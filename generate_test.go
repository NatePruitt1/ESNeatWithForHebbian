package esneatforhebbians

// AI-generated regression tests for AbstractNetwork (abstractnet.go) and the
// ES-HyperNEAT pipeline that builds it (generate.go and the files it calls).

import (
	"math"
	"testing"
)

// waveCPPN is a fixed connection CPPN with a smooth, varied weight pattern
// and all the outputs ConvertToHebbian needs.
type waveCPPN struct{}

func (waveCPPN) Compute(x1, y1, x2, y2, bias float64, networkType int) []float64 {
	dir := 1.0
	if math.Cos(3*x1+2*y2) < 0 {
		dir = -1
	}
	out := make([]float64, HebbianValueCount)
	out[WeightIndex] = math.Sin(3*(x2-x1))*math.Cos(2*(y2-y1)) + 0.2*x1*y2 + 0.1*bias
	out[AIndex] = 0.3 * math.Sin(x1+y2)
	out[BIndex] = 0.2 * math.Cos(x2)
	out[CIndex] = 0.1 * y1
	out[DIndex] = -0.05 * x2 * y2
	out[DirIndex] = dir
	out[EtaIndex] = 0.1 * (1 + math.Tanh(x2+float64(networkType)))
	return out
}
func (waveCPPN) GetOutputSize() int { return HebbianValueCount }

// smoothNodeCPPN chooses Sigmoid, Tanh or Gaussion depending on position.
type smoothNodeCPPN struct{}

func (smoothNodeCPPN) ComputeNode(x, y, bias float64, networkType int) []float64 {
	return []float64{-5, math.Sin(3 * x), math.Cos(2 * y), -5, x * y}
}
func (smoothNodeCPPN) GetOutputSize() int { return ActivationFuncCount }

var testSearch = SearchParams{
	InitialDepth: 2, MaxDepth: 4,
	DivisionThreshold: 0.03, VarianceThreshold: 0.03, BandThreshold: 0.3,
	IterationLevel: 2,
}

func generateWave(t *testing.T, recurrent bool) *AbstractNetwork {
	t.Helper()
	an, err := GenerateAbstractFromCPPNs(waveCPPN{}, smoothNodeCPPN{}, 1, 0, recurrent, 3, 2, testSearch)
	if err != nil {
		t.Fatalf("GenerateAbstractFromCPPNs: %v", err)
	}
	return an
}

// checksum depends on every connection's endpoints, weight, flag and order.
func checksum(an *AbstractNetwork) float64 {
	sum := 0.0
	for i, c := range an.Connections {
		f := 1.0
		if c.Forward {
			f = 2
		}
		sum += float64(i%97+1) * (c.Weight + 0.001*float64(c.From) + 0.0001*float64(c.To)) * f
	}
	for i, n := range an.Nodes {
		sum += float64(i%89+1) * (n.X + 2*n.Y + 0.1*float64(n.Activation))
	}
	return sum
}

func hebbianRun(t *testing.T, an *AbstractNetwork) []float64 {
	hn, err := an.ConvertToHebbian()
	if err != nil {
		t.Fatal(err)
	}
	var out []float64
	for i, in := range [][]float64{{1, 0, -1}, {0.5, 0.5, 0}, {-1, 1, 0.25}} {
		out = append(out, hn.Compute(in)...)
		hn.UpdateWeights(0.5 - 0.3*float64(i))
	}
	return out
}

func closeTo(a, b, tol float64) bool { return math.Abs(a-b) <= tol*math.Max(1, math.Abs(b)) }

// TestGenerateGolden pins the exact output of the whole search for one fixed
// CPPN pair. It fails if anything changes which nodes and connections are
// found, their order, weights or activations, or how the networks compute.
//
// If you change the search on purpose, check the new numbers make sense and
// update them here (the failure message prints them).
func TestGenerateGolden(t *testing.T) {
	golden := []struct {
		recurrent      bool
		nodes, conns   int
		forward        int
		checksum       float64
		networkOutputs []float64 // ConvertToNetwork, input {1, 0, -1}
		hebbianOutputs []float64 // ConvertToHebbian, see hebbianRun
	}{
		{true, 327, 12110, 5772, 232378.622286,
			[]float64{0.000244506321212, 0.949149141352},
			[]float64{0.717097650093, 0.39346364368, 1.66493089095e-16, 0.00264440363651, 2.06770238488e-16, 0.00101683186241}},
		{false, 239, 3768, 3768, 74932.8515857,
			[]float64{0.000590666273856, 0.80360955182},
			nil},
	}
	for _, g := range golden {
		an := generateWave(t, g.recurrent)
		forward := 0
		for _, c := range an.Connections {
			if c.Forward {
				forward++
			}
		}
		if len(an.Nodes) != g.nodes || len(an.Connections) != g.conns || forward != g.forward {
			t.Fatalf("recurrent=%v: %d nodes, %d connections (%d forward); want %d, %d (%d)",
				g.recurrent, len(an.Nodes), len(an.Connections), forward, g.nodes, g.conns, g.forward)
		}
		if sum := checksum(an); !closeTo(sum, g.checksum, 1e-9) {
			t.Fatalf("recurrent=%v: checksum %.12g, want %.12g", g.recurrent, sum, g.checksum)
		}
		net, err := an.ConvertToNetwork(g.recurrent)
		if err != nil {
			t.Fatal(err)
		}
		out := net.Compute([]float64{1, 0, -1})
		for i := range g.networkOutputs {
			if !closeTo(out[i], g.networkOutputs[i], 1e-9) {
				t.Fatalf("recurrent=%v: network outputs %.12g, want %.12g", g.recurrent, out, g.networkOutputs)
			}
		}
		if g.hebbianOutputs != nil {
			h := hebbianRun(t, an)
			for i := range g.hebbianOutputs {
				if !closeTo(h[i], g.hebbianOutputs[i], 1e-9) {
					t.Fatalf("Hebbian outputs %.12g, want %.12g", h, g.hebbianOutputs)
				}
			}
		}
	}
}

// TestGenerateStructure checks the rules every generated network must follow,
// independent of the exact numbers.
func TestGenerateStructure(t *testing.T) {
	for _, rec := range []bool{true, false} {
		an := generateWave(t, rec)
		if an.InputCount != 3 || an.OutputCount != 2 {
			t.Fatalf("counts %d/%d, want 3/2", an.InputCount, an.OutputCount)
		}
		for i := 0; i < 3; i++ {
			if n := an.Nodes[i]; n.Y != InputY || n.Activation != Linear {
				t.Fatalf("node %d should be a Linear input at y=%g: %+v", i, InputY, n)
			}
		}
		for i := 3; i < 5; i++ {
			if an.Nodes[i].Y != OutputY {
				t.Fatalf("node %d should be an output at y=%g", i, OutputY)
			}
		}
		hidden := an.Nodes[5:]
		for i := 1; i < len(hidden); i++ {
			if comparePoints(hidden[i-1].Point, hidden[i].Point) >= 0 {
				t.Fatalf("hidden nodes %d and %d are not sorted by y then x", i-1, i)
			}
		}
		for i, n := range an.Nodes[3:] {
			if n.Activation != chooseActivation(n.Values) {
				t.Fatalf("node %d's activation does not match its CPPN values", i+3)
			}
		}
		for i, c := range an.Connections {
			if c.Forward != (an.Nodes[c.To].Y > an.Nodes[c.From].Y) {
				t.Fatalf("connection %d: Forward=%v does not match the y rule", i, c.Forward)
			}
			if !rec && !c.Forward {
				t.Fatalf("connection %d is recurrent with recurrence off", i)
			}
			if c.Weight != c.Values[WeightIndex] {
				t.Fatalf("connection %d: weight %g differs from its CPPN value %g", i, c.Weight, c.Values[WeightIndex])
			}
		}
		if err := validateLayout(len(an.Nodes), an.InputCount, an.OutputCount, an.connections(), rec); err != nil {
			t.Fatal(err)
		}
	}
}

func TestGenerateIsDeterministic(t *testing.T) {
	a, b := generateWave(t, true), generateWave(t, true)
	if checksum(a) != checksum(b) || len(a.Connections) != len(b.Connections) {
		t.Fatal("two runs with the same CPPNs gave different networks")
	}
}

func TestGenerateRejectsSmallCPPNs(t *testing.T) {
	if _, err := GenerateAbstractFromCPPNs(waveCPPN{}, tooFewNodeOutputs{}, 1, 0, true, 3, 2, testSearch); err == nil {
		t.Error("a node CPPN with fewer outputs than activation functions should be an error")
	}
	if _, err := GenerateAbstractFromCPPNs(waveCPPN{}, smoothNodeCPPN{}, 1, 0, true, -1, 2, testSearch); err == nil {
		t.Error("a negative input count should be an error")
	}
}

type tooFewNodeOutputs struct{}

func (tooFewNodeOutputs) ComputeNode(x, y, bias float64, networkType int) []float64 {
	return []float64{x, y}
}
func (tooFewNodeOutputs) GetOutputSize() int { return 2 }

// ConvertToNetwork keeps weights and activations, and drops recurrent
// connections when recurrence is off.
// Layout: input 0, output 1, hidden 2.
func TestConvertToNetwork(t *testing.T) {
	an := &AbstractNetwork{
		InputCount: 1, OutputCount: 1,
		Nodes: []AbstractNode{
			{Point: Point{0, InputY}, Activation: Linear},
			{Point: Point{0, OutputY}, Activation: Tanh},
			{Point: Point{0, 0}, Activation: Gaussion},
		},
		Connections: []AbstractConnection{
			{Connection: Connection{From: 0, To: 2, Weight: 0.5, Forward: true}},
			{Connection: Connection{From: 2, To: 1, Weight: -2, Forward: true}},
			{Connection: Connection{From: 2, To: 2, Weight: 1, Forward: false}},
		},
	}
	ff, err := an.ConvertToNetwork(false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ff.Connections) != 2 {
		t.Fatalf("feed-forward conversion kept %d connections, want 2", len(ff.Connections))
	}
	x := 1.5
	want := math.Tanh(-2 * math.Exp(-(0.5*x)*(0.5*x)))
	if got := ff.Compute([]float64{x})[0]; !near(got, want) {
		t.Fatalf("got %g, want %g", got, want)
	}
	rec, err := an.ConvertToNetwork(true)
	if err != nil {
		t.Fatal(err)
	}
	if len(rec.Connections) != 3 {
		t.Fatalf("recurrent conversion kept %d connections, want 3", len(rec.Connections))
	}
}
