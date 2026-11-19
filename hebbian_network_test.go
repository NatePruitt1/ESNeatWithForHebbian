package esneatforhebbians

// AI-generated regression tests for HebbianNetwork (hebbian_network.go).
// Expected values are worked out by hand in each test.

import (
	"math"
	"strings"
	"testing"
)

func mustHebbian(t *testing.T, inputs, outputs int, nodes []Node, conns []Connection, plast []Plasticity) *HebbianNetwork {
	t.Helper()
	hn, err := NewHebbianNetwork(inputs, outputs, nodes, conns, plast)
	if err != nil {
		t.Fatalf("NewHebbianNetwork: %v", err)
	}
	return hn
}

// A HebbianNetwork that does not learn computes exactly what a Network with
// the same weights does, including recurrent connections.
func TestHebbianEvaluatesLikeNetwork(t *testing.T) {
	newNodes := func() []Node {
		return []Node{{}, {Activation: Tanh}, {Activation: Sigmoid, Bias: 0.2}, {Activation: Gaussion}}
	}
	newConns := func() []Connection {
		return []Connection{
			{From: 0, To: 2, Forward: true},
			{From: 2, To: 3, Forward: true},
			{From: 3, To: 2, Forward: false},
			{From: 3, To: 1, Forward: true},
			{From: 1, To: 3, Forward: false},
		}
	}
	plast := []Plasticity{{M: 1.5, Dir: 1}, {M: 0.5, Dir: -1}, {M: 2, Dir: 1}, {M: 1, Dir: 1}, {M: 0.3, Dir: -1}}

	conns := newConns()
	for i := range conns {
		conns[i].Weight = plast[i].Weight()
	}
	net := mustNetwork(t, 1, 1, newNodes(), conns, true)
	hn := mustHebbian(t, 1, 1, newNodes(), newConns(), plast)
	for step, x := range []float64{1, -0.5, 0.25, 2} {
		a, b := net.Compute([]float64{x})[0], hn.Compute([]float64{x})[0]
		if a != b {
			t.Fatalf("step %d: Network %g, HebbianNetwork %g", step, a, b)
		}
	}
}

// The constructor sets each weight from its plasticity, ignoring the Weight
// passed in.
func TestHebbianWeightComesFromPlasticity(t *testing.T) {
	hn := mustHebbian(t, 1, 1, make([]Node, 2),
		[]Connection{{From: 0, To: 1, Weight: 99, Forward: true}},
		[]Plasticity{{M: 2, Dir: -1}})
	if w := hn.Connections[0].Weight; w != -2 {
		t.Fatalf("weight %g, want M*Dir = -2", w)
	}
	if got := hn.Compute([]float64{3})[0]; got != -6 {
		t.Fatalf("output %g, want -6", got)
	}
}

// Each term of ΔM = Eta * fitness * (A*pre*post + B*pre + C*post + D), one at
// a time. One linear connection input -> output with M = 1, Dir = 1, so
// pre = x and post = x after Compute.
func TestHebbianUpdateRule(t *testing.T) {
	const x = 2.0 // pre = 2, post = 2
	cases := []struct {
		name    string
		p       Plasticity
		fitness float64
		wantM   float64
	}{
		{"A uses pre*post", Plasticity{A: 1, Eta: 0.5}, 1, 1 + 0.5*(2*2)},
		{"B uses pre", Plasticity{B: 1, Eta: 0.5}, 1, 1 + 0.5*2},
		{"C uses post", Plasticity{C: 1, Eta: 0.5}, 1, 1 + 0.5*2},
		{"D is constant", Plasticity{D: 1, Eta: 0.5}, 1, 1 + 0.5},
		{"fitness scales", Plasticity{D: 1, Eta: 0.5}, 3, 1 + 0.5*3},
		{"zero Eta does nothing", Plasticity{A: 1, B: 1, C: 1, D: 1}, 1, 1},
		{"negative fitness shrinks M", Plasticity{D: 1, Eta: 0.5}, -1, 0.5},
		{"M stops at 0", Plasticity{D: 1, Eta: 1}, -5, 0},
	}
	for _, c := range cases {
		p := c.p
		p.M, p.Dir = 1, 1
		hn := mustHebbian(t, 1, 1, make([]Node, 2), []Connection{{From: 0, To: 1, Forward: true}}, []Plasticity{p})
		hn.Compute([]float64{x})
		hn.UpdateWeights(c.fitness)
		if got := hn.Plasticity[0].M; !near(got, c.wantM) {
			t.Errorf("%s: M = %g, want %g", c.name, got, c.wantM)
		}
		if got := hn.Connections[0].Weight; !near(got, c.wantM) {
			t.Errorf("%s: weight = %g, want M*Dir = %g", c.name, got, c.wantM)
		}
	}
}

// B and C must use different ends of the connection. With a weight of 0.5,
// pre = 2 and post = 1, so B and C give different changes.
func TestHebbianPreAndPostAreDistinct(t *testing.T) {
	for _, c := range []struct {
		name  string
		p     Plasticity
		wantM float64
	}{
		{"B (pre = 2)", Plasticity{B: 1, Eta: 1}, 0.5 + 2},
		{"C (post = 1)", Plasticity{C: 1, Eta: 1}, 0.5 + 1},
	} {
		p := c.p
		p.M, p.Dir = 0.5, 1
		hn := mustHebbian(t, 1, 1, make([]Node, 2), []Connection{{From: 0, To: 1, Forward: true}}, []Plasticity{p})
		hn.Compute([]float64{2})
		hn.UpdateWeights(1)
		if got := hn.Plasticity[0].M; got != c.wantM {
			t.Errorf("%s: M = %g, want %g", c.name, got, c.wantM)
		}
	}
}

// With Dir = -1 the weight stays negative as M grows: learning changes the
// magnitude, never the sign.
func TestHebbianDirKeepsSign(t *testing.T) {
	hn := mustHebbian(t, 1, 1, make([]Node, 2), []Connection{{From: 0, To: 1, Forward: true}},
		[]Plasticity{{M: 1, Dir: -1, D: 1, Eta: 1}})
	hn.Compute([]float64{1})
	hn.UpdateWeights(1)
	if w := hn.Connections[0].Weight; w != -2 {
		t.Fatalf("weight %g, want -2", w)
	}
}

// Learning affects the next Compute call.
func TestHebbianLearningChangesNextOutput(t *testing.T) {
	hn := mustHebbian(t, 1, 1, make([]Node, 2), []Connection{{From: 0, To: 1, Forward: true}},
		[]Plasticity{{M: 1, Dir: 1, D: 1, Eta: 1}})
	if got := hn.Compute([]float64{1})[0]; got != 1 {
		t.Fatalf("before learning: %g, want 1", got)
	}
	hn.UpdateWeights(1) // M: 1 -> 2
	if got := hn.Compute([]float64{1})[0]; got != 2 {
		t.Fatalf("after learning: %g, want 2", got)
	}
}

func TestNewHebbianNetworkValidates(t *testing.T) {
	conn := []Connection{{From: 0, To: 1, Forward: true}}
	cases := []struct {
		name  string
		plast []Plasticity
		want  string
	}{
		{"missing plasticity", nil, "plasticity entries"},
		{"Dir not a sign", []Plasticity{{M: 1, Dir: 0.5}}, "Dir"},
		{"Dir zero", []Plasticity{{M: 1}}, "Dir"},
		{"negative M", []Plasticity{{M: -1, Dir: 1}}, "M is"},
		{"negative Eta", []Plasticity{{M: 1, Dir: 1, Eta: -0.1}}, "Eta"},
		{"NaN coefficient", []Plasticity{{M: 1, Dir: 1, B: math.NaN()}}, "B is"},
		{"infinite M", []Plasticity{{M: math.Inf(1), Dir: 1}}, "finite"},
	}
	for _, c := range cases {
		_, err := NewHebbianNetwork(1, 1, make([]Node, 2), []Connection{conn[0]}, c.plast)
		if err == nil || !strings.Contains(err.Error(), c.want) {
			t.Errorf("%s: got error %v, want one mentioning %q", c.name, err, c.want)
		}
	}
	// Layout rules still apply.
	if _, err := NewHebbianNetwork(1, 1, make([]Node, 2), []Connection{{From: 1, To: 0, Forward: true}}, []Plasticity{{M: 1, Dir: 1}}); err == nil {
		t.Error("a connection into an input should be an error")
	}
}

// ConvertToHebbian reads the plasticity from each connection's CPPN values:
// M = |weight|, and Dir, A, B, C, D, Eta from their own outputs.
func TestConvertToHebbian(t *testing.T) {
	values := make([]float64, HebbianValueCount)
	values[WeightIndex] = -0.6
	values[AIndex], values[BIndex], values[CIndex], values[DIndex] = 0.1, 0.2, 0.3, 0.4
	values[DirIndex] = -1
	values[EtaIndex] = 0.5
	an := &AbstractNetwork{
		InputCount: 1, OutputCount: 1,
		Nodes: []AbstractNode{{Activation: Linear}, {Activation: Tanh}},
		Connections: []AbstractConnection{{
			Connection: Connection{From: 0, To: 1, Weight: -0.6, Forward: true},
			Values:     values,
		}},
	}
	hn, err := an.ConvertToHebbian()
	if err != nil {
		t.Fatal(err)
	}
	want := Plasticity{A: 0.1, B: 0.2, C: 0.3, D: 0.4, Eta: 0.5, M: 0.6, Dir: -1}
	if hn.Plasticity[0] != want {
		t.Fatalf("plasticity %+v, want %+v", hn.Plasticity[0], want)
	}
	if hn.Connections[0].Weight != -0.6 || hn.Nodes[1].Activation != Tanh {
		t.Fatalf("connection %+v, output activation %d", hn.Connections[0], hn.Nodes[1].Activation)
	}

	an.Connections[0].Values = values[:EtaIndex]
	if _, err := an.ConvertToHebbian(); err == nil {
		t.Error("too few CPPN values should be an error")
	}
	an.Connections[0].Values = values
	values[DirIndex] = 0.5
	if _, err := an.ConvertToHebbian(); err == nil {
		t.Error("a Dir that is not a sign should be an error")
	}
}
