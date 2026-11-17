package esneatforhebbians

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

func randomDenseGenomes(rng *rand.Rand, s DenseSpec) ([]float64, []ActivationFuncID) {
	weights := make([]float64, s.WeightGenomeLength())
	for i := range weights {
		weights[i] = rng.NormFloat64()
	}
	acts := make([]ActivationFuncID, s.ActivationGenomeLength())
	for i := range acts {
		acts[i] = ActivationFuncID(rng.Intn(AllActivationFuncCount))
	}
	return weights, acts
}

func TestDenseGenomeLengths(t *testing.T) {
	// 6 inputs (bias at 4) -> [8, 8] -> 7 outputs:
	// layer 1: 8 * 6; layer 2: 8 * (8 + bias); outputs: 7 * (8 + bias).
	s := DenseSpec{Inputs: 6, Outputs: 7, Hidden: []int{8, 8}, OutputActs: make([]ActivationFuncID, 7), BiasInput: 4}
	if got, want := s.WeightGenomeLength(), 8*6+8*9+7*9; got != want {
		t.Fatalf("weight genome length %d, want %d", got, want)
	}
	if got := s.ActivationGenomeLength(); got != 16 {
		t.Fatalf("activation genome length %d, want 16", got)
	}
	s.BiasInput = -1
	if got, want := s.WeightGenomeLength(), 8*6+8*8+7*8; got != want {
		t.Fatalf("without bias: weight genome length %d, want %d", got, want)
	}
	s.Hidden = nil
	if got, want := s.WeightGenomeLength(), 7*6; got != want {
		t.Fatalf("no hidden layers: weight genome length %d, want %d", got, want)
	}
	if got := s.ActivationGenomeLength(); got != 0 {
		t.Fatalf("no hidden layers: activation genome length %d, want 0", got)
	}
}

// Inputs (x, bias) -> one hidden node -> one output:
// hidden = act(w0*x + w1*bias), output = Linear(w2*hidden + w3*bias).
func TestDenseComputesExpectedFunction(t *testing.T) {
	s := DenseSpec{Inputs: 2, Outputs: 1, Hidden: []int{1}, OutputActs: []ActivationFuncID{Linear}, BiasInput: 1}
	w := []float64{2, -1, 3, 0.5}
	for _, act := range []ActivationFuncID{Abs, Sine, Sign, Gaussion} {
		net, err := s.BuildNetwork(w, []ActivationFuncID{act})
		if err != nil {
			t.Fatal(err)
		}
		for _, x := range []float64{-1.5, 0.25} {
			want := 3*act.Activate(2*x-1) + 0.5
			if got := net.Compute([]float64{x, 1})[0]; math.Abs(got-want) > 1e-12 {
				t.Fatalf("act %d, x=%g: got %g, want %g", act, x, got, want)
			}
		}
	}
}

func TestDenseLayout(t *testing.T) {
	s := DenseSpec{Inputs: 3, Outputs: 2, Hidden: []int{4, 2}, OutputActs: []ActivationFuncID{Sign, Sigmoid}, BiasInput: 0}
	w, a := randomDenseGenomes(rand.New(rand.NewSource(1)), s)
	an, err := s.BuildAbstract(w, a)
	if err != nil {
		t.Fatal(err)
	}
	if len(an.Nodes) != 3+2+6 {
		t.Fatalf("%d nodes, want 11", len(an.Nodes))
	}
	for i := 0; i < 3; i++ {
		if an.Nodes[i].Posy != InputY || an.Nodes[i].ActivationFunc != Linear {
			t.Fatalf("node %d is not a linear input: %+v", i, an.Nodes[i])
		}
	}
	if an.Nodes[3].ActivationFunc != Sign || an.Nodes[4].ActivationFunc != Sigmoid || an.Nodes[3].Posy != OutputY {
		t.Fatal("output activations must come from OutputActs")
	}
	// Hidden nodes: layer 1 at y = -1/3, layer 2 at y = 1/3, in genome order.
	for i, n := range an.Nodes[5:] {
		wantY := -1.0 / 3
		if i >= 4 {
			wantY = 1.0 / 3
		}
		if math.Abs(n.Posy-wantY) > 1e-12 {
			t.Fatalf("hidden node %d at y=%g, want %g", i, n.Posy, wantY)
		}
		if n.ActivationFunc != a[i] {
			t.Fatalf("hidden node %d uses %d, want genome value %d", i, n.ActivationFunc, a[i])
		}
	}
	for i, c := range an.Connections {
		if !c.Forward || c.Weight != w[i] || len(c.Values) != 1 || c.Values[WeightIndex] != w[i] {
			t.Fatalf("connection %d: %+v", i, c)
		}
	}
	net, err := s.BuildNetwork(w, a)
	if err != nil {
		t.Fatal(err)
	}
	if len(net.Connections) != len(an.Connections) {
		t.Fatal("conversion dropped connections")
	}
}

func TestDenseRejectsBadInput(t *testing.T) {
	good := DenseSpec{Inputs: 2, Outputs: 1, Hidden: []int{2}, OutputActs: []ActivationFuncID{Linear}, BiasInput: 1}
	w := make([]float64, good.WeightGenomeLength())
	a := make([]ActivationFuncID, good.ActivationGenomeLength())
	if _, err := good.BuildAbstract(w, a); err != nil {
		t.Fatalf("good spec rejected: %v", err)
	}
	specCases := map[string]func(*DenseSpec){
		"no outputs":           func(s *DenseSpec) { s.Outputs = 0; s.OutputActs = nil },
		"empty layer":          func(s *DenseSpec) { s.Hidden = []int{0} },
		"output acts mismatch": func(s *DenseSpec) { s.OutputActs = nil },
		"bad output act":       func(s *DenseSpec) { s.OutputActs = []ActivationFuncID{99} },
		"bad bias input":       func(s *DenseSpec) { s.BiasInput = 5 },
	}
	for name, modify := range specCases {
		s := good
		modify(&s)
		if _, err := s.BuildAbstract(w, a); err == nil {
			t.Errorf("%s: expected an error", name)
		}
	}
	if _, err := good.BuildAbstract(w[1:], a); err == nil || !strings.Contains(err.Error(), "weight genome") {
		t.Errorf("short weight genome: %v", err)
	}
	if _, err := good.BuildAbstract(w, a[1:]); err == nil || !strings.Contains(err.Error(), "activation genome") {
		t.Errorf("short activation genome: %v", err)
	}
	if _, err := good.BuildAbstract(w, []ActivationFuncID{0, AllActivationFuncCount}); err == nil || !strings.Contains(err.Error(), "not a valid") {
		t.Errorf("invalid activation ID: %v", err)
	}
	bad := append([]float64(nil), w...)
	bad[2] = math.NaN()
	if _, err := good.BuildAbstract(bad, a); err == nil || !strings.Contains(err.Error(), "finite") {
		t.Errorf("NaN weight: %v", err)
	}
	// The CPPN builders check the input count for their role.
	if _, err := good.BuildCPPN(w, a); err == nil || !strings.Contains(err.Error(), "6 inputs") {
		t.Errorf("2-input spec as a connection CPPN: %v", err)
	}
	if _, err := good.BuildNodeCPPN(w, a); err == nil || !strings.Contains(err.Error(), "4 inputs") {
		t.Errorf("2-input spec as a node CPPN: %v", err)
	}
}

// End to end: random dense CPPNs whose fixed output functions enforce the
// Hebbian value rules drive ES-HyperNEAT, and every resulting connection
// passes HebbianConnection.Validate.
func TestDenseCPPNsBuildHebbianNetwork(t *testing.T) {
	connSpec := DenseSpec{Inputs: 6, Outputs: HebbianValueCount, Hidden: []int{8, 8}, BiasInput: 4}
	connSpec.OutputActs = make([]ActivationFuncID, HebbianValueCount)
	connSpec.OutputActs[WeightIndex] = Tanh
	connSpec.OutputActs[AIndex], connSpec.OutputActs[BIndex] = Tanh, Tanh
	connSpec.OutputActs[CIndex], connSpec.OutputActs[DIndex] = Tanh, Tanh
	connSpec.OutputActs[DirIndex] = Sign
	connSpec.OutputActs[EtaIndex] = Sigmoid

	nodeSpec := DenseSpec{Inputs: 4, Outputs: ActivationFuncCount, Hidden: []int{4}, BiasInput: 2,
		OutputActs: []ActivationFuncID{Linear, Linear, Linear, Linear, Linear}}

	rng := rand.New(rand.NewSource(7))
	built := 0
	for trial := 0; trial < 10; trial++ {
		cw, ca := randomDenseGenomes(rng, connSpec)
		nw, na := randomDenseGenomes(rng, nodeSpec)
		connCPPN, err := connSpec.BuildCPPN(cw, ca)
		if err != nil {
			t.Fatal(err)
		}
		nodeCPPN, err := nodeSpec.BuildNodeCPPN(nw, na)
		if err != nil {
			t.Fatal(err)
		}

		an, err := GenerateAbstractFromCPPNs(connCPPN, nodeCPPN, 1, 0, true, 3, 2, 3, 5, 1, 500, 0.03, 0.3, 0.03)
		if err != nil {
			t.Fatal(err)
		}
		hn, err := an.ConvertToHebbian(3, 2)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		hn.Compute([]float64{1, 0, -1})
		if len(hn.Connections) > 0 {
			built++
		}
	}
	t.Logf("%d of 10 random CPPN pairs produced connections", built)
	if built == 0 {
		t.Fatal("no trial produced any connections, so the validation rules were never exercised")
	}
}
