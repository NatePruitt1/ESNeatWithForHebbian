package esneatforhebbians

// AI-generated regression tests for Network (network.go) and the layout rules
// (layout.go). Expected values are worked out by hand in each test.

import (
	"math"
	"strings"
	"testing"
)

func mustNetwork(t *testing.T, inputs, outputs int, nodes []Node, conns []Connection, recurrent bool) *Network {
	t.Helper()
	net, err := NewNetwork(inputs, outputs, nodes, conns, recurrent)
	if err != nil {
		t.Fatalf("NewNetwork: %v", err)
	}
	return net
}

func near(a, b float64) bool { return math.Abs(a-b) < 1e-12 }

func TestActivationFunctions(t *testing.T) {
	cases := []struct {
		id   ActivationFuncID
		x    float64
		want float64
	}{
		{Linear, -2.5, -2.5},
		{Sigmoid, 0, 0.5},
		{Sigmoid, 2, 1 / (1 + math.Exp(-2))},
		{Sigmoid, -1000, 0},
		{Tanh, 1, math.Tanh(1)},
		{Step, 0, 0},
		{Step, 0.1, 1},
		{Gaussion, 0, 1},
		{Gaussion, -1, math.Exp(-1)},
		{Sine, math.Pi / 2, 1},
		{Abs, -3, 3},
		{Sign, -0.1, -1},
		{Sign, 0, 1},
	}
	for _, c := range cases {
		if got := c.id.Activate(c.x); !near(got, c.want) {
			t.Errorf("activation %d at %g = %g, want %g", c.id, c.x, got, c.want)
		}
	}
	if ActivationFuncID(AllActivationFuncCount).Valid() || ActivationFuncID(-1).Valid() || !Sign.Valid() {
		t.Error("Valid does not match the defined IDs")
	}
}

// Layout: input 0, output 1, hidden 2 and 3.
// hidden2 = Tanh(0.5 + 2*x), hidden3 = Linear(-1 + 3*hidden2), output = Sigmoid(hidden3).
func TestNetworkForwardChain(t *testing.T) {
	nodes := []Node{
		{},
		{Activation: Sigmoid},
		{Activation: Tanh, Bias: 0.5},
		{Activation: Linear, Bias: -1},
	}
	net := mustNetwork(t, 1, 1, nodes, []Connection{
		{From: 0, To: 2, Weight: 2, Forward: true},
		{From: 2, To: 3, Weight: 3, Forward: true},
		{From: 3, To: 1, Weight: 1, Forward: true},
	}, false)
	for _, x := range []float64{-1, 0, 0.7} {
		h2 := math.Tanh(0.5 + 2*x)
		h3 := -1 + 3*h2
		want := 1 / (1 + math.Exp(-h3))
		if got := net.Compute([]float64{x})[0]; !near(got, want) {
			t.Fatalf("x=%g: got %g, want %g", x, got, want)
		}
		if !near(net.Nodes[3].PreSum, h3) {
			t.Fatalf("x=%g: PreSum %g, want %g", x, net.Nodes[3].PreSum, h3)
		}
	}
}

// Inputs pass through unchanged, even if the node has another activation.
func TestNetworkInputsPassThrough(t *testing.T) {
	nodes := []Node{{Activation: Sigmoid}, {}}
	net := mustNetwork(t, 1, 1, nodes, []Connection{{From: 0, To: 1, Weight: 1, Forward: true}}, false)
	if got := net.Compute([]float64{3})[0]; got != 3 {
		t.Fatalf("got %g, want 3", got)
	}
}

// Each output reads its own connections; several outputs are returned in order.
// Layout: inputs 0, 1; outputs 2, 3, 4; hidden 5, 6.
func TestNetworkSeveralOutputs(t *testing.T) {
	net := mustNetwork(t, 2, 3, make([]Node, 7), []Connection{
		{From: 0, To: 5, Weight: 1, Forward: true},
		{From: 1, To: 6, Weight: 1, Forward: true},
		{From: 5, To: 2, Weight: 2, Forward: true},
		{From: 6, To: 3, Weight: 3, Forward: true},
		{From: 5, To: 4, Weight: 1, Forward: true},
		{From: 6, To: 4, Weight: 1, Forward: true},
		{From: 0, To: 4, Weight: 100, Forward: true}, // input straight to output
	}, false)
	got := net.Compute([]float64{1, 10})
	want := []float64{2, 30, 111}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("outputs %v, want %v", got, want)
		}
	}
}

// Recurrent connections read the previous step's value, so each one delays
// its signal by exactly one step, and the first step reads 0.
// Layout: input 0, output 1, hidden 2 (lower index) and 3 (higher index).
func TestNetworkRecurrentDelay(t *testing.T) {
	cases := []struct {
		name  string
		conns []Connection
		want  []float64 // outputs for inputs 1, 2, 3
	}{
		{"lateral, lower to higher index", []Connection{
			{From: 0, To: 2, Weight: 1, Forward: true},
			{From: 2, To: 3, Weight: 1, Forward: false},
			{From: 3, To: 1, Weight: 1, Forward: true},
		}, []float64{0, 1, 2}},
		{"backward, higher to lower index", []Connection{
			{From: 0, To: 3, Weight: 1, Forward: true},
			{From: 3, To: 2, Weight: 1, Forward: false},
			{From: 2, To: 1, Weight: 1, Forward: true},
		}, []float64{0, 1, 2}},
		// hidden = previous output; output = x + 10*hidden:
		// 1 + 10*0 = 1, then 2 + 10*1 = 12, then 3 + 10*12 = 123.
		{"from an output back into a hidden node", []Connection{
			{From: 0, To: 1, Weight: 1, Forward: true},
			{From: 1, To: 2, Weight: 1, Forward: false},
			{From: 2, To: 1, Weight: 10, Forward: true},
		}, []float64{1, 12, 123}},
		{"self-connection accumulates", []Connection{
			{From: 0, To: 2, Weight: 1, Forward: true},
			{From: 2, To: 2, Weight: 1, Forward: false},
			{From: 2, To: 1, Weight: 1, Forward: true},
		}, []float64{1, 3, 6}},
	}
	for _, c := range cases {
		net := mustNetwork(t, 1, 1, make([]Node, 4), c.conns, true)
		for step, x := range []float64{1, 2, 3} {
			if got := net.Compute([]float64{x})[0]; got != c.want[step] {
				t.Fatalf("%s: step %d output %g, want %g", c.name, step, got, c.want[step])
			}
		}
	}
}

func TestNetworkReset(t *testing.T) {
	net := mustNetwork(t, 1, 1, make([]Node, 3), []Connection{
		{From: 0, To: 2, Weight: 1, Forward: true},
		{From: 2, To: 2, Weight: 1, Forward: false},
		{From: 2, To: 1, Weight: 1, Forward: true},
	}, true)
	net.Compute([]float64{5})
	net.Compute([]float64{5})
	net.Reset()
	for i, n := range net.Nodes {
		if n.Current != 0 || n.Previous != 0 || n.PreSum != 0 {
			t.Fatalf("node %d not reset: %+v", i, n)
		}
	}
	if got := net.Compute([]float64{1})[0]; got != 1 {
		t.Fatalf("after Reset got %g, want 1", got)
	}
}

// Without recurrent connections, a network has no state between steps.
func TestNetworkFeedForwardIsStateless(t *testing.T) {
	net := mustNetwork(t, 2, 1, []Node{{}, {}, {Activation: Gaussion}, {}, {Activation: Sine}}, []Connection{
		{From: 0, To: 3, Weight: 1.5, Forward: true},
		{From: 1, To: 4, Weight: -2, Forward: true},
		{From: 3, To: 4, Weight: 0.5, Forward: true},
		{From: 4, To: 2, Weight: 1, Forward: true},
	}, false)
	in := []float64{0.3, -0.4}
	first := net.Compute(in)[0]
	net.Compute([]float64{9, 9})
	if again := net.Compute(in)[0]; again != first {
		t.Fatalf("same input gave %g then %g", first, again)
	}
}

func TestNetworkLayoutRules(t *testing.T) {
	// Layout for every case: input 0, output 1, hidden 2 and 3.
	cases := []struct {
		name      string
		conns     []Connection
		recurrent bool
		want      string // "" means valid; otherwise part of the error
	}{
		{"forward chain", []Connection{{From: 0, To: 2, Forward: true}, {From: 2, To: 3, Forward: true}, {From: 3, To: 1, Forward: true}}, false, ""},
		{"input straight to output", []Connection{{From: 0, To: 1, Forward: true}}, false, ""},
		{"recurrent backward", []Connection{{From: 3, To: 2}}, true, ""},
		{"recurrent self", []Connection{{From: 2, To: 2}}, true, ""},
		{"recurrent from output", []Connection{{From: 1, To: 2}}, true, ""},
		{"recurrent not allowed", []Connection{{From: 3, To: 2}}, false, "does not allow recurrent"},
		{"forward to lower hidden index", []Connection{{From: 3, To: 2, Forward: true}}, true, "higher index"},
		{"forward self", []Connection{{From: 2, To: 2, Forward: true}}, true, "higher index"},
		{"forward from output", []Connection{{From: 1, To: 2, Forward: true}}, true, "from output"},
		{"forward into input", []Connection{{From: 2, To: 0, Forward: true}}, true, "ends at input"},
		{"recurrent into input", []Connection{{From: 2, To: 0}}, true, "ends at input"},
		{"index out of range", []Connection{{From: 0, To: 9, Forward: true}}, true, "outside"},
	}
	for _, c := range cases {
		_, err := NewNetwork(1, 1, make([]Node, 4), c.conns, c.recurrent)
		switch {
		case c.want == "" && err != nil:
			t.Errorf("%s: unexpected error %v", c.name, err)
		case c.want != "" && (err == nil || !strings.Contains(err.Error(), c.want)):
			t.Errorf("%s: got error %v, want one mentioning %q", c.name, err, c.want)
		}
	}
	if _, err := NewNetwork(2, 2, make([]Node, 3), nil, false); err == nil {
		t.Error("more inputs and outputs than nodes should be an error")
	}
}

func TestNetworkRejectsInvalidActivation(t *testing.T) {
	if _, err := NewNetwork(1, 1, []Node{{}, {Activation: AllActivationFuncCount}}, nil, false); err == nil {
		t.Fatal("an invalid activation ID should be an error")
	}
}

func TestNetworkWrongInputCountPanics(t *testing.T) {
	net := mustNetwork(t, 2, 1, make([]Node, 3), nil, false)
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic for the wrong number of inputs")
		}
	}()
	net.Compute([]float64{1})
}
