package esneatforhebbians

import (
	"slices"
	"testing"
)

// fixed returns a connection with a fixed weight and no plasticity.
func fixed(from, to int, w float64, forward bool) HebbianConnection {
	return HebbianConnection{From: from, To: to, W: w, Forward: forward}
}

func linearNodes(n int) []HebbianNode {
	nodes := make([]HebbianNode, n)
	for i := range nodes {
		nodes[i].Activation = Linear
	}
	return nodes
}

func newNet(t *testing.T, inputs, outputs int, nodes []HebbianNode, conns []HebbianConnection) *HebbianNetwork {
	t.Helper()
	hn, err := NewHebbianNetwork(inputs, outputs, nodes, conns)
	if err != nil {
		t.Fatal(err)
	}
	return hn
}

func run(hn *HebbianNetwork, inputs ...float64) []float64 {
	var outs []float64
	for _, x := range inputs {
		outs = append(outs, hn.Compute([]float64{x})[0])
	}
	return outs
}

// Layout in these tests: node 0 is the input, node 1 the output, the rest hidden.

// A forward chain I -> H -> O reaches the output in the same step.
func TestHebbianForwardChain(t *testing.T) {
	hn := newNet(t, 1, 1, linearNodes(3), []HebbianConnection{
		fixed(0, 2, 1, true),
		fixed(2, 1, 1, true),
	})
	if got := run(hn, 2, 5, -1); !slices.Equal(got, []float64{2, 5, -1}) {
		t.Fatalf("outputs %v, want [2 5 -1]", got)
	}
}

// A lateral connection Ha -> Hb from a lower index to a higher one must still
// be delayed by one step, even though Ha is computed first.
func TestHebbianLateralIsDelayed(t *testing.T) {
	hn := newNet(t, 1, 1, linearNodes(4), []HebbianConnection{
		fixed(0, 2, 1, true),  // I -> Ha
		fixed(2, 3, 1, false), // Ha -> Hb, lateral
		fixed(3, 1, 1, true),  // Hb -> O
	})
	if got := run(hn, 1, 2, 3); !slices.Equal(got, []float64{0, 1, 2}) {
		t.Fatalf("outputs %v, want [0 1 2] (one step of delay)", got)
	}
}

// A backward connection from a higher-index hidden node is delayed by one step
// and does not panic on the very first call.
func TestHebbianBackwardIsDelayed(t *testing.T) {
	hn := newNet(t, 1, 1, linearNodes(4), []HebbianConnection{
		fixed(0, 3, 1, true),  // I -> Hhigh
		fixed(3, 2, 1, false), // Hhigh -> Hlow, backward
		fixed(2, 1, 1, true),  // Hlow -> O
	})
	if got := run(hn, 4, 5, 6); !slices.Equal(got, []float64{0, 4, 5}) {
		t.Fatalf("outputs %v, want [0 4 5]", got)
	}
}

// A self-connection with weight 1 makes the node an accumulator; Reset clears it.
func TestHebbianSelfConnectionAndReset(t *testing.T) {
	hn := newNet(t, 1, 1, linearNodes(3), []HebbianConnection{
		fixed(0, 2, 1, true),
		fixed(2, 2, 1, false), // self-connection
		fixed(2, 1, 1, true),
	})
	if got := run(hn, 1, 1, 1); !slices.Equal(got, []float64{1, 2, 3}) {
		t.Fatalf("outputs %v, want [1 2 3]", got)
	}
	hn.Reset()
	if got := run(hn, 1); !slices.Equal(got, []float64{1}) {
		t.Fatalf("after Reset got %v, want [1]", got)
	}
}

// Every output is computed, and each node reads its own incoming connections.
func TestHebbianSeveralOutputs(t *testing.T) {
	// 2 inputs (0, 1), 3 outputs (2, 3, 4), 2 hidden (5, 6).
	hn := newNet(t, 2, 3, linearNodes(7), []HebbianConnection{
		fixed(0, 5, 1, true),
		fixed(1, 6, 1, true),
		fixed(5, 2, 2, true),
		fixed(6, 3, 3, true),
		fixed(5, 4, 1, true),
		fixed(6, 4, 1, true),
	})
	got := hn.Compute([]float64{1, 10})
	if want := []float64{2, 30, 11}; !slices.Equal(got, want) {
		t.Fatalf("outputs %v, want %v", got, want)
	}
}

func TestHebbianActivationApplied(t *testing.T) {
	nodes := linearNodes(3)
	nodes[2].Activation = Step
	nodes[1].Activation = Tanh
	hn := newNet(t, 1, 1, nodes, []HebbianConnection{
		fixed(0, 2, 1, true),
		fixed(2, 1, 0.5, true),
	})
	if got, want := hn.Compute([]float64{3})[0], Tanh.Activate(0.5); got != want {
		t.Fatalf("output %g, want %g", got, want)
	}
}

// The C term must use the post-synaptic activation.
func TestHebbianUpdateUsesPostForC(t *testing.T) {
	hn := newNet(t, 1, 1, linearNodes(2), []HebbianConnection{
		{From: 0, To: 1, W: 2, M: 1, Dir: 1, C: 1, Eta: 1, Forward: true},
	})
	hn.Compute([]float64{0}) // pre = 0, post = 0
	hn.Nodes[1].Current = 3  // set post directly so pre and post differ
	hn.UpdateWeights(1)
	if c := hn.Connections[0]; c.M != 4 || c.W != 4 {
		t.Fatalf("M=%g W=%g, want 4 and 4 (M grows by C*post = 3)", c.M, c.W)
	}
}

func TestHebbianUpdateKeepsMNonNegative(t *testing.T) {
	hn := newNet(t, 1, 1, linearNodes(2), []HebbianConnection{
		{From: 0, To: 1, M: 0.5, Dir: -1, D: -10, Eta: 1, Forward: true},
	})
	hn.Compute([]float64{1})
	hn.UpdateWeights(1)
	if c := hn.Connections[0]; c.M != 0 || c.W != 0 {
		t.Fatalf("M=%g W=%g, want 0 and 0", c.M, c.W)
	}
}

func TestNewHebbianNetworkValidates(t *testing.T) {
	if _, err := NewHebbianNetwork(1, 1, linearNodes(2), []HebbianConnection{fixed(0, 5, 1, true)}); err == nil {
		t.Fatal("a connection to a missing node should be an error")
	}
	if _, err := NewHebbianNetwork(2, 2, linearNodes(3), nil); err == nil {
		t.Fatal("more inputs and outputs than nodes should be an error")
	}
}

func TestHebbianComputeWrongInputCountPanics(t *testing.T) {
	hn := newNet(t, 2, 1, linearNodes(3), nil)
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic for the wrong number of inputs")
		}
	}()
	hn.Compute([]float64{1})
}
