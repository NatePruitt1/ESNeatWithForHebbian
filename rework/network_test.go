// AI-generated: written by Claude (Claude Code, claude-opus-5-5) as a
// regression suite for the rework package. Review before relying on it.

package rework

import (
	"math"
	"testing"
)

// ---------- activations ----------

func TestActivationValues(t *testing.T) {
	cases := []struct {
		act  ActivationFunc
		x    float64
		want float64
	}{
		{Linear, -2.5, -2.5},
		{Sigmoid, 0, 0.5},
		{Tanh, 0.5, math.Tanh(0.5)},
		{Gaussian, 0, 1},
		{Gaussian, 1, math.Exp(-1)},
		{Sine, math.Pi / 2, 1},
		{Abs, -2, 2},
		{Step, 0, 0},
		{Step, 0.1, 1},
		{Sign, 0, 1},
		{Sign, -0.1, -1},
	}
	for _, c := range cases {
		if got := c.act.Compute(c.x); !approxEq(got, c.want) {
			t.Errorf("act %d at %v: got %v want %v", c.act, c.x, got, c.want)
		}
	}
}

func TestEveryActivationComputes(t *testing.T) {
	for a := ActivationFunc(0); a < ActivationFunctionCount; a++ {
		noPanic(t, func() { a.Compute(0.5) })
	}
}

func TestActivationValid(t *testing.T) {
	if !Linear.Valid() || !Sign.Valid() {
		t.Errorf("first/last activation reported invalid")
	}
	if ActivationFunc(-1).Valid() || ActivationFunc(ActivationFunctionCount).Valid() {
		t.Errorf("out-of-range activation reported valid")
	}
	mustPanic(t, func() { ActivationFunc(ActivationFunctionCount).Compute(0) })
}

func TestArgmax(t *testing.T) {
	if Argmax([]float64{}) != -1 {
		t.Errorf("empty slice")
	}
	if got := Argmax([]float64{1, 3, 2}); got != 1 {
		t.Errorf("got %d want 1", got)
	}
	if got := Argmax([]float64{5, 5, 1}); got != 0 {
		t.Errorf("ties should pick the first, got %d", got)
	}
}

// ---------- Forward ----------

func TestForwardLinear(t *testing.T) {
	ann := linearConnCPPN(1, 2, 3, 4, 0.5, Linear)
	got := ann.Forward([]float64{1, 1, 1, 1})
	if len(got) != 1 || got[0] != 10.5 {
		t.Fatalf("got %v, want [10.5]", got)
	}
}

func TestForwardMultipleOutputsInOrder(t *testing.T) {
	// [in, out0, out1]: out0 = 2*in, out1 = 3*in
	ann := ArtificialNeuralNet{InputCount: 1, OutputCount: 2, Nodes: make([]Node, 3),
		Connections: []Connection{
			{From: 0, To: 1, Weight: 2, Forward: true},
			{From: 0, To: 2, Weight: 3, Forward: true},
		}}
	got := ann.Forward([]float64{1})
	if len(got) != 2 || got[0] != 2 || got[1] != 3 {
		t.Fatalf("got %v, want [2 3]", got)
	}
}

func TestForwardHiddenChain(t *testing.T) {
	// [in, out, h1, h2]: in -> h1 -> h2 -> out
	ann := ArtificialNeuralNet{InputCount: 1, OutputCount: 1, Nodes: make([]Node, 4),
		Connections: []Connection{
			{From: 0, To: 2, Weight: 2, Forward: true},
			{From: 2, To: 3, Weight: 3, Forward: true},
			{From: 3, To: 1, Weight: 5, Forward: true},
		}}
	if got := ann.Forward([]float64{1})[0]; got != 30 {
		t.Fatalf("got %v, want 30", got)
	}
}

func TestForwardRecurrentReadsPrevious(t *testing.T) {
	// [in, out]: out = in + out(previous step)
	ann := ArtificialNeuralNet{InputCount: 1, OutputCount: 1, Nodes: make([]Node, 2),
		Connections: []Connection{
			{From: 0, To: 1, Weight: 1, Forward: true},
			{From: 1, To: 1, Weight: 1, Forward: false},
		}}
	for step, want := range []float64{1, 2, 3} {
		if got := ann.Forward([]float64{1})[0]; got != want {
			t.Fatalf("step %d: got %v want %v", step, got, want)
		}
	}
}

func TestForwardBiasAndActivation(t *testing.T) {
	ann := ArtificialNeuralNet{InputCount: 1, OutputCount: 1,
		Nodes:       []Node{{}, {Bias: -1, Activation: Abs}},
		Connections: []Connection{{From: 0, To: 1, Weight: 1, Forward: true}}}
	if got := ann.Forward([]float64{0.25})[0]; got != 0.75 {
		t.Fatalf("got %v want |0.25 - 1| = 0.75", got)
	}
}

func TestStepNodesCopiesCurrent(t *testing.T) {
	ann := ArtificialNeuralNet{Nodes: []Node{{Current: 1}, {Current: 2}}}
	ann.StepNodes()
	if ann.Nodes[0].Previous != 1 || ann.Nodes[1].Previous != 2 {
		t.Fatalf("got %+v", ann.Nodes)
	}
}

// ---------- Validate ----------

func TestValidateAcceptsMatchingSpec(t *testing.T) {
	ann := linearConnCPPN(1, 1, 1, 1, 0, Gaussian)
	spec := NetworkSpec{Inputs: 4, Outputs: 1,
		OutputShape: []map[ActivationFunc]bool{{Gaussian: true}}}
	noPanic(t, func() {
		if err := ann.Validate(spec); err != nil {
			t.Errorf("valid spec rejected: %v", err)
		}
	})
}

func TestValidateRejections(t *testing.T) {
	base := func() ArtificialNeuralNet { return linearConnCPPN(1, 1, 1, 1, 0, Gaussian) }
	recurrent := base()
	recurrent.Connections = append(recurrent.Connections, Connection{From: 4, To: 4, Forward: false})
	tooFew := base()
	tooFew.Nodes = tooFew.Nodes[:4]

	cases := []struct {
		name string
		ann  ArtificialNeuralNet
		spec NetworkSpec
	}{
		{"recurrent disallowed", recurrent, NetworkSpec{Inputs: 4, Outputs: 1}},
		{"input count", base(), NetworkSpec{Inputs: 3, Outputs: 1}},
		{"output count", base(), NetworkSpec{Inputs: 4, Outputs: 2}},
		{"too few nodes", tooFew, NetworkSpec{Inputs: 4, Outputs: 1}},
		{"wrong output activation", base(), NetworkSpec{Inputs: 4, Outputs: 1,
			OutputShape: []map[ActivationFunc]bool{{Sine: true}}}},
		{"activation present but false", base(), NetworkSpec{Inputs: 4, Outputs: 1,
			OutputShape: []map[ActivationFunc]bool{{Gaussian: false}}}},
		{"shape length mismatch", base(), NetworkSpec{Inputs: 4, Outputs: 1,
			OutputShape: []map[ActivationFunc]bool{{Gaussian: true}, {Gaussian: true}}}},
	}
	for _, c := range cases {
		noPanic(t, func() {
			if err := c.ann.Validate(c.spec); err == nil {
				t.Errorf("%s: accepted", c.name)
			}
		})
	}
}

func TestValidateAllowsRecurrentWhenSpecAllows(t *testing.T) {
	ann := linearConnCPPN(1, 1, 1, 1, 0, Linear)
	ann.Connections = append(ann.Connections, Connection{From: 4, To: 4, Forward: false})
	if err := ann.Validate(NetworkSpec{Inputs: 4, Outputs: 1, RecursiveAllowed: true}); err != nil {
		t.Fatalf("rejected: %v", err)
	}
}
