// AI-generated: written by Claude (Claude Code, claude-opus-5-5) as a
// regression suite for the rework package. Review before relying on it.

package rework

import "testing"

func TestBindConnRejections(t *testing.T) {
	ann := linearConnCPPN(1, 1, 1, 1, 0, Linear)
	cases := []struct {
		name string
		spec NetworkSpec
	}{
		{"fewer than 4 inputs", NetworkSpec{Inputs: 3, Outputs: 1}},
		{"no outputs", NetworkSpec{Inputs: 4, Outputs: 0}},
		{"recursion allowed", NetworkSpec{Inputs: 4, Outputs: 1, RecursiveAllowed: true}},
		{"spec does not match net", NetworkSpec{Inputs: 4, Outputs: 2}},
	}
	for _, c := range cases {
		if _, err := BindANNForConnectionGeneration(ann, c.spec); err == nil {
			t.Errorf("%s: bound", c.name)
		}
	}
}

func TestBindConnRejectsRecurrentCPPN(t *testing.T) {
	ann := linearConnCPPN(1, 1, 1, 1, 0, Linear)
	ann.Connections = append(ann.Connections, Connection{From: 4, To: 4, Weight: 1, Forward: false})
	if _, err := BindANNForConnectionGeneration(ann, connSpec); err == nil {
		t.Errorf("recurrent CPPN bound with RecursiveAllowed=false")
	}
	spec := connSpec
	spec.RecursiveAllowed = true
	if _, err := BindANNForConnectionGeneration(ann, spec); err == nil {
		t.Errorf("recurrent CPPN bound with RecursiveAllowed=true; Sample would not be pure")
	}
}

func TestBindNodeRejections(t *testing.T) {
	ann := constNodeCPPN(Sigmoid)
	cases := []struct {
		name string
		spec NetworkSpec
	}{
		{"fewer than 2 inputs", NetworkSpec{Inputs: 1, Outputs: ActivationFunctionCount}},
		{"too few outputs", NetworkSpec{Inputs: 2, Outputs: ActivationFunctionCount - 1}},
		{"recursion allowed", NetworkSpec{Inputs: 2, Outputs: ActivationFunctionCount, RecursiveAllowed: true}},
		{"spec does not match net", NetworkSpec{Inputs: 2, Outputs: ActivationFunctionCount + 1}},
	}
	for _, c := range cases {
		if _, err := BindANNForNodeGeneration(ann, c.spec); err == nil {
			t.Errorf("%s: bound", c.name)
		}
	}
}

func TestConnSampleArgumentOrder(t *testing.T) {
	// w = x1 + 10*y1 + 100*x2 + 1000*y2
	g := mustBindConn(t, linearConnCPPN(1, 10, 100, 1000, 0, Linear))
	w, vals := g.Sample(Point{1, 2}, Point{3, 4})
	if w != 4321 || vals[WEIGHT_INDEX] != w {
		t.Fatalf("got w=%v vals=%v, want 4321", w, vals)
	}
}

func TestConnSampleIsPure(t *testing.T) {
	g := mustBindConn(t, randomConnCPPN(1))
	a, _ := g.Sample(Point{0.1, 0.2}, Point{0.3, 0.4})
	g.Sample(Point{-0.9, 0.9}, Point{0.5, -0.5})
	b, _ := g.Sample(Point{0.1, 0.2}, Point{0.3, 0.4})
	if a != b {
		t.Fatalf("same query gave %v then %v", a, b)
	}
}

func TestConnSampleValuesNotShared(t *testing.T) {
	g := mustBindConn(t, linearConnCPPN(0, 0, 1, 0, 0, Linear))
	_, v1 := g.Sample(Point{}, Point{0.25, 0})
	_, v2 := g.Sample(Point{}, Point{0.5, 0})
	if v1[0] != 0.25 || v2[0] != 0.5 {
		t.Fatalf("value slices alias each other: %v %v", v1, v2)
	}
}

func TestNodeSamplePicksArgmax(t *testing.T) {
	for a := ActivationFunc(0); a < ActivationFunctionCount; a++ {
		b := mustBindNode(t, constNodeCPPN(a))
		act, vals := b.Sample(Point{0.3, -0.3})
		if act != a {
			t.Errorf("got %d want %d", act, a)
		}
		if len(vals) != ActivationFunctionCount {
			t.Errorf("got %d values", len(vals))
		}
	}
}

func TestNodeSampleArgumentOrder(t *testing.T) {
	// output 0 = x, output 1 = y
	ann := constNodeCPPN(Linear)
	ann.Nodes[2].Bias = 0
	ann.Connections = []Connection{
		{From: 0, To: 2, Weight: 1, Forward: true},
		{From: 1, To: 3, Weight: 1, Forward: true},
	}
	b := mustBindNode(t, ann)
	_, vals := b.Sample(Point{0.25, 0.75})
	if vals[0] != 0.25 || vals[1] != 0.75 {
		t.Fatalf("got %v, want x then y", vals[:2])
	}
}
