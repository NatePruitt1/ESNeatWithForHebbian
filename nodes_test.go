package esneatforhebbians

import "testing"

// posCPPN returns each node's position as its values, so tests can check them.
type posCPPN struct{}

func (posCPPN) QueryNode(x, y float64) []float64 { return []float64{x, y} }

func TestEvenlySpaced(t *testing.T) {
	cases := map[int][]float64{
		0: {},
		1: {0},
		2: {-1.0 / 3, 1.0 / 3},
		3: {-0.5, 0, 0.5},
	}
	for n, want := range cases {
		got := evenlySpaced(n)
		if len(got) != len(want) {
			t.Fatalf("evenlySpaced(%d) = %v, want %v", n, got, want)
		}
		for i := range want {
			if d := got[i] - want[i]; d > 1e-12 || d < -1e-12 {
				t.Fatalf("evenlySpaced(%d) = %v, want %v", n, got, want)
			}
		}
	}
}

func TestBuildRows(t *testing.T) {
	in, err := BuildInputNodes(3, posCPPN{})
	if err != nil {
		t.Fatal(err)
	}
	out, err := BuildOutputNodes(2, posCPPN{})
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range in {
		if n.y != InputY || n.kind != InputNode || n.values[0] != n.x || n.values[1] != n.y {
			t.Fatalf("bad input node %+v", n)
		}
	}
	for _, n := range out {
		if n.y != OutputY || n.kind != OutputNode || n.values[0] != n.x || n.values[1] != n.y {
			t.Fatalf("bad output node %+v", n)
		}
	}
	if in[0].x != -0.5 || in[1].x != 0 || in[2].x != 0.5 {
		t.Fatalf("input x positions %v %v %v", in[0].x, in[1].x, in[2].x)
	}
	if _, err := BuildInputNodes(-1, posCPPN{}); err == nil {
		t.Fatal("negative count should be an error")
	}
}

func TestHiddenNodes(t *testing.T) {
	h := NewHiddenNodes(posCPPN{})
	v := []float64{1}
	conns := []*SpacialConnection{
		NewSpacialConnection(0, -1, 0.25, 0.5, v),
		NewSpacialConnection(0.5, -1, 0.25, 0.5, v), // same target: no new node
		NewSpacialConnection(0, -1, -0.75, 0.125, v),
	}
	added := h.AddTargets(conns)
	if len(added) != 2 || h.Len() != 2 {
		t.Fatalf("added %d, len %d, want 2 and 2", len(added), h.Len())
	}
	if added[0].x != 0.25 || added[1].x != -0.75 || added[0].kind != HiddenNode {
		t.Fatalf("wrong nodes or order: %+v %+v", added[0], added[1])
	}
	if again := h.AddTargets(conns); len(again) != 0 {
		t.Fatalf("re-adding found %d new nodes", len(again))
	}
	if n, ok := h.Get(-0.75, 0.125); !ok || n != added[1] {
		t.Fatal("Get did not return the existing node")
	}
	if _, ok := h.Get(0.9, 0.9); ok {
		t.Fatal("Get found a node that was never added")
	}
	if n, added := h.Add(0.25, 0.5); added || n.values[0] != 0.25 {
		t.Fatal("Add should return the existing node")
	}
}
