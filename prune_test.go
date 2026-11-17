package esneatforhebbians

import (
	"slices"
	"testing"
)

func node(x, y float64, kind NodeKind) *SpacialNode {
	return &SpacialNode{x: x, y: y, kind: kind}
}

func conn(from, to *SpacialNode) *SpacialConnection {
	return NewSpacialConnection(from.x, from.y, to.x, to.y, []float64{1})
}

// TestPruneHandBuilt uses one input I, one output O and these hidden nodes:
//
//	I -> A -> B -> O      main path: kept
//	     A -> C -> A      cycle hanging off the path: kept
//	     B -> B           self-connection on the path: kept
//	I -> D                dead end (D reaches no output): removed
//	E -> O                no input reaches E: removed
//	F -> G -> F           island cycle, unattached: removed
//	C -> D                connection into the dead end: removed
func TestPruneHandBuilt(t *testing.T) {
	I := node(0, InputY, InputNode)
	O := node(0, OutputY, OutputNode)
	A := node(-0.5, -0.5, HiddenNode)
	B := node(-0.5, 0.5, HiddenNode)
	C := node(0.5, -0.5, HiddenNode)
	D := node(0.5, 0.5, HiddenNode)
	E := node(0.25, 0.25, HiddenNode)
	F := node(-0.25, 0.25, HiddenNode)
	G := node(-0.25, -0.25, HiddenNode)

	s := &substrate{
		inputs:  []*SpacialNode{I},
		hidden:  []*SpacialNode{A, B, C, D, E, F, G},
		outputs: []*SpacialNode{O},
		connections: []*SpacialConnection{
			conn(I, A), conn(A, B), conn(B, O),
			conn(A, C), conn(C, A),
			conn(B, B),
			conn(I, D),
			conn(E, O),
			conn(F, G), conn(G, F),
			conn(C, D),
		},
	}
	p := pruneSubstrate(s)

	if want := []*SpacialNode{A, B, C}; !slices.Equal(p.hidden, want) {
		t.Fatalf("kept hidden nodes %v, want A, B, C", p.hidden)
	}
	want := map[[2]*SpacialNode]bool{
		{I, A}: true, {A, B}: true, {B, O}: true, {A, C}: true, {C, A}: true, {B, B}: true,
	}
	byPos := map[position]*SpacialNode{}
	for _, n := range []*SpacialNode{I, O, A, B, C, D, E, F, G} {
		byPos[position{n.x, n.y}] = n
	}
	if len(p.connections) != len(want) {
		t.Fatalf("kept %d connections, want %d", len(p.connections), len(want))
	}
	for _, c := range p.connections {
		k := [2]*SpacialNode{byPos[position{c.fromx, c.fromy}], byPos[position{c.tox, c.toy}]}
		if !want[k] {
			t.Fatalf("unexpected connection (%g,%g)->(%g,%g)", c.fromx, c.fromy, c.tox, c.toy)
		}
	}
	if len(p.inputs) != 1 || len(p.outputs) != 1 {
		t.Fatal("inputs and outputs must always be kept")
	}
}

func TestPruneNoPath(t *testing.T) {
	I := node(0, InputY, InputNode)
	O := node(0, OutputY, OutputNode)
	A := node(0.5, 0.5, HiddenNode)
	p := pruneSubstrate(&substrate{
		inputs: []*SpacialNode{I}, hidden: []*SpacialNode{A}, outputs: []*SpacialNode{O},
		connections: []*SpacialConnection{conn(I, A)},
	})
	if len(p.hidden) != 0 || len(p.connections) != 0 || len(p.inputs) != 1 || len(p.outputs) != 1 {
		t.Fatalf("got %d hidden, %d connections; want only the input and output", len(p.hidden), len(p.connections))
	}
}

// TestPruneDiscovered checks pruning on a real search: afterwards every hidden
// node must be reachable from an input and reach an output, and pruning again
// must change nothing.
func TestPruneDiscovered(t *testing.T) {
	for _, level := range []int{0, 1, -1} {
		s := discover(t, level, 0)
		p := pruneSubstrate(s)
		checkInvariants(t, p)

		forward, backward := buildAdjacency(p.connections)
		fromInputs, toOutputs := reachable(p.inputs, forward), reachable(p.outputs, backward)
		for _, n := range p.hidden {
			if !fromInputs[position{n.x, n.y}] || !toOutputs[position{n.x, n.y}] {
				t.Fatalf("hidden node (%g,%g) kept but not on an input-output path", n.x, n.y)
			}
		}
		again := pruneSubstrate(p)
		if len(again.hidden) != len(p.hidden) || len(again.connections) != len(p.connections) {
			t.Fatal("pruning twice changed the result")
		}
		t.Logf("iterationLevel=%d: hidden %d -> %d, connections %d -> %d",
			level, len(s.hidden), len(p.hidden), len(s.connections), len(p.connections))
	}
}
