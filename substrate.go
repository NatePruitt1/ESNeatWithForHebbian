package esneatforhebbians

import (
	"fmt"
	"slices"
)

// This file implements Algorithm 3, steps 1-3, of ES-HyperNEAT: placing the
// inputs and outputs, and discovering hidden nodes and connections between
// them. Step 4 (pruning) is in prune.go, and generate.go runs everything in
// order.

// Inputs lie on the bottom edge of the substrate and outputs on the top edge.
// Hidden nodes are quadtree centers, which always lie strictly inside
// (-1, 1) x (-1, 1), so they never coincide with an input or output.
const (
	InputY  = -1.0
	OutputY = 1.0
)

// SearchParams controls the ES-HyperNEAT search.
type SearchParams struct {
	// InitialDepth is the quadtree depth every search subdivides to, and
	// MaxDepth the deepest it may go where the CPPN's weights vary. The finest
	// grid has 2^MaxDepth points per axis.
	InitialDepth, MaxDepth int

	// DivisionThreshold is the variance above which a quadtree node is
	// subdivided (Algorithm 1). VarianceThreshold is the variance below which
	// a region is treated as one point during extraction, and BandThreshold
	// how much a point's weight must differ from its neighbors to be
	// extracted (Algorithm 2).
	DivisionThreshold, VarianceThreshold, BandThreshold float64

	// IterationLevel is the number of hidden -> hidden exploration rounds; a
	// negative value keeps exploring until no new hidden nodes are found.
	IterationLevel int

	// MaxHiddenNodes stops hidden -> hidden exploration once this many hidden
	// nodes exist (it may overshoot by the targets of one search). Zero or
	// less means no limit.
	MaxHiddenNodes int
}

// substrate holds the nodes and connections found by Algorithm 3, before they
// are pruned and turned into an AbstractNetwork.
type substrate struct {
	inputs, outputs []AbstractNode
	hidden          []AbstractNode // in discovery order
	connections     []*SpacialConnection
}

// newNode creates a node at p with its node-CPPN values and the activation
// function those values choose.
func newNode(p Point, ncppn NodeQuery) AbstractNode {
	values := slices.Clone(ncppn.QueryNode(p))
	return AbstractNode{Point: p, Activation: chooseActivation(values), Values: values}
}

// evenlySpaced returns n x-positions that divide [-1, 1] into n+1 equal gaps,
// leaving the endpoints free: 1 -> [0], 2 -> [-1/3, 1/3], 3 -> [-0.5, 0, 0.5].
func evenlySpaced(n int) []float64 {
	xs := make([]float64, n)
	spacing := 2.0 / float64(n+1)
	for i := range xs {
		xs[i] = -1 + spacing*float64(i+1)
	}
	return xs
}

// buildRow creates n nodes evenly spaced along the horizontal line at y.
func buildRow(n int, y float64, ncppn NodeQuery) []AbstractNode {
	nodes := make([]AbstractNode, n)
	for i, x := range evenlySpaced(n) {
		nodes[i] = newNode(Point{x, y}, ncppn)
	}
	return nodes
}

// hiddenNodes collects the hidden nodes found during the search: at most one
// per point, in the order they were discovered.
type hiddenNodes struct {
	ncppn NodeQuery
	index map[Point]int // position in nodes
	nodes []AbstractNode
}

func newHiddenNodes(ncppn NodeQuery) *hiddenNodes {
	return &hiddenNodes{ncppn: ncppn, index: make(map[Point]int)}
}

func (h *hiddenNodes) has(p Point) bool {
	_, ok := h.index[p]
	return ok
}

// addTargets adds a hidden node at the target of every connection that does
// not already have one, and returns the points of the new nodes, in order.
// These are the nodes the next exploration round searches from.
func (h *hiddenNodes) addTargets(connections []*SpacialConnection) []Point {
	var added []Point
	for _, c := range connections {
		if h.has(c.To) {
			continue
		}
		h.index[c.To] = len(h.nodes)
		h.nodes = append(h.nodes, newNode(c.To, h.ncppn))
		added = append(added, c.To)
	}
	return added
}

// discoverSubstrate runs Algorithm 3, steps 1-3:
//
//  1. inputs -> hidden: search outgoing connections from every input; each
//     target becomes a hidden node.
//  2. hidden -> hidden: search outgoing connections from the hidden nodes found
//     in the previous round, for params.IterationLevel rounds. Connections to
//     nodes that already exist are kept, so this step can create recurrent
//     connections and self-connections.
//  3. hidden -> outputs: search incoming connections to every output and keep
//     only those whose source is an existing hidden node.
//
// If allowRecurrent is false, every search keeps only forward connections, so
// no recurrent connection enters the substrate.
func discoverSubstrate(cppn ConnectionQuery, ncppn NodeQuery, allowRecurrent bool, inputCount, outputCount int, params SearchParams) (*substrate, error) {
	if inputCount < 0 || outputCount < 0 {
		return nil, fmt.Errorf("input and output counts must not be negative, got %d and %d", inputCount, outputCount)
	}
	s := &substrate{
		inputs:  buildRow(inputCount, InputY, ncppn),
		outputs: buildRow(outputCount, OutputY, ncppn),
	}
	// Inputs pass sensor values through unchanged.
	for i := range s.inputs {
		s.inputs[i].Activation = Linear
	}

	// search runs Algorithms 1 and 2 around one point. Connections from
	// different searches always differ in their fixed endpoint, so each search
	// can start from an empty list without creating duplicates.
	search := func(p Point, outgoing bool) []*SpacialConnection {
		root := DivisionAndInitialization(p, outgoing, cppn, params.InitialDepth, params.MaxDepth, params.DivisionThreshold)
		found := PruningAndExtraction(p, nil, outgoing, cppn, root, params.VarianceThreshold, params.BandThreshold)
		if !allowRecurrent {
			found = forwardOnly(found)
		}
		return found
	}

	hidden := newHiddenNodes(ncppn)

	// Step 1: inputs -> hidden.
	var unexplored []Point
	for _, in := range s.inputs {
		found := search(in.Point, true)
		s.connections = append(s.connections, found...)
		unexplored = append(unexplored, hidden.addTargets(found)...)
	}

	// Step 2: hidden -> hidden.
	limitReached := func() bool {
		return params.MaxHiddenNodes > 0 && len(hidden.nodes) >= params.MaxHiddenNodes
	}
	for round := 0; len(unexplored) > 0 && (params.IterationLevel < 0 || round < params.IterationLevel); round++ {
		var next []Point
		for _, p := range unexplored {
			if limitReached() {
				break
			}
			found := search(p, true)
			s.connections = append(s.connections, found...)
			next = append(next, hidden.addTargets(found)...)
		}
		unexplored = next
	}

	// Step 3: hidden -> outputs.
	for _, out := range s.outputs {
		for _, c := range search(out.Point, false) {
			if hidden.has(c.From) {
				s.connections = append(s.connections, c)
			}
		}
	}

	s.hidden = hidden.nodes
	return s, nil
}
