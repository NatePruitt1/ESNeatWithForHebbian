package esneatforhebbians

import "fmt"

// Substrate layout: inputs lie on the bottom edge and outputs on the top edge.
// Hidden nodes are quadtree centers, which always lie strictly inside
// (-1, 1) x (-1, 1), so they can never coincide with an input or output.
const (
	InputY  = -1.0
	OutputY = 1.0
)

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

// buildRow creates n nodes of the given kind, evenly spaced along the
// horizontal line at y.
func buildRow(n int, y float64, kind NodeKind, ncppn NodeQuery) ([]*SpacialNode, error) {
	if n < 0 {
		return nil, fmt.Errorf("%s node count must not be negative, got %d", kind, n)
	}
	nodes := make([]*SpacialNode, n)
	for i, x := range evenlySpaced(n) {
		nodes[i] = NewSpacialNode(x, y, ncppn)
	}
	return nodes, nil
}

// BuildInputNodes creates n input nodes evenly spaced along y = InputY.
func BuildInputNodes(n int, ncppn NodeQuery) ([]*SpacialNode, error) {
	return buildRow(n, InputY, InputNode, ncppn)
}

// BuildOutputNodes creates n output nodes evenly spaced along y = OutputY.
func BuildOutputNodes(n int, ncppn NodeQuery) ([]*SpacialNode, error) {
	return buildRow(n, OutputY, OutputNode, ncppn)
}

// position identifies a point on the substrate. Exact float comparison is safe
// because quadtree centers are dyadic fractions, computed the same way each time.
type position struct {
	x, y float64
}

// HiddenNodes collects the hidden nodes found while building a network. It
// holds at most one node per position and remembers the order in which nodes
// were discovered.
type HiddenNodes struct {
	ncppn NodeQuery
	byPos map[position]*SpacialNode
	order []*SpacialNode
}

// NewHiddenNodes creates an empty set whose nodes take their values from ncppn.
func NewHiddenNodes(ncppn NodeQuery) *HiddenNodes {
	return &HiddenNodes{
		ncppn: ncppn,
		byPos: make(map[position]*SpacialNode),
	}
}

// Get returns the hidden node at (x, y), if there is one.
func (h *HiddenNodes) Get(x, y float64) (*SpacialNode, bool) {
	n, ok := h.byPos[position{x, y}]
	return n, ok
}

// Add returns the hidden node at (x, y), creating it if needed. added reports
// whether the node is new.
func (h *HiddenNodes) Add(x, y float64) (node *SpacialNode, added bool) {
	if n, ok := h.byPos[position{x, y}]; ok {
		return n, false
	}
	n := NewSpacialNode(x, y, h.ncppn)
	h.byPos[position{x, y}] = n
	h.order = append(h.order, n)
	return n, true
}

// AddTargets adds a hidden node at the target of every connection that does
// not already have one, and returns only the newly created nodes, in order.
// Use it after an outgoing search (Algorithm 3, steps 1 and 2): the new nodes
// are the ones the next round should explore.
func (h *HiddenNodes) AddTargets(connections []*SpacialConnection) []*SpacialNode {
	var added []*SpacialNode
	for _, c := range connections {
		if n, ok := h.Add(c.tox, c.toy); ok {
			added = append(added, n)
		}
	}
	return added
}

// Len returns the number of hidden nodes.
func (h *HiddenNodes) Len() int {
	return len(h.order)
}

// Nodes returns the hidden nodes in the order they were discovered. The slice
// belongs to h; do not modify it.
func (h *HiddenNodes) Nodes() []*SpacialNode {
	return h.order
}
