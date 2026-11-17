package esneatforhebbians

import (
	"fmt"
	"math"
)

// DenseSpec describes the shape of a fully connected, layered network and its
// fixed output activations. Two genomes fill it in: one weight per connection
// and one activation function per hidden node. It is built as an
// AbstractNetwork, so it goes through the same conversion and checks as any
// other network, and BuildCPPN / BuildNodeCPPN turn it straight into a CPPN:
//
//	spec := DenseSpec{Inputs: 6, Outputs: 1, Hidden: []int{4, 4},
//		OutputActs: []ActivationFuncID{Tanh}, BiasInput: 4}
//	weights := make([]float64, spec.WeightGenomeLength())
//	acts := make([]ActivationFuncID, spec.ActivationGenomeLength())
//	cppn, err := spec.BuildCPPN(weights, acts)
//
// Structure (fixed by the spec):
//
//   - Nodes use the usual [inputs..., outputs..., hidden...] layout. Inputs sit
//     at y = -1 and outputs at y = 1; hidden layer k of L sits at
//     y = -1 + 2k/(L+1). Within each row, x is evenly spaced. Hidden nodes are
//     listed layer by layer, left to right, which is sorted by y then x.
//   - Each hidden layer is fully connected from the layer below it (the
//     inputs, for the first layer), and the outputs are fully connected from
//     the last hidden layer (or from the inputs, if there are no hidden
//     layers). Every connection goes to a greater y, so all are forward and
//     the network is valid with recurrence off, as a CPPN requires.
//   - If BiasInput is an input index, that input is also connected to every
//     hidden and output node it does not already feed, giving each node its
//     own evolvable bias. Use -1 for no bias connections.
//
// Output activations are fixed by OutputActs, since a CPPN's output functions
// enforce its value rules (for example Sign for Dir, Sigmoid for Eta).
type DenseSpec struct {
	Inputs, Outputs int
	Hidden          []int              // layer sizes, bottom to top
	OutputActs      []ActivationFuncID // one fixed function per output
	BiasInput       int                // input connected to every node as a bias; -1 for none
}

// Validate checks that the spec describes a buildable network.
func (s DenseSpec) Validate() error {
	if s.Inputs < 1 || s.Outputs < 1 {
		return fmt.Errorf("dense spec needs at least 1 input and 1 output, got %d and %d", s.Inputs, s.Outputs)
	}
	for i, n := range s.Hidden {
		if n < 1 {
			return fmt.Errorf("hidden layer %d has %d nodes; every layer needs at least 1", i, n)
		}
	}
	if len(s.OutputActs) != s.Outputs {
		return fmt.Errorf("dense spec has %d outputs but %d OutputActs", s.Outputs, len(s.OutputActs))
	}
	for i, id := range s.OutputActs {
		if !id.Valid() {
			return fmt.Errorf("OutputActs[%d] is %d, not a valid activation function", i, id)
		}
	}
	if s.BiasInput < -1 || s.BiasInput >= s.Inputs {
		return fmt.Errorf("BiasInput is %d; it must be -1 or an input index 0..%d", s.BiasInput, s.Inputs-1)
	}
	return nil
}

// HiddenCount returns the total number of hidden nodes.
func (s DenseSpec) HiddenCount() int {
	total := 0
	for _, n := range s.Hidden {
		total += n
	}
	return total
}

// WeightGenomeLength returns the number of weights the build functions
// expect. Weights are used target by target: hidden layers bottom to top,
// then the outputs; each target node left to right. For each target, its
// sources come in order: the previous layer left to right, then the bias
// input if it is not already one of them.
func (s DenseSpec) WeightGenomeLength() int {
	return len(s.links())
}

// ActivationGenomeLength returns the number of activation functions the build
// functions expect: one per hidden node, in layout order (layer by layer,
// left to right). Any valid ActivationFuncID may be used, including the
// CPPN-only ones.
func (s DenseSpec) ActivationGenomeLength() int {
	return s.HiddenCount()
}

// rows returns the node indices of each row of the network, bottom to top:
// inputs, each hidden layer, then outputs.
func (s DenseSpec) rows() [][]int {
	seq := func(start, n int) []int {
		r := make([]int, n)
		for i := range r {
			r[i] = start + i
		}
		return r
	}
	rows := [][]int{seq(0, s.Inputs)}
	next := s.Inputs + s.Outputs
	for _, n := range s.Hidden {
		rows = append(rows, seq(next, n))
		next += n
	}
	return append(rows, seq(s.Inputs, s.Outputs))
}

// links returns every connection as (from, to), in weight-genome order.
func (s DenseSpec) links() [][2]int {
	rows := s.rows()
	var links [][2]int
	for r := 1; r < len(rows); r++ {
		for _, to := range rows[r] {
			for _, from := range rows[r-1] {
				links = append(links, [2]int{from, to})
			}
			// Rows above the first hidden layer do not see the inputs, so
			// give them the bias input separately.
			if s.BiasInput >= 0 && r > 1 {
				links = append(links, [2]int{s.BiasInput, to})
			}
		}
	}
	return links
}

// BuildAbstract builds the dense network described by s, with weights from
// weights and hidden activation functions from acts. It returns an error if
// the spec is invalid, a genome has the wrong length, a weight is not finite,
// or an activation function ID is not valid.
func (s DenseSpec) BuildAbstract(weights []float64, acts []ActivationFuncID) (*AbstractNetwork, error) {
	if err := s.Validate(); err != nil {
		return nil, err
	}
	if n := s.WeightGenomeLength(); len(weights) != n {
		return nil, fmt.Errorf("weight genome has %d values; this spec needs %d", len(weights), n)
	}
	if n := s.ActivationGenomeLength(); len(acts) != n {
		return nil, fmt.Errorf("activation genome has %d values; this spec needs %d", len(acts), n)
	}
	for i, id := range acts {
		if !id.Valid() {
			return nil, fmt.Errorf("activation genome value %d is %d, not a valid activation function", i, id)
		}
	}

	nodes := make([]AbstractNode, s.Inputs+s.Outputs+s.HiddenCount())
	for i, x := range evenlySpaced(s.Inputs) {
		nodes[i] = AbstractNode{Posx: x, Posy: InputY, ActivationFunc: Linear}
	}
	for o, x := range evenlySpaced(s.Outputs) {
		nodes[s.Inputs+o] = AbstractNode{Posx: x, Posy: OutputY, ActivationFunc: s.OutputActs[o]}
	}
	h := 0 // position among hidden nodes, for the activation genome
	for k, row := range s.rows()[1 : len(s.Hidden)+1] {
		y := -1 + 2*float64(k+1)/float64(len(s.Hidden)+1)
		xs := evenlySpaced(len(row))
		for i, n := range row {
			nodes[n] = AbstractNode{Posx: xs[i], Posy: y, ActivationFunc: acts[h]}
			h++
		}
	}

	links := s.links()
	connections := make([]AbstractConnection, len(links))
	for i, l := range links {
		w := weights[i]
		if math.IsNaN(w) || math.IsInf(w, 0) {
			return nil, fmt.Errorf("weight genome value %d is %g; weights must be finite", i, w)
		}
		connections[i] = AbstractConnection{
			From:    l[0],
			To:      l[1],
			Weight:  w,
			Values:  []float64{w},
			Forward: true,
		}
	}

	an := &AbstractNetwork{Nodes: nodes, Connections: connections}

	// The structure is forward-only by construction; check it with the same
	// rules the networks enforce, so a mistake here cannot slip through.
	check := make([]link, len(connections))
	for i, c := range connections {
		check[i] = link{c.From, c.To, an.Nodes[c.To].Posy > an.Nodes[c.From].Posy}
	}
	if err := validateLayout(len(nodes), s.Inputs, s.Outputs, check, false); err != nil {
		return nil, fmt.Errorf("dense network breaks the layout rules: %w", err)
	}
	return an, nil
}

// BuildNetwork builds the dense network as a feed-forward Network.
func (s DenseSpec) BuildNetwork(weights []float64, acts []ActivationFuncID) (*Network, error) {
	an, err := s.BuildAbstract(weights, acts)
	if err != nil {
		return nil, err
	}
	return an.ConvertToNetwork(s.Inputs, s.Outputs, false)
}

// BuildCPPN builds the dense network as a connection CPPN. The spec must have
// 6 inputs (x1, y1, x2, y2, bias, networkType); BiasInput is usually 4.
func (s DenseSpec) BuildCPPN(weights []float64, acts []ActivationFuncID) (*NetworkCPPN, error) {
	net, err := s.BuildNetwork(weights, acts)
	if err != nil {
		return nil, err
	}
	return NewNetworkCPPN(net)
}

// BuildNodeCPPN builds the dense network as a node CPPN. The spec must have 4
// inputs (x, y, bias, networkType); BiasInput is usually 2.
func (s DenseSpec) BuildNodeCPPN(weights []float64, acts []ActivationFuncID) (*NetworkNodeCPPN, error) {
	net, err := s.BuildNetwork(weights, acts)
	if err != nil {
		return nil, err
	}
	return NewNetworkNodeCPPN(net)
}
