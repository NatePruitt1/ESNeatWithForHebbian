package esneatforhebbians

import "fmt"

// Node is a neuron in a Network. Its value is
// Activation(Bias + sum of weighted inputs).
//
// Current is the value from the most recent step and Previous the one from
// the step before. Forward connections read Current (their source was already
// computed this step); recurrent connections read Previous, which gives them
// exactly one step of delay. PreSum is the most recent sum before the
// activation function.
type Node struct {
	Bias       float64
	Activation ActivationFuncID

	Current, Previous float64
	PreSum            float64
}

// Connection is a weighted connection From -> To between node indices.
// Forward says whether it reads its source's current value (true) or its
// previous one (false, recurrent).
type Connection struct {
	From, To int
	Weight   float64
	Forward  bool
}

// Network is the evaluation engine every network uses. Nodes follow the
// layout rules in layout.go, and each Compute call evaluates inputs, then
// hidden nodes in index order, then outputs.
//
// AllowRecurrent controls whether recurrent connections are permitted. A
// network used as a CPPN must have it off, so every query depends only on its
// inputs.
//
// HebbianNetwork embeds a Network and only adds plasticity, so everything
// here applies to it as well.
type Network struct {
	InputCount, OutputCount int
	AllowRecurrent          bool
	Nodes                   []Node
	Connections             []Connection

	incoming [][]int // incoming[n]: indices into Connections that end at node n
}

// NewNetwork checks the layout rules and every node's activation function, and
// prepares the network for Compute. All values start at 0.
func NewNetwork(inputCount, outputCount int, nodes []Node, connections []Connection, allowRecurrent bool) (*Network, error) {
	if err := validateLayout(len(nodes), inputCount, outputCount, connections, allowRecurrent); err != nil {
		return nil, err
	}
	for i, n := range nodes {
		if !n.Activation.Valid() {
			return nil, fmt.Errorf("node %d has activation function %d, which is not valid", i, n.Activation)
		}
	}

	incoming := make([][]int, len(nodes))
	for ci, c := range connections {
		incoming[c.To] = append(incoming[c.To], ci)
	}
	net := &Network{
		InputCount:     inputCount,
		OutputCount:    outputCount,
		AllowRecurrent: allowRecurrent,
		Nodes:          nodes,
		Connections:    connections,
		incoming:       incoming,
	}
	net.Reset()
	return net, nil
}

// Reset sets every value to 0. Call it between episodes, or before each
// independent input so that only forward paths affect the output.
func (net *Network) Reset() {
	for i := range net.Nodes {
		net.Nodes[i].Current, net.Nodes[i].Previous, net.Nodes[i].PreSum = 0, 0, 0
	}
}

// Compute runs one step and returns a new slice with the output values. It
// panics if len(inputs) is not InputCount.
func (net *Network) Compute(inputs []float64) []float64 {
	out := make([]float64, net.OutputCount)
	net.computeInto(inputs, out)
	return out
}

// computeInto is Compute without the allocation: it writes the outputs to out,
// which must have OutputCount elements.
func (net *Network) computeInto(inputs, out []float64) {
	if len(inputs) != net.InputCount {
		panic(fmt.Sprintf("Compute got %d inputs, want %d", len(inputs), net.InputCount))
	}
	// A network without recurrent connections never reads Previous.
	if net.AllowRecurrent {
		for i := range net.Nodes {
			net.Nodes[i].Previous = net.Nodes[i].Current
		}
	}
	for i, x := range inputs {
		net.Nodes[i].PreSum = x
		net.Nodes[i].Current = x
	}
	for n := net.InputCount + net.OutputCount; n < len(net.Nodes); n++ {
		net.computeNode(n)
	}
	for o := range out {
		out[o] = net.computeNode(net.InputCount + o)
	}
}

// computeNode sums node n's bias and weighted inputs, applies its activation
// function, and stores the result in Current.
func (net *Network) computeNode(n int) float64 {
	node := &net.Nodes[n]
	sum := node.Bias
	for _, ci := range net.incoming[n] {
		c := &net.Connections[ci]
		if c.Forward {
			sum += c.Weight * net.Nodes[c.From].Current
		} else {
			sum += c.Weight * net.Nodes[c.From].Previous
		}
	}
	node.PreSum = sum
	node.Current = node.Activation.Activate(sum)
	return node.Current
}
