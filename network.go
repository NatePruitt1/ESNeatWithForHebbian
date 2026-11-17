package esneatforhebbians

import "fmt"

// NetworkNode is a neuron in a Network. Its activation is
// Activation(Bias + sum of weighted inputs). A nil Activation means linear
// (the identity). Any function works, so a Network can use activation
// functions beyond ActivationFuncID (sine, absolute value, ...), which is what
// lets it act as a CPPN.
type NetworkNode struct {
	Bias       float64
	Activation func(float64) float64

	// Current is the activation from the most recent Compute call and Previous
	// the one from the call before it.
	Current, Previous float64
}

// NetworkConnection is a fixed-weight connection From -> To. A forward
// connection reads its source's value from the current step; a recurrent one
// (Forward == false) reads the value from the previous step.
type NetworkConnection struct {
	From, To int
	Weight   float64
	Forward  bool
}

// Network is a fixed-weight neural network that follows the same layout and
// evaluation rules as HebbianNetwork (see layout.go): nodes are laid out as
// [inputs..., outputs..., hidden...], and each step evaluates inputs, then
// hidden nodes in index order, then outputs.
//
// AllowRecurrent controls whether recurrent connections are permitted. A
// network used as a CPPN must have it off, so every query depends only on its
// inputs.
type Network struct {
	InputCount, OutputCount int
	AllowRecurrent          bool
	Nodes                   []NetworkNode
	Connections             []NetworkConnection

	incoming [][]int // incoming[n] holds indices into Connections ending at n
}

// NewNetwork checks the layout rules and prepares the network for Compute. It
// returns an error naming the first rule a connection breaks, including any
// recurrent connection when allowRecurrent is false. All activations start
// at 0.
func NewNetwork(inputCount, outputCount int, nodes []NetworkNode, connections []NetworkConnection, allowRecurrent bool) (*Network, error) {
	links := make([]link, len(connections))
	for i, c := range connections {
		links[i] = link{c.From, c.To, c.Forward}
	}
	if err := validateLayout(len(nodes), inputCount, outputCount, links, allowRecurrent); err != nil {
		return nil, err
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

// Reset sets every activation to 0. Call it between episodes, or before each
// independent input so that only forward paths affect the output. A network
// without recurrent connections has no state between steps, so it never needs
// a reset.
func (net *Network) Reset() {
	for i := range net.Nodes {
		net.Nodes[i].Current, net.Nodes[i].Previous = 0, 0
	}
}

// Compute runs one step and returns a new slice with the output activations.
// It panics if len(inputs) is not InputCount.
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
	if net.AllowRecurrent {
		for i := range net.Nodes {
			net.Nodes[i].Previous = net.Nodes[i].Current
		}
	}
	for i, x := range inputs {
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
	if node.Activation != nil {
		sum = node.Activation(sum)
	}
	node.Current = sum
	return sum
}

// ConvertToNetwork builds a fixed-weight Network from an AbstractNetwork,
// which already uses the [inputs..., outputs..., hidden...] layout with hidden
// nodes sorted by y. Each connection keeps its CPPN weight and Forward flag;
// each node keeps its activation function and gets a bias of 0.
//
// If allowRecurrent is false, the abstract network's recurrent connections
// are left out, giving a purely feed-forward network.
func (an *AbstractNetwork) ConvertToNetwork(inputCount, outputCount int, allowRecurrent bool) (*Network, error) {
	nodes := make([]NetworkNode, len(an.Nodes))
	for i, n := range an.Nodes {
		nodes[i] = NetworkNode{Activation: n.ActivationFunc.Activate}
	}
	connections := make([]NetworkConnection, 0, len(an.Connections))
	for _, c := range an.Connections {
		if !c.Forward && !allowRecurrent {
			continue
		}
		connections = append(connections, NetworkConnection{From: c.From, To: c.To, Weight: c.Weight, Forward: c.Forward})
	}
	return NewNetwork(inputCount, outputCount, nodes, connections, allowRecurrent)
}
