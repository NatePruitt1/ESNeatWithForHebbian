package esneatforhebbians

import (
	"fmt"
	"math"
)

// HebbianNode is a neuron in a HebbianNetwork.
//
// Current is the activation from the most recent Compute call and Previous is
// the one from the call before it. Forward connections read Current (their
// source has already been computed in this call); recurrent connections read
// Previous, which gives every recurrent connection exactly one call of delay.
type HebbianNode struct {
	Current, Previous float64
	PreSum            float64
	Activation        ActivationFuncID

	incoming []int // indices into HebbianNetwork.Connections; built by NewHebbianNetwork
}

type HebbianConnection struct {
	From, To                   int
	A, B, C, D, Eta, W, M, Dir float64
	Forward                    bool
}

// HebbianNetwork evaluates nodes in the same layout as AbstractNetwork: the
// first InputCount nodes are inputs, the next OutputCount are outputs, and the
// rest are hidden nodes sorted by increasing y. Each Compute call evaluates
// inputs, then hidden nodes in index order, then outputs, which computes every
// node after all of its forward sources.
type HebbianNetwork struct {
	InputCount, OutputCount int
	Nodes                   []HebbianNode
	Connections             []HebbianConnection
}

func NewHebbianNode(activation ActivationFuncID) HebbianNode {
	return HebbianNode{
		Current:    0.0,
		Previous:   0.0,
		Activation: activation,
		PreSum:     0.0,
		incoming:   make([]int, 0),
	}
}

// NewHebbianConnection creates a connection From -> To with Hebbian
// coefficients A, B, C, D and learning rate Eta. M_0 is the initial magnitude
// M, and the starting weight follows the same rule UpdateWeights uses:
// W = M * Dir.
func NewHebbianConnection(To, From int, A, B, C, D, Eta, M_0, Dir float64, forward bool) HebbianConnection {
	return HebbianConnection{
		From:    From,
		To:      To,
		A:       A,
		B:       B,
		C:       C,
		D:       D,
		Eta:     Eta,
		M:       M_0,
		Dir:     Dir,
		W:       M_0 * Dir,
		Forward: forward,
	}
}

// HebbianValueCount is the number of connection-CPPN outputs a Hebbian
// conversion reads: indices WeightIndex through EtaIndex in cppn.go.
const HebbianValueCount = EtaIndex + 1

// Validate checks the rules the connection CPPN is expected to enforce through
// its structure, so a CPPN that breaks them is caught when the network is
// built instead of producing a network that silently misbehaves:
//
//   - every value is finite (no NaN or ±Inf);
//   - M >= 0, since UpdateWeights keeps M non-negative and the sign of the
//     weight comes only from Dir;
//   - Dir is exactly -1 or +1, so it sets the weight's sign without scaling it;
//   - Eta >= 0: anti-Hebbian learning comes from negative fitness and
//     Dir = -1, so a negative learning rate would only add a second,
//     confusing sign flip;
//   - W == M * Dir, the relation UpdateWeights maintains.
func (c HebbianConnection) Validate() error {
	for _, v := range []struct {
		name  string
		value float64
	}{{"A", c.A}, {"B", c.B}, {"C", c.C}, {"D", c.D}, {"Eta", c.Eta}, {"M", c.M}, {"Dir", c.Dir}, {"W", c.W}} {
		if math.IsNaN(v.value) || math.IsInf(v.value, 0) {
			return fmt.Errorf("%s is %g; it must be finite", v.name, v.value)
		}
	}
	if c.M < 0 {
		return fmt.Errorf("M is %g; it must be >= 0", c.M)
	}
	if c.Dir != -1 && c.Dir != 1 {
		return fmt.Errorf("Dir is %g; it must be exactly -1 or +1", c.Dir)
	}
	if c.Eta < 0 {
		return fmt.Errorf("Eta is %g; it must be >= 0", c.Eta)
	}
	if c.W != c.M*c.Dir {
		return fmt.Errorf("W is %g but M * Dir is %g", c.W, c.M*c.Dir)
	}
	return nil
}

func (an *AbstractNetwork) ConvertToHebbian(inputCount, outputCount int) (*HebbianNetwork, error) {
	// Turn all of the abstract nodes into hebbian nodes.
	// Turn all of the abstract connections into hebbian connections. This involves bringing values
	// From a generic float slice to the actual variables.

	// The float slice values (which has M_0 at 0) are defined in cppn.go
	nodes := make([]HebbianNode, len(an.Nodes))
	for n, anode := range an.Nodes {
		nodes[n] = NewHebbianNode(anode.ActivationFunc)
	}

	connections := make([]HebbianConnection, len(an.Connections))
	for c, aconn := range an.Connections {
		if len(aconn.Values) < HebbianValueCount {
			return nil, fmt.Errorf("connection %d has %d CPPN values; Hebbian conversion needs %d", c, len(aconn.Values), HebbianValueCount)
		}
		connections[c] = NewHebbianConnection(aconn.To, aconn.From,
			aconn.Values[AIndex],
			aconn.Values[BIndex],
			aconn.Values[CIndex],
			aconn.Values[DIndex],
			aconn.Values[EtaIndex],
			math.Abs(aconn.Weight), // M_0
			aconn.Values[DirIndex], // Dir
			aconn.Forward)
		if err := connections[c].Validate(); err != nil {
			return nil, fmt.Errorf("connection %d (node %d -> %d): %w", c, aconn.From, aconn.To, err)
		}
	}

	return NewHebbianNetwork(inputCount, outputCount, nodes, connections)
}

// NewHebbianNetwork creates a network and builds each node's list of incoming
// connections, so Compute does not have to search the connection list. All
// activations start at 0. It returns an error naming the first layout rule a
// connection breaks (see layout.go); recurrent connections are always allowed.
func NewHebbianNetwork(inputCount, outputCount int, nodes []HebbianNode, connections []HebbianConnection) (*HebbianNetwork, error) {
	links := make([]link, len(connections))
	for i, c := range connections {
		links[i] = link{c.From, c.To, c.Forward}
	}
	if err := validateLayout(len(nodes), inputCount, outputCount, links, true); err != nil {
		return nil, err
	}
	hn := &HebbianNetwork{
		InputCount:  inputCount,
		OutputCount: outputCount,
		Nodes:       nodes,
		Connections: connections,
	}
	for i := range hn.Nodes {
		hn.Nodes[i].incoming = nil
	}
	for ci, c := range hn.Connections {
		hn.Nodes[c.To].incoming = append(hn.Nodes[c.To].incoming, ci)
	}
	hn.Reset()
	return hn, nil
}

// Reset sets every activation to 0. Call it between episodes, or before each
// input in one-shot tasks so that only forward paths affect the output.
func (hn *HebbianNetwork) Reset() {
	for i := range hn.Nodes {
		hn.Nodes[i].Current, hn.Nodes[i].Previous, hn.Nodes[i].PreSum = 0, 0, 0
	}
}

// UpdateWeights applies the ABCD Hebbian rule to every connection, using the
// activations from the most recent Compute call:
//
//	ΔM = Eta * fitness * (A*pre*post + B*pre + C*post + D)
//
// M is kept non-negative and the weight becomes W = M * d.
func (hn *HebbianNetwork) UpdateWeights(fitness float64) {
	for c := range hn.Connections {
		conn := &hn.Connections[c]
		pre := hn.Nodes[conn.From].Current
		post := hn.Nodes[conn.To].Current

		delM := conn.Eta * fitness * (conn.A*pre*post +
			conn.B*pre +
			conn.C*post +
			conn.D)

		conn.M = max(0, conn.M+delM)
		conn.W = conn.M * conn.Dir
	}
}

// Compute runs one step of the network and returns the output activations.
//
// Inputs pass through unchanged. Hidden nodes and then outputs are computed
// in index order; forward connections use their source's value from this
// step, and recurrent connections (lateral, backward and self-connections)
// use the value from the previous step. It panics if len(inputs) is not
// InputCount.
func (hn *HebbianNetwork) Compute(inputs []float64) []float64 {
	if len(inputs) != hn.InputCount {
		panic(fmt.Sprintf("Compute got %d inputs, want %d", len(inputs), hn.InputCount))
	}
	for i := range hn.Nodes {
		hn.Nodes[i].Previous = hn.Nodes[i].Current
	}

	for i, x := range inputs {
		hn.Nodes[i].PreSum = x
		hn.Nodes[i].Current = x
	}

	for n := hn.InputCount + hn.OutputCount; n < len(hn.Nodes); n++ {
		hn.computeNode(n)
	}

	outputs := make([]float64, hn.OutputCount)
	for o := range outputs {
		outputs[o] = hn.computeNode(hn.InputCount + o)
	}
	return outputs
}

// computeNode sums node n's weighted inputs, applies its activation function,
// and stores the result in Current.
func (hn *HebbianNetwork) computeNode(n int) float64 {
	node := &hn.Nodes[n]
	sum := 0.0
	for _, ci := range node.incoming {
		c := &hn.Connections[ci]
		src := &hn.Nodes[c.From]
		if c.Forward {
			sum += c.W * src.Current
		} else {
			sum += c.W * src.Previous
		}
	}
	node.PreSum = sum
	node.Current = node.Activation.Activate(sum)
	return node.Current
}
