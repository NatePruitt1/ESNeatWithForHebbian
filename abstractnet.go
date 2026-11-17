package esneatforhebbians

// AbstractNetwork is the hand-off between the search and the runtime
// networks. It keeps everything the search produced (positions, chosen
// activation functions, and every CPPN output) so that different runtime
// networks can be built from it: ConvertToNetwork gives a fixed-weight
// Network, and ConvertToHebbian a HebbianNetwork.
//
// Nodes follow the layout rules in layout.go: the first InputCount are
// inputs, the next OutputCount are outputs, and the rest are hidden nodes
// sorted by y (then x).
type AbstractNetwork struct {
	InputCount, OutputCount int
	Nodes                   []AbstractNode
	Connections             []AbstractConnection
}

// AbstractNode is a node on the substrate. Values holds all of the node CPPN's
// outputs at its Point; Activation is the function those outputs chose (always
// Linear for inputs).
type AbstractNode struct {
	Point
	Activation ActivationFuncID
	Values     []float64
}

// AbstractConnection is a runtime Connection plus all of the connection CPPN's
// outputs for it (laid out by the index constants in cppn.go).
type AbstractConnection struct {
	Connection
	Values []float64
}

// connections returns the runtime part of every connection.
func (an *AbstractNetwork) connections() []Connection {
	conns := make([]Connection, len(an.Connections))
	for i, c := range an.Connections {
		conns[i] = c.Connection
	}
	return conns
}

// nodes returns a runtime Node for every abstract node, with its activation
// function and a bias of 0.
func (an *AbstractNetwork) nodes() []Node {
	nodes := make([]Node, len(an.Nodes))
	for i, n := range an.Nodes {
		nodes[i] = Node{Activation: n.Activation}
	}
	return nodes
}

// ConvertToNetwork builds a fixed-weight Network. Each connection keeps its
// CPPN weight. If allowRecurrent is false, recurrent connections are left out,
// giving a purely feed-forward network.
func (an *AbstractNetwork) ConvertToNetwork(allowRecurrent bool) (*Network, error) {
	conns := an.connections()
	if !allowRecurrent {
		var forward []Connection
		for _, c := range conns {
			if c.Forward {
				forward = append(forward, c)
			}
		}
		conns = forward
	}
	return NewNetwork(an.InputCount, an.OutputCount, an.nodes(), conns, allowRecurrent)
}
