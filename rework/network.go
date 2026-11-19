package rework

import "errors"

type Node struct {
	Bias              float64
	Activation        ActivationFunc
	Current, Previous float64
}

type Connection struct {
	From, To int
	Weight   float64
	Forward  bool
}

// Networks order nodes as [inputs..., outputs..., hidden...]
// Hidden nodes are ordered so that they are safe to loop through.
type ArtificialNeuralNet struct {
	InputCount, OutputCount int
	Nodes                   []Node // Structure [inputs..., outputs..., hidden...]
	Connections             []Connection
}

type NetworkSpec struct {
	RecursiveAllowed bool
	Inputs           int
	Outputs          int
	OutputShape      []map[ActivationFunc]bool
}

type nodeSubstratePair struct {
	node  Node
	sNode substrateNode
}

func ANNFromSubstrate(substrate *substrate) *ArtificialNeuralNet {
	// convert the nodes.
	nodes := make([]Node, len(substrate.inputNodes)+len(substrate.outputNodes)+len(substrate.hiddenNodes))

	// We want to make a relation nodes and pointers to their substrateNodes
	nodeRelation := make([]nodeSubstratePair, len(nodes))
	for i := 0; i < len(substrate.inputNodes); i += 1 {
		nodeRelation[i].sNode = substrate.inputNodes[i]
	}

	for i := 0; i < len(substrate.outputNodes); i += 1 {
		nodeRelation[i+len(substrate.inputNodes)].sNode = substrate.outputNodes[i]
	}

	for i := 0; i < len(substrate.hiddenNodes); i += 1 {
		nodeRelation[i+len(substrate.inputNodes)+len(substrate.outputNodes)].sNode = substrate.hiddenNodes[i]
	}

	for n := range nodeRelation {
		convertSubstrateNodeInPair(&nodeRelation[n])
	}

	connections := make([]Connection, len(substrate.conns))

	// Convert connections by finding the nodes that match the connection.
	for c := range connections {
		fromPoint := substrate.conns[c].From
		toPoint := substrate.conns[c].To

		var fromNode int
		var toNode int
		for n := range nodeRelation {
			if nodeRelation[n].sNode.Point == fromPoint {
				fromNode = n
			}

			if nodeRelation[n].sNode.Point == toPoint {
				toNode = n
			}
		}

		newConn := Connection{
			From:    fromNode,
			To:      toNode,
			Weight:  substrate.conns[c].values[WEIGHT_INDEX],
			Forward: !substrate.conns[c].recurrent,
		}

		connections[c] = newConn
	}

	for n := range nodeRelation {
		nodes[n] = nodeRelation[n].node
	}

	return &ArtificialNeuralNet{
		InputCount:  len(substrate.inputNodes),
		OutputCount: len(substrate.outputNodes),
		Nodes:       nodes,
		Connections: connections,
	}
}

func convertSubstrateNodeInPair(pair *nodeSubstratePair) {
	pair.node.Activation = chooseActivation(pair.sNode.values[:ActivationFunctionCount])
	pair.node.Bias = pair.sNode.values[ActivationFunctionCount]
}

func (n *ArtificialNeuralNet) Validate(ns NetworkSpec) error {
	if !ns.RecursiveAllowed {
		for c := range n.Connections {
			if n.Connections[c].Forward == false {
				return errors.New("Recursive connections in this net is disallowed.")
			}
		}
	}

	if n.InputCount != ns.Inputs {
		return errors.New("Input count mismatch from spec.")
	}

	if n.OutputCount != ns.Outputs {
		return errors.New("Output count mismatch from spec.")
	}

	if len(n.Nodes) < n.InputCount+n.OutputCount {
		return errors.New("Not enough nodes to account for inputs and outputs.")
	}

	if ns.OutputShape != nil {
		if len(ns.OutputShape) != ns.Outputs {
			return errors.New("Inconsistent spec, members in outputshape do not match output count.")
		}

		for oNode := n.InputCount; oNode < n.InputCount+n.OutputCount; oNode += 1 {
			act := n.Nodes[oNode].Activation
			allowedMap := ns.OutputShape[oNode-n.InputCount]
			val, ok := allowedMap[act]
			if !ok || !val {
				return errors.New("Output shape does not match spec.")
			}
		}
	}

	return nil
}

func (n *ArtificialNeuralNet) Forward(inputs []float64) []float64 {
	n.StepNodes()

	//Load inputs
	for i := range n.Nodes[:n.InputCount] {
		n.Nodes[i].Current = inputs[i]
	}

	// Recursive connections are not allowed on inputs, so no more stepping is needed.
	// Go through each hidden node, and get all of its "in" connections.
	// Compute the nodes new value and continue.
	hnodeOffset := n.InputCount + n.OutputCount
	for hnode := hnodeOffset; hnode < len(n.Nodes); hnode += 1 {
		n.ComputeNode(hnode)
	}

	// Compute all of the output variables. It is not possible to have a recursive connection with an output as an To node.
	outputs := make([]float64, n.OutputCount)
	for oNode := n.InputCount; oNode < n.InputCount+n.OutputCount; oNode += 1 {
		outputs[oNode-n.InputCount] = n.ComputeNode(oNode)
	}

	return outputs
}

func (n *ArtificialNeuralNet) StepNodes() {
	for i := range n.Nodes {
		n.Nodes[i].Previous = n.Nodes[i].Current
	}
}

func (n *ArtificialNeuralNet) ComputeNode(node int) float64 {
	nodeRef := &n.Nodes[node]

	connections := make([]Connection, 0)
	for c := range n.Connections {
		if n.Connections[c].To == node {
			connections = append(connections, n.Connections[c])
		}
	}

	sum := 0.0
	for c := range connections {
		if connections[c].Forward {
			sum += connections[c].Weight * n.Nodes[connections[c].From].Current
		} else {
			sum += connections[c].Weight * n.Nodes[connections[c].From].Previous
		}
	}

	nodeRef.Current = nodeRef.Activation.Compute(sum + nodeRef.Bias)
	return nodeRef.Current
}
