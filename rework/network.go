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
