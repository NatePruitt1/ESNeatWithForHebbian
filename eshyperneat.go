package esneatforhebbians

import (
	"cmp"
	"fmt"
	"math"
	"slices"
)

// The algorithms in this file implement Algorithm 1 (DivisionAndInitialization)
// and Algorithm 2 (PruningAndExtraction) of ES-HyperNEAT, as described in:
//
//	S. Risi and K. O. Stanley, "An Enhanced Hypercube-Based Encoding for
//	Evolving the Placement, Density, and Connectivity of Neurons,"
//	Artificial Life, vol. 18, no. 4, pp. 331-363, 2012.
//	doi:10.1162/ARTL_a_00071
//
// One deliberate difference from the paper's pseudocode: Variance is computed
// over all leaf descendants of a node, not only its direct children.

// sample queries the CPPN for the connection between the fixed point (a, b) and
// (x, y) and returns all of its outputs. If outgoing is true, (a, b) is the
// source; otherwise it is the target. The result may be reused by the CPPN.
func sample(cppn ConnectionQuery, a, b, x, y float64, outgoing bool) []float64 {
	if outgoing {
		return cppn.Query(a, b, x, y)
	}
	return cppn.Query(x, y, a, b)
}

// sampleWeight is like sample but returns only the weight output.
func sampleWeight(cppn ConnectionQuery, a, b, x, y float64, outgoing bool) float64 {
	return sample(cppn, a, b, x, y, outgoing)[WeightIndex]
}

// DivisionAndInitialization (Algorithm 1) builds a quadtree over the substrate
// for the point (a, b), subdividing wherever the CPPN's weights vary.
func DivisionAndInitialization(a, b float64, outgoing bool, cppn ConnectionQuery, initialDepth, maxDepth int, divThreshold float64) *QuadPoint {
	root := NewQuadPoint(0, 0, 1, 1)
	queue := []*QuadPoint{root}
	for len(queue) > 0 {
		p := queue[0]
		queue[0] = nil
		queue = queue[1:]

		r := p.width / 2
		nextLevel := p.level + 1
		p.children[0] = NewQuadPoint(p.x-r, p.y-r, r, nextLevel)
		p.children[1] = NewQuadPoint(p.x-r, p.y+r, r, nextLevel)
		p.children[2] = NewQuadPoint(p.x+r, p.y-r, r, nextLevel)
		p.children[3] = NewQuadPoint(p.x+r, p.y+r, r, nextLevel)
		for _, c := range p.children {
			c.values = slices.Clone(sample(cppn, a, b, c.x, c.y, outgoing))
			c.w = c.values[WeightIndex]
		}

		if p.level < initialDepth || (p.level < maxDepth && Variance(p) > divThreshold) {
			queue = append(queue, p.children...)
		}
	}

	return root
}

// PruningAndExtraction (Algorithm 2) walks the quadtree under p and appends a
// connection for every low-variance point that lies inside a band of the
// weight pattern. Each connection carries all of the CPPN's outputs for that
// point, sharing the node's values slice. It returns the updated connections.
func PruningAndExtraction(a, b float64, connections []*SpacialConnection, outgoing bool, cppn ConnectionQuery, p *QuadPoint, varThreshold, bandThreshold float64) []*SpacialConnection {
	for _, c := range p.children {
		if c == nil {
			continue
		}
		if Variance(c) >= varThreshold {
			connections = PruningAndExtraction(a, b, connections, outgoing, cppn, c, varThreshold, bandThreshold)
			continue
		}

		dleft := math.Abs(c.w - sampleWeight(cppn, a, b, c.x-p.width, c.y, outgoing))
		dright := math.Abs(c.w - sampleWeight(cppn, a, b, c.x+p.width, c.y, outgoing))
		dtop := math.Abs(c.w - sampleWeight(cppn, a, b, c.x, c.y-p.width, outgoing))
		dbottom := math.Abs(c.w - sampleWeight(cppn, a, b, c.x, c.y+p.width, outgoing))

		if max(min(dtop, dbottom), min(dleft, dright)) <= bandThreshold {
			continue
		}

		var con *SpacialConnection
		if outgoing {
			con = NewSpacialConnection(a, b, c.x, c.y, c.values)
		} else {
			con = NewSpacialConnection(c.x, c.y, a, b, c.values)
		}
		if !containsConnection(connections, con) {
			connections = append(connections, con)
		}
	}

	return connections
}

// GenerateAbstractFromCPPNs builds a network from a connection CPPN and a node
// CPPN using ES-HyperNEAT's Algorithm 3.
//
// bias and networkType are passed to every query of both CPPNs. bias is a
// constant input (commonly 1); networkType selects which network to build, so
// calling this with networkType 0, 1, 2, ... builds several different
// networks from the same pair of CPPNs.
//
// allowRecurrent controls whether the network may contain recurrent
// connections (lateral, backward and self-connections; see layout.go). When it
// is false, the search keeps only forward connections, so hidden nodes that
// could only be reached or only reach an output through a recurrent
// connection are never added or are pruned, and the result is purely
// feed-forward. Use false for networks that will act as CPPNs.
//
// iterationLevel is the number of hidden -> hidden exploration rounds; a
// negative value keeps exploring until no new hidden nodes are found.
// maxHiddenNodes caps exploration when it is positive (see discoverSubstrate).
//
// The result always satisfies the layout rules in layout.go for inputSize
// inputs and outputSize outputs.
func GenerateAbstractFromCPPNs(connCPPN CPPN, nodeCPPN NodeCPPN, bias float64, networkType int, allowRecurrent bool, inputSize, outputSize, initialDepth, maxDepth, iterationLevel, maxHiddenNodes int, varianceThreshold, bandThreshold, divisionThreshold float64) (*AbstractNetwork, error) {
	if n := connCPPN.GetOutputSize(); n <= WeightIndex {
		return nil, fmt.Errorf("connection CPPN has %d outputs; it needs at least %d to provide a weight", n, WeightIndex+1)
	}
	if n, need := nodeCPPN.GetOutputSize(), ActivationChannel+ActivationFuncCount; n < need {
		return nil, fmt.Errorf("node CPPN has %d outputs; it needs at least %d to choose an activation function", n, need)
	}

	sub, err := discoverSubstrate(BindCPPN(connCPPN, bias, networkType), BindNodeCPPN(nodeCPPN, bias, networkType), allowRecurrent, inputSize, outputSize, initialDepth, maxDepth, iterationLevel, maxHiddenNodes, varianceThreshold, bandThreshold, divisionThreshold)
	if err != nil {
		return nil, err
	}

	// Algorithm 3, step 4: keep only what lies on an input-to-output path.
	// With recurrence off, the substrate has only forward connections here, so
	// this also removes nodes that only recurrent connections had joined up.
	sub = pruneSubstrate(sub)

	network, err := convertSubstrate(sub)
	if err != nil {
		return nil, err
	}

	// Final check: the network must satisfy the same layout rules Network and
	// HebbianNetwork enforce, including no recurrent connections when they are
	// not allowed. Failing here means a bug in the construction above.
	links := make([]link, len(network.Connections))
	for i, c := range network.Connections {
		links[i] = link{c.From, c.To, c.Forward}
	}
	if err := validateLayout(len(network.Nodes), inputSize, outputSize, links, allowRecurrent); err != nil {
		return nil, fmt.Errorf("generated network breaks the layout rules: %w", err)
	}
	return &network, nil
}

// substrate holds the nodes and connections found by Algorithm 3, steps 1-3.
type substrate struct {
	inputs      []*SpacialNode
	hidden      []*SpacialNode // in discovery order
	outputs     []*SpacialNode
	connections []*SpacialConnection
}

// discoverSubstrate runs Algorithm 3, steps 1-3:
//
//  1. inputs -> hidden: search outgoing connections from every input; each
//     target becomes a hidden node.
//  2. hidden -> hidden: search outgoing connections from the hidden nodes found
//     in the previous round, for iterationLevel rounds (or, if iterationLevel is
//     negative, until a round finds no new nodes). Connections to nodes that
//     already exist are kept, so this step can create recurrent connections
//     and self-connections.
//  3. hidden -> outputs: search incoming connections to every output and keep
//     only those whose source is an existing hidden node. No new nodes are
//     created.
//
// If allowRecurrent is false, every search keeps only forward connections, so
// no recurrent connection enters the substrate and step 2 never explores a
// node that only a recurrent connection reached.
//
// If maxHiddenNodes is positive, step 2 stops exploring new nodes once the
// hidden set reaches that size. It may overshoot by the targets of one search,
// since all of a search's targets are added together.
func discoverSubstrate(connCPPN ConnectionQuery, nodeCPPN NodeQuery, allowRecurrent bool, inputSize, outputSize, initialDepth, maxDepth, iterationLevel, maxHiddenNodes int, varianceThreshold, bandThreshold, divisionThreshold float64) (*substrate, error) {
	// Inputs sit evenly spaced on the bottom edge (y = -1) and outputs on the
	// top edge (y = 1), dividing [-1, 1] into n+1 equal gaps: 1 node is at
	// x = 0, 2 nodes are at x = -1/3 and 1/3.
	inputNodes, err := BuildInputNodes(inputSize, nodeCPPN)
	if err != nil {
		return nil, err
	}
	outputNodes, err := BuildOutputNodes(outputSize, nodeCPPN)
	if err != nil {
		return nil, err
	}

	// search runs Algorithms 1 and 2 from node n. Each search starts with an
	// empty connection list: connections from different searches always differ
	// in their fixed endpoint, so they can never be duplicates of each other.
	search := func(n *SpacialNode, outgoing bool) []*SpacialConnection {
		root := DivisionAndInitialization(n.x, n.y, outgoing, connCPPN, initialDepth, maxDepth, divisionThreshold)
		found := PruningAndExtraction(n.x, n.y, nil, outgoing, connCPPN, root, varianceThreshold, bandThreshold)
		if !allowRecurrent {
			found = forwardOnly(found)
		}
		return found
	}

	hidden := NewHiddenNodes(nodeCPPN)
	var connections []*SpacialConnection

	// Step 1: inputs -> hidden.
	var unexplored []*SpacialNode
	for _, in := range inputNodes {
		found := search(in, true)
		connections = append(connections, found...)
		unexplored = append(unexplored, hidden.AddTargets(found)...)
	}

	// Step 2: hidden -> hidden.
	limitReached := func() bool {
		return maxHiddenNodes > 0 && hidden.Len() >= maxHiddenNodes
	}
	for round := 0; len(unexplored) > 0 && (iterationLevel < 0 || round < iterationLevel); round++ {
		var next []*SpacialNode
		for _, n := range unexplored {
			if limitReached() {
				break
			}
			found := search(n, true)
			connections = append(connections, found...)
			next = append(next, hidden.AddTargets(found)...)
		}
		unexplored = next
	}

	// Step 3: hidden -> outputs.
	for _, out := range outputNodes {
		for _, c := range search(out, false) {
			if _, ok := hidden.Get(c.fromx, c.fromy); ok {
				connections = append(connections, c)
			}
		}
	}

	return &substrate{
		inputs:      inputNodes,
		hidden:      hidden.Nodes(),
		outputs:     outputNodes,
		connections: connections,
	}, nil
}

// convertSubstrate turns a pruned substrate into an AbstractNetwork. Nodes are
// ordered inputs, then outputs, then hidden nodes sorted by increasing y (ties
// broken by increasing x). Connections refer to nodes by index.
//
// Sorting by y makes the hidden nodes a valid evaluation order under the
// forward rule (a connection is forward when it goes to a strictly greater y):
// every forward connection between two hidden nodes goes from a lower index
// to a higher one, so visiting hidden nodes in index order computes each one
// after all of its forward sources. Inputs (y = -1) are below every hidden node
// and outputs (y = 1) above, so the full evaluation order is inputs, hidden
// nodes in index order, then outputs.
//
// Node values are not recomputed: every SpacialNode already holds its
// node-CPPN outputs from when it was created. Inputs always use Linear, so a
// sensor value passes through unchanged; outputs and hidden nodes use the
// activation function chosen by their CPPN outputs.
//
// It returns an error if a connection refers to a position with no node,
// which would mean the substrate is inconsistent.
func convertSubstrate(sub *substrate) (AbstractNetwork, error) {
	total := len(sub.inputs) + len(sub.outputs) + len(sub.hidden)
	nodes := make([]AbstractNode, 0, total)
	index := make(map[position]int, total)
	add := func(n *SpacialNode, activation ActivationFuncID) {
		index[position{n.x, n.y}] = len(nodes)
		nodes = append(nodes, AbstractNode{
			Posx:           n.x,
			Posy:           n.y,
			ActivationFunc: activation,
			Values:         n.values,
		})
	}
	for _, n := range sub.inputs {
		add(n, Linear)
	}
	for _, n := range sub.outputs {
		add(n, chooseActivation(n.values))
	}
	for _, n := range sortedByY(sub.hidden) {
		add(n, chooseActivation(n.values))
	}

	connections := make([]AbstractConnection, 0, len(sub.connections))
	for _, c := range sub.connections {
		from, ok := index[position{c.fromx, c.fromy}]
		if !ok {
			return AbstractNetwork{}, fmt.Errorf("connection source (%g, %g) has no node", c.fromx, c.fromy)
		}
		to, ok := index[position{c.tox, c.toy}]
		if !ok {
			return AbstractNetwork{}, fmt.Errorf("connection target (%g, %g) has no node", c.tox, c.toy)
		}

		forward := c.isForward()
		connections = append(connections, AbstractConnection{
			From:    from,
			To:      to,
			Weight:  c.weight,
			Values:  c.values,
			Forward: forward,
		})
	}

	return AbstractNetwork{
		Nodes:       nodes,
		Connections: connections,
	}, nil
}

// sortedByY returns a copy of nodes sorted by increasing y, then increasing x.
// Positions are unique, so the order is fully determined.
func sortedByY(nodes []*SpacialNode) []*SpacialNode {
	sorted := slices.Clone(nodes)
	slices.SortFunc(sorted, func(a, b *SpacialNode) int {
		if c := cmp.Compare(a.y, b.y); c != 0 {
			return c
		}
		return cmp.Compare(a.x, b.x)
	})
	return sorted
}
