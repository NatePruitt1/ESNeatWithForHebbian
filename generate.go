package esneatforhebbians

import (
	"fmt"
	"slices"
)

// GenerateAbstractFromCPPNs builds a network from a connection CPPN and a node
// CPPN using ES-HyperNEAT's Algorithm 3.
//
// bias and networkType are passed to every query of both CPPNs. bias is a
// constant input (commonly 1); networkType selects which network to build, so
// calling this with networkType 0, 1, 2, ... builds several networks from the
// same pair of CPPNs.
//
// allowRecurrent controls whether the network may contain recurrent
// connections (lateral, backward and self-connections). When it is false, the
// search keeps only forward connections, so the result is purely feed-forward.
//
// The result always satisfies the layout rules in layout.go.
func GenerateAbstractFromCPPNs(cppn CPPN, ncppn NodeCPPN, bias float64, networkType int, allowRecurrent bool, inputCount, outputCount int, params SearchParams) (*AbstractNetwork, error) {
	if n := cppn.GetOutputSize(); n <= WeightIndex {
		return nil, fmt.Errorf("connection CPPN has %d outputs; it needs at least %d to provide a weight", n, WeightIndex+1)
	}
	if n, need := ncppn.GetOutputSize(), ActivationChannel+ActivationFuncCount; n < need {
		return nil, fmt.Errorf("node CPPN has %d outputs; it needs at least %d to choose an activation function", n, need)
	}

	// Algorithm 3, steps 1-3: find the nodes and connections.
	s, err := discoverSubstrate(BindCPPN(cppn, bias, networkType), BindNodeCPPN(ncppn, bias, networkType), allowRecurrent, inputCount, outputCount, params)
	if err != nil {
		return nil, err
	}

	// Algorithm 3, step 4: keep only what lies on an input-to-output path.
	s = pruneSubstrate(s)

	an, err := toAbstractNetwork(s)
	if err != nil {
		return nil, err
	}

	// Final check against the same rules the networks enforce. Failing here
	// means a bug in the construction above, not bad CPPN output.
	if err := validateLayout(len(an.Nodes), an.InputCount, an.OutputCount, an.connections(), allowRecurrent); err != nil {
		return nil, fmt.Errorf("generated network breaks the layout rules: %w", err)
	}
	return an, nil
}

// toAbstractNetwork turns a pruned substrate into an AbstractNetwork. Nodes are
// ordered inputs, outputs, then hidden nodes sorted by y (then x), and
// connections refer to them by index.
//
// Sorting by y gives the evaluation order the layout rules need: a forward
// connection goes to a greater y, so between hidden nodes it always goes to a
// higher index.
func toAbstractNetwork(s *substrate) (*AbstractNetwork, error) {
	hidden := slices.Clone(s.hidden)
	slices.SortFunc(hidden, func(a, b AbstractNode) int { return comparePoints(a.Point, b.Point) })

	an := &AbstractNetwork{InputCount: len(s.inputs), OutputCount: len(s.outputs)}
	an.Nodes = slices.Concat(s.inputs, s.outputs, hidden)

	index := make(map[Point]int, len(an.Nodes))
	for i, n := range an.Nodes {
		index[n.Point] = i
	}
	for _, c := range s.connections {
		from, ok := index[c.From]
		if !ok {
			return nil, fmt.Errorf("connection source %v has no node", c.From)
		}
		to, ok := index[c.To]
		if !ok {
			return nil, fmt.Errorf("connection target %v has no node", c.To)
		}
		an.Connections = append(an.Connections, AbstractConnection{
			Connection: Connection{From: from, To: to, Weight: c.weight, Forward: c.isForward()},
			Values:     c.values,
		})
	}
	return an, nil
}
