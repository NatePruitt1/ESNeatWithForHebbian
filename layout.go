package esneatforhebbians

import "fmt"

// Network layout rules, shared by Network and HebbianNetwork.
//
// Nodes are laid out as [inputs..., outputs..., hidden...]: the first
// inputCount nodes are inputs, the next outputCount are outputs, and the rest
// are hidden nodes, which ES-HyperNEAT sorts by increasing y. Each step
// evaluates inputs, then hidden nodes in index order, then outputs.
//
// A forward connection reads its source's value from the current step, so its
// source must be evaluated before its target:
//
//   - it may not end at an input (inputs are set directly, never computed);
//   - it may not start at an output (outputs are evaluated last);
//   - between two hidden nodes it must go from a lower index to a higher one,
//     which is what the y-sorting guarantees under the forward rule.
//
// A recurrent connection reads its source's value from the previous step, so
// it may join any two nodes, except that it may not end at an input. Recurrent
// connections are only allowed when allowRecurrent is true.

// link is the part of a connection the layout rules look at.
type link struct {
	from, to int
	forward  bool
}

// validateLayout checks the layout rules above and returns the first one that
// is broken, naming the connection by its index.
func validateLayout(nodeCount, inputCount, outputCount int, links []link, allowRecurrent bool) error {
	if inputCount < 0 || outputCount < 0 || inputCount+outputCount > nodeCount {
		return fmt.Errorf("%d inputs and %d outputs do not fit in %d nodes", inputCount, outputCount, nodeCount)
	}
	hiddenStart := inputCount + outputCount
	isInput := func(n int) bool { return n < inputCount }
	isOutput := func(n int) bool { return n >= inputCount && n < hiddenStart }

	for i, l := range links {
		if l.from < 0 || l.from >= nodeCount || l.to < 0 || l.to >= nodeCount {
			return fmt.Errorf("connection %d (%d -> %d) refers to a node outside 0..%d", i, l.from, l.to, nodeCount-1)
		}
		if isInput(l.to) {
			return fmt.Errorf("connection %d (%d -> %d) ends at input node %d; inputs cannot receive connections", i, l.from, l.to, l.to)
		}
		if !l.forward {
			if !allowRecurrent {
				return fmt.Errorf("connection %d (%d -> %d) is recurrent, but this network does not allow recurrent connections", i, l.from, l.to)
			}
			continue
		}
		if isOutput(l.from) {
			return fmt.Errorf("connection %d (%d -> %d) is forward from output node %d; outputs are evaluated last, so it must be recurrent", i, l.from, l.to, l.from)
		}
		if l.from >= hiddenStart && l.to >= hiddenStart && l.from >= l.to {
			return fmt.Errorf("connection %d (%d -> %d) is forward between hidden nodes but does not go to a higher index; hidden nodes must be sorted so forward connections go to higher indices", i, l.from, l.to)
		}
	}
	return nil
}
