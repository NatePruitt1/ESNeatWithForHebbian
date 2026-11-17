package esneatforhebbians

import "fmt"

// Layout rules, shared by every network.
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
//     which is what the y-sorting guarantees.
//
// A recurrent connection reads its source's value from the previous step, so
// it may join any two nodes, except that it may not end at an input. Recurrent
// connections are only allowed when allowRecurrent is true.

// validateLayout checks the layout rules above and returns the first one that
// is broken, naming the connection by its index.
func validateLayout(nodeCount, inputCount, outputCount int, conns []Connection, allowRecurrent bool) error {
	if inputCount < 0 || outputCount < 0 || inputCount+outputCount > nodeCount {
		return fmt.Errorf("%d inputs and %d outputs do not fit in %d nodes", inputCount, outputCount, nodeCount)
	}
	hiddenStart := inputCount + outputCount
	isInput := func(n int) bool { return n < inputCount }
	isOutput := func(n int) bool { return n >= inputCount && n < hiddenStart }

	for i, c := range conns {
		if c.From < 0 || c.From >= nodeCount || c.To < 0 || c.To >= nodeCount {
			return fmt.Errorf("connection %d (%d -> %d) refers to a node outside 0..%d", i, c.From, c.To, nodeCount-1)
		}
		if isInput(c.To) {
			return fmt.Errorf("connection %d (%d -> %d) ends at input node %d; inputs cannot receive connections", i, c.From, c.To, c.To)
		}
		if !c.Forward {
			if !allowRecurrent {
				return fmt.Errorf("connection %d (%d -> %d) is recurrent, but this network does not allow recurrent connections", i, c.From, c.To)
			}
			continue
		}
		if isOutput(c.From) {
			return fmt.Errorf("connection %d (%d -> %d) is forward from output node %d; outputs are evaluated last, so it must be recurrent", i, c.From, c.To, c.From)
		}
		if c.From >= hiddenStart && c.To >= hiddenStart && c.From >= c.To {
			return fmt.Errorf("connection %d (%d -> %d) is forward between hidden nodes but does not go to a higher index; hidden nodes must be sorted so forward connections go to higher indices", i, c.From, c.To)
		}
	}
	return nil
}
