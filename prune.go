package esneatforhebbians

// This file implements Algorithm 3, step 4: remove every hidden node and
// connection that is not on a path from an input to an output.
//
// It runs two depth-first searches over the connection graph:
//
//   - forward, from all inputs, following connections from source to target:
//     this finds every point some input can reach;
//   - backward, from all outputs, following connections from target to
//     source: this finds every point that can reach some output.
//
// A hidden node is on an input-to-output path exactly when both searches
// reach it. A connection from -> to is on such a path exactly when an input
// reaches `from` and `to` reaches an output.
//
// Cycles need no special handling: each search marks a point as visited the
// first time it sees it and never expands it again.

// reachable runs a depth-first search from every start point at once,
// following graph, and returns the set of points it visits (including the
// starts). It uses an explicit stack, so long chains cannot overflow the call
// stack, and marks points visited as they are pushed, so none is pushed twice.
func reachable(starts []AbstractNode, graph map[Point][]Point) map[Point]bool {
	visited := make(map[Point]bool)
	var stack []Point
	for _, n := range starts {
		if !visited[n.Point] {
			visited[n.Point] = true
			stack = append(stack, n.Point)
		}
	}
	for len(stack) > 0 {
		p := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for _, next := range graph[p] {
			if !visited[next] {
				visited[next] = true
				stack = append(stack, next)
			}
		}
	}
	return visited
}

// pruneSubstrate returns a copy of s that keeps only the hidden nodes and
// connections on some path from an input to an output. Inputs and outputs are
// always kept, since they are the network's interface. The order of the
// remaining hidden nodes and connections is preserved.
func pruneSubstrate(s *substrate) *substrate {
	forward := make(map[Point][]Point)
	backward := make(map[Point][]Point)
	for _, c := range s.connections {
		forward[c.From] = append(forward[c.From], c.To)
		backward[c.To] = append(backward[c.To], c.From)
	}
	fromInputs := reachable(s.inputs, forward)
	toOutputs := reachable(s.outputs, backward)

	pruned := &substrate{inputs: s.inputs, outputs: s.outputs}
	for _, n := range s.hidden {
		if fromInputs[n.Point] && toOutputs[n.Point] {
			pruned.hidden = append(pruned.hidden, n)
		}
	}
	for _, c := range s.connections {
		if fromInputs[c.From] && toOutputs[c.To] {
			pruned.connections = append(pruned.connections, c)
		}
	}
	return pruned
}
