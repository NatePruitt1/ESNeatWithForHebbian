package esneatforhebbians

// This file implements Algorithm 3, step 4: remove every hidden node and
// connection that is not on a path from an input to an output.
//
// The idea is two depth-first searches over the connection graph:
//
//   - forward, from all inputs, following connections from source to target:
//     this finds every node that some input can reach;
//   - backward, from all outputs, following connections from target to
//     source: this finds every node that can reach some output.
//
// A hidden node lies on an input-to-output path exactly when both searches
// reach it: an input reaches it, and it reaches an output. The same test,
// applied to the two ends of a connection, decides which connections stay:
// a connection from -> to is on such a path exactly when an input reaches
// `from` and `to` reaches an output (input ... -> from -> to -> ... output).
//
// Cycles and self-connections need no special handling. Each search marks a
// node as visited the first time it sees it and never expands it again, so a
// loop is walked at most once. A cycle survives pruning if, and only if, it is
// attached to an input-to-output path.

// adjacency maps each node's position to the positions it is linked to.
type adjacency map[position][]position

// buildAdjacency returns the graph twice: forward (source -> targets) and
// backward (target -> sources).
func buildAdjacency(connections []*SpacialConnection) (forward, backward adjacency) {
	forward, backward = adjacency{}, adjacency{}
	for _, c := range connections {
		from := position{c.fromx, c.fromy}
		to := position{c.tox, c.toy}
		forward[from] = append(forward[from], to)
		backward[to] = append(backward[to], from)
	}
	return forward, backward
}

// reachable runs a depth-first search from every start node at once and
// returns the set of positions it visits, including the starts themselves.
//
// It uses an explicit stack instead of recursion, so a long chain of hidden
// nodes cannot overflow the call stack:
//
//  1. push every start node and mark it visited;
//  2. pop a node, and push each neighbor that has not been visited yet,
//     marking it visited as it is pushed;
//  3. repeat until the stack is empty.
//
// Marking on push (rather than on pop) means a node is never on the stack
// twice, so the search does at most one push per node and one look per link.
func reachable(starts []*SpacialNode, graph adjacency) map[position]bool {
	visited := make(map[position]bool)
	stack := make([]position, 0, len(starts))
	for _, n := range starts {
		p := position{n.x, n.y}
		if !visited[p] {
			visited[p] = true
			stack = append(stack, p)
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
// always kept, even if nothing connects to them, because they are the
// network's interface. The order of the remaining hidden nodes and
// connections is preserved.
func pruneSubstrate(s *substrate) *substrate {
	forward, backward := buildAdjacency(s.connections)
	fromInputs := reachable(s.inputs, forward)
	toOutputs := reachable(s.outputs, backward)

	var hidden []*SpacialNode
	for _, n := range s.hidden {
		p := position{n.x, n.y}
		if fromInputs[p] && toOutputs[p] {
			hidden = append(hidden, n)
		}
	}

	var connections []*SpacialConnection
	for _, c := range s.connections {
		if fromInputs[position{c.fromx, c.fromy}] && toOutputs[position{c.tox, c.toy}] {
			connections = append(connections, c)
		}
	}

	return &substrate{
		inputs:      s.inputs,
		hidden:      hidden,
		outputs:     s.outputs,
		connections: connections,
	}
}
