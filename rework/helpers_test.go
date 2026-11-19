// AI-generated: written by Claude (Claude Code, claude-opus-5-5) as a
// regression suite for the rework package. Review before relying on it.

package rework

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

// Specs used throughout the tests.
var connSpec = NetworkSpec{Inputs: 4, Outputs: 1}
var nodeSpec = NetworkSpec{Inputs: 2, Outputs: ActivationFunctionCount}

// linearConnCPPN builds a 4-input, 1-output connection CPPN:
//
//	w = act(a*x1 + b*y1 + c*x2 + d*y2 + bias)
//
// Layout: [x1, y1, x2, y2, out].
func linearConnCPPN(a, b, c, d, bias float64, act ActivationFunc) ArtificialNeuralNet {
	nodes := make([]Node, 5)
	nodes[4] = Node{Bias: bias, Activation: act}
	return ArtificialNeuralNet{
		InputCount: 4, OutputCount: 1, Nodes: nodes,
		Connections: []Connection{
			{From: 0, To: 4, Weight: a, Forward: true},
			{From: 1, To: 4, Weight: b, Forward: true},
			{From: 2, To: 4, Weight: c, Forward: true},
			{From: 3, To: 4, Weight: d, Forward: true},
		},
	}
}

// randomConnCPPN builds a feed-forward connection CPPN with one Gaussian
// hidden node and random weights from seed. Layout: [4 inputs, out, hidden].
func randomConnCPPN(seed int64) ArtificialNeuralNet {
	r := rand.New(rand.NewSource(seed))
	w := func() float64 { return r.Float64()*6 - 3 }
	nodes := make([]Node, 6)
	nodes[4] = Node{Bias: w(), Activation: Tanh}
	nodes[5] = Node{Bias: w(), Activation: Gaussian}
	var conns []Connection
	for i := 0; i < 4; i++ {
		conns = append(conns, Connection{From: i, To: 5, Weight: w(), Forward: true})
		conns = append(conns, Connection{From: i, To: 4, Weight: w(), Forward: true})
	}
	conns = append(conns, Connection{From: 5, To: 4, Weight: w(), Forward: true})
	return ArtificialNeuralNet{InputCount: 4, OutputCount: 1, Nodes: nodes, Connections: conns}
}

// layeredCPPN connects each point to the row 0.5 above it:
//
//	w = gaussian(k*(y2 - y1 - 0.5))
//
// With k = 4 and baseCfg it yields a connected multi-row substrate.
func layeredCPPN() ArtificialNeuralNet {
	const k = 4
	return linearConnCPPN(0, -k, 0, k, -k*0.5, Gaussian)
}

// constNodeCPPN builds a node CPPN whose argmax always picks act.
func constNodeCPPN(act ActivationFunc) ArtificialNeuralNet {
	nodes := make([]Node, 2+ActivationFunctionCount)
	nodes[2+int(act)].Bias = 1
	return ArtificialNeuralNet{InputCount: 2, OutputCount: ActivationFunctionCount, Nodes: nodes}
}

func mustBindConn(t *testing.T, ann ArtificialNeuralNet) *BoundConnectionGeneratingCPPN {
	t.Helper()
	b, err := BindANNForConnectionGeneration(ann, connSpec)
	if err != nil {
		t.Fatalf("bind connection CPPN: %v", err)
	}
	return b
}

func mustBindNode(t *testing.T, ann ArtificialNeuralNet) *BoundNodeGeneratingCPPN {
	t.Helper()
	b, err := BindANNForNodeGeneration(ann, nodeSpec)
	if err != nil {
		t.Fatalf("bind node CPPN: %v", err)
	}
	return b
}

func baseCfg() substrateGenConfig {
	return substrateGenConfig{
		connectionGeneratorSpec: connSpec,
		nodeGeneratorSpec:       nodeSpec,
		initialDepth:            3,
		maxDepth:                5,
		inputCount:              2,
		outputCount:             2,
		divThreshold:            0.03,
		varThreshold:            0.03,
		bandThreshold:           0.3,
		recursionAllowed:        true,
	}
}

// ---------- quadtree helpers ----------

// walk calls fn on every quadtree node with its parent (nil for the root).
func walk(p, parent *quadPoint, fn func(p, parent *quadPoint)) {
	fn(p, parent)
	if !p.isLeaf() {
		for _, c := range p.children {
			walk(c, p, fn)
		}
	}
}

func leaves(root *quadPoint) []*quadPoint {
	var out []*quadPoint
	walk(root, nil, func(p, _ *quadPoint) {
		if p.isLeaf() {
			out = append(out, p)
		}
	})
	return out
}

// checkTreeInvariants checks the structural rules of a quadtree built by
// divisionAndInitialization.
func checkTreeInvariants(t *testing.T, root *quadPoint) {
	t.Helper()
	walk(root, nil, func(p, parent *quadPoint) {
		if parent == nil {
			return
		}
		if p.width != parent.width/2 {
			t.Errorf("%v: width %v, parent %v", p.Point, p.width, parent.width)
		}
		if p.level != parent.level+1 {
			t.Errorf("%v: level %d, parent %d", p.Point, p.level, parent.level)
		}
		if math.Abs(p.X-parent.X) != p.width || math.Abs(p.Y-parent.Y) != p.width {
			t.Errorf("%v is not a quadrant centre of %v", p.Point, parent.Point)
		}
		if math.Abs(p.X) >= 1 || math.Abs(p.Y) >= 1 {
			t.Errorf("%v is not strictly inside (-1,1)^2", p.Point)
		}
		if len(p.values) == 0 || p.w != p.values[WEIGHT_INDEX] {
			t.Errorf("%v: w %v does not match values %v", p.Point, p.w, p.values)
		}
	})
}

// extract runs Algorithms 1 and 2 around fixed.
func extract(t *testing.T, ann ArtificialNeuralNet, fixed Point, outgoing bool, cfg substrateGenConfig) map[[2]Point]substrateConnection {
	t.Helper()
	g := mustBindConn(t, ann)
	root := divisionAndInitialization(fixed, outgoing, *g, cfg)
	return pruningAndExtraction(fixed, make(map[[2]Point]substrateConnection), g, root, outgoing, cfg)
}

// ---------- substrate helpers ----------

// comparePts orders points by Y, then X.
func comparePts(a, b Point) int {
	switch {
	case a.Y < b.Y:
		return -1
	case a.Y > b.Y:
		return 1
	case a.X < b.X:
		return -1
	case a.X > b.X:
		return 1
	}
	return 0
}

func sortedConns(cs []substrateConnection) []substrateConnection {
	out := slices.Clone(cs)
	slices.SortFunc(out, func(a, b substrateConnection) int {
		if c := comparePts(a.From, b.From); c != 0 {
			return c
		}
		return comparePts(a.To, b.To)
	})
	return out
}

// connMap builds a connection map from (from, to) pairs, each with weight 1.
func connMap(pairs ...[2]Point) map[[2]Point]substrateConnection {
	m := make(map[[2]Point]substrateConnection)
	for _, p := range pairs {
		m[p] = substrateConnection{From: p[0], To: p[1], values: []float64{1}}
	}
	return m
}

func nodesAt(ps ...Point) []substrateNode {
	out := make([]substrateNode, len(ps))
	for i, p := range ps {
		out[i] = substrateNode{Point: p}
	}
	return out
}

// nodesWithValues is like nodesAt but gives each node a full set of node-CPPN
// values, as generated nodes have.
func nodesWithValues(ps ...Point) []substrateNode {
	out := nodesAt(ps...)
	for i := range out {
		out[i].values = make([]float64, ActivationFunctionCount)
	}
	return out
}

func pointSetEqual(a, b map[Point]bool) bool {
	if len(a) != len(b) {
		return false
	}
	for k := range a {
		if !b[k] {
			return false
		}
	}
	return true
}

func gen(t *testing.T, conn ArtificialNeuralNet, cfg substrateGenConfig) *substrate {
	t.Helper()
	s, err := generateSubstrate(conn, constNodeCPPN(Sigmoid), cfg)
	if err != nil {
		t.Fatalf("generateSubstrate: %v", err)
	}
	return s
}

// checkSubstrate checks every rule a generated substrate must satisfy:
// inputs/outputs on the edges, hidden nodes unique and strictly inside, hidden
// nodes sorted by (Y, X), connection endpoints are kept nodes, recurrent flags
// match geometry, no recurrent connections when cfg.recursionAllowed is false,
// forward hidden connections go to a higher index, and every hidden node lies
// on an input -> output path.
func checkSubstrate(t *testing.T, s *substrate, cfg substrateGenConfig) {
	t.Helper()
	if len(s.inputNodes) != cfg.inputCount || len(s.outputNodes) != cfg.outputCount {
		t.Errorf("io counts %d/%d, want %d/%d", len(s.inputNodes), len(s.outputNodes), cfg.inputCount, cfg.outputCount)
	}
	isIn, isOut, idx := map[Point]bool{}, map[Point]bool{}, map[Point]int{}
	for _, n := range s.inputNodes {
		isIn[n.Point] = true
		if n.Y != -1 {
			t.Errorf("input at %v, want y = -1", n.Point)
		}
	}
	for _, n := range s.outputNodes {
		isOut[n.Point] = true
		if n.Y != 1 {
			t.Errorf("output at %v, want y = 1", n.Point)
		}
	}
	for i, n := range s.hiddenNodes {
		if _, dup := idx[n.Point]; dup {
			t.Errorf("duplicate hidden node %v", n.Point)
		}
		idx[n.Point] = i
		if math.Abs(n.X) >= 1 || math.Abs(n.Y) >= 1 {
			t.Errorf("hidden %v is not strictly inside", n.Point)
		}
		if isIn[n.Point] || isOut[n.Point] {
			t.Errorf("hidden %v collides with an input or output", n.Point)
		}
		if len(n.values) != ActivationFunctionCount {
			t.Errorf("hidden %v has %d values, want %d", n.Point, len(n.values), ActivationFunctionCount)
		}
	}
	if !slices.IsSortedFunc(s.hiddenNodes, func(a, b substrateNode) int { return comparePts(a.Point, b.Point) }) {
		t.Errorf("hidden nodes are not sorted by (Y, X)")
	}

	seen := map[[2]Point]bool{}
	fwdAdj, bwdAdj := map[Point][]Point{}, map[Point][]Point{}
	for _, c := range s.conns {
		k := [2]Point{c.From, c.To}
		if seen[k] {
			t.Errorf("duplicate connection %v", k)
		}
		seen[k] = true
		_, fromHidden := idx[c.From]
		_, toHidden := idx[c.To]
		if !(isIn[c.From] || fromHidden) {
			t.Errorf("connection %v->%v: source is not an input or kept hidden node", c.From, c.To)
		}
		if !(isOut[c.To] || toHidden) {
			t.Errorf("connection %v->%v: target is not an output or kept hidden node", c.From, c.To)
		}
		if wantRec := c.From.Y >= c.To.Y; c.recurrent != wantRec {
			t.Errorf("connection %v->%v: recurrent=%v, want %v (From.Y >= To.Y)", c.From, c.To, c.recurrent, wantRec)
		}
		if !cfg.recursionAllowed && (c.recurrent || c.From.Y >= c.To.Y) {
			t.Errorf("connection %v->%v is recurrent, but recursionAllowed is false", c.From, c.To)
		}
		if !c.recurrent && fromHidden && toHidden && idx[c.From] >= idx[c.To] {
			t.Errorf("forward connection %v->%v goes to a lower or equal hidden index", c.From, c.To)
		}
		if len(c.values) == 0 {
			t.Errorf("connection %v->%v has no values", c.From, c.To)
		}
		fwdAdj[c.From] = append(fwdAdj[c.From], c.To)
		bwdAdj[c.To] = append(bwdAdj[c.To], c.From)
	}

	bfs := func(starts []substrateNode, adj map[Point][]Point) map[Point]bool {
		seen := map[Point]bool{}
		var q []Point
		for _, s := range starts {
			seen[s.Point] = true
			q = append(q, s.Point)
		}
		for len(q) > 0 {
			p := q[0]
			q = q[1:]
			for _, n := range adj[p] {
				if !seen[n] {
					seen[n] = true
					q = append(q, n)
				}
			}
		}
		return seen
	}
	fromIn, toOut := bfs(s.inputNodes, fwdAdj), bfs(s.outputNodes, bwdAdj)
	for _, n := range s.hiddenNodes {
		if !fromIn[n.Point] || !toOut[n.Point] {
			t.Errorf("hidden %v is not on an input->output path (fromInputs=%v toOutputs=%v)", n.Point, fromIn[n.Point], toOut[n.Point])
		}
	}
}

// ---------- misc ----------

// noPanic runs fn and reports a panic as a test error instead of crashing.
func noPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("unexpected panic: %v", r)
		}
	}()
	fn()
}

// mustPanic fails the test if fn does not panic.
func mustPanic(t *testing.T, fn func()) {
	t.Helper()
	defer func() {
		if recover() == nil {
			t.Errorf("expected a panic")
		}
	}()
	fn()
}

func approxEq(a, b float64) bool { return math.Abs(a-b) < 1e-12 }
