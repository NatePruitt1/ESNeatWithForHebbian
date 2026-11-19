// AI-generated: written by Claude (Claude Code, claude-opus-5-5) as a
// regression suite for the rework package. Review before relying on it.

package rework

import (
	"slices"
	"testing"
)

// ---------- small pieces ----------

func TestGetWeight(t *testing.T) {
	c := substrateConnection{values: []float64{0.7, 2}}
	if c.GetWeight() != 0.7 {
		t.Fatalf("got %v", c.GetWeight())
	}
	mustPanic(t, func() { substrateConnection{}.GetWeight() })
}

func TestBuildRow(t *testing.T) {
	g := mustBindNode(t, constNodeCPPN(Sine))
	cases := map[int][]float64{0: {}, 1: {0}, 3: {-0.5, 0, 0.5}}
	for n, xs := range cases {
		row := buildRow(n, 1, *g)
		if len(row) != n {
			t.Fatalf("n=%d: got %d nodes", n, len(row))
		}
		for i, node := range row {
			if !approxEq(node.X, xs[i]) || node.Y != 1 {
				t.Errorf("n=%d node %d at %v, want (%v, 1)", n, i, node.Point, xs[i])
			}
			if len(node.values) != ActivationFunctionCount || Argmax(node.values[:ActivationFunctionCount]) != int(Sine) {
				t.Errorf("n=%d node %d has values %v", n, i, node.values)
			}
		}
	}
}

func TestDiscoverConnections(t *testing.T) {
	g := mustBindNode(t, constNodeCPPN(Sigmoid))
	in, a, b, known := Point{0, -1}, Point{0.5, 0}, Point{-0.5, 0}, Point{0.25, 0.25}
	seen := map[Point]bool{known: true}
	conns := connMap([2]Point{in, a}, [2]Point{in, b}, [2]Point{in, known}, [2]Point{a, b})

	got := discoverConnections(nil, seen, conns, *g, true)
	pts := map[Point]bool{}
	for _, n := range got {
		if pts[n.Point] {
			t.Errorf("node %v discovered twice", n.Point)
		}
		pts[n.Point] = true
		if len(n.values) != ActivationFunctionCount {
			t.Errorf("node %v has %d values", n.Point, len(n.values))
		}
	}
	if !pointSetEqual(pts, map[Point]bool{a: true, b: true}) {
		t.Errorf("discovered %v, want {a, b}", pts)
	}
	if !seen[a] || !seen[b] || !seen[known] {
		t.Errorf("seen map not updated: %v", seen)
	}

	// Running again finds nothing new.
	if again := discoverConnections(nil, seen, conns, *g, true); len(again) != 0 {
		t.Errorf("rediscovered %d nodes", len(again))
	}
}

func TestDiscoverConnectionsIncomingUsesFrom(t *testing.T) {
	g := mustBindNode(t, constNodeCPPN(Sigmoid))
	src, out := Point{0.5, 0.5}, Point{0, 1}
	got := discoverConnections(nil, map[Point]bool{}, connMap([2]Point{src, out}), *g, false)
	if len(got) != 1 || got[0].Point != src {
		t.Fatalf("got %v, want the source %v", got, src)
	}
}

func TestUnionConnections(t *testing.T) {
	p, q, r := Point{0, 0}, Point{0, 0.5}, Point{0.5, 0.5}
	a := connMap([2]Point{p, q})
	unionConnections(a, connMap([2]Point{q, r}, [2]Point{p, q}))
	if len(a) != 2 {
		t.Fatalf("got %d connections, want 2", len(a))
	}
	if _, ok := a[[2]Point{q, r}]; !ok {
		t.Errorf("q->r missing")
	}
}

// ---------- reachable ----------

func TestReachableForwardAndBackward(t *testing.T) {
	in, a, b, c, out := Point{0, -1}, Point{0, 0}, Point{0.5, 0}, Point{-0.5, 0}, Point{0, 1}
	conns := connMap([2]Point{in, a}, [2]Point{a, out}, [2]Point{in, b}, [2]Point{c, out})
	fwd := reachable(nodesAt(in), conns, false)
	bwd := reachable(nodesAt(out), conns, true)
	if want := map[Point]bool{in: true, a: true, b: true, out: true}; !pointSetEqual(fwd, want) {
		t.Errorf("forward %v want %v", fwd, want)
	}
	if want := map[Point]bool{out: true, a: true, c: true, in: true}; !pointSetEqual(bwd, want) {
		t.Errorf("backward %v want %v", bwd, want)
	}
}

func TestReachableCyclesAndSelfLoops(t *testing.T) {
	in, a, b := Point{0, -1}, Point{0, 0}, Point{0, 0.5}
	conns := connMap([2]Point{in, a}, [2]Point{a, b}, [2]Point{b, a}, [2]Point{b, b})
	if got := reachable(nodesAt(in), conns, false); len(got) != 3 {
		t.Fatalf("got %v", got)
	}
}

func TestReachableIncludesStartsAndMultipleSources(t *testing.T) {
	i1, i2, a := Point{-0.5, -1}, Point{0.5, -1}, Point{0, 0}
	got := reachable(nodesAt(i1, i2), connMap([2]Point{i1, a}), false)
	if want := map[Point]bool{i1: true, i2: true, a: true}; !pointSetEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
	if got := reachable(nodesAt(i1), nil, false); !pointSetEqual(got, map[Point]bool{i1: true}) {
		t.Fatalf("empty graph: got %v", got)
	}
}

// ---------- generateSubstrate ----------

func TestGenerateConstantIsEmpty(t *testing.T) {
	cfg := baseCfg()
	s := gen(t, linearConnCPPN(0, 0, 0, 0, 0.5, Linear), cfg)
	checkSubstrate(t, s, cfg)
	if len(s.hiddenNodes) != 0 || len(s.conns) != 0 {
		t.Errorf("constant CPPN gave %d hidden, %d conns", len(s.hiddenNodes), len(s.conns))
	}
}

func TestGenerateLayered(t *testing.T) {
	cfg := baseCfg()
	s := gen(t, layeredCPPN(), cfg)
	checkSubstrate(t, s, cfg)
	if len(s.hiddenNodes) == 0 {
		t.Fatal("layered CPPN produced no hidden nodes")
	}
	toOutput, fromInput := 0, 0
	for _, c := range s.conns {
		if c.To.Y == 1 {
			toOutput++
		}
		if c.From.Y == -1 {
			fromInput++
		}
	}
	if toOutput == 0 || fromInput == 0 {
		t.Errorf("from inputs: %d, to outputs: %d; want both > 0", fromInput, toOutput)
	}
}

// Regression for the case where forward discovery and the output search land
// on different quadtree rows: nothing connects, so pruning removes everything.
func TestGenerateMisalignedRowsArePrunedAway(t *testing.T) {
	cfg := baseCfg()
	const k = 8
	s := gen(t, linearConnCPPN(0, -k, 0, k, -k*0.5, Gaussian), cfg)
	checkSubstrate(t, s, cfg)
	if len(s.hiddenNodes) != 0 || len(s.conns) != 0 {
		t.Logf("misaligned layered CPPN now connects (%d hidden): update this test if that was intended", len(s.hiddenNodes))
	}
}

func TestGenerateDeterministic(t *testing.T) {
	cfg := baseCfg()
	a, b := gen(t, layeredCPPN(), cfg), gen(t, layeredCPPN(), cfg)
	if !slices.EqualFunc(a.hiddenNodes, b.hiddenNodes, func(x, y substrateNode) bool { return x.Point == y.Point }) {
		t.Errorf("hidden nodes differ between runs")
	}
	if !slices.EqualFunc(sortedConns(a.conns), sortedConns(b.conns), func(x, y substrateConnection) bool {
		return x.From == y.From && x.To == y.To && x.recurrent == y.recurrent
	}) {
		t.Errorf("connections differ between runs (after sorting)")
	}
}

func TestGenerateRandomInvariants(t *testing.T) {
	for _, allowed := range []bool{true, false} {
		cfg := baseCfg()
		cfg.maxDepth = 4
		cfg.recursionAllowed = allowed
		total, recurrent := 0, 0
		for seed := int64(0); seed < 40; seed++ {
			var s *substrate
			noPanic(t, func() {
				var err error
				if s, err = generateSubstrate(randomConnCPPN(seed), constNodeCPPN(Sigmoid), cfg); err != nil {
					t.Errorf("recursionAllowed=%v seed %d: %v", allowed, seed, err)
				}
			})
			if s == nil {
				continue
			}
			checkSubstrate(t, s, cfg)
			total += len(s.hiddenNodes)
			for _, c := range s.conns {
				if c.recurrent {
					recurrent++
				}
			}
		}
		if total == 0 {
			t.Errorf("recursionAllowed=%v: random CPPNs gave no hidden nodes; the invariants were not exercised", allowed)
		}
		if allowed && recurrent == 0 {
			t.Errorf("recursionAllowed=true: no recurrent connections; the recurrent invariants were not exercised")
		}
	}
}

// Removing recurrent connections and pruning again can only remove nodes and
// connections, never add them.
func TestGenerateFeedForwardIsSubsetOfRecurrent(t *testing.T) {
	cfg := baseCfg()
	cfg.maxDepth = 4
	smaller := 0
	for seed := int64(0); seed < 20; seed++ {
		cfg.recursionAllowed = true
		rec := gen(t, randomConnCPPN(seed), cfg)
		cfg.recursionAllowed = false
		ff := gen(t, randomConnCPPN(seed), cfg)
		checkSubstrate(t, ff, cfg)

		recNodes := map[Point]bool{}
		for _, n := range rec.hiddenNodes {
			recNodes[n.Point] = true
		}
		for _, n := range ff.hiddenNodes {
			if !recNodes[n.Point] {
				t.Errorf("seed %d: feed-forward hidden node %v is not in the recurrent substrate", seed, n.Point)
			}
		}
		recConns := map[[2]Point]bool{}
		for _, c := range rec.conns {
			recConns[[2]Point{c.From, c.To}] = true
		}
		for _, c := range ff.conns {
			if !recConns[[2]Point{c.From, c.To}] {
				t.Errorf("seed %d: feed-forward connection %v->%v is not in the recurrent substrate", seed, c.From, c.To)
			}
		}
		if len(ff.hiddenNodes) < len(rec.hiddenNodes) {
			smaller++
		}
	}
	if smaller == 0 {
		t.Errorf("removing recurrence never pruned a hidden node; the second prune was not exercised")
	}
}

// in -> a -> out, a -> c (forward), c -> a (recurrent). With recurrence, c is on
// a path (in -> a -> c -> a -> out) and is kept. Without it, c leads nowhere, so
// the second prune must drop c and a -> c.
func TestPruneGenerationRecurrence(t *testing.T) {
	in, a, c, out := Point{0, -1}, Point{0, 0}, Point{0, 0.5}, Point{0, 1}
	build := func() map[[2]Point]substrateConnection {
		return connMap([2]Point{in, a}, [2]Point{a, out}, [2]Point{a, c}, [2]Point{c, a})
	}

	for _, allowed := range []bool{true, false} {
		cfg := baseCfg()
		cfg.inputCount, cfg.outputCount = 1, 1
		cfg.recursionAllowed = allowed
		s := pruneGeneration(nodesWithValues(in), nodesWithValues(out), nodesWithValues(a, c), build(), cfg)
		checkSubstrate(t, s, cfg)

		gotNodes := map[Point]bool{}
		for _, n := range s.hiddenNodes {
			gotNodes[n.Point] = true
		}
		gotConns := map[[2]Point]bool{}
		for _, cn := range s.conns {
			gotConns[[2]Point{cn.From, cn.To}] = true
		}

		wantNodes := map[Point]bool{a: true}
		wantConns := map[[2]Point]bool{{in, a}: true, {a, out}: true}
		if allowed {
			wantNodes[c] = true
			wantConns[[2]Point{a, c}] = true
			wantConns[[2]Point{c, a}] = true
		}
		if !pointSetEqual(gotNodes, wantNodes) {
			t.Errorf("recursionAllowed=%v: hidden %v, want %v", allowed, gotNodes, wantNodes)
		}
		if len(gotConns) != len(wantConns) {
			t.Errorf("recursionAllowed=%v: connections %v, want %v", allowed, gotConns, wantConns)
		}
		for k := range wantConns {
			if !gotConns[k] {
				t.Errorf("recursionAllowed=%v: missing connection %v->%v", allowed, k[0], k[1])
			}
		}
	}
}

// A connection between two nodes at the same y, and a self-connection, are
// recurrent, so they must not survive when recursion is not allowed.
func TestPruneGenerationLateralAndSelfAreRecurrent(t *testing.T) {
	in, a, b, out := Point{0, -1}, Point{-0.5, 0}, Point{0.5, 0}, Point{0, 1}
	conns := connMap([2]Point{in, a}, [2]Point{a, b}, [2]Point{b, b}, [2]Point{b, out})

	cfg := baseCfg()
	cfg.inputCount, cfg.outputCount = 1, 1
	cfg.recursionAllowed = true
	s := pruneGeneration(nodesWithValues(in), nodesWithValues(out), nodesWithValues(a, b), conns, cfg)
	checkSubstrate(t, s, cfg)
	if len(s.hiddenNodes) != 2 {
		t.Errorf("recursionAllowed=true: kept %d hidden nodes, want 2", len(s.hiddenNodes))
	}

	cfg.recursionAllowed = false
	s = pruneGeneration(nodesWithValues(in), nodesWithValues(out), nodesWithValues(a, b),
		connMap([2]Point{in, a}, [2]Point{a, b}, [2]Point{b, b}, [2]Point{b, out}), cfg)
	checkSubstrate(t, s, cfg)
	if len(s.hiddenNodes) != 0 || len(s.conns) != 0 {
		t.Errorf("recursionAllowed=false: kept %d hidden, %d conns; the lateral a->b was the only link", len(s.hiddenNodes), len(s.conns))
	}
}

func TestGenerateInputAndOutputRows(t *testing.T) {
	cfg := baseCfg()
	cfg.inputCount, cfg.outputCount = 3, 1
	s := gen(t, layeredCPPN(), cfg)
	checkSubstrate(t, s, cfg)
	if s.outputNodes[0].X != 0 {
		t.Errorf("single output at %v, want x = 0", s.outputNodes[0].Point)
	}
}

func TestGenerateNoInputsMeansNoHidden(t *testing.T) {
	cfg := baseCfg()
	cfg.inputCount = 0
	s := gen(t, layeredCPPN(), cfg)
	if len(s.hiddenNodes) != 0 || len(s.conns) != 0 {
		t.Errorf("no inputs gave %d hidden, %d conns", len(s.hiddenNodes), len(s.conns))
	}
}

func TestGenerateRejectsBadCPPNs(t *testing.T) {
	cfg := baseCfg()
	badConn := linearConnCPPN(0, 0, 0, 0, 0, Linear)
	badConn.InputCount = 3
	if _, err := generateSubstrate(badConn, constNodeCPPN(Sigmoid), cfg); err == nil {
		t.Errorf("3-input connection CPPN accepted")
	}
	badNode := constNodeCPPN(Sigmoid)
	badNode.OutputCount = 3
	if _, err := generateSubstrate(layeredCPPN(), badNode, cfg); err == nil {
		t.Errorf("node CPPN with too few outputs accepted")
	}
	recurrentConn := layeredCPPN()
	recurrentConn.Connections = append(recurrentConn.Connections, Connection{From: 4, To: 4, Weight: 1, Forward: false})
	if _, err := generateSubstrate(recurrentConn, constNodeCPPN(Sigmoid), cfg); err == nil {
		t.Errorf("recurrent connection CPPN accepted")
	}
}
