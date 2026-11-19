// AI-generated: written by Claude (Claude Code, claude-opus-5-5) as a
// regression suite for the rework package. Review before relying on it.

package rework

import (
	"math"
	"testing"
)

// ---------- quadPoint ----------

func TestVarianceLeafIsZero(t *testing.T) {
	if v := newQuadPoint(Point{}, 1, 1).variance(); v != 0 {
		t.Fatalf("leaf variance %v", v)
	}
}

func TestVarianceFourChildren(t *testing.T) {
	p := newQuadPoint(Point{}, 1, 1)
	for i, w := range []float64{0, 0, 1, 1} {
		p.children[i] = &quadPoint{w: w}
	}
	if v := p.variance(); v != 0.25 {
		t.Fatalf("variance %v want 0.25", v)
	}
}

func TestVarianceUsesAllLeafDescendants(t *testing.T) {
	// One child is split further; variance is over the 7 leaves.
	p := newQuadPoint(Point{}, 1, 1)
	for i := range p.children {
		p.children[i] = &quadPoint{w: 0}
	}
	for i := range p.children[0].children {
		p.children[0].children[i] = &quadPoint{w: 7}
	}
	ws := collectLeaves(p, nil)
	if len(ws) != 7 {
		t.Fatalf("collected %d leaves, want 7", len(ws))
	}
	mean := 28.0 / 7
	want := (4*(7-mean)*(7-mean) + 3*mean*mean) / 7
	if v := p.variance(); !approxEq(v, want) {
		t.Fatalf("variance %v want %v", v, want)
	}
}

// ---------- Algorithm 1: division ----------

func TestDivisionConstantStopsAtInitialDepth(t *testing.T) {
	cfg := baseCfg()
	g := mustBindConn(t, linearConnCPPN(0, 0, 0, 0, 0.5, Linear))
	root := divisionAndInitialization(Point{0, -1}, true, *g, cfg)
	checkTreeInvariants(t, root)
	ls := leaves(root)
	if want := int(math.Pow(4, float64(cfg.initialDepth))); len(ls) != want {
		t.Errorf("%d leaves, want %d", len(ls), want)
	}
	for _, l := range ls {
		if l.level != cfg.initialDepth+1 {
			t.Errorf("leaf level %d, want %d", l.level, cfg.initialDepth+1)
		}
	}
}

func TestDivisionRespectsMaxDepth(t *testing.T) {
	cfg := baseCfg()
	cfg.divThreshold = 0
	for seed := int64(0); seed < 5; seed++ {
		g := mustBindConn(t, randomConnCPPN(seed))
		root := divisionAndInitialization(Point{0, -1}, true, *g, cfg)
		checkTreeInvariants(t, root)
		for _, l := range leaves(root) {
			if l.level > cfg.maxDepth+1 {
				t.Errorf("seed %d: leaf at level %d beyond maxDepth+1=%d", seed, l.level, cfg.maxDepth+1)
			}
		}
	}
}

func TestDivisionRefinesOnlyNearEdge(t *testing.T) {
	cfg := baseCfg()
	cfg.initialDepth = 1
	// w = 1 if target x > 0.3. An edge at x = 0 lies on cell borders and is
	// never refined, so use an off-grid edge.
	g := mustBindConn(t, linearConnCPPN(0, 0, 1, 0, -0.3, Step))
	root := divisionAndInitialization(Point{0, -1}, true, *g, cfg)
	checkTreeInvariants(t, root)
	deepest := 0
	for _, l := range leaves(root) {
		deepest = max(deepest, l.level)
		// A deep leaf exists only because its grandparent straddled the edge
		// (grandparent half-width = 4 * leaf width, so |X - 0.3| <= 7 * width).
		if l.level > 3 && math.Abs(l.X-0.3) > 7*l.width {
			t.Errorf("refined leaf %v (level %d) is far from the edge x=0.3", l.Point, l.level)
		}
	}
	if deepest <= 3 {
		t.Errorf("no refinement near the edge (deepest level %d)", deepest)
	}
}

func TestDivisionGridAlignedEdgeIsNotRefined(t *testing.T) {
	cfg := baseCfg()
	cfg.initialDepth = 1
	g := mustBindConn(t, linearConnCPPN(0, 0, 1, 0, 0, Step)) // edge at x = 0
	root := divisionAndInitialization(Point{0, -1}, true, *g, cfg)
	for _, l := range leaves(root) {
		if l.level > 3 {
			t.Fatalf("leaf %v at level %d: grid-aligned edge was refined", l.Point, l.level)
		}
	}
}

func TestDivisionDirection(t *testing.T) {
	cfg := baseCfg()
	g := mustBindConn(t, linearConnCPPN(1, 0, 0, 0, 0, Linear)) // w = source x
	fixed := Point{0.3, -1}
	for _, l := range leaves(divisionAndInitialization(fixed, true, *g, cfg)) {
		if l.w != fixed.X {
			t.Fatalf("outgoing: leaf %v w=%v, want fixed.X=%v", l.Point, l.w, fixed.X)
		}
	}
	for _, l := range leaves(divisionAndInitialization(fixed, false, *g, cfg)) {
		if l.w != l.X {
			t.Fatalf("incoming: leaf %v w=%v, want its own X", l.Point, l.w)
		}
	}
}

// ---------- Algorithm 2: pruning and extraction ----------

func TestPruneConstantFindsNothing(t *testing.T) {
	if got := extract(t, linearConnCPPN(0, 0, 0, 0, 0.7, Linear), Point{0, -1}, true, baseCfg()); len(got) != 0 {
		t.Fatalf("constant pattern produced %d connections", len(got))
	}
}

func TestPruneStepEdgeIsNotABand(t *testing.T) {
	// On one side of an edge the weight does not change, so min(...) is 0.
	for _, bias := range []float64{0, -0.3} {
		if got := extract(t, linearConnCPPN(0, 0, 1, 0, bias, Step), Point{0, -1}, true, baseCfg()); len(got) != 0 {
			t.Errorf("step edge (bias %v) produced %d connections", bias, len(got))
		}
	}
}

func TestPruneStripeFindsOnlyStripe(t *testing.T) {
	// w = gaussian(8*(y2 + 0.5)): a horizontal band at target y = -0.5.
	got := extract(t, linearConnCPPN(0, 0, 0, 8, 4, Gaussian), Point{0, -1}, true, baseCfg())
	if len(got) == 0 {
		t.Fatal("stripe produced no connections")
	}
	for _, c := range got {
		if math.Abs(c.To.Y+0.5) > 0.3 {
			t.Errorf("connection to %v is outside the stripe", c.To)
		}
	}
}

func TestPruneDirectionAndKeys(t *testing.T) {
	ann := linearConnCPPN(0, 0, 0, 8, 4, Gaussian)
	fixed := Point{0, -1}
	for _, outgoing := range []bool{true, false} {
		for k, c := range extract(t, ann, fixed, outgoing, baseCfg()) {
			if k != [2]Point{c.From, c.To} {
				t.Errorf("key %v does not match connection %v->%v", k, c.From, c.To)
			}
			if outgoing && c.From != fixed || !outgoing && c.To != fixed {
				t.Errorf("outgoing=%v: connection %v->%v does not use the fixed point", outgoing, c.From, c.To)
			}
			if len(c.values) == 0 || c.GetWeight() != c.values[WEIGHT_INDEX] {
				t.Errorf("connection %v->%v has bad values %v", c.From, c.To, c.values)
			}
		}
	}
}

func TestPruneConnectionsCarryCPPNOutput(t *testing.T) {
	ann := linearConnCPPN(0, 0, 0, 8, 4, Gaussian)
	g := mustBindConn(t, ann)
	fixed := Point{0, -1}
	for _, c := range extract(t, ann, fixed, true, baseCfg()) {
		w, _ := g.Sample(c.From, c.To)
		if c.GetWeight() != w {
			t.Errorf("%v->%v: stored weight %v, CPPN gives %v", c.From, c.To, c.GetWeight(), w)
		}
	}
}

func TestPruneTargetsStrictlyInside(t *testing.T) {
	for seed := int64(0); seed < 10; seed++ {
		for _, c := range extract(t, randomConnCPPN(seed), Point{0, -1}, true, baseCfg()) {
			if math.Abs(c.To.X) >= 1 || math.Abs(c.To.Y) >= 1 {
				t.Errorf("seed %d: target %v on or outside the edge", seed, c.To)
			}
		}
	}
}

func TestPruneZeroVarThresholdNoPanic(t *testing.T) {
	cfg := baseCfg()
	cfg.varThreshold = 0
	noPanic(t, func() { extract(t, randomConnCPPN(3), Point{0, -1}, true, cfg) })
}
