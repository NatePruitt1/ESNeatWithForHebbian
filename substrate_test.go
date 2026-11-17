package esneatforhebbians

import (
	"math"
	"testing"
)

// waveCPPN gives a weight pattern that varies with the offset between the two
// endpoints, so searches from different points find different connections.
type waveCPPN struct{}

func (waveCPPN) Compute(x1, y1, x2, y2, bias float64, networkType int) []float64 {
	return []float64{math.Sin(4*(x2-x1)) * math.Cos(4*(y2-y1)), x1 + x2}
}
func (waveCPPN) GetOutputSize() int { return 2 }

func discover(t *testing.T, iterationLevel, maxHidden int) *substrate {
	t.Helper()
	s, err := discoverSubstrate(BindCPPN(waveCPPN{}, 1, 0), posCPPN{}, true, 3, 2, 3, 5, iterationLevel, maxHidden, 0.03, 0.3, 0.03)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// checkInvariants verifies that every connection joins known nodes in a
// direction Algorithm 3 allows: inputs only send, outputs only receive.
func checkInvariants(t *testing.T, s *substrate) {
	t.Helper()
	kind := map[position]NodeKind{}
	for _, group := range [][]*SpacialNode{s.inputs, s.hidden, s.outputs} {
		for _, n := range group {
			if _, dup := kind[position{n.x, n.y}]; dup {
				t.Fatalf("two nodes at (%g, %g)", n.x, n.y)
			}
			kind[position{n.x, n.y}] = n.kind
		}
	}
	for _, c := range s.connections {
		from, ok1 := kind[position{c.fromx, c.fromy}]
		to, ok2 := kind[position{c.tox, c.toy}]
		if !ok1 || !ok2 {
			t.Fatalf("connection (%g,%g)->(%g,%g) has an unknown endpoint", c.fromx, c.fromy, c.tox, c.toy)
		}
		if from == OutputNode || to == InputNode {
			t.Fatalf("connection %s->%s is not allowed", from, to)
		}
		if from == InputNode && to == OutputNode {
			t.Fatal("unexpected direct input->output connection")
		}
	}
}

func count(s *substrate) (inHid, hidHid, hidOut int) {
	for _, c := range s.connections {
		switch {
		case c.fromy == InputY:
			inHid++
		case c.toy == OutputY:
			hidOut++
		default:
			hidHid++
		}
	}
	return
}

func TestDiscoverSubstrate(t *testing.T) {
	prev := -1
	for _, level := range []int{0, 1, 2} {
		s := discover(t, level, 0)
		checkInvariants(t, s)
		a, b, c := count(s)
		t.Logf("iterationLevel=%d: %d hidden, connections in->hid %d, hid->hid %d, hid->out %d", level, len(s.hidden), a, b, c)
		if level == 0 && b != 0 {
			t.Fatal("iterationLevel 0 should have no hidden->hidden connections")
		}
		if len(s.hidden) < prev {
			t.Fatal("more rounds should never find fewer hidden nodes")
		}
		prev = len(s.hidden)
		if c == 0 {
			t.Fatal("expected some hidden->output connections")
		}
	}
}

func TestDiscoverUntilDone(t *testing.T) {
	all := discover(t, -1, 0)
	checkInvariants(t, all)
	t.Logf("iterationLevel=-1: %d hidden nodes, %d connections", len(all.hidden), len(all.connections))
	if again := discover(t, 1000, 0); len(again.hidden) != len(all.hidden) {
		t.Fatalf("-1 found %d hidden nodes but 1000 rounds found %d", len(all.hidden), len(again.hidden))
	}

	limit := len(all.hidden) / 2
	capped := discover(t, -1, limit)
	checkInvariants(t, capped)
	t.Logf("maxHiddenNodes=%d: %d hidden nodes", limit, len(capped.hidden))
	if len(capped.hidden) >= len(all.hidden) {
		t.Fatal("the cap did not limit exploration")
	}
}

func TestDiscoverBadSizes(t *testing.T) {
	if _, err := discoverSubstrate(BindCPPN(waveCPPN{}, 1, 0), posCPPN{}, true, -1, 2, 3, 5, 1, 0, 0.03, 0.3, 0.03); err == nil {
		t.Fatal("negative input size should be an error")
	}
}
