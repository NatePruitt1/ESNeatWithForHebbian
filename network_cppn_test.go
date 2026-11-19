package esneatforhebbians

// AI-generated regression tests for using a Network as a CPPN
// (network_cppn.go) and for binding bias and network type (cppn.go).

import "testing"

// Connection CPPN network: inputs x1 y1 x2 y2 bias type (0-5), one output (6).
// output = 1*x1 + 10*y1 + 100*x2 + 1000*y2 + 10000*bias + 100000*type, so each
// input's position in the result shows it arrived in the right slot.
func TestNetworkCPPNInputOrder(t *testing.T) {
	var conns []Connection
	for i, w := range []float64{1, 10, 100, 1000, 10000, 100000} {
		conns = append(conns, Connection{From: i, To: 6, Weight: w, Forward: true})
	}
	cppn, err := NewNetworkCPPN(mustNetwork(t, 6, 1, make([]Node, 7), conns, false))
	if err != nil {
		t.Fatal(err)
	}
	if got := cppn.Compute(1, 2, 3, 4, 5, 6)[0]; got != 654321 {
		t.Fatalf("got %g, want 654321", got)
	}
	// BindCPPN passes bias and network type on every query.
	if got := BindCPPN(cppn, 5, 6).Query(Point{1, 2}, Point{3, 4})[0]; got != 654321 {
		t.Fatalf("bound query got %g, want 654321", got)
	}
	if cppn.GetOutputSize() != 1 {
		t.Fatalf("output size %d, want 1", cppn.GetOutputSize())
	}
}

// Node CPPN network: inputs x y bias type (0-3), one output (4).
func TestNetworkNodeCPPNInputOrder(t *testing.T) {
	var conns []Connection
	for i, w := range []float64{1, 10, 100, 1000} {
		conns = append(conns, Connection{From: i, To: 4, Weight: w, Forward: true})
	}
	ncppn, err := NewNetworkNodeCPPN(mustNetwork(t, 4, 1, make([]Node, 5), conns, false))
	if err != nil {
		t.Fatal(err)
	}
	if got := ncppn.ComputeNode(1, 2, 3, 4)[0]; got != 4321 {
		t.Fatalf("got %g, want 4321", got)
	}
	if got := BindNodeCPPN(ncppn, 3, 4).QueryNode(Point{1, 2})[0]; got != 4321 {
		t.Fatalf("bound query got %g, want 4321", got)
	}
}

// Queries are independent: a CPPN network has no state between them.
func TestNetworkCPPNQueriesAreIndependent(t *testing.T) {
	nodes := make([]Node, 8)
	nodes[7].Activation = Sine
	cppn, err := NewNetworkCPPN(mustNetwork(t, 6, 1, nodes, []Connection{
		{From: 0, To: 7, Weight: 2, Forward: true},
		{From: 2, To: 7, Weight: -1, Forward: true},
		{From: 7, To: 6, Weight: 1, Forward: true},
	}, false))
	if err != nil {
		t.Fatal(err)
	}
	first := cppn.Compute(0.3, 0, 0.8, 0, 1, 0)[0]
	cppn.Compute(-1, 1, 1, -1, 1, 3)
	if again := cppn.Compute(0.3, 0, 0.8, 0, 1, 0)[0]; again != first {
		t.Fatalf("same query gave %g then %g", first, again)
	}
}

func TestNetworkCPPNRules(t *testing.T) {
	if _, err := NewNetworkCPPN(mustNetwork(t, 6, 1, make([]Node, 7), nil, true)); err == nil {
		t.Error("a connection CPPN with recurrence on should be rejected")
	}
	if _, err := NewNetworkCPPN(mustNetwork(t, 4, 1, make([]Node, 5), nil, false)); err == nil {
		t.Error("a connection CPPN needs 6 inputs")
	}
	if _, err := NewNetworkCPPN(mustNetwork(t, 6, 0, make([]Node, 6), nil, false)); err == nil {
		t.Error("a connection CPPN needs at least 1 output")
	}
	if _, err := NewNetworkNodeCPPN(mustNetwork(t, 4, 1, make([]Node, 5), nil, true)); err == nil {
		t.Error("a node CPPN with recurrence on should be rejected")
	}
	if _, err := NewNetworkNodeCPPN(mustNetwork(t, 6, 1, make([]Node, 7), nil, false)); err == nil {
		t.Error("a node CPPN needs 4 inputs")
	}
}
