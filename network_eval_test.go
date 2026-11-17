package esneatforhebbians

import (
	"math"
	"slices"
	"strings"
	"testing"
)

func mustNetwork(t *testing.T, inputs, outputs int, nodes []NetworkNode, conns []NetworkConnection, recurrent bool) *Network {
	t.Helper()
	net, err := NewNetwork(inputs, outputs, nodes, conns, recurrent)
	if err != nil {
		t.Fatal(err)
	}
	return net
}

// Layout [input 0, output 1, hidden 2, hidden 3]: input -> A -> B -> output,
// with A's bias and activation applied.
func TestNetworkForwardChain(t *testing.T) {
	nodes := []NetworkNode{{}, {}, {Activation: math.Abs}, {Bias: 1}}
	net := mustNetwork(t, 1, 1, nodes, []NetworkConnection{
		{From: 0, To: 2, Weight: 1, Forward: true},
		{From: 2, To: 3, Weight: 3, Forward: true},
		{From: 3, To: 1, Weight: 2, Forward: true},
	}, false)
	// output = 2 * (1 + 3*|x|)
	for _, x := range []float64{-2, 0.5} {
		if got, want := net.Compute([]float64{x})[0], 2*(1+3*math.Abs(x)); got != want {
			t.Fatalf("x=%g: got %g, want %g", x, got, want)
		}
	}
}

func TestNetworkRecurrentIsDelayed(t *testing.T) {
	// input 0 -> hidden 2; hidden 2 -> itself (recurrent, accumulates); 2 -> output 1.
	net := mustNetwork(t, 1, 1, make([]NetworkNode, 3), []NetworkConnection{
		{From: 0, To: 2, Weight: 1, Forward: true},
		{From: 2, To: 2, Weight: 1, Forward: false},
		{From: 2, To: 1, Weight: 1, Forward: true},
	}, true)
	var got []float64
	for range 3 {
		got = append(got, net.Compute([]float64{1})[0])
	}
	if !slices.Equal(got, []float64{1, 2, 3}) {
		t.Fatalf("got %v, want [1 2 3]", got)
	}
	net.Reset()
	if v := net.Compute([]float64{1})[0]; v != 1 {
		t.Fatalf("after Reset got %g, want 1", v)
	}
}

// The layout rules apply to both network types, so test them through both
// constructors.
func TestLayoutRules(t *testing.T) {
	// Layout for every case: input 0, output 1, hidden 2, hidden 3.
	cases := []struct {
		name      string
		links     []link
		recurrent bool
		want      string // "" means valid
	}{
		{"forward chain", []link{{0, 2, true}, {2, 3, true}, {3, 1, true}}, false, ""},
		{"input straight to output", []link{{0, 1, true}}, false, ""},
		{"recurrent backward, allowed", []link{{3, 2, false}}, true, ""},
		{"recurrent self, allowed", []link{{2, 2, false}}, true, ""},
		{"recurrent from output, allowed", []link{{1, 2, false}}, true, ""},
		{"recurrent when not allowed", []link{{3, 2, false}}, false, "does not allow recurrent"},
		{"forward hidden to lower index", []link{{3, 2, true}}, true, "higher index"},
		{"forward hidden self", []link{{2, 2, true}}, true, "higher index"},
		{"forward from output", []link{{1, 2, true}}, true, "from output"},
		{"forward output to itself", []link{{1, 1, true}}, true, "from output"},
		{"forward into input", []link{{2, 0, true}}, true, "ends at input"},
		{"recurrent into input", []link{{2, 0, false}}, true, "ends at input"},
		{"index out of range", []link{{0, 9, true}}, true, "outside"},
	}
	for _, tc := range cases {
		nconns := make([]NetworkConnection, len(tc.links))
		hconns := make([]HebbianConnection, len(tc.links))
		for i, l := range tc.links {
			nconns[i] = NetworkConnection{From: l.from, To: l.to, Forward: l.forward}
			hconns[i] = HebbianConnection{From: l.from, To: l.to, Forward: l.forward}
		}
		_, nerr := NewNetwork(1, 1, make([]NetworkNode, 4), nconns, tc.recurrent)
		checks := map[string]error{"Network": nerr}
		if tc.recurrent { // Hebbian networks always allow recurrence
			_, checks["HebbianNetwork"] = NewHebbianNetwork(1, 1, make([]HebbianNode, 4), hconns)
		}
		for kind, err := range checks {
			switch {
			case tc.want == "" && err != nil:
				t.Errorf("%s, %s: unexpected error %v", tc.name, kind, err)
			case tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)):
				t.Errorf("%s, %s: got error %v, want one mentioning %q", tc.name, kind, err, tc.want)
			}
		}
	}
	if _, err := NewNetwork(2, 2, make([]NetworkNode, 3), nil, false); err == nil {
		t.Error("more inputs and outputs than nodes should be an error")
	}
}

func TestConvertToNetwork(t *testing.T) {
	// [input, output, hidden]: input -> hidden -> output, plus a recurrent
	// self-connection on the hidden node.
	an := &AbstractNetwork{
		Nodes: []AbstractNode{
			{Posy: InputY, ActivationFunc: Linear},
			{Posy: OutputY, ActivationFunc: Tanh},
			{Posy: 0, ActivationFunc: Gaussion},
		},
		Connections: []AbstractConnection{
			{From: 0, To: 2, Weight: 0.5, Forward: true},
			{From: 2, To: 1, Weight: -2, Forward: true},
			{From: 2, To: 2, Weight: 1, Forward: false},
		},
	}
	ff, err := an.ConvertToNetwork(1, 1, false)
	if err != nil {
		t.Fatal(err)
	}
	if len(ff.Connections) != 2 {
		t.Fatalf("feed-forward conversion kept %d connections, want 2 (recurrent one left out)", len(ff.Connections))
	}
	x := 1.5
	want := math.Tanh(-2 * math.Exp(-(0.5*x)*(0.5*x)))
	for range 2 { // no state, so repeated calls agree
		if got := ff.Compute([]float64{x})[0]; math.Abs(got-want) > 1e-12 {
			t.Fatalf("got %g, want %g", got, want)
		}
	}
	rec, err := an.ConvertToNetwork(1, 1, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(rec.Connections) != 3 {
		t.Fatalf("recurrent conversion kept %d connections, want 3", len(rec.Connections))
	}
}

// Networks generated by ES-HyperNEAT satisfy the layout rules, with and
// without recurrence.
func TestGeneratedNetworksFollowLayout(t *testing.T) {
	an, err := GenerateAbstractFromCPPNs(waveCPPN{}, &scoredNodeCPPN{}, 1, 0, true, 3, 2, 3, 5, -1, 0, 0.03, 0.3, 0.03)
	if err != nil {
		t.Fatal(err)
	}
	rec, err := an.ConvertToNetwork(3, 2, true)
	if err != nil {
		t.Fatalf("recurrent conversion: %v", err)
	}
	ff, err := an.ConvertToNetwork(3, 2, false)
	if err != nil {
		t.Fatalf("feed-forward conversion: %v", err)
	}
	t.Logf("%d nodes; %d connections with recurrence, %d without", len(an.Nodes), len(rec.Connections), len(ff.Connections))
	rec.Compute([]float64{1, 0, -1})
	ff.Compute([]float64{1, 0, -1})
}

// End to end: Networks act as both CPPNs for ES-HyperNEAT, and the result is
// identical to using hand-written Go CPPNs that compute the same functions.

// sineCPPN computes weight = sin(4*(x2 - x1)) + 0.3*y2 and extra = bias + networkType.
type sineCPPN struct{}

func (sineCPPN) Compute(x1, y1, x2, y2, bias float64, networkType int) []float64 {
	return []float64{math.Sin(4*(x2-x1)) + 0.3*y2, bias + float64(networkType)}
}
func (sineCPPN) GetOutputSize() int { return 2 }

// sineCPPNNetwork builds the same function as a Network.
// Layout: inputs 0-5 (x1 y1 x2 y2 bias type), outputs 6-7, hidden 8 (sine).
func sineCPPNNetwork(t *testing.T) *NetworkCPPN {
	nodes := make([]NetworkNode, 9)
	nodes[8].Activation = math.Sin
	net := mustNetwork(t, 6, 2, nodes, []NetworkConnection{
		{From: 0, To: 8, Weight: -4, Forward: true},
		{From: 2, To: 8, Weight: 4, Forward: true},
		{From: 8, To: 6, Weight: 1, Forward: true},
		{From: 3, To: 6, Weight: 0.3, Forward: true},
		{From: 4, To: 7, Weight: 1, Forward: true},
		{From: 5, To: 7, Weight: 1, Forward: true},
	}, false)
	c, err := NewNetworkCPPN(net)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

// rowNodeCPPN scores Tanh highest when y > 0, Sigmoid otherwise.
type rowNodeCPPN struct{}

func (rowNodeCPPN) ComputeNode(x, y, bias float64, networkType int) []float64 {
	return []float64{0, -y, y, 0, 0}
}
func (rowNodeCPPN) GetOutputSize() int { return 5 }

// rowNodeNetwork builds the same function as a Network.
// Layout: inputs 0-3 (x y bias type), outputs 4-8 (one per activation function).
func rowNodeNetwork(t *testing.T) *NetworkNodeCPPN {
	net := mustNetwork(t, 4, 5, make([]NetworkNode, 9), []NetworkConnection{
		{From: 1, To: 5, Weight: -1, Forward: true},
		{From: 1, To: 6, Weight: 1, Forward: true},
	}, false)
	c, err := NewNetworkNodeCPPN(net)
	if err != nil {
		t.Fatal(err)
	}
	return c
}

func TestNetworkAsCPPN(t *testing.T) {
	want, err := GenerateAbstractFromCPPNs(sineCPPN{}, rowNodeCPPN{}, 1, 2, true, 3, 2, 3, 5, 1, 0, 0.03, 0.3, 0.03)
	if err != nil {
		t.Fatal(err)
	}
	got, err := GenerateAbstractFromCPPNs(sineCPPNNetwork(t), rowNodeNetwork(t), 1, 2, true, 3, 2, 3, 5, 1, 0, 0.03, 0.3, 0.03)
	if err != nil {
		t.Fatal(err)
	}
	if len(got.Nodes) != len(want.Nodes) || len(got.Connections) != len(want.Connections) {
		t.Fatalf("network CPPN gave %d nodes / %d connections, Go CPPN gave %d / %d",
			len(got.Nodes), len(got.Connections), len(want.Nodes), len(want.Connections))
	}
	for i := range want.Nodes {
		g, w := got.Nodes[i], want.Nodes[i]
		if g.Posx != w.Posx || g.Posy != w.Posy || g.ActivationFunc != w.ActivationFunc {
			t.Fatalf("node %d differs: %+v vs %+v", i, g, w)
		}
	}
	for i := range want.Connections {
		g, w := got.Connections[i], want.Connections[i]
		if g.From != w.From || g.To != w.To || math.Abs(g.Weight-w.Weight) > 1e-12 || g.Forward != w.Forward || g.Values[1] != 3 {
			t.Fatalf("connection %d differs: %+v vs %+v", i, g, w)
		}
	}
	t.Logf("network CPPNs reproduced the Go CPPNs exactly: %d nodes, %d connections", len(got.Nodes), len(got.Connections))
}

func TestNetworkCPPNRules(t *testing.T) {
	recurrent := mustNetwork(t, 6, 1, make([]NetworkNode, 7), nil, true)
	if _, err := NewNetworkCPPN(recurrent); err == nil || !strings.Contains(err.Error(), "recurrent") {
		t.Fatalf("a network with recurrence on should not be a CPPN: %v", err)
	}
	if _, err := NewNetworkNodeCPPN(mustNetwork(t, 4, 1, make([]NetworkNode, 5), nil, true)); err == nil {
		t.Fatal("a node CPPN network with recurrence on should be rejected")
	}
	wrongInputs := mustNetwork(t, 2, 1, make([]NetworkNode, 3), nil, false)
	if _, err := NewNetworkCPPN(wrongInputs); err == nil {
		t.Fatal("a 2-input network should not be accepted as a connection CPPN")
	}
	if _, err := NewNetworkNodeCPPN(wrongInputs); err == nil {
		t.Fatal("a 2-input network should not be accepted as a node CPPN")
	}
}
