package esneatforhebbians

import (
	"math"
	"testing"
)

// scoredNodeCPPN gives 5 activation scores followed by one extra value (x + y),
// and counts how often it is called. The score for function (i + j) % 5 is
// highest, where i and j depend on the node's position, so different nodes
// pick different functions.
type scoredNodeCPPN struct{ calls int }

func (c *scoredNodeCPPN) ComputeNode(x, y, bias float64, networkType int) []float64 {
	c.calls++
	out := make([]float64, ActivationFuncCount+1)
	best := (int((x+1)*8) + int((y+1)*8)) % ActivationFuncCount
	out[best] = 1
	out[ActivationFuncCount] = x + y
	return out
}
func (*scoredNodeCPPN) GetOutputSize() int { return ActivationFuncCount + 1 }

func TestChooseActivation(t *testing.T) {
	cases := []struct {
		values []float64
		want   ActivationFuncID
	}{
		{[]float64{1, 0, 0, 0, 0}, Linear},
		{[]float64{0, 0, 0.5, 0.1, 0}, Tanh},
		{[]float64{-3, -2, -1, -5, 0}, Gaussion},
		{[]float64{0, 0, 0, 1, 0, 99}, Step}, // outputs past the channels are ignored
	}
	for _, c := range cases {
		if got := chooseActivation(c.values); got != c.want {
			t.Errorf("chooseActivation(%v) = %d, want %d", c.values, got, c.want)
		}
	}
}

func TestGenerateNetwork(t *testing.T) {
	ncppn := &scoredNodeCPPN{}
	net, err := GenerateAbstractFromCPPNs(waveCPPN{}, ncppn, 1, 0, true, 3, 2, 3, 5, 1, 0, 0.03, 0.3, 0.03)
	if err != nil {
		t.Fatal(err)
	}

	// Compare against the same search done step by step.
	discovered, err := discoverSubstrate(BindCPPN(waveCPPN{}, 1, 0), BindNodeCPPN(&scoredNodeCPPN{}, 1, 0), true, 3, 2, 3, 5, 1, 0, 0.03, 0.3, 0.03)
	if err != nil {
		t.Fatal(err)
	}
	sub := pruneSubstrate(discovered)

	// Every node's CPPN values are computed exactly once, when it is created,
	// and never again during conversion.
	created := len(discovered.inputs) + len(discovered.outputs) + len(discovered.hidden)
	if ncppn.calls != created {
		t.Fatalf("node CPPN called %d times, want %d (once per created node)", ncppn.calls, created)
	}

	if want := len(sub.inputs) + len(sub.outputs) + len(sub.hidden); len(net.Nodes) != want {
		t.Fatalf("got %d nodes, want %d", len(net.Nodes), want)
	}
	seen := map[ActivationFuncID]bool{}
	for i, n := range net.Nodes {
		switch {
		case i < 3:
			if n.Posy != InputY || n.ActivationFunc != Linear {
				t.Fatalf("node %d should be a Linear input, got %+v", i, n)
			}
		case i < 5:
			if n.Posy != OutputY {
				t.Fatalf("node %d should be an output, got %+v", i, n)
			}
		}
		if i >= 3 {
			if n.ActivationFunc != chooseActivation(n.Values) {
				t.Fatalf("node %d activation %d does not match its values", i, n.ActivationFunc)
			}
			seen[n.ActivationFunc] = true
		}
		if len(n.Values) != ActivationFuncCount+1 || n.Values[ActivationFuncCount] != n.Posx+n.Posy {
			t.Fatalf("node %d values %v not carried over", i, n.Values)
		}
	}
	if len(seen) < 2 {
		t.Fatalf("expected several activation functions, got %v", seen)
	}

	if len(net.Connections) != len(sub.connections) {
		t.Fatalf("got %d connections, want %d", len(net.Connections), len(sub.connections))
	}
	for i, c := range net.Connections {
		s := sub.connections[i]
		f, to := net.Nodes[c.From], net.Nodes[c.To]
		if f.Posx != s.fromx || f.Posy != s.fromy || to.Posx != s.tox || to.Posy != s.toy || c.Weight != s.weight {
			t.Fatalf("connection %d does not match its substrate connection", i)
		}
	}
	t.Logf("%d nodes (%d activation functions used), %d connections, %d node-CPPN calls",
		len(net.Nodes), len(seen), len(net.Connections), ncppn.calls)
}

// twoOutputNodeCPPN has too few outputs to choose an activation function.
type twoOutputNodeCPPN struct{}

func (twoOutputNodeCPPN) ComputeNode(x, y, bias float64, networkType int) []float64 {
	return []float64{x, y}
}
func (twoOutputNodeCPPN) GetOutputSize() int { return 2 }

func TestGenerateRejectsSmallNodeCPPN(t *testing.T) {
	if _, err := GenerateAbstractFromCPPNs(waveCPPN{}, twoOutputNodeCPPN{}, 1, 0, true, 3, 2, 3, 5, 1, 0, 0.03, 0.3, 0.03); err == nil {
		t.Fatal("a node CPPN with too few outputs for the activation channels should be an error")
	}
}

func TestConvertMissingEndpoint(t *testing.T) {
	in := &SpacialNode{x: 0, y: InputY, kind: InputNode}
	out := &SpacialNode{x: 0, y: OutputY, kind: OutputNode, values: make([]float64, ActivationFuncCount)}
	sub := &substrate{
		inputs:      []*SpacialNode{in},
		outputs:     []*SpacialNode{out},
		connections: []*SpacialConnection{NewSpacialConnection(0, InputY, 0.5, 0.5, []float64{1})},
	}
	if _, err := convertSubstrate(sub); err == nil {
		t.Fatal("a connection to a missing node should be an error")
	}
}

// typedCPPN shifts its weight pattern by networkType and records every bias
// and networkType it is called with.
type typedCPPN struct {
	biases map[float64]bool
	types  map[int]bool
}

func newTypedCPPN() *typedCPPN {
	return &typedCPPN{biases: map[float64]bool{}, types: map[int]bool{}}
}

func (c *typedCPPN) Compute(x1, y1, x2, y2, bias float64, networkType int) []float64 {
	c.biases[bias], c.types[networkType] = true, true
	shift := 0.7 * float64(networkType)
	return []float64{math.Sin(4*(x2-x1)+shift) * math.Cos(4*(y2-y1)-shift)}
}
func (*typedCPPN) GetOutputSize() int { return 1 }

// typedNodeCPPN records every bias and networkType it is called with.
type typedNodeCPPN struct {
	biases map[float64]bool
	types  map[int]bool
}

func (c *typedNodeCPPN) ComputeNode(x, y, bias float64, networkType int) []float64 {
	c.biases[bias], c.types[networkType] = true, true
	out := make([]float64, ActivationFuncCount)
	out[networkType%ActivationFuncCount] = 1
	return out
}
func (*typedNodeCPPN) GetOutputSize() int { return ActivationFuncCount }

func TestBiasAndNetworkTypeReachCPPNs(t *testing.T) {
	var sizes []int
	for networkType := 0; networkType < 3; networkType++ {
		cc := newTypedCPPN()
		nc := &typedNodeCPPN{biases: map[float64]bool{}, types: map[int]bool{}}
		net, err := GenerateAbstractFromCPPNs(cc, nc, 0.5, networkType, true, 3, 2, 3, 5, 1, 0, 0.03, 0.3, 0.03)
		if err != nil {
			t.Fatal(err)
		}
		for name, seen := range map[string]bool{
			"connection CPPN": len(cc.biases) == 1 && cc.biases[0.5] && len(cc.types) == 1 && cc.types[networkType],
			"node CPPN":       len(nc.biases) == 1 && nc.biases[0.5] && len(nc.types) == 1 && nc.types[networkType],
		} {
			if !seen {
				t.Fatalf("networkType %d: %s did not always get bias 0.5 and networkType %d", networkType, name, networkType)
			}
		}
		for i, n := range net.Nodes[3:] { // inputs are always Linear
			if n.ActivationFunc != ActivationFuncID(networkType%ActivationFuncCount) {
				t.Fatalf("networkType %d: node %d has activation %d", networkType, i+3, n.ActivationFunc)
			}
		}
		sizes = append(sizes, len(net.Connections))
	}
	t.Logf("connections for networkType 0, 1, 2: %v", sizes)
	if sizes[0] == sizes[1] && sizes[1] == sizes[2] {
		t.Fatal("different network types should give different networks")
	}
}

func TestSelfQueryPassesBiasAndType(t *testing.T) {
	cc := newTypedCPPN()
	SelfQuery{cc}.ComputeNode(0.25, 0.5, 2, 7)
	if !cc.biases[2] || !cc.types[7] {
		t.Fatal("SelfQuery did not pass bias and networkType through")
	}
}

// TestHiddenOrderedByY checks that the hidden nodes are sorted by y (then x)
// and that, under the forward rule (strictly increasing y), every forward
// connection between hidden nodes goes from a lower index to a higher one.
// The rule is computed from node positions here and compared against the
// Forward field.
func TestHiddenOrderedByY(t *testing.T) {
	const inputs, outputs = 3, 2
	for _, level := range []int{0, 1, -1} {
		net, err := GenerateAbstractFromCPPNs(waveCPPN{}, &scoredNodeCPPN{}, 1, 0, true, inputs, outputs, 3, 5, level, 0, 0.03, 0.3, 0.03)
		if err != nil {
			t.Fatal(err)
		}
		hidden := net.Nodes[inputs+outputs:]
		for i := 1; i < len(hidden); i++ {
			a, b := hidden[i-1], hidden[i]
			if a.Posy > b.Posy || (a.Posy == b.Posy && a.Posx >= b.Posx) {
				t.Fatalf("hidden nodes %d and %d out of order: (%g,%g) then (%g,%g)", i-1, i, a.Posx, a.Posy, b.Posx, b.Posy)
			}
		}
		forward, lateral := 0, 0
		for _, c := range net.Connections {
			from, to := net.Nodes[c.From], net.Nodes[c.To]
			isForward := to.Posy > from.Posy
			if c.Forward != isForward {
				t.Fatalf("connection (%g,%g)->(%g,%g) has Forward=%v", from.Posx, from.Posy, to.Posx, to.Posy, c.Forward)
			}
			if isForward {
				forward++
			} else {
				lateral++
			}
			hiddenToHidden := c.From >= inputs+outputs && c.To >= inputs+outputs
			if isForward && hiddenToHidden && c.From >= c.To {
				t.Fatalf("forward hidden connection %d -> %d goes to a lower index", c.From, c.To)
			}
			if c.From < inputs && !isForward {
				t.Fatal("input connections must always be forward")
			}
			if c.To >= inputs && c.To < inputs+outputs && !isForward {
				t.Fatal("connections into outputs must always be forward")
			}
		}
		t.Logf("iterationLevel=%d: %d hidden nodes, %d forward and %d lateral/backward connections", level, len(hidden), forward, lateral)
	}
}
