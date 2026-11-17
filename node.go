package esneatforhebbians

// NodeCPPN computes per-node values (for example bias, time constant, or
// activation-function selectors) from a node's absolute position on the
// substrate. Each output is one channel of a 2D map over the substrate.
// bias and networkType have the same meaning as for CPPN.
//
// Every call must return the same number of outputs. An implementation may
// reuse the returned slice between calls; callers copy the values they keep.
type NodeCPPN interface {
	ComputeNode(x, y, bias float64, networkType int) []float64
	GetOutputSize() int
}

// NodeQuery is a NodeCPPN with its bias and network type fixed, so that only
// the position remains. BindNodeCPPN creates it.
type NodeQuery interface {
	QueryNode(x, y float64) []float64
}

// BindNodeCPPN fixes bias and networkType for every query made through the
// result.
func BindNodeCPPN(c NodeCPPN, bias float64, networkType int) NodeQuery {
	return boundNodeCPPN{c, bias, networkType}
}

type boundNodeCPPN struct {
	cppn        NodeCPPN
	bias        float64
	networkType int
}

func (b boundNodeCPPN) QueryNode(x, y float64) []float64 {
	return b.cppn.ComputeNode(x, y, b.bias, b.networkType)
}

// NodeKind says what role a node plays in the network.
type NodeKind int

const (
	InputNode NodeKind = iota
	HiddenNode
	OutputNode
)

func (k NodeKind) String() string {
	switch k {
	case InputNode:
		return "input"
	case HiddenNode:
		return "hidden"
	case OutputNode:
		return "output"
	}
	return "unknown"
}

// SpacialNode is a neuron on the substrate. values holds the node CPPN's
// outputs at (x, y).
type SpacialNode struct {
	x, y   float64
	values []float64
}

// NewSpacialNode creates a node at (x, y) and fills its values from ncppn.
func NewSpacialNode(x, y float64, ncppn NodeQuery) *SpacialNode {
	out := ncppn.QueryNode(x, y)
	return &SpacialNode{
		x:      x,
		y:      y,
		values: append([]float64(nil), out...),
	}
}
