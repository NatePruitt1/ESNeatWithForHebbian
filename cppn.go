package esneatforhebbians

// Connection-CPPN output indices. WeightIndex is the weight ES-HyperNEAT
// searches on; the others are read by ConvertToHebbian.
const (
	WeightIndex = 0
	AIndex      = 1
	BIndex      = 2
	CIndex      = 3
	DIndex      = 4
	DirIndex    = 5
	EtaIndex    = 6
)

// HebbianValueCount is the number of connection-CPPN outputs ConvertToHebbian
// reads: WeightIndex through EtaIndex.
const HebbianValueCount = EtaIndex + 1

// CPPN computes the outputs for a connection from (x1, y1) to (x2, y2).
//
// bias is a constant input, the same for every query in one network build.
// networkType selects which network is being built, so one CPPN can encode
// several networks; how it is fed into the CPPN is up to the implementation.
//
// The output at WeightIndex is the connection weight. Any further outputs (for
// example the Hebbian coefficients) do not affect the search; they are carried
// along onto the extracted connections.
//
// Every call must return the same number of outputs, at least one. An
// implementation may reuse the returned slice between calls; callers copy the
// values they keep.
type CPPN interface {
	Compute(x1, y1, x2, y2, bias float64, networkType int) []float64
	GetOutputSize() int
}

// NodeCPPN computes per-node values from a node's position. Outputs
// ActivationChannel onward score the activation functions (see
// activations.go). bias and networkType mean the same as for CPPN, and the
// same rules about outputs apply.
type NodeCPPN interface {
	ComputeNode(x, y, bias float64, networkType int) []float64
	GetOutputSize() int
}

// ConnectionQuery is a CPPN with its bias and network type fixed, so that only
// positions remain. The search works on this view; BindCPPN creates it.
type ConnectionQuery interface {
	Query(from, to Point) []float64
}

// NodeQuery is a NodeCPPN with its bias and network type fixed. BindNodeCPPN
// creates it.
type NodeQuery interface {
	QueryNode(p Point) []float64
}

// BindCPPN fixes bias and networkType for every query made through the result.
func BindCPPN(c CPPN, bias float64, networkType int) ConnectionQuery {
	return boundCPPN{c, bias, networkType}
}

// BindNodeCPPN fixes bias and networkType for every query made through the
// result.
func BindNodeCPPN(c NodeCPPN, bias float64, networkType int) NodeQuery {
	return boundNodeCPPN{c, bias, networkType}
}

type boundCPPN struct {
	cppn        CPPN
	bias        float64
	networkType int
}

func (b boundCPPN) Query(from, to Point) []float64 {
	return b.cppn.Compute(from.X, from.Y, to.X, to.Y, b.bias, b.networkType)
}

type boundNodeCPPN struct {
	cppn        NodeCPPN
	bias        float64
	networkType int
}

func (b boundNodeCPPN) QueryNode(p Point) []float64 {
	return b.cppn.ComputeNode(p.X, p.Y, b.bias, b.networkType)
}
