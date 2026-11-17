package esneatforhebbians

// WeightIndex is the position of the connection weight in a CPPN's outputs.
const WeightIndex = 0
const AIndex = 1
const BIndex = 2
const CIndex = 3
const DIndex = 4
const DirIndex = 5
const EtaIndex = 6

// CPPN computes the outputs for a connection from (x1, y1) to (x2, y2).
//
// bias is a constant input, the same for every query in one network build; it
// lets the CPPN shift its outputs independently of position. networkType is a
// counter that selects which network is being built, so one CPPN can encode
// several networks (for example, one per role in a multi-network agent). How
// networkType is fed into the CPPN (raw, scaled, one-hot, ...) is up to the
// implementation.
//
// The output at WeightIndex is the connection weight, which ES-HyperNEAT uses
// to decide where to subdivide and which connections to extract. Any further
// outputs (for example Hebbian A, B, C, D coefficients) do not affect the
// search; they are stored on the quadtree nodes and copied onto the extracted
// connections.
//
// Every call must return the same number of outputs, at least one. An
// implementation may reuse the returned slice between calls; callers copy the
// values they keep.
type CPPN interface {
	Compute(x1, y1, x2, y2, bias float64, networkType int) []float64
	GetOutputSize() int
}

// ConnectionQuery is a CPPN with its bias and network type fixed, so that only
// positions remain. Algorithms 1 and 2 work on this view; BindCPPN creates it.
type ConnectionQuery interface {
	Query(x1, y1, x2, y2 float64) []float64
}

// BindCPPN fixes bias and networkType for every query made through the result.
func BindCPPN(c CPPN, bias float64, networkType int) ConnectionQuery {
	return boundCPPN{c, bias, networkType}
}

type boundCPPN struct {
	cppn        CPPN
	bias        float64
	networkType int
}

func (b boundCPPN) Query(x1, y1, x2, y2 float64) []float64 {
	return b.cppn.Compute(x1, y1, x2, y2, b.bias, b.networkType)
}
