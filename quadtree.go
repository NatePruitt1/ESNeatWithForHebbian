package esneatforhebbians

// QuadPoint is a node in the quadtree that ES-HyperNEAT uses to sample the
// CPPN's weight pattern. It covers the square centered on its Point with
// half-width width. values holds every CPPN output sampled at its Point, and w
// is a copy of values[WeightIndex], the weight the search runs on. A leaf has
// no children (all four are nil).
type QuadPoint struct {
	Point
	width    float64
	level    int
	w        float64
	values   []float64
	children [4]*QuadPoint
}

func NewQuadPoint(p Point, width float64, level int) *QuadPoint {
	return &QuadPoint{Point: p, width: width, level: level}
}

// isLeaf reports whether p has no children.
func (p *QuadPoint) isLeaf() bool {
	return p.children[0] == nil
}

// collectLeaves appends the weights of all leaf descendants of p to out.
func collectLeaves(p *QuadPoint, out []float64) []float64 {
	if p.isLeaf() {
		return append(out, p.w)
	}
	for _, c := range p.children {
		out = collectLeaves(c, out)
	}
	return out
}

// Variance returns the variance of the weights of all leaf descendants of p.
// A leaf has no spread, so its variance is 0.
func Variance(p *QuadPoint) float64 {
	if p.isLeaf() {
		return 0
	}
	ws := collectLeaves(p, nil)
	mean := 0.0
	for _, w := range ws {
		mean += w
	}
	mean /= float64(len(ws))
	v := 0.0
	for _, w := range ws {
		d := w - mean
		v += d * d
	}
	return v / float64(len(ws))
}
