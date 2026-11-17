package esneatforhebbians

// QuadPoint is a node in the quadtree that ES-HyperNEAT uses to sample the
// CPPN's weight pattern. width is the half-width of the square region the node
// covers, centered on (x, y). values holds every CPPN output sampled at (x, y),
// and w is a copy of values[WeightIndex], the weight the search runs on.
type QuadPoint struct {
	children []*QuadPoint
	width    float64
	level    int
	x        float64
	y        float64
	w        float64
	values   []float64
}

func NewQuadPoint(x, y, width float64, level int) *QuadPoint {
	return &QuadPoint{
		children: make([]*QuadPoint, 4),
		width:    width,
		level:    level,
		x:        x,
		y:        y,
		w:        0.0,
	}
}

// collectLeaves appends the weights of all leaf descendants of p to out.
func collectLeaves(p *QuadPoint, out []float64) []float64 {
	leaf := true
	for _, c := range p.children {
		if c != nil {
			leaf = false
			out = collectLeaves(c, out)
		}
	}
	if leaf {
		out = append(out, p.w)
	}
	return out
}

// Variance returns the variance of the weights of all leaf descendants of p.
// A leaf has no spread, so its variance is 0.
func Variance(p *QuadPoint) float64 {
	if p.children[0] == nil {
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
