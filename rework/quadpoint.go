package rework

type quadPoint struct {
	Point
	width    float64
	level    int
	w        float64
	values   []float64
	children [4]*quadPoint
}

func newQuadPoint(pos Point, width float64, level int) *quadPoint {
	return &quadPoint{
		Point: pos,
		width: width,
		level: level,
	}
}

func (qp *quadPoint) isLeaf() bool {
	return qp.children[0] == nil
}

func collectLeaves(p *quadPoint, out []float64) []float64 {
	if p.isLeaf() {
		return append(out, p.w)
	} else {
		for _, c := range p.children {
			out = collectLeaves(c, out)
		}
	}
	return out
}

func (p *quadPoint) variance() float64 {
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
