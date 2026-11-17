package esneatforhebbians

import (
	"math"
	"slices"
)

// This file implements Algorithm 1 (DivisionAndInitialization) and Algorithm 2
// (PruningAndExtraction) of ES-HyperNEAT, as described in:
//
//	S. Risi and K. O. Stanley, "An Enhanced Hypercube-Based Encoding for
//	Evolving the Placement, Density, and Connectivity of Neurons,"
//	Artificial Life, vol. 18, no. 4, pp. 331-363, 2012.
//	doi:10.1162/ARTL_a_00071
//
// Both algorithms search around one fixed point (the paper's (a, b)). If
// outgoing is true, the fixed point is the source of every connection queried;
// otherwise it is the target.
//
// One deliberate difference from the paper's pseudocode: Variance is computed
// over all leaf descendants of a node, not only its direct children.

// sample queries the CPPN for the connection between fixed and p and returns
// all of its outputs. The result may be reused by the CPPN.
func sample(cppn ConnectionQuery, fixed, p Point, outgoing bool) []float64 {
	if outgoing {
		return cppn.Query(fixed, p)
	}
	return cppn.Query(p, fixed)
}

// sampleWeight is like sample but returns only the weight output.
func sampleWeight(cppn ConnectionQuery, fixed, p Point, outgoing bool) float64 {
	return sample(cppn, fixed, p, outgoing)[WeightIndex]
}

// DivisionAndInitialization (Algorithm 1) builds a quadtree over the substrate
// for the fixed point, subdividing wherever the CPPN's weights vary.
func DivisionAndInitialization(fixed Point, outgoing bool, cppn ConnectionQuery, initialDepth, maxDepth int, divThreshold float64) *QuadPoint {
	root := NewQuadPoint(Point{0, 0}, 1, 1)
	queue := []*QuadPoint{root}
	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]

		r := p.width / 2
		next := p.level + 1
		p.children[0] = NewQuadPoint(Point{p.X - r, p.Y - r}, r, next)
		p.children[1] = NewQuadPoint(Point{p.X - r, p.Y + r}, r, next)
		p.children[2] = NewQuadPoint(Point{p.X + r, p.Y - r}, r, next)
		p.children[3] = NewQuadPoint(Point{p.X + r, p.Y + r}, r, next)
		for _, c := range p.children {
			c.values = slices.Clone(sample(cppn, fixed, c.Point, outgoing))
			c.w = c.values[WeightIndex]
		}

		if p.level < initialDepth || (p.level < maxDepth && Variance(p) > divThreshold) {
			queue = append(queue, p.children[:]...)
		}
	}
	return root
}

// PruningAndExtraction (Algorithm 2) walks the quadtree under p and appends a
// connection for every low-variance point that lies inside a band of the
// weight pattern. Each connection carries all of the CPPN's outputs for that
// point. It returns the updated connections.
func PruningAndExtraction(fixed Point, connections []*SpacialConnection, outgoing bool, cppn ConnectionQuery, p *QuadPoint, varThreshold, bandThreshold float64) []*SpacialConnection {
	if p.isLeaf() {
		return connections
	}
	for _, c := range p.children {
		if Variance(c) >= varThreshold {
			connections = PruningAndExtraction(fixed, connections, outgoing, cppn, c, varThreshold, bandThreshold)
			continue
		}

		// Band test: how much the weight changes one step to each side.
		d := p.width
		dleft := math.Abs(c.w - sampleWeight(cppn, fixed, Point{c.X - d, c.Y}, outgoing))
		dright := math.Abs(c.w - sampleWeight(cppn, fixed, Point{c.X + d, c.Y}, outgoing))
		dtop := math.Abs(c.w - sampleWeight(cppn, fixed, Point{c.X, c.Y - d}, outgoing))
		dbottom := math.Abs(c.w - sampleWeight(cppn, fixed, Point{c.X, c.Y + d}, outgoing))
		if max(min(dtop, dbottom), min(dleft, dright)) <= bandThreshold {
			continue
		}

		var con *SpacialConnection
		if outgoing {
			con = NewSpacialConnection(fixed, c.Point, c.values)
		} else {
			con = NewSpacialConnection(c.Point, fixed, c.values)
		}
		if !containsConnection(connections, con) {
			connections = append(connections, con)
		}
	}
	return connections
}
