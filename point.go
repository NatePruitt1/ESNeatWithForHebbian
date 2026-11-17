package esneatforhebbians

import "cmp"

// Point is a position on the substrate, which spans [-1, 1] x [-1, 1]. It is
// comparable, so it can be used as a map key. Exact comparison is safe because
// substrate positions are computed the same way every time (quadtree centers
// are exact binary fractions).
type Point struct {
	X, Y float64
}

// comparePoints orders points by Y, then by X. Sorting hidden nodes this way
// gives the evaluation order the layout rules need.
func comparePoints(a, b Point) int {
	if c := cmp.Compare(a.Y, b.Y); c != 0 {
		return c
	}
	return cmp.Compare(a.X, b.X)
}
