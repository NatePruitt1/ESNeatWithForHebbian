package esneatforhebbians

import "cmp"

// Argmax returns the index of the largest element of s (the first one, if
// several are equal), or -1 if s is empty.
func Argmax[T cmp.Ordered](s []T) int {
	if len(s) == 0 {
		return -1
	}

	maxIdx := 0
	for i := 1; i < len(s); i++ {
		if s[i] > s[maxIdx] {
			maxIdx = i
		}
	}
	return maxIdx
}
