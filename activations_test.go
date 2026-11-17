package esneatforhebbians

import (
	"math"
	"testing"
)

func TestActivate(t *testing.T) {
	cases := []struct {
		id   ActivationFuncID
		x    float64
		want float64
	}{
		{Linear, -2.5, -2.5},
		{Linear, 3, 3},
		{Sigmoid, 0, 0.5},
		{Sigmoid, 2, 1 / (1 + math.Exp(-2))},
		{Sigmoid, -1000, 0}, // very negative input must not produce NaN
		{Sigmoid, 1000, 1},
		{Tanh, 0, 0},
		{Tanh, 1, math.Tanh(1)},
		{Tanh, -1, -math.Tanh(1)},
		{Step, -0.1, 0},
		{Step, 0, 0},
		{Step, 0.1, 1},
		{Gaussion, 0, 1},
		{Gaussion, 1, math.Exp(-1)},
		{Gaussion, -1, math.Exp(-1)},
		{Sine, 0, 0},
		{Sine, math.Pi / 2, 1},
		{Sine, -1, math.Sin(-1)},
		{Abs, -2.5, 2.5},
		{Abs, 3, 3},
		{Sign, -0.1, -1},
		{Sign, 0, 1},
		{Sign, 7, 1},
	}
	for _, c := range cases {
		got := c.id.Activate(c.x)
		if math.Abs(got-c.want) > 1e-12 || math.IsNaN(got) {
			t.Errorf("ActivationFuncID(%d).Activate(%g) = %g, want %g", c.id, c.x, got, c.want)
		}
	}
}

func TestActivateUnknownPanics(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("expected a panic for an unknown activation function ID")
		}
	}()
	ActivationFuncID(AllActivationFuncCount).Activate(1)
}
