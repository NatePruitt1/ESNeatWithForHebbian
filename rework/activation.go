package rework

import (
	"fmt"
	"math"
)

type ActivationFunc int

const (
	Linear   ActivationFunc = 0
	Sigmoid  ActivationFunc = 1
	Tanh     ActivationFunc = 2
	Gaussian ActivationFunc = 3
	Sine     ActivationFunc = 4
	Abs      ActivationFunc = 5
	Step     ActivationFunc = 6
	Sign     ActivationFunc = 7
)

const ActivationFunctionCount = 8

func (act ActivationFunc) Valid() bool {
	return act >= 0 && act < ActivationFunctionCount
}

func (act ActivationFunc) Compute(x float64) float64 {
	switch act {
	case Linear:
		return x
	case Sigmoid:
		return 1 / (1 + math.Exp(-x))
	case Step:
		if x > 0 {
			return 1
		} else {
			return 0
		}
	case Gaussian:
		return math.Exp(-x * x)
	case Sine:
		return math.Sin(x)
	case Abs:
		return math.Abs(x)
	case Sign:
		if x < 0 {
			return -1
		} else {
			return 1
		}
	case Tanh:
		return math.Tanh(x)
	}

	panic(fmt.Sprintf("unknown activation function %d", int(act)))
}
