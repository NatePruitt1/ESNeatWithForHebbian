package esneatforhebbians

import (
	"fmt"
	"math"
)

type ActivationFuncID int

const (
	Linear   ActivationFuncID = 0
	Sigmoid  ActivationFuncID = 1
	Tanh     ActivationFuncID = 2
	Step     ActivationFuncID = 3
	Gaussion ActivationFuncID = 4

	// The functions below are for CPPNs. Substrate nodes cannot choose them
	// (see ActivationFuncCount), but networks built directly, such as dense
	// CPPN networks, can use them.
	Sine ActivationFuncID = 5
	Abs  ActivationFuncID = 6
	Sign ActivationFuncID = 7
)

// ActivationFuncCount is the number of activation functions a substrate node
// can choose from: IDs 0 through ActivationFuncCount-1. Node CPPNs score
// exactly these, so adding CPPN-only functions does not change how many node
// CPPN outputs are needed.
const ActivationFuncCount = 5

// AllActivationFuncCount is the number of activation function IDs Activate
// accepts, including the CPPN-only ones.
const AllActivationFuncCount = 8

// Valid reports whether id is an activation function Activate accepts.
func (id ActivationFuncID) Valid() bool {
	return id >= 0 && id < AllActivationFuncCount
}

// ActivationChannel is the first node-CPPN output used to choose a node's
// activation function. Outputs ActivationChannel through
// ActivationChannel+ActivationFuncCount-1 each score one function, in ID order;
// the highest score wins. Other outputs are free for other node values.
const ActivationChannel = 0

// chooseActivation picks the activation function whose channel in values has
// the highest score. values must have at least
// ActivationChannel+ActivationFuncCount elements.
func chooseActivation(values []float64) ActivationFuncID {
	return ActivationFuncID(Argmax(values[ActivationChannel : ActivationChannel+ActivationFuncCount]))
}

// Activate applies the activation function to x:
//
//	Linear    x
//	Sigmoid   1 / (1 + e^-x)       range (0, 1), 0.5 at x = 0
//	Tanh      tanh(x)              range (-1, 1)
//	Step      1 if x > 0, else 0
//	Gaussion  e^(-x^2)             range (0, 1], peak 1 at x = 0
//	Sine      sin(x)               range [-1, 1]
//	Abs       |x|                  range [0, inf)
//	Sign      -1 if x < 0, else +1 exactly -1 or +1 (for Dir outputs)
//
// It panics for an ID that is not one of the constants above, since that means
// a node was built with an invalid activation function.
func (id ActivationFuncID) Activate(x float64) float64 {
	switch id {
	case Linear:
		return x
	case Sigmoid:
		return 1 / (1 + math.Exp(-x))
	case Tanh:
		return math.Tanh(x)
	case Step:
		if x > 0 {
			return 1
		}
		return 0
	case Gaussion:
		return math.Exp(-x * x)
	case Sine:
		return math.Sin(x)
	case Abs:
		return math.Abs(x)
	case Sign:
		if x < 0 {
			return -1
		}
		return 1
	}
	panic(fmt.Sprintf("unknown activation function ID %d", int(id)))
}
