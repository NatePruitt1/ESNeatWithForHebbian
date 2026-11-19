package rework

import "errors"

const WEIGHT_INDEX = 0

type BoundConnectionGeneratingCPPN struct {
	ArtificialNeuralNet
}

func BindANNForConnectionGeneration(ann ArtificialNeuralNet, providingSpec NetworkSpec) (*BoundConnectionGeneratingCPPN, error) {
	if providingSpec.Inputs < 4 {
		return nil, errors.New("A connection generating cppn spec must have atleast 4 inputs allowed.")
	}

	if providingSpec.Outputs < 1 {
		return nil, errors.New("A connection generating cppn spec must have atleast 1 output allowed.")
	}

	if providingSpec.RecursiveAllowed {
		return nil, errors.New("CPPNs may not be recurrent.")
	}

	if err := ann.Validate(providingSpec); err != nil {
		return nil, err
	}

	return &BoundConnectionGeneratingCPPN{
		ArtificialNeuralNet: ann,
	}, nil
}

func (ann *BoundConnectionGeneratingCPPN) Sample(source, sink Point) (float64, []float64) {
	output := ann.Forward([]float64{source.X, source.Y, sink.X, sink.Y})
	return output[WEIGHT_INDEX], output
}

type BoundNodeGeneratingCPPN struct {
	ArtificialNeuralNet
}

func BindANNForNodeGeneration(ann ArtificialNeuralNet, providingSpec NetworkSpec) (*BoundNodeGeneratingCPPN, error) {
	if providingSpec.Inputs < 2 {
		return nil, errors.New("A node generating cppn spec must have atleast 2 inputs allowed.")
	}

	if providingSpec.Outputs < ActivationFunctionCount {
		return nil, errors.New("A node generating cppn spec must have atleast as many outputs as activation functions.")
	}

	if providingSpec.RecursiveAllowed {
		return nil, errors.New("CPPNs may not be recurrent.")
	}

	if err := ann.Validate(providingSpec); err != nil {
		return nil, err
	}

	return &BoundNodeGeneratingCPPN{
		ArtificialNeuralNet: ann,
	}, nil
}

func (ann *BoundNodeGeneratingCPPN) Sample(source Point) (ActivationFunc, []float64) {
	output := ann.Forward([]float64{source.X, source.Y})
	return ActivationFunc(Argmax(output[:ActivationFunctionCount])), output
}
