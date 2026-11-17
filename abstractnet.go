package esneatforhebbians

type AbstractNode struct {
	Posx, Posy     float64
	ActivationFunc ActivationFuncID
	Values         []float64 // all node-CPPN outputs at (Posx, Posy)
}

type AbstractConnection struct {
	From, To int //index into nodes
	Weight   float64
	Values   []float64
	Forward  bool
}

type AbstractNetwork struct {
	Nodes       []AbstractNode // First i are the input nodes, then o output nodes, and then all of the hidden nodes sorted by increasing y (then x).
	Connections []AbstractConnection
}
