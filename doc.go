// Package esneatforhebbians builds neural networks with ES-HyperNEAT and runs
// them either with fixed weights or with Hebbian plasticity.
//
// # Pipeline
//
//	CPPN + NodeCPPN
//	   │  GenerateAbstractFromCPPNs (ES-HyperNEAT, Algorithms 1-3)
//	   ▼
//	AbstractNetwork        positions, activation functions, all CPPN outputs
//	   │  ConvertToNetwork / ConvertToHebbian
//	   ▼
//	Network / HebbianNetwork   ready to Compute
//
// # Files, in pipeline order
//
//	point.go           Point, the position type used everywhere on the substrate
//	activations.go     activation function IDs and Activate
//	cppn.go            the CPPN interfaces and the output index constants
//	quadtree.go        QuadPoint and Variance
//	eshyperneat.go     Algorithm 1 (division) and Algorithm 2 (extraction)
//	connection.go      SpacialConnection, a connection found by the search
//	substrate.go       Algorithm 3 steps 1-3: inputs, hidden discovery, outputs
//	prune.go           Algorithm 3 step 4: remove nodes off every input-output path
//	generate.go        GenerateAbstractFromCPPNs: runs the steps above in order
//	abstractnet.go     AbstractNetwork, the hand-off between search and runtime
//	layout.go          the layout rules every network must follow
//	network.go         Network: the one evaluation engine
//	hebbian_network.go HebbianNetwork: a Network plus Hebbian plasticity
//	network_cppn.go    adapters that use a Network as a CPPN
//	utils.go           small helpers (Argmax)
//
// # Layout rules
//
// Every network lists its nodes as [inputs..., outputs..., hidden...], with
// hidden nodes sorted by increasing y (then x). A connection is forward when it
// goes to a strictly greater y; forward connections read the current step's
// values, all others (lateral, backward, self) read the previous step's. Each
// step evaluates inputs, then hidden nodes in order, then outputs. See
// layout.go for the exact rules.
package esneatforhebbians
