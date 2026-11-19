package rework

type PlasticityParameters struct {
	A, B, C, D, M, Dir, LearningRate float64
}

type HebbianNet struct {
	ArtificialNeuralNet
	Plasticity []PlasticityParameters
}

func (hn *HebbianNet) Update() {
	for c := range hn.Connections {
		hn.UpdateConn(c, 1.0)
	}
}

func (hn *HebbianNet) UpdateWithFitness(fitness float64) {
	for c := range hn.Connections {
		hn.UpdateConn(c, fitness)
	}
}

func (hn *HebbianNet) UpdateConn(conn int, fitness float64) {
	connection := &hn.Connections[conn]
	params := &hn.Plasticity[conn]
	postSynaptic := hn.Nodes[connection.To].Current
	preSynaptic := hn.GetPreSynaptic(conn)

	delM := params.LearningRate * fitness * (params.A*postSynaptic*preSynaptic +
		params.B*postSynaptic +
		params.C*preSynaptic +
		params.D)

	params.M = max(0, params.M+delM)
	connection.Weight = params.M * params.Dir
}

func (hn *HebbianNet) GetPreSynaptic(c int) float64 {
	from := hn.Nodes[hn.Connections[c].From]
	if hn.Connections[c].Forward {
		return from.Current
	} else {
		return from.Previous
	}
}
