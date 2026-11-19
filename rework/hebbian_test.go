// AI-generated: written by Claude (Claude Code, claude-opus-5-5) as a
// regression suite for the rework package. Review before relying on it.

package rework

import "testing"

// hebbPair builds a two-node Hebbian net: node 0 (source) -> node 1 (target).
func hebbPair(src, srcPrev, dst float64, forward bool, p PlasticityParameters) *HebbianNet {
	return &HebbianNet{
		ArtificialNeuralNet: ArtificialNeuralNet{
			Nodes:       []Node{{Current: src, Previous: srcPrev}, {Current: dst}},
			Connections: []Connection{{From: 0, To: 1, Forward: forward}},
		},
		Plasticity: []PlasticityParameters{p},
	}
}

func TestHebbianTerms(t *testing.T) {
	// source = 2, target = 3
	cases := []struct {
		name string
		p    PlasticityParameters
		want float64
	}{
		{"A: pre*post", PlasticityParameters{A: 1, Dir: 1, LearningRate: 1}, 6},
		{"B: post (target)", PlasticityParameters{B: 1, Dir: 1, LearningRate: 1}, 3},
		{"C: pre (source)", PlasticityParameters{C: 1, Dir: 1, LearningRate: 1}, 2},
		{"D: constant", PlasticityParameters{D: 1, Dir: 1, LearningRate: 1}, 1},
		{"learning rate scales", PlasticityParameters{D: 1, Dir: 1, LearningRate: 0.5}, 0.5},
		{"starts from M", PlasticityParameters{D: 1, M: 4, Dir: 1, LearningRate: 1}, 5},
	}
	for _, c := range cases {
		hn := hebbPair(2, 0, 3, true, c.p)
		hn.Update()
		if m := hn.Plasticity[0].M; !approxEq(m, c.want) {
			t.Errorf("%s: M = %v, want %v", c.name, m, c.want)
		}
		if w := hn.Connections[0].Weight; !approxEq(w, c.want) {
			t.Errorf("%s: weight = %v, want %v", c.name, w, c.want)
		}
	}
}

func TestHebbianRecurrentUsesPreviousSource(t *testing.T) {
	hn := hebbPair(2, 5, 3, false, PlasticityParameters{C: 1, Dir: 1, LearningRate: 1})
	hn.Update()
	if m := hn.Plasticity[0].M; m != 5 {
		t.Errorf("M = %v, want 5 (source's previous value)", m)
	}
}

func TestHebbianMClampsAtZero(t *testing.T) {
	hn := hebbPair(2, 0, 3, true, PlasticityParameters{D: -10, M: 1, Dir: 1, LearningRate: 1})
	hn.Update()
	if hn.Plasticity[0].M != 0 || hn.Connections[0].Weight != 0 {
		t.Errorf("M = %v weight = %v, want both 0", hn.Plasticity[0].M, hn.Connections[0].Weight)
	}
}

func TestHebbianDirSetsSign(t *testing.T) {
	hn := hebbPair(2, 0, 3, true, PlasticityParameters{D: 1, M: 1, Dir: -1, LearningRate: 1})
	hn.Update()
	if hn.Plasticity[0].M != 2 || hn.Connections[0].Weight != -2 {
		t.Errorf("M = %v weight = %v, want M=2 weight=-2", hn.Plasticity[0].M, hn.Connections[0].Weight)
	}
}

func TestHebbianNegativeFitnessIsAntiHebbian(t *testing.T) {
	hn := hebbPair(2, 0, 3, true, PlasticityParameters{B: 1, M: 5, Dir: 1, LearningRate: 1})
	hn.UpdateWithFitness(-1)
	if m := hn.Plasticity[0].M; m != 2 {
		t.Errorf("M = %v, want 5 - 3 = 2", m)
	}
}

func TestHebbianUpdateIsFitnessOne(t *testing.T) {
	p := PlasticityParameters{A: 0.3, B: -0.2, C: 0.1, D: 0.05, M: 1, Dir: 1, LearningRate: 0.7}
	a, b := hebbPair(2, 1, 3, true, p), hebbPair(2, 1, 3, true, p)
	a.Update()
	b.UpdateWithFitness(1)
	if a.Plasticity[0].M != b.Plasticity[0].M {
		t.Errorf("Update gave %v, UpdateWithFitness(1) gave %v", a.Plasticity[0].M, b.Plasticity[0].M)
	}
}

func TestHebbianUpdatesEveryConnection(t *testing.T) {
	hn := &HebbianNet{
		ArtificialNeuralNet: ArtificialNeuralNet{
			Nodes: []Node{{Current: 1}, {Current: 1}, {Current: 1}},
			Connections: []Connection{
				{From: 0, To: 1, Forward: true},
				{From: 1, To: 2, Forward: true},
			},
		},
		Plasticity: []PlasticityParameters{
			{D: 1, Dir: 1, LearningRate: 1},
			{D: 2, Dir: 1, LearningRate: 1},
		},
	}
	hn.Update()
	if hn.Connections[0].Weight != 1 || hn.Connections[1].Weight != 2 {
		t.Errorf("weights %v %v, want 1 2", hn.Connections[0].Weight, hn.Connections[1].Weight)
	}
}
