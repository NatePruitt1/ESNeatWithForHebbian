package esneatforhebbians

import (
	"math"
	"strings"
	"testing"
)

// hebbianValues returns a valid connection-CPPN output slice with the given
// weight and direction, laid out by the indices in cppn.go.
func hebbianValues(weight, dir float64) []float64 {
	v := make([]float64, HebbianValueCount)
	v[WeightIndex] = weight
	v[AIndex], v[BIndex], v[CIndex], v[DIndex] = 0.1, 0.2, 0.3, 0.4
	v[EtaIndex] = 0.5
	v[DirIndex] = dir
	return v
}

// smallAbstract is input 0 -> hidden 2 -> output 1, with the given values on
// the second connection.
func smallAbstract(values []float64) *AbstractNetwork {
	return &AbstractNetwork{
		Nodes: []AbstractNode{
			{Posx: 0, Posy: InputY, ActivationFunc: Linear},
			{Posx: 0, Posy: OutputY, ActivationFunc: Tanh},
			{Posx: 0, Posy: 0, ActivationFunc: Sigmoid},
		},
		Connections: []AbstractConnection{
			{From: 0, To: 2, Weight: 0.8, Values: hebbianValues(0.8, 1), Forward: true},
			{From: 2, To: 1, Weight: values[WeightIndex], Values: values, Forward: true},
		},
	}
}

func TestConvertToHebbian(t *testing.T) {
	hn, err := smallAbstract(hebbianValues(-0.6, -1)).ConvertToHebbian(1, 1)
	if err != nil {
		t.Fatal(err)
	}
	c := hn.Connections[1]
	if c.From != 2 || c.To != 1 {
		t.Fatalf("endpoints %d -> %d, want 2 -> 1", c.From, c.To)
	}
	// M_0 comes from |weight| and Dir from the direction output (the swap fix).
	if c.M != 0.6 || c.Dir != -1 || c.W != -0.6 {
		t.Fatalf("M=%g Dir=%g W=%g, want M=0.6 Dir=-1 W=-0.6", c.M, c.Dir, c.W)
	}
	if c.A != 0.1 || c.B != 0.2 || c.C != 0.3 || c.D != 0.4 || c.Eta != 0.5 || !c.Forward {
		t.Fatalf("coefficients not carried over: %+v", c)
	}
	if hn.Nodes[1].Activation != Tanh || hn.Nodes[2].Activation != Sigmoid {
		t.Fatal("activation functions not carried over")
	}
}

func TestConvertToHebbianRejectsBadValues(t *testing.T) {
	cases := []struct {
		name   string
		modify func(v []float64) []float64
		want   string
	}{
		{"too few values", func(v []float64) []float64 { return v[:EtaIndex] }, "needs"},
		{"Dir not a sign", func(v []float64) []float64 { v[DirIndex] = 0.5; return v }, "Dir"},
		{"Dir zero", func(v []float64) []float64 { v[DirIndex] = 0; return v }, "Dir"},
		{"negative Eta", func(v []float64) []float64 { v[EtaIndex] = -0.1; return v }, "Eta"},
		{"NaN coefficient", func(v []float64) []float64 { v[BIndex] = math.NaN(); return v }, "B"},
		{"infinite weight", func(v []float64) []float64 { v[WeightIndex] = math.Inf(1); return v }, "finite"},
	}
	for _, tc := range cases {
		values := tc.modify(hebbianValues(0.5, 1))
		_, err := smallAbstract(values).ConvertToHebbian(1, 1)
		if err == nil {
			t.Errorf("%s: expected an error", tc.name)
			continue
		}
		if !strings.Contains(err.Error(), tc.want) || !strings.Contains(err.Error(), "connection 1") {
			t.Errorf("%s: error %q should mention %q and connection 1", tc.name, err, tc.want)
		}
	}
}

func TestHebbianConnectionValidate(t *testing.T) {
	ok := NewHebbianConnection(1, 0, 0, 0, 0, 0, 0.1, 2, -1, true)
	if err := ok.Validate(); err != nil {
		t.Fatalf("valid connection rejected: %v", err)
	}
	negM := NewHebbianConnection(1, 0, 0, 0, 0, 0, 0.1, -2, 1, true)
	if err := negM.Validate(); err == nil || !strings.Contains(err.Error(), "M is") {
		t.Fatalf("negative M not caught: %v", err)
	}
	drifted := ok
	drifted.W = 5
	if err := drifted.Validate(); err == nil || !strings.Contains(err.Error(), "M * Dir") {
		t.Fatalf("W != M*Dir not caught: %v", err)
	}
}
