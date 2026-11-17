package esneatforhebbians

import (
	"fmt"
	"math"
)

// HebbianNetwork is a Network whose weights change with the ABCD Hebbian rule.
// It evaluates exactly like a Network (Compute, Reset and the layout rules are
// the embedded Network's) and only adds per-connection plasticity:
// Plasticity[i] belongs to Connections[i]. Recurrent connections are always
// allowed.
type HebbianNetwork struct {
	Network
	Plasticity []Plasticity
}

// Plasticity holds one connection's Hebbian parameters. The connection's
// weight is M * Dir: M is a non-negative magnitude that learning changes, and
// Dir (exactly -1 or +1) fixes the weight's sign.
//
//	ΔM = Eta * fitness * (A*pre*post + B*pre + C*post + D)
//
// Anti-Hebbian learning comes from negative fitness and Dir = -1, never from a
// negative Eta.
type Plasticity struct {
	A, B, C, D float64
	Eta        float64
	M          float64
	Dir        float64
}

// Weight returns the connection weight these parameters give.
func (p Plasticity) Weight() float64 {
	return p.M * p.Dir
}

// Validate checks the rules the connection CPPN is expected to enforce through
// its structure, so a CPPN that breaks them is caught when the network is
// built instead of producing a network that silently misbehaves:
//
//   - every value is finite (no NaN or ±Inf);
//   - M >= 0, since learning keeps M non-negative;
//   - Dir is exactly -1 or +1;
//   - Eta >= 0.
func (p Plasticity) Validate() error {
	for _, v := range []struct {
		name  string
		value float64
	}{{"A", p.A}, {"B", p.B}, {"C", p.C}, {"D", p.D}, {"Eta", p.Eta}, {"M", p.M}, {"Dir", p.Dir}} {
		if math.IsNaN(v.value) || math.IsInf(v.value, 0) {
			return fmt.Errorf("%s is %g; it must be finite", v.name, v.value)
		}
	}
	if p.M < 0 {
		return fmt.Errorf("M is %g; it must be >= 0", p.M)
	}
	if p.Dir != -1 && p.Dir != 1 {
		return fmt.Errorf("Dir is %g; it must be exactly -1 or +1", p.Dir)
	}
	if p.Eta < 0 {
		return fmt.Errorf("Eta is %g; it must be >= 0", p.Eta)
	}
	return nil
}

// NewHebbianNetwork builds a HebbianNetwork. plasticity must have one entry
// per connection, and each entry must pass Validate. Every connection's Weight
// is set from its plasticity (M * Dir), so the Weight fields passed in are
// ignored.
func NewHebbianNetwork(inputCount, outputCount int, nodes []Node, connections []Connection, plasticity []Plasticity) (*HebbianNetwork, error) {
	if len(plasticity) != len(connections) {
		return nil, fmt.Errorf("got %d plasticity entries for %d connections", len(plasticity), len(connections))
	}
	for i, p := range plasticity {
		if err := p.Validate(); err != nil {
			return nil, fmt.Errorf("connection %d (node %d -> %d): %w", i, connections[i].From, connections[i].To, err)
		}
		connections[i].Weight = p.Weight()
	}
	net, err := NewNetwork(inputCount, outputCount, nodes, connections, true)
	if err != nil {
		return nil, err
	}
	return &HebbianNetwork{Network: *net, Plasticity: plasticity}, nil
}

// ConvertToHebbian builds a HebbianNetwork, reading each connection's
// plasticity from its CPPN outputs (see the index constants in cppn.go): the
// starting magnitude M is |weight|, and Dir, A, B, C, D and Eta come from
// their own outputs.
func (an *AbstractNetwork) ConvertToHebbian() (*HebbianNetwork, error) {
	plasticity := make([]Plasticity, len(an.Connections))
	for i, c := range an.Connections {
		if len(c.Values) < HebbianValueCount {
			return nil, fmt.Errorf("connection %d has %d CPPN values; Hebbian conversion needs %d", i, len(c.Values), HebbianValueCount)
		}
		plasticity[i] = Plasticity{
			A:   c.Values[AIndex],
			B:   c.Values[BIndex],
			C:   c.Values[CIndex],
			D:   c.Values[DIndex],
			Eta: c.Values[EtaIndex],
			M:   math.Abs(c.Weight),
			Dir: c.Values[DirIndex],
		}
	}
	return NewHebbianNetwork(an.InputCount, an.OutputCount, an.nodes(), an.connections(), plasticity)
}

// UpdateWeights applies the Hebbian rule (see Plasticity) to every connection,
// using the node values from the most recent Compute call, then sets each
// connection's weight to M * Dir.
func (hn *HebbianNetwork) UpdateWeights(fitness float64) {
	for i := range hn.Connections {
		c := &hn.Connections[i]
		p := &hn.Plasticity[i]
		pre := hn.Nodes[c.From].Current
		post := hn.Nodes[c.To].Current

		dM := p.Eta * fitness * (p.A*pre*post + p.B*pre + p.C*post + p.D)
		p.M = max(0, p.M+dM)
		c.Weight = p.Weight()
	}
}
