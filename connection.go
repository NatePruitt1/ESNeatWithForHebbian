package esneatforhebbians

// SpacialConnection is a connection between two points on the substrate.
// values holds every CPPN output for the connection, and weight is a copy of
// values[WeightIndex].
type SpacialConnection struct {
	tox, fromx float64
	toy, fromy float64
	weight     float64
	values     []float64
}

// NewSpacialConnection creates a connection that keeps values as given, without
// copying it; values must have at least one element.
func NewSpacialConnection(fromx, fromy, tox, toy float64, values []float64) *SpacialConnection {
	return &SpacialConnection{
		tox:    tox,
		fromx:  fromx,
		toy:    toy,
		fromy:  fromy,
		weight: values[WeightIndex],
		values: values,
	}
}

// isForward reports whether c is a forward connection under the forward rule:
// it goes to a strictly greater y. Lateral (equal y), backward and
// self-connections are recurrent.
func (c *SpacialConnection) isForward() bool {
	return c.toy > c.fromy
}

// forwardOnly returns the forward connections in connections, in order.
func forwardOnly(connections []*SpacialConnection) []*SpacialConnection {
	kept := connections[:0:0]
	for _, c := range connections {
		if c.isForward() {
			kept = append(kept, c)
		}
	}
	return kept
}

// sameEndpoints reports whether a and b connect the same two points.
func sameEndpoints(a, b *SpacialConnection) bool {
	return a.fromx == b.fromx && a.fromy == b.fromy && a.tox == b.tox && a.toy == b.toy
}

// containsConnection reports whether connections already holds a connection
// with the same endpoints as con.
func containsConnection(connections []*SpacialConnection, con *SpacialConnection) bool {
	for _, c := range connections {
		if sameEndpoints(c, con) {
			return true
		}
	}
	return false
}
