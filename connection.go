package esneatforhebbians

// SpacialConnection is a connection found by the search, between two points
// on the substrate. values holds every CPPN output for the connection, and
// weight is a copy of values[WeightIndex].
type SpacialConnection struct {
	From, To Point
	weight   float64
	values   []float64
}

// NewSpacialConnection creates a connection that keeps values as given, without
// copying it; values must have at least one element.
func NewSpacialConnection(from, to Point, values []float64) *SpacialConnection {
	return &SpacialConnection{
		From:   from,
		To:     to,
		weight: values[WeightIndex],
		values: values,
	}
}

// isForward reports whether c is a forward connection: it goes to a strictly
// greater y. Lateral (equal y), backward and self-connections are recurrent.
func (c *SpacialConnection) isForward() bool {
	return c.To.Y > c.From.Y
}

// forwardOnly returns the forward connections in connections, in order.
func forwardOnly(connections []*SpacialConnection) []*SpacialConnection {
	var kept []*SpacialConnection
	for _, c := range connections {
		if c.isForward() {
			kept = append(kept, c)
		}
	}
	return kept
}

// containsConnection reports whether connections already holds a connection
// with the same endpoints as con.
func containsConnection(connections []*SpacialConnection, con *SpacialConnection) bool {
	for _, c := range connections {
		if c.From == con.From && c.To == con.To {
			return true
		}
	}
	return false
}
