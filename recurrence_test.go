package esneatforhebbians

import "testing"

// onPath reports, for each node of an, whether it lies on a path from an input
// to an output, following only the given connections.
func onPath(an *AbstractNetwork, inputs, outputs int, keep func(AbstractConnection) bool) []bool {
	fwd := make([][]int, len(an.Nodes))
	bwd := make([][]int, len(an.Nodes))
	for _, c := range an.Connections {
		if keep(c) {
			fwd[c.From] = append(fwd[c.From], c.To)
			bwd[c.To] = append(bwd[c.To], c.From)
		}
	}
	reach := func(starts []int, graph [][]int) []bool {
		seen := make([]bool, len(an.Nodes))
		stack := append([]int(nil), starts...)
		for _, s := range starts {
			seen[s] = true
		}
		for len(stack) > 0 {
			n := stack[len(stack)-1]
			stack = stack[:len(stack)-1]
			for _, m := range graph[n] {
				if !seen[m] {
					seen[m] = true
					stack = append(stack, m)
				}
			}
		}
		return seen
	}
	var ins, outs []int
	for i := 0; i < inputs; i++ {
		ins = append(ins, i)
	}
	for o := 0; o < outputs; o++ {
		outs = append(outs, inputs+o)
	}
	a, b := reach(ins, fwd), reach(outs, bwd)
	both := make([]bool, len(an.Nodes))
	for i := range both {
		both[i] = a[i] && b[i]
	}
	return both
}

func TestGenerateWithoutRecurrence(t *testing.T) {
	const inputs, outputs = 3, 2
	for _, level := range []int{1, -1} {
		rec, err := GenerateAbstractFromCPPNs(waveCPPN{}, &scoredNodeCPPN{}, 1, 0, true, inputs, outputs, 3, 5, level, 0, 0.03, 0.3, 0.03)
		if err != nil {
			t.Fatal(err)
		}
		ff, err := GenerateAbstractFromCPPNs(waveCPPN{}, &scoredNodeCPPN{}, 1, 0, false, inputs, outputs, 3, 5, level, 0, 0.03, 0.3, 0.03)
		if err != nil {
			t.Fatal(err)
		}

		// Every connection is forward, under both the flag and the y rule.
		for i, c := range ff.Connections {
			if !c.Forward || ff.Nodes[c.To].Posy <= ff.Nodes[c.From].Posy {
				t.Fatalf("iterationLevel=%d: connection %d is recurrent", level, i)
			}
		}
		// Every hidden node is on a forward-only input-to-output path, so
		// pruning removed the nodes recurrent connections had joined up.
		for i, ok := range onPath(ff, inputs, outputs, func(AbstractConnection) bool { return true }) {
			if i >= inputs+outputs && !ok {
				t.Fatalf("iterationLevel=%d: hidden node %d is not on a forward path", level, i)
			}
		}
		// Converting without recurrence drops nothing, and the result is a
		// valid feed-forward Network.
		net, err := ff.ConvertToNetwork(inputs, outputs, false)
		if err != nil {
			t.Fatal(err)
		}
		if len(net.Connections) != len(ff.Connections) {
			t.Fatalf("conversion dropped %d connections", len(ff.Connections)-len(net.Connections))
		}
		net.Compute([]float64{1, 0, -1})

		// Compare with simply dropping recurrent connections from the
		// recurrent network: those nodes would be stranded.
		stranded := 0
		for i, ok := range onPath(rec, inputs, outputs, func(c AbstractConnection) bool { return c.Forward }) {
			if i >= inputs+outputs && !ok {
				stranded++
			}
		}
		t.Logf("iterationLevel=%d: with recurrence %d nodes / %d connections; without %d nodes / %d connections; dropping recurrent connections afterwards would strand %d hidden nodes",
			level, len(rec.Nodes), len(rec.Connections), len(ff.Nodes), len(ff.Connections), stranded)
	}
}

func TestForwardOnly(t *testing.T) {
	v := []float64{1}
	conns := []*SpacialConnection{
		NewSpacialConnection(0, -0.5, 0, 0.5, v),   // forward
		NewSpacialConnection(0, 0.5, 0, -0.5, v),   // backward
		NewSpacialConnection(-0.5, 0, 0.5, 0, v),   // lateral
		NewSpacialConnection(0.25, 0, 0.25, 0, v),  // self
		NewSpacialConnection(0, -1, 0.75, 0.25, v), // forward from an input row
	}
	got := forwardOnly(conns)
	if len(got) != 2 || got[0] != conns[0] || got[1] != conns[4] {
		t.Fatalf("forwardOnly kept %d connections, want the 2 forward ones in order", len(got))
	}
	if len(conns) != 5 || conns[1].fromy != 0.5 {
		t.Fatal("forwardOnly must not modify its input")
	}
}
