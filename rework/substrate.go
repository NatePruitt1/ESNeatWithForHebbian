package rework

import (
	"cmp"
	"math"
	"slices"
)

type Point struct {
	X, Y float64
}

type substrateNode struct {
	Point
	values []float64
}

type substrateConnection struct {
	From, To  Point
	values    []float64
	recurrent bool
}

func (sC substrateConnection) GetWeight() float64 {
	if len(sC.values) < 1 {
		panic("Attempt to get weight from empty substrate connection")
	}

	return sC.values[0]
}

type substrate struct {
	inputNodes, outputNodes, hiddenNodes []substrateNode
	conns                                []substrateConnection
	connMap                              map[[2]Point]substrateConnection
}

type substrateGenConfig struct {
	connectionGeneratorSpec                         NetworkSpec
	nodeGeneratorSpec                               NetworkSpec
	initialDepth, maxDepth, inputCount, outputCount int
	divThreshold, bandThreshold, varThreshold       float64
	recursionAllowed                                bool
}

// perform steps 1 and 2, and then for step 3, we prune all connections that do not end in moving from input to output.
// We then sort the nodes by their y connection, and then connections by incoming, so the order would be [inputs->hidden, hidden->hidden, hidden->output]
func generateSubstrate(connectionGenerator ArtificialNeuralNet, nodeGenerator ArtificialNeuralNet, cfg substrateGenConfig) (*substrate, error) {

	boundConnGenerator, boundNodeGenerator, err := prepareForGeneration(connectionGenerator, nodeGenerator, cfg)
	if err != nil {
		return nil, err
	}

	inputNodes := buildRow(cfg.inputCount, -1, *boundNodeGenerator)
	outputNodes := buildRow(cfg.outputCount, 1, *boundNodeGenerator)

	connections := make(map[[2]Point]substrateConnection, 0)

	hiddenNodesList := make([]substrateNode, 0)
	hiddenNodesMap := make(map[Point]bool, 0)

	for i := range inputNodes {
		root := divisionAndInitialization(inputNodes[i].Point, true, *boundConnGenerator, cfg)

		newConnections := pruningAndExtraction(inputNodes[i].Point, make(map[[2]Point]substrateConnection, 0), boundConnGenerator, root, true, cfg)
		hiddenNodesList = discoverConnections(hiddenNodesList, hiddenNodesMap, newConnections, *boundNodeGenerator, true)

		unionConnections(connections, newConnections)
	}

	unexploredHiddens := slices.Clone(hiddenNodesList)
	exploredHiddens := make([]substrateNode, 0)
	for len(unexploredHiddens) > 0 {
		// Branch out from this hidden node and add the found nodes and connections into the fold.
		toExplore := unexploredHiddens[0]
		unexploredHiddens = unexploredHiddens[1:]
		exploredHiddens = append(exploredHiddens, toExplore)

		root := divisionAndInitialization(toExplore.Point, true, *boundConnGenerator, cfg)
		newConnections := pruningAndExtraction(toExplore.Point, make(map[[2]Point]substrateConnection), boundConnGenerator, root, true, cfg)
		unexploredHiddens = discoverConnections(unexploredHiddens, hiddenNodesMap, newConnections, *boundNodeGenerator, true)
		unionConnections(connections, newConnections)
	}

	hiddenNodesList = exploredHiddens

	for i := range outputNodes {
		root := divisionAndInitialization(outputNodes[i].Point, false, *boundConnGenerator, cfg)

		newConnections := pruningAndExtraction(outputNodes[i].Point, make(map[[2]Point]substrateConnection, 0), boundConnGenerator, root, false, cfg)

		// Avoid new useless nodes being made in this finishing step.
		for k, c := range newConnections {
			if hiddenNodesMap[c.From] {
				connections[k] = c
			}
		}
	}

	return pruneGeneration(inputNodes, outputNodes, hiddenNodesList, connections, cfg), nil
}

func prepareForGeneration(connectionGenerator ArtificialNeuralNet, nodeGenerator ArtificialNeuralNet, cfg substrateGenConfig) (*BoundConnectionGeneratingCPPN, *BoundNodeGeneratingCPPN, error) {
	// We need to build the input and output nodes. This is to support the loops for part 1 and 2 of the algorithm 3.
	if err := connectionGenerator.Validate(cfg.connectionGeneratorSpec); err != nil {
		return nil, nil, err
	}

	if err := nodeGenerator.Validate(cfg.nodeGeneratorSpec); err != nil {
		return nil, nil, err
	}

	boundConnGenerator, err := BindANNForConnectionGeneration(connectionGenerator, cfg.connectionGeneratorSpec)
	if err != nil {
		return nil, nil, err
	}

	boundNodeGenerator, err := BindANNForNodeGeneration(nodeGenerator, cfg.nodeGeneratorSpec)
	if err != nil {
		return nil, nil, err
	}

	return boundConnGenerator, boundNodeGenerator, nil
}

func pruneGeneration(inputNodes []substrateNode, outputNodes []substrateNode, hiddenNodes []substrateNode, connections map[[2]Point]substrateConnection, cfg substrateGenConfig) *substrate {
	connSlice := connMapToSlice(connections)

	substrate := &substrate{
		inputNodes:  inputNodes,
		outputNodes: outputNodes,
		hiddenNodes: hiddenNodes,
		conns:       connSlice,
		connMap:     connections,
	}

	sortNodes(substrate)
	prune(substrate)
	selectRecursive(substrate, cfg)

	return substrate
}

func sortNodes(substrate *substrate) {
	slices.SortFunc(substrate.hiddenNodes, func(a, b substrateNode) int {
		if c := cmp.Compare(a.Y, b.Y); c != 0 {
			return c
		} else {
			return cmp.Compare(a.X, b.X)
		}
	})
}

func selectRecursive(substrate *substrate, cfg substrateGenConfig) {
	if !cfg.recursionAllowed {
		substrate.conns = slices.DeleteFunc(substrate.conns, func(elem substrateConnection) bool {
			return elem.recurrent
		})
		substrate.connMap = connSliceToMap(substrate.conns)

		prune(substrate)
	}
}

func prune(sub *substrate) {
	fromInputs := reachable(sub.inputNodes, sub.connMap, false)
	toOutputs := reachable(sub.outputNodes, sub.connMap, true)

	keptNodes := make([]substrateNode, 0)
	for _, node := range sub.hiddenNodes {
		_, inInputs := fromInputs[node.Point]
		_, inOutputs := toOutputs[node.Point]
		if inInputs && inOutputs {
			keptNodes = append(keptNodes, node)
		}
	}

	// Mark all connections as recurrent or not. We do this by ensuring that the y value of the to and strictly greater than from.
	// It is impossible to generate recurrent connections sourced at input and output, we generate input outgoing in places impossible
	// to reach for the quad tree, and ones with output as the source would also be unreachable.
	keptConnections := make([]substrateConnection, 0)
	for _, conn := range sub.connMap {
		_, iok := fromInputs[conn.From]
		_, ook := toOutputs[conn.To]
		if iok && ook {
			conn.recurrent = conn.From.Y >= conn.To.Y
			keptConnections = append(keptConnections, conn)
		}
	}

	connMap := make(map[[2]Point]substrateConnection)
	for _, conn := range keptConnections {
		connMap[[2]Point{conn.From, conn.To}] = conn
	}

	sub.hiddenNodes = keptNodes
	sub.connMap = connMap
	sub.conns = keptConnections
}

func connSliceToMap(connections []substrateConnection) map[[2]Point]substrateConnection {
	connMap := make(map[[2]Point]substrateConnection)
	for _, conn := range connections {
		connMap[[2]Point{conn.From, conn.To}] = conn
	}

	return connMap
}

func connMapToSlice(connections map[[2]Point]substrateConnection) []substrateConnection {
	connSlice := make([]substrateConnection, len(connections))
	i := 0
	for _, v := range connections {
		connSlice[i] = v
		i += 1
	}
	return connSlice
}

func reachable(starts []substrateNode, connections map[[2]Point]substrateConnection, reverse bool) map[Point]bool {
	adj := make(map[Point][]Point)
	for k := range connections {
		from, to := k[0], k[1]
		if reverse {
			from, to = to, from
		}
		adj[from] = append(adj[from], to)
	}

	seen := make(map[Point]bool)
	var frontier []Point
	for _, s := range starts {
		if !seen[s.Point] {
			seen[s.Point] = true
			frontier = append(frontier, s.Point)
		}
	}

	for len(frontier) > 0 {
		cur := frontier[0]
		frontier = frontier[1:]
		for _, next := range adj[cur] {
			if !seen[next] {
				seen[next] = true
				frontier = append(frontier, next)
			}
		}
	}
	return seen
}

func discoverConnections(nodeList []substrateNode, nodeMap map[Point]bool, connections map[[2]Point]substrateConnection, boundNodeGenerator BoundNodeGeneratingCPPN, outgoing bool) []substrateNode {
	for c := range connections {
		// all of the "from" nodes here are input nodes.
		nodePos := connections[c].To
		if !outgoing {
			nodePos = connections[c].From
		}

		if _, ok := nodeMap[nodePos]; !ok {
			_, vals := boundNodeGenerator.Sample(nodePos)
			nodeList = append(nodeList, substrateNode{
				Point:  nodePos,
				values: vals,
			})

			nodeMap[nodePos] = true
		}
	}

	return nodeList
}

func unionConnections(a, b map[[2]Point]substrateConnection) {
	for bK := range b {
		a[bK] = b[bK]
	}
}

func buildRow(n int, y float64, nodeGenerator BoundNodeGeneratingCPPN) []substrateNode {
	spacing := 2.0 / float64(n+1)
	points := make([]Point, n)

	for x := range points {
		points[x] = Point{-1 + spacing*float64(x+1), y}
	}

	nodes := make([]substrateNode, n)
	for p := range points {
		_, vals := nodeGenerator.Sample(points[p])
		nodes[p] = substrateNode{
			Point:  points[p],
			values: vals,
		}
	}

	return nodes
}

func divisionAndInitialization(fixed Point, outgoing bool, connectionGenerator BoundConnectionGeneratingCPPN, cfg substrateGenConfig) *quadPoint {
	root := newQuadPoint(Point{0, 0}, 1, 1)
	queue := []*quadPoint{root}
	for len(queue) > 0 {
		p := queue[0]
		queue = queue[1:]

		r := p.width / 2
		next := p.level + 1
		p.children[0] = newQuadPoint(Point{p.X - r, p.Y - r}, r, next)
		p.children[1] = newQuadPoint(Point{p.X - r, p.Y + r}, r, next)
		p.children[2] = newQuadPoint(Point{p.X + r, p.Y - r}, r, next)
		p.children[3] = newQuadPoint(Point{p.X + r, p.Y + r}, r, next)
		for _, c := range p.children {
			if outgoing {
				c.w, c.values = connectionGenerator.Sample(fixed, c.Point)
			} else {
				c.w, c.values = connectionGenerator.Sample(c.Point, fixed)
			}
		}

		if p.level < cfg.initialDepth || (p.level < cfg.maxDepth && p.variance() > cfg.divThreshold) {
			queue = append(queue, p.children[:]...)
		}
	}
	return root
}

func pruningAndExtraction(fixed Point, connections map[[2]Point]substrateConnection, connGenerator *BoundConnectionGeneratingCPPN, p *quadPoint, outgoing bool, cfg substrateGenConfig) map[[2]Point]substrateConnection {
	// If division hit a maxDepth wall or other configuration stops, this protects pruning logic.
	if p.isLeaf() {
		return connections
	}

	for c := range p.children {
		child := p.children[c]
		if child.variance() >= cfg.varThreshold {
			pruningAndExtraction(fixed, connections, connGenerator, child, outgoing, cfg)
		} else {
			var dleftw, drightw, dtopw, dbottomw float64
			if outgoing {
				dleftw, _ = connGenerator.Sample(fixed, Point{child.X - p.width, child.Y})
				drightw, _ = connGenerator.Sample(fixed, Point{child.X + p.width, child.Y})
				dtopw, _ = connGenerator.Sample(fixed, Point{child.X, child.Y - p.width})
				dbottomw, _ = connGenerator.Sample(fixed, Point{child.X, child.Y + p.width})
			} else {
				dleftw, _ = connGenerator.Sample(Point{child.X - p.width, child.Y}, fixed)
				drightw, _ = connGenerator.Sample(Point{child.X + p.width, child.Y}, fixed)
				dtopw, _ = connGenerator.Sample(Point{child.X, child.Y - p.width}, fixed)
				dbottomw, _ = connGenerator.Sample(Point{child.X, child.Y + p.width}, fixed)
			}

			dleft := math.Abs(child.w - dleftw)
			dright := math.Abs(child.w - drightw)
			dtop := math.Abs(child.w - dtopw)
			dbottom := math.Abs(child.w - dbottomw)

			if max(min(dtop, dbottom), min(dright, dleft)) > cfg.bandThreshold {
				var con substrateConnection
				if outgoing {
					con = substrateConnection{
						From:   fixed,
						To:     child.Point,
						values: child.values,
					}
				} else {
					con = substrateConnection{
						From:   child.Point,
						To:     fixed,
						values: child.values,
					}
				}
				connections[[2]Point{con.From, con.To}] = con
			}
		}
	}
	return connections
}
