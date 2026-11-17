package esneatforhebbians

// AI-generated test code (see the summary at the bottom of this file for
// everything this test depends on).
//
// TestSolveXORWithGA evolves pairs of dense CPPNs (a connection CPPN and a
// node CPPN, each described by a Genome) with a simple genetic algorithm.
// Each pair is turned into a feed-forward network with ES-HyperNEAT, and the
// network is scored on XOR.
//
// Run it with logging:
//
//	go test -run TestSolveXORWithGA -v
//
// Override the main settings without editing the file:
//
//	go test -run TestSolveXORWithGA -v -args -xor.seed=3 -xor.pop=80 -xor.gens=300
//
// It is skipped under -short.

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"runtime"
	"slices"
	"sort"
	"strings"
	"sync"
	"testing"
	"time"
)

// ---------------------------------------------------------------------------
// Hyperparameters
// ---------------------------------------------------------------------------

// xorConfig holds every setting the experiment uses. defaultXORConfig gives
// the values the test runs with; the flags below override a few of them.
type xorConfig struct {
	// Genetic algorithm.
	Seed              int64
	PopulationSize    int
	Generations       int
	Elites            int     // best individuals copied unchanged into the next generation
	TournamentSize    int     // individuals compared when picking a parent
	CrossoverRate     float64 // chance a child is a crossover of two parents rather than a copy of one
	CrossoverKeep     float64 // chance each gene comes from the fitter parent during crossover
	MutationsPerChild int     // point mutations applied to each genome of every non-elite child
	ActMutationProb   float64 // chance a point mutation changes an activation rather than a weight
	WeightRange       float64 // weights are drawn uniformly from [-WeightRange, WeightRange]
	Workers           int     // parallel evaluations; results do not depend on this

	// Success: every case's output must be within SolveTolerance of its
	// target (0.5 would only require the right side of 0.5).
	SolveTolerance float64

	// CPPN shapes. Output activations are fixed; hidden activations evolve.
	ConnSpec DenseSpec
	NodeSpec DenseSpec

	// ES-HyperNEAT.
	CPPNBias          float64 // value of the CPPNs' bias input
	InitialDepth      int
	MaxDepth          int
	IterationLevel    int
	MaxHiddenNodes    int
	VarianceThreshold float64
	BandThreshold     float64
	DivisionThreshold float64

	// Logging.
	LogEvery int // log a summary line every LogEvery generations (and the last one)
}

func defaultXORConfig() xorConfig {
	hidden := []int{4, 4, 4, 4}
	return xorConfig{
		Seed:              1,
		PopulationSize:    100,
		Generations:       500,
		Elites:            4,
		TournamentSize:    3,
		CrossoverRate:     0.5,
		CrossoverKeep:     0.7,
		MutationsPerChild: 2,
		ActMutationProb:   0.2,
		WeightRange:       3.0,
		Workers:           runtime.NumCPU(),

		SolveTolerance: 0.3,

		// Connection CPPN: inputs x1, y1, x2, y2, bias, networkType; one
		// output, the weight. It is Linear so weights are not limited to
		// ±1; across seeds 1-5 this solved XOR more reliably than Tanh.
		ConnSpec: DenseSpec{
			Inputs: 6, Outputs: 1, Hidden: hidden,
			OutputActs: []ActivationFuncID{Linear},
			BiasInput:  4,
		},
		// Node CPPN: inputs x, y, bias, networkType; one linear score per
		// substrate activation function (the highest score is chosen).
		NodeSpec: DenseSpec{
			Inputs: 4, Outputs: ActivationFuncCount, Hidden: hidden,
			OutputActs: []ActivationFuncID{Linear, Linear, Linear, Linear, Linear},
			BiasInput:  2,
		},

		CPPNBias:          1,
		InitialDepth:      2,
		MaxDepth:          4,
		IterationLevel:    1,
		MaxHiddenNodes:    40,
		VarianceThreshold: 0.03,
		BandThreshold:     0.1,
		DivisionThreshold: 0.03,

		LogEvery: 5,
	}
}

var (
	xorSeed = flag.Int64("xor.seed", 0, "XOR GA: random seed (0 keeps the default)")
	xorPop  = flag.Int("xor.pop", 0, "XOR GA: population size (0 keeps the default)")
	xorGens = flag.Int("xor.gens", 0, "XOR GA: generations (0 keeps the default)")
)

func (c *xorConfig) applyFlags() {
	if *xorSeed != 0 {
		c.Seed = *xorSeed
	}
	if *xorPop != 0 {
		c.PopulationSize = *xorPop
	}
	if *xorGens != 0 {
		c.Generations = *xorGens
	}
}

func (c xorConfig) validate() error {
	switch {
	case c.PopulationSize < 2:
		return fmt.Errorf("PopulationSize must be at least 2")
	case c.Elites < 0 || c.Elites >= c.PopulationSize:
		return fmt.Errorf("Elites must be in 0..PopulationSize-1")
	case c.TournamentSize < 1:
		return fmt.Errorf("TournamentSize must be at least 1")
	case c.MutationsPerChild < 0:
		return fmt.Errorf("MutationsPerChild must not be negative")
	case c.Workers < 1:
		return fmt.Errorf("Workers must be at least 1")
	case c.SolveTolerance <= 0 || c.SolveTolerance > 0.5:
		return fmt.Errorf("SolveTolerance must be in (0, 0.5]")
	}
	if err := c.ConnSpec.Validate(); err != nil {
		return fmt.Errorf("ConnSpec: %w", err)
	}
	if err := c.NodeSpec.Validate(); err != nil {
		return fmt.Errorf("NodeSpec: %w", err)
	}
	return nil
}

// ---------------------------------------------------------------------------
// Problem data
// ---------------------------------------------------------------------------

// The substrate has 3 inputs (x1, x2 and a constant 1, since substrate nodes
// have no bias of their own) and 1 output.
const (
	xorSubstrateInputs  = 3
	xorSubstrateOutputs = 1
)

// xorCases are the four XOR cases as substrate inputs and the target output.
var xorCases = []struct {
	in     []float64
	target float64
}{
	{[]float64{0, 0, 1}, 0},
	{[]float64{0, 1, 1}, 1},
	{[]float64{1, 0, 1}, 1},
	{[]float64{1, 1, 1}, 0},
}

// ---------------------------------------------------------------------------
// Genomes
// ---------------------------------------------------------------------------

// Genome holds the evolvable values of one dense CPPN: one weight per
// connection and one activation function per hidden node.
type Genome struct {
	weights []float64
	acts    []ActivationFuncID
}

// RandomGenome returns a genome of the right size for spec, with weights drawn
// uniformly from [-weightRange, weightRange] and random activation functions.
func RandomGenome(rng *rand.Rand, spec DenseSpec, weightRange float64) Genome {
	g := Genome{
		weights: make([]float64, spec.WeightGenomeLength()),
		acts:    make([]ActivationFuncID, spec.ActivationGenomeLength()),
	}
	for i := range g.weights {
		g.weights[i] = randomWeight(rng, weightRange)
	}
	for i := range g.acts {
		g.acts[i] = ActivationFuncID(rng.Intn(AllActivationFuncCount))
	}
	return g
}

func randomWeight(rng *rand.Rand, weightRange float64) float64 {
	return rng.Float64()*weightRange*2 - weightRange
}

// Clone returns a copy of g that shares no memory with it, so changing the
// copy never changes the original.
func (g Genome) Clone() Genome {
	return Genome{weights: slices.Clone(g.weights), acts: slices.Clone(g.acts)}
}

// Mutate returns a copy of g with one point mutation: with probability
// actProb an activation function is replaced by a random one, otherwise a
// weight is replaced by a random value. g is not changed.
func (g Genome) Mutate(rng *rand.Rand, actProb, weightRange float64) Genome {
	g = g.Clone()
	if len(g.acts) > 0 && (len(g.weights) == 0 || rng.Float64() < actProb) {
		g.acts[rng.Intn(len(g.acts))] = ActivationFuncID(rng.Intn(AllActivationFuncCount))
	} else if len(g.weights) > 0 {
		g.weights[rng.Intn(len(g.weights))] = randomWeight(rng, weightRange)
	}
	return g
}

// Crossover returns a child of g (the fitter parent) and o. Each gene comes
// from g with probability keep, otherwise from o. Neither parent is changed.
func (g Genome) Crossover(rng *rand.Rand, o Genome, keep float64) Genome {
	child := g.Clone()
	for i := range child.weights {
		if rng.Float64() >= keep {
			child.weights[i] = o.weights[i]
		}
	}
	for i := range child.acts {
		if rng.Float64() >= keep {
			child.acts[i] = o.acts[i]
		}
	}
	return child
}

// xorIndividual is one member of the population: a connection CPPN genome and
// a node CPPN genome, evolved together.
type xorIndividual struct {
	conn, node Genome
}

func (ind xorIndividual) clone() xorIndividual {
	return xorIndividual{conn: ind.conn.Clone(), node: ind.node.Clone()}
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

// xorResult is everything evaluation learns about one individual.
type xorResult struct {
	fitness float64    // in [0, 1]; -1 if the network could not be built or produced NaN
	mse     float64    // mean squared error over the four cases
	correct int        // cases whose output is within SolveTolerance of the target
	outputs [4]float64 // the network's output for each case
	nodes   int        // nodes in the generated network
	conns   int        // connections in the generated network
	err     error      // why the network could not be built, if it could not
}

const xorFailedFitness = -1

func (r xorResult) solved() bool { return r.err == nil && r.correct == len(xorCases) }

// buildXORNetwork turns an individual into the network that is scored.
func buildXORNetwork(cfg xorConfig, ind xorIndividual) (*AbstractNetwork, *Network, error) {
	connCPPN, err := cfg.ConnSpec.BuildCPPN(ind.conn.weights, ind.conn.acts)
	if err != nil {
		return nil, nil, fmt.Errorf("connection CPPN: %w", err)
	}
	nodeCPPN, err := cfg.NodeSpec.BuildNodeCPPN(ind.node.weights, ind.node.acts)
	if err != nil {
		return nil, nil, fmt.Errorf("node CPPN: %w", err)
	}
	an, err := GenerateAbstractFromCPPNs(connCPPN, nodeCPPN, cfg.CPPNBias, 0, false,
		xorSubstrateInputs, xorSubstrateOutputs,
		cfg.InitialDepth, cfg.MaxDepth, cfg.IterationLevel, cfg.MaxHiddenNodes,
		cfg.VarianceThreshold, cfg.BandThreshold, cfg.DivisionThreshold)
	if err != nil {
		return nil, nil, fmt.Errorf("ES-HyperNEAT: %w", err)
	}
	net, err := an.ConvertToNetwork(xorSubstrateInputs, xorSubstrateOutputs, false)
	if err != nil {
		return nil, nil, fmt.Errorf("conversion: %w", err)
	}
	return an, net, nil
}

// evaluateXOR scores an individual. Fitness is 1 minus the mean squared
// error, with each case's squared error capped at 1 so that one wild output
// cannot dominate; a perfect network scores 1.
func evaluateXOR(cfg xorConfig, ind xorIndividual) xorResult {
	_, net, err := buildXORNetwork(cfg, ind)
	if err != nil {
		return xorResult{fitness: xorFailedFitness, err: err}
	}
	r := xorResult{nodes: len(net.Nodes), conns: len(net.Connections)}
	capped := 0.0
	for i, c := range xorCases {
		out := net.Compute(c.in)[0]
		if math.IsNaN(out) {
			return xorResult{fitness: xorFailedFitness, nodes: r.nodes, conns: r.conns, err: fmt.Errorf("output is NaN")}
		}
		r.outputs[i] = out
		e := (out - c.target) * (out - c.target)
		r.mse += e / float64(len(xorCases))
		capped += min(e, 1) / float64(len(xorCases))
		if math.Abs(out-c.target) < cfg.SolveTolerance {
			r.correct++
		}
	}
	r.fitness = 1 - capped
	return r
}

// evaluateAll scores every individual using cfg.Workers goroutines. Each
// evaluation builds its own CPPNs and network, so they share nothing, and the
// results depend only on the individuals, not on the number of workers.
func evaluateAll(cfg xorConfig, pop []xorIndividual) []xorResult {
	results := make([]xorResult, len(pop))
	jobs := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < cfg.Workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				results[i] = evaluateXOR(cfg, pop[i])
			}
		}()
	}
	for i := range pop {
		jobs <- i
	}
	close(jobs)
	wg.Wait()
	return results
}

// ---------------------------------------------------------------------------
// Genetic algorithm
// ---------------------------------------------------------------------------

// tournament returns the index (into ranked, best first) of the best of
// size randomly chosen individuals. Since ranked is sorted, that is simply the
// smallest index drawn.
func tournament(rng *rand.Rand, n, size int) int {
	best := rng.Intn(n)
	for i := 1; i < size; i++ {
		if j := rng.Intn(n); j < best {
			best = j
		}
	}
	return best
}

// nextGeneration builds a new population from ranked (sorted best first):
// the elites unchanged, then children made by tournament selection,
// optional crossover, and point mutations on both genomes.
func nextGeneration(rng *rand.Rand, cfg xorConfig, ranked []xorIndividual) []xorIndividual {
	next := make([]xorIndividual, 0, cfg.PopulationSize)
	for i := 0; i < cfg.Elites; i++ {
		next = append(next, ranked[i].clone())
	}
	for len(next) < cfg.PopulationSize {
		a := tournament(rng, len(ranked), cfg.TournamentSize)
		child := ranked[a].clone()
		if rng.Float64() < cfg.CrossoverRate {
			b := tournament(rng, len(ranked), cfg.TournamentSize)
			better, other := ranked[min(a, b)], ranked[max(a, b)]
			child = xorIndividual{
				conn: better.conn.Crossover(rng, other.conn, cfg.CrossoverKeep),
				node: better.node.Crossover(rng, other.node, cfg.CrossoverKeep),
			}
		}
		for m := 0; m < cfg.MutationsPerChild; m++ {
			child.conn = child.conn.Mutate(rng, cfg.ActMutationProb, cfg.WeightRange)
			child.node = child.node.Mutate(rng, cfg.ActMutationProb, cfg.WeightRange)
		}
		next = append(next, child)
	}
	return next
}

// rank sorts the population and its results together, best fitness first.
// The sort is stable, so ties keep their order and runs are reproducible.
func rank(pop []xorIndividual, results []xorResult) {
	idx := make([]int, len(pop))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return results[idx[a]].fitness > results[idx[b]].fitness })
	p, r := slices.Clone(pop), slices.Clone(results)
	for i, j := range idx {
		pop[i], results[i] = p[j], r[j]
	}
}

// ---------------------------------------------------------------------------
// Logging
// ---------------------------------------------------------------------------

func logConfig(t *testing.T, cfg xorConfig) {
	t.Helper()
	t.Logf("XOR GA settings:")
	t.Logf("  seed=%d population=%d generations=%d elites=%d tournament=%d workers=%d",
		cfg.Seed, cfg.PopulationSize, cfg.Generations, cfg.Elites, cfg.TournamentSize, cfg.Workers)
	t.Logf("  success: every output within %.2f of its target", cfg.SolveTolerance)
	t.Logf("  crossover rate=%.2f keep=%.2f | mutations/child=%d act-mutation prob=%.2f weight range=±%.1f",
		cfg.CrossoverRate, cfg.CrossoverKeep, cfg.MutationsPerChild, cfg.ActMutationProb, cfg.WeightRange)
	t.Logf("  conn CPPN: %d inputs, hidden %v, %d outputs %v, bias input %d -> %d weights, %d activations",
		cfg.ConnSpec.Inputs, cfg.ConnSpec.Hidden, cfg.ConnSpec.Outputs, cfg.ConnSpec.OutputActs, cfg.ConnSpec.BiasInput,
		cfg.ConnSpec.WeightGenomeLength(), cfg.ConnSpec.ActivationGenomeLength())
	t.Logf("  node CPPN: %d inputs, hidden %v, %d outputs %v, bias input %d -> %d weights, %d activations",
		cfg.NodeSpec.Inputs, cfg.NodeSpec.Hidden, cfg.NodeSpec.Outputs, cfg.NodeSpec.OutputActs, cfg.NodeSpec.BiasInput,
		cfg.NodeSpec.WeightGenomeLength(), cfg.NodeSpec.ActivationGenomeLength())
	t.Logf("  ES-HyperNEAT: depth %d..%d, iteration level %d, max hidden %d, thresholds variance=%.3f band=%.3f division=%.3f, CPPN bias %.1f",
		cfg.InitialDepth, cfg.MaxDepth, cfg.IterationLevel, cfg.MaxHiddenNodes,
		cfg.VarianceThreshold, cfg.BandThreshold, cfg.DivisionThreshold, cfg.CPPNBias)
}

// logGeneration writes one summary line for a ranked generation.
func logGeneration(t *testing.T, gen int, results []xorResult, elapsed time.Duration) {
	t.Helper()
	failed, empty := 0, 0
	sum := 0.0
	for _, r := range results {
		sum += r.fitness
		switch {
		case r.err != nil:
			failed++
		case r.conns == 0:
			empty++
		}
	}
	best := results[0]
	t.Logf("gen %4d | best %.4f (mse %.4f, %d/4 correct, %d nodes, %d conns) | mean %.4f median %.4f | failed %d, no connections %d | %v",
		gen, best.fitness, best.mse, best.correct, best.nodes, best.conns,
		sum/float64(len(results)), results[len(results)/2].fitness, failed, empty, elapsed.Round(time.Millisecond))
}

// logBest describes the best individual and the network it builds.
func logBest(t *testing.T, cfg xorConfig, ind xorIndividual, r xorResult) {
	t.Helper()
	t.Logf("best individual: fitness %.4f, mse %.5f, %d/4 correct", r.fitness, r.mse, r.correct)
	for i, c := range xorCases {
		t.Logf("  XOR(%g, %g) = %.4f (target %g)", c.in[0], c.in[1], r.outputs[i], c.target)
	}
	an, _, err := buildXORNetwork(cfg, ind)
	if err != nil {
		t.Logf("  network could not be rebuilt: %v", err)
		return
	}
	hidden := an.Nodes[xorSubstrateInputs+xorSubstrateOutputs:]
	counts := map[ActivationFuncID]int{}
	for _, n := range hidden {
		counts[n.ActivationFunc]++
	}
	var acts []string
	for id := ActivationFuncID(0); id < AllActivationFuncCount; id++ {
		if counts[id] > 0 {
			acts = append(acts, fmt.Sprintf("%d×%d", counts[id], id))
		}
	}
	t.Logf("  network: %d hidden nodes (activation ID counts: %s), %d connections, output activation %d",
		len(hidden), strings.Join(acts, " "), len(an.Connections), an.Nodes[xorSubstrateInputs].ActivationFunc)
	t.Logf("  conn CPPN hidden activations: %v", ind.conn.acts)
	t.Logf("  node CPPN hidden activations: %v", ind.node.acts)
}

// ---------------------------------------------------------------------------
// The test
// ---------------------------------------------------------------------------

func TestSolveXORWithGA(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping the XOR GA in -short mode")
	}
	cfg := defaultXORConfig()
	cfg.applyFlags()
	if err := cfg.validate(); err != nil {
		t.Fatal(err)
	}
	logConfig(t, cfg)

	rng := rand.New(rand.NewSource(cfg.Seed))
	pop := make([]xorIndividual, cfg.PopulationSize)
	for i := range pop {
		pop[i] = xorIndividual{
			conn: RandomGenome(rng, cfg.ConnSpec, cfg.WeightRange),
			node: RandomGenome(rng, cfg.NodeSpec, cfg.WeightRange),
		}
	}

	start := time.Now()
	var results []xorResult
	for gen := 0; gen < cfg.Generations; gen++ {
		results = evaluateAll(cfg, pop)
		rank(pop, results)

		last := gen == cfg.Generations-1 || results[0].solved()
		if gen%cfg.LogEvery == 0 || last {
			logGeneration(t, gen, results, time.Since(start))
		}
		if results[0].solved() {
			t.Logf("solved XOR in generation %d after %v", gen, time.Since(start).Round(time.Millisecond))
			break
		}
		if gen < cfg.Generations-1 {
			pop = nextGeneration(rng, cfg, pop)
		}
	}

	if errs := results[0].err; errs != nil {
		t.Logf("best individual failed to build: %v", errs)
	}
	logBest(t, cfg, pop[0], results[0])
	if !results[0].solved() {
		t.Fatalf("XOR not solved in %d generations (best %d/4 correct, mse %.4f)", cfg.Generations, results[0].correct, results[0].mse)
	}
}

// ---------------------------------------------------------------------------
// Everything this test depends on
// ---------------------------------------------------------------------------
//
// Defined in this file (AI-generated test code):
//   - settings (including SolveTolerance): xorConfig, defaultXORConfig, the -xor.seed / -xor.pop /
//     -xor.gens flags, applyFlags, validate
//   - problem data: xorSubstrateInputs, xorSubstrateOutputs, xorCases
//   - genomes: Genome, RandomGenome, randomWeight, Clone, Mutate, Crossover,
//     xorIndividual
//   - evaluation: xorResult, xorFailedFitness, buildXORNetwork, evaluateXOR,
//     evaluateAll
//   - GA: tournament, nextGeneration, rank
//   - logging: logConfig, logGeneration, logBest
//
// Library code it calls:
//   - DenseSpec (dense_spec.go): Validate, WeightGenomeLength,
//     ActivationGenomeLength, BuildCPPN, BuildNodeCPPN
//   - GenerateAbstractFromCPPNs (eshyperneat.go)
//   - AbstractNetwork.ConvertToNetwork, Network.Compute (network.go)
//   - ActivationFuncID constants, ActivationFuncCount, AllActivationFuncCount
//     (activations.go)
