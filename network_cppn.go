package esneatforhebbians

import "fmt"

// NetworkCPPN uses a Network as a connection CPPN. The network's inputs are,
// in order: x1, y1, x2, y2, bias, networkType (passed as a float64). Its
// outputs are the CPPN's outputs, laid out by the indices in cppn.go.
//
// The network must not allow recurrent connections, so each query depends only
// on its inputs. A NetworkCPPN reuses its buffers between queries, so it is
// not safe for concurrent use; give each goroutine its own.
type NetworkCPPN struct {
	net     *Network
	in, out []float64
}

// NewNetworkCPPN wraps net as a connection CPPN. net must have exactly 6
// inputs, at least 1 output, and AllowRecurrent off.
func NewNetworkCPPN(net *Network) (*NetworkCPPN, error) {
	if err := checkCPPNNetwork(net, 6, "a connection CPPN network", "x1, y1, x2, y2, bias, networkType"); err != nil {
		return nil, err
	}
	return &NetworkCPPN{net: net, in: make([]float64, 6), out: make([]float64, net.OutputCount)}, nil
}

// Compute implements CPPN. The returned slice is reused by the next query.
func (c *NetworkCPPN) Compute(x1, y1, x2, y2, bias float64, networkType int) []float64 {
	c.in[0], c.in[1], c.in[2], c.in[3], c.in[4], c.in[5] = x1, y1, x2, y2, bias, float64(networkType)
	c.net.computeInto(c.in, c.out)
	return c.out
}

// GetOutputSize implements CPPN.
func (c *NetworkCPPN) GetOutputSize() int { return c.net.OutputCount }

// NetworkNodeCPPN uses a Network as a node CPPN. The network's inputs are, in
// order: x, y, bias, networkType (passed as a float64). The same rules as for
// NetworkCPPN apply.
type NetworkNodeCPPN struct {
	net     *Network
	in, out []float64
}

// NewNetworkNodeCPPN wraps net as a node CPPN. net must have exactly 4 inputs,
// at least 1 output, and AllowRecurrent off.
func NewNetworkNodeCPPN(net *Network) (*NetworkNodeCPPN, error) {
	if err := checkCPPNNetwork(net, 4, "a node CPPN network", "x, y, bias, networkType"); err != nil {
		return nil, err
	}
	return &NetworkNodeCPPN{net: net, in: make([]float64, 4), out: make([]float64, net.OutputCount)}, nil
}

// ComputeNode implements NodeCPPN. The returned slice is reused by the next
// query.
func (c *NetworkNodeCPPN) ComputeNode(x, y, bias float64, networkType int) []float64 {
	c.in[0], c.in[1], c.in[2], c.in[3] = x, y, bias, float64(networkType)
	c.net.computeInto(c.in, c.out)
	return c.out
}

// GetOutputSize implements NodeCPPN.
func (c *NetworkNodeCPPN) GetOutputSize() int { return c.net.OutputCount }

func checkCPPNNetwork(net *Network, inputs int, what, inputNames string) error {
	if net.AllowRecurrent {
		return fmt.Errorf("%s must be built with recurrent connections turned off", what)
	}
	if net.InputCount != inputs {
		return fmt.Errorf("%s needs %d inputs (%s), got %d", what, inputs, inputNames, net.InputCount)
	}
	if net.OutputCount == 0 {
		return fmt.Errorf("%s needs at least 1 output", what)
	}
	return nil
}
