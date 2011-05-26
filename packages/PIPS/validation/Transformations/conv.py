from pyps import workspace, module
with workspace("conv.c") as w:
	c=w.fun.conv
	w.activate(module.must_regions)
	w.activate(module.region_chains)
	w.activate(module.rice_regions_dependence_graph)
	c.loops(0).unroll(rate=4)
	c.delay_communications(accel_load="=")
	c.partial_eval()
	c.loop_fusion()
	c.display()

