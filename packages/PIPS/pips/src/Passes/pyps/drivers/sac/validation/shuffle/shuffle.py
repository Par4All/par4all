import pyps,os
pyps.workspace.delete("shuffle")
simdc = "../../impl/SIMD.c"
os.system("gcc -E -DRWBITS=128 %s |cproto >./SIMD.h" % simdc)
with pyps.workspace("shuffle.c", simdc, cppflags="-DRWBITS=128", name="shuffle", deleteOnClose=False) as ws:
	ws.activate("REGION_CHAINS")
	ws.activate("RICE_REGIONS_DEPENDENCE_GRAPH")
	ws.activate("PRECONDITIONS_INTER_FULL")
	ws.activate("TRANSFORMERS_INTER_FULL")
	ws.props.sac_simd_register_width=128
	f=ws.fun.shuffle
	f.simdizer()
	f.display()
	ws.save(rep="sac")
