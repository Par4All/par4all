from pyps import workspace
with workspace("properties0.c",verbose=False) as w:
	print w.props.KERNEL_LOAD_STORE_FORCE_LOAD
	w.props.KERNEL_LOAD_STORE_FORCE_LOAD=True
	print w.props.kernel_load_store_force_load
