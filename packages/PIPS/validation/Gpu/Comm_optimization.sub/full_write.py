from validation import vworkspace

with vworkspace() as w:
    w.activate("must_regions")
    kernels = w.filter(lambda m: m.name.startswith("kernel"))
    kernels.flag_kernel()
    w.fun.main.validate_phases("kernel_data_mapping")


