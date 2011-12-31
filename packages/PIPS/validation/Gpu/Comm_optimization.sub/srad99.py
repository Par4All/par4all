from validation import vworkspace

with vworkspace() as w:
    w.activate("must_regions")
    w.fun.main.privatize_module()
    w.fun.main.coarse_grain_parallelization()
    w.fun.main.gpu_ify()
    w.filter(lambda m: m.name.startswith('p4a_launcher')).flag_kernel()
    w.fun.main.validate_phases("kernel_data_mapping")


