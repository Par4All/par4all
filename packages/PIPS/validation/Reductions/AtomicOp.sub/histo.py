from validation import vworkspace


with vworkspace() as w:
    w.fun.histogram.privatize_module()

    w.fun.histogram.validate_phases('coarse_grain_parallelization',
                                    'coarse_grain_parallelization_with_reduction',
                                    'replace_reduction_with_atomic')


