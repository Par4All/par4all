from validation import vworkspace

with vworkspace() as w:
    w.activate("must_regions")
    w.fun.kernel.flag_kernel()
    w.fun.main.kernel_data_mapping()
    w.all_functions.display()
    #w.all_functions.display("PRINT_CODE_REGIONS")



