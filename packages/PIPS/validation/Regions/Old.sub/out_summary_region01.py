from validation import vworkspace

with vworkspace() as w:
    w.fun.my_getopt_long.validate_phases("print_code_out_regions")
