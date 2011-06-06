from validation import vworkspace

with vworkspace() as w:
    w.fun.getopt_long.validate_phases("out_summary_regions")
