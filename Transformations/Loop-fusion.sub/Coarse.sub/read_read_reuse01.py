from validation import vworkspace

with vworkspace() as w:
    w.all_functions.validate_phases("loop_fusion_with_regions")
    print "// With read_read dependence :"
    w.props.keep_read_read_dependence = True
    w.all_functions.validate_phases("loop_fusion_with_regions")

