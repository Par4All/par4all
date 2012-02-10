from validation import vworkspace

with vworkspace("include/par4all.c") as w:
    h=w.compile()
    ref= w.run(h)
    w.props.constant_path_effects=False
    w.props.isolate_statement_label="holy" 
    w.activate("must_regions")
    w.fun.pain.validate_phases("print_code_regions","isolate_statement")
    h=w.compile()
    w.run(h)
    print ref, ':', w.run(h)
