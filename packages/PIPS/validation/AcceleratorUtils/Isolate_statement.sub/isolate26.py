from validation import vworkspace

try:
    with vworkspace() as w:
        h=w.compile()
        ref= w.run(h)
        w.add_source("include/par4all.c")
        w.props.constant_path_effects=False
        w.props.kernel_load_store_scalar=True
        w.props.isolate_statement_label="holy" # it's magic
        w.activate("must_regions")
        w.fun.pain.validate_phases("print_code_regions","isolate_statement")
        h=w.compile()
        w.run(h)
        print ref, ':', w.run(h)
except:
    print "statement isolation failed as expected"
