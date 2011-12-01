from validation import vworkspace

with vworkspace() as w:
    w.props.prettyprint_blocks = True
    w.all_functions.display('PRINT_CODE_CUMULATED_EFFECTS')
    w.all_functions.display('PRINT_CODE_IN_EFFECTS')

