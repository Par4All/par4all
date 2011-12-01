from validation import vworkspace as workspace

with workspace() as w:
    w.all_functions.validate_phases("print_code_cumulated_reductions")
