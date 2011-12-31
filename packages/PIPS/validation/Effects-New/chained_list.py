from validation import vworkspace

with vworkspace() as w:
    w.props.constant_path_effects = False
    w.fun.append_int_to_list.display('print_code_proper_effects')
    w.fun.append_int_to_list.atomic_chains()

