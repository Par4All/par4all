from validation import vworkspace

with vworkspace() as w:
  w.activate('must_regions')
  w.props.constant_path_effects = False
  w.all_functions.display('PRINT_CODE_REGIONS')
  w.all_functions.display('PRINT_CODE_IN_REGIONS')
  w.all_functions.display('PRINT_CODE_OUT_REGIONS')

