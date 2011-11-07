from validation import vworkspace

with vworkspace() as w:
  w.activate('must_regions')
  w.all_functions.display('PRINT_CODE_REGIONS')

