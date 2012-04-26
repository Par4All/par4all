from validation import vworkspace

with vworkspace("compilation_unit01_bis.c") as w:
  w.activate('must_regions')
  w.all_functions.display('PRINT_CODE_REGIONS')
  w.all_functions.display('PRINT_CODE_IN_REGIONS')
  w.all_functions.display('PRINT_CODE_OUT_REGIONS')

