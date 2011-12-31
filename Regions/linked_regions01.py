from validation import vworkspace
from os import environ

environ["REGIONS_DEBUG_LEVEL"]= "8"
environ["REGIONS_OPERATORS_DEBUG_LEVEL"]= "8"
environ["EFFECTS_OPERATORS_DEBUG_LEVEL"]= "8"
environ["EFFECTS_DEBUG_LEVEL"]= "8"
environ["SC_DEBUG_LEVEL"]= "5"

with vworkspace() as w:
  w.props.constant_path_effects = False
  w.activate('must_regions')
  w.all_functions.display('PRINT_CODE_REGIONS')

