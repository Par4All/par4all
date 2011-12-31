from validation import vworkspace
import openmp

with vworkspace() as w:
  w.props.flatten_code_unroll = False
  w.all_functions.validate_phases("openmp","flatten_code")
