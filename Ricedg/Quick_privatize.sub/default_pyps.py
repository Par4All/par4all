# Do not activate phase privatize_module.
# This directory is intended to test the quick privatization
# called from ricedg.

from validation import vworkspace


with vworkspace() as w:
  w.props.memory_effects_only = False
  w.all_functions.validate_phases("internalize_parallel_code")
