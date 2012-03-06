from validation import vworkspace
import validate_fusion
import os

with vworkspace() as w:
    w.all_functions.validate_fusion(parallelize=True,flatten=False)


