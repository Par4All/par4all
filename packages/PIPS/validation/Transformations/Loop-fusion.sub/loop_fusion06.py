from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace
import validate_fusion
import os

os.environ['WSPACE']="loop_fusion06"
os.environ['FILE']=os.environ['WSPACE']+".c"

with vworkspace() as w:
    w.all_functions.validate_fusion(parallelize=True,flatten=False)


