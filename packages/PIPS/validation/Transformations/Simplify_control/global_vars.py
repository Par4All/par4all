from __future__ import with_statement # this is to work with python2.5

from validation import vworkspace

with vworkspace() as w:
    w['global_vars!'].validate_phases()

