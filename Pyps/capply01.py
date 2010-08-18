# convertion of Semantics/all04.tpips into pyps
# tests usage of capply
from pyps import *
w = workspace(['capply01.f'],activates=['TRANSFORMERS_INTER_FULL','INTERPROCEDURAL_SUMMARY_PRECONDITION','PRECONDITIONS_INTER_FULL','PRINT_CODE_CUMULATED_EFFECTS'])
print " Initial code with preconditions for ALL04 after cleanup"

w.all.display()
# there is a capply there
w.all.partial_eval()
w.all.display()
# and there
w.all.suppress_dead_code()
w.all.display()
# end
w.close()
