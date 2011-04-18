


import pyps

pyps.workspace.delete("scope01")
w= pyps.workspace('scope01.c',name="scope01")

r = w.fun.Run



w.activate("print_chains_graph")
#w.activate("print_whole_dependence_graph")


print "*********************** Without scope filtering ***********************"
w.props.atomic_chains_filter_scope = False
r.atomic_chains()
r.display("dg_file");

print ""
print ""
print ""
print ""
print "************************* With scope filtering ************************"
w.props.atomic_chains_filter_scope = True
r.atomic_chains()
r.display("dg_file");


