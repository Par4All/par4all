from validation import vworkspace

with vworkspace() as w:
    r = w.fun.Run

    w.activate("print_chains_graph")
    #w.activate("print_whole_dependence_graph")

    print "*********************** Without points-to analysis ***********************"
    r.atomic_chains()
    r.display(rc="dg_file");

    print ""
    print ""
    print ""
    print ""
    print "************************* With points-to analysis ************************"
    w.activate("proper_effects_with_points_to")
    r.atomic_chains()
    r.display(rc="dg_file");


