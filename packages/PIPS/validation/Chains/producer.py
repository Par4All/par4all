from validation import vworkspace

with vworkspace("consumer.c") as w:
    r = w.fun.producer_consumer
    w.activate("print_chains_graph")
    r.atomic_chains()
    r.display(rc="dg_file");

