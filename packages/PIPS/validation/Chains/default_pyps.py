from validation import vworkspace
import pypsex

with vworkspace() as w:
    w.props.prettyprint_statement_number=True
    w.props.memory_effects_only = False
    w.all_functions.display("print_code")
    w.all_functions.display("print_code_proper_effects")
    #w.props.print_dependence_graph_using_sru_format=True
    print w.all_functions.dump_chains_or_dg("chains")
    w.all_functions.display("print_code_proper_regions")
    w.activate("region_chains")
    print w.all_functions.dump_chains_or_dg("chains")
