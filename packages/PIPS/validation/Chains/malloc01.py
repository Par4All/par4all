from validation import vworkspace
import pypsex
#import os

with vworkspace() as w:
    #os.environ['CHAINS_DEBUG_LEVEL']='5'
    w.props.prettyprint_statement_number=True
    w.props.must_regions = True
    w.props.constant_path_effects = False
    w.fun.main.display("print_code_proper_effects")
    w.props.print_dependence_graph_using_sru_format=True
    print w.all_functions.dump_chains_or_dg("chains")

