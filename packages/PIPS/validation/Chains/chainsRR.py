from validation import vworkspace
import pypsex 

with vworkspace() as w:
    w.props.keep_read_read_dependence=True
    w.props.print_dependence_graph_using_sru_format=True
    w.all_functions.dump_chains_or_dg("chains")

