/* 	$Id$	 */
/* header file built by cproto */
#ifndef icfg_header_included
#define icfg_header_included
/* 
   icfg.h -- acronym
   include file for Interprocedural Control Flow Graph
 */

#define ICFG_NOT_FOUND NULL
#define ICFG_OPTIONS "tcdDIFl"

#define ICFG_CALLEES_TOPO_SORT "ICFG_CALLEES_TOPO_SORT"
#define ICFG_DRAW "ICFG_DRAW"
#define ICFG_DEBUG "ICFG_DEBUG"
#define ICFG_DEBUG_LEVEL "ICFG_DEBUG_LEVEL"
#define ICFG_DOs "ICFG_DOs"
#define ICFG_IFs "ICFG_IFs"
#define ICFG_DV "ICFG_DV"
#define ICFG_FLOATs "ICFG_FLOATs"
#define ICFG_SHORT_NAMES "ICFG_SHORT_NAMES"

#define ICFG_DECOR "ICFG_DECOR"

#define ICFG_DECOR_NONE 1
#define ICFG_DECOR_PRECONDITIONS 2
#define ICFG_DECOR_TRANSFORMERS 3
#define ICFG_DECOR_PROPER_EFFECTS 4
#define ICFG_DECOR_CUMULATED_EFFECTS 5
#define ICFG_DECOR_REGIONS 6
#define ICFG_DECOR_IN_REGIONS 7
#define ICFG_DECOR_OUT_REGIONS 8
#define ICFG_DECOR_COMPLEXITIES 9
#define ICFG_DECOR_TOTAL_PRECONDITIONS 10
#define ICFG_DECOR_FILTERED_PROPER_EFFECTS 11


#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#define CALL_MARK "CALL_MARK@@@@"
#define ICFG_SCAN_INDENT 4
#define ADD_ELEMENT_TO_LIST( _list, _type, _element) \
    (_list = gen_nconc( _list, CONS( _type, _element, NIL)))

/* toposort.c */
extern list module_name_to_callees(string /*module_name*/);
extern list module_to_callees(entity /*mod*/);
extern void topological_number_assign_to_module(hash_table /*hash_module_to_depth*/, entity /*mod*/, int /*n*/);
extern list module_list_sort(hash_table /*hash_module_to_depth*/, list /*current_list*/, entity /*mod*/, int /*n*/);
extern list topologically_sorted_module_list(entity /*mod*/);
extern void print_module_name_to_toposorts(string /*module_name*/);
/* icfg_scan.c */
extern void icfg_error_handler(void);
extern void print_module_icfg(entity /*module*/);
extern void print_module_icfg_with_decoration(entity /*module*/, text (* /*deco*/)(string));
/* print.c */
extern bool generic_print_icfg(string /*module_name*/);
extern bool parametrized_print_icfg(string /*module_name*/, bool /*print_ifs*/, bool /*print_dos*/, text (* /*deco*/)(string));
extern bool print_icfg(string /*module_name*/);
extern bool print_icfg_with_complexities(string /*module_name*/);
extern bool print_icfg_with_preconditions(string /*module_name*/);
extern bool print_icfg_with_total_preconditions(string /*module_name*/);
extern bool print_icfg_with_transformers(string /*module_name*/);
extern bool print_icfg_with_proper_effects(string /*module_name*/);
extern bool print_icfg_with_filtered_proper_effects(string /*module_name*/);
extern bool print_dvicfg_with_filtered_proper_effects(string /*module_name*/);
extern bool print_icfg_with_cumulated_effects(string /*module_name*/);
extern bool print_icfg_with_regions(string /*module_name*/);
extern bool print_icfg_with_in_regions(string /*module_name*/);
extern bool print_icfg_with_out_regions(string /*module_name*/);
extern bool print_icfg_with_loops(string /*module_name*/);
extern bool print_icfg_with_loops_complexities(string /*module_name*/);
extern bool print_icfg_with_loops_preconditions(string /*module_name*/);
extern bool print_icfg_with_loops_total_preconditions(string /*module_name*/);
extern bool print_icfg_with_loops_transformers(string /*module_name*/);
extern bool print_icfg_with_loops_proper_effects(string /*module_name*/);
extern bool print_icfg_with_loops_cumulated_effects(string /*module_name*/);
extern bool print_icfg_with_loops_regions(string /*module_name*/);
extern bool print_icfg_with_loops_in_regions(string /*module_name*/);
extern bool print_icfg_with_loops_out_regions(string /*module_name*/);
extern bool print_icfg_with_control(string /*module_name*/);
extern bool print_icfg_with_control_complexities(string /*module_name*/);
extern bool print_icfg_with_control_preconditions(string /*module_name*/);
extern bool print_icfg_with_control_total_preconditions(string /*module_name*/);
extern bool print_icfg_with_control_transformers(string /*module_name*/);
extern bool print_icfg_with_control_proper_effects(string /*module_name*/);
extern bool print_icfg_with_control_cumulated_effects(string /*module_name*/);
extern bool print_icfg_with_control_regions(string /*module_name*/);
extern bool print_icfg_with_control_in_regions(string /*module_name*/);
extern bool print_icfg_with_control_out_regions(string /*module_name*/);
/* util.c */
extern void safe_free_vertex(vertex /*ver*/, list /*l_of_vers*/);
extern list safe_add_vertex_to_list(vertex /*ver*/, list /*l_of_vers*/);
extern list list_of_connected_nodes(vertex /*ver*/, list /*l_of_vers*/);
extern string remove_newline_of_string(string /*s*/);
extern vertex get_vertex_by_string(string /*str_name*/, list /*l_of_vers*/);
extern string sentence_to_string(sentence /*sen*/);
extern list safe_make_successor(vertex /*ver_parent*/, vertex /*ver_child*/, list /*l_of_vers*/);
extern void print_graph_of_text_to_daVinci(FILE */*f_out*/, list /*l_of_vers*/);
extern void print_graph_daVinci_from_starting_node(FILE * /*f_out*/, vertex /*start_ver*/);
extern void print_marged_text_from_starting_node(FILE * /*fd*/, int /*margin*/, vertex /*start_ver*/, list /*l_of_vers*/);
extern bool make_resource_from_starting_node
(string /*mod_name*/, string /*res_name*/, string /*file_ext*/, vertex /*start_ver*/, list /*l_of_vers*/, bool /*res_text_type*/);
extern list get_list_of_variable_to_filter(void);
extern list effects_filter(list /*l_effs*/, list /*l_ents*/);
#endif /* icfg_header_included */
