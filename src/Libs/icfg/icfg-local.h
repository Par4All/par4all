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
#define DVICFG_DECOR_FILTERED_PROPER_EFFECTS 12

/* util.c */
extern void safe_free_vertex(vertex /*ver*/, list /*l_of_vers*/);
extern list safe_add_vertex_to_list(vertex /*ver*/, list /*l_of_vers*/);
extern list list_of_connected_nodes(vertex /*ver*/, list /*l_of_vers*/);
extern string remove_newline_of_string(string);
extern string add_flash_newline_to_string(string);
extern vertex get_vertex_by_string(string /*str_name*/, list /*l_of_vers*/);
extern string sentence_to_string(sentence /*sen*/);
extern list safe_make_successor(vertex /*ver_parent*/, vertex /*ver_child*/, list /*l_of_vers*/);
extern void print_graph_of_text_to_daVinci(FILE * /*fd*/, graph /*g_in*/);
extern void print_graph_daVinci_with_starting_node(FILE * /*fd*/, vertex /*star_ver*/);
extern list get_list_of_variable_to_filter();
extern list effects_filter(list /*l_effs*/, list /*l_ents*/);


