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

/*************written by Dat*************/
typedef text (*get_text_function)(list);
/****************************************/
