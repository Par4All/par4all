/*#define REGULAR_VERSION "Regular Version"
  #define PARALLEL_VERSION "Parallel Version"
  */

/* Labels for menu Edit/View (These definitions are almost automatically available as aliases
   in wpips.rc; FI) */
#define USER_VIEW "User View"
#define SEQUENTIAL_VIEW "Sequential View"
#define PARALLEL_VIEW "Parallel View"
#define CALLGRAPH_VIEW "Callgraph View"
#define ICFG_VIEW "ICFG View"
#define DISTRIBUTED_VIEW "Distributed View"
#define DEPENDENCE_GRAPH_VIEW "Dependence Graph View"
#define FLINT_VIEW "Flint View"

/* Labels for menu Transform */
#define PARALLELIZE_TRANSFORM "! Parallelize"
#define PRIVATIZE_TRANSFORM "Privatize"
#define DISTRIBUTE_TRANSFORM "Distribute"
#define PARTIAL_EVAL_TRANSFORM "Partial Eval"
#define UNROLL_TRANSFORM "Loop Unroll"
#define STRIP_MINE_TRANSFORM "Strip Mining"
#define LOOP_INTERCHANGE_TRANSFORM "Loop Interchange"
#define SUPPRESS_DEAD_CODE_TRANSFORM "! Dead Code Elimination"
#define ATOMIZER_TRANSFORM "! Atomize"
#define REDUCTIONS_TRANSFORM "!! Reductions"
#define STATIC_CONTROLIZE_TRANSFORM "Static Controlize"

#define SEMANTICS_ANALYZE "Semantics"
#define CALLGRAPH_ANALYZE "Call Graph"

#define FULL_DG_PROPS "Full Dependence Graph"
#define FAST_DG_PROPS "Fast Dependence Graph"

#define SMALL_BUFFER_LENGTH 256
#define LARGE_BUFFER_LENGTH 256

#define MESSAGE_BUFFER_LENGTH 128
#define TEXT_BUFFER_LENGTH 1024

/* How many display wondows can be opened : */
#define MAX_NUMBER_OF_WPIPS_WINDOWS 9
#define INITIAL_NUMBER_OF_WPIPS_WINDOWS 2
extern int number_of_wpips_windows;


extern Frame main_frame, 
    schoose_frame, 
    mchoose_frame, 
    log_frame, 
    edit_frame[MAX_NUMBER_OF_WPIPS_WINDOWS], 
    help_frame, 
    query_frame,
	properties_frame;

extern Panel main_panel,
    status_panel,
    query_panel,
    mchoose_panel,
    schoose_panel,
    help_panel;

typedef enum {PIPS_ICON, ICFG_ICON, WP65_PE_ICON, WP65_bank_ICON, callgraph_ICON,
		parallel_ICON, sequential_ICON, user_ICON, LAST_ICON} icon_list;


extern char *strdup(), *re_comp();
extern int re_exec();


typedef bool success ;

/* Manques dans les .h de PIPS : */
extern int get_bool_property();

/* Contourne certains manques de .h Sun. RK, 14/01/93.
	Doit disparai^tre avec un nouveau compilateur. */
/* extern char *vsprintf(char *, const char *, void *); */
extern char *vsprintf();
extern int      fprintf();
extern int      pclose();
extern int      textsw_possibly_normalize();
