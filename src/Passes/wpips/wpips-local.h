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
/*
extern char *vsprintf();
extern int      fprintf();
extern int      pclose();
extern int      textsw_possibly_normalize();
*/

/* This variable is used to indicate wether wpips is in the Emacs
   mode: */
extern bool wpips_emacs_mode;
