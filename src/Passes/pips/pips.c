#include <stdio.h>
#include <setjmp.h>
#include <stdlib.h>
#include <unistd.h>

#include "genC.h"
#include "ri.h"
#include "graph.h"
#include "makefile.h"

#include "misc.h"
#include "ri-util.h"
#include "complexity_ri.h"

#include "constants.h"
#include "resources.h"

#include "database.h"
#include "pipsdbm.h"

#include "pipsmake.h"

#include "top-level.h"

jmp_buf pips_top_level;

extern void (*pips_error_handler)();
extern void (*pips_log_handler)();
extern void (*pips_warning_handler)();
extern void set_bool_property();

char *usage = 
	"Usage: %s [-f F]* [-m M] [-s S]* [-p P] [-b B] [-(0|1) T]* wspace\n"
	"\t-f F: source file F\n"
	"\t-m M: module M\n"
	"\t-s S: select rule S\n"
	"\t-p P: perform rule P\n"
	"\t-b B: build resource B\n"
	"\t-(0|1) T: set boolean property T to FALSE or TRUE\n" ;

static char *wspace = NULL;
static char *module = NULL;
static char *performed_rule = NULL;
static list build_resource_names = NIL;
static list source_files = NIL;
static list selected_rules = NIL;

void parse_arguments(argc, argv)
int argc;
char * argv[];
{
    int c;
    extern char *optarg;
    extern int optind;

    if (argc < 2) {
	fprintf(stderr, usage, argv[0]);
	exit(1);
    }

    while ((c = getopt(argc, argv, "f:m:s:p:b:1:0:")) != -1)
	switch (c) {
	case 'f':
	    source_files= gen_nconc(source_files, 
				    CONS(STRING, optarg, NIL));
	    break;
	case 'm':
	    module= optarg;
	    break;
	case 's':
	    selected_rules= gen_nconc(selected_rules, 
				      CONS(STRING, optarg, NIL));
	    break;
	case 'p':
	    performed_rule= optarg;
	    break;
	case 'b':
	    build_resource_names = gen_nconc(build_resource_names,
					     CONS(STRING, optarg, NIL));
	    break;
	/*
	 * next two added to deal with boolean properties directly
	 * FC, 27/03/95
	 */
	case '1':
	    set_bool_property(optarg, TRUE);
	    break;
	case '0':
	    set_bool_property(optarg, FALSE);
	    break;
	case '?':
	    fprintf(stderr, usage, argv[0]);
	    exit(1);
	    ;
	}
    
    if (argc != optind + 1) {
	user_warning("parse_argument", 
		     ((argc < (optind + 1)) ?
		     "Too few arguments\n" : "illegal argument: %s\n"),
		     argv[optind + 1]);
	fprintf(stderr, usage, argv[0]);
	exit(1);
    }
    wspace= argv[argc - 1];
}

void
select_rule(rule_name)
char *rule_name;
{
    user_log("Selecting rule: %s\n", rule_name);

    activate(rule_name);

    if(get_debug_level()>5)
	fprint_activated(stderr);
}

void main(argc, argv)
int argc;
char * argv[];
{
    initialize_newgen();
    initialize_sc((char*(*)(Variable)) entity_local_name); 

    parse_arguments(argc, argv);

    debug_on("PIPSMAKE_DEBUG_LEVEL");

    initialize_signal_catcher();

    if (source_files != NIL) {
	/* Workspace must be created */
	db_create_program(wspace);
	
	MAPL(f_cp, {
	    debug(1, "main", "processing file %s\n", STRING(CAR(f_cp)));
	    process_user_file( STRING(CAR(f_cp)) );
	}, source_files);

	wspace = db_get_current_program_name();
	user_log("Workspace %s created and opened\n", wspace);
    }
    else {
	/* Workspace must be opened */
	if (make_open_program(wspace) == NULL) {
	    user_log("Cannot open workspace %s\n", wspace);
	    exit(1);
	}
	else {
	    user_log("Workspace %s opened\n", wspace);
	}
    }

    /* Open module */
    if (module != NULL) {
	/* CA - le 040293- remplacement de db_open_module(module) par */
	open_module(module);
    }
    else {
	open_module_if_unique();
    }


    /* Make everything */
    if(setjmp(pips_top_level)) {
	/* no need to pop_pips_context() at top-level */
	make_close_program();
	exit(1);
    }
    else {
	push_pips_context(&pips_top_level);
	if (selected_rules != NIL) {
	    /* Select rules */
	    MAPL(r_cp, {
		select_rule(STRING(CAR(r_cp)));
	    }, selected_rules);
	}

	if (performed_rule != NULL) {
	    /* Perform rule */
	    user_log("Request: perform rule %s for module %s.\n", 
		     performed_rule, module);
	    apply(performed_rule, module);
	    user_log("%s performed for %s.\n", performed_rule, module);
	}

	if (build_resource_names != NIL) {
	    /* Build resource */
	    MAPL(crn, {
		string build_resource_name = STRING(CAR(crn));
		user_log("Request: build resource %s for module %s.\n", 
			 build_resource_name, module);
		make(build_resource_name, module);
		user_log("%s build for %s.\n", build_resource_name, module);
	    }, build_resource_names);
	}
    }

    make_close_program();

    /* without exit, program returns any value! */
    exit(0);
}
