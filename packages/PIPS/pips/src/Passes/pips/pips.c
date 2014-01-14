/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "effects.h"
#include "properties.h"

#include "misc.h"
#include "newgen.h"
#include "ri-util.h"
#include "complexity_ri.h"

#include "constants.h"
#include "resources.h"

#include "database.h"
#include "pipsdbm.h"

#include "pipsmake.h"

#include "top-level.h"

extern void (*pips_error_handler)();
extern void (*pips_log_handler)();
extern void (*pips_warning_handler)();

static char *usage = 
  "Usage: %s [-v] [-f F]* [-m M] [-s S]* [-p P] [-b B] [-(0|1) T]* wspace\n"
  "\t-v: pips version (which pips/ARCH)\n"
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
static gen_array_t source_files = NULL;
static list selected_rules = NIL;

static void pips_parse_arguments(int argc, char * argv[])
{
    int c;
    extern char *optarg;
    extern int optind;
    extern char * soft_revisions;
    extern char * soft_date;
    source_files = gen_array_make(5);

    while ((c = getopt(argc, argv, "vf:m:s:p:b:1:0:")) != -1)
	switch (c) {
	case 'v':
	    fprintf(stdout, 
		    "tpips: (%s)\n"
		    "ARCH=" STRINGIFY(SOFT_ARCH) "\n"
		    "REVS=\n"
		    "%s" 
		    "DATE=%s\n", 
		    argv[0], soft_revisions, soft_date);
	    exit(0);
	    break;
	case 'f':
	    gen_array_append(source_files, optarg);
	    break;
	case 'm':
	    module= optarg;
	    break;
	case 's':
	    selected_rules = 
		gen_nconc(selected_rules, CONS(STRING, optarg, NIL));
	    break;
	case 'p':
	    performed_rule = optarg;
	    break;
	case 'b':
	    build_resource_names = 
		gen_nconc(build_resource_names, CONS(STRING, optarg, NIL));
	    break;

	/* next two added to deal with boolean properties directly
	 * FC, 27/03/95
	 */
	case '1':
	    set_bool_property(optarg, true);
	    break;
	case '0':
	    set_bool_property(optarg, false);
	    break;
	case '?':
	    fprintf(stderr, usage, argv[0]);
	    exit(1);
	    ;
	}
    
    if (argc < 2) {
	fprintf(stderr, usage, argv[0]);
	exit(1);
    }

    if (argc != optind + 1) {
	user_warning("pips_parse_argument", 
		     ((argc < (optind + 1)) ?
		     "Too few arguments\n" : "illegal argument: %s\n"),
		     argv[optind + 1]);
	fprintf(stderr, usage, argv[0]);
	exit(1);
    }
    wspace= argv[argc - 1];
}

static void
select_rule(rule_name)
char *rule_name;
{
    user_log("Selecting rule: %s\n", rule_name);

    activate(rule_name);

    ifdebug(5) fprint_activated(stderr);
}

/* Pips user log */

static void pips_user_log(const char * fmt, va_list args)
{
    FILE * log_file = get_log_file();

    if(log_file!=NULL) {
        /* it seems one cannot use args twice... so a copy is made here... */
        va_list args_copy;
	va_copy(args_copy, args);
	if (vfprintf(log_file, fmt, args_copy) <= 0) {
	    perror("pips_user_log");
	    abort();
	}
	else
	    fflush(log_file);
	va_end(args_copy);
    }

    if(!get_bool_property("USER_LOG_P"))
	return;

    /* It goes to stderr to have only displayed files on stdout */
    (void) vfprintf(stderr, fmt, args);
    fflush(stderr);
}

int 
pips_main(int argc, char ** argv)
{
    bool success = true;
    pips_checks();

    initialize_newgen();
    initialize_sc((char*(*)(Variable)) entity_local_name); 
    set_exception_callbacks(push_pips_context, pop_pips_context);

    pips_parse_arguments(argc, argv);

    debug_on("PIPS_DEBUG_LEVEL");
    initialize_signal_catcher();
    pips_log_handler = pips_user_log;

    CATCH(any_exception_error)
    {
	/* no need to pop_pips_context() at top-level */
	/* FI: are you sure make_close_program() cannot call user_error() ? */
	close_workspace(true);
	success = false;
    }
    TRY
    {
	/* Initialize workspace
	 */
	if (gen_array_nitems(source_files)>0) {
	    if(db_create_workspace(wspace)) {
		create_workspace(source_files);
	    }
	    else {
		user_log("Cannot create workspace %s!\n", wspace);
		exit(1);
	    }
	} else {
	    /* Workspace must be opened */
	    if (!open_workspace(wspace)) {
		user_log("Cannot open workspace %s!\n", wspace);
		exit(1);
	    }
	}

	/* Open module
	 */
	if (module != NULL) 
	    open_module(module);
	else 
	    open_module_if_unique();

	/* Activate rules
	 */
	if (success && selected_rules) 
	{
	    MAP(STRING, r, select_rule(r), selected_rules);
	}

	/* Perform applies
	 */
	if (success && performed_rule && module) 
	{
	    success = safe_apply(performed_rule, module);
	    if (success) {
		user_log("%s performed for %s.\n", 
			 performed_rule, module);
	    }
	    else {
		user_log("Cannot perform %s for %s.\n", 
			 performed_rule, module);
	    }
	}

	/* Build resources
	 */
	if (success && build_resource_names && module) 
	{
	    /* Build resource */
	    MAPL(crn, {
		string build_resource_name = STRING(CAR(crn));
		success = safe_make(build_resource_name, module);
		if (!success) 
		{
		    user_log("Cannot build %s for %s.\n", 
			     build_resource_name, module);
		    break;
		}
	    }, build_resource_names);
	}
	
	/* whether success or not... */
	close_workspace(true);
	/* pop_performance_spy(stderr, "pips"); */
	/* check debug level if no exception occured */

	UNCATCH(any_exception_error);
    }

    debug_off();

    return !success;
}

/* end of it.
 */
