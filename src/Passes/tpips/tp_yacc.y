/* $Id$
 */

%token OPEN
%token CREATE
%token CLOSE
%token DELETE
%token MODULE
%token MAKE
%token APPLY
%token DISPLAY
%token ACTIVATE
%token SET_PROPERTY
%token GET_PROPERTY
%token SET_ENVIRONMENT
%token GET_ENVIRONMENT
%token INFO
%token CDIR
%token PWD

%token OWNER_NAME
%token OWNER_ALL
%token OWNER_PROGRAM
%token OWNER_MAIN
%token OWNER_MODULE
%token OWNER_CALLERS
%token OWNER_CALLEES

%token OPENPAREN
%token COMMA
%token CLOSEPAREN
%token EQUAL

%token NAME
%token A_STRING

%type <name>   NAME A_STRING propname phasename resourcename
%type <status> line instruction
%type <status> i_open i_create i_close i_delete i_module i_make i_pwd
%type <status> i_apply i_activate i_display i_get i_setenv i_getenv i_cd
%type <name> rulename filename 
%type <array> filename_list
%type <rn> resource_id rule_id
%type <owner> owner list_of_owner_name

%{
#include <stdlib.h>
#include <stdio.h>
#include <string.h>     
#include <sys/param.h>
#include <unistd.h>


#include "genC.h"

#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"

#include "ri-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "phases.h"
#include "properties.h"
#include "pipsmake.h"

#include "top-level.h"
#include "tpips.h"

extern FILE * yyin; 

/* Default comand to print a file (if $PAGER is not set) */
#define CAT_COMMAND "cat"

/********************************************************** static variables */
extern bool tpips_execution_mode;

#define YYERROR_VERBOSE 1 /* MUCH better error messages with bison */

extern int yylex(void);
extern void yyerror(char *);

static void
free_owner_content(res_or_rule * pr)
{
    gen_map(free, pr->the_owners);
    gen_free_list(pr->the_owners);
    free(pr->the_name);
    pr->the_owners = NIL;
    pr->the_name = NULL;
}

void 
close_workspace_if_opened(void)
{
    if (db_get_current_workspace_name() != NULL)
	close_workspace();
}

static void
set_env(string var, string val)
{
    string ival = getenv(var);
    if (!ival || !same_string_p(val, ival))
	putenv(strdup(concatenate(var, "=", val, 0)));
}

%}

%union {
    int status;
    string name;
    res_or_rule rn;
    list owner;
    gen_array_t array;
}

%%

line:   instruction 
    ;

instruction:
	  i_open
	| i_create
  	| i_close
	| i_delete
	| i_module
	| i_make
	| i_apply
	| i_display
	| i_activate
	| i_get
	| i_getenv
	| i_setenv
	| i_cd
	| i_pwd
	| error {$$ = FALSE;}
	;

i_cd: CDIR NAME 
	{
	    user_log("cd %s\n", $2);
	    if (chdir($2)) fprintf(stderr, "error while changing directory\n");
	    free($2);
	}
	;

i_pwd: PWD
	{
	    char pathname[MAXPATHLEN];
	    fprintf(stdout, "current working directory: %s\n", 
		    (char*) getwd(pathname));
	}
	;

i_getenv: GET_ENVIRONMENT NAME
	{
	    string val = getenv($2);
	    user_log("getenv %s\n", $2);
	    if (val) fprintf(stdout, "%s=%s\n", $2, val);
	    else fprintf(stdout, "%s is not defined\n", $2);
	    free($2);
	}
	;

i_setenv: SET_ENVIRONMENT NAME NAME
	{
	    set_env($2, $3);
	    user_log("setenv %s %s\n", $2, $3);
	    free($2); free($3);
	}
	| SET_ENVIRONMENT NAME EQUAL NAME
	{
	    set_env($2, $4);
	    user_log("setenv %s %s\n", $2, $4);
	    free($2); free($4);
	}
	;

i_open:	OPEN NAME 
	{
	    string main_module_name;

	    pips_debug(7,"reduce rule i_open\n");

	    if (tpips_execution_mode) {
		if (db_get_current_workspace_name() != NULL)
		    close_workspace ();
		if (( $$ = open_workspace ($2)))
		{
		    main_module_name = get_first_main_module();
		    
		    if (!string_undefined_p(main_module_name)) {
			/* Ok, we got it ! Now we select it: */
			user_log("Main module PROGRAM \"%s\" selected.\n",
				 main_module_name);
			lazy_open_module(main_module_name);
		    }
		}
	    }
	}
	;

i_create: CREATE NAME /* workspace name */ filename_list /* fortran files */
	{
	    string main_module_name;
	    pips_debug(7,"reduce rule i_create\n");
	    
	    if (tpips_execution_mode) {
		if (workspace_exists_p($2))
		    pips_user_error
			("Workspace %s already exists. Delete it!\n", $2);
		else {
		  if(db_create_workspace ($2))
		  {
		      if(!create_workspace ($3))
		      {
			  /* string wname = db_get_current_workspace_name();*/
			  db_close_workspace();
			  (void) delete_workspace($2);
			pips_user_error("Could not create workspace %s\n", $2);
		      }

		      free($2);
		      main_module_name = get_first_main_module();
		      
		      if (!string_undefined_p(main_module_name)) {
			  /* Ok, we got it ! Now we select it: */
			  user_log("Main module PROGRAM \"%s\" selected.\n",
				   main_module_name);
			  lazy_open_module(main_module_name);
		      }
		      $$ = TRUE;
		  }
		  else {
		      pips_user_error("Cannot create directory for workspace,"
				      " check rights!\n");
		  }
		}
	    }
	    gen_array_full_free($3);
	}
	;

i_close: CLOSE
	{
	    pips_debug(7,"reduce rule i_close\n");

	    if (tpips_execution_mode) {
		if (db_get_current_workspace_name() != NULL) {
		    close_workspace ();
		    $$ = TRUE;
		}
		else {
		    pips_user_error("No workspace to close. "
				    "Open or create one!\n");
		    $$ = FALSE;
		}	
		$$ = TRUE;
	    }
	}
	;

i_delete: DELETE NAME /* workspace name */
	{
	    pips_debug(7,"reduce rule i_delete\n");

	    if (tpips_execution_mode) {
		string wname = db_get_current_workspace_name();
		if ((wname != NULL) && same_string_p(wname, $2)) {
		    user_error ("delete",
				"Close before delete: Workspace %s is open\n",
				wname);
		    $$ = FALSE;
		} else {
		    if(workspace_exists_p($2)) {
			if(delete_workspace ($2)) {
			    /* In case of problem, user_error() has been
			       called, so it is OK now !!*/
			    user_log ("Workspace %s deleted.\n", $2);
			    $$ = TRUE;
			}
			else {
			pips_user_error("Could not delete workspace %s\n", $2);
			}
		    }
		    else {
			pips_user_error("%s: No such workspace\n", $2);
		    }
		}
	    }
	}
	;

i_module: MODULE NAME /* module name */
	{
	    pips_debug(7,"reduce rule i_module\n");

	    if (tpips_execution_mode) {
		if (db_get_current_workspace_name()) {
		    lazy_open_module (strupper($2,$2));
		    $$ = TRUE;
		} else {
		    pips_user_error("No workspace open. "
				    "Open or create one!\n");
		    $$ = FALSE;
		}
	    }
	    free($2);
	}
	;

i_make:	MAKE resource_id
	{
	    bool result = TRUE;
	    pips_debug(7,"reduce rule i_make\n");

	    if (tpips_execution_mode) {
		string save_current_module_name = 
		    db_get_current_module_name()?
		    strdup(db_get_current_module_name()): NULL;

		MAPL(e, {
		    string mod_name = STRING(CAR(e));
		    
		    if (mod_name != NULL)
		    {
			if (safe_make ($2.the_name, mod_name) == FALSE) {
			    result = FALSE;
			    break;
			}
		    }
		    else
			pips_user_warning("Select a module first!\n");
		}, $2.the_owners);

		/* restore the initial current module, if there was one */
		if(save_current_module_name!=NULL) {
		    if (db_get_current_module_name())
			db_reset_current_module_name();
		    db_set_current_module_name(save_current_module_name);
		    free(save_current_module_name);
		}
	    }
	    $$ = result;
	    free_owner_content(&$2);
	}
	;

i_apply: APPLY rule_id
	{
	    bool result = TRUE;
	    /* keep track of the current module, if there is one */

	    pips_debug(7,"reduce rule i_apply\n");

	    if (tpips_execution_mode) {
		string save_current_module_name = 
		    db_get_current_module_name()?
		    strdup(db_get_current_module_name()): NULL;

		if(!db_get_current_workspace_name()) {
		    user_error("apply", "Open or create a workspace first!\n");
		}
	    
		MAP(STRING, mod_name, 
		{
		    if (mod_name != NULL)
		    {
			if (safe_apply ($2.the_name, mod_name) == FALSE) {
			    result = FALSE;
			    break;
			}
		    }
		    else
			pips_user_warning("Select a module first!\n");
		}, $2.the_owners);

		/* restore the initial current module, if there was one */
		if(save_current_module_name!=NULL) {
		    if (db_get_current_module_name())
			db_reset_current_module_name();
		    db_set_current_module_name(save_current_module_name);
		    free(save_current_module_name);
		}
	    }
	    $$ = result;
	    free_owner_content(&$2);
	}
	;

i_display: DISPLAY resource_id
	{
	    pips_debug(7,"reduce rule i_display\n");
	    if (tpips_execution_mode) {
		string pager;
		string save_current_module_name = 
		    db_get_current_module_name()?
		    strdup(db_get_current_module_name()): NULL;

		if(!db_get_current_workspace_name()) {
		    user_error("display",
			       "Open or create a workspace first!\n");
		}

		if ( (isatty(0)) || (!(pager = getenv("PAGER"))))
		    pager = CAT_COMMAND;
		
		MAPL(e, {
		    string mod_name = STRING(CAR(e));
		    
		    if (mod_name != NULL)
		    {
			string fname;
			lazy_open_module (mod_name);
			fname = build_view_file($2.the_name);
			if (fname == NULL)
			    user_error("display",
				       "Cannot build view file %s\n",
				       $2.the_name);
		    
			safe_system(concatenate(pager, " ", fname, NULL));
			free(fname);
		    }
		    else
			pips_user_warning("Select a module first!\n");

		}, $2.the_owners);

		/* restore the initial current module, if there was one */
		if(save_current_module_name!=NULL) {
		    if (db_get_current_module_name())
			db_reset_current_module_name();
		    db_set_current_module_name(save_current_module_name);
		    free(save_current_module_name);
		}
	    }
	    $$ = TRUE;
	    free_owner_content(&$2);
	}
	;

i_activate: ACTIVATE rulename
	{
	    pips_debug(7,"reduce rule i_activate\n");
	    if (tpips_execution_mode) {

		if(!db_get_current_workspace_name()) {
		    user_error("activate",
			       "Open or create a workspace first!\n");
		}
		
		user_log("Selecting rule: %s\n", $2);
		activate ($2);
		$$ = TRUE;
	    }
	}
	;

i_get: GET_PROPERTY propname
	{
	    pips_debug(7,"reduce rule i_get (%s)\n", $2);
	    
	    if (tpips_execution_mode) {
		fprint_property(stdout, $2);
		$$ = TRUE;
	    }
	}
	;

rulename: phasename
	;

filename_list: filename_list filename
	{
	    gen_array_append($1, $2);
	    $$ = $1;
	}
	| filename
	{ 
	    $$ = gen_array_make(0);
	    gen_array_append($$, $1);
	}
	;

filename: NAME
	;

resource_id: resourcename owner
	{
	    pips_debug(7,"reduce rule resource_id (%s)\n",$<name>2);
	    $$.the_name = $1;
	    $$.the_owners = $2;
	}
	;

rule_id: phasename owner
	{
	    pips_debug(7,"reduce rule rule_id (%s)\n",$1);
	    $$.the_name = $1;
	    $$.the_owners = $2;
	}
	;

owner:	OPENPAREN OWNER_ALL CLOSEPAREN
	{
	    int i;
	    list result = NIL;
	    pips_debug(7,"reduce rule owner (ALL)\n");

	    if (tpips_execution_mode) {
		gen_array_t modules = db_get_module_list();
		int nmodules = gen_array_nitems(modules);
		message_assert("some modules", nmodules>0);
		for(i=0; i<nmodules; i++) {
		    string n =  gen_array_item(modules, i);
		    result = CONS(STRING, strdup(n), result);
		}
		gen_array_full_free(modules);
		result = gen_nreverse(result);
		pips_assert("length ok", nmodules==gen_length(result));
		$$ = result;
	    }
	}
	| OPENPAREN OWNER_PROGRAM CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner (PROGRAM)\n");
	    if (tpips_execution_mode) {
		$$ =CONS(STRING, strdup(db_get_current_workspace_name()), NIL);
	    }
	}
	| OPENPAREN OWNER_MAIN CLOSEPAREN
	{
	    int i;
	    int number_of_main = 0;
	    
	    pips_debug(7,"reduce rule owner (MAIN)\n");
	    $$ = NIL;

	    if (tpips_execution_mode) {
		gen_array_t modules = db_get_module_list();
		int nmodules = gen_array_nitems(modules);
		message_assert("some modules", nmodules>0);

		for(i=0; i<nmodules; i++) {
		    string on = gen_array_item(modules, i);
		    entity mod = local_name_to_top_level_entity(on);

		    if (!entity_undefined_p(mod) && entity_main_module_p(mod))
		    {
			if (number_of_main)
			    pips_internal_error("More the one main\n");
			
			number_of_main++;
			$$ = CONS(STRING, strdup(on), NIL);
		    }
		}
		gen_array_full_free(modules);
	    }
	}
	| OPENPAREN OWNER_MODULE CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner (MODULE)\n");

	    if (tpips_execution_mode) {
		$$ = CONS(STRING, strdup(db_get_current_module_name()), NIL);
	    }
	}
	| OPENPAREN OWNER_CALLEES CLOSEPAREN
	{
	    callees called_modules;
	    list lcallees;

	    pips_debug(7,"reduce rule owner (CALLEES)\n");

	    if (tpips_execution_mode) {
		if (safe_make(DBR_CALLEES, db_get_current_module_name())
		    == FALSE) {
		  pips_internal_error("Cannot make callees for %s\n",
				      db_get_current_module_name());
		}

		called_modules = (callees) 
		db_get_memory_resource(DBR_CALLEES,
				       db_get_current_module_name(),TRUE);
		lcallees = callees_callees(called_modules);

		for($$=NIL; lcallees; POP(lcallees))
		    $$ = CONS(STRING, strdup(STRING(CAR(lcallees))), $$);
	    }
	}
	| OPENPAREN OWNER_CALLERS CLOSEPAREN
	{
	    callees caller_modules;
	    list lcallers;

	    pips_debug(7,"reduce rule owner (CALLERS)\n");
	    if (tpips_execution_mode) {
		if (!safe_make(DBR_CALLERS, db_get_current_module_name()))
		    pips_internal_error("Cannot make callers for %s\n",
					db_get_current_module_name());

		caller_modules = (callees) 
		    db_get_memory_resource(DBR_CALLERS,
					   db_get_current_module_name(),TRUE);

		lcallers = callees_callees(caller_modules);
		for($$=NIL; lcallers; POP(lcallers))
		    $$ = CONS(STRING, strdup(STRING(CAR(lcallers))), $$);
	    }
	}
	|
	OPENPAREN list_of_owner_name CLOSEPAREN
	{ $$ = $2; }
	|
	{
	    pips_debug(7,"reduce rule owner (none)\n");

	    if (tpips_execution_mode) {
		$$ = CONS(STRING, strdup(db_get_current_module_name()), NIL);
	    }
	}
	| { $$ = NIL;}
	;

list_of_owner_name: NAME
	{ $$ = CONS(STRING, $1, NIL); }
	| list_of_owner_name NAME
	{ $$ = CONS(STRING, $2, $1); }
	;

propname: NAME
	{
	    if (!property_name_p($1)) 
		yyerror("expecting a property name\n");
	    $$ = $1;
	}
	;

phasename: NAME
	{
	    if (!phase_name_p($1)) 
		yyerror("expecting a phase name\n");
	    $$ = $1;
	}
	;

resourcename: NAME
	{
	    if (!resource_name_p($1)) 
		yyerror("expecting a resource name\n");
	    $$ = $1;
	}
	;

%%
