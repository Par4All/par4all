
%token SEPARATOR
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
%token OWNER_NAME
%token OWNER_ALL
%token OWNER_PROGRAM
%token OWNER_MAIN
%token OWNER_MODULE
%token OWNER_CALLERS
%token OWNER_CALLEES
%token MNAME
%token WORKSPACE
%token OPENPAREN
%token COMMA
%token CLOSEPAREN
%token PROPNAME
%token FILE_NAME
%token RESOURCENAME
%token PHASENAME
%token UNKNOWN_CHAR
%token SETVALUE
%token IDENT

%type <name>   IDENT
%type <status> line
%type <status> instruction
%type <status> i_open
%type <status> i_create
%type <status> i_close
%type <status> i_delete
%type <status> i_module
%type <status> i_make
%type <status> i_apply
%type <status> i_activate
%type <status> i_display
%type <status> i_get
%type <name> rulename
%type <name> filename
%type <args> filename_list
%type <rn> resource_id
%type <rn> rule_id
%type <owner> owner
%type <owner> list_of_owner_name
%type <status> sep_list
%type <status> opt_sep_list

%{
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>

#include "genC.h"

#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"

#include "ri-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "phases.h"
#include "builder_map.h"
#include "properties.h"
#include "pipsmake.h"

#include "top-level.h"
#include "tpips.h"

extern FILE * yyin; 

/* Default comand to print a file (if $PAGER is not set) */
#define CAT_COMMAND "cat"

/********************************************************** static variables */
static gen_array_t the_file_list = gen_array_undefined;
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


%}

%union {
    int status;
    string name;
    res_or_rule rn;
    list owner;
    gen_array_t *args;
}

%%

line:
	instruction
	{ $$ = $1; return $1;}
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
	| error {$$ = FALSE;}
	;

i_open:
	OPEN
	sep_list
	WORKSPACE /* workspace name */
	opt_sep_list
	{
	    string main_module_name;

	    debug(7,"yyparse","reduce rule i_open\n");

	    if (tpips_execution_mode) {
		if (db_get_current_workspace_name() != NULL)
		    close_workspace ();
		if (( $$ = open_workspace (yylval.name)))
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

i_create:
	CREATE
	sep_list
	WORKSPACE /* workspace name */
	{
	    $<name>$ = yylval.name;
	    if (!gen_array_undefined_p(the_file_list)) 
		gen_array_full_free(the_file_list);
	    the_file_list = gen_array_make(0);
	}
	sep_list
	filename_list /* list of fortran files */
	opt_sep_list
	{
	    string main_module_name;
	    
	    debug(7,"yyparse","reduce rule i_create\n");
	    
	    if (tpips_execution_mode) {
		if (workspace_exists_p($<name>4))
		    pips_user_error
			("Workspace %s already exists. Delete it!\n",
			 $<name>4);
		else {
		  if(db_create_workspace ((string) $<name>4))
		  {
		    if(!create_workspace (the_file_list))
		    {
			/* string wname = db_get_current_workspace_name();*/
			db_close_workspace();
			(void) delete_workspace($<name>4);
			pips_user_error("Could not create workspace"
					" %s\n", $<name>4);
		    }

		    free($<name>4);
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
	    if (!gen_array_undefined_p(the_file_list)) {
		gen_array_full_free(the_file_list);
		the_file_list = gen_array_undefined;
	    }
	}
	;

i_close:
	CLOSE
	opt_sep_list
	{
	    debug(7,"yyparse","reduce rule i_close\n");

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

i_delete:
	DELETE
	sep_list
	WORKSPACE /* workspace name */
	opt_sep_list
	{
	    char *t = yylval.name;

	    debug(7,"yyparse","reduce rule i_delete\n");

	    if (tpips_execution_mode) {
		string wname = db_get_current_workspace_name();
		if ((wname != NULL) && same_string_p(wname, t)) {
		    user_error ("delete",
				"Close before delete: Workspace %s is open\n",
				wname);
		    $$ = FALSE;
		} else {
		    if(workspace_exists_p(t)) {
			if(delete_workspace (t)) {
			    /* In case of problem, user_error() has been
			       called, so it is OK now !!*/
			    user_log ("Workspace %s deleted.\n", t);
			    $$ = TRUE;
			}
			else {
			    user_error("delete",
				       "Could not delete workspace %s\n", t);
			}
		    }
		    else {
			    user_error("delete",
				       "%s: No such workspace\n", t);
		    }
		}
	    }
	}
	;

i_module:
	MODULE
	sep_list
	WORKSPACE /* module name */
	opt_sep_list
	{
	    char *t = yylval.name;

	    pips_debug(7,"reduce rule i_module\n");

	    if (tpips_execution_mode) {
		if (db_get_current_workspace_name()) {
		    lazy_open_module (strupper(t,t));
		    $$ = TRUE;
		} else {
		    pips_user_error("No workspace open. "
				    "Open or create one!\n");
		    $$ = FALSE;
		}
	    }
	}
	;

i_make:
	MAKE
	sep_list
	resource_id
	opt_sep_list
	{
	    bool result = TRUE;

	    debug(7,"yyparse","reduce rule i_make\n");

	    if (tpips_execution_mode) {
		string save_current_module_name = 
		    db_get_current_module_name()?
		    strdup(db_get_current_module_name()): NULL;

		MAPL(e, {
		    string mod_name = STRING(CAR(e));
		    
		    if (mod_name != NULL)
		    {
			if (safe_make ($3.the_name, mod_name) == FALSE) {
			    result = FALSE;
			    break;
			}
		    }
		    else
			user_warning("make", "Select a module first!\n");
		}, $3.the_owners);

		/* restore the initial current module, if there was one */
		if(save_current_module_name!=NULL) {
		    if (db_get_current_module_name())
			db_reset_current_module_name();
		    db_set_current_module_name(save_current_module_name);
		    free(save_current_module_name);
		}
	    }
	    $$ = result;
	    free_owner_content(&$3);
	}
	;

i_apply:
	APPLY
	sep_list
	rule_id
	opt_sep_list
	{
	    bool result = TRUE;
	    /* keep track of the current module, if there is one */

	    debug(7,"yyparse","reduce rule i_apply\n");

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
			if (safe_apply ($3.the_name, mod_name) == FALSE) {
			    result = FALSE;
			    break;
			}
		    }
		    else
			pips_user_warning("Select a module first!\n");
		}, $3.the_owners);

		/* restore the initial current module, if there was one */
		if(save_current_module_name!=NULL) {
		    if (db_get_current_module_name())
			db_reset_current_module_name();
		    db_set_current_module_name(save_current_module_name);
		    free(save_current_module_name);
		}
	    }
	    $$ = result;
	    free_owner_content(&$3);
	}
	;

i_display:
	DISPLAY
	sep_list
	resource_id
	opt_sep_list
	{
	    debug(7,"yyparse","reduce rule i_display\n");

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
			fname = build_view_file($3.the_name);
			if (fname == NULL)
			    user_error("display",
				       "Cannot build view file %s\n",
				       $3.the_name);
		    
			safe_system(concatenate(pager, " ", fname, NULL));
			free(fname);
		    }
		    else
			pips_user_warning("Select a module first!\n");

		}, $3.the_owners);

		/* restore the initial current module, if there was one */
		if(save_current_module_name!=NULL) {
		    if (db_get_current_module_name())
			db_reset_current_module_name();
		    db_set_current_module_name(save_current_module_name);
		    free(save_current_module_name);
		}
	    }
	    $$ = TRUE;
	    free_owner_content(&$3);
	}
	;

i_activate:
	ACTIVATE
	sep_list
	rulename
	opt_sep_list
	{
	    debug(7,"yyparse","reduce rule i_activate\n");
	    if (tpips_execution_mode) {

		if(!db_get_current_workspace_name()) {
		    user_error("activate",
			       "Open or create a workspace first!\n");
		}
		
		user_log("Selecting rule: %s\n", $3);
		activate ($3);
		$$ = TRUE;
	    }
	}
	;

i_get:
	GET_PROPERTY
	sep_list
	PROPNAME
	opt_sep_list
	{
	    debug(7,"yyparse","reduce rule i_get (%s)\n",
		  yylval.name);
	    
	    if (tpips_execution_mode) {
		fprint_property(stdout, yylval.name);
		$$ = TRUE;
	    }
	}
	;

rulename:
	PHASENAME
	{
	    debug(7,"yyparse","reduce rule rulename (%s)\n",yylval.name);
	    $$ = yylval.name;
	}
	;

filename_list:
	filename_list
	sep_list
	filename
	{
	    pips_debug(7,"reduce rule filename_list\n");
	    gen_array_append(the_file_list, $3);
	}
	|
	filename
	/* opt_sep_list */
	{ /* the opt_sep_list enables trailing blanks... */
	    pips_debug(7,"reduce rule filename_list (%s)\n", $1);
	    gen_array_append(the_file_list, $1);
	}
	;

filename:
	FILE_NAME
	{ $$ = yylval.name;}
	;

resource_id:
	RESOURCENAME
	{ $<name>$ = yylval.name;}
	owner
	{
	    pips_debug(7,"reduce rule resource_id (%s)\n",$<name>2);

	    $$.the_name = $<name>2;
	    $$.the_owners = $3;
	}
	;

rule_id:
	PHASENAME
	{ $<name>$ = yylval.name;}
	owner
	{
	    pips_debug(7,"reduce rule rule_id (%s)\n",$<name>2);

	    $$.the_name = $<name>2;
	    $$.the_owners = $3;
	}
	;

owner:
	OPENPAREN
	OWNER_ALL
	CLOSEPAREN
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
		pips_assert("length ok", nmodules==gen_length(result));
		$$ = result;
	    }
	}
	|
	OPENPAREN
	OWNER_PROGRAM
	CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner (PROGRAM)\n");
	    if (tpips_execution_mode) {
		$$ =CONS(STRING, strdup(db_get_current_workspace_name()), NIL);
	    }
	}
	|
	OPENPAREN
	OWNER_MAIN
	CLOSEPAREN
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
	|
	OPENPAREN
	OWNER_MODULE
	CLOSEPAREN
	{
	    debug(7,"yyparse","reduce rule owner (MODULE)\n");

	    if (tpips_execution_mode) {
		$$ = CONS(STRING, strdup(db_get_current_module_name()), NIL);
	    }
	}
	|
	OPENPAREN
	OWNER_CALLEES
	CLOSEPAREN
	{
	    callees called_modules;
	    list lcallees;
	    list ps;
	    list result = NIL;

	    debug(7,"yyparse","reduce rule owner (CALLEES)\n");

	    if (tpips_execution_mode) {
		if (safe_make(DBR_CALLEES, db_get_current_module_name())
		    == FALSE) {
		  pips_error("ana_syn.y", "Cannot make callees for %s\n",
			     db_get_current_module_name());
		}

		called_modules = (callees) 
		db_get_memory_resource(DBR_CALLEES,
				       db_get_current_module_name(),TRUE);
		lcallees = callees_callees(called_modules);

		for (ps = lcallees; ps != NIL; ps = CDR(ps)) {
		    string on = STRING(CAR(ps));
		    result = gen_nconc(result, CONS(STRING, strdup(on), NIL));
		}
		$$ = result;
	    }
	}
	|
	OPENPAREN
	OWNER_CALLERS
	CLOSEPAREN
	{
	    callees caller_modules;
	    list lcallers;
	    list ps;
	    list result = NIL;

	    debug(7,"yyparse","reduce rule owner (CALLERS)\n");
	    if (tpips_execution_mode) {
		if (safe_make(DBR_CALLERS, db_get_current_module_name())
		    == FALSE) {
		  pips_error("ana_syn.y", "Cannot make callers for %s\n",
			     db_get_current_module_name());
		}

		caller_modules = (callees) 
		    db_get_memory_resource(DBR_CALLERS,
					   db_get_current_module_name(),TRUE);
		lcallers = callees_callees(caller_modules);

		for (ps = lcallers; ps != NIL; ps = CDR(ps)) {
		    string on = STRING(CAR(ps));
		    result = gen_nconc(result, CONS(STRING, strdup(on), NIL));
		}
		$$ = result;
	    }
	}
	|
	OPENPAREN
	list_of_owner_name
	{ $$ = $2; }
	|
	{
	    debug(7,"yyparse","reduce rule owner (none)\n");

	    if (tpips_execution_mode) {
		$$ = CONS(STRING, strdup(db_get_current_module_name()), NIL);
	    }
	}
	;

list_of_owner_name:
	OWNER_NAME
	{ $$ = (list) yylval.name; }
	COMMA
	list_of_owner_name
	{
	    pips_debug(7, "reduce rule owner list (name = %s)\n",$<name>2);

	    if (tpips_execution_mode) {
		char *c = $<name>2;
		strupper (c, c);
		$$ = CONS(STRING, strdup(c), $4);
	    }
	}
	|
	OWNER_NAME
	{ $$ = (list) yylval.name; }
	CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner list(name = %s)\n",$<name>2);

	    if (tpips_execution_mode) {
		char *c = $<name>2;
		strupper (c, c);
		$$ = CONS(STRING, strdup(c), NIL);
	    }
	}
	;

opt_sep_list:
	sep_list {$$ = 0;}
	| {$$ = 0;}
	;

sep_list:
	sep_list
	SEPARATOR
	{$$ = $1;}
	|
	SEPARATOR 
	{
	    pips_debug(7,"reduce separator list\n");
	    $$ = 0;
	}
	;

%%
