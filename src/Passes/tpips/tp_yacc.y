
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

/********************************************************** static functions */
static void print_property(char*,property);

/********************************************************** static variables */
static t_file_list the_file_list;
extern bool tpips_execution_mode;

#define YYERROR_VERBOSE 1 /* MUCH better error messages with bison */

extern int yylex(void);
extern void yyerror(char *);

%}

%union {
    int status;
    string name;
    res_or_rule rn;
    list owner;
    t_file_list *args;
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
/*
	    free (yylval.name);
*/
	}
	;

i_create:
	CREATE
	sep_list
	WORKSPACE /* workspace name */
	{
	    $<name>$ = yylval.name;
	    the_file_list.argc = 0;
	}
	sep_list
	filename_list /* list of fortran files */
	opt_sep_list
	{
	    string main_module_name;
	    
	    debug(7,"yyparse","reduce rule i_create\n");
	    
	    if (tpips_execution_mode) {
		if (workspace_exists_p($<name>4))
		    user_error
			("create", "Workspace %s already exists. Delete it!\n",
			 $<name>4);
		else {
		  if(db_create_workspace ((string) $<name>4))
		  {
		    if(!create_workspace (&the_file_list.argc, 
					  the_file_list.argv)) 
		    {
			/* string wname = db_get_current_workspace_name();*/
			db_close_workspace();
			delete_workspace($<name>4);
			user_error("create", "Could not create workspace"
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
		    user_error("create", "Cannot create directory for workspace,"
				    " check rights!\n");
		  }
		}
	    }
/*
	    while (the_file_list.argc--)
		free (the_file_list.argv[the_file_list.argc]);
	    free ($<name>4);
*/
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
		    user_error ("close",
				"No workspace to close. Open or create one!\n");
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

	    debug(7,"yyparse","reduce rule i_module\n");

	    if (tpips_execution_mode) {
		if (db_get_current_workspace_name()) {
		    lazy_open_module (strupper(t,t));
		    $$ = TRUE;
		} else {
		    user_error ("module",
				"No workspace open. Open or create one!\n");
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
		    db_get_current_module_name() == NULL? 
		    NULL : strdup(db_get_current_module_name());

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
		    {
			user_warning("make", "Select a module first!\n");
		    }
		}, $3.the_owners);

		db_set_current_module_name(save_current_module_name);
		if(save_current_module_name!=NULL)
		    free(save_current_module_name);
	    }
	    $$ = result;
/*	    free ($3.the_name);
	    gen_free_list ($3.the_owners);
*/
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
		    db_get_current_module_name() == NULL? 
		    NULL : strdup(db_get_current_module_name());

		if(db_get_current_workspace()==database_undefined) {
		    user_error("apply", "Open or create a workspace first!\n");
		}
	    
		MAPL(e, {
		    string mod_name = STRING(CAR(e));
		    
		    if (mod_name != NULL)
		    {
			if (safe_apply ($3.the_name, mod_name) == FALSE) {
			    result = FALSE;
			    break;
			}
		    }
		    else
		    {
			user_warning("apply", "Select a module first!\n");
		    }
		}, $3.the_owners);
		/* restore the initial current module, if there was one */
		db_set_current_module_name(save_current_module_name);
		if(save_current_module_name!=NULL)
		    free(save_current_module_name);
	    }
	    $$ = result;
/*	    free ($3.the_name);
	    gen_free_list ($3.the_owners);
*/
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
		    db_get_current_module_name() == NULL? 
		    NULL : strdup(db_get_current_module_name());

		if(db_get_current_workspace()==database_undefined) {
		    user_error("display",
			       "Open or create a workspace first!\n");
		}

		if ( (isatty(0)) || (!(pager = getenv("PAGER"))))
		    pager = CAT_COMMAND;
		
		MAPL(e, {
		    string file;
		    string mod_name = STRING(CAR(e));
		    
		    if (mod_name != NULL)
		    {
			lazy_open_module (mod_name);
			file = build_view_file($3.the_name);
			if (file == NULL)
			    user_error("display",
				       "Cannot build view file %s\n",
				       $3.the_name);
		    
			safe_system(concatenate(pager, " ", file, NULL));
		    }
		    else
		    {
			user_warning("display", "Select a module first!\n");
		    }

		}, $3.the_owners);
		db_set_current_module_name(save_current_module_name);
		if(save_current_module_name!=NULL)
		    free(save_current_module_name);
	    }
	    $$ = TRUE;
/*
	    free ($3.the_name);
	    gen_free_list ($3.the_owners);
*/
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

		if(db_get_current_workspace()==database_undefined) {
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
	    property p;

	    debug(7,"yyparse","reduce rule i_get (%s)\n",
		  yylval.name);
	    
	    if (tpips_execution_mode) {
		p = get_property (yylval.name);
	    
		print_property(yylval.name, p);
		$$ = TRUE;
	    }
/*
	    free (yylval.name);
*/
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
	    debug(7,"yyparse","reduce rule filename_list (%s)\n", $1);

	    if (the_file_list.argc < FILE_LIST_MAX_LENGTH) {
		the_file_list.argv[the_file_list.argc] = $3;
		the_file_list.argc++;
	    } else {
	      pips_error("tpips",
			 "Too many files - Resize FILE_LIST_MAX_LENGTH\n");
	    }
	    
	}
	|
	filename
	/* opt_sep_list */
	{ /* the opt_sep_list enables trailing blanks... */
	    debug(7,"yyparse","reduce rule filename_list (%s)\n", $1);

	    if (the_file_list.argc < FILE_LIST_MAX_LENGTH) {
		the_file_list.argv[the_file_list.argc] = $1;
		the_file_list.argc++;
	    } else {
	      pips_error("tpips",
			 "Too many files (2) - Resize FILE_LIST_MAX_LENGTH\n");
	    }
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
	    debug(7,"yyparse","reduce rule resource_id (%s)\n",$<name>2);

	    $$.the_name = $<name>2;
	    $$.the_owners = $3;
	}
	;

rule_id:
	PHASENAME
	{ $<name>$ = yylval.name;}
	owner
	{
	    debug(7,"yyparse","reduce rule rule_id (%s)\n",$<name>2);

	    $$.the_name = $<name>2;
	    $$.the_owners = $3;
	}
	;

owner:
	OPENPAREN
	OWNER_ALL
	CLOSEPAREN
	{
	    int nmodules = 0;
	    char *module_list[ARGS_LENGTH];
	    int i;
	    list result = NIL;

	    debug(7,"yyparse","reduce rule owner (ALL)\n");

	    if (tpips_execution_mode) {
		db_get_module_list(&nmodules, module_list);
		message_assert("yyparse", nmodules>0);
		for(i=0; i<nmodules; i++) {
		    string on = module_list[i];

		    result = gen_nconc(result, 
				       CONS(STRING, on, NIL));
		}
		$$ = result;
	    }
	}
	|
	OPENPAREN
	OWNER_PROGRAM
	CLOSEPAREN
	{
	    debug(7,"yyparse","reduce rule owner (PROGRAM)\n");

	    if (tpips_execution_mode) {
		$$ = CONS(STRING, db_get_current_workspace_name (), NIL);
	    }
	}
	|
	OPENPAREN
	OWNER_MAIN
	CLOSEPAREN
	{
	    int nmodules = 0;
	    char *module_list[ARGS_LENGTH];
	    int i;
	    int number_of_main = 0;

	    debug(7,"yyparse","reduce rule owner (MAIN)\n");

	    if (tpips_execution_mode) {
		db_get_module_list(&nmodules, module_list);
		message_assert("yyparse", nmodules>0);
		for(i=0; i<nmodules; i++) {
		    string on = module_list[i];

		    if (entity_main_module_p
			(local_name_to_top_level_entity(on)) == TRUE)
		    {
			if (number_of_main)
			    pips_error("build_real_resources",
				       "More the one main\n");
			
			number_of_main++;
			$$ = CONS(STRING, on, NIL);
		    }
		}
	    }
	}
	|
	OPENPAREN
	OWNER_MODULE
	CLOSEPAREN
	{
	    debug(7,"yyparse","reduce rule owner (MODULE)\n");

	    if (tpips_execution_mode) {
		$$ = CONS(STRING, db_get_current_module_name (), NIL);
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
		    result = gen_nconc(result, CONS(STRING, on, NIL));
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
		    result = gen_nconc(result, CONS(STRING, on, NIL));
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
		$$ = CONS(STRING, db_get_current_module_name (), NIL);
	    }
	}
	;

list_of_owner_name:
	OWNER_NAME
	{ $$ = (list) yylval.name; }
	COMMA
	list_of_owner_name
	{
	    debug(7,"yyparse",
		  "reduce rule owner list (name = %s)\n",$<name>2);

	    if (tpips_execution_mode) {
		char *c = $<name>2;
		strupper (c, c);
		$$ = gen_nconc($4,CONS(STRING, c, NIL));
	    }
	}
	|
	OWNER_NAME
	{ $$ = (list) yylval.name; }
	CLOSEPAREN
	{
	    debug(7,"yyparse","reduce rule owner list(name = %s)\n",$<name>2);

	    if (tpips_execution_mode) {
		char *c = $<name>2;
		strupper (c, c);
		$$ = CONS(STRING, c, NIL);
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
	    debug(7,"yyparse","reduce separator list\n");
	    $$ = 0;
	}
	;

%%

void close_workspace_if_opened()
{
    if (db_get_current_workspace_name() != NULL)
	close_workspace ();
}

static void print_property(char* pname, property p)
{
    switch (property_tag(p))
    {
    case is_property_bool:
    {
		fprintf (stderr, "%s = %s\n",
			pname,
			get_bool_property (pname) == TRUE ?
			"TRUE" : "FALSE");
		break;
    }
    case is_property_int:
    {
		fprintf (stderr, "%s = %d\n",
			pname,
			get_int_property (pname));
		break;
    }
    case is_property_string:
    {
		fprintf (stderr, "%s = %s\n",
			pname,
			get_string_property (pname));
		break;
    }
    }
}
