
%token DOT
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
%token EXIT
%token OWNER_ALL
%token OWNER_PROGRAM
%token OWNER_MAIN
%token OWNER_MODULE
%token OWNER_CALLERS
%token OWNER_CALLEES
%token NAME
%token FILE_NAME
%token PATH_NAME
%token UNKNOW_CHAR

%type <status> lines
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
%type <status> i_exit
%type <status> i_display
%type <name> rulename
%type <name> filename
%type <name> filename_list
%type <rn> resource_id
%type <rn> rule_id
%type <owner> owner

%{
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"

#include "pipsdbm.h"
#include "ri-util.h"
#include "resources.h"
#include "phases.h"
#include "builder_map.h"
#include "properties.h"
#include "pipsmake.h"

#include "list.h"
#include "ri-util.h"
#include "top-level.h"
#include "tpips.h"

extern char yytext[];
extern FILE * yyin; 


%}

%union {
	int status;
	string name;
	res_or_rule rn;
	list owner;
}

%%

lines:
	lines line
	|
	{ $$ = TRUE; /* OK */ }
	;

line:
	instruction
	SEPARATOR
	{ $$ = $1; }
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
	| i_exit
	| error
      { $$ = FALSE; }
	;

i_open:
	OPEN
	NAME /* workspace name */
	{
		$$ = open_program (yytext);
	}
	;

i_create:
	CREATE
	NAME /* workspace name */
	filename_list /* list of fortran files */
	{
	    /* now no more than one file is OK */
		int the_argc;
		char *the_argv;

		the_argv = $3;
		the_argc = 1;

		db_create_workspace (yytext);
		create_program (&the_argc, &the_argv);

		$$ = TRUE;
	}
	;

i_close:
	CLOSE
	{
		close_program ();
		$$ = TRUE;
	}
	;

i_delete:
	DELETE
	NAME /* workspace name */
	{
		$$ = TRUE;
	}
	;

i_module:
	MODULE
	NAME /* module name */
	{
		lazy_open_module (yytext);
		$$ = TRUE;
	}
	;

i_make:
	MAKE
	resource_id
	{
		MAPL(e, {
			if (safe_make_p ($2.the_name, (string) e) == FALSE) {
				$$ = FALSE;
				break;
			}
		}, $2.the_owners);

		$$ = TRUE;
	}
	;

i_apply:
	APPLY
	rule_id
	{
		MAPL(e, {
			safe_apply ($2.the_name, (string) e);
		}, $2.the_owners);

		$$ = TRUE;
	}
	;

i_display:
	DISPLAY
	resource_id
	{
		MAPL(e, {

			lazy_open_module ((string) e);
			fprintf(stdout,"---  %s for %s\n", $2.the_name, (string) e);
			fputs(build_view_file($2.the_name), stdout);

		}, $2.the_owners);

		$$ = TRUE;
	}
	;

i_activate:
	ACTIVATE
	rulename
	{
		activate ($2);
		$$ = TRUE;
	}
	;

i_exit:
	EXIT
	{
		/* should check if program is closed ... */
		close_program ();
		return TRUE;
	}
	;

rulename:
	NAME { $$ = yytext;}
	;

filename:
	  NAME { $$ = yytext;}
	| FILE_NAME { $$ = yytext;}
	| PATH_NAME { $$ = yytext;}
	;	

filename_list:
	filename {$$ = $1;}
	;

resource_id:
	NAME
	{
		$$.the_name = yytext;
		$$.the_owners = CONS(STRING, db_get_current_module_name(),NIL);
	}
	|
	owner DOT NAME
	{
		$$.the_name = yytext;
		$$.the_owners = $1;
	}
	;

rule_id:
	resource_id
	;

owner:
	  OWNER_ALL
	{
		
	}
	| OWNER_PROGRAM
    {
		$$ = CONS(STRING, db_get_current_program_name (), NIL);
	}
	| OWNER_MAIN
    {
	    int nmodules = 0;
	    char *module_list[ARGS_LENGTH];
	    int i;
	    int number_of_main = 0;

	    db_get_module_list(&nmodules, module_list);
	    pips_assert("tpips:yacc", nmodules>0);
	    for(i=0; i<nmodules; i++) {
			string on = module_list[i];

			if (entity_main_module_p(
				local_name_to_top_level_entity(on)) == TRUE)
			{
				if (number_of_main)
				pips_error("build_real_resources", "More the one main\n");

				number_of_main++;
				$$ = CONS(STRING, on, NIL);
			}
	    }
	}
	| OWNER_MODULE
    {
		$$ = CONS(STRING, db_get_current_module_name (), NIL);
	}
	| OWNER_CALLEES
	{
	    callees called_modules;
	    list lcallees;
		list ps;
		list result = NIL;

	    if (safe_make_p(DBR_CALLEES, db_get_current_module_name())
			== FALSE)
			YYERROR;	

	    called_modules = (callees) 
		db_get_memory_resource(DBR_CALLEES, db_get_current_module_name(), TRUE);
	    lcallees = callees_callees(called_modules);

	    for (ps = lcallees; ps != NIL; ps = CDR(ps)) {
		string on = STRING(CAR(ps));

		result = gen_nconc(result, 
				   CONS(STRING, on, NIL));
	    }

		$$ = result;
	}
	| OWNER_CALLERS
	{
	    callees caller_modules;
	    list lcallers;
		list ps;
		list result = NIL;

	    if (safe_make_p(DBR_CALLERS, db_get_current_module_name())
			== FALSE)
			YYERROR;	

	    caller_modules = (callees) 
		db_get_memory_resource(DBR_CALLERS, db_get_current_module_name(), TRUE);
	    lcallers = callees_callees(caller_modules);

	    for (ps = lcallers; ps != NIL; ps = CDR(ps)) {
		string on = STRING(CAR(ps));

		result = gen_nconc(result, 
				   CONS(STRING, on, NIL));
	    }

		$$ = result;
	}
	| NAME
	{
		$$ = CONS(STRING, yytext, NIL);
	}
	;

%%

void yyerror(s)
char * s;
{
	fprintf(stderr, "[yacc] %s near %s\n", s, yytext);
	fprintf(stderr, "[yacc] unparsed text:\n");

	return;
}

