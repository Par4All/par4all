
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
%token SET
%token GET
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
%token OPENBRACE
%token COMMA
%token CLOSEBRACE
%token DOT
%token PROPNAME
%token FILE_NAME
%token RESOURCENAME
%token PHASENAME
%token UNKNOW_CHAR

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
%type <status> i_set
%type <status> i_get
%type <name> rulename
%type <name> filename_list
%type <rn> resource_id
%type <rn> rule_id
%type <owner> owner
%type <owner> list_of_owner_name
%type <status> sep_list
%type <status> opt_sep_list

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

#define MORE_COMMAND "less "

static void print_property(char*,property);

%}

%union {
	int status;
	string name;
	res_or_rule rn;
	list owner;
}

%%

line:
	opt_sep_list
	instruction
	opt_sep_list
	{ $$ = $2; return TRUE;}
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
	| i_set
	| i_get
	| error
      { $$ = FALSE; }
	;

i_open:
	OPEN
	sep_list
	WORKSPACE /* workspace name */
	{
		string main_module_name;

		debug(7,"tp_parse","reduce rule i_open\n");
		if (db_get_current_program_name() != NULL)
			close_program ();
		$$ = open_program (yylval.name);
      
		main_module_name = get_first_main_module();

		if (!string_undefined_p(main_module_name)) {
			/* Ok, we got it ! Now we select it: */
			user_log("Main module PROGRAM \"%s\" found.\n", main_module_name);
			lazy_open_module(main_module_name);
		}
		$$ = TRUE;
	}
	;

i_create:
	CREATE
	sep_list
	WORKSPACE /* workspace name */
	{ $$ = (int) yylval.name;}
	sep_list
	filename_list /* list of fortran files */
	{
	    /* now no more than one file is OK */
		int the_argc;
		char *the_argv;
		string main_module_name;

		the_argv = $6;
		the_argc = 1;

		debug(7,"tp_parse","reduce rule i_create\n");
		db_create_workspace ((string) $<name>4);
		free($<name>4);
		create_program (&the_argc, &the_argv);

		main_module_name = get_first_main_module();

		if (!string_undefined_p(main_module_name)) {
			/* Ok, we got it ! Now we select it: */
			user_log("Main module PROGRAM \"%s\" found.\n", main_module_name);
			lazy_open_module(main_module_name);
		}		
		$$ = TRUE;
	}
	;

i_close:
	CLOSE
	{
		debug(7,"tp_parse","reduce rule i_close\n");
		if (db_get_current_program_name() != NULL) {
			close_program ();
			$$ = TRUE;
		} else {
			user_warning ("tp_parse","No workspace to close\n");
			$$ = FALSE;
		}	
		$$ = TRUE;
	}
	;

i_delete:
	DELETE
	sep_list
	WORKSPACE /* workspace name */
	{
		debug(7,"tp_parse","reduce rule i_delete\n");
		$$ = TRUE;
	}
	;

i_module:
	MODULE
	sep_list
	WORKSPACE /* module name */
	{
		char *t = yylval.name;

		debug(7,"tp_parse","reduce rule i_module\n");
		lazy_open_module (strupper(t,t));
		free(t);
		$$ = TRUE;
	}
	;

i_make:
	MAKE
	sep_list
	resource_id
	{
		debug(7,"tp_parse","reduce rule i_make\n");
		MAPL(e, {
			if (safe_make_p ($3.the_name, STRING(CAR(e))) == FALSE) {
				$$ = FALSE;
				break;
			}
		}, $3.the_owners);

		$$ = TRUE;
	}
	;

i_apply:
	APPLY
	sep_list
	rule_id
	{
		debug(7,"tp_parse","reduce rule i_apply\n");
		MAPL(e, {
			safe_apply ($3.the_name, STRING(CAR(e)));
		}, $3.the_owners);

		$$ = TRUE;
	}
	;

i_display:
	DISPLAY
	sep_list
	resource_id
	{
		debug(7,"tp_parse","reduce rule i_display\n");
	
		MAPL(e, {
			lazy_open_module (STRING(CAR(e)));
			fprintf(stdout,"---  %s for %s\n",
					$3.the_name,
					STRING(CAR(e)));
			system(concatenate(MORE_COMMAND,
					build_view_file($3.the_name),
					NULL));

		}, $3.the_owners);

		$$ = TRUE;
	}
	;

i_activate:
	ACTIVATE
	sep_list
	rulename
	{
		debug(7,"tp_parse","reduce rule i_activate\n");
		activate ($3);
		$$ = TRUE;
	}
	;

i_set:
	SET
	sep_list
	PROPNAME
	{ $$ = yylval.name;}
	sep_list
	WORKSPACE
	{
		property p;

		p = get_property ($<name>4);

		strupper (yylval.name,yylval.name);

		debug(7,"tp_parse","reduce rule i_set(%s,%s)\n",
		      $<name>4,yylval.name);

		switch (property_tag(p))
		{
		case is_property_bool:
			{
				if (!strcmp ("TRUE",yylval.name))
					set_bool_property ($<name>4, TRUE);
				else if (!strcmp ("FALSE",yylval.name))
					set_bool_property ($<name>4, FALSE);
				else {
					yyerror ("type mismatch");
					return FALSE;
				    }
				break;
			}
		case is_property_int:
			{
				char **ptr;
				long l;

				l = strtol (yylval.name, ptr, 0);
				if (**ptr != '\0') {
					yyerror ("type mismatch");
					return FALSE;
				} else
					set_int_property($<name>4, (int) l);
				break;
			}
		case is_property_string:
			{
				set_string_property($<name>4, yylval.name);
				break;
			}
		}
		print_property($<name>4,p);
		$$ = TRUE;
	}


i_get:
	GET
	sep_list
	PROPNAME
	{
		property p;

		debug(7,"tp_parse","reduce rule i_get (%s)\n",
		      yylval.name);

		strupper (yylval.name,yylval.name);
		p = get_property (yylval.name);

		print_property(yylval.name, p);
		$$ = TRUE;
	}


rulename:
	PHASENAME
	{
 		debug(7,"tp_parse","reduce rule rulename (%s)\n",yylval.name);
		$$ = yylval.name;
	}
	;

filename_list:
	FILE_NAME
	{
 		debug(7,"tp_parse","reduce rule filename_list (%s)\n",yylval.name);
		$$ = yylval.name;
	}
	;

resource_id:
	RESOURCENAME
	{
 		debug(7,"tp_parse","reduce rule resource_id (%s)\n",yylval.name);
		$$.the_name = yylval.name;
		$$.the_owners=CONS(STRING,db_get_current_module_name(),NIL);
	}
	|
	owner RESOURCENAME
	{
 		debug(7,"tp_parse","reduce rule resource_id (%s)\n",yylval.name);
		$$.the_name = yylval.name;
		$$.the_owners = $1;
	}
	;

rule_id:
	PHASENAME
	{
 		debug(7,"tp_parse","reduce rule rule_id (%s)\n",yylval.name);
		$$.the_name = yylval.name;
		$$.the_owners=CONS(STRING,db_get_current_module_name(),NIL);
	}
	|
	owner PHASENAME
	{
 		debug(7,"tp_parse","reduce rule rule_id (%s)\n",yylval.name);
		$$.the_name = yylval.name;
		$$.the_owners = $1;
	}
	;

owner:
	  OWNER_ALL
	{
	    int nmodules = 0;
	    char *module_list[ARGS_LENGTH];
	    int i;
		list result = NIL;

 		debug(7,"tp_parse","reduce rule owner (ALL)\n");
	    db_get_module_list(&nmodules, module_list);
	    pips_assert("tpips:yacc", nmodules>0);
	    for(i=0; i<nmodules; i++) {
		string on = module_list[i];

		result = gen_nconc(result, 
				   CONS(STRING, on, NIL));
	    }

		$$ = result;		
	}
	| OWNER_PROGRAM
    {
 		debug(7,"tp_parse","reduce rule owner (PROGRAM)\n");
		$$ = CONS(STRING, db_get_current_program_name (), NIL);
	}
	| OWNER_MAIN
    {
	    int nmodules = 0;
	    char *module_list[ARGS_LENGTH];
	    int i;
	    int number_of_main = 0;

 		debug(7,"tp_parse","reduce rule owner (MAIN)\n");
	    db_get_module_list(&nmodules, module_list);
	    pips_assert("tpips:yacc", nmodules>0);
	    for(i=0; i<nmodules; i++) {
			string on = module_list[i];

			if (entity_main_module_p(
				local_name_to_top_level_entity(on)) == TRUE)
			{
				if (number_of_main)
				pips_error("build_real_resources",
					   "More the one main\n");

				number_of_main++;
				$$ = CONS(STRING, on, NIL);
			}
	    }
	}
	| OWNER_MODULE
    {
 		debug(7,"tp_parse","reduce rule owner (MODULE)\n");
		$$ = CONS(STRING, db_get_current_module_name (), NIL);
	}
	| OWNER_CALLEES
	{
	    callees called_modules;
	    list lcallees;
		list ps;
		list result = NIL;

 		debug(7,"tp_parse","reduce rule owner (CALLEES)\n");
	    if (safe_make_p(DBR_CALLEES, db_get_current_module_name())
			== FALSE)
			YYERROR;	

	    called_modules = (callees) 
		db_get_memory_resource(DBR_CALLEES, db_get_current_module_name(),TRUE);
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

 		debug(7,"tp_parse","reduce rule owner (CALLERS)\n");
	    if (safe_make_p(DBR_CALLERS, db_get_current_module_name())
			== FALSE)
			YYERROR;	

	    caller_modules = (callees) 
		db_get_memory_resource(DBR_CALLERS, db_get_current_module_name(),TRUE);
	    lcallers = callees_callees(caller_modules);

	    for (ps = lcallers; ps != NIL; ps = CDR(ps)) {
		string on = STRING(CAR(ps));

		result = gen_nconc(result, 
				   CONS(STRING, on, NIL));
	    }

		$$ = result;
	}
	|
	OWNER_NAME
	{ $$ = yylval.name; }
	DOT
	{
		char *c = $<name>2;

 		debug(7,"tp_parse","reduce rule owner (name = %s)\n",c);
		strupper (c, c);
		$$ = CONS(STRING, c, NIL);
	}
	| OPENBRACE
	list_of_owner_name
	{ $$ = $2; }
	;

list_of_owner_name:
	OWNER_NAME
	{ $$ = yylval.name; }
	COMMA
	list_of_owner_name
	{
		char *c = $<name>2;
 		debug(7,"tp_parse",
			  "reduce rule owner list (name = %s)\n",c);

		c[strlen(c) - 1] = '\0'; /* skip the comma */
		$$ = gen_nconc($4,CONS(STRING, c, NIL));
	}
	|
	OWNER_NAME
	{ $$ = yylval.name; }
	CLOSEBRACE
	DOT
	{
		char *c = $<name>2;

 		debug(7,"tp_parse","reduce rule owner list(name = %s)\n",c);
		strupper (c, c);
		$$ = CONS(STRING, c, NIL);
	}
	;

opt_sep_list:
	sep_list {$$ = 0;}
	|
	{$$ = 0;}
	;

sep_list:
	sep_list SEPARATOR {$$ = $1;}
	|
	SEPARATOR {$$ = 0;}
	;

%%

void yyerror(s)
char * s;
{
    fprintf(stderr, "[yyparse] %s near %s\n", s, yytext);
    tpips_lex_print_pos(stderr);
    return;
}

void close_workspace_if_opened()
{
    if (db_get_current_program_name() != NULL)
	close_program ();
}	

static void print_property(char* pname, property p)
{
    switch (property_tag(p))
    {
    case is_property_bool:
    {
		printf ("%s = %s\n",
			pname,
			get_bool_property (pname) == TRUE ?
			"TRUE" : "FALSE");
		break;
    }
    case is_property_int:
    {
		printf ("%s = %d\n",
			pname,
			get_int_property (pname));
		break;
    }
    case is_property_string:
    {
		printf ("%s = %s\n",
			pname,
			get_string_property (pname));
		break;
    }
    }
}
