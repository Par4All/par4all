/* $Id$
 *
 * Each full syntax looks for ENDOFLINE so as to check that the right
 * number of arguments is matched.
 *
 * $Log: tp_yacc.y,v $
 * Revision 1.70  1997/12/12 12:25:24  coelho
 * guarded unknown rule to behave like a shell...
 *
 * Revision 1.69  1997/12/11 16:17:02  coelho
 * fixed log on shells.
 *
 * Revision 1.68  1997/12/11 16:08:02  coelho
 * implicit shell added.
 *
 * Revision 1.67  1997/12/10 12:28:57  coelho
 * internal cat for display.
 *
 * Revision 1.66  1997/12/05 16:50:03  coelho
 * i_checkpoint added.
 *
 * Revision 1.65  1997/12/05 13:28:20  coelho
 * capply rule added.
 *
 * Revision 1.64  1997/11/27 13:34:04  coelho
 * some user errors moved as user warnings in delete...
 *
 * Revision 1.63  1997/11/27 12:14:52  coelho
 * list of command are ok.
 *
 */

%token TK_OPEN
%token TK_CREATE
%token TK_CLOSE
%token TK_CHECKPOINT
%token TK_DELETE
%token TK_MODULE
%token TK_MAKE
%token TK_APPLY
%token TK_CAPPLY
%token TK_DISPLAY
%token TK_REMOVE
%token TK_ACTIVATE
%token TK_SET_PROPERTY
%token TK_GET_PROPERTY
%token TK_SET_ENVIRONMENT
%token TK_GET_ENVIRONMENT
%token TK_CDIR
%token TK_INFO TK_PWD TK_HELP
%token TK_SOURCE
%token TK_SHELL TK_ECHO TK_UNKNOWN
%token TK_QUIT TK_EXIT
%token TK_LINE

%token TK_OWNER_NAME
%token TK_OWNER_ALL
%token TK_OWNER_PROGRAM
%token TK_OWNER_MAIN
%token TK_OWNER_MODULE
%token TK_OWNER_CALLERS
%token TK_OWNER_CALLEES

%token TK_OPENPAREN
%token TK_COMMA
%token TK_CLOSEPAREN
%token TK_EQUAL

%token TK_NAME
%token TK_A_STRING
%token TK_ENDOFLINE

%type <name>   TK_NAME TK_A_STRING TK_LINE TK_UNKNOWN
%type <status> command commands
%type <status> i_open i_create i_close i_delete i_module i_make i_pwd i_source
%type <status> i_apply i_activate i_display i_get i_setenv i_getenv i_cd i_rm
%type <status> i_info i_shell i_echo i_setprop i_quit i_exit i_help i_capply
%type <status> i_checkpoint i_unknown
%type <name> rulename filename propname phasename resourcename
%type <array> filename_list
%type <rn> resource_id rule_id
%type <array> owner list_of_owner_name

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

/********************************************************** static variables */
extern bool tpips_execution_mode;

#define YYERROR_VERBOSE 1 /* MUCH better error messages with bison */

extern void tpips_set_line_to_parse(string);
extern int yylex(void);
extern void yyerror(char *);

static void
free_owner_content(res_or_rule * pr)
{
    gen_array_full_free(pr->the_owners), pr->the_owners = NULL;
    free(pr->the_name), pr->the_name = NULL;
}

void 
close_workspace_if_opened(void)
{
    if (db_get_current_workspace_name())
	close_workspace();
}

static void
set_env_log_and_free(string var, string val)
{
    string ival = getenv(var);
    if (!ival || !same_string_p(val, ival))
	putenv(strdup(concatenate(var, "=", val, 0)));
    user_log("setenv %s \"%s\"\n", var, val);
    free(var); free(val);
}

/* display a resource using $PAGER if defined and stdout on a tty.
 */
static bool
display_a_resource(string rname, string mname)
{
    string fname, pager = getenv("PAGER");
    if (!isatty(fileno(stdout))) pager = NULL;

    lazy_open_module (mname);
    fname = build_view_file(rname);
    if (!fname) pips_user_error("Cannot build view file %s\n", rname);

    if (pager)
    {
	safe_system(concatenate(pager, " ", fname, 0));
    }
    else
    {
	FILE * in = safe_fopen(fname, "r");
	safe_cat(stdout, in);
	safe_fclose(in, fname);
    }
    free(fname);
    return TRUE;
}

static bool
remove_a_resource(string rname, string mname)
{
    if (db_resource_p(rname, mname))
	db_delete_resource(rname, mname);
    else
	pips_user_warning("no resource %s(%s) to delete\n", rname, mname);
    return TRUE;
}

/* apply what to all resources in res
 */
static bool
perform(bool (*what)(string, string), res_or_rule * res)
{
    bool result = TRUE;
    
    if (tpips_execution_mode)
    {
	string save_current_module_name;

	if(!db_get_current_workspace_name())
	    pips_user_error("Open or create a workspace first!\n");

	/* push the current module. */
	save_current_module_name = 
	    db_get_current_module_name()?
	    strdup(db_get_current_module_name()): NULL;
	
	GEN_ARRAY_MAP(mod_name, 
        {
	    if (mod_name != NULL)
	    {
		if (what(res->the_name, mod_name) == FALSE) {
		    result = FALSE;
		    break;
		}
	    }
	    else
		pips_user_warning("Select a module first!\n");
	}, res->the_owners);
	
	/* restore the initial current module, if there was one */
	if(save_current_module_name!=NULL) {
	    if (db_get_current_module_name())
		db_reset_current_module_name();
	    db_set_current_module_name(save_current_module_name);
	    free(save_current_module_name);
	}
    }
    free_owner_content(res);
    return result;
}

static void 
tp_system(string s)
{
    int status;
    user_log("shell%s%s\n", (s[0]==' '|| s[0]=='\t')? "": " ", s);
    status = system(s);
    fflush(stdout);
    if (status) 
	pips_user_warning("shell returned status (%d.%d)\n", 
			  status%256, status/256);
}

%}

%union {
    int status;
    string name;
    res_or_rule rn;
    gen_array_t array;
}

%%

commands: commands command { $$ = $1 && $2; } 
	| command
	;

command: TK_ENDOFLINE { /* may be empty! */ }
	| i_open 
	| i_create
  	| i_close
	| i_delete
	| i_checkpoint
	| i_module
	| i_make
	| i_apply
	| i_capply
	| i_display
	| i_rm
	| i_activate
	| i_get
	| i_getenv
	| i_setenv
	| i_cd
	| i_pwd
	| i_source
	| i_info
	| i_echo
	| i_shell
	| i_setprop
	| i_quit
	| i_exit
	| i_help
	| i_unknown
	| error {$$ = FALSE;}
	;

i_quit: TK_QUIT TK_ENDOFLINE 
	{
	    tpips_close();
	    exit(0);
	}
	;

i_exit: TK_EXIT TK_ENDOFLINE 
	{
	    exit(0); /* rather rough! */
	}
	;

i_help: TK_HELP TK_NAME TK_ENDOFLINE 
	{
	    tpips_help($2);
	    free($2);
	}
	| TK_HELP TK_ENDOFLINE
	{
	    tpips_help("");
	}
	;

i_setprop: TK_SET_PROPERTY TK_LINE TK_ENDOFLINE
	{
	    user_log("setproperty%s\n", $2);
	    parse_properties_string($2);
	    fflush(stdout);
	    free($2);
	}
	;

i_shell: TK_SHELL TK_ENDOFLINE 
	{
	    tp_system("${SHELL:-sh}");
	}
	| TK_SHELL TK_LINE TK_ENDOFLINE 
	{ 
	    tp_system($2); free($2);
	}
	;

i_unknown: TK_UNKNOWN TK_ENDOFLINE
	{ 
	    if (tpips_behaves_like_a_shell || 
		get_bool_property("TPIPS_IS_A_SHELL"))
	    {
		pips_user_warning("implicit shell command assumed!\n");
		tp_system($1); 
	    }
	    else
	    {
		pips_user_warning("\n\n"
		    "\tMaybe you intended to execute a direct shell command.\n"
		    "\tThis convinient feature is desactivated by default.\n"
		    "\tTo enable it, you can run tpips with the -s option,\n"
		    "\tor do \"setproperty TPIPS_IS_A_SHELL=TRUE\"\n\n");
	    }
	    free($1);
	}
	;

i_echo: TK_ECHO TK_LINE TK_ENDOFLINE
	{
	    string s = $2;
	    user_log("echo%s\n", $2); 
	    skip_blanks(s);
	    fprintf(stdout,"%s\n",s);
	    fflush(stdout);
	    free($2);
	}
	| TK_ECHO TK_ENDOFLINE /* there may be no text at all. */
	{
	    user_log("echo\n");
	    fprintf(stdout,"\n");
	    fflush(stdout);
	}
	;

i_info: TK_INFO TK_ENDOFLINE
	{
	    user_log("info: workspace is %s, module is %s\n",
		     db_get_current_workspace_name()? 
		     db_get_current_workspace_name(): "<none>",
		     db_get_current_module_name()?
		     db_get_current_module_name(): "<none>");
	}
	;

i_cd: TK_CDIR TK_NAME TK_ENDOFLINE
	{
	    user_log("cd %s\n", $2);
	    if (chdir($2)) fprintf(stderr, "error while changing directory\n");
	    free($2);
	}
	;

i_pwd: TK_PWD TK_ENDOFLINE
	{
	    char pathname[MAXPATHLEN];
	    fprintf(stdout, "current working directory: %s\n", 
		    (char*) getwd(pathname));
	    fflush(stdout);
	}
	;

i_getenv: TK_GET_ENVIRONMENT TK_NAME TK_ENDOFLINE
	{
	    string val = getenv($2);
	    user_log("getenv %s\n", $2);
	    if (val) fprintf(stdout, "%s=%s\n", $2, val);
	    else fprintf(stdout, "%s is not defined\n", $2);
	    free($2);
	}
	;

i_setenv: TK_SET_ENVIRONMENT TK_NAME TK_NAME TK_ENDOFLINE
	{ set_env_log_and_free($2, $3);	}
	| TK_SET_ENVIRONMENT TK_NAME TK_EQUAL TK_NAME TK_ENDOFLINE
	{ set_env_log_and_free($2, $4);	}
	| TK_SET_ENVIRONMENT TK_NAME TK_A_STRING TK_ENDOFLINE
	{ set_env_log_and_free($2, $3);	}
	| TK_SET_ENVIRONMENT TK_NAME TK_EQUAL TK_A_STRING TK_ENDOFLINE
	{ set_env_log_and_free($2, $4);	}
	;

i_checkpoint: TK_CHECKPOINT TK_ENDOFLINE
	{
	    if (tpips_execution_mode) 
	    {
		if (db_get_current_workspace_name())
		    checkpoint_workspace();
		else
		    pips_user_error("Cannot checkpoint, no workspace!\n");
	    }
	}
	;

i_open:	TK_OPEN TK_NAME TK_ENDOFLINE
	{
	    string main_module_name;

	    pips_debug(7,"reduce rule i_open\n");

	    if (tpips_execution_mode) {
		if (db_get_current_workspace_name() != NULL) {
		    pips_user_warning("Closing workspace %s "
				      "before opening %s!\n",
				      db_get_current_workspace_name(), $2);
		    close_workspace();
		}
		if (!workspace_exists_p($2))
		    pips_user_error("No workspace %s to open!\n", $2);

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
		free($2);
	    }
	}
	;

i_create: TK_CREATE TK_NAME /* workspace name */ 
		filename_list /* fortran files */ TK_ENDOFLINE
	{
	    string main_module_name;
	    pips_debug(7,"reduce rule i_create\n");
	    
	    if (tpips_execution_mode) {
		if (workspace_exists_p($2))
		    pips_user_error
			("Workspace %s already exists. Delete it!\n", $2);
		else if (db_get_current_workspace_name()) {
		    pips_user_error("Close current workspace %s before "
				    "creating another!\n", 
				    db_get_current_workspace_name());
		} else {
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

i_close: TK_CLOSE TK_ENDOFLINE
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

i_delete: TK_DELETE TK_NAME /* workspace name */ TK_ENDOFLINE
	{
	    pips_debug(7,"reduce rule i_delete\n");

	    if (tpips_execution_mode) {
		string wname = db_get_current_workspace_name();
		if ((wname != NULL) && same_string_p(wname, $2)) {
		    pips_user_error("Close before delete: "
				    "Workspace %s is open\n", wname);
		    $$ = FALSE;
		} else {
		    if(workspace_exists_p($2)) 
		    {
			if(delete_workspace ($2)) {
			    /* In case of problem, user_error() has been
			       called, so it is OK now !!*/
			    user_log ("Workspace %s deleted.\n", $2);
			    $$ = TRUE;
			}
			else {
			    pips_user_warning(
				"Could not delete workspace %s\n", $2);
			}
		    }
		    else {
			pips_user_warning("%s: No such workspace\n", $2);
		    }
		}
	    }
	}
	;

i_module: TK_MODULE TK_NAME /* module name */ TK_ENDOFLINE
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

i_make:	TK_MAKE resource_id TK_ENDOFLINE
	{
	    pips_debug(7, "reduce rule i_make\n");
	    $$ = perform(safe_make, &$2);
	}
	;

i_apply: TK_APPLY rule_id TK_ENDOFLINE
	{
	    pips_debug(7,"reduce rule i_apply\n");
	    $$ = perform(safe_apply, &$2);
	}
	;

i_capply: TK_CAPPLY rule_id TK_ENDOFLINE
	{
	    pips_debug(7, "reduce rule i_capply\n");
	    $$ = safe_concurrent_apply($2.the_name, $2.the_owners);
	}
	;

i_display: TK_DISPLAY resource_id TK_ENDOFLINE
	{
	    pips_debug(7,"reduce rule i_display\n");
	    $$ = perform(display_a_resource, &$2);
	}
	;

i_rm: TK_REMOVE resource_id TK_ENDOFLINE
	{
	    pips_debug(7,"reduce rule i_rm\n");
	    $$ = perform(remove_a_resource, &$2);	    
	}
	;

i_activate: TK_ACTIVATE rulename TK_ENDOFLINE
	{
	    pips_debug(7,"reduce rule i_activate\n");
	    if (tpips_execution_mode) {

		if(!db_get_current_workspace_name())
		    pips_user_error("Open or create a workspace first!\n");
		
		user_log("Selecting rule: %s\n", $2);
		activate ($2);
		$$ = TRUE;
	    }
	}
	;

i_get: TK_GET_PROPERTY propname TK_ENDOFLINE
	{
	    pips_debug(7,"reduce rule i_get (%s)\n", $2);
	    
	    if (tpips_execution_mode) {
		fprint_property(stdout, $2);
		$$ = TRUE;
	    }
	}
	;

i_source: TK_SOURCE filename_list TK_ENDOFLINE
	{
	    int n = gen_array_nitems($2), i=0;
	    for(; i<n; i++) { 
		string name = gen_array_item($2, i);
		FILE * sourced = fopen(name, "r");
		if (!sourced) {
		    perror("while sourcing");
		    gen_array_full_free($2);
		    pips_user_error("cannot source file %s\n", name);
		}
		tpips_process_a_file(sourced, FALSE);
		fclose(sourced);
	    }
	    gen_array_full_free($2);
	    tpips_set_line_to_parse(""); /* humm... */
	}

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

filename: TK_NAME
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

owner:	TK_OPENPAREN TK_OWNER_ALL TK_CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner (ALL)\n");
	    if (tpips_execution_mode) 
	    {
		if (!db_get_current_workspace_name())
		    pips_user_error("No current workspace! "
				    "create or open one!\n");
		else 
		    $$ = db_get_module_list();
	    }
	}
	| TK_OPENPAREN TK_OWNER_PROGRAM TK_CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner (PROGRAM)\n");
	    if (tpips_execution_mode)
	    {
		$$ = gen_array_make(0);
		gen_array_dupappend($$, "");
	    }
	}
	| TK_OPENPAREN TK_OWNER_MAIN TK_CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner (MAIN)\n");

	    if (tpips_execution_mode) {
		if (!db_get_current_workspace_name())
		    pips_user_error("No current workspace! "
				    "create or open one!\n");
		else {
		    gen_array_t modules = db_get_module_list();
		    int nmodules = gen_array_nitems(modules),
			number_of_main = 0;
		    message_assert("some modules", nmodules>0);
		    $$ = gen_array_make(0);
		    
		    GEN_ARRAY_MAP(on, 
		    {
			entity mod = local_name_to_top_level_entity(on);

			if (!entity_undefined_p(mod) && 
			    entity_main_module_p(mod))
			{
			    if (number_of_main)
				pips_internal_error("More the one main\n");
			
			    number_of_main++;
			    gen_array_dupappend($$, on);
			}
		    },
			modules);
		    gen_array_full_free(modules);
		}
	    }
	}
	| TK_OPENPAREN TK_OWNER_MODULE TK_CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner (MODULE)\n");
	    if (tpips_execution_mode) {
		string n = db_get_current_module_name();
		$$ = gen_array_make(0);
		if (n) gen_array_dupappend($$, n);
	    }
	}
	| TK_OPENPAREN TK_OWNER_CALLEES TK_CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner (CALLEES)\n");

	    if (tpips_execution_mode) 
	    {
		callees called_modules;

		if (!safe_make(DBR_CALLEES, db_get_current_module_name()))
		  pips_internal_error("Cannot make callees for %s\n",
				      db_get_current_module_name());
		
		called_modules = (callees) 
		    db_get_memory_resource(DBR_CALLEES,
					   db_get_current_module_name(),TRUE);

		$$ = gen_array_from_list(callees_callees(called_modules));
	    }
	}
	| TK_OPENPAREN TK_OWNER_CALLERS TK_CLOSEPAREN
	{
	    pips_debug(7,"reduce rule owner (CALLERS)\n");
	    if (tpips_execution_mode) 
	    {
		callees caller_modules;

		if (!safe_make(DBR_CALLERS, db_get_current_module_name()))
		    pips_internal_error("Cannot make callers for %s\n",
					db_get_current_module_name());

		caller_modules = (callees) 
		    db_get_memory_resource(DBR_CALLERS,
					   db_get_current_module_name(),TRUE);
		
		$$ = gen_array_from_list(callees_callees(caller_modules));
	    }
	}
	|
	TK_OPENPAREN list_of_owner_name TK_CLOSEPAREN
	{ $$ = $2; }
	|
	{
	    pips_debug(7,"reduce rule owner (none)\n");
	    if (tpips_execution_mode) 
	    {
		string n = db_get_current_module_name();
		$$ = gen_array_make(0);
		if (n) gen_array_dupappend($$, n);
	    }
	}
	;

list_of_owner_name: TK_NAME
	{ $$ = gen_array_make(0); gen_array_append($$, strupper($1,$1)); }
	| list_of_owner_name TK_NAME
	    { gen_array_append($1, strupper($2,$2)); $$ = $1; }
	| list_of_owner_name TK_COMMA TK_NAME
	    { gen_array_append($1, strupper($3,$3)); $$ = $1; }
	;

propname: TK_NAME
	{
	    if (!property_name_p($1)) 
		yyerror("expecting a property name\n");
	    $$ = $1;
	}
	;

phasename: TK_NAME
	{
	    if (!phase_name_p($1)) 
		yyerror("expecting a phase name\n");
	    $$ = $1;
	}
	;

resourcename: TK_NAME
	{
	    if (!resource_name_p($1)) 
		yyerror("expecting a resource name\n");
	    $$ = $1;
	}
	;

%%
