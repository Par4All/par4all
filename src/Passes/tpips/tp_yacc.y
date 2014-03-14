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
/*
 * Each full syntax looks for ENDOFLINE so as to check that the right
 * number of arguments is matched.
 */

%token TK_OPEN TK_CREATE TK_CLOSE TK_CHECKPOINT TK_DELETE
%token TK_MODULE
%token TK_MAKE TK_APPLY TK_CAPPLY TK_DISPLAY
%token TK_REMOVE TK_ACTIVATE
%token TK_SET_PROPERTY TK_GET_PROPERTY
%token TK_SET_ENVIRONMENT TK_GET_ENVIRONMENT TK_UNSET_ENVIRONMENT
%token TK_CDIR TK_INFO TK_PWD TK_HELP TK_SHOW TK_SOURCE
%token TK_SHELL TK_ECHO TK_UNKNOWN
%token TK_QUIT TK_EXIT
%token TK_LINE TK_CHECKACTIVE TK_VERSION TK_TOUCH

%token TK_OWNER_NAME
%token TK_OWNER_ALL
%token TK_OWNER_ALLFUNC
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
%type <status> i_checkpoint i_show i_unknown i_checkactive i_touch i_unsetenv
%type <name> rulename filename propname phasename resourcename workspace_name
%type <array> filename_list
%type <rn> resource_id rule_id
%type <array> owner list_of_owner_name

%{
#ifdef HAVE_CONFIG_H
	#include "pips_config.h"
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/param.h>
#include <unistd.h>

#include "linear.h"
#include "genC.h"

#include "ri.h"
#include "database.h"

#include "misc.h"

#include "ri-util.h" /* ri needed for statement_mapping in pipsdbm... */
#include "pipsdbm.h"
#include "resources.h"
#include "phases.h"
#include "properties.h"
#include "pipsmake.h"

#include "top-level.h"
#include "tpips.h"

/********************************************************** static variables */
extern bool tpips_execution_mode;
extern bool consistency_enforced_p;
static bool processing_started_p=false;

#define YYERROR_VERBOSE 1 /* MUCH better error messages with bison */

extern void tpips_set_line_to_parse(string);
extern void tpips_lex_print_pos(FILE *);
extern int yylex(void);
extern void yyerror(const char *);

static void free_owner_content(res_or_rule * pr)
{
	gen_array_full_free(pr->the_owners), pr->the_owners = NULL;
	free(pr->the_name), pr->the_name = NULL;
}

static void set_env_log_and_free(string var, string val)
{
	string ival = getenv(var);
	if (!ival || !same_string_p(val, ival)) {
		putenv(strdup(concatenate(var, "=", val, NULL)));
  }
	user_log("setenv %s \"%s\"\n", var, val);
	free(var); free(val);
}

/* forward.
 */
static bool perform(bool (*)(const char*, const char*), res_or_rule *);

static void try_to_parse_everything_just_in_case(void)
{
	gen_array_t modules = db_get_module_list();
	res_or_rule * pr = (res_or_rule*) malloc(sizeof(res_or_rule));
	pr->the_owners = modules;
	pr->the_name = strdup(DBR_CALLEES);
	perform(safe_make, pr);
	free(pr);
}

/* try hard to open a module.
 */
static bool tp_set_current_module(const char* name)
{
	bool ok = lazy_open_module(name);
	if (!ok)
	{
		/* This is courageous, but makes debugging harder... */
		try_to_parse_everything_just_in_case();
		ok = lazy_open_module(name);
		if (!ok)
		{
			/* Neglect the return code because lazy_open_module() should fail if
				 safe_make fails(). This is a stupid short cut... */
			ok = safe_make(DBR_CODE, name);
			if (ok)
			{
				ok = lazy_open_module(name);
				pips_assert("should be able to open module if code just made...", ok);
			}
		}
	}
	return ok;
}

/* display a resource using $PAGER if defined and stdout on a tty.
 */
static bool display_a_resource(const char* rname, const char* mname)
{
	string fname;
	bool ret;

	if (!tp_set_current_module(mname))
	{
		pips_user_error("could not find module %s to display\n", mname);
	}

	fname = build_view_file(rname);

	if (!fname)
	{
		pips_user_error("Cannot build view file %s\n", rname);
		free(fname);
		return false;
	}

	if (jpips_is_running)
	{
		/* Should tell about what it is?
		 * What about special formats, such as graphs and all?
		 */
		jpips_tag2("show", fname);
		ret = true;
	}
	else
	{
		ret = safe_display(fname);
	}

	free(fname);
	return ret;
}

static bool remove_a_resource(const char* rname, const char* mname)
{
	if (db_resource_p(rname, mname))
		db_delete_resource(rname, mname);
	else
		pips_user_warning("no resource %s[%s] to delete.\n", rname, mname);
	return true;
}

/* tell pipsdbm that the resource is up to date.
 * may be useful if some transformations are applied
 * which do not change the results of some analyses.
 * under the responsability of the user, obviously...
 */
static bool touch_a_resource(const char* rname, const char* mname)
{
	if (db_resource_p(rname, mname))
		db_touch_resource(rname, mname);
	else
		pips_user_warning("no resource %s[%s] to delete.\n", rname, mname);
	return true;
}

static bool just_show(const char* rname, const char* mname)
{
	string file;

	if (!db_resource_p(rname, mname)) {
		pips_user_warning("no resource %s[%s].\n", rname, mname);
		return false;
	}

	if (!displayable_file_p(rname)) {
		pips_user_warning("resource %s cannot be displayed.\n", rname);
		return false;
	}

	/* now returns the name of the file.
	 */
	file = db_get_memory_resource(rname, mname, true);
	fprintf(stdout, "resource %s[%s] is file %s\n", rname, mname, file);

	return true;
}

/* perform "what" to all resources in "res". res is freed. Several rules
call it: display, apply. */
static bool perform(bool (*what)(const char*, const char*), res_or_rule * res)
{
	bool result = true;

	if (tpips_execution_mode)
	{
		string save_current_module_name;

		if(!db_get_current_workspace_name())
			pips_user_error("Open or create a workspace first!\n");

		/* This may be always trapped earlier in the parser by rule "owner". */
        if(gen_array_nitems(res->the_owners)==0) {
            free_owner_content(res);
            pips_user_error("Empty action: no argument!\n");
        }

		/* push the current module. */
		save_current_module_name =
			db_get_current_module_name()?
			strdup(db_get_current_module_name()): NULL;

		GEN_ARRAY_FOREACH(string, mod_name, res->the_owners)
    {
      if (mod_name != NULL)
      {
        if (!what(res->the_name, mod_name)) {
          result = false;
          break;
        }
      }
      else
        pips_user_warning("Select a module first!\n");
    }

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

static void tp_system(const char* s)
{
	int status;
	user_log("shell%s%s\n", (s[0]==' '|| s[0]=='\t')? "": " ", s);
	status = system(s);
	fflush(stdout);

	if (status)
	{
		pips_user_warning("shell returned status (%d.%d)\n",
											status/256, status%256);

	/* generate user error if not interactive,
	 * so as to abort quickly in scripts...
	 */
		if (!tpips_is_interactive)
			pips_user_error("shell error (%d) in tpips script\n", status);
	}
}

static bool tp_close_the_workspace(const char* s)
{
	bool result = true;

	pips_debug(7, "reduce rule i_close\n");

	if (tpips_execution_mode)
	{
		string current = db_get_current_workspace_name();

		if (current!=NULL && s!=NULL)
		{
			if (same_string_p(s, current))
			{
				close_workspace(false);
				result = true;
			}
			else
			{
				pips_user_error("must close the current workspace!\n");
				result = false;
			}
		}
		else {
			pips_user_error("No workspace to close. Open or create one!\n");
			result = false;
		}
		result = true;
	}

	return result;
}

static void tp_some_info(const char* about)
{
	if (same_string_p(about, "workspace"))
	{
		string ws = db_get_current_workspace_name();
		fprintf(stdout, "%s\n", ws? ws: "");
		if (jpips_is_running) jpips_tag2("workspace", ws? ws: "<none>");
	}
	else if (same_string_p(about, "module"))
	{
		string m = db_get_current_module_name();
		fprintf(stdout, "%s\n", m? m: "");
		if (jpips_is_running) jpips_tag2("module", m? m: "<none>");
	}
	else if (same_string_p(about, "modules") &&
					 db_get_current_workspace_name())
	{
		gen_array_t modules = db_get_module_list();
		int n = gen_array_nitems(modules), i;

		if (jpips_is_running)
		{
			jpips_begin_tag("modules");
			jpips_add_tag("");
		}

		for(i=0; i<n; i++)
		{
			string m = gen_array_item(modules, i);
			if (jpips_is_running) jpips_add_tag(m);
			else fprintf(stdout, "%s ", m);

		}
		if (jpips_is_running) jpips_end_tag();

		gen_array_full_free(modules);
	}
	else if (same_string_p(about, "directory"))
	{
	char pathname[MAXPATHLEN];
	fprintf(stdout, "%s\n", (char*) getcwd(pathname, MAXPATHLEN));
	if (jpips_is_running)
		jpips_tag2("directory", (char*) getcwd(pathname, MAXPATHLEN));
	}

	fprintf(stdout, "\n");
}

/* returns an array with the main inside.
   if not found, try to build all callees to parse all sources
   and maybe find it.
 */
static gen_array_t get_main(void)
{
	gen_array_t result = gen_array_make(0), modules;
	int number_of_main = 0;
	int n = 0;
	string main_name = get_first_main_module();

	if (!string_undefined_p(main_name))
	{
		gen_array_append(result, main_name);
		return result;
	}

	/* else try something else just in case...
   * well, it looks rather useless, maybe.
   */
	while (number_of_main==0 && n<2)
	{
		n++;
		modules = db_get_module_list();

		if (n==2)
		{
			pips_user_warning("no main directly found, parsing...\n");
			try_to_parse_everything_just_in_case();
		}

		GEN_ARRAY_FOREACH(string, on, modules)
    {
      entity mod = local_name_to_top_level_entity(on);

      if (!entity_undefined_p(mod) && entity_main_module_p(mod))
      {
        if (number_of_main)
          pips_user_error("More than one main\n");
        number_of_main++;
        gen_array_dupappend(result, on);
      }
    }
		gen_array_full_free(modules);
	}

	return result;
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
        | i_open {processing_started_p=true;}
	| i_create {processing_started_p=false;}
	| i_close {processing_started_p=false;}
	| i_delete {processing_started_p=false;}
	| i_checkpoint
	| i_module
	| i_make {processing_started_p=true;}
	| i_apply {processing_started_p=true;}
	| i_capply {processing_started_p=true;}
	| i_display {processing_started_p=true;}
	| i_show
	| i_rm
	| i_activate
	| i_checkactive
	| i_get
	| i_getenv
	| i_setenv
	| i_unsetenv
	| i_cd
	| i_pwd
	| i_source
	| i_info
	| i_echo
	| i_shell
	| i_setprop
	| i_quit
	| i_version
	| i_exit
	| i_help
	| i_touch
	| i_unknown
	| error {$$ = false;}
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

i_version: TK_VERSION TK_ENDOFLINE
	{
		fprintf(stdout,
						// "tpips: (%s)\n"
						"ARCH=" STRINGIFY(SOFT_ARCH) "\n"
						"REVS=\n"
						"%s"
						"DATE=%s\n"
						"CC_VERSION=%s\n",
						soft_revisions, soft_date, cc_version);
		fflush(stdout);
	}
	;

i_help: TK_HELP TK_NAME TK_ENDOFLINE
	{
		tpips_help($2); free($2);
	}
	| TK_HELP TK_ENDOFLINE
	{
		tpips_help("");
	}
	;

i_setprop: TK_SET_PROPERTY TK_LINE TK_ENDOFLINE
	{
	  consistency_enforced_p = get_bool_property("CONSISTENCY_ENFORCED_P");
	  if(!consistency_enforced_p || !processing_started_p) {
		user_log("setproperty %s\n", $2);
		reset_property_error(); // We start again at tpips
					// level and should be able to
					// avoid the fatal loop...
		parse_properties_string($2);
		if(processing_started_p) {
		  pips_user_warning("Properties should not be updated during "
				    "tpips processing."
				    " Move the setproperty statement at the "
				    "beginning of your tpips script.\n");
		}
	  }
	  else {
	    pips_user_error("Properties should not be updated during tpips "
			    "processing."
			    " Move the setproperty statement at the beginning "
			    "of your tpips script.\n");
	  }
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
		if (tpips_behaves_like_a_shell())
		{
			pips_user_warning("implicit shell command assumed!\n");
			tp_system($1);
		}
		else
		{
			tpips_lex_print_pos(stderr);
			if(get_bool_property("ABORT_ON_USER_ERROR"))
				pips_user_error(
					"\n\n"
					"\tMaybe you intended to execute a direct shell command.\n"
					"\tThis convenient feature is desactivated by default.\n"
					"\tTo enable it, you can run tpips with the -s option,\n"
					"\tor do \"setproperty TPIPS_IS_A_SHELL=TRUE\",\n"
					"\tor do \"setenv TPIPS_IS_A_SHELL=TRUE\".\n"
					"\tOtherwise use ! or \"shell\" as a command prefix.\n\n");
			else
				pips_user_warning(
					"\n\n"
					"\tMaybe you intended to execute a direct shell command.\n"
					"\tThis convenient feature is desactivated by default.\n"
					"\tTo enable it, you can run tpips with the -s option,\n"
					"\tor do \"setproperty TPIPS_IS_A_SHELL=TRUE\",\n"
					"\tor do \"setenv TPIPS_IS_A_SHELL=TRUE\".\n"
					"\tOtherwise use ! or \"shell\" as a command prefix.\n\n");
		}
		free($1);
	}
	;

i_echo: TK_ECHO TK_LINE TK_ENDOFLINE
	{
		string s = $2;
		user_log("echo %s\n", $2);
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

i_info: TK_INFO TK_NAME TK_ENDOFLINE
	{
		tp_some_info($2);
		free($2);
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
						(char*) getcwd(pathname, MAXPATHLEN));
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

// hmmm... not very convincing
i_setenv: TK_SET_ENVIRONMENT TK_NAME TK_NAME TK_ENDOFLINE
	{ set_env_log_and_free($2, $3);	}
	| TK_SET_ENVIRONMENT TK_NAME TK_EQUAL TK_NAME TK_ENDOFLINE
	{ set_env_log_and_free($2, $4);	}
	| TK_SET_ENVIRONMENT TK_NAME TK_A_STRING TK_ENDOFLINE
	{ set_env_log_and_free($2, $3);	}
	| TK_SET_ENVIRONMENT TK_NAME TK_EQUAL TK_A_STRING TK_ENDOFLINE
	{ set_env_log_and_free($2, $4);	}
	| TK_SET_ENVIRONMENT TK_NAME filename_list TK_ENDOFLINE
	{ set_env_log_and_free($2, strdup(string_array_join($3, " ")));	}
	| TK_SET_ENVIRONMENT TK_NAME TK_EQUAL filename_list TK_ENDOFLINE
	{ set_env_log_and_free($2, strdup(string_array_join($4, " ")));	}
	;

i_unsetenv: TK_UNSET_ENVIRONMENT TK_NAME TK_ENDOFLINE
  { unsetenv($2); }
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

i_open:	TK_OPEN workspace_name TK_ENDOFLINE
	{
		string main_module_name;

		pips_debug(7,"reduce rule i_open\n");

		if (tpips_execution_mode) {
			if (db_get_current_workspace_name() != NULL) {
				pips_user_warning("Closing workspace %s "
													"before opening %s!\n",
													db_get_current_workspace_name(), $2);
				close_workspace(false);
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
					free(main_module_name);
				}
			}
		}
		free($2);
	}
	;

workspace_name: TK_NAME
	{
		if(workspace_name_p($1))
			$$ = $1;
		else
			pips_user_error("workspace name %s contains invalid char(s)\n", $1);
	}

i_create: TK_CREATE workspace_name /* workspace name */
		filename_list /* source files */ TK_ENDOFLINE
	{
		string main_module_name;
		pips_debug(7,"reduce rule i_create\n");

		if (tpips_execution_mode) {
			if (workspace_exists_p($2))
				pips_user_error("Workspace %s already exists. Delete it!\n", $2);
			else if (db_get_current_workspace_name()) {
				pips_user_error("Close current workspace %s before "
												"creating another!\n",
												db_get_current_workspace_name());
			}
			else
			{
				if (db_create_workspace($2))
				{
		if (!create_workspace($3))
		{
			db_close_workspace(false);
			/* If you need to preserve the workspace
				 for debugging purposes, use property
				 ABORT_ON_USER_ERROR */
			if(!get_bool_property("ABORT_ON_USER_ERROR")) {
				user_log("Deleting workspace...\n");
				delete_workspace($2);
			}
			pips_user_error("Could not create workspace %s\n", $2);
		}

		main_module_name = get_first_main_module();

		if (!string_undefined_p(main_module_name)) {
			/* Ok, we got it ! Now we select it: */
			user_log("Main module PROGRAM \"%s\" selected.\n",
							 main_module_name);
			lazy_open_module(main_module_name);
			free(main_module_name);
		}
		$$ = true;
				}
				else {
					pips_user_error("Cannot create directory for workspace"
													", check rights!\n");
				}
			}
		}
		free($2);
		gen_array_full_free($3);
	}
	;

i_close: TK_CLOSE /* assume current workspace */ TK_ENDOFLINE
	{
		$$ = tp_close_the_workspace(db_get_current_workspace_name());
	}
	| TK_CLOSE TK_NAME /* workspace name */ TK_ENDOFLINE
	{
		$$ = tp_close_the_workspace($2);
		free($2);
	}
	;

i_delete: TK_DELETE workspace_name /* workspace name */ TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_delete\n");

		if (tpips_execution_mode) {
		string wname = db_get_current_workspace_name();
		if ((wname != NULL) && same_string_p(wname, $2)) {
			pips_user_error("Close before delete: "
											"Workspace %s is open\n", wname);
			$$ = false;
		} else {
			if(workspace_exists_p($2))
			{
				if(delete_workspace ($2)) {
					/* In case of problem, user_error() has been
						 called, so it is OK now !!*/
					user_log ("Workspace %s deleted.\n", $2);
					$$ = true;
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
		free($2);
	}
	;

i_module: TK_MODULE TK_NAME /* module name */ TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_module\n");

		if (tpips_execution_mode) {
			if (db_get_current_workspace_name()) {
				$$ = tp_set_current_module($2 /*strupper($2,$2)*/);
		        free($2);
			} else {
		        free($2);
				pips_user_error("No workspace open. Open or create one!\n");
				$$ = false;
			}
		}
        else free($2);
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

i_touch: TK_TOUCH resource_id TK_ENDOFLINE
	{
		pips_debug(7, "reduce rule i_touch\n");
		$$ = perform(touch_a_resource, &$2);
	}
	;

i_show: TK_SHOW resource_id TK_ENDOFLINE
	{
		pips_debug(7, "reduce rule i_show\n");
		$$ = perform(just_show, &$2);
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
		if (tpips_execution_mode)
		{
			if(!db_get_current_workspace_name())
				pips_user_error("Open or create a workspace first!\n");

		user_log("Selecting rule: %s\n", $2);
		activate($2);
		$$ = true;
		}
		free($2);
	}
	;

i_checkactive: TK_CHECKACTIVE resourcename TK_ENDOFLINE
	{
		string ph = active_phase_for_resource($2);
		fprintf(stdout, "resource %s built by phase %s\n", $2, ph);
		if (jpips_is_running)
		{
			jpips_begin_tag("result");
			jpips_add_tag(ph);
			jpips_end_tag();
		}
		free($2);
	}
	;

i_get: TK_GET_PROPERTY propname TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_get (%s)\n", $2);

		if (tpips_execution_mode) {
			fprint_property(stdout, $2);
			if (jpips_is_running)
			{
				jpips_begin_tag("result");
				jpips_add_tag("");
				fprint_property_direct(jpips_out_file(), $2);
				jpips_end_tag();
			}
			$$ = true;
		}
		free($2);
	}
	;

i_source: TK_SOURCE filename_list TK_ENDOFLINE
	{
		int n = gen_array_nitems($2), i=0;
		bool saved_tpips_is_interactive = tpips_is_interactive;
		tpips_is_interactive = false;
		CATCH(user_exception_error)
		{
			/* cleanup */
			gen_array_full_free($2);
			tpips_set_line_to_parse(""); /* humm... */
			tpips_is_interactive = saved_tpips_is_interactive;
			RETHROW();
		}
		TRY
		{
			for(; i<n; i++)
			{
				string name = gen_array_item($2, i);
				FILE * sourced = fopen(name, "r");
				if (!sourced) {
					perror("while sourcing");
					/* just in case, maybe tpips_init is not yet performed. */
					if (tpips_init_done)
						/* this performs a throw... */
						pips_user_error("cannot source file '%s'\n", name);
					else
					{
						fprintf(stderr, "cannot source file '%s'\n", name);
						break;
					}
				}
				else
				{
					tpips_process_a_file(sourced, name, false);
					fclose(sourced);
				}
			}
			gen_array_full_free($2);
			tpips_set_line_to_parse(""); /* humm... */
			tpips_is_interactive = saved_tpips_is_interactive;
			UNCATCH(user_exception_error);
		}
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
				pips_user_error("No current workspace! create or open one!\n");
			else
				$$ = db_get_module_list();
		}
	}
  | TK_OPENPAREN TK_OWNER_ALLFUNC TK_CLOSEPAREN
	{
		pips_debug(7,"reduce rule owner (ALL)\n");
		if (tpips_execution_mode)
		{
			if (!db_get_current_workspace_name())
				pips_user_error("No current workspace! create or open one!\n");
			else
				$$ = db_get_function_list();
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
				pips_user_error("No current workspace! create or open one!\n");
			else {
				$$ = get_main();
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
															 db_get_current_module_name(),true);

			$$ = gen_array_from_list(callees_callees(called_modules));
		}
	}
	| TK_OPENPAREN TK_OWNER_CALLERS TK_CLOSEPAREN
	{
		pips_debug(7,"reduce rule owner (CALLERS)\n");
		if (tpips_execution_mode)
		{
                        $$ = get_callers(db_get_current_module_name());
		}
	}
	| TK_OPENPAREN list_of_owner_name TK_CLOSEPAREN
	{ $$ = $2; }
	| /* No explicit argument */
	{
		pips_debug(7,"reduce rule owner (none)\n");
		if (tpips_execution_mode)
		{
			string n = db_get_current_module_name();
			$$ = gen_array_make(0);
			if (n)
				gen_array_dupappend($$, n);
			else {
				string wsn = db_get_current_workspace_name();
				/* pips_internal_error("No current module name\n"); */
				if (wsn==NULL)
					pips_user_error(
						"No current workspace. Open or create one first!\n");
				else
		pips_user_error(
			"No current module has been defined, explicitly or implictly.\n"
			"Please specify a module name as argument or check that"
			" the current workspace \"%s\" contains one main module"
			" or no more than one module.\n",
			wsn);
			}
		}
	}
	;

list_of_owner_name: TK_NAME
	{
		$$ = gen_array_make(0); gen_array_append($$, $1);
	}
	| list_of_owner_name TK_NAME
	{ gen_array_append($1, $2 /* strupper($2,$2) */); $$ = $1; }
	| list_of_owner_name TK_COMMA TK_NAME
	{ gen_array_append($1, $3 /* strupper($3,$3) */); $$ = $1; }
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
