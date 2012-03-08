/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
%type <status> i_checkpoint i_show i_unknown i_checkactive i_touch
%type <name> rulename filename propname phasename resourcename workspace_name
%type <array> filename_list
%type <rn> resource_id rule_id
%type <array> owner list_of_owner_name

%{

#ifdef HAVE_CONFIG_H
	#include "pips_config.h"
#endif
#include <stdlib.h>
#include <ctype.h>
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

#include "loop_phases.h"

/********************************************************** static variables */
extern bool tpips_execution_mode;


static string pips_cpp_flags = "";
static string default_props = "";
static string loop_label= "tpips2pyps undefined label";

static bool workspace_exist_p = false;

#define YYERROR_VERBOSE 1 /* MUCH better error messages with bison */

#define JUMPL "\n\n"

extern void tpips_set_line_to_parse(string);
extern int yylex(void);
extern void yyerror(const char *);

#define CURRENT_MODULE

static void strtolower(string s) {
  char *c = s;
  while(c && *c!='\0') {
    *c = tolower(*c);
    c++;
  }
}

bool loop_phase_p(string phase){
  char **c = loop_phases;
  while(*c) {
    if(strcmp(phase,*c)==0) {
      return true;
    }
    c++;
  }
  return false;
}

static void perform( string action, bool res_as_action_p, res_or_rule * res)
{
  GEN_ARRAY_FOREACH(string, mod_name, res->the_owners)
  {
    if (mod_name != NULL) {
      string standard_fun = "";
      if(strncmp(mod_name,"program",strlen("program")) != 0
         && strncmp(mod_name,"all_functions",strlen("all_functions")) != 0
         && strncmp(mod_name,"main_module",strlen("main_module")) != 0
         && strncmp(mod_name,"current_module",strlen("current_module")) != 0) {
        standard_fun = "fun.";
      }
      if(res_as_action_p) {
        strtolower(res->the_name);

        string loops_access = strdup("");
        if(loop_phase_p(res->the_name)) {
          free(loops_access);
          asprintf(&loops_access,"loops(\"%s\").",loop_label);
        }

        printf("# %s %s\n"
               "w.%s%s.%s%s()" JUMPL,
               action,res->the_name, standard_fun,
               (string)mod_name,loops_access,res->the_name);

        free(loops_access);
      } else {
        printf("# %s %s\n"
               "w.%s%s.%s(\"%s\")" JUMPL,
               action,res->the_name, standard_fun,
               (string)mod_name,action,res->the_name);
      }
    }
    else {
      pips_user_warning("Select a module first!\n");
    }
  }
}

static void print_putenv(string key, string value){
  if(strcmp(key,"PIPS_CPP_FLAGS")==0) {
    pips_cpp_flags = strdup(value);
  } else {
    printf("os.environ['''%s''']='''%s'''" JUMPL,key, value);
  }
}


static void set_env_log_and_free(string var, string val)
{
  string ival = getenv(var);
  if (!ival || !same_string_p(val, ival)) {
    putenv(strdup(concatenate(var, "=", val, NULL)));
    }
  //user_log("#setenv %s \"%s\"\n", var, val);
  free(var); free(val);
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
	| i_show
	| i_rm
	| i_activate
	| i_checkactive
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
	| i_version
	| i_exit
	| i_help
	| i_touch
	| i_unknown
	| error
	{
	  printf("\n");
	  $$ = false;
	}
	;

i_quit: TK_QUIT TK_ENDOFLINE
	{
		printf("sys.exit()" JUMPL);
	}
	;


i_exit: TK_EXIT TK_ENDOFLINE
	{
    printf("sys.exit()" JUMPL);
	}
	;

i_version: TK_VERSION TK_ENDOFLINE
	{
		printf("print \""
						"ARCH=" STRINGIFY(SOFT_ARCH) "\\n"
						"Unsupported\""JUMPL);
	}
	;

i_help: TK_HELP TK_NAME TK_ENDOFLINE
	{
		printf("#unsupported : help '%s'" JUMPL,$2);
	}
	| TK_HELP TK_ENDOFLINE
	{
    printf("#unsupported : help" JUMPL);
	}
	;

i_setprop: TK_SET_PROPERTY TK_LINE TK_ENDOFLINE
	{
    string s = $2;
    skip_blanks(s);
    char *c = s;
    while(*c!='\0') {
      if(*c==' ' || *c=='\t') {
        *c='=';
        break;
      }
      *c=tolower(*c);
      c++;
    }
    if(strncmp(s,"loop_label=",strlen("loop_label="))==0) {
      loop_label = s;
    } else {
      if(workspace_exist_p) {
        printf("w.props.%s" JUMPL,s);
      } else {
        // postpone to workspace creation
        string buf = strdup(default_props);
        asprintf(&default_props, "w.props.%s\n%s" JUMPL,s,buf);
        free(buf);
      }
    }
	}
	;

i_shell: TK_SHELL TK_ENDOFLINE
	{
    printf("subprocess.call(\"${SHELL:-sh}\\n\", shell=True)" JUMPL);
	}
	| TK_SHELL TK_LINE TK_ENDOFLINE
	{
    printf("subprocess.call('''%s\\n''', shell=True)" JUMPL,$2);
	}
	;

i_unknown: TK_UNKNOWN TK_ENDOFLINE
	{
    printf("# Unknown cmd : '%s'" JUMPL, $1);
	}
	;

i_echo: TK_ECHO TK_LINE TK_ENDOFLINE
	{
    string s = $2;
		skip_blanks(s);
		printf("print '''%s'''" JUMPL,s);
	}
	| TK_ECHO TK_ENDOFLINE /* there may be no text at all. */
	{
    printf("print \"\"" JUMPL);
	}
	;

i_info: TK_INFO TK_NAME TK_ENDOFLINE
	{
    printf("# unsupported 'info %s'" JUMPL,$2);
	}
	;

i_cd: TK_CDIR TK_NAME TK_ENDOFLINE
	{
    printf("os.chdir('''%s''')" JUMPL,$2);
	}
	;

i_pwd: TK_PWD TK_ENDOFLINE
	{
    printf("print \"current working directory: \" + os.getcwd()" JUMPL);
	}
	;

i_getenv: TK_GET_ENVIRONMENT TK_NAME TK_ENDOFLINE
	{
    printf("print os.getenv('''%s''')" JUMPL,$2);
    string val = getenv($2);
    user_log("getenv %s\n", $2);
    if (val) fprintf(stdout, "%s=%s\n", $2, val);
    else fprintf(stdout, "%s is not defined\n", $2);
    free($2);
	}
	;

i_setenv: TK_SET_ENVIRONMENT TK_NAME TK_NAME TK_ENDOFLINE
	{ print_putenv($2, $3);
	  set_env_log_and_free($2, $3); }
	| TK_SET_ENVIRONMENT TK_NAME TK_EQUAL TK_NAME TK_ENDOFLINE
  { print_putenv($2, $4);
    set_env_log_and_free($2, $4); }
	| TK_SET_ENVIRONMENT TK_NAME TK_A_STRING TK_ENDOFLINE
  { print_putenv($2, $3);
    set_env_log_and_free($2, $3); }
	| TK_SET_ENVIRONMENT TK_NAME TK_EQUAL TK_A_STRING TK_ENDOFLINE
  { print_putenv($2, $4);
    set_env_log_and_free($2, $4); }
	| TK_SET_ENVIRONMENT TK_NAME filename_list TK_ENDOFLINE
  {
	  string s = strdup(string_array_join($3, " "));
	  print_putenv($2, s);
	  set_env_log_and_free($2, s);
  }
	| TK_SET_ENVIRONMENT TK_NAME TK_EQUAL filename_list TK_ENDOFLINE
  {
    string s = strdup(string_array_join($4, " "));
    print_putenv($2, s);
    set_env_log_and_free($2, s);
  }
	;

i_checkpoint: TK_CHECKPOINT TK_ENDOFLINE
	{
    printf("print \"Unsupported checkpoint, see git functionnality\"" JUMPL);
	}
	;

i_open:	TK_OPEN workspace_name TK_ENDOFLINE
	{
    printf("print \"Unsupported open !\"" JUMPL);
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
		pips_debug(7,"reduce rule i_create\n");
		gen_array_t files = $3;
		int i, argc = gen_array_nitems(files);

		printf("#Creating workspace\nw = workspace(");

    for ( i = 0; i < argc; i++ ) {
      string filename = gen_array_item( files, i );
      printf("\"%s\",",filename);
    }

    if(pips_cpp_flags[0]!='\0') {
      printf("cppflags=\'\'\'%s\'\'\',",pips_cpp_flags);
    }


    printf("name=\"%s\")" JUMPL, $2);

		gen_array_full_free($3);
		workspace_exist_p = true;

		// Add default workspace properties (defined before in tpips)
    if(default_props && *default_props != '\0') {
      printf("#Add some default property\n%s" JUMPL,default_props);
    }

    $$ = true;
	}
	;

i_close: TK_CLOSE /* assume current workspace */ TK_ENDOFLINE
	{
    printf("#Closing workspace\nw.close()" JUMPL);
    $$ = true;
	}
	| TK_CLOSE TK_NAME /* workspace name */ TK_ENDOFLINE
	{
		printf("#Closing workspace %s\nw.close()" JUMPL,$2);
		workspace_exist_p = false;
    $$ = true;
	}
	;

i_delete: TK_DELETE workspace_name /* workspace name */ TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_delete\n");
    printf("#Deleting workspace\nworkspace.delete(\"%s\")" JUMPL,$2);
    $$ = true;
	}
	;

i_module: TK_MODULE TK_NAME /* module name */ TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_module\n");
    printf("# tpips : module %s\nw.set_current_active_module(\"%s\")" JUMPL, $2, $2);
    $$ = true;
	}
	;

i_make:	TK_MAKE resource_id TK_ENDOFLINE
	{
		pips_debug(7, "reduce rule i_make\n");
    printf("# tpips : make \n# UNSUPPORTED" JUMPL);
    $$ = true;
	}
	;

i_apply: TK_APPLY rule_id TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_apply\n");
		res_or_rule * res = &$2;
		perform("apply" , true, res);
    $$ = true;
	}
	;

i_capply: TK_CAPPLY rule_id TK_ENDOFLINE
	{
		pips_debug(7, "reduce rule i_capply\n");
    res_or_rule * res = &$2;
    GEN_ARRAY_FOREACH(string, mod_name, res->the_owners)
    {
      if (mod_name != NULL) {
        printf("# tpips capply as a simple apply\n"
               "w.fun.%s.%s()" JUMPL, (string)mod_name, res->the_name);
      }
      else {
        pips_user_warning("Select a module first!\n");
      }
    }
    $$ = true;
	}
	;

i_display: TK_DISPLAY resource_id TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_display\n");
    res_or_rule * res = &$2;
    perform("display", false, res);
    $$ = true;
	}
	;

i_touch: TK_TOUCH resource_id TK_ENDOFLINE
	{
		pips_debug(7, "reduce rule i_touch\n");
    printf("# tpips : touch \n# UNSUPPORTED" JUMPL);
    $$ = true;
	}
	;

i_show: TK_SHOW resource_id TK_ENDOFLINE
	{
		pips_debug(7, "reduce rule i_show\n");
    printf("# tpips : show\n# UNSUPPORTED" JUMPL);
    $$ = true;
	}
	;

i_rm: TK_REMOVE resource_id TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_rm\n");
    printf("# tpips : rm\n# UNSUPPORTED" JUMPL);
    $$ = true;
	}
	;

i_activate: TK_ACTIVATE rulename TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_activate\n");
    printf("# tpips : activate %s\nw.activate(\"%s\")" JUMPL, $2,$2);
		$$ = true;
	}
	;

i_checkactive: TK_CHECKACTIVE resourcename TK_ENDOFLINE
	{
   printf("# tpips : checkactive %s\n# UNSUPPORTED" JUMPL, $2);
	}
	;

i_get: TK_GET_PROPERTY propname TK_ENDOFLINE
	{
		pips_debug(7,"reduce rule i_get (%s)\n", $2);
    string s = $2;
    skip_blanks(s);
    printf("# tpips : get %s\nprint w.props.%s" JUMPL, s,s);
    $$ = true;
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
    string n = "all";
    $$ = gen_array_make(0);
    gen_array_dupappend($$, n);
	}
  | TK_OPENPAREN TK_OWNER_ALLFUNC TK_CLOSEPAREN
	{
		pips_debug(7,"reduce rule owner (ALL)\n");
    string n = "all_functions";
    $$ = gen_array_make(0);
    gen_array_dupappend($$, n);
	}
	| TK_OPENPAREN TK_OWNER_PROGRAM TK_CLOSEPAREN
	{
		pips_debug(7,"reduce rule owner (PROGRAM)\n");
		if (tpips_execution_mode)
		{
			$$ = gen_array_make(0);
      gen_array_dupappend($$, "current_module");
//      gen_array_dupappend($$, "program");
		}
	}
	| TK_OPENPAREN TK_OWNER_MAIN TK_CLOSEPAREN
	{
		pips_debug(7,"reduce rule owner (MAIN)\n");
    $$ = gen_array_make(0);
    gen_array_dupappend($$, "main_module");
	}
	| TK_OPENPAREN TK_OWNER_MODULE TK_CLOSEPAREN
	{
		pips_debug(7,"reduce rule owner (MODULE)\n");
		if (tpips_execution_mode) {
			$$ = gen_array_make(0);
			gen_array_dupappend($$, "current_module");
		}
	}
	| TK_OPENPAREN TK_OWNER_CALLEES TK_CLOSEPAREN
	{
		pips_debug(7,"reduce rule owner (CALLEES)\n");
    $$ = gen_array_make(0);
    gen_array_dupappend($$, "current_module.callees");
	}
	| TK_OPENPAREN TK_OWNER_CALLERS TK_CLOSEPAREN
	{
		pips_debug(7,"reduce rule owner (CALLERS)\n");
    $$ = gen_array_make(0);
    gen_array_dupappend($$, "current_module.callers");
	}
	| TK_OPENPAREN list_of_owner_name TK_CLOSEPAREN
	{ $$ = $2; }
	| /* No explicit argument */
	{
		pips_debug(7,"reduce rule owner (none)\n");
		$$ = gen_array_make(0);
		gen_array_dupappend($$, "current_module");
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
//		if (!property_name_p($1))
//			yyerror("expecting a property name\n");
		$$ = $1;
	}
	;

phasename: TK_NAME
	{
//		if (!phase_name_p($1))
//			yyerror("expecting a phase name\n");
		$$ = $1;
	}
	;

resourcename: TK_NAME
	{
//		if (!resource_name_p($1))
//			yyerror("expecting a resource name\n");
		$$ = $1;
	}
	;

%%
