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
#include <stdarg.h>
#include <unistd.h>
#include <strings.h>
#include <sys/param.h>

/* Sometimes, already included by unistd.h */
#include <getopt.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "database.h"

#include "misc.h"
#include "newgen.h"
#include "database.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "properties.h"
#include "constants.h"
#include "resources.h"
#include "pipsmake.h"
#include "preprocessor.h"
#include "top-level.h"

#include "tpips2pyps.h"

/********************************************************** Static variables */

bool tpips_execution_mode = true;
bool tpips_is_interactive = false;

static bool tpips_is_a_shell = false;
static FILE * logfile;

/* current file being processed */
static FILE * current_file = NULL;
static string current_name = "<unknown>";
static int current_line = 0;

void tpips_next_line(void)
{
	current_line++;
}

int tpips_current_line(void)
{
	return current_line;
}

string tpips_current_name(void)
{
	return current_name;
}

extern int tgetnum();
extern void tp_restart( FILE * ); /* tp_lex.c */

/*************************************************************** Some Macros */


static bool prefix_equal_p(string str, string prf)
{
	skip_blanks(str);
	return !strncmp(str, prf, strlen(prf));
}

static bool string_is_true(string s)
{
	return s && (*s=='1' || *s=='t' || *s=='T' || *s=='y' || *s=='Y' ||
							 *s=='o' || *s=='O');
}

/* Whether pips should behave as a shell. Can be turned on from
 * the command line, from properties and from the environment.
 * Default is false.
 */
#define TPIPS_IS_A_SHELL "TPIPS_IS_A_SHELL"

bool tpips_behaves_like_a_shell(void)
{
	return tpips_is_a_shell || get_bool_property(TPIPS_IS_A_SHELL) ||
		string_is_true(getenv(TPIPS_IS_A_SHELL));
}

/*************************************************** FILE OR TTY INTERACTION */

/* returns the next line from the input, interactive tty or file...
 * the final \n does not appear.
 */
static char * get_next_line(char * prompt)
{
	tpips_next_line();
	return safe_readline(current_file);
}

/* returns an allocated line read, including continuations.
 * may return NULL at end of file.
 */
static char * tpips_read_a_line(char * main_prompt)
{
	char *line;
	int l;

	line = get_next_line(main_prompt);

	/* handle backslash-style continuations
	 */
	while (line && (l=strlen(line), l>1 && line[l-1]==TPIPS_CONTINUATION_CHAR))
	{
		char * next = get_next_line(TPIPS_SECONDARY_PROMPT);
		line[l-1] = '\0';
		char * tmp = strdup(concatenate(line, next, NULL));
		free(line); if (next) free(next);
		line = tmp;
	}

	if (logfile && line)
		fprintf(logfile,"%s\n",line);

	pips_debug(3, "line is --%s--\n", line);

	return line;
}

/************************************************* TPIPS HANDLERS FOR PIPS */

/* Tpips user request */

#define BEGIN_RQ	"begin_user_request"
#define END_RQ		"end_user_request"

static string tpips_user_request(const char * fmt, va_list args)
{
	char * response;

	debug_on("TPIPS_DEBUG_LEVEL");

	response = tpips_read_a_line(TPIPS_REQUEST_PROMPT);

	pips_debug(2, "returning --%s--\n", response? response: "<NULL>");

	debug_off();

	return response;
}

/* Tpips user error */
#define BEGIN_UE	"begin_user_error"
#define END_UE		"end_user_error"

static void tpips_user_error(
	const char * calling_function_name,
	const char * a_message_format,
	va_list * some_arguments)
{
  va_list save;
  va_copy(save, *some_arguments);
	/* print name of function causing error and
	 * print out remainder of message
	 */
	fprintf(stderr, "user error in %s: ", calling_function_name);
	append_to_warning_file(calling_function_name, "user error\n",
			       some_arguments);
	vfprintf(stderr, a_message_format, *some_arguments);
	append_to_warning_file(calling_function_name,
			       a_message_format,
			       &save);

	/* terminate PIPS request */
	if (get_bool_property("ABORT_ON_USER_ERROR"))
	{
		pips_user_warning("Abort on user error requested!\n");
		abort();
	}

	THROW(user_exception_error);
}


static void close_workspace_if_opened(bool is_quit)
{
	if (db_get_current_workspace_name())
		close_workspace(is_quit);
}

void tpips_close(void)
{
	/*   close history: truncate list and write history file
	 */
	close_workspace_if_opened(true);

	if (logfile) {
		safe_fclose (logfile, "the log file");
		logfile = NULL;
	}
}

/* in lex file
 */
extern void tpips_set_line_to_parse(char*);
extern char * tpips_get_line_to_parse(void);

static void handle(string line)
{
	tpips_set_line_to_parse(line);

	/* parse if non-null line */
	if (*tpips_get_line_to_parse()) {
		tp_init_lex ();
		tp_parse ();
	}

	fflush(stderr);
	fflush(stdout);
}

/*************************************************************** DO THE JOB */

/* whether some substitutions are needed...
 * variables are restricted to the ${xxx} syntax.
 */
static bool line_with_substitutions(string line)
{
	static char SHELL_CHARS[] = "${`*?"; /* autres : ~ ??? */
	while (*line)
	{
		if (strchr(SHELL_CHARS, *line))
			return true;
		line++;
	}
	return false;
}

/* returns an allocated string after shell substitutions.
 */
static string tp_substitutions(string line)
{
	string substituted;

	/* shell and comments are not substituted...
	 */
	if (!prefix_equal_p(line, "shell") && !prefix_equal_p(line, "!")
			&& !prefix_equal_p(line, "#") && line_with_substitutions(line))
	{
		/* substitutions are performed by forking sh;-)
		 * however sh does not understand ~
		 */
		substituted = safe_system_substitute(line);
		if (!substituted)
		{
			tpips_init();
			pips_user_warning("error in shell substitutions...\n");
			substituted = strdup(line);
		}
		if (line_with_substitutions(substituted))
		{
			pips_user_warning("maybe error in substituted lines:\n\t%s\n"
												"For instance, check location of your source files.\n",
												substituted);
		}
	}
	else
		substituted = strdup(line);

	pips_debug(2, "after substitution: %s\n", substituted);
	return substituted;
}

/* variable globale, utilisee par le parser helas */
bool tpips_init_done = false;

void tpips_init(void)
{
	if (tpips_init_done) return;

	pips_checks();

	initialize_newgen();
	initialize_sc((char*(*)(Variable))entity_local_name);
	/* set_exception_callbacks(push_pips_context, pop_pips_context); */
	/* initialize_signal_catcher(); */

	set_bool_property("ABORT_ON_USER_ERROR", false); /* ??? */

	pips_log_handler = smart_log_handler;
	pips_request_handler = tpips_user_request;
	pips_error_handler = tpips_user_error;

	tpips_init_done = true;
}

static bool blank_or_comment_line_p(string line)
{
	skip_blanks(line);
	return line[0]==TPIPS_COMMENT_PREFIX || line[0]=='\0';
}

static void tpips_exec(char * line)
{
	pips_debug(3, "considering line: %s\n", line? line: " --- empty ---");

	/* does not make much sense here... FC.
		 if (interrupt_pipsmake_asap_p())
		 {
		 user_log("signal occured, closing workspace...\n");
		 close_workspace_if_opened();
		 }
	*/

	CATCH(any_exception_error)
	{
		pips_debug(2, "restating tpips scanner\n");
		tp_restart(tp_in);
	}
	TRY
	{
		char * sline; /* after environment variable substitution */

		pips_debug(2, "restarting tpips scanner\n");
		tp_restart(tp_in);

		/* leading setenv/getenv in a tpips script are performed
		 * PRIOR to pips initialization, hence the environment variable
		 * NEWGEN_MAX_TABULATED_ELEMENTS can be taken into account
		 * for a run. little of a hack.
		 */
		if (!tpips_init_done &&
				strncmp(line, SET_ENV, strlen(SET_ENV))!=0 &&
				strncmp(line, GET_ENV, strlen(GET_ENV))!=0 &&
				strncmp(line, TPIPS_SOURCE, strlen(TPIPS_SOURCE))!=0 &&
				!blank_or_comment_line_p(line))
			tpips_init();

		sline = tp_substitutions(line);
		handle(sline);
		free(sline), sline = (char*) NULL;

		UNCATCH(any_exception_error);
	}
}

/* processing command line per line.
 * might be called recursively thru source.
 */
void tpips_process_a_file(FILE * file, string name, bool use_rl)
{
	char * line;

	// PUSH
	FILE * saved_file = current_file;
	string saved_name = current_name;
	int saved_line = current_line;

	/* push globals */
	current_file = file;
	current_name = name;
	current_line = 0;

	/* interactive loop
	 */
	while ((line = tpips_read_a_line(TPIPS_PRIMARY_PROMPT)))
	{
		tpips_exec(line);
		free(line);
	}

	// POP
	current_file = saved_file;
	current_name = saved_name;
	current_line = saved_line;
}

/* default .tpipsrc is $HOME/.tpipsrc. the returned string is allocated.
 */
static string default_tpipsrc(void)
{
	return strdup(concatenate(getenv("HOME"), "/.tpipsrc", NULL));
}

extern char *optarg;
extern int optind;

static void parse_arguments(int argc, char * argv[])
{
	int c, opt_ind;
	string tpipsrc = default_tpipsrc();
	static struct option lopts[] = {
		{ "version", 0, NULL, 'v' },
		{ "jpips", 0, NULL, 'j' },
		{ "help", 0, NULL, 'h' },
		{ "shell", 0, NULL, 's' },
		{ "log", 1, NULL, 'l' },
		{ "rc", 1, NULL, 'r' },
		{ NULL, 0, NULL, 0 }
	};

	while ((c = getopt_long(argc, argv, "ane:l:h?vscr:jwx", lopts, &opt_ind))
				 != -1) {
		switch (c) {
		  default:
		    fprintf(stderr, "%s : I don't know what to do :-(\n", argv[0]);
		    exit(1);
		}
	}

	/* sources ~/.tpipsrc or the like, if any.
   */
	if (tpipsrc)
	{
		if (file_exists_p(tpipsrc))
		{
			FILE * rc = fopen(tpipsrc, "r");
			if (rc)
			{
				user_log("sourcing tpips rc file: %s\n", tpipsrc);
				tpips_process_a_file(rc, tpipsrc, false);
				fclose(rc);
			}
		}
		free(tpipsrc), tpipsrc=NULL;
	}

	if (argc == optind)
	{
		/* no arguments, parses stdin. */
		bool use_rl = isatty(0);
		tpips_is_interactive = use_rl;
		pips_debug(1, "reading from stdin, which is%s a tty\n",
							 use_rl ? "" : " not");
		tpips_process_a_file(stdin, "<stdin>", use_rl);
	}
	else
	{
		/* process file arguments. */
		while (optind < argc)
		{
			string tps = NULL, saved_srcpath = NULL;
			FILE * toprocess = (FILE*) NULL;
			bool use_rl = false;

			if (same_string_p(argv[optind], "-"))
			{
				tps = strdup("-");
				toprocess = stdin;
				use_rl = isatty(0);
				tpips_is_interactive = use_rl;
			}
			else
			{
				tpips_is_interactive = false;
				tps = find_file_in_directories(argv[optind],
																			 getenv("PIPS_SRCPATH"));
				if (tps)
				{
					/* the tpips dirname is appended to PIPS_SRCPATH */
					string dir = pips_dirname(tps);
					set_script_directory_name(dir);
					saved_srcpath = pips_srcpath_append(dir);
					free(dir), dir = NULL;

					if ((toprocess = fopen(tps, "r"))==NULL)
					{
						perror(tps);
						fprintf(stderr, "[TPIPS] cannot open \"%s\"\n", tps);
						free(tps), tps=NULL;
					}

					use_rl = false;
				}
				else
					fprintf(stderr, "[TPIPS] \"%s\" not found...\n",
									argv[optind]);
			}

			if (tps)
			{
				pips_debug(1, "reading from file %s\n", tps);

				tpips_process_a_file(toprocess, tps, use_rl);

					safe_fclose(toprocess, tps);
				if (!same_string_p(tps, "-"))
					free(tps), tps = NULL;
			}

			if (saved_srcpath)
			{
				pips_srcpath_set(saved_srcpath);
				free(saved_srcpath), saved_srcpath = NULL;
			}

			optind++;
		}
	}
}

/* MAIN: interactive loop and history management.
 */
int tpips_main(int argc, char * argv[])
{
	debug_on("TPIPS_DEBUG_LEVEL");
	pips_log_handler = smart_log_handler;

  {
		char pid[20];
		sprintf(pid, "PID=%d", (int) getpid());
		pips_assert("not too long", strlen(pid)<20);
		putenv(pid);
	}

  // Put header
  printf("from pyps import workspace\n");
  printf("import os,sys,subprocess\n\n");


  printf("#Hack\ntrue=TRUE=True\nfalse=FALSE=False\n\n");


  parse_arguments(argc, argv);
	fprintf(stdout, "\n\n#Add a final breaking line for compatibility with tpips\n"
	        "print \"\"\n");	/* for Ctrl-D terminations */
	tpips_close();
	return 0;			/* statement not reached ... */
}


/*************************************************************** IS IT A... */

#define CACHED_STRING_LIST(NAME)																				\
	bool NAME##_name_p(string name)																				\
	{																																			\
		static hash_table cache = NULL;																			\
		if (!cache) {																												\
			char ** p;																												\
			cache = hash_table_make(hash_string,															\
															2*sizeof(tp_##NAME##_names)/sizeof(char*));	\
			for (p=tp_##NAME##_names; *p; p++)																\
				hash_put(cache, *p, (char*) 1);																	\
		}																																		\
																																				\
		return hash_get(cache, name)!=HASH_UNDEFINED_VALUE;									\
	}
