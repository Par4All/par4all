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

#include <readline/readline.h>
#include <readline/history.h>

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

#include "tpips.h"
#include "completion_list.h"

/********************************************************** Static variables */

bool tpips_execution_mode = true;
bool tpips_is_interactive = false;

static bool tpips_is_a_shell = false;
static bool use_readline = false;
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

#define tpips_usage                                                     \
  "Usage: %s [-nscvh?jwa] "                                             \
  "[-l logfile] [-r rcfile] [-e tpips-cmds] tpips-scripts\n"            \
  "\t-n: no execution mode. just to check a script for syntax errors\n" \
  "\t-s: behaves like a shell. tpips commands simply extend a shell.\n" \
  "\t-c: behaves like a command, not a shell (it is the default).\n"    \
  "\t-h: this help. (also -?)\n"                                        \
  "\t-v: display version and architecture informations.\n"              \
  "\t-a: create a logfile automatically.\n"                             \
  "\t-j: jpips special mode.\n"                                         \
  "\t-w: starts with a wrapper (jpips special again)...\n"              \
  "\t-l  logfile: log to logfile.\n"                                    \
  "\t-r  rcfile: tpips rc file to source. (default ~/.tpipsrc)\n"       \
  "\t-e  tpips-cmds: here tpips commands.\n"                            \
  "\n"

#define SEPARATOR_P(c) (index (" \t", c))

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
 * Default is FALSE.
 */
#define TPIPS_IS_A_SHELL "TPIPS_IS_A_SHELL"

bool tpips_behaves_like_a_shell(void)
{
  return tpips_is_a_shell || get_bool_property(TPIPS_IS_A_SHELL) ||
    string_is_true(getenv(TPIPS_IS_A_SHELL));
}

/********************************************************************* JPIPS */

/* jpips specials.
 *
 * #jpips: modules MAIN FOO BLA...
 * #jpips: prop ...
 * #jpips: done
 * #jpips: begin_user_error
 * ...
 * #jpips: end_user_error
 * #jpips: show ...
 * #jpips: {begin,end}_user_request
 * #jpips:
 */

#define JPIPS_TAG  "#jpips:"

bool jpips_is_running = false;

/* Ronan suggested a signal driven handling of jpips requests,
 * so as to let the prompt available for direct commands.
 * This seems great, but is not implemented at the time.
 * This would mean interruption driven tpips execution from jpips.
 * The executions should not interfere...
 * SIGIO handling on in_from_jpips...
 * forward to readline maybe...
 * a new -J option?
 * how to link C FILE* to unix file descriptors?
 * ? f = fopen(); dup2(..., fileno(f)); OR freopen()...
 */
static FILE * in_from_jpips;
static FILE * out_to_jpips;

FILE * jpips_out_file(void)
{
  return out_to_jpips;
}

void jpips_begin_tag(string s)
{
  fprintf(out_to_jpips, JPIPS_TAG " %s", s);
}

void jpips_add_tag(string s)
{
  fprintf(out_to_jpips, " %s", s);
}

void jpips_end_tag(void)
{
  fprintf(out_to_jpips, "\n");
  fflush(out_to_jpips);
}

void jpips_tag(string s)
{
  jpips_begin_tag(s);
  jpips_end_tag();
}

void jpips_tag2(string s1, string s2)
{
  jpips_begin_tag(s1);
  jpips_add_tag(s2);
  jpips_end_tag();
}

void jpips_done(void)
{
  jpips_tag("done");
}

void jpips_string(const char* a_message_format, va_list *some_arguments)
{
  vfprintf(out_to_jpips, a_message_format, *some_arguments);
  fflush(out_to_jpips);
}

#include <stdarg.h>

void jpips_printf(const string format, ...)
{
  va_list some_arguments;
  va_start(some_arguments, format);
  (void) vfprintf(out_to_jpips, format, some_arguments);
  va_end(some_arguments);
}

/********************************************************** TPIPS COMPLETION */

static char ** current_completion_array;

enum COMPLETION_TYPES {
  COMP_NONE,
  COMP_FILENAME,
  COMP_MODULE,
  COMP_RULE,
  COMP_RESOURCE,
  COMP_PROPERTY,
  COMP_HELP_TOPIC,
  COMP_FILE_RSC
};

struct t_completion_scheme
{
  char *fun_name;
  int first_completion_type;
  int other_completion_type;
};

static struct t_completion_scheme completion_scheme[] =
{
{ SHELL_ESCAPE, COMP_FILENAME,   COMP_FILENAME },
{ TPIPS_SOURCE,  COMP_FILENAME,   COMP_FILENAME },
{ CHANGE_DIR,   COMP_FILENAME,   COMP_NONE },
{ QUIT,    COMP_NONE,       COMP_NONE },
{ "checkpoint", COMP_NONE,       COMP_NONE },
{ HELP,    COMP_HELP_TOPIC, COMP_NONE },
{ ECHO_N,    COMP_NONE,       COMP_NONE },
{ "open",       COMP_NONE,       COMP_NONE },
{ "create",     COMP_NONE,       COMP_FILENAME },
{ "close",      COMP_NONE,       COMP_NONE },
{ "delete",     COMP_NONE,       COMP_NONE },
{ "module",     COMP_MODULE,       COMP_NONE },
{ "make",       COMP_RESOURCE,   COMP_NONE },
{ "remove",  COMP_RESOURCE,   COMP_NONE },
{ "apply",      COMP_RULE,       COMP_NONE },
{ "capply",      COMP_RULE,       COMP_NONE },
{ "display",    COMP_FILE_RSC,   COMP_NONE },
{ "activate",   COMP_RULE,       COMP_NONE },
{ SET_ENV,  COMP_NONE,   COMP_NONE },
{ GET_ENV,  COMP_NONE,   COMP_NONE },
{ "unsetenv",   COMP_NONE, COMP_NONE },
{ SET_PROP,     COMP_PROPERTY,   COMP_NONE },
{ GET_PROP,     COMP_PROPERTY,   COMP_NONE },
{ "info",       COMP_NONE,   COMP_NONE },
{ "show",  COMP_RESOURCE,   COMP_NONE },
{ (char*)NULL,  COMP_FILENAME,   COMP_FILENAME } /* default: files... */
};

static char *tp_help_topics[] =
{
  "readline", "create","close","delete","echo","module","activate",
  "make","apply","capply","display",SET_ENV, SET_PROP,GET_PROP,SHELL_ESCAPE,
  CHANGE_DIR,QUIT,"source", HELP,"rule","resource","owner", "remove",
  "checkpoint", "info", "show", "checkactive", (char*)NULL
};

/* Generator function for command completion.  STATE lets us know whether
 * to start from scratch; without any state (i.e. STATE == 0), then we
 * start at the top of the list.
 */
static char * fun_generator(const char *texte, int state)
{
  static int list_index, len;
  char *name;

  /* If this is a new word to complete, initialize now.  This includes
     saving the length of TEXT for efficiency, and initializing the index
     variable to 0. */
  if (!state)
  {
    list_index = 0;
    len = strlen (texte);
  }

  /* Return the next name which partially matches from the command list. */
  while ((name = completion_scheme[list_index].fun_name))
  {
    list_index++;

    if (strncmp (name, texte, len) == 0)
      return (strdup(name));
  }

  /* If no names matched, then return NULL. */
  return NULL;
}


/* Build an array with the names of all available modules. */

char **get_module_names(void)
{
  static char **names = NULL;
  static gen_array_t modules = NULL;

  if (names != NULL) {
    /* Free after a previous use: */
    free(names);
    gen_array_full_free(modules);
    /* By default, no available module: */
    names = NULL;
  }

  /* Mainly inspired from wpips/emacs.c

     Overkilling since most of time, the module list does not change but I
     guess there is no indicator in PIPS to tell some modules have been
     created or destroyed. */
  if (db_get_current_workspace_name() != NULL) {
    int module_list_length, i;
    modules = db_get_module_list();
    module_list_length = gen_array_nitems(modules);
    /* Note that since calloc initialize the memory to 0, this array will
       end with a NULL pointer as expected. */
    names = calloc(module_list_length + 1, sizeof(char *));
    for(i = 0; i < module_list_length; i++)
      names[i] = gen_array_item(modules, i);
  }
  return names;
}

/* Generator function for param. completion.  STATE lets us know whether
 * to start from scratch; without any state (i.e. STATE == 0), then we
 * start at the top of the list.
 */
static char * param_generator(const char *texte, int state)
{
  static int list_index, len;
  char *name;

  /* If this is a new word to complete, initialize now.  This includes
     saving the length of TEXT for efficiency, and initializing the index
     variable to 0. */
  if (!state)
  {
    int number_of_sep = 0;
    int current_pos = 0;
    struct t_completion_scheme * cs = completion_scheme;
    int completion_type;

    pips_debug (9, "completing parameters\n\n");

    /*We should count the number of separator before the actual pos*/
    while (rl_line_buffer[current_pos])
    {
      if (SEPARATOR_P(rl_line_buffer[current_pos]))
      {
        number_of_sep ++;
        current_pos++;
        while ((rl_line_buffer[current_pos]) &&
               (SEPARATOR_P(rl_line_buffer[current_pos])))
          current_pos++;
      }
      else
        current_pos++;
    }
    pips_debug (9, "%d separator have been found on line\n\n",
                number_of_sep);

    /* We scan the array of function to find
     the used function */
    while ((cs->fun_name) &&
           !prefix_equal_p(rl_line_buffer, cs->fun_name))
    {
      cs++;
      pips_debug (9, "text is '%s', function found is '%s'\n\n",
                  rl_line_buffer,
                  cs->fun_name!=NULL? cs->fun_name : "<none>");
    }

    /* Now we can determine the completion type */
    if (number_of_sep == 1)
      completion_type = cs->first_completion_type;
    else
      completion_type = cs->other_completion_type;

    pips_debug (9, "completion type %d has been selected\n\n",
                completion_type);

    switch (completion_type)
    {
    case COMP_NONE:
      current_completion_array = NULL;
      break;
    case COMP_FILENAME:
#define RESERVED_FOR_FILENAME (char**)"should not appear"
      current_completion_array = RESERVED_FOR_FILENAME;
      break;
    case COMP_MODULE:
      current_completion_array = get_module_names();
      break;
    case COMP_RULE:
      current_completion_array = tp_phase_names;
      break;
    case COMP_RESOURCE:
      current_completion_array = tp_resource_names;
      break;
    case COMP_PROPERTY:
      current_completion_array = tp_property_names;
      break;
    case COMP_HELP_TOPIC:
      current_completion_array = tp_help_topics;
      break;
    case COMP_FILE_RSC:
      current_completion_array = tp_file_rsc_names;
      break;
    default:
      current_completion_array = NULL;
    }
    list_index = 0;
    len = strlen (texte);
  }

  if (current_completion_array == NULL)
    return NULL;
  else if (current_completion_array == RESERVED_FOR_FILENAME)
    return rl_filename_completion_function(texte,state);

  /* Return the next name which partially matches from the command list. */
  while ((name = current_completion_array[list_index]))
  {
  list_index++;

  if (strncmp (name, texte, len) == 0)
    return (strdup(name));
  }

  /* If no names matched, then return NULL. */
  return NULL;
}

/* Attempt to complete on the contents of TEXT.  START and END show the
 * region of TEXT that contains the word to complete.  We can use the
 * entire line in case we want to do some simple parsing.  Return the
 * array of matches, or NULL if there aren't any.
 */
static char ** fun_completion(char *texte, int start, int end)
{

  char **matches;

  matches = (char **)NULL;

  /* If this word is at the start of the line, then it is a command
     to complete.  Otherwise it is the name of a file in the current
     directory. */
  if (start == 0)
  {
    pips_debug (9, "completing function (START = %d, END= %d)\n\n",
                start, end);
    matches = rl_completion_matches (texte , fun_generator);
  }
  return matches;
}

/* Tell the GNU Readline library how to complete.  We want to try to complete
 * on command names if this is the first word in the line, or on filenames
 * if not.
 */
static void initialize_readline(void)
{
  /* Allow conditional parsing of the ~/.inputrc file. */
  rl_readline_name = "Tpips";

  /* allow "." to separate words */
  rl_basic_word_break_characters = " \t\n\"\\@$><=;|&{(";

  /* Tell the completer that we want a crack first. */
#if defined (_RL_FUNCTION_TYPEDEF)
  rl_attempted_completion_function = (rl_completion_func_t *) fun_completion;
#else
  rl_attempted_completion_function = (CPPFunction *) fun_completion;
#endif

  /* function for completing parameters */
  rl_completion_entry_function = (rl_compentry_func_t *) param_generator;
}


/*************************************************** FILE OR TTY INTERACTION */

/* returns the next line from the input, interactive tty or file...
 * the final \n does not appear.
 */
static char * get_next_line(char * prompt)
{
  tpips_next_line();
  return use_readline? readline(prompt): safe_readline(current_file);
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

#define BEGIN_RQ  "begin_user_request"
#define END_RQ    "end_user_request"

static string tpips_user_request(const char * fmt, va_list args)
{
  char * response;

  debug_on("TPIPS_DEBUG_LEVEL");

  if (jpips_is_running)
  {
    jpips_tag(BEGIN_RQ);
    jpips_string( fmt, (va_list*)&args);
    jpips_printf("\n");
    jpips_tag(END_RQ);
  }
  else if (use_readline)
  {
    (void) fprintf(stdout,"\nWaiting for your response: ");
    (void) vfprintf(stdout, fmt, args);
    fflush(stdout);
  }

  response = tpips_read_a_line(TPIPS_REQUEST_PROMPT);

  pips_debug(2, "returning --%s--\n", response? response: "<NULL>");

  debug_off();

  return response;
}

/* Tpips user error */
#define BEGIN_UE  "begin_user_error"
#define END_UE    "end_user_error"

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

  if (jpips_is_running)
  {
    jpips_tag(BEGIN_UE);
    jpips_printf("%s\n", calling_function_name);
    jpips_string( a_message_format, some_arguments);
    jpips_tag(END_UE);
  }

  /* terminate PIPS request */
  if (get_bool_property("ABORT_ON_USER_ERROR"))
  {
    pips_user_warning("Abort on user error requested!\n");
    abort();
  }

  THROW(user_exception_error);
}

/*  returns the allocated full tpips history file name, i.e.
 *  - $TPIPS_HISTORY (if any)
 *  - $HOME/"TPIPS_HISTORY"
 */
static string default_hist_file_name(void)
{
  string home, hist = getenv(TPIPS_HISTENV);
  if (hist) return strdup(hist);
  /* else builds the default name.
   */
  home = getenv("HOME");
  return strdup(concatenate(home? home: "", "/", TPIPS_HISTORY, NULL));
}

static void initialize_tpips_history(void)
{
  string file_name = default_hist_file_name();
  // read the history file, then point to the last entry.
  using_history();
  read_history(file_name);
  free(file_name);
  history_set_pos(history_length);
}

/* Handlers
 */
#define TP_HELP(prefix, simple, full)            \
  if (!*line || prefix_equal_p(line, prefix)) {  \
    printf(simple); if (*line) printf(full);}

void tpips_help(string line)
{
  skip_blanks(line);

  printf("\n");
  TP_HELP("readline", "* readline interaction facilities\n",
          "\ttry <tab><tab> for automatic completion\n"
          "\temacs-tyle editing capabilities (see man readline)\n");
  TP_HELP("create", "create   <workspace-name> <file-name>...\n",
          "\tcreate a new worspace from a list of fortran files\n"
          "\tfirst delete the workspace if it exists\n");
  TP_HELP("open", "open     <workspace-name>\n",
          "\topen an existing workspace\n");
  TP_HELP("checkactive", "checkactive <resourcename>\n",
          "\ttell which phase would produce this resource.\n");
  TP_HELP("checkpoint", "checkpoint\n",
          "\tcheckpoint the current workspace\n");
  TP_HELP("close", "close\n",
          "\tclose the current opened workspace\n");
  TP_HELP("delete", "delete   <workspace-name>\n",
          "\tdelete an existing workspace\n");
  TP_HELP("module", "module   <module-name>\n",
          "\tselect a module from an opened workspace\n");
  TP_HELP("activate", "activate <rule-name>\n",
          "\ttell a rule to be active\n");
  TP_HELP("make", "make     <resourcename([OWNER])>\n",
          "\tbuild a resource\n"
          "\n\tExamples:\n\n"
          "\t\t make PRINTED_FILE\n"
          "\t\t make CALLGRAPH_FILE[my_module]\n"
          "\t\t make DG_FILE[%%ALL]\n"
          "\t\t make ICFG_FILE[%%CALLEES]\n\n");
  TP_HELP("apply", "apply    <rulename[(OWNER)]>\n",
          "\tmake the produced resources of a rule\n"
          "\n\tExamples:\n\n"
          "\t\t apply PRINT_SOURCE_WITH_REGIONS\n"
          "\t\t apply HPFC_CLOSE[my_module]\n"
          "\t\t apply PRINT_CODE[%%ALL]\n"
          "\t\t apply PRINT_ICFG[%%CALLEES]\n");
  TP_HELP("capply", "capply    <rulename[(OWNER)]>\n",
          "\tconcurrently apply a transformation rule\n"
          "\n\tExamples:\n\n"
          "\t\t apply SUPPRESS_DEAD_CODE[%%ALL]\n"
          "\t\t apply PARTIAL_EVAL[%%CALLEES]\n");
  TP_HELP("display", "display  <fileresourcename([OWNER])>\n",
          "\tprint a file resource\n"
          "\n\tExamples:\n\n"
          "\t\t display PRINTED_FILE\n"
          "\t\t display CALLGRAPH_FILE[my_module]\n"
          "\t\t display DG_FILE[%%ALL]\n"
          "\t\t display ICFG_FILE[%%CALLEES]\n\n");
  TP_HELP("remove", "remove  <resourcename([OWNER])>\n",
          "\tremove a resource from the database.\n");
  TP_HELP("cd", "cd       <dirname>\n",
          "\tchange directory\n");
  TP_HELP("pwd", "pwd\n", "\tprint current working directory\n");
  TP_HELP("setenv", "setenv    <name>=<value>\n",
          "\tchange environment\n");
  TP_HELP("unsetenv", "unsetenv    <name>\n",
          "\tremove variable from environment\n");
  TP_HELP("getenv", "getenv   <name>\n",
          "\tprint from environment (echo ${<name>} also ok)\n");
  TP_HELP("setproperty", "setproperty <name>=<value>\n",
          "\tchange property\n");
  TP_HELP(GET_PROP, GET_PROP " <name>\n",
          "\t print property\n");
  TP_HELP("echo", "echo     <string>\n",
          "\tprint the string\n");
  TP_HELP("quit", "quit\n",
          "\texit tpips (you should close the workspace before\n");
  TP_HELP("exit", "exit\n",
          "\texit tpips quickly (rhough!)\n");
  TP_HELP("source", "source <filenames...>\n",
          "\tread tpips commands from files.\n");
  TP_HELP("help", "help     (<help-item>)\n",
          "\tprint a list of all the commands or a \"detailled\""
          " description of one\n");
  TP_HELP("show", "show     <resourcename([OWNER])>\n",
          "\treturns the file of this resource\n");
  TP_HELP("info", "info <name>\n",
          "\tprint information about <name>\n"
          "\tname: module, modules, workspace, directory\n");
  TP_HELP("shell", "shell   [<shell-function>]\n",
          "\tallow shell functions call\n");
  TP_HELP("version", "version\n",
          "\tshow tpips version informations, such as:\n"
          "\t\trepository revisions used by the compilation\n"
          "\t\tdate of compilation\n"
          "\t\tcompiler used\n");
  TP_HELP("owner", "- owner : variable*\n",
          "\tList of available owners:\n"
          "\t\t%%MODULE\n"
          "\t\t%%ALL\n"
          "\t\t%%ALLFUNC\n"
          "\t\t%%PROGRAM\n"
          "\t\t%%CALLEES\n"
          "\t\t%%CALLERS\n"
          "\t\t<module_name>\n");
  TP_HELP("*", "* default rule...\n",
          "\tan implicit \"shell\" is assumed.\n");

  if (!*line || prefix_equal_p(line,"rulename") ||
      prefix_equal_p(line,"rule")) {
    printf("* rulename : variable*\n");
    if (*line) {
      char ** ps = tp_phase_names;
      int big_size = 0;
      int current_size;
      int columns, count;

      while (*ps) {
        current_size = strlen (*ps);
        if (big_size < current_size)
          big_size = current_size;
        ps++;
      }
      big_size++;
      /* get the number of colunms for 80 chars */
      columns = tgetnum ("co");
      pips_debug (1, "number of columns is %d\n", columns);
      columns = (columns > 0) ? columns /big_size : 1;
      count = 1;
      printf("\tList of available rules\n");
      ps = tp_phase_names;
      while (*ps)
      {
        printf("%-*s",big_size,*ps);
        if ((count % columns) == 0)
          printf("\n");
        ps++;
        count++;
      }
    }
  }
  if (!*line || prefix_equal_p(line,"resourcename") ||
      prefix_equal_p(line,"resource")) {
    printf("* resourcename : variable*\n");
    if (*line) {
      char ** ps = tp_resource_names;
      int big_size = 0;
      int current_size;
      int columns, count;

      while (*ps)
      {
        current_size = strlen (*ps);
        if (big_size < current_size)
          big_size = current_size;
        ps++;
      }
      /* get the number of colunms for 80 chars */
      columns = tgetnum ("co");
      big_size++;
      pips_debug (1, "number of columns is %d\n", columns);
      columns = (columns > 0) ? columns /big_size : 1;
      count = 1;
      printf("\tList of available resources\n");
      ps = tp_resource_names;
      while (*ps)
      {
        printf("%-*s",big_size,*ps);
        if ((count % columns) == 0)
          printf("\n");
        ps++;
        count++;
      }
    }
  }
  printf("\n");
  fflush(stdout);
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
  if (use_readline)
  {
    char *file_name = default_hist_file_name();
    stifle_history(TPIPS_HISTORY_LENGTH);
    write_history(file_name);
  }

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
      if (get_bool_property("ABORT_ON_USER_ERROR")) abort();
      substituted = strdup(line);
    }
    if (line_with_substitutions(substituted))
    {
      // not sure whether there is really an error, so we cannot abort
      tpips_init();
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

  in_from_jpips = stdin;
  out_to_jpips = stdout;

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
    if (use_readline && line)
      add_history(strdup(line));

    pips_debug(2, "restarting tpips scanner\n");
    tp_restart(tp_in);

    /* leading setenv/getenv in a tpips script are performed
     * PRIOR to pips initialization, hence the environment variable
     * NEWGEN_MAX_TABULATED_ELEMENTS can be taken into account
     * for a run. little of a hack. That results in a core
     * dump when the tpips script starts with setenv
     * commands generating user warnings because those
     * imply a check of the property NO_USER_WARNING. Also
     * errors are likely to lead to a check of
     * ABORT_ON_USER_ERROR. And properties cannot be used
     * before tpips_init() has been executed. So
     * pips_user_warning() have to be protected in tpips.c
     * by a preliminary call to tpips_init()
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
  static bool readline_initialized = false;
  char * line;

  // PUSH
  FILE * saved_file = current_file;
  string saved_name = current_name;
  int saved_line = current_line;
  bool saved_use_rl = use_readline;

  /* push globals */
  current_file = file;
  current_name = name;
  current_line = 0;
  use_readline = use_rl;

  if (use_readline && !readline_initialized)
  {
    initialize_readline();
    initialize_tpips_history();
    readline_initialized = true;
  }

  /* interactive loop
   */
  while ((line = tpips_read_a_line(TPIPS_PRIMARY_PROMPT)))
  {
    tpips_exec(line);
    free(line);
    if (jpips_is_running && file==stdin) jpips_done();
  }

  // POP
  current_file = saved_file;
  current_name = saved_name;
  current_line = saved_line;
  use_readline = saved_use_rl;
}

/* default .tpipsrc is $HOME/.tpipsrc. the returned string is allocated.
 */
static string default_tpipsrc(void)
{
  return strdup(concatenate(getenv("HOME"), "/.tpipsrc", NULL));
}

extern char *optarg;
extern int optind;

static void open_logfile(string filename, char opt)
{
  if (logfile)
  {
    fprintf(logfile,
            "# logfile moved to %s by -%c tpips option\n", filename, opt);
    safe_fclose(logfile, "the current log file");
  }
  logfile = safe_fopen(filename, "w");
}

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
         != -1)
  {
    switch (c)
    {
    case 'j':
      jpips_is_running = true;
      /* -j => -a */
    case 'a':
    {
      string filename = safe_new_tmp_file("tpips_session");
      fprintf(stderr, "tpips session logged in \"%s\"\n", filename);
      open_logfile(filename, c);
      free(filename);
      break;
    }
    case 's':
      tpips_is_a_shell = true;
      break;
    case 'c':
      tpips_is_a_shell = false;
      break;
    case 'l':
      open_logfile(optarg, c);
      break;
    case 'h':
    case '?':
      fprintf (stderr, tpips_usage, argv[0]);
      return;
      break;
    case 'n':
      tpips_execution_mode = false;
      break;
    case 'e':
      tpips_exec(optarg);
      break;
    case 'v':
      fprintf(stdout,
              "tpips: (%s)\n"
              "ARCH=" STRINGIFY(SOFT_ARCH) "\n"
              "REVS=\n"
              "%s"
              "DATE=%s\n"
              "CC_VERSION=%s\n",
              argv[0], soft_revisions, soft_date, cc_version);
      exit(0);
      break;
    case 'r':
      free(tpipsrc);
      tpipsrc = strdup(optarg);
      break;
    case 'w':
      tpips_wrapper(); /* the wrapper process will never return */
      break;
    case 'x':
      /* tpips could start an xterm and redirect its stdin/stdout
       * on it under this option. not implemented yet.
       */
      break;
    default:
      fprintf(stderr, tpips_usage, argv[0]);
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
  variable_debug_name = (char * (*)(Variable)) entity_local_name;
  debug_on("TPIPS_DEBUG_LEVEL");
  pips_log_handler = smart_log_handler;
  initialize_signal_catcher();
  /* I need this one right now, as tpips init may be called too late. */
  set_exception_callbacks(push_pips_context, pop_pips_context);

  {
    char pid[20];
    sprintf(pid, "PID=%d", (int) getpid());
    pips_assert("not too long", strlen(pid)<20);
    putenv(pid);
  }

  parse_arguments(argc, argv);
  fprintf(stdout, "\n");  /* for Ctrl-D terminations */
  tpips_close();
  return 0;      /* statement not reached ... */
}


/*************************************************************** IS IT A... */

#define CACHED_STRING_LIST(NAME)                                        \
  bool NAME##_name_p(string name)                                       \
  {                                                                     \
    static hash_table cache = NULL;                                     \
    if (!cache) {                                                       \
      char ** p;                                                        \
      cache = hash_table_make(hash_string,                              \
                            2*sizeof(tp_##NAME##_names)/sizeof(char*)); \
      for (p=tp_##NAME##_names; *p; p++)                                \
        hash_put(cache, *p, (char*) 1);                                 \
    }                                                                   \
                                                                        \
    return hash_get(cache, name)!=HASH_UNDEFINED_VALUE;                 \
  }

CACHED_STRING_LIST(phase)
CACHED_STRING_LIST(resource)
CACHED_STRING_LIST(property)
