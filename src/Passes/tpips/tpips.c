/*
 * $Id$
 *
 * $Log: tpips.c,v $
 * Revision 1.78  1997/12/02 17:47:13  coelho
 * don't initialize of blank or comment line...
 *
 * Revision 1.77  1997/11/27 13:16:18  coelho
 * initialization of toprocess.
 *
 * Revision 1.76  1997/11/27 12:52:44  coelho
 * does not core dump if .tpips file is not found.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <unistd.h>
#include <setjmp.h>
#include <strings.h>
#include <sys/param.h>

#include "readline.h"
#include "history.h"

#include "genC.h"
#include "ri.h"
#include "database.h"
#include "graph.h"
#include "makefile.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "properties.h"
#include "constants.h"
#include "resources.h"
#include "pipsmake.h"

#include "top-level.h"
#include "tpips.h"
#include "completion_list.h"

/********************************************************** Static variables */

bool tpips_execution_mode = TRUE;

static bool use_readline = FALSE;
static FILE * logfile;
static FILE * current_file; /* current file being processed */

extern int tgetnum();
extern void tp_restart( FILE * ); /* tp_lex.c */

#define tpips_usage \
  "Usage: %s [-n] [-h/?] [-v] [-l logfile] [-e tpips-cmds] tpips-scripts\n"

#define before_initial_prompt \
  "tpips (ARCH=" SOFT_ARCH ")\n\n" \
  "  (c) 1988-1997 Centre de Recherche en Informatique,\n" \
  "                École des mines de Paris, France.\n\n" \
  "  URL: http://www.cri.ensmp.fr/pips\n" \
  "  MAIL: pipsgroup@cri.ensmp.fr\n\n" \
  "  This software is provided as is, under the terms of the GPL.\n" \
  "  It includes software from GNU and Berkeley.\n\n"

/*************************************************************** Some Macros */

#define SEPARATOR_P(c) (index (" \t", c))
#define PREFIX_EQUAL_P(str, prf) (strncmp(str, prf, strlen(prf))==0)


/********************************** Some static functions forward definition */
static char **fun_completion();
static char *fun_generator(char*,int);
static char *param_generator(char*, int);
static void initialize_readline();
static char * tpips_read_a_line(char *);

/***************************************************** Some static variables */

static char ** current_completion_array;

/****************************************** Parameter Completion definitions */

enum COMPLETION_TYPES {
    COMP_NONE,
    COMP_FILENAME,
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
{ "source",	COMP_FILENAME,   COMP_FILENAME },
{ CHANGE_DIR,   COMP_FILENAME,   COMP_NONE },
{ QUIT,         COMP_NONE,       COMP_NONE },
{ HELP,         COMP_HELP_TOPIC, COMP_NONE },
{ ECHO,         COMP_NONE,       COMP_NONE },
{ "open",       COMP_NONE,       COMP_NONE },
{ "create",     COMP_NONE,       COMP_FILENAME },
{ "close",      COMP_NONE,       COMP_NONE },
{ "delete",     COMP_NONE,       COMP_NONE },
{ "module",     COMP_NONE,       COMP_NONE },
{ "make",       COMP_RESOURCE,   COMP_NONE },
{ "apply",      COMP_RULE,       COMP_NONE },
{ "display",    COMP_FILE_RSC,   COMP_NONE },
{ "activate",   COMP_RULE,       COMP_NONE },
{ SET_ENV,	COMP_NONE,	 COMP_NONE },
{ GET_ENV, 	COMP_NONE, 	 COMP_NONE },
{ SET_PROP,     COMP_PROPERTY,   COMP_NONE },
{ GET_PROP,     COMP_PROPERTY,   COMP_NONE },
{ "info",       COMP_NONE,   	 COMP_NONE },
{ (char*)NULL,  COMP_NONE,       COMP_NONE }
};

static char *tp_help_topics[] = 
{
    "readline", "create","close","delete","echo","module","activate",
    "make","apply","display",SET_ENV, SET_PROP,GET_PROP,SHELL_ESCAPE,
    CHANGE_DIR,QUIT,"source", HELP,"rule","resource","owner", "remove",
    (char*)NULL
};

/************************************************* TPIPS HANDLERS FOR PIPS */
static void 
tpips_user_log(char *fmt, va_list args)
{
    FILE * log_file = get_log_file();

    /* It goes to stderr to have only displayed files on stdout.
     */
    vfprintf(stderr, fmt, args); 
    fflush(stderr);

    if (!log_file || !get_bool_property("USER_LOG_P"))
	return;

    if (vfprintf(log_file, fmt, args) <= 0) {
	perror("tpips_user_log");
	abort();
    }
    else fflush(log_file);
}

/* Tpips user request */
static string 
tpips_user_request(string fmt, va_list args)
{
    char * response;

    debug_on("TPIPS_DEBUG_LEVEL");

    if (use_readline) {
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

static void 
tpips_user_error(string calling_function_name,
		 string a_message_format,
		 va_list *some_arguments)
{
    /* extern jmp_buf pips_top_level; */
    jmp_buf * ljbp = 0;

   /* print name of function causing error */
   (void) fprintf(stderr, "user error in %s: ", calling_function_name);

   /* print out remainder of message */
   (void) vfprintf(stderr, a_message_format, * some_arguments);

   /* terminate PIPS request */
   if (get_bool_property("ABORT_ON_USER_ERROR")) {
       user_warning("tpips_user_error", "Abort on user error requested!\n");
       abort();
   }
   else
       /* longjmp(pips_top_level, 2); */
      ljbp = top_pips_context_stack();
      longjmp(*ljbp, 2);
}

/*  returns the full tpips history file name, i.e.
 *  - $TPIPS_HISTORY (if any)
 *  - $HOME/"TPIPS_HISTORY"
 */
static string 
default_hist_file_name(void)
{
    string home, hist = getenv(TPIPS_HISTENV), tmp;

    if (hist) return hist;

    /* else builds the default name. memory leak.
     */
    home = getenv("HOME");
    tmp = (char*) malloc(sizeof(char)*(strlen(home)+strlen(TPIPS_HISTORY)+2));
    if (!tmp) pips_exit(1, "memory exhausted\n");
    (void) sprintf(tmp, "%s/%s", home, TPIPS_HISTORY);

    return tmp;
}

static string 
initialize_tpips_history(void)
{
    HIST_ENTRY * last_entry;
    char *file_name = default_hist_file_name();
    
    /*  initialize history: 
     *  read the history file, then point to the last entry.
     */
    using_history();
    read_history(file_name);
    history_set_pos(history_length);
    last_entry = previous_history();

    /* last points to the last history line of any.
     * used to avoid to put twice the same line.
     */
    return last_entry ? last_entry->line : NULL ;
}

/* Handlers
 */
#define TP_HELP(prefix, simple, full)		\
  if (!*line || PREFIX_EQUAL_P(line, prefix)) {	\
      printf(simple); if (*line) printf(full);}

void 
tpips_help(string line)
{
    skip_blanks(line);

    printf("\n");
    TP_HELP("readline", "* readline interaction facilities\n",
	    "\ttry <tab><tab> for automatic completion\n"
	    "\temacs-tyle editing capabilities (see man readline)\n")
    TP_HELP("create", "create   <workspace-name> <file-name>...\n",
	 "\tcreate a new worspace from a list of fortran files\n"
	 "\tfirst delete the workspace if it exists\n");
    TP_HELP("open", "open     <workspace-name>\n",
	 "\topen an existing workspace\n");
    TP_HELP("close", "close\n",
	 "\tclose the current opened workspace\n");
    TP_HELP("delete", "delete   <workspace-name>\n",
	 "\tdelete an existing workspace\n");
    TP_HELP("module", "module   <module-name>\n",
	 "\tselect a module from an opened workspace\n");
    TP_HELP("activate", "activate <rule-name>\n",
	 "\ttell a rule to be active\n");
    TP_HELP("make", "make     <resourcename[(OWNER)]>\n",
	 "\tbuild a resource\n"
	 "\n\tExamples:\n\n"
	 "\t\t make PRINTED_FILE\n"
	 "\t\t make CALLGRAPH_FILE(my_module)\n"
	 "\t\t make DG_FILE($ALL)\n"
	 "\t\t make ICFG_FILE($CALLEES)\n\n");
    TP_HELP("apply", "apply    <rulename[(OWNER)]>\n",
	 "\tmake the produced resources of a rule\n"
	 "\n\tExamples:\n\n"
	 "\t\t apply PRINT_SOURCE_WITH_REGIONS\n"
	 "\t\t apply HPFC_CLOSE(my_module)"
	 "\t\t apply PRINT_CODE($ALL)\n"
	 "\t\t apply PRINT_ICFG($CALLEES)\n");
    TP_HELP("display", "display  <resourcename[(OWNER)]>\n",
	 "\tprint a resource\n"
	 "\n\tExamples:\n\n"
	 "\t\t display PRINTED_FILE\n"
	 "\t\t display CALLGRAPH_FILE(my_module)\n"
	 "\t\t display DG_FILE($ALL)\n"
	 "\t\t display ICFG_FILE($CALLEES)\n\n");
    TP_HELP("remove", "remove	<resourcename[(OWNER)]>\n",
	    "\tremove a resource from the database.\n");
    TP_HELP("cd", "cd       <dirname>\n",
	 "\tchange directory\n");
    TP_HELP("pwd", "pwd\n", "\tprint current working directory\n");
    TP_HELP("setenv", "setenv    <name>=<value>\n",
	 "\tchange environment\n");
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
    TP_HELP("help", "help     [<help-item>]\n",
	 "\tprint a list of all the commands or a \"detailled\""
	 " description of one\n");
    TP_HELP("shell", "shell   [<shell-function>]\n",
	 "\tallow shell functions call\n");
    TP_HELP("owner", "* owner : variable*\n",
	 "\tList of available owners:\n"
	 "\t\t$MODULE\n"
	 "\t\t$ALL\n"
	 "\t\t$PROGRAM\n"
	 "\t\t$CALLEES\n"
	 "\t\t$CALLERS\n"
	 "\t\t<module_name>\n");

    if (!*line || PREFIX_EQUAL_P(line,"rulename") ||
	PREFIX_EQUAL_P(line,"rule")) {
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
	    debug (1,"help_handler","number of columns is %d\n",
		   columns);
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
    if (!*line || PREFIX_EQUAL_P(line,"resourcename") ||
	PREFIX_EQUAL_P(line,"resource")) {
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
	    debug (1,"help_handler","number of columns is %d\n",
		   columns);
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

    close_workspace_if_opened();
    if (logfile) {
	safe_fclose (logfile, "the log file");
	logfile = NULL;
    }
}

/* in lex file
 */
extern void tpips_set_line_to_parse(char*);
extern char * tpips_get_line_to_parse(void);

static void 
handle(string line)
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

/* returns the next line from the input, interactive of file...
 * the final \n does not appear.
 */
static char * get_next_line(char * prompt)
{
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
	char *tmp, *next = get_next_line(TPIPS_SECONDARY_PROMPT);
	line[l-1] = '\0';
	tmp = strdup(concatenate(line, next, NULL));
	free(line); if (next) free(next);
	line = tmp;
    }

    if (logfile && line)
	fprintf(logfile,"%s\n",line);

    pips_debug(3, "line is --%s--\n", line);

    return line;
}

/* simple direct dynamic buffer management. FC.
 * it can be used to accumulate chars, one by one of strings by strings.
 */
static char * sbuffer = NULL;
static int sbufsize = 0;
static void init_sbuffer(void)
{ 
    if (sbuffer) return;
    sbufsize = 64; 
    sbuffer = (char*) malloc(sbufsize); 
    if (!sbuffer) pips_exit(3, "memory exhausted\n");
}
/* appends a char at pos
 */
static int add_sbuffer_char(int pos, char c)
{ 
    if (pos>=sbufsize) { 
	sbufsize*=2; 
	sbuffer = realloc(sbuffer, sbufsize); 
	if (!sbuffer) pips_exit(3, "memory exhausted\n");
    }
    sbuffer[pos] = c;
    return pos+1;
}
/* appends a string at pos
 */
static int add_sbuffer_string(int pos, char * word)
{
    while (word && *word) {
	pos = add_sbuffer_char(pos, *word);
	word++;
    }
    return pos;
}
/* looks for a {} enclosed name from env.
 * if found, returns a pointer to the name, and the line is skipped.
 * if not, returns a pointer to the initial position and NULL (for the name)
 */
static char * skip_env_name(char * line, char** name)
{
    char * s = line;

    if (s && *s && (s[0]!='$' || s[1]!='{')) {
	*name = NULL;
	return s;
    }
    
    s+=2; *name=s;

    while (*s && *s!='}') s++;

    if (*s=='}') {
	*s = '\0'; return s+1;
    } else {
	*name = NULL; return line;
    }
}
/* substitute environemnt variables in line. 
 * returns a newly allocated string.
 */
static char * substitute_variables(char * line)
{
    int pos=0;
    init_sbuffer();
    while (*line) {
	if (*line!='$') {
	    pos = add_sbuffer_char(pos, *line);
	    line++;
	} else { /* *line=='$' */
	    char * name, * nl;
	    nl = skip_env_name(line, &name);
	    if (nl==line) { /* no name found */
		pos = add_sbuffer_char(pos, *line);
		line++;
	    } else {
		line=nl;
		if (name) {
		    pips_debug(1, "substituting $%s\n", name);
		    pos = add_sbuffer_string(pos, getenv(name));
		}
	    }
	}
    }
    add_sbuffer_char(pos, '\0');

    pips_debug(1, "returning: %s\n", sbuffer);

    return strdup(sbuffer);
}

static char * last = NULL;

static bool tpips_init_done = FALSE;
void 
tpips_init(void)
{
    if (tpips_init_done) return;

    pips_checks();

    initialize_newgen();
    initialize_sc((char*(*)(Variable))entity_local_name);
    initialize_signal_catcher();

    set_bool_property("ABORT_ON_USER_ERROR", FALSE); /* ??? */

    pips_log_handler = tpips_user_log;
    pips_request_handler = tpips_user_request;
    pips_error_handler = tpips_user_error;

    tpips_init_done = TRUE;
}

static bool
blank_or_comment_line_p(string line)
{
    skip_blanks(line);
    return line[0]==TPIPS_COMMENT_PREFIX || line[0]=='\0';
}

void 
tpips_exec(char * line)
{
    jmp_buf pips_top_level;

    pips_debug(3, "considering line: %s\n", line? line: " --- empty ---");

    if (signal_occured()) {
	user_log("signal occured, closing workspace...\n");
	close_workspace_if_opened();
    }

    if (setjmp(pips_top_level)) 
    {
	pips_debug(2, "restating tpips scanner\n");
	tp_restart(tp_in);
    }
    else 
    {
	char * sline; /* after environment variable substitution */
	
	push_pips_context(&pips_top_level);
	/*   add to history if not the same as the last one.
	 */
	if (use_readline &&
	    (line && *line && ((last && strcmp(last, line)!=0) || (!last))) &&
	    (strncmp (line, QUIT, strlen(QUIT))))
	{
	    add_history(line);
	    last = strdup(line);
	}

	/*   call the appropriate handler.
	 */
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
	    !blank_or_comment_line_p(line))
	    tpips_init();
	
	sline = substitute_variables(line);
	handle(sline);
	free(sline), sline = (char*) NULL;
    }

    pop_pips_context();
}

/* processing command line per line. might be called recursively thru source.
 */
void 
tpips_process_a_file(FILE * file, bool use_rl)
{
    static readline_initialized = FALSE;
    char * line;
    FILE * saved = current_file;
    bool saved_use_rl = use_readline;

    /* push globals */
    current_file = file;
    use_readline = use_rl;

    if ((use_readline) && (readline_initialized == FALSE))
    {
	initialize_readline ();
	last = initialize_tpips_history();
	readline_initialized = TRUE;

	fprintf(stdout, before_initial_prompt);
    }

    /* interactive loop */
    while ((line = tpips_read_a_line(TPIPS_PRIMARY_PROMPT)))
	tpips_exec(line);

    /* pop globals */
    current_file = saved;
    use_readline = saved_use_rl;
}

static void 
parse_arguments(int argc, char * argv[])
{
    int c;
    extern char *optarg;
    extern int optind;

    while ((c = getopt(argc, argv, "ne:l:h?v")) != -1) {
	switch (c)
	{
	case 'l':
	    logfile = safe_fopen (optarg,"w");
	    break;
	case 'h':
	case '?':
	    fprintf (stderr, tpips_usage, argv[0]);
	    return;
	    break;
	case 'n':
	    tpips_execution_mode = FALSE;
	    break;
	case 'e':
	    tpips_exec(optarg);
	    break;
	case 'v': 
	    fprintf(stderr, "tpips: (ARCH=%s) %s\n", SOFT_ARCH, argv[0]);
            break;
	}
    }

    if (argc == optind)
    {
	/* no arguments, parses stdin. */
	bool use_rl = isatty(0);
	pips_debug(1, "reading from stdin, which is%s a tty\n",
		   use_rl ? "" : " not");
	tpips_process_a_file(stdin, use_rl);
    }
    else 
    {
	/* process file arguments. */
	while (optind < argc)
	{
	    string tps = NULL, saved_srcpath = NULL;
	    FILE * toprocess = (FILE*) NULL;
	    bool use_rl = FALSE;

	    if (same_string_p(argv[optind], "-")) 
	    {
		tps = strdup("-");
		toprocess = stdin;
		use_rl = isatty(0);
	    }
	    else
	    {
		tps = find_file_in_directories(argv[optind], 
					       getenv("PIPS_SRCPATH"));
		if (tps)
		{
		    /* the tpips dirname is appended to PIPS_SRCPATH */
		    string dir = pips_dirname(tps);
		    saved_srcpath = pips_srcpath_append(dir);
		    free(dir), dir = NULL;
		    
		    if ((toprocess = fopen(tps, "r"))==NULL)
		    {
			perror(tps);
			fprintf(stderr, "[TPIPS] cannot open \"%s\"\n", tps);
			free(tps), tps=NULL;
		    }

		    use_rl = FALSE;
		}
		else
		    fprintf(stderr, "[TPIPS] \"%s\" not found...\n", 
			    argv[optind]);
	    }
	    
	    if (tps)
	    {
		pips_debug(1, "reading from file %s\n", tps);
	    
		tpips_process_a_file (toprocess, use_rl);

		if (!same_string_p(tps, "-"))
		    safe_fclose(toprocess, tps);
		
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
int 
tpips_main(int argc, char * argv[])
{
    debug_on("TPIPS_DEBUG_LEVEL");
    pips_log_handler = tpips_user_log;
    {
	string pid = (char*) malloc(sizeof(char)*20);
	sprintf(pid, "PID=%d", (int) getpid());
	pips_assert("not too long", strlen(pid)<20);
	putenv(pid);
    }
    parse_arguments(argc, argv);
    fprintf(stdout, "\n");	/* for Ctrl-D terminations */
    tpips_close();
    return 0;			/* statement not reached ... */
}


/* Tell the GNU Readline library how to complete.  We want to try to complete
 * on command names if this is the first word in the line, or on filenames
 * if not. 
 */
static void 
initialize_readline(void)
{
    /* Allow conditional parsing of the ~/.inputrc file. */
    rl_readline_name = "Tpips";

    /* allow "." to separate words */
    rl_basic_word_break_characters = " \t\n\"\\@$><=;|&{(";

    /* Tell the completer that we want a crack first. */
    rl_attempted_completion_function = (CPPFunction *)fun_completion;
    /* function for completing parameters */
    rl_completion_entry_function = (Function *)param_generator;
}

/* Attempt to complete on the contents of TEXT.  START and END show the
 * region of TEXT that contains the word to complete.  We can use the
 * entire line in case we want to do some simple parsing.  Return the
 * array of matches, or NULL if there aren't any. 
 */
static char **
fun_completion(char *texte, int start, int end)
{

    char **matches;
     
    matches = (char **)NULL;
     
    /* If this word is at the start of the line, then it is a command
       to complete.  Otherwise it is the name of a file in the current
       directory. */
    if (start == 0)
    {
	debug (9,"fun_completion",
	       "completing function (START = %d, END= %d)\n\n",
	       start, end);
	matches = completion_matches (texte , fun_generator);
    }
    return (matches);
}

/* Generator function for command completion.  STATE lets us know whether
 * to start from scratch; without any state (i.e. STATE == 0), then we
 * start at the top of the list. 
 */
static char *
fun_generator(char *texte, int state)
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
    return ((char *)NULL);
}

/* Generator function for param. completion.  STATE lets us know whether
 * to start from scratch; without any state (i.e. STATE == 0), then we
 * start at the top of the list. 
 */
static char *
param_generator(char *texte, int state)
{
    static int list_index, len;
    char *name;
     
    /* If this is a new word to complete, initialize now.  This includes
       saving the length of TEXT for efficiency, and initializing the index
       variable to 0. */
    if (!state)
    {
	char **matches;
	int number_of_sep = 0;
	int current_pos = 0;
	struct t_completion_scheme * cs = completion_scheme;
	int completion_type;
 	matches = (char **)NULL;

	debug (9,"param_generator",
	       "completing parameters\n\n");

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
	debug (9,"param_generator",
	       "%d separator have been found on line\n\n",
	       number_of_sep);

	/* We scan the array of function to find
	   the used function */
	while ((cs->fun_name) &&
	       !PREFIX_EQUAL_P(rl_line_buffer, cs->fun_name))
	{
	    cs++;
	    
	    debug (9,"param_generator",
		   "text is '%s', function found is '%s'\n\n",
		   rl_line_buffer,
		   cs->fun_name  != NULL? cs->fun_name : "<none>");
	}

	/* Now we can determine the completion type */
	if (number_of_sep == 1)
	    completion_type = cs->first_completion_type;
	else
	    completion_type = cs->other_completion_type;

	debug (9,"param_generator",
	       "completion type %d has been selected\n\n",
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
	return filename_completion_function(texte,state);
    
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

/*************************************************************** IS IT A... */

#define CACHED_STRING_LIST(NAME)					\
bool NAME##_name_p(string name)						\
{									\
    static hash_table cache = NULL;					\
									\
    if (!cache) {							\
	char ** p;							\
	cache = hash_table_make(hash_string, 				\
		       2*sizeof(tp_##NAME##_names)/sizeof(char*));	\
	for (p=tp_##NAME##_names; *p; p++)				\
	    hash_put(cache, *p, (char*) 1);				\
    }									\
									\
    return hash_get(cache, name)!=HASH_UNDEFINED_VALUE;			\
}

CACHED_STRING_LIST(phase)
CACHED_STRING_LIST(resource)
CACHED_STRING_LIST(property)
