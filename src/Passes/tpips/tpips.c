#include <stdio.h>
#include <fcntl.h>
#include <string.h>
#include <stdarg.h>
#include <setjmp.h>
#include <stdlib.h>
#include <strings.h>
#include <unistd.h>

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

static bool use_readline;
static FILE *logfile;
static FILE * current_file;
extern int tgetnum();
static char *usage = "Usage: %s [-n] [-h] [-l logfilename] sourcefile...\n";

/*************************************************************** Some Macros */

#define TPIPS_PROMPT "tpips> " 		/* prompt for readline  */
#define TPIPS_HISTENV "TPIPS_HISTORY"	/* history file env variable */
#define TPIPS_HISTORY_LENGTH 100	/* max length of history file */
#define TPIPS_COMMENT_PREFIX '#'	/* comment prefix */
#define HIST ".tpips.history" 		/* default history file */
#define TPIPS_REQUEST_BUFFER_LENGTH 100
#define SHELL_ESCAPE "shell" 		/* ! used for history reference */
#define CHANGE_DIR   "cd "
#define QUIT         "quit"
#define HELP         "help"
#define ECHO         "echo"

#define SEPARATOR_P(c) (index (" \t", c))
#define PREFIX_EQUAL_P(str, prf) (strncmp(str, prf, strlen(prf))==0)


/********************************** Some static functions forward definition */
static char **fun_completion();
static char *fun_generator(char*,int);
static char *param_generator(char*, int);
static void initialize_readline();

/***************************************************** Some static variables */

static char * line_to_parse;
static char * line_parsed;
static char ** current_completion_array;

/****************************************** Parameter Completion definitions */

enum COMPLETION_TYPES {
    COMP_NONE,
    COMP_FILENAME,
    COMP_RULE,
    COMP_RESOURCE,
    COMP_PROPERTY,
    COMP_HELP_TOPIC
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
{ "display",    COMP_RESOURCE,   COMP_NONE },
{ "activate",   COMP_RULE,       COMP_NONE },
{ "set",        COMP_PROPERTY,   COMP_NONE },
{ "get",        COMP_PROPERTY,   COMP_NONE },
{ "info",       COMP_PROPERTY,   COMP_NONE },
{ (char*)NULL,  COMP_NONE,       COMP_NONE }
};

static char *tp_help_topics[] = 
{
    "create","close","delete","echo","module","activate",
    "make","apply","display","set","get",SHELL_ESCAPE,
    CHANGE_DIR,QUIT,HELP,"rule","resource","owner",
    (char*)NULL
};

/************************************************* TPIPS HANDLERS FOR PIPS */
static void tpips_user_log(char *fmt, va_list args)
{
    FILE * log_file = get_log_file();

    if(log_file!=NULL) {
	if (vfprintf(log_file, fmt, args) <= 0) {
	    perror("tpips_user_log");
	    abort();
	}
	else
	    fflush(log_file);
    }

    if(get_bool_property("USER_LOG_P")==FALSE)
	return;

    /* It goes to stderr to have only displayed files on stdout */
    /* if no logfile, nowhere. FC. */
    (void) vfprintf(stderr, fmt, args); fflush(stderr);
}

/* Tpips user request */

static string tpips_user_request(fmt, args)
char *fmt;
va_list args;
{
    static char buf[TPIPS_REQUEST_BUFFER_LENGTH];

    /* GO: Don't print the request if we are in batch mode */
    if (use_readline) {
	fprintf(stderr,"\nWaiting for your response: ");
	(void) vfprintf(stderr, fmt, args);
	fflush(stderr);
    }
    return gets(buf);
}

/* Tpips user error */

static void tpips_user_error(char * calling_function_name,
			     char * a_message_format,
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
      abort();
   }
   else
       /* longjmp(pips_top_level, 2); */
      ljbp = top_pips_context_stack();
      longjmp(*ljbp, 2);
}

/*  returns the full tpips history file name, i.e.
 *  - $TPIPS_HISTORY (if any)
 *  - $HOME/"HIST"
 */
static char *default_hist_file_name()
{
    char *home, *hist = getenv(TPIPS_HISTENV);

    if (hist) return hist;

    /* else builds the default name. memory leak.
     */
    home = getenv("HOME");
    return sprintf((char*) malloc(sizeof(char)*(strlen(home)+strlen(HIST)+2)),
		   "%s/%s", home, HIST);
}

static char * initialize_tpips_history()
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
static void cdir_handler(char * line)
{
    user_log("%s\n", line);
    if (chdir(line+strlen(CHANGE_DIR)))
	fprintf(stderr, "error while changing directory\n");
}

static void shell_handler(char * line)
{
    line += strlen(SHELL_ESCAPE);
    while ((*line ==' ') || (*line == '\t'))
	line++;
    if (*line) {
	user_log("shell %s\n", line);
	system(line);
    }
    else
	system("sh");
}

static void echo_handler(char * line)
{
    /* skip the key word and a blank character */
    user_log("%s\n", line); 
    line += strlen(ECHO) + 1;
    fprintf(stdout,"%s\n",line);
    fflush(stdout);
}

static void help_handler(char * line)
{
    char *tmpline;
    /* skip the help word */
    line += strlen(HELP);
    /* skip blanks and tabs */
    while ((*line)==' ' || (*line)=='\t')
	line++;

    /* forget spaces afeer */
    tmpline = line;
    while ((*tmpline) && (*tmpline !=' ') && (*tmpline != '\t'))
	tmpline++;
    if (*tmpline)
	*tmpline = '\0';
    
    printf("\n");
    if (PREFIX_EQUAL_P("create",line)) {
	printf("create   <workspace-name> <file-name>...\n");
	if (*line) {
	    printf("\tcreate a new worspace from a list of fortran files\n");
	    printf("\tfirst delete the workspace if it exists\n");
	}
    }
    if (PREFIX_EQUAL_P("open",line)) {
	printf("open     <workspace-name>\n");
	if (*line) {
	    printf("\topen an existing workspace\n");
	}
    }
    if (PREFIX_EQUAL_P("close",line)) {
	printf("close    <workspace-name>\n");
	if (*line) {
	    printf("\tclose an opened workspace\n");
	}
    }
    if (PREFIX_EQUAL_P("delete",line)) {
	printf("delete   <workspace-name>\n");
	if (*line) {
	    printf("\tdelete an existing workspace\n");
	}
    }
    if (PREFIX_EQUAL_P("module",line)) {
	printf("module   <module-name>\n");
	if (*line) {
	    printf("\tselect a module from an opened workspace\n");
	}
    }
    if (PREFIX_EQUAL_P("activate",line)) {
	printf("activate <rule-name>\n");
	if (*line) {
	    printf("\ttell a rule to be active\n");
	}
    }
    if (PREFIX_EQUAL_P("make",line)) {
	printf("make     <resourcename[(OWNER)]>\n");
	if (*line) {
	    printf("\tbuild a resource\n");
	    printf("\n\tExamples:\n\n");
	    printf("\t\t make PRINTED_FILE\n");
	    printf("\t\t make CALLGRAPH_FILE(my_module)\n");
	    printf("\t\t make DG_FILE($ALL)\n");
	    printf("\t\t make ICFG_FILE($CALLEES)\n\n");
	}
    }
    if (PREFIX_EQUAL_P("apply",line)) {
	printf("apply    <rulename[(OWNER)]>\n");
	if (*line) {
	    printf("\tmake the produced resources of a rule\n");
	    printf("\n\tExamples:\n\n");
	    printf("\t\t apply PRINT_SOURCE_WITH_REGIONS\n");
	    printf("\t\t apply HPFC_CLOSE(my_module)");
	    printf("\t\t apply PRINT_CODE($ALL)\n");
	    printf("\t\t apply PRINT_ICFG($CALLEES)\n");
	}
    }
    if (PREFIX_EQUAL_P("display",line)) {
	printf("display  <resourcename[(OWNER)]>\n");
	if (*line) {
	    printf("\tprint a resource\n");
	    printf("\n\tExamples:\n\n");
	    printf("\t\t display PRINTED_FILE\n");
	    printf("\t\t display CALLGRAPH_FILE(my_module)\n");
	    printf("\t\t display DG_FILE($ALL)\n");
	    printf("\t\t display ICFG_FILE($CALLEES)\n\n");
	}
    }
    if (PREFIX_EQUAL_P("cd",line)) {
	printf("cd       <dirname>\n");
	if (*line) {
	    printf("\tchange directory\n");
	}
    }
    if (PREFIX_EQUAL_P("echo",line)) {
	printf("echo     <string>\n");
	if (*line) {
	    printf("\tprint the string\n");
	}
    }
    if (PREFIX_EQUAL_P("quit",line)) {
	printf("quit\n");
	if (*line) {
	    printf("\texit from tpips (you should close the workspace before\n");
	}
    }
    if (PREFIX_EQUAL_P("help",line)) {
	printf("help     [<help-item>]\n");
	if (*line) {
	    printf("\tprint a list of all the commands or a \"detailled\" description of one\n");
	}
    }
    if (PREFIX_EQUAL_P("\\",line)) {
	printf("shell   [<shell-function>]\n");
	if (*line) {
	    printf("\tallow shell functions call\n");
	}
    }
    if (PREFIX_EQUAL_P("rulename",line) ||
	PREFIX_EQUAL_P("rule",line)) {
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
    if (PREFIX_EQUAL_P("resourcename",line) ||
	PREFIX_EQUAL_P("resource",line)) {
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
    if (PREFIX_EQUAL_P("owner",line)) {
	printf("* owner : variable *\n");
	if (*line) {
	    printf("\tList of available owners:\n");
	    printf("\t\t$MODULE\n");
	    printf("\t\t$ALL\n");
	    printf("\t\t$PROGRAM\n");
	    printf("\t\t$CALLEES\n");
	    printf("\t\t$CALLERS\n");
	    printf("\t\t<module_name>\n");
	}
    }
    
    printf("\n");
}

static void quit_handler(char * line)
{
    /* FI: cannot be done here because debug_on() was called
       in another function. Fortunately, it does not matter. */
    /* debug_off(); */

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
    exit(0);
}

static void default_handler(char * line)
{
    /* skip blanks */
    while (((*line) == ' ') ||
	   ((*line) == '\t') ||
	   ((*line) == '\n'))
	line++;

    /* store line pointer */
    line_parsed = line_to_parse = line;

    /* cut line at the beginning of comments */
    while (*line)
    {
	if ((*line) == TPIPS_COMMENT_PREFIX)
	{
	    *line = 0;
	    break;
	}
	line++;
    }

    /* parse if non-null line */
    if (*line_to_parse) {
	tp_init_lex ();
	tp_parse ();
    }

    /* error if some characters are still here */
    if (*line_to_parse) {
 	tp_error("syntax error: cannot parse the end of the line");
    }
}

/***************************************** Handler for trivial functions ... */

struct t_handler 
{
    char * name;
    void (*function)(char *);
} ;

static struct t_handler handlers[] =
{
  { QUIT,		quit_handler },
  { CHANGE_DIR, 	cdir_handler },
  { SHELL_ESCAPE, 	shell_handler },
  { HELP,		help_handler },
  { ECHO,		echo_handler },
  { (char *) NULL, 	default_handler}
};

static void (*find_handler(char* line))(char *)
{
    struct t_handler * x = handlers;
    while ((x->name) && !PREFIX_EQUAL_P(line, x->name)) x++;
    return x->function;
}

/*********************************************************** DOING THE JOB */

static char * read_a_line(char * prompt)
{
#define MAX_LINE_LENGTH  1024
    static char line[MAX_LINE_LENGTH];
    char *logline;

    if (use_readline)
	logline = readline(prompt);
    else
	/* GO: Please FC don't put safe_fgets here or Validate it !! :-) */
	logline = fgets(line, MAX_LINE_LENGTH, current_file);

    if (logfile && logline)
	fprintf(logfile,"%s\n",logline);

    return logline;
}

static void process_a_file()
{
    char *last = NULL;
    char *line;
    jmp_buf pips_top_level;
    static readline_initialized = FALSE;

    if ((use_readline) && (readline_initialized == FALSE))
    {
	initialize_readline ();
	last = initialize_tpips_history();
	readline_initialized = TRUE;
    }

    /*  interactive loop
     */
    while ((line = read_a_line(TPIPS_PROMPT)))
    {
	if (setjmp(pips_top_level)) {
#ifdef FLEX_SCANNER
	    tp_restart(tp_in);
#endif
	}
	else {
	    push_pips_context(&pips_top_level);
	    /*   add to history if not the same as the last one.
	     */
	    if (use_readline &&
	  (line && *line && ((last && strcmp(last, line)!=0) || (!last))) &&
		(strncmp (line,QUIT,strlen (QUIT))))
	    {
		add_history(line);
		last = strdup(line);
	    }
	    /*   calls the appropriate handler.
	     */
	    (find_handler(line))(line);
	}
	pop_pips_context();
    }
}

static void parse_arguments(int argc, char * argv[])
{
    int c;
    extern char *optarg;
    extern int optind;

    while ((c = getopt(argc, argv, "nl:h")) != -1)
	switch (c)
	{
	case 'l':
	    logfile = safe_fopen (optarg,"w");
	    break;
	case 'h':
	    fprintf (stderr,usage, argv[0]);
	    return;
	    break;
	case 'n':
	    set_bool_property ("TPIPS_NO_EXECUTION_MODE", TRUE);
	    break;
	}

    if (argc == optind ) {
	use_readline = isatty(0);
	pips_debug(1, "reading from stdin, which %s a tty\n",
		   use_readline ? "is" : "is not");
	current_file = stdin;
	process_a_file();
    }
    else {
	while (optind < argc ) {
	    if (same_string_p(argv[optind], "-")) {
		current_file = stdin;
		use_readline = isatty(0);
	    }
	    else {
		if((current_file = fopen(argv[optind], "r"))==NULL) {
		    perror("tpips");
		    exit(1);
		}
		else {
		    use_readline = FALSE;
		}
	    }
	    
	    pips_debug(1, "reading from file %s\n",argv[optind]);
	    
	    process_a_file ();
	    if (!same_string_p(argv[optind], "-"))
		safe_fclose(current_file, argv[optind]);
	    
	    optind++;
	}
    }
}

/* MAIN: interactive loop and history management.
 */
int main(int argc, char * argv[])
{
    debug_on("TPIPS_DEBUG_LEVEL");

    initialize_newgen();
    initialize_sc((char*(*)(Variable))entity_local_name);
    initialize_signal_catcher();

    set_bool_property("ABORT_ON_USER_ERROR", FALSE);
    pips_log_handler = tpips_user_log;
    pips_request_handler = tpips_user_request;
    pips_error_handler = tpips_user_error;

    parse_arguments(argc, argv);

    fprintf(stdout, "\n");	/* for Ctrl-D terminations */
    quit_handler("quit");
    exit (0);			/* statement not reached ... */
}

int tpips_lex_input ()
{
    char c =*line_to_parse;

    debug(9,"tpips_lex_input","input char '%c'(0x%2x) from input\n", c, c);
    if (c)
	line_to_parse++;
    return (int) c;
}

void tpips_lex_unput(int c)
{

    debug(9,"tpips_lex_unput","unput char '%c'(0x%2x)\n",
	  c,c);
    *(--line_to_parse) = (char) c;
}

void tpips_lex_print_pos(FILE* fout)
{
    fprintf (fout,"%s\n",line_parsed);
    fprintf(fout,"%*s^\n",(int)((long) line_to_parse - (long)line_parsed),"");
}


/*
 * Tell the GNU Readline library how to complete.  We want to try to complete
 * on command names if this is the first word in the line, or on filenames
 * if not. 
 */
static void initialize_readline ()
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

/*
 * Attempt to complete on the contents of TEXT.  START and END show the
 * region of TEXT that contains the word to complete.  We can use the
 * entire line in case we want to do some simple parsing.  Return the
 * array of matches, or NULL if there aren't any. 
 */
static char **fun_completion(char *texte, int start, int end)
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

/*
 * Generator function for command completion.  STATE lets us know whether
 * to start from scratch; without any state (i.e. STATE == 0), then we
 * start at the top of the list. 
 */
static char *fun_generator(char *texte, int state)
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

/*
 * Generator function for param. completion.  STATE lets us know whether
 * to start from scratch; without any state (i.e. STATE == 0), then we
 * start at the top of the list. 
 */
static char *param_generator(char *texte, int state)
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
