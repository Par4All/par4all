/*
 * $Id$
 */

typedef struct {
    gen_array_t the_owners;
    string the_name;
} res_or_rule;

extern int tp_lex();
extern int tp_parse();
extern void tp_error();
extern void tp_init_lex();
extern void tp_begin_key();
extern void tp_begin_fname();
extern void close_workspace_if_opened();

extern FILE * tp_in;
#ifdef FLEX_SCANNER
extern void tp_restart(FILE *);
#endif

#define TPIPS_PRIMARY_PROMPT 	"tpips> " /* prompt for readline  */
#define TPIPS_REQUEST_PROMPT    "tpips-request> "
#define TPIPS_SECONDARY_PROMPT 	"> "
#define TPIPS_CONTINUATION_CHAR '\\'

#define TPIPS_HISTENV "TPIPS_HISTORY"	/* history file env variable */
#define TPIPS_HISTORY_LENGTH 100	/* max length of history file */
#define TPIPS_COMMENT_PREFIX '#'	/* comment prefix */
#define TPIPS_HISTORY ".tpips.history" 	/* default history file */

#define SHELL_ESCAPE "shell" 		/* ! used for history reference */
#define CHANGE_DIR   "cd"

#define SET_ENV	     "setenv"
#define GET_ENV	     "getenv"

#define SET_PROP     "setproperty"
#define GET_PROP     "getproperty"

#define QUIT         "quit"
#define HELP         "help"
#define ECHO         "echo"

#define skip_blanks(str) \
  while (*str && (*str==' ' || *str=='\t' || *str=='\n')) str++
