/* $RCSfile: tpips-local.h,v $ (version $Revision$)
 * $Date: 1997/03/07 14:14:07 $, 
 */

/* FI: temporary storage of file names could be avoided with better yacc rules
 * 272 files in Zebulon
 */
#define FILE_LIST_MAX_LENGTH 300

typedef struct _t_file_list {
	int argc;
	char *argv[FILE_LIST_MAX_LENGTH];
}t_file_list;

typedef struct _res_or_rule {
	list the_owners;
	string the_name;
}res_or_rule;

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
#define TPIPS_SECONDARY_PROMPT 	"> "
#define TPIPS_CONTINUATION_CHAR '\\'

#define TPIPS_HISTENV "TPIPS_HISTORY"	/* history file env variable */
#define TPIPS_HISTORY_LENGTH 100	/* max length of history file */
#define TPIPS_COMMENT_PREFIX '#'	/* comment prefix */
#define TPIPS_HISTORY ".tpips.history" 	/* default history file */
#define TPIPS_REQUEST_BUFFER_LENGTH 100

#define SHELL_ESCAPE "shell" 		/* ! used for history reference */
#define CHANGE_DIR   "cd "
#define SET_ENV	     "setenv "
#define SET_PROP     "setproperty "

#define GET_PROP     "getproperty"
#define QUIT         "quit"
#define HELP         "help"
#define ECHO         "echo"

/* end of $RCSfile: tpips-local.h,v $ 
 */
