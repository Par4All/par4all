/* $RCSfile: tpips-local.h,v $ (version $Revision$)
 * $Date: 1996/12/30 17:52:47 $, 
 */
#define FILE_LIST_MAX_LENGTH 200

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

/* end of $RCSfile: tpips-local.h,v $ 
 */
