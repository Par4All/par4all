
#define FILE_LIST_MAX_LENGTH 10

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
