
typedef struct _res_or_rule {
	list the_owners;
	string the_name;
}res_or_rule;

extern int tp_lex();
extern int tp_parse();
extern void tp_init_lex();
extern void tp_begin_key();
extern void tp_begin_fname();
extern void close_workspace_if_opened();
