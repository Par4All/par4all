/* From readmakefile.y sunce the .y is not passed through cproto (RK): */
extern makefile parse_makefile();
extern rule find_rule_by_phase(string pname);
extern bool close_makefile();
extern makefile open_makefile();
typedef bool (*pipsmake_callback_handler_type)(void);

