extern char* (*default_variable_to_string)(Variable);
extern int sc_debug_level;
#define get_sc_debug_level() sc_debug_level
#define ifscdebug(l) if (get_sc_debug_level()>=l)


