#ifndef SPAGHETTIFY_DEFS
#define SPAGHETTIFY_DEFS

#define INDEX_VARIABLE_NAME "INDEX"
#define BEGIN_VARIABLE_NAME "BEGIN"
#define END_VARIABLE_NAME "END"
#define INCREMENT_VARIABLE_NAME "INCREMENT"

statement spaghettify_loop (statement stat, string module_name);

statement spaghettify_whileloop (statement stat, string module_name);

statement spaghettify_forloop (statement stat, string module_name);

statement spaghettify_test (statement stat, string module_name);

#endif
