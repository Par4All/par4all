 /* include file for transformer library */

/* prefix used for value entity names; no conflict should occur with user
 * function names as long as they are restricted to 6 characters
 */
#define SEMANTICS_MODULE_NAME "SEMANTICS"

#define SEMANTICS_SEPARATOR '#'

/* internal entity names (FI: I should have used suffixes to be consistent with external
 * suffixes */
#define OLD_VALUE_PREFIX "o#"
#define INTERMEDIATE_VALUE_PREFIX "i#"
/* external suffixes */
#define NEW_VALUE_SUFFIX "#new"
#define OLD_VALUE_SUFFIX "#init"
#define INTERMEDIATE_VALUE_SUFFIX "#int"

/*VARARGS2*/
void debug();

/*VARARGS2*/
void pips_error();

/*VARARGS2*/
char * concatenate();
