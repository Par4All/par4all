 /* include file for transformer library */

/* prefix used for value entity names; no conflict should occur with user
 * function names as long as they are restricted to 6 characters
 */
#define SEMANTICS_MODULE_NAME "*SEMANTICS*"

/* Must be used in suffixes and prefixes below */
#define SEMANTICS_SEPARATOR '#'

/* internal entity names (FI: I should have used suffixes to be consistent with external
 * suffixes */
#define OLD_VALUE_PREFIX "o#"
#define INTERMEDIATE_VALUE_PREFIX "i#"
#define TEMPORARY_VALUE_PREFIX "t#"

/* external suffixes (NEW_VALUE_SUFFIX is not used, new values are represented
 * by the variable itself, i.e. new value suffix is the empty string "")
 */
#define NEW_VALUE_SUFFIX "#new"
#define OLD_VALUE_SUFFIX "#init"
#define INTERMEDIATE_VALUE_SUFFIX "#int"
