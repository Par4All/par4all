/* Legal characters to start a comment line
 *
 * '\n' is added to cope with empty lines
 * Empty lines with SPACE and TAB characters 
 * are be preprocessed and reduced to an empty line by GetChar().
 */
#define START_COMMENT_LINE "CcDd*!#\n"

extern FILE * syn_in; /* the file read in by the scanner */

/* definition of implementation dependent constants */

#include "constants.h"

#define HASH_SIZE 1013
#define FORMATLENGTH (4096)
#define LOCAL static

#ifndef abs
#define abs(v) (((v) < 0) ? -(v) : (v))
#endif

/* extern char * getenv(); */

#define Warning(f,m) \
(user_warning(f,"Warning between lines %d and %d\n%s\n",line_b_I,line_e_I,m) )

#define FatalError(f,m) \
(pips_error(f,"Fatal error between lines %d and %d\n%s\n",line_b_I,line_e_I,m))

extern char * strdup(const char *);
