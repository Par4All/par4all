/* hmmm...
 */
#ifndef _STDARG_H
/* If we have already used stdarg.h do not include: */
#include <varargs.h>
#endif
#include <setjmp.h>


#define ifdebug(l) if(the_current_debug_level>=(l))

/* these macros use the GNU extensions that allow to know the 
 * function name thru the special __FUNCTION__ macro. Also variable
 * number of arguments macros are used.
 * if not available, other macros or function calls are generated.
 * Fabien.
 */
#ifdef __GNUC__
#define pips_debug(level, format, args...)\
 ifdebug(level) fprintf(stderr, "[" __FUNCTION__  "] " format, ##args);
#define pips_user_warning(format, args...)\
  user_warning(__FUNCTION__, format, ##args)
#define pips_user_error(format, args...)\
  user_error(__FUNCTION__, format, ##args)
#define pips_assert(what, predicate)\
  if(!(predicate)){\
    (void) fprintf(stderr, \
    "[%s] (%s:%d) assertion failed\n", __FUNCTION__, __FILE__, __LINE__);\
    (void) fprintf(stderr, "\n %s not verified\n\n", what); abort();}
#define pips_internal_error(format, args...)\
  pips_error(__FUNCTION__, format, ##args)
#else
#define pips_debug pips_debug_function
#define pips_user_warning pips_user_warning_function
#define pips_user_error pips_user_error_function
#define pips_internal_error pips_internal_error_function
#define pips_assert(what, predicate)\
  if(!(predicate)){\
    (void) fprintf(stderr, \
    "(%s:%d) assertion failed\n", __FILE__, __LINE__);\
    (void) fprintf(stderr, "\n %s not verified\n\n", what); abort();}
#endif

/* USE message_assert or assert instead, defined in newgen.
 */
/* #define pips_assert(f, p) pips_assert_function((f), (p), __LINE__, __FILE__)
 */

#define same_string_p(s1, s2) (strcmp((s1), (s2)) == 0)

/* Constant used to dimension arrays in wpips and pipsmake */
#define ARGS_LENGTH 512

/* MAXPATHLEN is defined in <sys/param.h> for SunOS... but not for all OS! */
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

/* hmmm. ???
 */
extern char *re_comp();
extern int re_exec();
