#ifndef _STDARG_H
/* If we have already used stdarg.h do not include: */
#include <varargs.h>
#endif
#include <setjmp.h>

#define ifdebug(l) if(the_current_debug_level>=(l))

/* a debug macro that generates automatically the function name if available.
 */
#ifdef __GNUC__
#define pips_debug(level, format, args...)\
 ifdebug(level) fprintf(stderr, "[" __FUNCTION__  "] " format, ##args);
#else
#define pips_debug pips_debug_function
#endif

#define pips_assert(f, p) pips_assert_function((f), (p), __LINE__, __FILE__)

#define same_string_p(s1, s2) (strcmp((s1), (s2)) == 0)

/* Constant used to dimension arrays in wpips and pipsmake */
#define ARGS_LENGTH 512

/* MAXPATHLEN is defined in <sys/param.h> for SunOS... but not for all OS! */
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

extern char *re_comp();
extern int re_exec();
