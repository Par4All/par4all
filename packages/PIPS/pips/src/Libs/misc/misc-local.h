/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

#include <stdarg.h>
#include <stdlib.h>
#include <setjmp.h>

/* To generate a string from a macro: */
#define STRINGIFY_SECOND_STAGE(symbol) #symbol
/* If not using this 2-stage macro evaluation, the generated string is not
   the value of the macro but the name of the macro... Who said C was a
   simple language? :-/ */
#define STRINGIFY(symbol) STRINGIFY_SECOND_STAGE(symbol)


/* Measurement type for mem_spy.c */
typedef enum {SBRK_MEASURE, NET_MEASURE, GROSS_MEASURE} measurement_type;


#define pips_unknown_function "Unknown Function Name"

/* these macros use the GNU extensions that allow to know the
 * function name thru the special __FUNCTION__ macro. Also variable
 * number of arguments macros are used.
 * if not available, other macros or function calls are generated.
 * Fabien.
 */
#if (defined(__GNUC__))
#define pips_where(out) \
  fprintf(out, "[%s] (%s:%d) ", __FUNCTION__, __FILE__, __LINE__)
#define debug_on(env) debug_on_function(env, __FUNCTION__, __FILE__, __LINE__)
#define debug_off() debug_off_function(__FUNCTION__, __FILE__, __LINE__)
/* Use the old "do {...} while(0)" C hack to allow a macro with arbitrary
   code to be used everywhere: */
#define pips_debug(level, format, args...)				\
  do {									\
    ifdebug(level) fprintf(stderr, "[%s] " format, __FUNCTION__ ,	\
			   ##args);					\
  } while(0)
#define pips_user_warning(format, args...)\
  user_warning(__FUNCTION__, format, ##args)
#define pips_user_error(format, args...)\
  user_error(__FUNCTION__, format, ##args)
#define pips_user_irrecoverable_error(format, args...)\
  user_irrecoverable_error(__FUNCTION__, format, ##args)
#define pips_internal_error(format, args...)\
  pips_error(__FUNCTION__, "(%s:%d) " format, __FILE__ , __LINE__ , ##args)
#else
#define pips_where(out) \
  fprintf(out, "[%s] (%s:%d) ", pips_unknown_function, __FILE__, __LINE__)
#define debug_on(env) \
  debug_on_function(env, pips_unknown_function, __FILE__, __LINE__)
#define debug_off() \
  debug_off_function(pips_unknown_function, __FILE__, __LINE__)
#define pips_debug pips_debug_function
#define pips_user_warning pips_user_warning_function
#define pips_user_error pips_user_error_function
#define pips_user_irrecoverable_error user_irrecoverable_error_function
#define pips_internal_error pips_internal_error_function
#endif

/* common macros, two flavors depending on NDEBUG */
#ifdef NDEBUG

#define pips_assert(what, predicate)
#define pips_user_assert(what, predicate)
#define ifdebug(l) if(0)

#else

#define pips_assert(what, predicate)					\
  do {									\
    if(!(predicate)) {							\
      (void)pips_internal_error("assertion failed\n\n '%s' not verified\n\n",what);\
      abort();								\
    }									\
  } while(0)

#define pips_user_assert(what, predicate)				\
  do {									\
    if(!(predicate)) {							\
      (void)pips_internal_error("assertion failed\n\n '%s' not verified\n\n",what);\
      pips_user_error("this is a USER ERROR, I guess\n");		\
    };									\
  } while(0)

#define ifdebug(l) if(the_current_debug_level>=(l))

#endif

#define pips_exit(code, format, args...)\
   pips_user_warning(format, ##args), exit(code)

/* FI:need to breakpoint while inlining is available */
/* #define same_string_p(s1, s2) (strcmp((s1), (s2)) == 0)*/
#define same_string_p(s1, s2) function_same_string_p(s1,s2)
#define same_stringn_p(a,b,c) (!strncmp((a),(b),(c)))

/* MAXPATHLEN is defined in <sys/param.h> for SunOS... but not for all OS! */
#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

#define PIPS_CATCH(what) \
   if (push_debug_status(), \
       setjmp(*push_exception_on_stack(what, __CURRENT_FUNCTION_NAME__, \
      	                    __FILE__, __LINE__, pop_debug_status)))

/* SG moved there from transformation.h */
#define SIGN_EQ(a,b) ((((a)>0 && (b)>0) || ((a)<0 && (b)<0)) ? true : false)
#define FORTRAN_DIV(n,d) (SIGN_EQ((n),(d)) ? ABS(n)/ABS(d) : -(ABS(n)/ABS(d)))
#define C_DIVISION(n,d) ((n)/(d))
#define FORTRAN_MOD(n,m) (SIGN_EQ((n),(m)) ? ABS(n)%ABS(m) : -(ABS(n)%ABS(m)))
#define C_MODULO(n,m) ((n)%(m))

// redundant declaration to ease bootstrapping
extern int the_current_debug_level;
