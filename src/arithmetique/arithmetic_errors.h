/* 
 * $Id$
 *
 * managing arithmetic errors...
 * detecting and managing arithmetic errors on Values should be
 * systematic. These macros gives a C++ look and feel to this
 * management. 
 *
 * (c) CA et FC, Sept 1997
 *
 * $Log: arithmetic_errors.h,v $
 * Revision 1.18  1998/10/24 09:21:45  coelho
 * const added to constants.
 *
 */
#include <setjmp.h>

/* the index points to the first available chunck for a new context...
 */

#define MAX_STACKED_CONTEXTS 20
extern jmp_buf global_exception_stack[MAX_STACKED_CONTEXTS];
extern int     global_exception_type[MAX_STACKED_CONTEXTS];
extern int     global_exception_index;
extern int     global_exception_thrown;

extern const unsigned int overflow_error;
extern const unsigned int simplex_arithmetic_error;
extern const unsigned int user_exception_error;
extern const unsigned int any_exception_error;

/* declaration of "exception"  to keep  2 potential types:
   extern int or extern jmp_buf
*/

/* set DEBUG_GLOBAL_EXCEPTIONS for debugging information
 */

#if defined(DEBUG_GLOBAL_EXCEPTIONS)
#define exception_debug(msg, n, what) 			\
  fprintf(stderr, "%s %d - %d (%s %s %d)\n", 		\
      msg, what, n, __FUNCTION__, __FILE__, __LINE__),
#else
#define exception_debug(msg, n, what) 
#endif

#define exception_debug_push(what) \
  exception_debug("PUSH ", global_exception_index, what)
#define exception_debug_pop(what) \
  exception_debug("POP  ", global_exception_index-1, what)
#define exception_debug_throw(what) \
  exception_debug("THROW", global_exception_index-1, what)

#define EXCEPTION extern unsigned int

#define global_exception_index_decr                                     \
    (global_exception_index > 0 ? --global_exception_index:             \
     (print_exception_stack_error(0),1))

#define THROW(what) \
    (exception_debug_throw(what) throw_exception(what))

#define PUSH_AND_FORWARD_EXCEPTION(what)				\
    (exception_debug_push(what) 					\
      global_exception_index==MAX_STACKED_CONTEXTS?			\
     (print_exception_stack_error(1),1):	                        \
     (global_exception_type[global_exception_index]=what,		\
      setjmp(global_exception_stack[global_exception_index++])))

#define CATCH(what) if PUSH_AND_FORWARD_EXCEPTION(what)

#define UNCATCH(what)						\
    (exception_debug_pop(what) 					\
     global_exception_type[global_exception_index_decr]!=what?	\
	(print_exception_stack_error(2), 0): 1)

#define TRY else

/* end of it.
 */
