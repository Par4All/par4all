/* 
 * $Id$
 *
 * managing arithmetic errors...
 * detecting and managing arithmetic errors on Values should be
 * systematic. These macros gives a C++ look and feel to this
 * management. 
 *
 * (c) CA et FC, Sept 1997
 */
#include <setjmp.h>

/* the index points to the first available chunck for a new context...
 */

#define MAX_STACKED_CONTEXTS 20
extern jmp_buf global_exception_stack[MAX_STACKED_CONTEXTS];
extern int     global_exception_type[MAX_STACKED_CONTEXTS];
extern int     global_exception_index;
extern int     global_exception_thrown;

extern int overflow_error;
extern int  simplex_arithmetic_error;

/* declaration of "exception"  to keep  2 potential types:
   extern int or extern jmp_buf
*/

#define DEBUG_LINEAR_EXCEPTIONS

#if defined(DEBUG_LINEAR_EXCEPTIONS)
#define exception_debug(msg, what) 			\
  fprintf(stderr, "%s %d (%s %s %d)\n", 		\
	  msg, what, __FUNCTION__, __FILE__, __LINE__)
#else
#define exception_debug(msg, what) 1
#endif

#define exception_push(what) exception_debug("PUSH", what)
#define exception_pop(what)  exception_debug("POP", what)

#define EXCEPTION extern int

#define global_exception_index_decr                                     \
    (global_exception_index > 0 ? --global_exception_index:             \
     (print_exception_stack_error(0),1))

#define THROW(what) \
    (throw_exception(what))

#define PUSH_AND_FORWARD_EXCEPTION(what)				\
    (exception_push(what), 						\
      global_exception_index==MAX_STACKED_CONTEXTS?			\
     (print_exception_stack_error(1),1):	                        \
     (global_exception_type[global_exception_index]=what,		\
      setjmp(global_exception_stack[global_exception_index++])))

#define CATCH(what) if PUSH_AND_FORWARD_EXCEPTION(what)

#define UNCATCH(what)						\
    (exception_pop(what), 					\
     global_exception_type[global_exception_index_decr]!=what?	\
	print_exception_stack_error(2): 1)

#define TRY else

/* end of $RCSfile: arithmetic_errors.h,v $
 */
