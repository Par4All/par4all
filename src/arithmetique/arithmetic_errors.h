/* $RCSfile: arithmetic_errors.h,v $ (version $Revision$)
 * $Date: 1997/09/08 19:51:03 $, 
 *
 * managing arithmetic errors...
 * detecting and managing arithmetic errors on Values should be
 * systematic. These macros gives a C++ look and feel to this
 * management. 
 *
 * (c) FC et CA , Sept 1997
 */
#include <setjmp.h>

/* the index points to the first available chunck for a new context...
 */

#define MAX_STACKED_CONTEXTS 20
extern jmp_buf global_exception_stack[MAX_STACKED_CONTEXTS];
extern int     global_exception_type[MAX_STACKED_CONTEXTS];
extern int     global_exception_index = 0;
extern int     global_exception_thrown = 0;

extern int overflow_error = 1;
extern int  simplex_arithmetic_error=2;

/* declaration of "exception"  to keep  2 potential types:
   extern int or extern jmp_buf
*/

#define EXCEPTION extern int

#define global_exception_index_decr                                     \
    (global_exception_index > 0 ? --global_exception_index:             \
     (print_exception_stack_error(0),1))

#define THROW(what) \
    (throw_exception(what))

#define PUSH_AND_FORWARD_EXCEPTION(what)				\
    (global_exception_index==MAX_STACKED_CONTEXTS?			\
     (print_exception_stack_error(1),1):	                        \
     (global_exception_type[global_exception_index]=what,		\
      setjmp(global_exception_stack[global_exception_index++])))

#define CATCH(what) if PUSH_AND_FORWARD_EXCEPTION(what)

#define UNCATCH(what) \
    (global_exception_type[global_exception_index_decr]==what)

#define TRY else

/* end of $RCSfile: arithmetic_errors.h,v $
 */
