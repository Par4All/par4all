/* $RCSfile: errors.c,v $ (version $Revision$)
 * $Date: 1996/08/09 17:57:14 $, 
 */

#include <stdio.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"

/* the index points to the first available chunck for a new context...
 */
#define MAX_STACKED_CONTEXTS 20
jmp_buf global_exception_stack[MAX_STACKED_CONTEXTS];
int     global_exception_type[MAX_STACKED_CONTEXTS];
int     global_exception_index = 0;
int     global_exception_thrown = 0;

#define global_exception_index_decr
    (global_exception_index>0? --global_exception_index:
     (fprintf(stacks, "global exception stack underflow\n"), abort, -1))

#define THROW_EXCEPTION 
    longjmp(global_exception_stack[global_exception_index_decr])

#define THROW(what) 
    (global_exception_thrown=what, THROW_EXCEPTION)

#define PUSH_ANY_EXCEPTION() ???

#define PUSH_AND_FORWARD_EXCEPTION(what)				\
    (global_exception_index==MAX_STACKED_CONTEXTS?			\
     (fprintf(stacks, "global exception stack overflow\n"), abort, -1):	\
     (global_exception_type[global_exception_index]=what,		\
      setjmp(global_exception_stack[global_exception_index++]) &&	\
      global_exception_type[global_exception_index]==what? 		\
      1: THROW_EXCEPTION))

#define CATCH(what) if PUSH_AND_FORWARD_EXCEPTION(what)

#define UNCATCH(what) \
    (global_exception_type[global_exception_index_decr]==what)

#define TRY else

#define RETHROW(what) rethrow_if_any(what)

/* forward if possible?
 * this is not compatible with the c++ behavior.
 * moreover it's an exensive default behavior (the whole stack
 * must be checked as the common behavior under an exception)
 */
void 
rethrow_if_any(int what) 
{
    int i=0;
    assert(global_exception_thrown==what);
    for(; i<global_exception_index; i++)
	if (global_exception_type[global_exception_index]==what) 
	    THROW_EXCEPTION;
    global_exception_thrown=0;
}

/* end of $RCSfile: errors.c,v $
 */
