/*  STACK MANAGEMENT -- headers
 *  
 * $RCSfile: newgen_stack.h,v $ version $Revision$
 * $Date: 1995/02/01 10:50:03 $, 
 * got on %D%, %T%
 *
 *  - a stack is declared with type stack (internals not visible from here)
 *  - it is allocated with stack_make(newgen domain, bulk size)
 *  - it is freed with stack_free(stack)
 *  - stack_size(stack) returns the size
 *  - stack_empty_p(stack) tells whether the stack is empty or not
 *  - stack_{push,pop,head,replace} do what you may expect from them
 *
 *  Fabien COELHO 05/12/94
 */

#ifndef STACK_INCLUDED
#define STACK_INCLUDED

struct __stack_head;
typedef struct __stack_head *stack;

#define STACK_NULL ((stack) NULL)
#define STACK_NULL_P(s) ((s)==STACK_NULL)

#define stack_undefined  ((stack)-14)
#define stack_undefined_p(s) ((s)==stack_undefined)

extern stack stack_make GEN_PROTO((int, int));
extern void stack_free GEN_PROTO((stack));

extern bool stack_empty_p GEN_PROTO((stack));
extern int stack_size GEN_PROTO((stack));

extern void stack_push GEN_PROTO((char*, stack));
extern char *stack_pop GEN_PROTO((stack));
extern char *stack_head GEN_PROTO((stack));
extern char *stack_replace GEN_PROTO((char*, stack));

extern void stack_info GEN_PROTO((FILE*, stack));

#endif

/*  That is all
 */
