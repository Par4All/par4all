/*  STACK MANAGEMENT -- headers
 *  
 * $RCSfile: newgen_stack.h,v $ version $Revision$
 * $Date: 1995/02/02 18:10:38 $, 
 * got on %D%, %T%
 *
 *  - a stack is declared with type stack (internals not visible from here!)
 *  - a stack_iterator allows to iterate over the items in a stack.
 *  - allocation with stack_make(newgen domain, bulk size)
 *  - free with stack_free(stack)
 *  - stack_size(stack) returns the size
 *  - stack_empty_p(stack) tells whether the stack is empty or not
 *  - stack_{push,pop,head,replace} do what you may expect from them
 *  - stack_info gives informations about the stack
 *  - stack_map applies the function on all the items in stack.
 *  - stack_iterator_{init,next,end_p,clean} to iterate.
 *  - see STACK_MAP for instance.
 *
 *  Fabien COELHO 05/12/94
 */

#ifndef STACK_INCLUDED
#define STACK_INCLUDED

/*  encapsulated types
 */
struct __stack_head;
typedef struct __stack_head *stack;
struct __stack_iterator;
typedef struct __stack_iterator *stack_iterator;

/*  defines for empty values
 */
#define STACK_NULL ((stack) NULL)
#define STACK_NULL_P(s) ((s)==STACK_NULL)

#define stack_undefined  ((stack)-14)
#define stack_undefined_p(s) ((s)==stack_undefined)

/*   allocation
 */
extern stack stack_make GEN_PROTO((int, int)); /* type, size */
extern void stack_free GEN_PROTO((stack));

/*   miscellaneous
 */
extern void stack_map GEN_PROTO((stack, void(*)()));
extern int stack_empty_p GEN_PROTO((stack));
extern int stack_size GEN_PROTO((stack));
extern void stack_info GEN_PROTO((FILE*, stack));

/*   stack use
 */
extern void stack_push GEN_PROTO((char*, stack));
extern char *stack_pop GEN_PROTO((stack));
extern char *stack_head GEN_PROTO((stack));
extern char *stack_replace GEN_PROTO((char*, stack));

/*   stack iterator
 *
 *   This way the stack type is fully encapsulated, but
 *   it is not very efficient, due to the many function calls.
 *   Consider gen_map first which has a very small overhead.
 */
extern stack_iterator stack_iterator_init GEN_PROTO((stack));
extern char *stack_iterator_next GEN_PROTO((stack_iterator));
extern int stack_iterator_end_p GEN_PROTO((stack_iterator));
extern void stack_iterator_clean GEN_PROTO((stack_iterator*));

/* applies _code on the items of _stack downward , with _item of _itemtype.
 */
#define STACK_MAP(_item, _itemtype, _code, _stack) \
{\
    stack_iterator _i = stack_iterator_init(_stack);\
    _itemtype _item;\
    while(!(stack_iterator_end_p(_i)))\
    {\
	_item = (_itemtype) stack_iterator_next(_i);\
	_code;\
    }\
    stack_iterator_clean(&_i);\
}

#endif

/*  That is all
 */
