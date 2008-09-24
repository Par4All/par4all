/*  STACK MANAGEMENT -- headers
 *
 * $Id$
 *
 *  - a stack is declared with type stack (internals not visible from here!)
 *  - a stack_iterator allows to iterate over the items in a stack.
 *  - allocation with stack_make(newgen domain, bucket size)
 *  - free with stack_free(stack)
 *  - stack_size(stack) returns the size
 *  - stack_empty_p(stack) tells whether the stack is empty or not
 *  - stack_{push,pop,head,replace} do what you may expect from them
 *  - stack_info gives informations about the stack
 *  - stack_map applies the function on all the items in stack.
 *  - stack_iterator_{init,next_and_go,end} to iterate.
 *  - see STACK_MAP for instance.
 *
 *  newgen_assert should be included before.
 *
 *  Fabien COELHO 05/12/94
 */

#ifndef STACK_INCLUDED
#define STACK_INCLUDED

#ifdef __STDC__
#define _PROTO(x) x
#else
#define _PROTO(x) ()
#endif

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

#define STACK_CHECK(s)\
  message_assert("stack null or undefined",\
                 !STACK_NULL_P(s) && !stack_undefined_p(s))

/*   allocation
 */
extern stack stack_make _PROTO((int, int, int)); /* type, bucket_size, policy */
extern void stack_free _PROTO((stack*));
extern stack stack_copy _PROTO((stack));

/*   observers
 */
extern int stack_size _PROTO((stack));
extern int stack_type _PROTO((stack));
extern int stack_bsize _PROTO((stack));
extern int stack_policy _PROTO((stack));
extern int stack_max_extent _PROTO((stack));

/*   miscellaneous
 */
extern int stack_consistent_p _PROTO((stack));
extern int stack_empty_p _PROTO((stack));
extern void stack_info _PROTO((FILE*, stack));
extern void stack_map _PROTO((stack, void(*)()));

/*   stack use
 */
extern void stack_push _PROTO((void*, stack));
extern void *stack_pop _PROTO((stack));
extern void *stack_head _PROTO((stack));
extern void *stack_replace _PROTO((void*, stack));

/*   stack iterator
 *
 *   This way the stack type is fully encapsulated, but
 *   it is not very efficient, due to the many function calls.
 *   Consider gen_map first which has a very small overhead.
 */
extern stack_iterator stack_iterator_init _PROTO((stack, int)); /* X-ward */
extern int stack_iterator_next_and_go _PROTO((stack_iterator, void**));
extern void stack_iterator_end _PROTO((stack_iterator*));
extern int stack_iterator_end_p _PROTO((stack_iterator)); /* not needed */

/* applies _code on the items of _stack downward , with _item of _itemtype.
 */
#define STACK_MAP_X(_item, _itemtype, _code, _stack, _downwards)	\
  {									\
    stack_iterator _i = stack_iterator_init(_stack, _downwards);	\
    void * _vs_item;							\
    while (stack_iterator_next_and_go(_i, &_vs_item))			\
    {									\
      _itemtype _item = (_itemtype) _vs_item;				\
      _code;								\
    }									\
    stack_iterator_end(&_i);						\
  }

#define STACK_MAP(_item, _itemtype, _code, _stack)	\
  STACK_MAP_X(_item, _itemtype, _code, _stack, 1)

#undef _PROTO
#endif

/*  That is all
 */
