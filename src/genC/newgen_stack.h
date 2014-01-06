/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of NewGen.

  NewGen is free software: you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation, either version 3 of the License, or any later version.

  NewGen is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
  License for more details.

  You should have received a copy of the GNU General Public License along with
  NewGen.  If not, see <http://www.gnu.org/licenses/>.

*/
/*  STACK MANAGEMENT -- headers
 *
 *  - a stack is declared with type stack (internals not visible from here!)
 *  - a stack_iterator allows to iterate over the items in a stack.
 *  - allocation with stack_make(newgen domain, bucket size)
 *  - free with stack_free(stack)
 *  - stack_size(stack) returns the number of elements stacked
 *  - stack_empty_p(stack) tells whether the stack is empty or not
 *    stack_empty_p(stack)==(stack_size(stack)==0)
 *  - stack_{push,pop,head,replace} do what you may expect from them
 *  - stack_info gives informations about the stack
 *  - stack_map applies the function on all the items in stack.
 *  - stack_iterator_{init,next_and_go,end} to iterate.
 *  - see STACK_MAP for instance.
 *
 *  newgen_assert should be included before.
 *
 *  Fabien COELHO 05/12/1994
 */

#ifndef STACK_INCLUDED
#define STACK_INCLUDED

/*  encapsulated types
 */
typedef struct __stack_head *stack;
typedef struct __stack_iterator *stack_iterator;

/*  defines for empty values
 */
#define STACK_NULL ((stack) NULL)
#define STACK_NULL_P(s) ((s)==STACK_NULL)

#define stack_undefined  ((stack)-14)
#define stack_undefined_p(s) ((s)==stack_undefined)

#define STACK_CHECK(s)						\
  message_assert("stack null or undefined",			\
                 !STACK_NULL_P(s) && !stack_undefined_p(s))

/*   allocation
 */
extern stack stack_make (int, int, int); /* type, bucket_size, policy */
extern void stack_free (stack*);
extern stack stack_copy (const stack);

/*   observers
 */
extern int stack_size(const stack);
extern int stack_type(const stack);
extern int stack_bsize(const stack);
extern int stack_policy(const stack);
extern int stack_max_extent(const stack);

/*   miscellaneous
 */
extern int stack_consistent_p(const stack);
extern bool stack_empty_p(const stack);
extern void stack_info(FILE*, const stack);
extern void stack_map(const stack, gen_iter_func_t);

/*   stack use
 */
extern void stack_push(void*, stack);
extern void *stack_pop(stack);
extern void *stack_head(const stack);
extern void *stack_nth(const stack, int);
extern void *stack_replace(void*, stack);

/*   stack iterator
 *
 *   This way the stack type is fully encapsulated, but
 *   it is not very efficient, due to the many function calls.
 *   Consider "stack_map" first which has a very small overhead.
 */
extern stack_iterator stack_iterator_init(const stack, bool); /* X-ward */
extern bool stack_iterator_next_and_go(stack_iterator, void**);
extern void stack_iterator_end(stack_iterator*);
extern bool stack_iterator_end_p(stack_iterator); /* not needed */

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
  STACK_MAP_X(_item, _itemtype, _code, _stack, true)
#endif

/*  That is all
 */
