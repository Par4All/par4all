/*  STACK MANAGEMENT -- headers
 *
 *  - a stack is declared with type gen_stack
 *  - it is allocated with gen_stack_make(newgen domain, bulk size)
 *  - it is freed with gen_stack_free(stack)
 *  - gen_stack_size(stack) returns the size
 *  - gen_stack_empty_p(stack) tells whether the stack is empty or not
 *  - gen_push, pop, head, replace do what you may expect from them
 *
 *  Fabien COELHO 05/12/94
 */

#ifndef STACK_INCLUDED
#define STACK_INCLUDED

typedef struct gen_stack_bulk
{
  int n_item;                 /* next available item in the bulk */
  int max_items;              /* the maximun number of items of this bulk */
  chunk **items;              /* the items (only pointers at the moment) */
  struct gen_stack_bulk *succ;/* the next bulk */
}
  gen_stack_bulk, *gen_stack_ptr;

typedef struct 
{
  int size;
  int type; /* as BASIC, LIST, EXTERNAL and so on */
  int max_extent;
  gen_stack_ptr stack;
  gen_stack_ptr available;
}
  gen_stack_head, *gen_stack;

#define GEN_STACK_NULL ((gen_stack) NULL)
#define GEN_STACK_NULL_P(s) ((s)==GEN_STACK_NULL)

#define gen_stack_undefined  ((gen_stack)-14)
#define gen_stack_undefined_p(s) ((s)==gen_stack_undefined)

extern gen_stack gen_stack_make GEN_PROTO((int, int));
extern void gen_stack_free GEN_PROTO((gen_stack));

extern bool gen_stack_empty_p GEN_PROTO((gen_stack));
extern int gen_stack_size GEN_PROTO((gen_stack));

extern void gen_push GEN_PROTO((chunk*, gen_stack));
extern chunk *gen_pop GEN_PROTO((gen_stack));
extern chunk *gen_head GEN_PROTO((gen_stack));
extern chunk *gen_replace GEN_PROTO((chunk*, gen_stack));

#endif

/*  That is all
 */
