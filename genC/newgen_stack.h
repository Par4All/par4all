/*  STACK MANAGEMENT -- headers
 *
 *  - a stack is declared with type stack
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

/* the stack bulks, that is arrays containing the elements
 */
typedef struct _stack_bulk
{
  int n_item;                 /* next available item in the bulk */
  int max_items;              /* the maximun number of items of this bulk */
  char **items;               /* the items (only pointers at the moment) */
  struct _stack_bulk *succ;   /* the next bulk */
}
  _stack_bulk, *_stack_ptr;

/*  the stack head
 */
typedef struct 
{
  int size;
  int type; /* as BASIC, LIST, EXTERNAL, CHUNK and so on */
  int max_extent;
  _stack_ptr stack;
  _stack_ptr available;
}
  _stack_head, *stack;

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

#endif

/*  That is all
 */
