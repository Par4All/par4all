/*
 * STACK MANAGEMENT
 *
 * Fabien COELHO, 05/12/94 
 *
 * Could be integrated in Newgen as a building type (as lists, mappings...).
 * there is no actual need of such a type on the functional point of view.
 * I put it there since it may be much more efficient than lists.
 *
 * More thoughts needed. I should had an iterator...
 *
 * $RCSfile: stack.c,v $ version $Revision$
 * $Date: 1995/02/01 10:59:09 $, 
 * got on %D%, %T%
 */

#include <stdio.h>
extern int fprintf();
#include "malloc.h"
#include "newgen_assert.h"
#include "newgen_types.h"

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
typedef struct __stack_head
{
  int size;
  int type; /* as BASIC, LIST, EXTERNAL, CHUNK and so on */
  int max_extent;
  _stack_ptr stack;
  _stack_ptr available;
}
  _stack_head; /* and *stack */

#include "newgen_stack.h"

/*  usefull defines
 */

#define STACK_PTR_NULL ((_stack_ptr) NULL)
#define STACK_PTR_NULL_P(s) ((s)==STACK_PTR_NULL)

#define STACK_DEFAULT_SIZE 30

/*  tools
 */
static int number_of_bulks(x)
_stack_ptr x;
{
    int n=0;
    for(; !STACK_PTR_NULL_P(x); x=x->succ, n++);
    return(n);
}

/* allocates a bulk of size size
 */
static _stack_ptr allocate_bulk(size)
int size;
{
    _stack_ptr x = (_stack_ptr) malloc(sizeof(_stack_bulk));
    
    x->n_item = 0;
    x->max_items = size;
    x->items = (char **) malloc(sizeof(char *)*size);
    x->succ = STACK_PTR_NULL;

    return(x);
}

/* search for a new bulk, first in the available list,
 * if none are available, a new bulk is allocated
 */
static _stack_ptr find_or_allocate(s)
stack s;
{
    if (!STACK_PTR_NULL_P(s->available))
    {
	_stack_ptr x = s->available;

	s->available = (s->available)->succ;
	x->succ = STACK_PTR_NULL; /*  clean the bulk to be returned */
	return(x);
    }
    else
	return(allocate_bulk((s->stack)->max_items));
}

/* ALLOCATEs a new stack of type
 */
stack stack_make(type, size)
int type, size;
{
    stack s = malloc(sizeof(_stack_head));

    if (size<10) size=STACK_DEFAULT_SIZE; /* not too small */

    s->size = 0;
    s->type = type;
    s->max_extent = 0;
    s->stack = allocate_bulk(size);
    s->available = STACK_PTR_NULL;
 
    return(s);
}

void stack_info(f, s)
FILE *f;
stack s;
{
    fprintf(f, "stack_info about stack 0x%x\n", (unsigned int) s);

    if (STACK_NULL_P(s))
    {
	fprintf(f, " - is null\n");
	return;
    }
    /* else */
    if (stack_undefined_p(s))
    {
	fprintf(f, " - is undefined\n");
	return;
    }
    /* else */
    fprintf(f, " - type %d, size %d, max extent %d\n", 
	    s->type, s->size, s->max_extent);
    fprintf(f, " - bulks: %d in use, %d available\n",
	    number_of_bulks(s->stack), number_of_bulks(s->available));
}

/* FREEs the stack
 */

static void free_bulk(x)
_stack_ptr x;
{
    free(x->items), x->items = (char **) NULL, free(x);
}

static void free_bulks(x)
_stack_ptr x;
{
    _stack_ptr tmp;

    while(!STACK_PTR_NULL_P(x))
    {
	tmp=x, x=x->succ, tmp->succ=STACK_PTR_NULL;
	free_bulk(tmp);
    }
}

void stack_free(s)
stack s;
{
    free_bulks(s->stack), s->stack=STACK_PTR_NULL;
    free_bulks(s->available), s->available=STACK_PTR_NULL;
    free(s);
}

/* MISC
 */
int stack_size(s)
stack s;
{
    assert(!STACK_NULL_P(s) && !stack_undefined_p(s));
    return(s->size);
}

bool stack_empty_p(s)
stack s;
{
    assert(!STACK_NULL_P(s) && !stack_undefined_p(s));
    return(s->size==0);
}

/* PUSHes the item on stack s
 *
 * a new bulk is allocated if necessary. 
 * the size it the same than the initial bulk size. 
 * Other policies may be considered.
 */
void stack_push(item, s)
char *item;
stack s;
{
    _stack_ptr x = s->stack;

    assert(!STACK_PTR_NULL_P(x));

    if (x->n_item == x->max_items)
    {
	_stack_ptr saved = x;

	x = find_or_allocate(x->max_items);
	x->succ = saved;
	s->stack = x;
    }

    /*   PUSH!
     */
    s->size++; 
    if (s->size > s->max_extent) s->max_extent = s->size;
    x->items[x->n_item++] = item;
}

/* POPs one item from stack s
 *
 * the empty bulks are not freed here. 
 * stack_free does the job.
 */
char *stack_pop(s)
stack s;
{
    _stack_ptr x = s->stack;

    if (x->n_item==0)
    {
	_stack_ptr saved = x->succ;

	x->succ = s->available, s->available = x;
	s->stack = saved, x = saved;
    }

    assert(!STACK_PTR_NULL_P(x) && x->n_item>0);

    /*   POP!
     */
    s->size--; 
    return(x->items[--x->n_item]);
}

/* returns the item on top of stack s
 */
char *stack_head(s)
stack s;
{
    _stack_ptr x = s->stack;

    if (x->n_item==0) x = x->succ;

    assert(!STACK_PTR_NULL_P(x) && x->n_item>0);

    /*   HEAD
     */
    return(x->items[(x->n_item)-1]);
}

/* REPLACEs the item on top of stack s, and returns the old item
 */
char *stack_replace(item, s)
char *item;
stack s;
{
    char *old = stack_pop(s);

    stack_push(item, s);
    return(old);
}

/*  that is all
 */
