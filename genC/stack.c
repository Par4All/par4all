/*
 * STACK MANAGEMENT
 *
 * Fabien COELHO, 05/12/94 
 *
 * Could be integrated in Newgen as a building type (as lists, mappings...).
 * there is no actual need of such a type on the functional point of view.
 * I put it there since it may be much more efficient than lists.
 *
 * More thoughts needed. 
 */

#include <stdio.h>
extern int fprintf();
#include "malloc.h"

#include "genC.h"

/*  usefull defines
 */

#define STACK_NULL ((gen_stack_ptr) NULL)
#define STACK_NULL_P(s) ((s)==STACK_NULL)

#define GEN_STACK_DEFAULT_SIZE 30

/* allocates a bulk of size size
 */
static gen_stack_ptr allocate_bulk(size)
int size;
{
    gen_stack_ptr 
	x = (gen_stack_ptr) malloc(sizeof(gen_stack_bulk));
    
    x->n_item = 0;
    x->max_items = size;
    x->items = (chunk **) malloc(sizeof(chunk *)*size);
    x->succ = STACK_NULL;

    return(x);
}

/* search for a new bulk, first in the available list,
 * if non are available, a new bulk is allocated
 */
static gen_stack_ptr find_or_allocate(s)
gen_stack s;
{
    if (!STACK_NULL_P(s->available))
    {
	gen_stack_ptr 
	    x = s->available;

	s->available = (s->available)->succ;

	/*  clean the bulk to be returned
	 */
	x->succ = STACK_NULL;
	return(x);
    }
    else
	return(allocate_bulk((s->stack)->max_items));
}

/* ALLOCATEs a new stack of type
 */
gen_stack gen_stack_make(type, size)
int type, size;
{
    gen_stack 
	s = malloc(sizeof(gen_stack_head));

    if (size<1) size=GEN_STACK_DEFAULT_SIZE;

    s->size = 0;
    s->type = type;
    s->max_extent = 0;
    s->stack = allocate_bulk(size);
    s->available = STACK_NULL;
 
    return(s);
}

/* FREEs the stack
 */

static void free_bulk(x)
gen_stack_ptr x;
{
    free(x->items), x->items = (chunk **) NULL, free(x);
}

static void free_bulks(x)
gen_stack_ptr x;
{
    if (!STACK_NULL_P(x))
	free_bulks(x->succ), x->succ=STACK_NULL, free_bulk(x);
}

void gen_stack_free(s)
gen_stack s;
{
    free_bulks(s->stack), s->stack = STACK_NULL;
    free_bulks(s->available), s->available = STACK_NULL;
    free(s);
}

/* MISC
 */
int gen_stack_size(s)
gen_stack s;
{
    return(s->size);
}

bool gen_stack_empy_p(s)
gen_stack s;
{
    return(s->size==0);
}

/* PUSHes the item on stack s
 *
 * a new bulk is allocated if necessary. 
 * the size it the same than the initial bulk size. 
 * Other policies may be considered.
 */
void gen_push(item, s)
chunk *item;
gen_stack s;
{
    gen_stack_ptr x = s->stack;

    assert(!STACK_NULL_P(x));

    if (x->n_item == x->max_items)
    {
	gen_stack_ptr saved = x;

	x = find_or_allocate(x->max_items);
	x->succ = saved;
	s->stack = x;
    }

    /*   PUSH!
     */
    s->size++; 
    if (s->size > s->max_extent) 
	s->max_extent = s->size;
    x->items[x->n_item++] = item;
}

/* POPs one item from stack s
 *
 * the empty bulks are not freed here. 
 * gen_stack_free does the job.
 */
chunk *gen_pop(s)
gen_stack s;
{
    gen_stack_ptr x = s->stack;

    if (x->n_item==0)
    {
	gen_stack_ptr saved = x->succ;

	x->succ = s->available, s->available = x;
	s->stack = saved, x = saved;
    }

    assert(!STACK_NULL_P(x) && x->n_item>0);

    /*   POP!
     */
    s->size--; 
    return(x->items[--x->n_item]);
}

/* returns the item on top of stack s
 */
chunk *gen_head(s)
gen_stack s;
{
    gen_stack_ptr x = s->stack;

    if (x->n_item==0) x = x->succ;

    assert(!STACK_NULL_P(x) && x->n_item>0);

    /*   HEAD
     */
    return(x->items[(x->n_item)-1]);
}

/* REPLACEs the item on top of stack s
 */
chunk *gen_replace(item, s)
chunk *item;
gen_stack s;
{
    chunk *old = gen_pop(s);

    gen_push(item, s);
    return(old);
}

/*  that is all
 */
