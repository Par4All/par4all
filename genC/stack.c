/*
 * STACK MANAGEMENT
 *
 * Fabien COELHO, 05/12/94 
 *
 * Could be integrated in Newgen as a building type (as lists, mappings...).
 * there is no actual need of such a type on the functional point of view.
 * I put it there since it may be much more efficient than lists.
 * stack print out and read in functions would be needed. (direction problem).
 *
 * More thoughts needed. 
 *
 * $RCSfile: stack.c,v $ version $Revision$
 * $Date: 1995/02/02 18:18:25 $, 
 * got on %D%, %T%
 */

#include <stdio.h>
extern int fprintf();
#include "malloc.h"
#include "newgen_assert.h"
#include "newgen_types.h" /* just for GEN_PROTO */
#include "newgen_stack.h"

/*
 *   STACK STRUCTURES
 *
 */

/* the stack bulks, that is arrays containing the elements
 */
typedef struct __stack_bulk
{
    int n_item;                 /* next available item in the bulk */
    int max_items;              /* the maximun number of items of this bulk */
    char **items;               /* the items (only pointers at the moment) */
    struct __stack_bulk *succ;  /* the next bulk */
}
   _stack_bulk, *_stack_ptr;

/*  the stack head
 */
typedef struct __stack_head
{
    int size;        /* current number of elements in stack */
    int type;        /* as BASIC, LIST, EXTERNAL, CHUNK, domain? */
    int policy;      /* may be used to indicate an allocation policy */
    int bulk_size;   /* reference bulk size for allocation */
    int n_bulks;     /* number of allocated bulks */
    int max_extent;  /* maximum extension of the stack */
    _stack_ptr stack;/* bulks in use by the stack */
    _stack_ptr avail;/* allocated bulks not in use */
}
    _stack_head; /* and also *stack (in headers) */

/*  usefull defines
 */
#define STACK_PTR_NULL ((_stack_ptr) NULL)
#define STACK_PTR_NULL_P(x) ((x)==STACK_PTR_NULL)
#define STACK_DEFAULT_SIZE 30

/*
 *   STACK ITERATOR
 *
 */
typedef struct __stack_iterator
{
    _stack_ptr bulk; /* current bulk */
    int index;       /* current index in bulk */
}
    _stack_iterator; /* and also *stack_iterator (in headers) */

#define STACK_ITERATOR_END_P(i) STACK_PTR_NULL_P(i->bulk)
#define DEFINE_ITERATOR(i,blk,idx) i->bulk=(blk), i->index=(idx);
#define UPDATE_ITERATOR(i) \
  if (i->index==-1) \
    i->bulk = i->bulk->succ, i->index = (i->bulk) ? (i->bulk->n_item)-1 : -1;

stack_iterator stack_iterator_init(s)
stack s;
{
    stack_iterator i=(stack_iterator) malloc(sizeof(_stack_iterator));

    message_assert("null stack", !STACK_NULL_P(s));
    message_assert("undefined stack", !stack_undefined_p(s));

    if ((s->size)==0)
	DEFINE_ITERATOR(i,STACK_PTR_NULL,-1)
    else
    {
	DEFINE_ITERATOR(i,s->stack,(s->stack->n_item)-1);
	UPDATE_ITERATOR(i);
    }
    
    return(i);
}

char *stack_iterator_next(i)
stack_iterator i;
{
    char *result;

    assert(!STACK_ITERATOR_END_P(i));
    result=(i->bulk->items)[(i->index)--];
    UPDATE_ITERATOR(i);
    
    return(result);
}

int stack_iterator_end_p(i)
stack_iterator i;
{
    return(STACK_ITERATOR_END_P(i));
}

void stack_iterator_clean(pi)
stack_iterator *pi;
{
    free(*pi), *pi=(stack_iterator)NULL;
}

/*
 *     STACK ALLOCATION
 *
 */

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
    if (!STACK_PTR_NULL_P(s->avail))
    {
	_stack_ptr x = s->avail;

	s->avail = (s->avail)->succ;
	x->succ = STACK_PTR_NULL; /*  clean the bulk to be returned */
	return(x);
    }
    else
    {
	s->n_bulks++;
	return(allocate_bulk(s->bulk_size)); /* may depend from the policy? */
    }
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
    s->policy = (-1); /* not used */
    s->bulk_size = size;
    s->max_extent = 0;
    s->n_bulks = 0;
    s->stack = allocate_bulk(size);
    s->avail = STACK_PTR_NULL;
 
    return(s);
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
    free_bulks(s->avail), s->avail=STACK_PTR_NULL;
    free(s);
}

/* 
 *    STACK MISCELLANEOUS
 *
 */
int stack_size(s)
stack s;
{
    assert(!STACK_NULL_P(s) && !stack_undefined_p(s));
    return(s->size);
}

int stack_empty_p(s)
stack s;
{
    assert(!STACK_NULL_P(s) && !stack_undefined_p(s));
    return(s->size==0);
}

/*   APPLY f to all items of stack s;
 */
void stack_map(s, f)
stack s;
void (*f)();
{
    _stack_ptr x;
    int i;

    for(x=s->stack; x!=NULL; x=x->succ)
	for(i=(x->n_item)-1; i>=0; i--)
	    (*f)(x->items[i]);
}

static int number_of_bulks(x)
_stack_ptr x;
{
    int n=0;
    for(; !STACK_PTR_NULL_P(x); x=x->succ, n++);
    return(n);
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
	    number_of_bulks(s->stack), number_of_bulks(s->avail));
}

/*
 *     STACK USE
 *
 */

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

	x->succ = s->avail, s->avail = x;
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
    _stack_ptr x = s->stack;
    char *old;

    if (x->n_item==0) x = x->succ;
    assert(!STACK_PTR_NULL_P(x) && x->n_item>0);

    /*    REPLACE
     */
    old = x->items[(x->n_item)-1],
    x->items[(x->n_item)-1] = item;

    return(old);
}

/*  that is all
 */
