/*
 * STACK MANAGEMENT
 *
 * Fabien COELHO, 05/12/94 
 *
 * Could be integrated in Newgen as a building type (as lists, mappings...).
 * there is no actual need of such a type on the functional point of view.
 * I put it there since it may be much more efficient than lists.
 * Stack print out and read in functions would be needed. (direction problem).
 *
 * More thoughts needed. 
 *
 * $RCSfile: stack.c,v $ version $Revision$
 * $Date: 1995/02/03 10:11:36 $, 
 * got on %D%, %T%
 */

#include <stdio.h>
extern int fprintf();
#include "malloc.h"
#include "newgen_assert.h"
#include "newgen_stack.h"

/*
 *   STACK STRUCTURES
 *
 */

/* the stack bulks, i.e. arrays containing the elements
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
    int max_extent;  /* maximum extension of the stack */
    _stack_ptr stack;/* bulks in use by the stack */
    _stack_ptr avail;/* allocated bulks not in use anymore */
    int bulk_size;   /* reference bulk size for allocation */
    int n_bulks;     /* number of allocated bulks */
    int type;        /* as BASIC, LIST, EXTERNAL, CHUNK, domain? */
    int policy;      /* may be used to indicate an allocation policy */
}
    _stack_head; /* and also *stack (in headers) */

/*  usefull defines
 */
#define STACK_PTR_NULL ((_stack_ptr) NULL)
#define STACK_PTR_NULL_P(x) ((x)==STACK_PTR_NULL)
#define STACK_DEFAULT_SIZE 50

/*
 *   STACK ITERATOR
 *
 */
typedef struct __stack_iterator
{
    _stack_ptr bulk; /* current bulk */
    int downward;    /* true if downward iterations */
    int index;       /* current index in bulk */
    _stack_ptr list; /* all bulks */
}
    _stack_iterator; /* and also *stack_iterator (in headers) */

static void update_iterator_upward(i)
stack_iterator i;
{
    _stack_ptr x=i->list;

    while(!STACK_PTR_NULL_P(x) && x->succ!=i->bulk) 
	x=x->succ;

    i->bulk=x;
    i->index=0;
}

#define STACK_ITERATOR_END_P(i) STACK_PTR_NULL_P(i->bulk)
#define DEFINE_ITERATOR(i,blk,idx,dwn,lst) \
    i->bulk=(blk), i->index=(idx), i->list=lst, i->downward=dwn;
#define UPDATE_ITERATOR(i) \
  if (i->downward) \
  {\
    if (i->index==-1) \
      i->bulk = i->bulk->succ,\
      i->index = (i->bulk) ? (i->bulk->n_item)-1 : -1;\
  }\
  else\
  {\
    if (i->index==i->bulk->n_item)\
      update_iterator_upward(i);\
  }

stack_iterator stack_iterator_init(s, down)
stack s;
int down;
{
    stack_iterator i=(stack_iterator) malloc(sizeof(_stack_iterator));

    STACK_CHECK(s);

    if ((s->size)==0)
	DEFINE_ITERATOR(i, STACK_PTR_NULL, -1, down, STACK_PTR_NULL)
    else
    {
	if (down)
	{
	    DEFINE_ITERATOR(i, s->stack, (s->stack->n_item)-1, down, s->stack);
	    UPDATE_ITERATOR(i);
	}
	else
	{
	    DEFINE_ITERATOR(i, STACK_PTR_NULL, 0, down, s->stack);
	    update_iterator_upward(i);
	}
    }
    
    return(i);
}

int stack_iterator_next_and_go(i, pitem)
stack_iterator i;
char **pitem;
{
    if (STACK_ITERATOR_END_P(i))
    {
	*pitem = (char*) NULL;
	return(0);
    }
    else
    {
	*pitem = (i->bulk->items)[i->index];
	i->index += i->downward ? -1 : 1;
	UPDATE_ITERATOR(i);
	return(1);
    }
}

int stack_iterator_end_p(i)
stack_iterator i;
{
    return(STACK_ITERATOR_END_P(i));
}

void stack_iterator_end(pi)
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
stack stack_make(type, bulk_size, policy)
int type, bulk_size, policy;
{
    stack s = malloc(sizeof(_stack_head));

    if (bulk_size<10) bulk_size=STACK_DEFAULT_SIZE; /* not too small */

    s->size = 0;
    s->type = type;
    s->policy = policy; /* not used */
    s->bulk_size = bulk_size;
    s->max_extent = 0;
    s->n_bulks = 0;
    s->stack = allocate_bulk(bulk_size);
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

void stack_free(ps)
stack *ps;
{
    free_bulks((*ps)->stack), (*ps)->stack=STACK_PTR_NULL;
    free_bulks((*ps)->avail), (*ps)->avail=STACK_PTR_NULL;
    free(*ps); *ps = STACK_NULL;
}

/* 
 *    STACK MISCELLANEOUS
 *
 */
#define STACK_OBSERVER(name, what)\
int stack_##name(s) stack s; { STACK_CHECK(s); return(what);}

STACK_OBSERVER(size, s->size);
STACK_OBSERVER(bulk_size, s->bulk_size);
STACK_OBSERVER(policy, s->policy);
STACK_OBSERVER(max_extent, s->max_extent);
STACK_OBSERVER(empty_p, s->size==0);
STACK_OBSERVER(consistent_p, 1); /* well, it is not implemented */

#undef STACK_OBSERVER

/*   APPLY f to all items of stack s;
 */
void stack_map(s, f)
stack s;
void (*f)();
{
    _stack_ptr x;
    int i;

    STACK_CHECK(s);

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
