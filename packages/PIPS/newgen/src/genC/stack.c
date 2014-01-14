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
/*
 * STACK MANAGEMENT
 *
 * Fabien COELHO, 05/12/1994
 *
 * Could be integrated in Newgen as a building type (as lists, mappings).
 * there is no actual need of such a type on the functional point of view.
 * I put it there since it may be much more efficient than lists.
 * Stack print out and read in functions would be needed.
 * (direction problem).
 *
 * More thoughts needed.
 */
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include "genC.h"
#include "newgen_include.h"

/*   STACK STRUCTURES
 */

/* the stack buckets, i.e. arrays containing the elements
 */
typedef struct __stack_bucket
{
  size_t n_item;                 /* next available item in the bucket */
  size_t max_items;              /* the max number of items of this bucket */
  void ** items;                 /* the items (only pointers at the moment) */
  struct __stack_bucket *succ;/* the next bucket */
  /* we could keep the previous bucket? */
}
  _stack_bucket, *_stack_ptr;

/*  the stack head
 */
typedef struct __stack_head
{
  size_t size;        /* current number of elements in stack */
  size_t max_extent;  /* maximum extension of the stack */
  _stack_ptr stack;/* buckets in use by the stack */
  _stack_ptr avail;/* allocated buckets not in use anymore */
  size_t bucket_size; /* reference bucket size for allocation */
  size_t n_buckets;   /* number of allocated buckets */
  int type;        /* as BASIC, LIST, EXTERNAL, CHUNK, domain? */
  int policy;      /* may be used to indicate an allocation policy */
}
    _stack_head; /* and also *stack (in headers) */

/*  usefull defines
 */
#define STACK_PTR_NULL ((_stack_ptr) NULL)
#define STACK_PTR_NULL_P(x) ((x)==STACK_PTR_NULL)
#define STACK_DEFAULT_SIZE 50

/*   STACK ITERATOR
 */
typedef struct __stack_iterator
{
  _stack_ptr bucket; /* current bucket */
  bool downward;      /* true if downward iterations */
  size_t index;      /* current index in bucket */
  _stack_ptr list;   /* all buckets */
}
    _stack_iterator; /* and also *stack_iterator (in headers) */

static void update_iterator_upward(stack_iterator i)
{
  _stack_ptr x=i->list;

  while(!STACK_PTR_NULL_P(x) && x->succ!=i->bucket)
    x=x->succ;

  i->bucket=x, i->index=0;

  if (x && i->bucket->n_item==0)
    i->bucket = STACK_PTR_NULL;
}

#define STACK_ITERATOR_END_P(i) STACK_PTR_NULL_P(i->bucket)

#define DEFINE_ITERATOR(i,blk,idx,dwn,lst)				\
  {									\
    i->bucket=(blk);							\
    i->index=(idx);							\
    i->list=lst;							\
    i->downward=dwn;							\
  }

#define UPDATE_ITERATOR_DOWNWARD(i)					\
  if (i->index == (size_t) -1)						\
    {									\
      i->bucket = i->bucket->succ;					\
      i->index = (i->bucket) ? (i->bucket->n_item)-1 : (size_t) -1;	\
    }

#define UPDATE_ITERATOR_UPWARD(i)				\
  if (i->index==i->bucket->n_item)				\
    update_iterator_upward(i);

#define NEXT_ITERATION(i)			\
  if (i->downward)				\
  {						\
    i->index--; UPDATE_ITERATOR_DOWNWARD(i);	\
  }						\
  else						\
  {						\
    i->index++; UPDATE_ITERATOR_UPWARD(i);	\
  }

static void stack_iterator_internal_init
  (const stack s, int down, stack_iterator i)
{
  STACK_CHECK(s);

  if ((s->size)==0)
    DEFINE_ITERATOR(i, STACK_PTR_NULL, -1, down, STACK_PTR_NULL)
  else
  {
    if (down)
    {
      DEFINE_ITERATOR(i, s->stack, (s->stack->n_item)-1, down, s->stack);
      UPDATE_ITERATOR_DOWNWARD(i);
    }
    else
    {
      DEFINE_ITERATOR(i, STACK_PTR_NULL, 0, down, s->stack);
      update_iterator_upward(i); /* NOT the define! */
    }
  }
}

stack_iterator stack_iterator_init(const stack s, bool down)
{
  stack_iterator i = (stack_iterator) malloc(sizeof(_stack_iterator));
  stack_iterator_internal_init(s, down, i);
  return i;
}

bool stack_iterator_next_and_go(stack_iterator i, void ** pitem)
{
  if (STACK_ITERATOR_END_P(i))
  {
    *pitem = (void*) NULL;
    return false;
  }
  else
  {
    *pitem = (i->bucket->items)[i->index];
    NEXT_ITERATION(i);
    return true;
  }
}

bool stack_iterator_end_p(stack_iterator i)
{
  return STACK_ITERATOR_END_P(i);
}

void stack_iterator_end(stack_iterator * pi)
{
  (*pi)->bucket = NEWGEN_FREED;
  (*pi)->downward = 0;
  (*pi)->index = 0;
  (*pi)->list = NEWGEN_FREED;
  free(*pi);
  *pi=(stack_iterator) NULL;
}

/*
 *     STACK ALLOCATION
 *
 */

/* allocates a bucket of size size
 */
static _stack_ptr allocate_bucket(int size)
{
  _stack_ptr x = (_stack_ptr) malloc(sizeof(_stack_bucket));
  message_assert("pointer was allocated", x);

  x->n_item = 0;
  x->max_items = size;
  x->items = (void **) malloc(sizeof(void *)*size);
  message_assert("pointer was allocated", x->items);
  x->succ = STACK_PTR_NULL;

  return x;
}

/* search for a new bucket, first in the available list,
 * if none are available, a new bucket is allocated
 */
static _stack_ptr find_or_allocate(stack s)
{
  if (!STACK_PTR_NULL_P(s->avail))
  {
    _stack_ptr x = s->avail;
    s->avail = (s->avail)->succ;
    x->succ = STACK_PTR_NULL; /*  clean the bucket to be returned */
    return x;
  }
  else
  {
    s->n_buckets++;
    /* may depend from the policy? */
    return allocate_bucket(s->bucket_size);
  }
}

/* ALLOCATEs a new stack of @p type

   @param type record newgen domain of stack contents. should be used
   to check the type of appended elements.

   @param bucket_size is the number of elements in the elemental stack
   container. If you now you will have big stacks, try big numbers here to
   save memory.

   @param policy not used, 0 is fine.
 */
stack stack_make(int type, int bucket_size, int policy)
{
  stack s = (stack) malloc(sizeof(_stack_head));
  message_assert("pointer was allocated", s);

  if (bucket_size<10) bucket_size=STACK_DEFAULT_SIZE; /* not too small */

  s->size = 0;
  s->type = type;
  s->policy = policy; /* not used */
  s->bucket_size = bucket_size;
  s->max_extent = 0;
  s->n_buckets = 0;
  s->stack = allocate_bucket(bucket_size);
  s->avail = STACK_PTR_NULL;

  return s;
}

/* duplicate a stack with its contents.
 */
stack stack_copy(const stack s)
{
  stack n = stack_make(s->type, s->bucket_size, s->policy);
  STACK_MAP_X(i, void*, stack_push(i, n), s, 0);
  return n;
}

/* FREEs the stack
 */
static void free_bucket(_stack_ptr x)
{
  gen_free_area(x->items, x->max_items*sizeof(void*));
  gen_free_area((void**) x, sizeof(_stack_bucket));
}

static void free_buckets(_stack_ptr x)
{
  _stack_ptr tmp;
  while(!STACK_PTR_NULL_P(x))
  {
    tmp=x, x=x->succ, tmp->succ=STACK_PTR_NULL;
    free_bucket(tmp);
  }
}

void stack_free(stack * ps)
{
  free_buckets((*ps)->stack), (*ps)->stack=STACK_PTR_NULL;
  free_buckets((*ps)->avail), (*ps)->avail=STACK_PTR_NULL;
  gen_free_area((void**) *ps, sizeof(stack_head));
  *ps = STACK_NULL;
}

/*    STACK MISCELLANEOUS
 */
#define STACK_OBSERVER(name, what)				\
  int stack_##name(const stack s) { STACK_CHECK(s); return(what); }

/* Here we define stack_size(), stack_bsize(), stack_policy() and
   stack_max_extent(): */
STACK_OBSERVER(size, s->size)
STACK_OBSERVER(bsize, s->bucket_size)
STACK_OBSERVER(policy, s->policy)
STACK_OBSERVER(max_extent, s->max_extent)
STACK_OBSERVER(consistent_p, 1) /* well, it is not implemented */

bool stack_empty_p(const stack s)
{
  STACK_CHECK(s);
  return s->size==0;
}

#undef STACK_OBSERVER

/*   APPLY f to all items of stack s;
 */
void stack_map(const stack s, gen_iter_func_t f)
{
  _stack_ptr x;
  int i;

  STACK_CHECK(s);

  for(x=s->stack; x!=NULL; x=x->succ)
    for(i=(x->n_item)-1; i>=0; i--)
      (*f)(x->items[i]);
}

static int number_of_buckets(_stack_ptr x)
{
  int n=0;
  for(; !STACK_PTR_NULL_P(x); x=x->succ, n++);
  return n;
}

void stack_info(FILE * f, const stack s)
{
  fprintf(f, "stack_info about stack %p\n", s);

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
  fprintf(f, " - type %d, size %zd, max extent %zd\n",
	  s->type, s->size, s->max_extent);
  fprintf(f, " - buckets: %d in use, %d available\n",
	  number_of_buckets(s->stack), number_of_buckets(s->avail));
}

/*     STACK USE
 */

/* PUSHes the item on stack s
 *
 * a new bucket is allocated if necessary.
 * the size it the same than the initial bucket size.
 * Other policies may be considered.
 */
void stack_push(void * item, stack s)
{
  _stack_ptr x = s->stack;

  assert(!STACK_PTR_NULL_P(x));

  if (x->n_item == x->max_items)
  {
    _stack_ptr saved = x;
    x = find_or_allocate(s);
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
 * the empty buckets are not freed here.
 * stack_free does the job.
 */
void *stack_pop(stack s)
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
  return x->items[--x->n_item];
}

/* returns the item on top of stack s
 */
void *stack_head(const stack s)
{
  _stack_ptr x = s->stack;
  if (x->n_item==0) x = x->succ;
  assert(!STACK_PTR_NULL_P(x) && x->n_item>0);

  /*   HEAD
   */
  return x->items[(x->n_item)-1];
}

/* returns the nth item starting from the head and counting from 1,
   when possible, or NULL, elsewhere.

   stack_nth(stack,1)==stack_head(stack) if stack_size(stack)>=1.
 */
void *stack_nth(const stack s, int nskip)
{
  void * value = NULL;
  message_assert("positive nskip", nskip>=0);
  // message_assert("deep enough stack", stack_size(s)>=nskip);
  _stack_iterator si;
  stack_iterator_internal_init(s, true, &si);
  while (nskip && stack_iterator_next_and_go(&si, &value))
    nskip--;
  return value;
}

/* REPLACEs the item on top of stack s, and returns the old item
 */
void *stack_replace(void * item, stack s)
{
  _stack_ptr x = s->stack;
  void *old;

  if (x->n_item==0) x = x->succ;
  assert(!STACK_PTR_NULL_P(x) && x->n_item>0);

  /*    REPLACE
   */
  old = x->items[(x->n_item)-1];
  x->items[(x->n_item)-1] = item;

  return old;
}


/*  that is all
 */
