/*
 * $Id$
 *
 * A pointer oriented hash table, to be used for variable sets.
 * It is a simplified version of what is in newgen.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/* #define LINEAR_HASHTABLE_DEBUG 1 */

#if defined(LINEAR_HASHTABLE_DEBUG)
#define debug_assert(a) assert(a)
#else
#define debug_assert(a)
#endif

#define debug_assert_coherent(h) debug_assert(linear_hashtable_coherent_p(h))

typedef enum { false, true } boolean;

#define FREE_CHUNK	((void *) 0)
#define EMPTIED_CHUNK	((void *) -1)

/* hidden structure to store the hashtable.
 */
typedef struct linear_hashtable_st 
{
  int nitems;	/* number of association stored */
  int size;	/* size of internal array */
  void ** keys; /* array for keys */
  void ** vals; /* array for vals */
}
  * linear_hashtable_pt;

/* returns the location to put or get k in h.
 */
static int key_location(linear_hashtable_pt h, void * k, boolean toget)
{
  register int index = ((((int)k)&(0x7fffffff))%(h->size));

  while (!(h->keys[index]==FREE_CHUNK ||
	   h->keys[index]==k ||
	   (toget && h->keys[index]==EMPTIED_CHUNK)))
    index = (index+1) % h->size;

  debug_assert(index>=0 && index<h->size &&
	       (h->keys[index]==FREE_CHUNK ||
		h->keys[index]==EMPTIED_CHUNK ||
		h->keys[index]==k));

  return index;
}

/********************************************************************* DEBUG */

static void linear_hashtable_dump(linear_hashtable_pt h)
{
  register int i;

  fprintf(stderr, "[linear_hashtable_dump] hash=%p size=%d nitems=%d\n",
	  h, h->size, h->nitems);
  
  for (i=0; i<h->size; i++)
  {
    register void * k = h->keys[i];
    fprintf(stderr, "%d (%d): 0x%p -> 0x%p\n", 
	    i, 
	    (k!=FREE_CHUNK && k!=EMPTIED_CHUNK)? 
	        key_location(h, k, true):  -1, 
	    k, h->vals[i]);
  }

  fprintf(stderr, "[linear_hashtable_dump] done.\n");
}

/* check hashtable coherency */
boolean linear_hashtable_coherent_p(linear_hashtable_pt h)
{
  register int i, n;

  /* coherent size/nitems. */
  if (h->nitems >= h->size) 
    return false;

  /* check number of item stored. */
  for(i=0, n=0; i<h->size; i++)
  {
    register void * k = h->keys[i];
    if (k!=FREE_CHUNK && k!=EMPTIED_CHUNK)
      n++;
  }

  if (n!=h->nitems)
    return false;

  /* check keys */
  for (i=0, n=0; i<h->size; i++)
  {
    register void * k = h->keys[i];
    if (k!=FREE_CHUNK && k!=EMPTIED_CHUNK)
    {
      register int index = key_location(h, k, true);
      if (index!=i) return false;
    } 
  }

  return true;
}

/********************************************************************* BUILD */

/* constructor.
 * returns a newly allocated hashtable.
 */
linear_hashtable_pt linear_hashtable_make(void)
{
  linear_hashtable_pt h;
  register int i, size = 17;

  h = (linear_hashtable_pt) malloc(sizeof(struct linear_hashtable_st));
  assert(h);

  h->size = size;
  h->nitems = 0;
  h->keys = (void **) malloc(sizeof(void *)*size);
  h->vals = (void **) malloc(sizeof(void *)*size);

  assert(h->keys && h->vals);

  for (i=0; i<size; i++)
    h->keys[i] = FREE_CHUNK, 
      h->vals[i] = FREE_CHUNK;

  debug_assert_coherent(h);

  return h;
}

/* destructor */
void linear_hashtable_free(linear_hashtable_pt h)
{
  debug_assert_coherent(h);
  
  free(h->keys);
  free(h->vals);
  free(h);
}

static void linear_hashtable_extend(linear_hashtable_pt h)
{
  register void ** oldkeys, ** oldvals;
  register int i, oldsize, moved_nitems;

  debug_assert_coherent(h);

  moved_nitems = h->nitems;
  oldsize = h->size;
  oldkeys = h->keys;
  oldvals = h->vals;

  h->size = 2*oldsize + 1;
  h->keys = (void**) malloc(sizeof(void *)*h->size);
  h->vals = (void**) malloc(sizeof(void *)*h->size);
  assert(h->keys && h->vals);

  for (i=0; i<h->size; i++)
    h->keys[i] = FREE_CHUNK,
      h->vals[i] = FREE_CHUNK;

  for (i=0; i<oldsize; i++)
  {
    register void * k = oldkeys[i];
    if (k!=FREE_CHUNK && k!=EMPTIED_CHUNK)
    {
      register int index = key_location(h, k, false);
      h->keys[index] = k;
      h->vals[index] = oldvals[i];
      moved_nitems--;
    }
  }

  assert(moved_nitems==0);
  free(oldkeys);
  free(oldvals);

  debug_assert_coherent(h);
}

/*********************************************************************** USE */

static void linear_hashtable_internal_put
    (linear_hashtable_pt h, void * k, void * v, boolean once)
{
  register int index;

  assert(k!=FREE_CHUNK && k!=EMPTIED_CHUNK);
  debug_assert_coherent(h);

  if ((h->nitems*2) > h->size) { /* 50% */
    linear_hashtable_extend(h);
  }

  index = key_location(h, k, true); /* is it already in? */
  if (h->keys[index]!=k) /* no */
    index = key_location(h, k, false);
  else 
    assert(!once);

  debug_assert(index>=0 && index<h->size &&
	       (h->keys[index]==FREE_CHUNK || 
		h->keys[index]==EMPTIED_CHUNK ||
		h->keys[index]==k));

  if (h->keys[index]!=k)
    h->nitems++;

  h->keys[index] = k;
  h->vals[index] = v;

  debug_assert_coherent(h);
}

void linear_hashtable_put(linear_hashtable_pt h, void * k, void * v)
{
  linear_hashtable_internal_put(h, k, v, false);
}

void linear_hashtable_put_once(linear_hashtable_pt h, void * k, void * v)
{
  linear_hashtable_internal_put(h, k, v, true);
}

boolean linear_hashtable_isin(linear_hashtable_pt h, void * k)
{
  return h->keys[key_location(h, k, true)]==k;
}

boolean linear_hashtable_remove(linear_hashtable_pt h, void * k)
{
  register int index = key_location(h, k, true);

  if (h->keys[index]==k) 
  {
    h->keys[index] = EMPTIED_CHUNK;
    h->vals[index] = FREE_CHUNK;
    h->nitems--;
    return true;
  }

  return false;
}

void * linear_hashtable_get(linear_hashtable_pt h, void * k)
{
  return h->keys[key_location(h, k, true)];
}

int linear_hashtable_nitems(linear_hashtable_pt h)
{
  return h->nitems;
}
