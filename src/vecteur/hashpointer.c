/*
 * $Id$
 *
 * A pointer oriented hash table, to be used for variable sets.
 * It is a simplified version of what is in newgen.
 */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef enum { false, true } boolean;

#define FREE_CHUNK	((void *) 0)
#define EMPTIED_CHUNK	((void *) -1)

typedef struct linear_hashtable_st 
{
  int nitems;	/* number of association stored */
  int size;	/* size of internal array */
  void ** keys; /* array for keys */
  void ** vals; /* array for vals */
}
  * linear_hashtable_pt;

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

  return h;
}

void linear_hashtable_free(linear_hashtable_pt h)
{
  free(h->keys);
  free(h->vals);
  free(h);
}

void linear_hashtable_put(linear_hashtable_pt, void *, void *);

static void linear_hashtable_extend(linear_hashtable_pt h)
{
  void ** oldkeys, ** oldvals;
  register int i, oldsize, oldnitems;

  oldnitems = h->nitems;
  oldsize = h->size;
  oldkeys = h->keys;
  oldvals = h->vals;

  h->nitems = 0;
  h->size = 2*oldsize + 1;
  h->keys = (void**) malloc(sizeof(void *)*h->size);
  h->vals = (void**) malloc(sizeof(void *)*h->size);
  assert(h->keys && h->vals);

  for (i=0; i<h->size; i++)
    h->keys[i] = FREE_CHUNK,
      h->vals[i] = FREE_CHUNK;

  for (i=0; i<oldsize; i++)
  {
    if (oldkeys[i]!=FREE_CHUNK && oldkeys[i]!=EMPTIED_CHUNK)
    {
      linear_hashtable_put(h, oldkeys[i], oldvals[i]);
      oldnitems--;
    }
  }

  assert(oldnitems==0);
  free(oldkeys);
  free(oldvals);
}

/* returns the location to put or get k in h.
 */
static int key_location(linear_hashtable_pt h, void * k, boolean toget)
{
  register int hashed = ((((int)k)&(0x7fffffff))%(h->size));

  while (h->keys[hashed]!=FREE_CHUNK &&
	 h->keys[hashed]!=k &&
	 (toget && h->keys[hashed]==EMPTIED_CHUNK))
    hashed++, hashed %= h->size;

  return hashed;
}

void linear_hashtable_put(linear_hashtable_pt h, void * k, void * v)
{
  register int hashed;

  if ((h->nitems<<2) > h->size) {
    linear_hashtable_extend(h);
  }

  hashed = key_location(h, k, false);
  
  h->keys[hashed] = k;
  h->vals[hashed] = v;
  h->nitems++;
}

boolean linear_hashtable_isin(linear_hashtable_pt h, void * k)
{
  return h->keys[key_location(h, k, true)]==k;
}

boolean linear_hashtable_remove(linear_hashtable_pt h, void * k)
{
  register int hashed = key_location(h, k, true);

  if (h->keys[hashed]==k) 
  {
    h->keys[hashed] = EMPTIED_CHUNK;
    h->vals[hashed] = FREE_CHUNK;
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
