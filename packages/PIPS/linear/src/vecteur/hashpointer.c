/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

/*
 * A pointer oriented hash table, to be used for variable sets.
 * It is a simplified version of what is in newgen. 
 * It is fully standalone.
 */
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <assert.h>
#include "boolean.h"

/* expected headers: the internal structure does not need to be available!
struct linear_hashtable_st;
typedef struct linear_hashtable_st * linear_hashtable_pt;
 */
/* #define LINEAR_HASHTABLE_DEBUG 1 */

#if defined(LINEAR_HASHTABLE_DEBUG)
#define debug_assert(a) assert(a)
#else
#define debug_assert(a)
#endif

#define debug_assert_coherent(h) debug_assert(linear_hashtable_coherent_p(h))

#define FREE_CHUNK	((void *) 0)
#define EMPTIED_CHUNK	((void *) -1)

typedef struct {
  void * key;
  void * val;
} paire;

/* hidden structure to store the hashtable.
 */
typedef struct linear_hashtable_st 
{
  size_t nitems;	    /* number of association stored */
  size_t size;	    /* size of internal array */
  paire * contents; /* array storing key&value paires */
}
  * linear_hashtable_pt;

/* returns the location to put or get k in h.
 */
static uintptr_t key_location(linear_hashtable_pt h, void * k, bool toget)
{
  uintptr_t index = ((((uintptr_t) k)
		      & ~(((uintptr_t)1) << ((sizeof(uintptr_t)*CHAR_BIT) - 1)))
		     % (h->size));
  uintptr_t loop = h->size;

  while (loop-- && !(h->contents[index].key==FREE_CHUNK ||
		   h->contents[index].key==k ||
		   (toget && h->contents[index].key==EMPTIED_CHUNK)))
    index = (index+1) % h->size;

  assert(!(!loop && !toget)); /* should not loop to put! */

  /* if !loop and toget, the initial index is returned.
   * it is checked against the expected key before returning the value.
   */
  debug_assert(index>=0 && index<h->size &&
	       (h->contents[index].key==FREE_CHUNK ||
		h->contents[index].key==EMPTIED_CHUNK ||
		(h->contents[index].key==k || (!loop && toget))));

  return index;
}

/********************************************************************* DEBUG */

static void linear_hashtable_print(FILE * file, linear_hashtable_pt h)
{
  uintptr_t i;

  fprintf(file, "[linear_hashtable_dump] hash=%p size=%zd nitems=%zd\n",
	  h, h->size, h->nitems);
  
  for (i=0; i<h->size; i++)
  {
    void * k = h->contents[i].key;
    fprintf(file, "%zd (%td): 0x%p -> 0x%p\n", 
	    i, 
	    (k!=FREE_CHUNK && k!=EMPTIED_CHUNK)? key_location(h, k, true): ~(uintptr_t)0,
	    k, h->contents[i].val);
  }

  fprintf(file, "[linear_hashtable_dump] done.\n");
}

/* convenient function to be called from gdb. */
void linear_hashtable_dump(linear_hashtable_pt h)
{
  linear_hashtable_print(stderr, h);
}

/* check hashtable coherency */
bool linear_hashtable_coherent_p(linear_hashtable_pt h)
{
  uintptr_t i, n;

  /* coherent size/nitems. */
  if (h->nitems >= h->size) 
    return false;

  /* check number of item stored. */
  for(i=0, n=0; i<h->size; i++)
  {
    void * k = h->contents[i].key;
    if (k!=FREE_CHUNK && k!=EMPTIED_CHUNK)
    {
      /* check key index */
      uintptr_t index = key_location(h, k, true);
      if (index!=i) return false;
      n++;
    }
  }

  if (n!=h->nitems)
    return false;

  return true;
}

/********************************************************************* BUILD */

/* size of internal table. 
 * should be a not too big odd number.
 */
#define HASHTABLE_INITIAL_SIZE (17)

/* constructor.
 * returns a newly allocated hashtable.
 */
linear_hashtable_pt linear_hashtable_make(void)
{
  linear_hashtable_pt h;
  register int i, size = HASHTABLE_INITIAL_SIZE;

  h = (linear_hashtable_pt) malloc(sizeof(struct linear_hashtable_st));
  assert(h); /* check malloc */

  h->size = size;
  h->nitems = 0;
  h->contents = (paire*) malloc(sizeof(paire)*size);

  assert(h->contents); /* check malloc */

  for (i=0; i<size; i++)
    h->contents[i].key = FREE_CHUNK, 
      h->contents[i].val = FREE_CHUNK;

  debug_assert_coherent(h);

  return h;
}

/* destructor */
void linear_hashtable_free(linear_hashtable_pt h)
{
  debug_assert_coherent(h);
  
  free(h->contents);
  free(h);
}

static void linear_hashtable_extend(linear_hashtable_pt h)
{
  paire * oldcontents;
  uintptr_t i, oldsize, moved_nitems;

  debug_assert_coherent(h);

  moved_nitems = h->nitems;
  oldcontents = h->contents;
  oldsize = h->size;

  h->size = 2*oldsize + 1;
  h->contents = (paire*) malloc(sizeof(paire)*h->size);
  assert(h->contents); /* check malloc */

  for (i=0; i<h->size; i++)
    h->contents[i].key = FREE_CHUNK,
      h->contents[i].val = FREE_CHUNK;

  for (i=0; i<oldsize; i++)
  {
    register void * k = oldcontents[i].key;
    if (k!=FREE_CHUNK && k!=EMPTIED_CHUNK)
    {
      register int index = key_location(h, k, false);
      h->contents[index].key = k;
      h->contents[index].val = oldcontents[i].val;
      moved_nitems--;
    }
  }

  assert(moved_nitems==0); /* the expected number of items was moved. */
  free(oldcontents);

  debug_assert_coherent(h);
}

/*********************************************************************** USE */

static void linear_hashtable_internal_put
    (linear_hashtable_pt h, void * k, void * v, bool once)
{
  register int index;

  assert(k!=FREE_CHUNK && k!=EMPTIED_CHUNK); /* no special values! */
  debug_assert_coherent(h);

  if ((h->nitems*2) > h->size) { /* 50% limit to extend */
    linear_hashtable_extend(h);
  }

  index = key_location(h, k, true); /* is it already in? */
  if (h->contents[index].key!=k) /* no */
    index = key_location(h, k, false); /* where should it be put? */
  else 
    assert(!once); /* if it is in, the once option must not be set */

  if (h->contents[index].key!=k) /* update number of stored items */
    h->nitems++;

  h->contents[index].key = k;
  h->contents[index].val = v;

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

bool linear_hashtable_isin(linear_hashtable_pt h, void * k)
{
  return h->contents[key_location(h, k, true)].key==k;
}

bool linear_hashtable_remove(linear_hashtable_pt h, void * k)
{
  register int index = key_location(h, k, true);

  if (h->contents[index].key==k) 
  {
    h->contents[index].key = EMPTIED_CHUNK;
    h->contents[index].val = FREE_CHUNK;
    h->nitems--;
    return true;
  }

  return false;
}

void * linear_hashtable_get(linear_hashtable_pt h, void * k)
{
  register int index = key_location(h, k, true);
  return h->contents[index].key==k ? h->contents[index].val: FREE_CHUNK;
}

int linear_hashtable_nitems(linear_hashtable_pt h)
{
  return h->nitems;
}
