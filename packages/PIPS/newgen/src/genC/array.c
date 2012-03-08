/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include "genC.h"

#define GEN_ARRAY_SIZE_INCREMENT (50)

struct _gen_array_chunk_t {
    size_t size;
    int nitems;
    void ** array;
};

gen_array_t
gen_array_make(size_t size)
{
    gen_array_t a;
    size_t i;
    if (size<=0) size= GEN_ARRAY_SIZE_INCREMENT; /* default size */
    a = (gen_array_t) malloc(sizeof(struct _gen_array_chunk_t));
    message_assert("array ok", a);
    a->size = size;
    a->nitems = 0; /* number of items stored */
    a->array = (void**) malloc(size*sizeof(void*));
    message_assert("malloc ok", a->array);
    for (i=0; i<size; i++) a->array[i] = (void*) NULL;
    return a;
}

static void
gen_array_resize(gen_array_t a, int min)
{
  int N = GEN_ARRAY_SIZE_INCREMENT;
  size_t nsize = ((min%N)==0)?min:((int)(min/N) +1)*N;
  size_t i;
  /* int nsize = a->size+GEN_ARRAY_SIZE_INCREMENT, i;*/
  a->array = (void**) realloc(a->array, nsize*sizeof(void*));
  message_assert("realloc ok", a->array);
  for (i=a->size; i<nsize; i++) 
    a->array[i] = (void*) NULL;
  a->size = nsize;
}

void
gen_array_free(gen_array_t a)
{
  gen_free_area(a->array, a->size*sizeof(void*));
  gen_free_area((void**)a, sizeof(struct _gen_array_chunk_t));
}

void
gen_array_full_free(gen_array_t a)
{
  size_t i;
  for (i=0; i<a->size; i++)
    if (a->array[i])
      free(a->array[i]); /* what is it? */
  gen_array_free(a);
}

void
gen_array_addto(gen_array_t a, size_t i, void * what)
{
    if (i>=a->size) gen_array_resize(a,i+1);
    message_assert("valid index", /* 0<=i &&  */ i < a->size);
    if (a->array[i]!=(void *)NULL) a->nitems--;
    a->array[i] = what;
    if (a->array[i]!=(void *)NULL) a->nitems++;
}

void 
gen_array_remove(gen_array_t a, size_t i) 
{
  message_assert("valid index", /* 0<=i && */ i < a->size);
  if (a->array[i]!=(void *)NULL) a->nitems--;
  a->array[i] = (void *)NULL;
}

void 
gen_array_append(gen_array_t a, void * what)
{
    gen_array_addto(a, a->nitems, what);
}

void
gen_array_dupaddto(gen_array_t a, size_t i, void * what)
{
    gen_array_addto(a, i, strdup(what));
}

void
gen_array_dupappend(gen_array_t a, void * what)
{
    gen_array_append(a, strdup(what));
}

/* Observers...
 */
void **
gen_array_pointer(const gen_array_t a)
{
    return a->array;
}

size_t
gen_array_nitems(const gen_array_t a)
{
    return a->nitems;
}

size_t
gen_array_size(const gen_array_t a)
{
    return a->size;
}

void *
gen_array_item(const gen_array_t a, size_t i)
{
  message_assert("valid index", /* 0<=i && */ i < a->size);
  return a->array[i];
}

/* Sort: assumes that the items are the first ones.
 */
static int
gen_array_cmp(const void * a1, const void * a2)
{
  return strcmp(* (char **) a1, * (char **) a2);
}

void
gen_array_sort_with_cmp(gen_array_t a, int (*cmp)(const void *, const void *))
{
  qsort(a->array, a->nitems, sizeof(void *), cmp);
}

void
gen_array_sort(gen_array_t a)
{
  gen_array_sort_with_cmp(a, gen_array_cmp);
}

gen_array_t
gen_array_from_list(list /* of string */ ls)
{
    gen_array_t a = gen_array_make(0);
    MAP(STRING, s, gen_array_dupappend(a, s), ls);
    return a;
}

list // of string
list_from_gen_array(gen_array_t a)
{
  list ls = NIL;
  GEN_ARRAY_FOREACH(string, s, a)
    ls = CONS(string, strdup(s), ls);
  return ls;
}

/* Join a string array with a string separator.

   @param array is the string array
   @param separator is the string separator

   @return a string in a concatenate buffer, so it needs to be strdup()ed
   quickly if it is expected to last some time in the caller...

   It is similar to the join() string method in Python. Using the function
   with ["foo", "bar", "daurade"] and "," should return the string
   "foo,bar,daurade".
*/
string string_array_join(gen_array_t array, string separator)
{
  string join = "";
  bool first_iteration = true;

  GEN_ARRAY_FOREACH(string, s, array)
  {
    if (! first_iteration)
      join = concatenate(join, separator, NULL);
    else
      first_iteration = false;
    join = concatenate(join, s, NULL);
  }

  return join;
}
