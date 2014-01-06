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
  management of tabulated domains.
*/
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <stdint.h>
#include <setjmp.h>

#include "genC.h"
#include "newgen_include.h"

/*********************************************************** TABULATED STUFF */

/* table of tabulated elements.

   I guess it could be stored directly in the gen_binding,
   and the index could be dropped.
 */

#define TABULATED_ELEMENTS_SIZE (500)
#define TABULATED_ELEMENTS_INCR (10000)

struct _gtp
{
  int domain;           /* domain number */
  int size;		/* current allocated size */
  int used;		/* how many are used */
  gen_chunk * table;	/* actual table */
  hash_table names;     /* names to index for fast search */
};

static gen_tabulated_p get_tabulated_from_domain(int domain)
{
  check_domain(domain);
  message_assert("domain is tabulated", IS_TABULATED(Domains+domain));
  return Domains[domain].tabulated;
}

gen_tabulated_p gen_init_tabulated(int domain)
{
  int i;
  gen_chunk * t;
  gen_tabulated_p gtp;

  gtp = (gen_tabulated_p) alloc(sizeof(struct _gtp));

  gtp->size = TABULATED_ELEMENTS_SIZE;
  gtp->used = 0;
  gtp->domain = domain;
  t = (gen_chunk*) alloc(sizeof(gen_chunk)*TABULATED_ELEMENTS_SIZE);

  for (i=0; i<TABULATED_ELEMENTS_SIZE; i++)
    t[i].p = gen_chunk_undefined;

  gtp->table = t;
  gtp->names = hash_table_make(hash_string, 0);

  return gtp;
}

static void extends_tabulated(gen_tabulated_p gtp)
{
  register int nsize, i;
  gen_chunk * t;

  nsize = gtp->size + TABULATED_ELEMENTS_INCR;

  t = (gen_chunk*) realloc(gtp->table, sizeof(gen_chunk)*nsize);

  message_assert("realloc ok", t);

  for (i=gtp->size; i<nsize; i++)
    t[i].p = gen_chunk_undefined;

  gtp->size = nsize;
  gtp->table = t;
}

/* WARNING: it is not reentrant... */
gen_chunk * gen_tabulated_fake_object_hack(int domain)
{
  static gen_chunk c[2];
  static struct intlist il;

  gen_tabulated_p gtp = get_tabulated_from_domain(domain);

  c[0].i = Tabulated_bp-Domains;
  c[1].p = gtp->table;

  il.val = gtp->size; /* max_tabulated_elements() */
  il.cdr = (struct intlist *) NULL;

  Tabulated_bp->domain->ar.element = &Domains[domain];
  Tabulated_bp->domain->ar.dimensions = &il;

  return c;
}

/* apply fp to domain... */
void gen_mapc_tabulated(void (*fp)(gen_chunk*), int domain)
{
  gen_tabulated_p gtp = get_tabulated_from_domain(domain);
  register int i, size = gtp->size;

  for (i=0; i<size; i++)
  {
    gen_chunk e = gtp->table[i];
    if (e.p && e.p != gen_chunk_undefined)
    {
      message_assert("tabulated ok", e.p->i==domain);
      fp(e.p);
    }
  }
}

/* returns the list of entities with this caracteristics. */
list gen_filter_tabulated(bool (*filter)(gen_chunk*), int domain)
{
  gen_tabulated_p gtp = get_tabulated_from_domain(domain);
  register int i, size = gtp->size;
  list l;

  for (l=NIL, i=0; i<size; i++) {
    gen_chunk * o = gtp->table[i].p;
    if (o && o != gen_chunk_undefined) {
      if (filter(o)) l = CONS(CHUNK, o, l);
    }
  }

  return l;
}

/* add tabulated in table. returns its index.
 */
static int gen_put_tabulated(int domain, gen_chunk * gc)
{
  gen_tabulated_p gtp = get_tabulated_from_domain(domain);
  register int i, size = gtp->size;

  if (gtp->used + 10 > gtp->size) {
    extends_tabulated(gtp);
  }

  message_assert("large enough", gtp->used+10 <= gtp->size);

  for (i = gtp->used? gtp->used: 1; i != gtp->used-1 ; i%=(size-1), i++)
  {
    if (gtp->table[i].p == gen_chunk_undefined)
    {
      gtp->table[i].p = gc;
      gtp->used++;
      return i;
    }
  }

  /* should not get there */
  fatal("cannot put tabulated dom=%d... no space available!", domain);
  return 0;
}

static void gen_put_tabulated_name(int domain, char * id, _int number)
{
  gen_tabulated_p gtp = get_tabulated_from_domain(domain);
  message_assert("positive tabulated number", number>0);
  hash_put(gtp->names, id, (void *) number);
}


/* deletes obj from the tabulated names...
 */
static void gen_delete_tabulated_name(gen_chunk * obj)
{
  char * key = (obj+2)->s;
  void * okey, * val;
  int domain = obj->i;
  gen_tabulated_p gtp = get_tabulated_from_domain(domain);

  val = hash_delget(gtp->names, key, &okey);
  if (val == HASH_UNDEFINED_VALUE)
    fatal("gen_delete_tabulated_name: clearing unexisting (%s)\n", key);

  // free(okey);
}

static _int gen_get_tabulated_name_basic(int domain, const char * id)
{
  gen_tabulated_p gtp = get_tabulated_from_domain(domain);
  return (_int) hash_get(gtp->names, id);
}

void * gen_find_tabulated(const char * key, int domain)
{
  _int number = gen_get_tabulated_name_basic(domain, key);

  if (number == (_int) HASH_UNDEFINED_VALUE)
  {
    return gen_chunk_undefined;
  }
  else
  {
    gen_tabulated_p gtp = get_tabulated_from_domain(domain);
    message_assert("valid tabulated number", number>=0 && number<gtp->size);
    return gtp->table[number].p;
  }
}

static void positive_number(gen_chunk * o)
{
  message_assert("positive tabulated number", (o+1)->i>0);
}

int gen_read_and_check_tabulated(FILE *file, int create_p)
{
  int domain = gen_read_tabulated(file, create_p);
  gen_mapc_tabulated(positive_number, domain);
  return domain;
}

/******************************************************************** IN/OUT */

/* GEN_CLEAR_TABULATED_ELEMENT only clears the entry for object OBJ in the
   gen_tabulated_ and gen_tabulated_names tables. */

void gen_clear_tabulated_element(gen_chunk * obj)
{
  register int domain = quick_domain_index(obj);
  gen_tabulated_p gtp = get_tabulated_from_domain(domain);
  int number = (obj+1)->i;

  message_assert("correct object to delete", gtp->table[number].p == obj);

  gen_delete_tabulated_name(obj);
  gtp->table[number].p = gen_chunk_undefined;
  gtp->used--;
}

gen_chunk *
gen_do_enter_tabulated(int domain, string id, gen_chunk * cp, bool is_a_ref)
{
  int number = gen_put_tabulated(domain, cp);
  (cp+1)->i = is_a_ref? -number: number; /* stores - if ref */
  message_assert("name pointer ok", (cp+2)->s == id);
  gen_put_tabulated_name(domain, id, number);
  return cp;
}

/* ENTER_TABULATED_DEF enters a new definition (previous refs are allowed if
   ALLOW_REF) in the INDEX tabulation table of the DOMAIN, with the unique
   ID and value CHUNKP.
 */
gen_chunk *
gen_enter_tabulated(int domain, string id, gen_chunk * cp, bool allow_ref)
{
  gen_chunk * gp = gen_find_tabulated(id, domain);

  if (gp==gen_chunk_undefined)
  {
    cp = gen_do_enter_tabulated(domain, id, cp, false);
  }
  else /* already in, redefine */
  {
    register int i, size, number = (gp+1)->i;

    if (number>0)
      fprintf(stderr, "warning: '%s' of %d redefined\n", id, domain);
    else
      number = -number;

    if (!allow_ref)
      fprintf(stderr, "unexpected reference to '%s' of %d\n", id, domain);

    size = gen_size(domain);

    message_assert("same name", same_string_p((gp+2)->s, (cp+2)->s));
    message_assert("same domain", gp->i == cp->i);

    for (i=3; i<size; i++)
      gp[i] = cp[i];

    free((cp+2)->s), free(cp), cp = gp;
    (cp+1)->i = number;
  }

  return cp;
}



