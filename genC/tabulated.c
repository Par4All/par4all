/*
 * $Id$
 *
 * management of tabulated domains.
 *
 */
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include <stdlib.h>
#include <setjmp.h>

#include "genC.h"
#include "newgen_include.h"

/*********************************************************** TABULATED STUFF */

/* table of tabulated elements.
   
   I guess it could be stored directly in the gen_binding,
   and the index could be dropped.
 */

#define TABULATED_ELEMENTS_SIZE (1000)
#define TABULATED_ELEMENTS_INCR (10000)

typedef struct
{
  int domain;           /* domain number */
  int index;            /* index in table */
  int size;		/* current allocated size */
  int used;		/* how many are used */
  gen_chunk * table;	/* actual table */
} 
  gen_tabulated_t, * gen_tabulated_p;

static gen_tabulated_p all_tabulateds = NULL;

#define all_tabulateds_initialized() \
  message_assert("all tabulated is initialized", all_tabulateds)

static gen_tabulated_p get_tabulated_from_index(int index)
{
  all_tabulateds_initialized();
  check_index(index);
  return all_tabulateds + index;
}

static gen_tabulated_p get_tabulated_from_domain(int domain)
{
  gen_tabulated_p gtp;
  check_domain(domain);

  gtp = get_tabulated_from_index(Domains[domain].index);
  
  message_assert("coherent domain", gtp->domain==domain);
  return gtp;
}

static void init_tabulated(int index)
{
  int i;
  gen_chunk * t;
  gen_tabulated_p gtp = get_tabulated_from_index(index);

  gtp->size = TABULATED_ELEMENTS_SIZE;
  gtp->used = 0;
  gtp->index = index;
  gtp->domain = -1; /* not assigned yet. */
  t = (gen_chunk*) alloc(sizeof(gen_chunk)*gtp->size);
  
  for (i=0; i<TABULATED_ELEMENTS_SIZE; i++)
    t[i].p = gen_chunk_undefined;
  
  gtp->table = t;
}

void gen_init_all_tabulateds(void)
{
  int i;
  message_assert("all tabulated not initialized", !all_tabulateds);

  all_tabulateds = (gen_tabulated_p) 
    alloc(sizeof(gen_tabulated_t)*MAX_TABULATED);

  for (i=0; i<MAX_TABULATED; i++)
    init_tabulated(i);
}

/* bof */
void gen_init_tabulated_set_domain(int index, int domain)
{
  gen_tabulated_p gtp = get_tabulated_from_index(index);
  message_assert("same index, not set yet, coherent", 
		 gtp->index==index && gtp->domain==-1 &&
		 Domains[domain].index == index);
  gtp->domain = domain;
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

  for (l=NIL, i=0; i<size; i++)
    if (gtp->table[i].p != gen_chunk_undefined) {
      register gen_chunk * obj = gtp->table[i].p;
      if (filter(obj)) l = CONS(CHUNK, obj, l);
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

/***************************************************** GEN_TABULATED_NAMES */

/* I guess the chunks could be avoided?
 */
static hash_table gen_tabulated_names = NULL;

void gen_init_gen_tabulated_names(void)
{
  message_assert("NULL table", !gen_tabulated_names);
  /* can avoid to allocate a holder since positive ints are stored... */
  message_assert("hack assumption", ((int)HASH_UNDEFINED_VALUE)<0);
  gen_tabulated_names = hash_table_make(hash_string, 1000);
}

void gen_lazy_init_gen_tabulated_names(void)
{
  if (!gen_tabulated_names) gen_init_gen_tabulated_names();
}

void gen_close_gen_tabulated_names(void)
{
  message_assert("defined table", gen_tabulated_names);
  hash_table_free(gen_tabulated_names);
  gen_tabulated_names = NULL;
}

#define check_gen_tabulated_names() \
  message_assert("gen_tabulated_names defined", gen_tabulated_names)

/* returns a pointer to a static string "number|name"
 */
static char * gen_build_unique_tabulated_name(int domain, char * name)
{
  int len = strlen(name);
  
  /* persistant buffer */
  static int size = 0;
  static char * buffer = 0;
  
  if (len+30>size) {
    size = len+30;
    if (buffer) free(buffer);
    buffer = (char*) malloc(sizeof(char)*size);
    if (!buffer) fatal("build_unique_tabulated_name: memory exhausted\n");
  }
  
  sprintf(buffer, "%d%c%s", domain, HASH_SEPAR, name);
  return buffer;
}

static char * build_unique_tabulated_name_for_obj(gen_chunk * obj)
{
  char * name = (obj+HASH_OFFSET)->s;
  int domain = quick_domain_index(obj);
  return gen_build_unique_tabulated_name(domain, name);
}	

/* deletes obj from the tabulated names...
 */
static void gen_delete_tabulated_name(gen_chunk * obj)
{
  char * key = build_unique_tabulated_name_for_obj(obj);
  void * okey, * val;
  check_gen_tabulated_names();
  
  val = hash_delget(gen_tabulated_names, key, &okey);
  if (val == HASH_UNDEFINED_VALUE)
    fatal("gen_delete_tabulated_name: clearing unexisting (%s)\n", key);
  
  free(okey);
}

static int gen_get_tabulated_name_basic(int domain, char * id)
{
  char * key = gen_build_unique_tabulated_name(domain, id);
  check_gen_tabulated_names();
  return (int) hash_get(gen_tabulated_names, key);
}

static void gen_put_tabulated_name(int domain, char * id, int number)
{
  char * key = gen_build_unique_tabulated_name(domain, id);
  check_gen_tabulated_names();
  message_assert("positive tabulated number", number>0);
  hash_put(gen_tabulated_names, strdup(key), (void*)number);
}

void * gen_find_tabulated(char * key, int domain)
{
  int number = gen_get_tabulated_name_basic(domain, key);

  if (number == (int) HASH_UNDEFINED_VALUE)
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
  gen_put_tabulated_name(domain, id, number);
  return cp;
}

/* ENTER_TABULATED_DEF enters a new definition (previous refs are allowed if
   ALLOW_REF) in the INDEX tabulation table of the DOMAIN, with the unique
   ID and value CHUNKP.
 */
gen_chunk * 
gen_enter_tabulated(int domain, string id, gen_chunk * cp, int allow_ref)
{
  gen_chunk * gp = gen_find_tabulated(id, domain);
  
  if (gp==gen_chunk_undefined)
  {
    cp = gen_do_enter_tabulated(domain, id, cp, FALSE);
  }
  else
  {
    register int i, size, number = (gp+1)->i;

    if (number>0) 
      fprintf(stderr, "warning: '%d|%s' redefined\n", domain, id);
    else
      number = -number;

    if (!allow_ref)
      fprintf(stderr, "unexpected reference to '%d|%s'\n", domain, id);

    size = gen_size(domain);
    
    for (i=0; i<size; i++) 
      gp[i] = cp[i];

    free(cp), cp = gp;
    (cp+1)->i = number;
  }

  return cp;
}



