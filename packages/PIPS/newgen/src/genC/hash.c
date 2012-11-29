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
/*

	-- NewGen Project

	The NewGen software has been designed by Remi Triolet and Pierre
	Jouvelot (Ecole des Mines de Paris). This prototype implementation
	has been written by Pierre Jouvelot.

	This software is provided as is, and no guarantee whatsoever is
	provided regarding its appropriate behavior. Any request or comment
	should be sent to newgen@cri.ensmp.fr.

	(C) Copyright Ecole des Mines de Paris, 1989

*/
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#include "newgen_types.h"
#include "genC.h"
#include "newgen_include.h"
#include "newgen_hash.h"

/* Some predefined values for the key
 */
#define HASH_ENTRY_FREE ((void *) 0)
#define HASH_ENTRY_FREE_FOR_PUT ((void *) -1)

typedef struct {
  void * key;
  void * val;
} hash_entry;

typedef void * (*hash_key_func_t)(const void *);
typedef void (*hash_free_func_t)(void *);

struct __hash_table
{
  hash_key_type type;		// the type of keys...
  size_t size;			// size of actual array
  size_t n_entry;		// number of associations stored
  hash_rank_t rank;		// how to compute rank for key
  hash_equals_t equals;		// how to compare keys
  hash_key_func_t store_key;	// possibly duplicate the key when storing
  hash_free_func_t delete_key;	// possibly free the memory for the key...
  hash_entry *array;		// actual array
  size_t limit;			// max entries before reallocation

  /* keep statistics on the life time of the hash table... FC 04/06/2003 */
  size_t n_free_for_puts,
    n_put, n_get, n_del, n_upd,
    n_put_iter, n_get_iter, n_del_iter, n_upd_iter;
};

/* Constant to find the end of the prime numbers table
 */
#define END_OF_SIZE_TABLE (0)

/* Hash function to get the index of the array from the key
 */
/* Does not work :
   #if sizeof(uintptr_t) == 8
   so go into less portable way: */
#if __WORDSIZE == 64
#define RANK(key, size) ((((_uint)(key)) ^ 0xC001BabeFab1C0e1LLU)%(size))
#else
/* Fabien began with this joke... :-) */
#define RANK(key, size) ((((_uint)(key)) ^ 0xfab1c0e1U)%(size))
#endif

/* Private functions
 */
static void hash_enlarge_table(hash_table htp);
static hash_entry * hash_find_entry(const hash_table htp,
				    const void * key,
				    _uint * prank,
				    _uint * stats);
static string hash_print_key(hash_key_type, const void *);

static int hash_int_equal(const int, const int);
static _uint hash_int_rank(const void*, size_t);
static int hash_pointer_equal(const void*, const void*);
static _uint hash_pointer_rank(const void*, size_t);
static int hash_string_equal(const char*, const char*);
extern _uint hash_string_rank(const void*, size_t);
static int hash_chunk_equal(const gen_chunk*, const gen_chunk*) ;
static _uint hash_chunk_rank(const gen_chunk*, size_t);

/* list of the prime numbers from 17 to 2^31-1
 * used as allocated size
 */
static size_t prime_list[] = {
    7,
    17,
    37,
    71,
    137,
    277,
    547,
    1091,
    2179,
    4357,
    8707,
    17417,
    34819,
    69653,
    139267,
    278543,
    557057,
    1114117,
    2228243,
    4456451,
    8912921,
    17825803,
    35651593,
    71303171,
    142606357,
    285212677,
    570425377,
    1140850699,
    END_OF_SIZE_TABLE
};

/* returns the maximum number of things to hold in a table
 */
static size_t hash_size_limit(size_t current_size)
{
  /* 25.0% : ((current_size)>>2)
   * 50.0% : ((current_size)>>1)
   * next values are TOO MUCH!
   * 62.5% : (((current_size)>>1)+((current_size)>>3))
   * 75.0% : (((current_size)>>1)+((current_size)>>2))
   */
  return current_size>>1;
}

/* Now we need the table size to be a prime number.
 * So we need to retrieve the next prime number in a list.
 */
static size_t get_next_hash_table_size(size_t size)
{
  size_t * p_prime = prime_list;
  while (*p_prime <= size) {
    message_assert("size too big ", *p_prime != END_OF_SIZE_TABLE);
    p_prime++;
  }
  return *p_prime;
}

/* internal variable to know we should warm or not
 */
static bool should_i_warn_on_redefinition = true;

/* these function set the variable should_i_warn_on_redefinition
   to the value true or false */

void hash_warn_on_redefinition(void)
{
  should_i_warn_on_redefinition = true;
}

void hash_dont_warn_on_redefinition(void)
{
  should_i_warn_on_redefinition = false;
}

bool hash_warn_on_redefinition_p(void)
{
  return should_i_warn_on_redefinition;
}

static void * hash_store_string(const void * s)
{
  return strdup((char*) s);
}

static void hash_free_string(void * s)
{
  free(s);
}

/* this function makes a hash table of size size. if size is less or
   equal to zero a default size is used. the type of keys is given by
   key_type (see hash.txt for further details; where is hash.txt?).

   private_equal_p() is a predicate to decide if two elements are equal
   or not

   private_rank() returns an integer in interval [0,size-1]

   if private_equal_p(e1, e2) then private_rank(e1)==private_rank(e2)
   or results are unpredicatbale.

   No functionality has been used or tested for hash_type==hash_private.
 */
hash_table
hash_table_generic_make(hash_key_type key_type,
			size_t size,
			hash_equals_t private_equal_p,
			hash_rank_t private_rank)
{
  size_t i;
  hash_table htp;

  if (size<HASH_DEFAULT_SIZE) size=HASH_DEFAULT_SIZE - 1;
  // get the next prime number in the table
  size = get_next_hash_table_size(size);

  htp = (hash_table) alloc(sizeof(struct __hash_table));
  message_assert("allocated", htp);

  htp->type = key_type;
  htp->size = size;
  htp->n_entry = 0;
  htp->limit = hash_size_limit(size);
  htp->array = (hash_entry*) alloc(size*sizeof(hash_entry));

  // initialize statistics
  htp->n_free_for_puts = 0;
  htp->n_put = 0;
  htp->n_get = 0;
  htp->n_del = 0;
  htp->n_upd = 0;

  htp->n_put_iter = 0;
  htp->n_get_iter = 0;
  htp->n_del_iter = 0;
  htp->n_upd_iter = 0;

  for (i = 0; i < size; i++)
    htp->array[i].key = HASH_ENTRY_FREE;

  htp->store_key = NULL;
  htp->delete_key = NULL;

  switch(key_type)
  {
  case hash_string:
    htp->equals = (int(*)(const void*,const void*)) hash_string_equal;
    htp->rank = hash_string_rank;
    htp->store_key = hash_store_string;
    htp->delete_key = hash_free_string;
    break;
  case hash_int:
    htp->equals = (int(*)(const void*,const void*)) hash_int_equal;
    htp->rank = hash_int_rank;
    break;
  case hash_chunk:
    htp->equals = (int(*)(const void*,const void*)) hash_chunk_equal;
    htp->rank = (_uint (*)(const void*, _uint)) hash_chunk_rank;
    break;
  case hash_pointer:
    htp->equals = hash_pointer_equal;
    htp->rank = hash_pointer_rank;
    break;
  case hash_private:
    htp->equals = private_equal_p;
    htp->rank = private_rank;
    break;
  default:
    fprintf(stderr, "[make_hash_table] bad type %d\n", key_type);
    abort();
  }

  return htp;
}

hash_table hash_table_make(hash_key_type key_type, size_t size)
{
  message_assert("key_type is not hash_private for this interface",
		 key_type!=hash_private);
  /* Use default functions for equality check and rank computation. */
  return hash_table_generic_make(key_type, size, NULL, NULL);
}

static size_t max_size_seen = 0;

/* Clears all entries of a hash table HTP. [pj] */
void hash_table_clear(hash_table htp)
{
  hash_entry * p, * end ;

  if (htp->size > max_size_seen) {
    max_size_seen = htp->size;
#ifdef DBG_HASH
    fprintf(stderr, "[hash_table_clear] maximum size is %d\n", max_size_seen);
#endif
  }

  end = htp->array + htp->size ;
  htp->n_entry = 0 ;

  for ( p = htp->array ; p < end ; p++ ) {
    p->key = HASH_ENTRY_FREE ;
  }
}

/* this function deletes a hash table that is no longer useful. unused
 memory is freed. */

void hash_table_free(hash_table htp)
{
  // free allocated keys if necessary
  if (htp->delete_key)
  {
    size_t i;
    for (i=0; i<htp->size; i++)
      if (htp->array[i].key != HASH_ENTRY_FREE &&
	  htp->array[i].key != HASH_ENTRY_FREE_FOR_PUT)
	htp->delete_key(htp->array[i].key);
  }
  gen_free_area((void**) htp->array, htp->size*sizeof(hash_entry));
  gen_free_area((void**) htp, sizeof(struct __hash_table));
}

/* hash_put which allows silent overwrite...
 */
void hash_overwrite(hash_table htp, const void * key, const void * val)
{
  if (hash_defined_p(htp, key))
    hash_update(htp, key, val);
  else
    hash_put(htp, key, val);
}

/* This functions stores a couple (key,val) in the hash table pointed to
   by htp. If a couple with the same key was already stored in the table
   and if hash_warn_on_redefintion was requested, hash_put complains but
   replace the old value by the new one. This is a potential source for a
   memory leak. If the value to store is HASH_UNDEFINED_VALUE or if the key
   is HASH_ENTRY_FREE or HASH_ENTRY_FREE_FOR_INPUT, hash_put
   aborts. The restrictions on the key should be avoided by changing
   the implementation. The undefined value should be
   user-definable. It might be argued that users should be free to
   assign HASH_UNDEFINED_VALUE, but they can always perform hash_del()
   to get the same result */

void hash_put(hash_table htp, const void * key, const void * val)
{
  _uint rank;
  hash_entry * hep;

  if (htp->n_entry+1 >= (htp->limit))
    hash_enlarge_table(htp);

  message_assert("legal input key",
		 key!=HASH_ENTRY_FREE && key!=HASH_ENTRY_FREE_FOR_PUT);
  message_assert("input value must be defined", val!=HASH_UNDEFINED_VALUE);

  htp->n_put++;
  hep = hash_find_entry(htp, key, &rank, &htp->n_put_iter);

  if (hep->key != HASH_ENTRY_FREE && hep->key != HASH_ENTRY_FREE_FOR_PUT) {
    if (should_i_warn_on_redefinition && hep->val != val) {
      (void) fprintf(stderr, "[hash_put] key redefined: %s\n",
		     hash_print_key(htp->type, key));
    }
    hep->val = (void *) val;
  }
  else {
    if (hep->key == HASH_ENTRY_FREE_FOR_PUT)
      htp->n_free_for_puts--;
    htp->n_entry += 1;
    hep->key = htp->store_key? htp->store_key(key): (void*) key;
    hep->val = (void *) val;
  }
}

/* deletes key from the hash table. returns the val and key
 */
void *
hash_delget(
    hash_table htp,
    const void * key,
    void ** pkey)
{
    hash_entry * hep;
    void *val;
    _uint rank;

    /* FI: the stack is destroyed by assert; I need to split the
       statement to put a breakpoint just before the stack
       disappears. */
    if(!(key!=HASH_ENTRY_FREE && key!=HASH_ENTRY_FREE_FOR_PUT))
      message_assert("legal input key",
		     key!=HASH_ENTRY_FREE && key!=HASH_ENTRY_FREE_FOR_PUT);

    htp->n_del++;
    hep = hash_find_entry(htp, key, &rank, &htp->n_del_iter);

    if (hep->key != HASH_ENTRY_FREE && hep->key != HASH_ENTRY_FREE_FOR_PUT) {
	// argh... was hep->key... cannot return it!
	if (htp->delete_key)
	  htp->delete_key(hep->key), *pkey = NULL;
	else
	  *pkey = hep->key;
	val = hep->val;
	htp->array[rank].key = HASH_ENTRY_FREE_FOR_PUT;
	htp->array[rank].val = NULL;
	htp->n_entry -= 1;
	htp->n_free_for_puts++;
	return val;
    }

    *pkey = 0;
    return HASH_UNDEFINED_VALUE;
}

/* this function removes from the hash table pointed to by htp the
   couple whose key is equal to key. nothing is done if no such couple
   exists. ??? should I abort ? (FC)
 */
void * hash_del(hash_table htp, const void * key)
{
    void * tmp;
    return hash_delget(htp, key, &tmp);
}

/* this function retrieves in the hash table pointed to by htp the
   couple whose key is equal to key. the HASH_UNDEFINED_VALUE pointer is
   returned if no such couple exists. otherwise the corresponding value
   is returned. */
void * hash_get(const hash_table htp, const void * key)
{
  hash_entry * hep;
  _uint n;

  /* FI: the stack is destroyed by assert; I need to split the
     statement to put a breakpoint just before the stack
     disappears. */
  if(!(key!=HASH_ENTRY_FREE && key!=HASH_ENTRY_FREE_FOR_PUT))
    message_assert("legal input key",
		   key!=HASH_ENTRY_FREE && key!=HASH_ENTRY_FREE_FOR_PUT);

  if (!htp->n_entry)
    return HASH_UNDEFINED_VALUE;

  /* else may be there */
  htp->n_get++;
  hep = hash_find_entry(htp, key, &n, &htp->n_get_iter);

  return hep->key!=HASH_ENTRY_FREE &&
    hep->key!=HASH_ENTRY_FREE_FOR_PUT ? hep->val : HASH_UNDEFINED_VALUE;
}


/* Like hash_get() but returns an empty list instead of
   HASH_UNDEFINED_VALUE when a key is not found */
list hash_get_default_empty_list(const hash_table h, const void * k) {
  list l = (list) hash_get(h, k);

  return (l == (list) HASH_UNDEFINED_VALUE) ? NIL : l;
}


/* true if key has e value in htp.
 */
bool hash_defined_p(const hash_table htp, const void * key)
{
  return hash_get(htp, key)!=HASH_UNDEFINED_VALUE;
}

/* update key->val in htp, that MUST be pre-existent.
 */
void hash_update(hash_table htp, const void * key, const void * val)
{
  hash_entry * hep;
  _uint n;

  message_assert("illegal input key",
		 key!=HASH_ENTRY_FREE && key!=HASH_ENTRY_FREE_FOR_PUT);

  htp->n_upd++;
  hep = hash_find_entry(htp, key, &n, &htp->n_upd_iter);

  message_assert("some previous entry", htp->equals(hep->key, key));

  hep->val = (void *) val;
}

/* this function prints the header of the hash_table pointed to by htp
 * on the opened stream fout.
 */
void hash_table_print_header(hash_table htp, FILE *fout)
{
  fprintf(fout, "hash_key_type:     %d\n", htp->type);
  fprintf(fout, "size:         %zd\n", htp->size);
  /* to be used by pips, we should not print this
     as it is only for debugging NewGen and it is not important data
     I (go) comment it.
     fprintf(fout, "limit    %d\n", htp->limit);
  */
  fprintf(fout, "n_entry: %zd\n", htp->n_entry);
}

/* this function prints the content of the hash_table pointed to by htp
on stderr. it is mostly useful when debugging programs. */

void hash_table_print(hash_table htp)
{
  size_t i;

  hash_table_print_header (htp,stderr);

  for (i = 0; i < htp->size; i++) {
    hash_entry he;

    he = htp->array[i];

    if (he.key != HASH_ENTRY_FREE && he.key != HASH_ENTRY_FREE_FOR_PUT) {
      fprintf(stderr, "%zd %s %p\n",
	      i, hash_print_key(htp->type, he.key),
	      he.val);
    }
  }
}

/* This function prints the content of the hash_table pointed to by htp
on file descriptor f, using functions key_to_string and value_to string
to display the mapping. it is mostly useful when debugging programs. */

void hash_table_fprintf(FILE * f, gen_string_func_t key_to_string,
		gen_string_func_t value_to_string, const hash_table htp)
{
    size_t i;

    hash_table_print_header (htp,f);

    for (i = 0; i < htp->size; i++) {
	hash_entry he;

	he = htp->array[i];

	if (he.key != HASH_ENTRY_FREE && he.key != HASH_ENTRY_FREE_FOR_PUT) {
	    fprintf(f, "%s -> %s\n",
		    key_to_string(he.key), value_to_string(he.val));
	}
    }
}

void hash_table_dump(const hash_table htp)
{
    size_t i;

    hash_table_print_header (htp,stderr);

    for (i = 0; i < htp->size; i++) {
	hash_entry he = htp->array[i];

	if (he.key != HASH_ENTRY_FREE && he.key != HASH_ENTRY_FREE_FOR_PUT)
	  fprintf(stderr, "%zd: %p -> %p\n", i, he.key, he.val);
	else if(he.key == HASH_ENTRY_FREE && he.key != HASH_ENTRY_FREE_FOR_PUT)
	  fprintf(stderr, "%zd: FREE\n", i);
	else
	  fprintf(stderr, "%zd: FREE FOR PUT\n", i);
    }
}

/* function to enlarge the hash_table htp.
 * the new size will be first number in the array prime_numbers_for_table_size
 * that will be greater or equal to the actual size
 */

static void
hash_enlarge_table(hash_table htp)
{
  hash_entry * old_array;
  size_t i, old_size;

  old_size = htp->size;
  old_array = htp->array;

  htp->size++;
  /* Get the next prime number in the table */
  htp->size = get_next_hash_table_size(htp->size);
  htp->array = (hash_entry *) alloc(htp->size* sizeof(hash_entry));
  htp->limit = hash_size_limit(htp->size);

  for (i = 0; i < htp->size ; i++)
    htp->array[i].key = HASH_ENTRY_FREE;

  for (i = 0; i < old_size; i++)
  {
    hash_entry he;
    he = old_array[i];

    if (he.key != HASH_ENTRY_FREE && he.key != HASH_ENTRY_FREE_FOR_PUT) {
      hash_entry * nhep;
      _uint rank;

      htp->n_put++;
      nhep = hash_find_entry(htp, he.key, &rank, &htp->n_put_iter);

      if (nhep->key != HASH_ENTRY_FREE) {
	fprintf(stderr, "[hash_enlarge_table] fatal error\n");
	abort();
      }
      htp->array[rank] = he;
    }
  }
  gen_free_area((void**)old_array, old_size*sizeof(hash_entry));
}

/* en s'inspirant vaguement de
 *   Fast Hashing of Variable-Length Text Strings
 *   Peter K. Pearson
 *   CACM vol 33, nb 6, June 1990
 * qui ne donne qu'une valeur entre 0 et 255
 *
 * unsigned int T[256] with random values
 * unsigned int h = 0;
 * for (char * s = (char*) key; *s; s++)
 *   h = ROTATION(...,h) ^ T[ (h^(*s)) % 256];
 * mais...
 */
_uint hash_string_rank(const void * key, size_t size)
{
  _uint v = 0;
  const char * s;
  for (s = (char*) key; *s; s++)
    /* FC: */ v = ((v<<7) | (v>>25)) ^ *s;
    /* GO: v <<= 2, v += *s; */
  return v % size;
}

static _uint hash_int_rank(const void * key, size_t size)
{
  return RANK(key, size);
}

static _uint hash_pointer_rank(const void * key, size_t size)
{
  return RANK(key, size);
}

static _uint hash_chunk_rank(const gen_chunk * key, size_t size)
{
  return RANK(key->i, size);
}

static int hash_string_equal(const char * key1, const char * key2)
{
  if (key1==key2)
    return true;
  /* else check contents */
  for(; *key1 && *key2; key1++, key2++)
    if (*key1!=*key2)
      return false;
  return *key1==*key2;
}

static int hash_int_equal(const int key1, const int key2)
{
  return key1 == key2;
}

static int hash_pointer_equal(const void * key1, const void * key2)
{
  return key1 == key2;
}

static int hash_chunk_equal(const gen_chunk * key1, const gen_chunk * key2)
{
  return key1->p == key2->p;
}

static char * hash_print_key(hash_key_type t, const void * key)
{
  static char buffer[32]; /* even 8 byte pointer => ~16 chars */

  if (t == hash_string)
    return (char*) key; /* hey, this is just what we need! */
  /* Use extensive C99 printf formats and stdint.h types to avoid
     truncation warnings: */
  else if (t == hash_int)
    sprintf(buffer, "%td", (_int) key);
  else if (t == hash_pointer || t == hash_private)
    sprintf(buffer, "%p", key);
  else if (t == hash_chunk)
    sprintf(buffer, "%zx", (_uint) ((gen_chunk *)key)->p);
  else {
    fprintf(stderr, "[hash_print_key] bad type : %d\n", t);
    abort();
  }

  return buffer;
}


/* distinct primes for long cycle incremental search */
static int inc_prime_list[] = {
    2,   3,   5,  11,  13,  19,  23,  29,  31,  41,
   43,  47,  53,  59,  61,  67,  73,  79,  83,  89,
   97, 101, 103, 107, 109, 113, 127, 131, 139, 149,
  151
};

#define HASH_INC_SIZE (31) /* 31... (yes, this one is prime;-) */


/* return where the key is [to be] stored, depending on the operation.
 */
static hash_entry *
hash_find_entry(hash_table htp,
		const void * key,
		_uint *prank,
		_uint * stats)
{
  register size_t
    r_init = (*(htp->rank))(key, htp->size),
    r = r_init,
    /* history of r_inc value
     * RT: 1
     * GO: 1 + abs(r_init)%(size-1)
     * FC: inc_prime_list[ RANK(r_init, HASH_INC_SIZE) ]
     * FC rationnal: if r_init is perfect, 1 is fine...
     *    if it is not perfect, let us randmize here some more...
     *    I'm not sure the result is any better than 1???
     *    It does seems to help significantly on some examples...
     */
    r_inc  = inc_prime_list[ RANK(r_init, HASH_INC_SIZE) ],
    r_first_free = r_init;
  bool first_free_found = false;
  hash_entry he;

  while (1)
  {
    /* FC 05/06/2003
     * if r_init is randomized (i.e. perfect hash function)
     * and r_inc does not kill everything (could it?)
     * if p is the filled proportion for the table, 0<=p<1
     * we should have number_of_iterations
     *        = \Sigma_{i=1}{\infinity} i*(1-p)p^{i-1}
     * this formula must simplify somehow... = 1/(1-p) ?
     * 0.20   => 1.25
     * 0.25   => 1.33..
     * 0.33.. => 1.50
     * 0.50   => 2.00
     * 0.66.. => 3.00
     * 0.70   => 3.33
     */
    if (stats) (*stats)++;

    /* the r-th entry */
    he = htp->array[r];

    /* a possible never used place is found, stop seeking!
     * and return first free found or current if none.
     */
    if (he.key == HASH_ENTRY_FREE) {
      if (first_free_found)
	r = r_first_free;
      break;
    }

    /* this is a possible place for storing, but the key may be there anyway...
     * so we keep on seeking, but keep the first found place.
     */
    if (he.key == HASH_ENTRY_FREE_FOR_PUT && !first_free_found) {
      r_first_free = r;
      first_free_found = true;
    }

    /* the key is found! */
    if (he.key != HASH_ENTRY_FREE_FOR_PUT &&
	(*(htp->equals))(he.key, key))
      break;

    /* GO: it is not anymore the next slot, we skip some of them depending
     * on the reckonned increment
     */
    r = (r + r_inc) % htp->size;

    /* argh! we have made a full round...
     * it may happen after many put and del, if the table contains no FREE,
     * but only many FREE_FOR_PUT instead.
     */
    if(r == r_init) {
      message_assert("one free place was seen", first_free_found);
      r = r_first_free;
      break;
    }
  }

  *prank = r;
  return &(htp->array[r]);
}

/* now we define observers in order to
 * hide the hash_table type...
 */
int hash_table_entry_count(hash_table htp)
{
    return htp->n_entry;
}

/* returns the size of the internal array.
 */
int hash_table_size(hash_table htp)
{
    return htp->size;
}

/* returns the type of the hash_table.
 */
hash_key_type hash_table_type(hash_table htp)
{
    return htp->type;
}

/*
 * This function allows a hash_table scanning
 * First you give a NULL hentryp and get the key and val
 * After you give the previous hentryp and so on
 * at the end NULL is returned
 */
void *
hash_table_scan(hash_table htp,
		void * hentryp_arg,
		void ** pkey,
		void ** pval)
{
  hash_entry * hentryp = (hash_entry *) hentryp_arg;
  hash_entry * hend = htp->array + htp->size;

  if (!hentryp)	hentryp = (void*) htp->array;

  while (hentryp < hend)
  {
    void *key = hentryp->key;

    if ((key !=HASH_ENTRY_FREE) && (key !=HASH_ENTRY_FREE_FOR_PUT))
    {
      *pkey = key;
      if (pval) *pval = hentryp->val;
      return hentryp + 1;
    }
    hentryp++;
  }
  return NULL;
}

int hash_table_own_allocated_memory(hash_table htp)
{
    return htp ?
      sizeof(struct __hash_table) + sizeof(hash_entry)*(htp->size) : 0 ;
}

/***************************************************************** MAP STUFF */
/* newgen mapping to newgen hash...
 */
void * hash_map_get(const hash_table h, const void * k)
{
  gen_chunk key, * val;
  key.e = (void *) k;
  val = (gen_chunk*) hash_get(h, &key);
  if (val==HASH_UNDEFINED_VALUE)
    fatal("no value correspond to key %p", k);
  return val->e;
}

bool hash_map_defined_p(const hash_table h, const void * k)
{
  gen_chunk key;
  key.e = (void *) k;
  return hash_defined_p(h, &key);
}

void hash_map_put(hash_table h, const void * k, const void * v)
{
  gen_chunk
    * key = (gen_chunk*) alloc(sizeof(gen_chunk)),
    * val = (gen_chunk*) alloc(sizeof(gen_chunk));
  key->e = (void *) k;
  val->e = (void *) v;
  hash_put(h, key, val);
}

void * hash_map_del(hash_table h, const void * k)
{
  gen_chunk key, * oldkeychunk, * val;
  void * result;

  key.e = (void *) k;
  val = hash_delget(h, &key, (void**) &oldkeychunk);
  message_assert("defined value (entry to delete must be defined!)",
		 val!=HASH_UNDEFINED_VALUE);
  result = val->e;

  oldkeychunk->p = NEWGEN_FREED;
  free(oldkeychunk);

  val->p = NEWGEN_FREED;
  free(val);

  return result;
}

void hash_map_update(hash_table h,  const void * k, const void * v)
{
  hash_map_del(h, k);
  hash_map_put(h, k, v);
}

/* Because the hash table data structure is hidden in this source
   file hash.c and not exported via the newgen_include.h, it is not
   possible to access its fields in other files, e.g. set.c. */
hash_equals_t hash_table_equals_function(hash_table h)
{
  return h->equals;
}

hash_rank_t hash_table_rank_function(hash_table h)
{
  return h->rank;
}
