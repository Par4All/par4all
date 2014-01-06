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
   Pierre Jouvelot (3 Avril 1989)

   Set package for any type of pointer.

   To avoid sharing problem, all the routines are 3-adress: S1 = S2 op S3.
   It is up to the user to know what to do (e.g., freeing some temporary
   memory storage) before S1 is assigned a new value.

   Shallow copies are used. If set A is assigned to set B, side
   effects on elements of A are visible in B.

   Formal parameters are modified in functions which makes the stack
   misleading when debugging with gdb.

   The sets are implemented with hash-tables. Each element is stored
   as key and value.

  */
#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include "genC.h"
#include "newgen_set.h"

/* FI: I do not understand why the type is duplicated at the set
   level. Is there a potential consistency problem with the hash
   table type? Is this a consequence of the decision to hide the
   actual hash_table data structure?
*/
struct _set_chunk {
  hash_table table;
  set_type type;
};

#define INITIAL_SET_SIZE 10

/* return the internal hash table of set s
 */
hash_table set_private_get_hash_table(const set s)
{
  return s->table;
}

/* return the type of set s
 */
set_type set_get_type(const set s)
{
  return s->type;
}

/* Create an empty set of any type */
/* discrepancy: size_t sometimes, _uint elsewhere */
/* why not use the functional types now defined in newgen_hash.h? */
set set_generic_make(set_type typ,
		     hash_equals_t private_equal_p,
		     hash_rank_t private_rank)
{
  set hp = (set) alloc(sizeof(struct _set_chunk));
  message_assert("allocated", hp);

  hp->table = hash_table_generic_make( typ,
				       INITIAL_SET_SIZE,
				       private_equal_p,
				       private_rank ) ;
  hp->type = typ ;
  return hp;
}

/* Create an empty set of any type but hash_private */
set set_make(set_type typ)
{
  message_assert("typ is not set_private", typ!=set_private);
  /* Use default functions for equality check and rank computation. */
  return set_generic_make(typ, NULL, NULL);
}

/* create a singleton set of any type but hash_private
 *
 * use set_add_element() instead for hash_private
 */
set set_singleton(set_type type, const void * p)
{
  set s = set_make( type ) ;
  hash_put( s->table, p, p ) ;
  return s;
}

/* Assign a set with the content of another set.

   @param s1 the set to write into
   @param s2 the set to copy

   If the same set is given twice, nothing is done.

   @return the target set.
*/
set set_assign(set s1, const set s2)
{
  if (s1 == s2) {
    return s1;
  }
  else {
    set_clear(s1);
    HASH_MAP(k, v, hash_put( s1->table, k, v ), s2->table);
    return s1;
  }
}

/* @return duplicated set
 */
set set_dup(const set s)
{
  set n = set_make(s->type);
  HASH_MAP(k, v, hash_put(n->table, k, v ), s->table);
  return n;
}

/* @return s1 = s2 u { e }.
 */
set set_add_element(set s1, const set s2, const void * e)
{
  if( s1 == s2 ) {
    if (! set_belong_p(s1, e))
      hash_put(s1->table, e, e);
    return( s1 ) ;
  }
  else {
    set_clear( s1 ) ;
    HASH_MAP( k, v, {hash_put( s1->table, k, v ) ;}, s2->table ) ;
    if (! set_belong_p(s1, e))
      hash_put(s1->table, e, e);
    return( s1 ) ;
  }
}


/* @return s1 = s2 u { list of e }.
 */
set set_add_elements(set s1, const set s2, const void * e, ...)
{
  bool first = true;
  va_list args;
  /* The statement list */
  /* Analyze in args the variadic arguments that may be after s2: */
  va_start(args, e);
 /* Since a variadic function in C must have at least 1 non variadic
     argument (here the s), just skew the varargs analysis: */
  e = va_arg(args, void *);
  while(e) {
    // may copy s2 on first time
    s1 = set_add_element(s1, first? s2: s1, e);
    first = false;
    // get next element
    e = va_arg(args, void *);
  }
  /* Release the variadic analyzis: */
  va_end(args);
  return s1;
}
/* @return whether e \in s.
 */
bool set_belong_p(const set s, const void * e)
{
  return hash_get(s->table, e) != HASH_UNDEFINED_VALUE;
}

/* @return whether all items in l are in s
 */
bool list_in_set_p(const list l, const set s)
{
  FOREACH(chunk, c, l)
    if (!set_belong_p(s, c))
      return false;
  return true;
}

/* @return s1 = s2 u s3.
 */
set set_union(set s1, const set s2, const set s3)
{
  // do not warn on redefinitions when computing an union.
  bool warning = hash_warn_on_redefinition_p();
  if (warning) hash_dont_warn_on_redefinition();
  if( s1 != s3 ) {
    set_assign(s1, s2) ;
    HASH_MAP( k, v, hash_put( s1->table, k, v), s3->table ) ;
  }
  else {
    HASH_MAP( k, v, hash_put( s1->table, k, v), s2->table ) ;
  }
  if (warning) hash_warn_on_redefinition();
  return s1;
}

/* @return s1 = s2 n s3.
 */
set set_intersection(set s1, const set s2, const set s3)
{
  if( s1 != s2 && s1 != s3 ) {
    set_clear( s1 ) ;
    HASH_MAP( k, v, {if( hash_get( s2->table, k )
			 != HASH_UNDEFINED_VALUE )
	  hash_put( s1->table, k, v ) ;},
      s3->table ) ;
    return( s1 ) ;
  }
  else {
    set tmp = set_generic_make( s1->type,
				hash_table_equals_function(s1->table),
				hash_table_rank_function(s1->table) ) ;

    HASH_MAP( k, v, {if( hash_get( s1->table, k )
			 != HASH_UNDEFINED_VALUE )
	  hash_put( tmp->table, k, v ) ;},
      (s1 == s2) ? s3->table : s2->table ) ;
    set_assign( s1, tmp ) ;
    set_free( tmp ) ;
    return( s1 ) ;
  }
}

/* @return s1 = s2 - s3.
 */
set set_difference(set s1, const set s2, const set s3)
{
  set_assign(s1, s2);
  HASH_MAP(k, ignore, hash_del(s1->table, k), s3->table);
  return s1;
}

/* @return s1 = s2 - { e }.
 */
set set_del_element(set s1, const set s2, const void * e)
{
  set_assign( s1, s2 ) ;
  hash_del( s1->table, e );
  return s1;
}

/* May be useful for string sets ... NOT TESTED
 *
 * FI:Confusing for Newgen users because gen_free() is expected?
 */
set set_delfree_element(set s1, const set s2, const void * e)
{
  void * pe;
  set_assign(s1, s2);
  (void) hash_delget(s1->table, e ,&pe);
  free(pe);
  return s1;
}

/* returns whether s1 n s2 <> 0
 * complexity of the intersection
 */
bool set_intersection_p(const set s1, const set s2)
{
  bool non_empty_intersection;
  if (set_empty_p(s1) || set_empty_p(s2))
    non_empty_intersection = false;
  else
  {
    set inter = set_make(set_pointer);
    set_intersection(inter, s1, s2);
    non_empty_intersection = !set_empty_p(inter);
    set_free(inter);
  }
  return non_empty_intersection;
}

/* return whether s1 \included s2
 */
bool set_inclusion_p(const set s1, const set s2)
{
  if (s1==s2) return true;
  SET_FOREACH(void *, i, s1)
    if (!set_belong_p(s2, i))
      return false;
  return true;
}

/* returns whether s1 == s2
 */
bool set_equal_p(const set s1, const set s2)
{
  if (s1==s2) return true;
  return set_size(s1)==set_size(s2) &&
    set_inclusion_p(s1, s2) && set_inclusion_p(s2, s1);
}

/* Assign the empty set to s
 * s := {}
 */
set set_clear(set s)
{
  hash_table_clear(s->table);
  return s;
}

void set_free(set s)
{
  hash_table_free(s->table);
  gen_free_area((void**) s, sizeof(struct _set_chunk));
}

/* Free several sets in one call. Useful when many sets are used
   simultaneously. */
void sets_free(set s,...)
{
  va_list args;

  /* Analyze in args the variadic arguments that may be after t: */
  va_start(args, s);
  /* Since a variadic function in C must have at least 1 non variadic
     argument (here the s), just skew the varargs analysis: */
  do {
    set_free(s);
    /* Get the next argument */
    s = va_arg(args, set);
  } while(s!=NULL);
  /* Release the variadic analysis */
  va_end(args);
}

/* returns the number of items in s.
 */
int set_size(const set s)
{
  return hash_table_entry_count(s->table);
}

/* tell whether set s is empty.
 * returnn s=={}
 */
bool set_empty_p(const set s)
{
  return set_size(s)==0;
}

void gen_set_closure_iterate(void (*iterate)(void *, set),
			     set initial,
			     bool dont_iterate_twice)
{
  set curr, next, seen;
  set_type t = initial->type;

  seen = set_generic_make(t,
			  hash_table_equals_function(initial->table),
			  hash_table_rank_function(initial->table));
  curr = set_generic_make(t,
			  hash_table_equals_function(initial->table),
			  hash_table_rank_function(initial->table));
  next = set_generic_make(t,
			  hash_table_equals_function(initial->table),
			  hash_table_rank_function(initial->table));

  set_assign(curr, initial);

  while (!set_empty_p(curr))
  {
    SET_FOREACH(void *, x, curr)
      iterate(x, next);

    if (dont_iterate_twice)
    {
      set_union(seen, seen, curr);
      set_difference(curr, next, seen);
    }
    else
    {
      set_assign(curr, next);
    }
    set_clear(next);
  }

  set_free(curr);
  set_free(seen);
  set_free(next);

}

/* a set-based implementation of gen_closure
 * that does not go twice in the same object.
 * FC 27/10/95.
 */
void gen_set_closure(void (*iterate)(void *, set), set initial)
{
  gen_set_closure_iterate(iterate, initial, true);
}

int set_own_allocated_memory(const set s)
{
  return sizeof(struct _set_chunk)+hash_table_own_allocated_memory(s->table);
}

/**
 * create a list from a set
 * the set is not freed
 * @warning no assumption can be made on the ordering of returned list
 * @param s set where the data are
 *
 * @return an allocated list of elements from s
 */
list set_to_list(const set s)
{
  list l =NIL;
  SET_FOREACH(void *, v, s)
    l = gen_cons(v,l);
  return l;
}

/* @return a sorted list from a set.
 * provide comparison function as gen_sort_list, which calls "qsort".
 */
list set_to_sorted_list(const set s, gen_cmp_func_t cmp)
{
  list l = set_to_list(s);
  gen_sort_list(l, cmp);
  return l;
}

/**
 * add list l items to set s, which is returned.
 *
 * @param s modified set
 * @param l provided list
 */
set set_append_list(set s, const list l)
{
  FOREACH(chunk, i, l)
    set_add_element(s, s, i);
  return s;
}

/**
 * assigns a list contents to a set
 * all duplicated elements are lost
 *
 * @param s set being assigned to.
 * @param l list to turn into a set
 */
set set_assign_list(set s, const list l)
{
  set_clear(s);
  return set_append_list(s, l);
}

/************************************************** UTILITIES FOR DEBUGGING */

#include "newgen_string_buffer.h"

/* convert set s to a string_buffer
 * used internally, but may be exported if needed.
 */
static string_buffer
set_to_string_buffer(string name, const set s, gen_string_func_t item_name)
{
  string_buffer sb = string_buffer_make(true);
  if (name)
    string_buffer_cat(sb, name, " = (", itoa(set_size(s)), ") { ", NULL);
  else
    string_buffer_cat(sb, " (", itoa(set_size(s)), ") { ", NULL);

  bool first = true;
  SET_FOREACH(void *, i, s)
  {
    if (first) first = false;
    else string_buffer_append(sb, ", ");
    string_buffer_append(sb, item_name(i));
  }

  string_buffer_append(sb, " }");
  return sb;
}

/* return allocated string for set s
 * @param name set description
 * @param s the set
 * @param item_name user function to build a string for each item
 */
string set_to_string(string name, const set s, gen_string_func_t item_name)
{
  string_buffer sb = set_to_string_buffer(name, s, item_name);
  string res = string_buffer_to_string(sb);
  string_buffer_free(&sb);
  return res;
}

/* print set s to file stream out.
 */
void set_fprint
  (FILE * out, string name, const set s, gen_string_func_t item_name)
{
  string_buffer sb = set_to_string_buffer(name, s, item_name);
  string_buffer_to_file(sb, out);
  putc('\n', out);
  string_buffer_free(&sb);
}
