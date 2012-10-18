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

#ifndef SET_INCLUDED
#define SET_INCLUDED
/*

   Pierre Jouvelot (3 Avril 1989)

   Set (of CHAR *) package interface.

   WARNING: You aren't allowed to use = or == between sets. Always use
   SET_ASSIGN and SET_EQUAL.
*/

#include "newgen_types.h"
#include "newgen_hash.h"

typedef struct _set_chunk * set;

/* Note: hash_chunk is not included in set_type */
typedef enum {
  set_string = hash_string ,
  set_int = hash_int,
  set_pointer = hash_pointer,
  set_private = hash_private
} set_type ;

#define set_undefined ((set)(-16))
#define set_undefined_p(s) ((s)==set_undefined)

// old compatibility, do not use
#define set_equal(s1,s2) set_equal_p(s1,s2)

#define SET_MAP(element,code,the_set)               \
  {                                                 \
    HASH_MAP(_set_map_key, element, code,           \
             set_private_get_hash_table(the_set));  \
  }

/**
 * enumerate set elements in their internal order.
 * caution, this enumeration is not deterministic!
 *
 * @param var_type is the plain type name (*not* capitalized).
 * @param var variable name, unique in scope
 * @param the_set expression that lead to a set, for instance a variable
 *
 * SET_FOREACH(var_type, var, the_set) {
 *   instructions;
 * }
 *
 * note that due to variables which are declared in the current scope:
 * - the "var" name must be unique in the scope, and is used as a
 *   suffix for declaring temporaries.
 * - put braces around the macro when using it as a loop body or
 *   condition case.
 *
 * Ronan, I wish to avoid an ugly double macro expansion hack here.
 * Just change the scalar variable name "var" if need be.
 */
#define SET_FOREACH(type_name, the_item, the_set) \
  hash_table _hash_##the_item =                   \
    set_private_get_hash_table(the_set);          \
  void * _point_##the_item = NULL;                \
  type_name the_item;                             \
  for (; (_point_##the_item =                     \
          hash_table_scan(_hash_##the_item,				\
                          _point_##the_item,      \
                          (void **) &the_item,    \
                          NULL));)

/* what about this replacement?
#define SET_MAP(the_item, the_code, the_set)		\
  { SET_FOREACH(void *, the_item, the_set) the_code; }
*/

/* functions implemented in set.c */
// CONSTRUCTORS
extern set set_generic_make(set_type, hash_equals_t, hash_rank_t);
extern set set_make(set_type);
extern set set_singleton(set_type, const void *);
extern set set_dup(const set);
// DESTRUCTORA
extern void set_free(set);
extern void sets_free(set,...);
// OBSERVERS
extern int set_size(const set);
extern int set_own_allocated_memory(const set);
extern set_type set_get_type(const set);
// do not call this one, please...
extern hash_table set_private_get_hash_table(const set);
// TESTS
extern bool set_belong_p(const set, const void *);
extern bool list_in_set_p(const list, const set);
extern bool set_equal_p(const set, const set);
extern bool set_empty_p(const set);
extern bool set_inclusion_p(const set, const set);
extern bool set_intersection_p(const set, const set);
// OPERATIONS
extern set set_clear(set);
extern set set_assign(set, const set);
extern set set_append_list(set, const list);
extern set set_assign_list(set, const list);
extern set set_add_element(set, const set, const void *);
extern set set_add_elements(set, const set, const void * e, ...);
extern set set_union(set, const set, const set);
extern set set_intersection(set, const set, const set);
extern set set_difference(set, const set, const set);
extern set set_del_element(set, const set, const void *);
extern set set_delfree_element(set, const set, const void *);
extern void gen_set_closure_iterate(void (*)(void *, set), set, bool);
extern void gen_set_closure(void (*)(void *, set), set);
// CONVERSIONS
extern string set_to_string(string, const set, gen_string_func_t);
extern void set_fprint(FILE *, string, const set, gen_string_func_t);
extern list set_to_sorted_list(const set, gen_cmp_func_t);
// no not use set_to_list, the output is not deterministic
extern list set_to_list(const set);
// See set_append_list for the conversion from list to set

#endif // SET_INCLUDED
