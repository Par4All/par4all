/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

#define SET_MAP(element,code,the_set)					\
  {									\
    HASH_MAP(_set_map_key, element, code,				\
	     set_private_get_hash_table(the_set));			\
  }

/* functions implemented in set.c */
// CONSTRUCTORS
extern set set_generic_make(set_type, hash_equals_t, hash_rank_t);
extern set set_make(set_type);
extern set set_singleton(set_type, void *);
extern set set_dup(set);
// DESTRUCTOR
extern void set_free(set);
// OBSERVERS
extern int set_size(set);
extern int set_own_allocated_memory(set);
extern set_type set_get_type(set);
// do not call this one, please...
extern hash_table set_private_get_hash_table(set);
// TESTS
extern bool set_belong_p(set s, void *e);
extern bool list_in_set_p(list, set);
extern bool set_equal_p(set, set);
extern bool set_empty_p(set);
extern bool set_inclusion_p(set, set);
// OPERATIONS
extern set set_clear(set);
extern set set_assign(set, set);
extern set set_append_list(set, list);
extern set set_assign_list(set, list);
extern set set_add_element(set, set, void *);
extern set set_union(set, set, set);
extern set set_intersection(set, set, set);
extern set set_difference(set, set, set);
extern set set_del_element(set, set, void *);
extern set set_delfree_element(set, set, void *);
extern void gen_set_closure_iterate(void (*)(void *, set), set, bool);
extern void gen_set_closure(void (*)(void *, set), set);
// conversions
extern list set_to_sorted_list(set, int (*)(const void *, const void *));
// no not use set_to_list, the output is not deterministic
extern list set_to_list(set);

#endif // SET_INCLUDED
