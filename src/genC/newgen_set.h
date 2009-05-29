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

typedef enum {
    set_string = hash_string ,
    set_int = hash_int,
    set_pointer = hash_pointer
} set_type ;

typedef struct {
    hash_table table ;
    set_type type ;
} set_chunk, *set ;

#define set_undefined ((set)(-16))
#define set_undefined_p(s) ((s)==set_undefined)

#define SET_MAP(element,code,set) \
    { HASH_MAP(_set_map_key, element, code, (set)->table); }

/* functions declared in set.c */

extern set set_add_element GEN_PROTO (( set, set, void *)) ;
extern set set_del_element GEN_PROTO(( set, set, void *)) ;
extern set set_assign GEN_PROTO(( set, set )) ;
extern bool set_belong_p GEN_PROTO(( set, void *)) ;
extern void set_clear GEN_PROTO(( set )) ;
extern set set_difference GEN_PROTO(( set, set, set )) ;
extern bool set_equal GEN_PROTO(( set, set )) ;
extern void set_free GEN_PROTO(( set )) ;
extern set set_intersection GEN_PROTO(( set, set, set )) ;
extern set set_make GEN_PROTO(( set_type )) ;
extern set set_singleton GEN_PROTO(( set_type, void * )) ;
extern set set_union GEN_PROTO(( set, set, set )) ;
extern bool set_empty_p GEN_PROTO(( set )) ;
extern void gen_set_closure GEN_PROTO(( void(*)(void*,set), set )) ;
extern void gen_set_closure_iterate GEN_PROTO((void(*)(void*,set), set, bool));
extern int set_own_allocated_memory GEN_PROTO((set));

#endif
