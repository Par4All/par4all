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
 * These are the functions defined in the Newgen mapping library.
 *
 * This is a temporary implementation used for the Pips Project. The
 * general notion of mapping (i.e., functions) is going to be implemented
 * shortly inside Newgen.
 *
 * This version uses pointers to statements as keys in hash_tables which
 * causes bugs when code in unloaded on and then reloaded from disk.
 * Francois Irigoin, 1 February 1993
 *
 * a useful macro which generates the declaration of a static variable of type
 * statement_mapping, and related functions :
 *       set_**_map
 *       load_statement_**_map
 *       store_statement_**_map
 *       reset_**_map
 *
 * BA, august 26, 1993
 *
 * this macro is redefined here as GENERIC_CURRENT_MAPPING, which uses
 * that type as a parameter, to allow other mappings than statement's ones.
 * I need entity_mapping, control_mapping
 * there is no way to define macros inside the macro, so I cannot
 * generate the usual macros:-(, The code defining them will have to be
 * replicated (done in mapping.h).
 *
 * FC, Feb 21, 1994
 */

#ifndef GENERIC_MAPPING_INCLUDED
#define GENERIC_MAPPING_INCLUDED

/* PIPS level:
 *
 * GENERIC_MAPPING(PREFIX, name, result, type)
 *
 * This macros are obsolete. Declare the mappings in newgen instead,
 * and use the generic functions!
 *
 * name: name of the mapping
 * result: type of the result
 * type: type of the mapping key
 * // CTYPE: type of the mapping key in capital letters
 *
 * a static variable 'name'_map of type 'type' is declared
 * and can be accessed thru the definefd functions.
 *
 * the mapping is   'name' = 'type' -> 'result'
 */
#define GENERIC_MAPPING(PREFIX, name, result, type)			\
static type##_mapping name##_map = hash_table_undefined;		\
PREFIX bool __attribute__ ((unused)) name##_map_undefined_p() {		\
  return name##_map == hash_table_undefined;				\
}									\
PREFIX void __attribute__ ((unused)) set_##name##_map(type##_mapping m) { \
  message_assert("mapping undefined", name##_map == hash_table_undefined); \
  name##_map = m;							\
}									\
PREFIX type##_mapping __attribute__ ((unused)) get_##name##_map() {	\
  return name##_map;							\
}									\
PREFIX void __attribute__ ((unused)) reset_##name##_map() {		\
  name##_map = hash_table_undefined;					\
}									\
PREFIX void __attribute__ ((unused)) free_##name##_map() {		\
  hash_table_free(name##_map); name##_map = hash_table_undefined;	\
}									\
PREFIX void __attribute__ ((unused)) make_##name##_map() {		\
  name##_map = hash_table_make(hash_pointer, HASH_DEFAULT_SIZE);	\
}									\
PREFIX result __attribute__ ((unused)) load_##type##_##name(type s) {	\
  result t;								\
  message_assert("key defined", s != type##_undefined);			\
  t = (result)(intptr_t) hash_get((hash_table) (name##_map), (void*) (s));	\
  if (t ==(result)(intptr_t) HASH_UNDEFINED_VALUE)				\
    t = result##_undefined;						\
  return t;								\
}									\
PREFIX void __attribute__ ((unused)) delete_##type##_##name(type s) {	\
  message_assert("key defined", s != type##_undefined);			\
  (void) hash_del((hash_table) (name##_map), (void*) (s));		\
}									\
PREFIX bool __attribute__ ((unused)) type##_##name##_undefined_p(type s) { \
  return(load_##type##_##name(s)==result##_undefined);			\
}									\
PREFIX void __attribute__ ((unused)) store_##type##_##name(type s, result t) { \
  message_assert("key defined", s != type##_undefined);			\
  message_assert("value defined", t != result##_undefined);		\
  hash_put((hash_table) name##_map, (void*) s, (void*)(intptr_t) t);		\
}									\
PREFIX void __attribute__ ((unused)) update_##type##_##name(type s, result t) {	\
  message_assert("key defined", s != type##_undefined);			\
  message_assert("value defined", t != result##_undefined);		\
  hash_update((hash_table) name##_map, (void*) s, (void*)(intptr_t) t);		\
}

#define GENERIC_CURRENT_MAPPING(name, result, type) \
        GENERIC_MAPPING(extern, name, result, type)

/*  to allow mappings local to a file.
 *  it seems not to make sense, but I like the interface.
 *  FC 27/12/94
 */
#define GENERIC_LOCAL_MAPPING(name, result, type) \
        GENERIC_MAPPING(static, name, result, type)

/* end GENERIC_MAPPING_INCLUDED */
#endif
