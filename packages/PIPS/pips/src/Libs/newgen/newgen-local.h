/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

/* mapping.h inclusion
 *
 * I do that because this file was previously included in genC.h,
 * but the macros defined here use ri types (statement, entity...).
 * three typedef are also included here. ri.h is a prerequisit for 
 * mapping.h.
 *
 * FC, Feb 21, 1994
 */

#define STATEMENT_ORDERING_UNDEFINED (-1)

/* these macros are obsolete! newgen functions (->) should be used
 * instead
 */
#ifndef STATEMENT_MAPPING_INCLUDED
#define STATEMENT_MAPPING_INCLUDED
typedef hash_table statement_mapping;
#define MAKE_STATEMENT_MAPPING() \
     (statement_mapping) hash_table_make(hash_pointer, 0)
#define FREE_STATEMENT_MAPPING(map) \
    (hash_table_free((hash_table) (map)))
#define SET_STATEMENT_MAPPING(map, stat, val) \
    hash_put((hash_table) (map), (char *)(stat), (char *)(val))
#define GET_STATEMENT_MAPPING(map, stat) \
    hash_get((hash_table) (map), (char *) (stat))
#define STATEMENT_MAPPING_COUNT(map) \
    hash_table_entry_count((hash_table) map)
#define STATEMENT_MAPPING_MAP(s, v, code, h) \
        HASH_MAP(s, v, code, h)
/*
 * this warrant BA Macro Compatibility:
 */
#define DEFINE_CURRENT_MAPPING(name, type) \
        GENERIC_CURRENT_MAPPING(name, type, statement)
#endif

#ifndef ENTITY_MAPPING_INCLUDED
#define ENTITY_MAPPING_INCLUDED
typedef hash_table entity_mapping;
#define MAKE_ENTITY_MAPPING() \
    ((entity_mapping) hash_table_make(hash_pointer, 0))
#define FREE_ENTITY_MAPPING(map) \
    (hash_table_free((hash_table) (map)))
#define SET_ENTITY_MAPPING(map, ent, val) \
    hash_put((hash_table) (map), (char *)(ent), (char *)(val))
#define GET_ENTITY_MAPPING(map, ent) \
    hash_get((hash_table) (map), (char *)(ent))
#define ENTITY_MAPPING_COUNT(map) \
    hash_table_entry_count((hash_table) map)
#define ENTITY_MAPPING_MAP(s, v, code, h) \
        HASH_MAP(s, v, code, h)
#endif

#ifndef CONTROL_MAPPING_INCLUDED
#define CONTROL_MAPPING_INCLUDED
typedef hash_table control_mapping;
#define MAKE_CONTROL_MAPPING() \
    ((control_mapping) hash_table_make(hash_pointer, 0))
#define FREE_CONTROL_MAPPING(map) \
    (hash_table_free((hash_table) (map)))
#define SET_CONTROL_MAPPING(map, cont, val) \
    hash_put((hash_table) (map), (char *)(cont), (char *)(val))
#define GET_CONTROL_MAPPING(map, cont) \
    hash_get((hash_table) (map), (char *)(cont))
#define CONTROL_MAPPING_COUNT(map) \
    hash_table_entry_count((hash_table) map)
#define CONTROL_MAPPING_MAP(s, v, code, h) \
        HASH_MAP(s, v, code, h)
#endif
