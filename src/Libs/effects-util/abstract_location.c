/*

  $Id: eval.c 17426 2010-06-25 09:24:14Z creusillet $

  Copyright 1989-2010 MINES ParisTech
  Copyright 2009-2010 HPC Project

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
/* package abstract location. Amira Mensi 2010
 *
 * File: abstract_location.c
 *
 *
 * This file contains various useful functions to modelize a heap.
 *
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "points_to_private.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text.h"
#include "text-util.h"
#include "misc.h"
#include "properties.h"


/*
  Heap Modelization

  See pipsmake-rc.tex, propertiews ABSTRACT_HEAP_LOCATIONS and
  ALIASING_ACROSS_TYPES, for documentation.
*/

/* check if an entity b may represent a bucket or a set of buckets in
   the heap. */
bool entity_heap_location_p(entity b)
{
  bool bucket_p = entity_all_heap_locations_p(b) ||
    entity_all_module_heap_locations_p(b);

  if(!bucket_p) {
  }

  return bucket_p;
}

entity entity_flow_or_context_sentitive_heap_location(int stmt_number, type t)
{
  entity e;
  string m = i2a(stmt_number);
  string s = strdup(concatenate(get_current_module_name(),
				MODULE_SEP_STRING,
				HEAP_AREA_LOCAL_NAME,
				"_l_",
				m,
				NULL));

  e = find_or_create_entity(s);
  if(type_undefined_p(entity_type(e))) {
    entity f = get_current_module_entity();
    entity a = module_to_heap_area(f);
    ram r = make_ram(f, a, UNKNOWN_RAM_OFFSET, NIL);

    /* FI: Beware, the symbol table is updated but this is not
       reflected in pipsmake.rc */
    entity_type(e) = t;
    entity_storage(e) = make_storage_ram(r);
    entity_initial(e) = make_value_unknown();
    (void) add_C_variable_to_area(a, e);
      
      
  }
  else {
    /* We might be in trouble, unless a piece of code is
       reanalyzed. Let's assume the type is unchanged */
    pips_assert("The type is unchanged",
		type_equal_p(t, entity_type(e)));
  }
  return e;

}

bool entity_flow_or_context_sentitive_heap_location_p(entity e)
{
  bool result = false;
  string ln = entity_local_name(e);
  string found = strstr(ln, ANYWHERE_LOCATION);

  pips_debug(9, "input entity: %s\n", ln);
  pips_debug(9, "found (1) = : %s\n", found);

  if (found == NULL) 
    {
      found = strstr(ln, HEAP_AREA_LOCAL_NAME);
      pips_debug(9, "found (2) = : %s\n", found);
      if (found!=NULL)
	{
	  size_t found_n = strspn(found, HEAP_AREA_LOCAL_NAME);
	  ln = &found[found_n];
	  pips_debug(0, "ln : %s\n", ln);
	  found = strstr(ln, "_l_");
	  pips_debug(0, "found (3) = : %s\n", found);
	  result = (found != NULL);
	}
      else
	result = false;
    }
  else
    result = false;
  pips_debug(9, "result = %d\n", (int) result);
  return result;
}

/* to handle malloc instruction : type t = (cast)
 * malloc(sizeof(expression). This function return a reference
 * according to the value of the property ABSTRACT_HEAP_LOCATIONS */
/* the list of callers will be added to ensure the context sensitive
   property. We should keep in mind that context and sensitive
   properties are orthogonal and we should modify them in pipsmake.*/
reference malloc_to_abstract_location(reference lhs,
				      type var_t,
				      type cast_t,
				      expression sizeof_exp,
				      entity f,
				      int stmt_number)
{
  reference r = reference_undefined;
  entity e = entity_undefined;
  //string st, s;
  string opt = get_string_property("ABSTRACT_HEAP_LOCATIONS");
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");

  /* in case we want an anywhere abstract heap location : the property
     ABSTRACT_HEAP_LOCATIONS is set to "unique" and a unique abstract
     location is used for all heap buckets. */
  if(strcmp(opt, "unique")==0){
    if(type_sensitive_p) {
      e = entity_all_heap_locations_typed(var_t);
      r = make_reference(e , NIL);
    }
    else {
      e = entity_all_heap_locations();
      r = make_reference(e , NIL);
    }
  }

  /* in case the property ABSTRACT_HEAP_LOCATIONS is set to
     "insensitive": an abstract location is used for each function. */
  else if(strcmp(opt, "insensitive")==0){
    if(type_sensitive_p) {
      e = entity_all_module_heap_locations_typed(get_current_module_entity(),
						 var_t);
      r = make_reference(e , NIL);
    }
    else {
      e = entity_all_module_heap_locations(f);
      r = make_reference(e , NIL);
    }
  }

  /* in case the property ABSTRACT_HEAP_LOCATIONS is set to
     "flow-sensitive" or "context-sensitive".

     No difference here between the two values. The diffferent
     behavior will show when points-to and effects are translated at
     call sites

     At this level, we want to use the statement number or the
     statement ordering. The statement ordering is safer because two
     statements or more can be put on the same line. The statement
     number is more user-friendly.

     There is no need to distinguish between types since the statement
     ordering is at least as effective.
  */
  else if(strcmp(opt, "flow-sensitive")==0
	  || strcmp(opt, "context-sensitive")==0 ){
    e = entity_flow_or_context_sentitive_heap_location(stmt_number, var_t);
    r = make_reference(e , NIL);
  }
  else {
    pips_user_error("Unrecognized value for property ABSTRACT_HEAP_LOCATION:"
		    " \"%s\"", opt);
  }

  pips_debug(8, "Reference to ");

  return r;
}
