/* package abstract location. Amira Mensi 2010
 *
 * File: abstract_location.c
 *
 *
 * This file contains various useful functions, some of which should be moved
 * elsewhere, maybe in ri-util.
 *
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "text-util.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "properties.h"
#include "preprocessor.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "alias-classes.h"

/* to handle malloc instruction : type t = (cast)
 * malloc(sizeof(expression). This function return a reference
 * according to the value of the property ABSTRACT_HEAP_LOCATION */
/* the list of callers will be added to ensure the context sensitive
   property. We should keep in mind that context and sensitive
   properties are orthogonal and we should modify them in pipsmake.*/
reference malloc_to_abstract_location(reference lhs,
				      type var_t,
				      type cast_t,
				      expression sizeof_exp,
				      entity module_name,
				      int stmt_number)
{
  reference r = reference_undefined;
  entity e = entity_undefined;
  string st, s;

  /* in case we want an anywhere abstract heap location : the property
     ABSTRACT_HEAP_LOCATIONS is set to "unique"*/
  if(strcmp(get_string_property("ABSTRACT_HEAP_LOCATIONS"),"unique")==0){
    e = entity_all_locations();
    r = make_reference(e , NIL);
  }

  /* in case the property ABSTRACT_HEAP_LOCATIONS is set to "insensitive"*/
  else if(strcmp(get_string_property("ABSTRACT_HEAP_LOCATIONS"),"insensitive")==0){
    e = entity_all_module_heap_locations(get_current_module_entity());
    r = make_reference(e , NIL);
  }

  /* in case the property ABSTRACT_HEAP_LOCATIONS is set to "flow-sensitive"*/
  else if(strcmp(get_string_property("ABSTRACT_HEAP_LOCATIONS"),"flow-sensitive")==0){
    e = entity_all_module_heap_locations(get_current_module_entity());
    st = itoa(stmt_number);
    s = strdup(concatenate(entity_name(e),"[", st, "]",NIL));
    entity ee = find_or_create_entity(s);
    if(entity_undefined_p(ee)) {
      area a = make_area(0,NIL); /* Size and layout are unknown */
      type t = make_type_area(a);
      ee = make_entity(s,
		       t, make_storage_rom(), make_value_unknown());
    }
    r = make_reference(ee , NIL);
  }

  /* in case the property ABSTRACT_HEAP_LOCATIONS is set to
     "context-sensitive"*/
  else if(strcmp(get_string_property("ABSTRACT_HEAP_LOCATIONS"),
	    "context-sensitive")==0){

    e = entity_all_module_heap_locations(get_current_module_entity());
    st = itoa(stmt_number);
    s = strdup(concatenate(entity_name(e),"[", st,"]",NIL));
    entity ee = find_or_create_entity(strdup(s));
    if(entity_undefined_p(ee)) {
      area a = make_area(0,NIL); /* Size and layout are unknown */
      type t = make_type_area(a);
      ee = make_entity(strdup(s),
		       t, make_storage_rom(), make_value_unknown());
    }
    r = make_reference(ee , NIL);
  }

  return r;
}
