/* package abstract location. Amira Mensi 2010
 *
 * File: abstract_location.c
 *
 *
 * This file contains various useful functions to modelize a heap,
 * some of which/most should be moved elsewhere, probably in ri-util.
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
  string st, s;
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

     No difference here. The diffferent behavior will show when
     points-to and effects are translated at call sites
  */
  else if(strcmp(opt, "flow-sensitive")==0
	  || strcmp(opt, "context-sensitive")==0 ){
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
  else {
    pips_user_error("Unrecognized value for property ABSTRACT_HEAP_LOCATION:"
		    " \"%s\"", opt);
  }

  pips_debug(8, "Reference to ");

  return r;
}
