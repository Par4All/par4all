#include <stdlib.h>
#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"
#include "makefile.h"
#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "syntax.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "pipsmake.h"
#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "transformations.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"

static int pointer_index = 1;

/* --------------------------------Interprocedural Analysis-----------------------*/
/*This package computes the points-to interprocedurally.*/
void points_to_forward_translation()
{

}

void points_to_backward_translation()
{
}

/* We want a recursive descent on the type of the formal parameter,
 * once we found a pointer type we beguin a recursive descent until
 * founding a basic case. Then we beguin the ascent and the creation
 * gradually of the points_to_stub by calling pointer_formal_parameter_to_stub_points_to() */
set formal_points_to_parameter(cell c)
{
	
	reference r = reference_undefined;
	entity e = entity_undefined;
	type fpt = type_undefined;
	set pt_in = set_generic_make(set_private,
				     points_to_equal_p,
				     points_to_rank);
	r = cell_reference(c);
	e = reference_variable(r);
	fpt = ultimate_type(entity_type(e));
 	if(type_variable_p(fpt)){
		/* We ignor dimensions for the being, descriptors are not
		 * implemented yet...Amira Mensi*/
		basic fpb = variable_basic(type_variable(fpt));
		switch(basic_tag(fpb)){
		case is_basic_int:
			break;
		case is_basic_float:
			break;
		case is_basic_logical:
			break;
		case is_basic_overloaded:
			break;
		case is_basic_complex:
			break;
		case is_basic_pointer:{
		  pt_in = set_union(pt_in, pt_in, pointer_formal_parameter_to_stub_points_to(fpt,copy_cell(c)));
		  /* what about storage*/
		  break;
		}
		case is_basic_derived:
			break;
		case is_basic_string:
			break;
		case is_basic_typedef:
			break;
		case is_basic_bit:
			break;
		default: pips_error("basic_equal_p", "unexpected tag %d\n", basic_tag(fpb));
		}
	}
	return pt_in;
}


/* To create the points-to stub assiciated to the formal parameter,
 * the sink name is a concatenation of the formal parmater and the POINTS_TO_MODULE_NAME */
points_to create_stub_points_to(cell c, type t)
{
	cell sink = cell_undefined;
	reference r = reference_undefined;
	entity e = entity_undefined;
	reference sink_ref = reference_undefined;
	approximation rel = approximation_undefined;
	points_to pt_to = points_to_undefined;
	r = cell_reference(copy_cell(c));
	e = reference_variable(r);
	string s = strdup(concatenate("_", entity_user_name(e),"_", i2a(pointer_index), NULL));
	string formal_name = strdup(concatenate(POINTS_TO_MODULE_NAME,MODULE_SEP_STRING, s, NULL));
	entity formal_parameter = gen_find_entity(formal_name);
	if(entity_undefined_p(formal_parameter)) {
	  formal_parameter = make_entity(formal_name,
					 t, make_storage_rom(), make_value_unknown());
	}
	sink_ref = copy_reference(make_reference(formal_parameter, NIL));
	sink = make_cell_reference(sink_ref);
	rel = make_approximation_exact();
	pt_to = make_points_to(copy_cell(c), sink, rel,
			       make_descriptor_none());
	pointer_index ++;
	return pt_to;
}


/* Input : a formal parameter which is a pointer and its type.
   output : a set of points-to where sinks are stub points-to.
   we descent recursively until reaching a basic type, then we call
   create_stub_points_to()to generate the adequate points-to.
*/
set  pointer_formal_parameter_to_stub_points_to(type pt, cell c)
{
  reference r = reference_undefined;
  entity e = entity_undefined;
  points_to pt_to = points_to_undefined;
  set pt_in = set_generic_make(set_private,
			       points_to_equal_p,
			       points_to_rank);
  /* maybe should be removed if we have already called ultimate type
   * in formal_points_to_parameter() */

  type upt = type_to_pointed_type(pt);
  r = cell_reference(copy_cell(c));
  e = reference_variable(r);

  if(type_variable_p(upt)){
    if(array_entity_p(e)){
      /* We ignor dimensions for the being, descriptors are not
       * implemented yet...Amira Mensi*/
      ;
      /* ultimate_type() returns a wrong type for arrays. For
       * example for type int*[10] it returns int*[10] instead of int[10]. */
    }
    else {
      basic fpb = variable_basic(type_variable(upt));
      switch(basic_tag(fpb)){
      case is_basic_int:{
	pt_to = create_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_float:{
	pt_to = create_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_logical:
	break;
      case is_basic_overloaded:
	break;
      case is_basic_complex:
	break;
      case is_basic_pointer:{
	//pt = type_to_pointed_type(pt);
	pt_to = create_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	cell sink = points_to_sink(pt_to);
	pt_in = set_union(pt_in, pt_in,pointer_formal_parameter_to_stub_points_to(upt, sink));
	/* what about storage*/
	break;
      }
      case is_basic_derived:
	break;
      case is_basic_bit:
	break;
      case is_basic_string:{
	pt_to = create_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_typedef:{
	pt_to = create_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      default: pips_error("basic_equal_p", "unexpected tag %d\n", basic_tag(fpb));
      }
    }
  }
  else if(type_functional_p(upt))
    ;/*we don't know how to handle pointers to functions: nothing to
       be done for points-to analysis. */
  else if(type_void_p(upt)) {
    /* Create a target of unknown type */
    pt_to = create_stub_points_to(c, make_type_unknown() /*memory leak?*/);
    pt_in = set_add_element(pt_in, pt_in, (void*) pt_to );
  }
  else
    //we don't know how to handle other types
    pips_internal_error("Unexpected type\n");

  return pt_in;


}


bool intraprocedural_summary_points_to_analysis(char * module_name)
{
  entity module;
  type t;
  statement module_stat;
  list pt_list = NIL;
  list dcl = NIL;
  list params = NIL;
  set pts_to_set = set_generic_make(set_private,
				    points_to_equal_p,points_to_rank);

  set_current_module_entity(module_name_to_entity(module_name));
  module = get_current_module_entity();

  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE,
						       module_name, TRUE));
  module_stat = get_current_module_statement();
  t = entity_type(module);

  debug_on("POINTS_TO_DEBUG_LEVEL");

  pips_debug(1, "considering module %s\n", module_name);

  /* Properties */
  if(get_bool_property("ALIASING_ACROSS_FORMAL_PARAMETERS"))
    pips_user_warning("Property ALIASING_ACROSS_FORMAL_PARAMETERS"
		      " is ignored\n");
  if(get_bool_property("ALIASING_ACROSS_TYPES"))
    pips_user_warning("Property ALIASING_ACROSS_TYPES"
		      " is ignored\n");
  if(get_bool_property("ALIASING_INSIDE_DATA_STRUCTURE"))
    pips_user_warning("Property ALIASING_INSIDE_DATA_STRUCTURE"
		      " is ignored\n");

  if(type_functional_p(t)){
    functional f = type_functional(t);
    params = functional_parameters(f);
    FOREACH(PARAMETER, p, params){
		dummy d = parameter_dummy(p);
		if(dummy_identifier_p(d)){
			entity e = dummy_identifier(d);
			reference r = make_reference(e, NIL);
			cell c = make_cell_reference(r);
			pts_to_set = set_union(pts_to_set, pts_to_set,formal_points_to_parameter(c));
		}
	}
  }
  else
    pips_user_warning("The module %s is not a function\n", module_name);

  pt_list = set_to_sorted_list(pts_to_set,
			       (int(*)
				(const void*,const void*))
			       compare_points_to_location);
  points_to_list summary_pts_to_list = make_points_to_list(pt_list);
  points_to_list_consistent_p(summary_pts_to_list);
  DB_PUT_MEMORY_RESOURCE
    (DBR_SUMMARY_POINTS_TO_LIST, module_name, summary_pts_to_list);

  reset_current_module_entity();
  reset_current_module_statement();

  debug_off();

  bool good_result_p = TRUE;
  return (good_result_p);
}
