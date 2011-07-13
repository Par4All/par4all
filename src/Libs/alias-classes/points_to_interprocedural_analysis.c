#include <stdlib.h>
#include <stdio.h>
/* For strdup: */
#define _GNU_SOURCE
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "parser_private.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
#include "pipsdbm.h"
#include "resources.h"
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
	fpt = basic_concrete_type(entity_type(e));
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
		case is_basic_derived:{
		  pt_in = set_union(pt_in, pt_in, derived_formal_parameter_to_stub_points_to(fpt,copy_cell(c)));
			break;
		}
		case is_basic_string:
			break;
		case is_basic_typedef:{
		  pt_in = set_union(pt_in, pt_in, typedef_formal_parameter_to_stub_points_to(fpt,copy_cell(c)));
			break;
		}
		case is_basic_bit:
			break;
		default: pips_internal_error("unexpected tag %d", basic_tag(fpb));
		}
	}
	return pt_in;
}


/* To create the points-to stub assiciated to the formal parameter,
 * the sink name is a concatenation of the formal parmater and the POINTS_TO_MODULE_NAME */
points_to create_stub_points_to(cell c, type t,__attribute__ ((__unused__)) basic b)
{
  basic bb = basic_undefined;
  expression ex = make_unbounded_expression();
  reference r = cell_to_reference(copy_cell(c));
  entity e = reference_variable(r);
  string s = strdup(concatenate("_", entity_user_name(e),"_", i2a(pointer_index), NULL));
  string formal_name = strdup(concatenate(get_current_module_name() /* POINTS_TO_MODULE_NAME */ ,MODULE_SEP_STRING, s, NULL));
  entity formal_parameter = gen_find_entity(formal_name);
  type pt = type_to_pointed_type(t);
  if(type_variable_p(pt))
    bb = variable_basic(type_variable(pt));
  type tt = make_type_variable(
			       make_variable(copy_basic(bb),
					     CONS(DIMENSION,
						  make_dimension(int_to_expression(0),ex),NIL),
					     NIL));
  if(entity_undefined_p(formal_parameter)) {
    formal_parameter = make_entity(formal_name,
				   tt,
				   make_storage_formal(
						       make_formal(
								    get_current_module_entity() 
								   /* module_name_to_entity(POINTS_TO_MODULE_NAME) */,
								   pointer_index)),
				   make_value_unknown());
  }
  /* expression ex_sink =  entity_to_expression(formal_parameter); */
  /* set_methods_for_proper_simple_effects(); */
  /* effect ef = effect_undefined; */
  /* list l1 = generic_proper_effects_of_complex_address_expression(ex_sink, &ef, */
  /* 									 true); */
  reference sink_ref = make_reference(formal_parameter, CONS(EXPRESSION, int_to_expression(0), NIL));
  /* reference_indices_(sink_ref)= CONS(EXPRESSION, int_to_expression(0), NIL); */
  /* effects_free(l1); */
  /* generic_effects_reset_all_methods(); */
  cell sink_cell = make_cell_reference(sink_ref);
  approximation rel = make_approximation_exact();
  points_to pt_to = make_points_to(copy_cell(c), sink_cell, rel,
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
      basic fpb = variable_basic(type_variable(upt));
      if(basic_pointer_p(fpb)){
	expression ind = make_unbounded_expression();
	reference r = make_reference(e, CONS(EXPRESSION, ind, NULL));
	cell c = make_cell_reference(r);
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
    }
    }
    else {
      basic fpb = variable_basic(type_variable(upt));
      switch(basic_tag(fpb)){
      case is_basic_int:{
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_float:{
	pt_to = create_stub_points_to(c, upt, fpb);
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
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	cell sink = points_to_sink(pt_to);
	pt_in = set_union(pt_in, pt_in,pointer_formal_parameter_to_stub_points_to(upt, sink));
	/* what about storage*/
	break;
      }
      case is_basic_derived:{
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_bit:
	break;
      case is_basic_string:{
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_typedef:{
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      default: pips_internal_error("unexpected tag %d", basic_tag(fpb));
	break;
      }
    }
  }
  else if(type_functional_p(upt))
    ;/*we don't know how to handle pointers to functions: nothing to
       be done for points-to analysis. */
  else if(type_void_p(upt)) {
    /* Create a target of unknown type */
    pt_to = create_stub_points_to(c, make_type_unknown(), basic_undefined /*memory leak?*/);
    pt_in = set_add_element(pt_in, pt_in, (void*) pt_to );
  }
  else
    //we don't know how to handle other types
    pips_internal_error("Unexpected type");

  return pt_in;


}


/* Input : a formal parameter which is a derived.
   output : a set of points-to where sinks are stub points-to.
*/
set  derived_formal_parameter_to_stub_points_to(type pt, cell c)
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
      if( basic_derived_p(fpb)) {
	type t = entity_type(e);
	expression ex = entity_to_expression(e);
	if(type_variable_p(t)){
	  basic vb = variable_basic(type_variable(t));
	  if(basic_derived_p(vb)){
	    entity ed = basic_derived(vb);
	    type et = entity_type(ed);
	    if(type_struct_p(et)){
	      list l1 = type_struct(et);
	      FOREACH(ENTITY, i, l1){
	
		expression ef = entity_to_expression(i);
		if(expression_pointer_p(ef)){
		  type ent_type = entity_type(i);
		  fpb =  variable_basic(type_variable(ent_type));
		  expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
						  ex,
						  ef);
		  set_methods_for_proper_simple_effects();
		  effect ef = effect_undefined;
		  list l_ef = NIL;
		  list l1 = generic_proper_effects_of_complex_address_expression(ex1, &l_ef,
										 true);
		  ef = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
		  gen_free_list(l_ef); /* free the spine */

		  reference source_ref =  effect_any_reference(ef);
		  effects_free(l1);
		  generic_effects_reset_all_methods();
		  cell source_cell = make_cell_reference(source_ref);
		  pt_to = create_stub_points_to(source_cell, ent_type, fpb);
		  pt_in = set_add_element(pt_in, pt_in,
					  (void*) pt_to );
		}

	      }
	    }
	  }
	}
      }
    }
  }
	    
  return pt_in;


}

/* Input : a formal parameter which is a typedef.
 */
set  typedef_formal_parameter_to_stub_points_to(type pt, cell c)
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
      if(basic_typedef_p(fpb)){
	    entity e1  = basic_typedef(fpb);
	    type t1 = entity_type(e1);
	    if(entity_variable_p(e1)){
	      basic b2 =  variable_basic(type_variable(t1));
		if(basic_derived_p(b2)){
		  entity e2  = basic_derived(b2);
		  /* l = points_to_init_derived(e, e2); */
		  type t = entity_type(e2);
		  expression ex = entity_to_expression(e);
		  if(type_struct_p(t)){
		    list l1 = type_struct(t);
		    FOREACH(ENTITY, i, l1){
		      expression ef = entity_to_expression(i);
		      if(expression_pointer_p(ef)){
			type ent_type = entity_type(i);
			expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
						  ex,
						  ef);
			set_methods_for_proper_simple_effects();
			effect ef = effect_undefined;
			list l_ef = NIL;
			list l1 = generic_proper_effects_of_complex_address_expression(ex1, &l_ef,
										       true);
			ef = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
			gen_free_list(l_ef); /* free the spine */

			reference source_ref =  effect_any_reference(ef);
			effects_free(l1);
			generic_effects_reset_all_methods();
			cell source_cell = make_cell_reference(source_ref);
			pt_to = create_stub_points_to(source_cell, ent_type, fpb);
			pt_in = set_add_element(pt_in, pt_in,
						(void*) pt_to );
		      }
		    }
		  }
		}
	    }
      }
    }
  }

  return pt_in;
}

bool intraprocedural_summary_points_to_analysis(char * module_name)
{
  entity module;
  type t;
  //statement module_stat;
  list pt_list = NIL;
  //list dcl = NIL;
  //list params = NIL;
  set pts_to_set = set_generic_make(set_private,
				    points_to_equal_p,points_to_rank);

  set_current_module_entity(module_name_to_entity(module_name));
  module = get_current_module_entity();

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
    list dl = code_declarations(value_code(entity_initial(module)));

    FOREACH(ENTITY, fp, dl) {
      if(formal_parameter_p(fp)) {
	reference r = make_reference(fp, NIL);
	cell c = make_cell_reference(r);
	pts_to_set = set_union(pts_to_set, pts_to_set,
			       formal_points_to_parameter(c));
      }
    }
  }
  else
    pips_user_error("The module %s is not a function.\n", module_name);

  pt_list = set_to_sorted_list(pts_to_set,
			       (int(*)
				(const void*,const void*))
			       points_to_compare_cells);
  points_to_list summary_pts_to_list = make_points_to_list(pt_list);
  points_to_list_consistent_p(summary_pts_to_list);
  DB_PUT_MEMORY_RESOURCE
    (DBR_SUMMARY_POINTS_TO_LIST, module_name, summary_pts_to_list);

  reset_current_module_entity();
  debug_off();

  bool good_result_p = true;
  return (good_result_p);
}
