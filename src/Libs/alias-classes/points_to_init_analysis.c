#include <stdlib.h>
#include <stdio.h>
/* For strdup: */
// Already defined elsewhere
//#define _GNU_SOURCE
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

/* --------------------------------Interprocedural Points-to Analysis-----------------------*/
/* This package computes the points-to interprocedurally.
 *
 * See Chapter ? in Amira Mensi's PhD dissertation.
 */

void points_to_forward_translation()
{

}

void points_to_backward_translation()
{
}
/* We want a recursive descent on the type of the formal parameter,
 * once we found a pointer type we beguin a recursive descent until
 * founding a basic case. Then we beguin the ascent and the creation
 * gradually of the points_to_stub by calling
 * pointer_formal_parameter_to_stub_points_to().
 *
 * FI->AM: as I would rather work on-demand, this function should be
 * useless. I fixed it nevertheless because it seems better for
 * EffectsWithPointsTo, which does not seem to allocate the new
 * points-to stubs it needs.
 */
set formal_points_to_parameter(cell c)
{
  reference r = reference_undefined;
  type fpt = type_undefined;
  set pt_in = set_generic_make(set_private,
			       points_to_equal_p,
			       points_to_rank);
 
  r = cell_to_reference(c);
  bool to_be_freed = false;
  /* fpt = entity_basic_concrete_type(e); */
  fpt = cell_reference_to_type(r,&to_be_freed);
  if(type_variable_p(fpt)){
    /* We ignor dimensions for the time being, descriptors are not
     * implemented yet...Amira Mensi*/
    basic fpb = variable_basic(type_variable(fpt));
    if(array_type_p(fpt)) {
      pt_in = set_union(pt_in, pt_in, array_formal_parameter_to_stub_points_to(fpt,c));
    }
    else {
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
	pt_in = set_union(pt_in, pt_in, pointer_formal_parameter_to_stub_points_to(fpt,c));
	/* what about storage*/
	break;
      }
      case is_basic_derived:{
	pt_in = set_union(pt_in, pt_in, derived_formal_parameter_to_stub_points_to(fpt,c));
	break;
      }
      case is_basic_string:
	break;
      case is_basic_typedef:{
	pt_in = set_union(pt_in, pt_in, typedef_formal_parameter_to_stub_points_to(fpt,c));
	break;
      }
      case is_basic_bit:
	break;
      default: pips_internal_error("unexpected tag %d", basic_tag(fpb));
      }
    }
  }
  if (to_be_freed) free_type(fpt);
  return pt_in;

}

/* Allocate a new stub entity for entity "e" and with type "t".
 *
 * Type "t" could be derived from "e" since it should be the pointed
 * type of "e"'s type
 */
entity create_stub_entity(entity e, type t)
{
  // local name for the stub
  string s = string_undefined;
  string en = (string) entity_user_name(e);

  // FI: guarantee about *local* new name uniqueness?
  if(formal_parameter_p(e)) {
    // Naming for sinks of formal parameters: use their offsets
    formal f = storage_formal( entity_storage(e) );
    int off = formal_offset(f);
    s = strdup(concatenate("_", en,"_", i2a(off), NULL));
  }
  else if(top_level_entity_p(e)){ // FI: global_variable_p()
    // Naming for sinks of global variable: use their offsets
    int off = ram_offset(storage_ram(entity_storage(e)));
    s = strdup(concatenate("_", en,"_", i2a(off), NULL));
  }
  else if(static_global_variable_p(e)){ // "static int i;"
    // Naming for sinks of static global variable: use their offsets
    int off = ram_offset(storage_ram(entity_storage(e)));
    s = strdup(concatenate("_", en,"_", i2a(off), NULL));
  }
  else if(entity_stub_sink_p(e)) {
    // Naming for sinks of stubs: repeat their last suffix
    char *suffix = strrchr(en,'_');
    s = strdup(concatenate( en, suffix, NULL )); 
  }
  
  // FI: the stub entity already exists?
  string formal_name = strdup(concatenate(get_current_module_name(),
					  MODULE_SEP_STRING, s, NULL));
  entity stub = gen_find_entity(formal_name);
  // FI: I expect here a pips_assert("The stub cannot exist",
  // entity_undefined_p(stub));

  // Compute the pointed type
  type pt = type_undefined;
  if(type_variable_p(t)){
    basic bb = variable_basic(type_variable(t));
    basic base = copy_basic(bb);
    bool type_strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
    // FI: we have a probleme here when type_strict_p is set to false
    //
    // We want pointers to element to be promoted to pointers to
    // arrays, for instance, "int * p;" is compatible with "p++;" in
    // spite of the standard.
    //
    // However, we cannot do that with struct and union because the
    // fields are replaced by subscript.
    //
    // To sum up, things are better if the typing is
    // strict... although most C programs are not strictly typed.
    if(!type_strict_p && !struct_type_p(t)) {
      expression ex = make_unbounded_expression();
      dimension d = make_dimension(int_to_expression(0),ex);
      variable v = make_variable(base,
				 CONS(DIMENSION, d ,NIL),
				 NIL);
      pt = make_type_variable(v);
    }
    else
      pt = copy_type(t);
  } 
  else if (type_void_p(t)){
    pt = make_type_void(NIL);
  }
 
  // If entity "stub" does not already exist, create it.
  if(entity_undefined_p(stub)) {
    entity DummyTarget = FindOrCreateEntity(POINTER_DUMMY_TARGETS_AREA_LOCAL_NAME,
					    POINTER_DUMMY_TARGETS_AREA_LOCAL_NAME);
    entity_kind(DummyTarget) = ENTITY_POINTER_DUMMY_TARGETS_AREA;
    stub = make_entity(formal_name,
		       pt,
		       make_storage_ram(make_ram(get_current_module_entity(),DummyTarget, UNKNOWN_RAM_OFFSET, NIL)),
		       make_value_unknown());
  
  }

  return stub;
}


/* To create the points-to between a formal parameter or a global
 * variable or another stub on one hand, and another new stub on the
 * other.
 *
 * the sink name is a concatenation of the formal parameter and the
 * POINTS_TO_MODULE_NAME.
 *
 * FI: Is it sufficient to generate stubs for foo(int *p) and bar(double *p)?
 *
 * FI: no, it is useless, since they are created on demand; but this
 * function is reused on demand.
 */
points_to create_stub_points_to(cell c, type t,
				__attribute__ ((__unused__)) basic b)
{
  points_to pt_to = points_to_undefined;
  //basic bb = basic_undefined;
  //type pt = type_undefined;
  //expression ex = make_unbounded_expression();
  reference sink_ref = reference_undefined;
  cell source_cell = copy_cell(c);
  reference r = cell_any_reference(source_cell);
  entity e = reference_variable(r);

  //const char * en = entity_user_name(e); 
  //string s = NULL;

  entity formal_parameter = create_stub_entity(e, t);

  bool type_strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
  if(!type_strict_p && !derived_type_p(t))
    sink_ref = make_reference(formal_parameter,
			      CONS(EXPRESSION, int_to_expression(0), NIL));
  else
    sink_ref = make_reference(formal_parameter,  NIL);

  cell sink_cell = make_cell_reference(sink_ref);

  /* FI: if we single out the NULL pointer value, this has to be a may
   * points-to. Another discussion about stubs is based on the fact
   * that they may represent several cells at the call site, although
   * they are only one cell at run-time and for any specific execution
   * of the called function.
   *
   * Singling out the NULL pointers is useful to exploit conditions in
   * tests and while loops. It may also lead to more precise
   * fix-points for the points-to graph.
  */
  // approximation rel = make_approximation_exact();
  approximation rel = make_approximation_may();

  pt_to = make_points_to(source_cell, sink_cell, rel,
			 make_descriptor_none());
  pointer_index ++;
  
  return pt_to;
}

/* To create the points-to stub associated to the formal parameter,
 * the sink name is a concatenation of the formal parmater and the
 * POINTS_TO_MODULE_NAME.
 */
points_to create_pointer_to_array_stub_points_to(cell c, type t,__attribute__ ((__unused__)) basic b)
{ 
  list l_ind = NIL;
  basic bb = basic_undefined;
  expression ex = make_unbounded_expression();
  expression l = expression_undefined;
  expression u = expression_undefined;
  reference sink_ref = reference_undefined;
  cell source_cell = copy_cell(c);
  reference r = cell_any_reference(source_cell);
  entity e = reference_variable(r);
  const char * en = entity_user_name(e); 
  string s = NULL;
  if( formal_parameter_p(e) ) {
    formal f = storage_formal( entity_storage(e) );
    int off = formal_offset(f);
    s = strdup(concatenate("_", en,"_", i2a(off), NULL));
  }
  else {
    char *suffix = strrchr(en,'_');
    s = strdup(concatenate( en, suffix, NULL )); 
  }
  
  string formal_name = strdup(concatenate(get_current_module_name() ,MODULE_SEP_STRING, s, NULL));
  entity formal_parameter = gen_find_entity(formal_name);
  type pt = type_undefined;
  bool type_strict_p = !get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
  bb = variable_basic(type_variable(t));
  list l_dim = variable_dimensions(type_variable(t));
  basic base = copy_basic(bb);
  FOREACH(DIMENSION, d, l_dim){
    l = dimension_lower(d);
    u = dimension_upper(d);
    l_ind = CONS(EXPRESSION, l, NIL);
    l_ind = gen_nconc(l_ind, (CONS(EXPRESSION, u, NIL)));
  }
  
 
  if(type_strict_p)
    pt = make_type_variable(
			    make_variable(base,
					  CONS(DIMENSION,
					       make_dimension(int_to_expression(0),ex),NIL),
					  NIL));
  else
    pt = copy_type(t);    

  if(entity_undefined_p(formal_parameter)) {
    formal_parameter = make_entity(formal_name,
				   pt,
				   make_storage_rom(),
				   make_value_unknown());
  }

  if(type_strict_p)
    sink_ref = make_reference(formal_parameter, CONS(EXPRESSION, int_to_expression(0), NIL));
  else if((int)gen_length(l_dim)>1){
    sink_ref = make_reference(formal_parameter,l_ind);
  }
  else
    sink_ref = make_reference(formal_parameter, CONS(EXPRESSION, int_to_expression(0), NIL));

  cell sink_cell = make_cell_reference(sink_ref);
  approximation rel = make_approximation_exact();
  points_to pt_to = make_points_to(source_cell, sink_cell, rel,
				   make_descriptor_none());
  pointer_index ++;
  return pt_to;
}


/* Input : a formal parameter which is a pointer and its type.
 *
 * Output : a set of points-to where sinks are stub points-to.
 * we descent recursively until reaching a basic type, then we call
 *  create_stub_points_to()to generate the adequate points-to.
 *
 * FI: I do not know if I want to keep using this function because
 * stubs are not created on demand and because some are certainly not
 * useful for the points-to analysis. But they may be useful for
 * client analysis... However, client analyses will have to create
 * more such stubs...
 */
set pointer_formal_parameter_to_stub_points_to(type pt, cell c)
{
  points_to pt_to = points_to_undefined;
  set pt_in = set_generic_make(set_private,
			       points_to_equal_p,
			       points_to_rank);
  /* maybe should be removed if we have already called ultimate type
   * in formal_points_to_parameter() */

  /* The pointer may be NULL or undefined. We neglect undefined/nowhere */
  cell nc = copy_cell(c);
  cell null_c = make_null_pointer_value_cell();
  points_to npt = make_points_to(nc, null_c,
				 make_approximation_may(),
				 make_descriptor_none());
  pt_in = add_arc_to_pt_map(npt, pt_in);

  /* The pointer may points towards another object (or set of object) */
  type upt = type_to_pointed_type(pt);
  if( type_variable_p(upt) ){
    basic fpb = variable_basic(type_variable(upt));
    if( array_type_p(upt) ){
      pt_to = create_pointer_to_array_stub_points_to(c, upt , fpb);
      pt_in = set_add_element(pt_in, pt_in,
			      (void*) pt_to );
    }
    else {
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
      case is_basic_logical:{
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_overloaded:{
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_complex:{
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_pointer:{
	pt_to = create_stub_points_to(c, upt, fpb); 
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	cell sink = points_to_sink(pt_to);
	set tmp = pointer_formal_parameter_to_stub_points_to(upt, sink);
	pt_in = set_union(pt_in, pt_in,tmp);
	set_free(tmp);
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
    // FI: we probably should change this
    ;/*we don't know how to handle pointers to functions: nothing to
       be done for points-to analysis. */
  else if(type_void_p(upt)) {
    /* Create a target of unknown type */
    pt_to = create_stub_points_to(c, upt/* make_type_unknown() */, basic_undefined /*memory leak?*/);
    pt_in = set_add_element(pt_in, pt_in, (void*) pt_to );
  }
  else
    //we don't know how to handle other types
    pips_internal_error("Unexpected type");

  return pt_in;


}



/* Input : a formal parameter which has a derived type (FI, I guess).
 *  output : a set of points-to where sinks are stub points-to.
 *
 * FI: a lot of rewrite needed to simplify. Also, do not forget that
 * NULL maybe the received value. 
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
  r = cell_any_reference(c);
  e = reference_variable(r);

  if(type_variable_p(upt)){
    if(array_entity_p(e)){
      /* We ignore dimensions for the time being, descriptors are not
       * implemented yet...Amira Mensi*/
      ;
      /* ultimate_type() returns a wrong type for arrays. For
       * example for type int*[10] it returns int*[10] instead of int[10]. */
    }
    else {
      basic fpb = variable_basic(type_variable(upt));
      if( basic_derived_p(fpb)) {
	type t = ultimate_type(entity_type(e));
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
 *
 * FI: a formal parameter cannot be a typedef, but it can be typed
 * with a typedefined type.
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
  r = cell_any_reference(c);
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


set array_formal_parameter_to_stub_points_to(type t,cell c)
{
  set pt_in = set_generic_make(set_private,
			       points_to_equal_p,
			       points_to_rank);
  basic fpb = variable_basic(type_variable(t));
  if(basic_pointer_p(fpb)) {
    reference r = cell_any_reference(c);
    entity e = reference_variable(r);
    expression ind = make_unbounded_expression();
    reference ref = make_reference(e, CONS(EXPRESSION, ind, NULL));
    reference_consistent_p(ref);
    cell cel = make_cell_reference(ref);
    type pt = basic_pointer(fpb);
    points_to pt_to = create_stub_points_to(cel, pt, fpb);
    pt_in = set_add_element(pt_in, pt_in,
			    (void*) pt_to );
  }

  return pt_in;

}
