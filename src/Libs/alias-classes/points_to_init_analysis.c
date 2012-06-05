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

cell create_scalar_stub_sink_cell(entity v, type t, int d)
{
  entity formal_parameter = create_stub_entity(v, t);
  reference sink_ref = reference_undefined;

  /* Do we want to assume that "int * p;" defines a pointer to an
   * array of unbounded dimension?
   */
  bool type_strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
  if(!type_strict_p
     // && !derived_type_p(t)
     && !type_functional_p(t)
     && !array_type_p(t)) {
    //sink_ref =
    //make_reference(formal_parameter,
    //		     CONS(EXPRESSION, make_unbounded_expression(), NIL));
    //sink_ref =
    //  make_reference(formal_parameter,
    //		     CONS(EXPRESSION, int_to_expression(0), NIL));
    sink_ref = make_reference(formal_parameter,  NIL);
  }
  else {
    if(array_type_p(t)) {
      // sink_ref =
      // make_reference(formal_parameter,
      // CONS(EXPRESSION, make_unbounded_expression(), NIL));
      sink_ref = make_reference(formal_parameter,  NIL);
    }
    else
      sink_ref = make_reference(formal_parameter,  NIL);
  }

  int i;
  list sl = NIL;
  for(i=0;i<d;i++) {
    sl = CONS(EXPRESSION, make_unbounded_expression(), sl);
  }
  reference_indices(sink_ref) = sl;

  cell sink_cell = make_cell_reference(sink_ref);
  return sink_cell;
}


/* To create the points-to between a formal parameter or a global
 * variable or another stub on one hand, and another new stub on the
 * other. Or to create a points-to for any reference to a formal
 * parameter, a global variable or a points-to stub.
 *
 * The type of cell c and the type of the sink that is generated must
 * fit in some complicated way:
 *
 * 1. Do we consider types to be strict or do we allow pointer
 * arithmetic, which implies that pointers to scalar in fact points to
 * arrays?
 *
 * 2. When the source is an array of pointers, do we add its
 * dimensions to the type of the sink?
 *
 * 3. When the source is a really subscripted reference to an array of
 * pointers, how do we generate the subscripts of the sink? Especially
 * if the source is partially subscripted?
 *
 * This function must be consistent with type compatibility checks
 * used in points-to analysis.
 *
 * The sink name is a concatenation of the formal parameter and the
 * POINTS_TO_MODULE_NAME.
 *
 * Also, we have a choice: either points to the first element of an
 * implicit array or points towards the array itself. To be consistent
 * with the interpretation of "p=a;" and "p=&a[0]", we chose to points
 * towards the object itself. But Pass effects_with_points_to seems to
 * expect pointers to the first array element. A normalization
 * function could be used.
 *
 * As regards Point 3, Beatrice Creusillet also suggests to check if
 * the source cell is an array and then to generate a special sink
 * with an unbounded expression to express the fact that several
 * targets are defined at once and that the set of arcs is between any
 * element of the source and any target.
 *
 * FI: if we single out the NULL pointer value, this has to be a may
 * points-to. Another discussion about stubs is based on the fact that
 * they may represent several cells at the call site, although they
 * are only one cell at run-time and for any specific execution of the
 * called function.
 *
 * Singling out the NULL pointers is useful to exploit conditions in
 * tests and while loops. It may also lead to more precise
 * fix-points for the points-to graph.
 *
 * The cell "c" is not embedded in the generated points-to "pt_to". A
 * copy is allocated. The output has no sharing with the input
 * parameters.
 */
points_to create_stub_points_to(cell c, // source of the points-to
				type st, // expected type for the sink cell 
				// or the sink cell reference...
				__attribute__ ((__unused__)) basic b)
{
  points_to pt_to = points_to_undefined;
  //reference sink_ref = reference_undefined;
  cell source_cell = copy_cell(c);
  reference r = cell_any_reference(source_cell);
  entity v = reference_variable(r);
  list sl = reference_indices(r); // They may include fields as well
                                  // as usual array subscripts
  //int rd = (int) gen_length(sl); // FI: To be used later

  // FI->AM: we do not resolve the typedef, nor the dimensions hidden by
  // the typedefs...
  type vt = entity_type(v);
  variable vv = type_variable(vt);
  list dl = variable_dimensions(vv);
  int vd = (int) gen_length(dl);

  //pips_assert("source dimension is well known", source_dim==vd);

  cell sink_cell = cell_undefined;

  if(vd==0) {
    // "st" can be an array type
    // variable_entity_dimension();
    // variable_dimension_number();
    sink_cell = create_scalar_stub_sink_cell(v, st, vd);
  }
  else {
    // The source is an array of pointers of you do not know what...
    list ndl = gen_full_copy_list(dl);
    // Add these dimensions to "st"
    type nst = copy_type(st);
    // FI: quid of arrays of functions, type equivalent to pointers to
    // functions?
    pips_assert("type_variable_p(nst)", type_variable_p(nst));
    variable nstv = type_variable(st);
    variable_dimensions(nstv) = gen_nconc(ndl, variable_dimensions(nstv));
    sink_cell = create_scalar_stub_sink_cell(v, st, vd);
    reference r = cell_any_reference(sink_cell);
    // Add the missing subscripts to the sink cell reference, if they
    // are not added by create_scalar_stub_sink()
    /*
    int i;
    for(i=0;i<vd;i++) {
      expression use = make_unbounded_expression();
      reference_indices(r) = gen_nconc(CONS(EXPRESSION, use, NIL),
				       reference_indices(r));
    }
    */

    //pips_internal_error("Not implemented yet!\n");
  }

  bool null_initialization_p = get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");
  approximation rel = null_initialization_p? make_approximation_may():
    make_approximation_exact();
  pt_to = make_points_to(source_cell, sink_cell, rel,
			 make_descriptor_none());
  pointer_index ++;
  
  return pt_to;
}

/* Take into account the POINTS_TO_STRICT_POINTER_TYPE to allocate a
 * sink cell of type "t" if the strictness is requested and of type
 * "array of t" if not.
 */
points_to create_advanced_stub_points_to(cell c, type t)
{
  points_to pt = points_to_undefined;
  bool strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
  if(strict_p)
    pt = create_stub_points_to(c, t, basic_undefined);
  else {
    /* assume that pointer always points towards an array of
       unknown dimension. */
    type at = type_to_array_type(t);
    pt = create_stub_points_to(c, at, basic_undefined);
    // FI: I do not know if we should free t [and/or at]
  }
  return pt;
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
  bool type_strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
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
    entity DummyTarget = FindOrCreateEntity(POINTER_DUMMY_TARGETS_AREA_LOCAL_NAME,
					    POINTER_DUMMY_TARGETS_AREA_LOCAL_NAME);
    // FI->AM: weird, it is redone when the entity already exists
    entity_kind(DummyTarget) = ENTITY_POINTER_DUMMY_TARGETS_AREA;
    storage rs = make_storage_ram(make_ram(get_current_module_entity(),
					   DummyTarget,
					   UNKNOWN_RAM_OFFSET, NIL));
    formal_parameter = make_entity(formal_name,
				   pt,
				   // FI->AM: if it is made rom, then
				   // the entitY is no longer
				   // recognized by entity_stub_sink_p()
				   // make_storage_rom(),
				   rs,
				   make_value_unknown());
  }

  if(type_strict_p)
    sink_ref = make_reference(formal_parameter, CONS(EXPRESSION, int_to_expression(0), NIL));
  else if((int)gen_length(l_dim)>1){
    sink_ref = make_reference(formal_parameter,l_ind);
  }
  else {
    // sink_ref = make_reference(formal_parameter, CONS(EXPRESSION, int_to_expression(0), NIL));
    // FI: no reason to index an array; see "p = &a;"
    sink_ref = make_reference(formal_parameter, NIL);
  }

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
  // AM: Get the property POINTS_TO_NULL_POINTER_INITIALIZATION
  bool null_initialization_p = get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");
  if(null_initialization_p) {
  cell nc = copy_cell(c);
  cell null_c = make_null_pointer_value_cell();
  points_to npt = make_points_to(nc, null_c,
				 make_approximation_may(),
				 make_descriptor_none());
  pt_in = add_arc_to_pt_map(npt, pt_in);
  }

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
	// type st = type_undefined; // sink type
	pt_to = create_advanced_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_float:{
	pt_to = create_advanced_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_logical:{
	pt_to = create_advanced_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_overloaded:{
	// FI: Oops, what are we doing here?
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_complex:{
	pt_to = create_advanced_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_pointer:{
	pt_to = create_advanced_stub_points_to(c, upt); 
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	cell sink = points_to_sink(pt_to);
	if(false) {
	  /* Recursive descent for pointers: the new sink becomes the
	     new source... FI: I do not think this is useful because
	     they will be created on demand... */
	  set tmp = pointer_formal_parameter_to_stub_points_to(upt, sink);
	  pt_in = set_union(pt_in, pt_in,tmp);
	  set_free(tmp);
	}
	/* what about storage*/
	break;
      }
      case is_basic_derived:{
	pt_to = create_advanced_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_bit:
	pips_internal_error("Not implemented.\n");
	break;
      case is_basic_string:{
	// FI: I'm not too sure about what to do for strings...
	pt_to = create_stub_points_to(c, upt, fpb);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_typedef:{
	pt_to = create_advanced_stub_points_to(c, upt);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      default: pips_internal_error("unexpected tag %d", basic_tag(fpb));
	break;
      }
    }
  }
  else if(type_functional_p(upt)) {
    pt_to = create_stub_points_to(c, upt, basic_undefined);
    add_arc_to_pt_map(pt_to, pt_in);
}
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
		  type p_ent_type = type_to_pointed_type(ent_type);
		  cell source_cell = make_cell_reference(source_ref);
		  pt_to = create_stub_points_to(source_cell, p_ent_type, fpb);
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


/* t is supposed to be a concrete type */
set array_formal_parameter_to_stub_points_to(type t, cell c)
{
  set pt_in = set_generic_make(set_private,
			       points_to_equal_p,
			       points_to_rank);
  basic fpb = variable_basic(type_variable(t));
  if(basic_pointer_p(fpb)) {
    reference r = cell_any_reference(c);
    entity e = reference_variable(r);
    //expression ind = make_unbounded_expression();
    //reference ref = make_reference(e, CONS(EXPRESSION, ind, NULL));
    reference ref = make_reference(e, NULL);
    reference_consistent_p(ref);
    cell cel = make_cell_reference(ref);
    type pt = copy_type(basic_pointer(fpb));
    list sl = NIL;
    if(array_type_p(t)) {
      // FI->BC: to follow Beatrice's idea that dimensions of the sources
      // have to be replicated in the target
      // I am afraid this is going to make typing and subscript analysis impossible
      // I do not see the point of addind a unique extra-dimension
      // The fun part: imagine a n-D array of pointers towards m-D arrays...
      // The target stub is an n+m-D array...
      list dl = variable_dimensions(type_variable(t));
      list ndl = gen_full_copy_list(dl);
      variable_dimensions(type_variable(pt))
	= gen_nconc(ndl, variable_dimensions(type_variable(pt)));
      int d = gen_length(dl), i;
      for(i=0;i<d; i++) {
	sl = CONS(EXPRESSION, make_unbounded_expression(), sl);
      }
    }
    // FI: d is recomputed below in the same way
    points_to pt_to = create_stub_points_to(cel, pt, fpb/*, d*/);
    cell source = points_to_source(pt_to);
    reference sr = cell_any_reference(source);
    reference_indices(sr) = gen_nconc(sl, reference_indices(sr));
    pt_in = set_add_element(pt_in, pt_in,
			    (void*) pt_to );
  }

  return pt_in;

}
