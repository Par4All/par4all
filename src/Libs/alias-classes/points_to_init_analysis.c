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
  type pt = type_undefined; // FI: should be copy_type(t)
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
    // FI: this is dealt with at a much higher level
    if(false && !type_strict_p && !struct_type_p(t)) {
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
  else if (type_functional_p(t)){
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

/* Create a stub entity "se" for entity "v" with type "t" and return a
 * cell based on a reference to the stub entity "se" with "d"
 * unbounded subscripts to account for the dimension of the source and
 * a zero subscript for implicit array.
 *
 * It is not clear if type "t" already accounts for the extra "d"
 * dimensions... Can we assert "t==points_to_cell_to_type(sink_cell)"?
 */
cell create_scalar_stub_sink_cell(entity v, type t, int d)
{
  // FI: "d" must already have been taken into account in "t" for the
  // typing of "stub_entity" to be correct...
  entity stub_entity = create_stub_entity(v, t);
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
    //make_reference(stub_entity,
    //		     CONS(EXPRESSION, make_unbounded_expression(), NIL));
    //sink_ref =
    //  make_reference(stub_entity,
    //		     CONS(EXPRESSION, int_to_expression(0), NIL));
    sink_ref = make_reference(stub_entity,  NIL);
  }
  else {
    // FI: I do not think we can have a reliable test here to decide
    // if a zero subscript must be added or not
    if(array_type_p(t) && d==0) {
      // For the initialization of pi in Pointers/inc01, a zero
      // subscript is needed
      // FI: to get p->_p_1[0]
      // FI: why not a set of zero subscripts in some other cases?
      sink_ref =
	make_reference(stub_entity,
		       CONS(EXPRESSION, make_zero_expression(), NIL));
      // FI: to get p->_p_1
      // sink_ref = make_reference(stub_entity,  NIL);
      // FI: I may also want _argv_2[*] if the source is an array and I
      // am in the initialization phase... See for instance malloc02()
      // See also below the loop over d
    }
    else
      sink_ref = make_reference(stub_entity,  NIL);
  }

  int i;
  list sl = NIL;
  for(i=0;i<d;i++) {
    // FI: to be understood; parameter "d" is passed to reflect the
    // dimensions of the source
    sl = CONS(EXPRESSION, make_unbounded_expression(), sl);
    // sl = CONS(EXPRESSION, int_to_expression(0), sl);
    ;
  }

  reference_indices(sink_ref) = gen_nconc(sl, reference_indices(sink_ref));

  cell sink_cell = make_cell_reference(sink_ref);
  return sink_cell;
}

/* Count the number of array indices and ignore the field
 * subscripts.
 */
int points_to_indices_to_array_index_number(list sl)
{
  int c = 0;
  FOREACH(EXPRESSION, s, sl) {
    if(!expression_reference_p(s))
      c++;
  }
  return c;
}

/* FI: probably a duplicate... */
void points_to_indices_to_unbounded_indices(list sl)
{
  list csl = NIL;
  for(csl = sl; !ENDP(csl); POP(csl)) {
    expression se = EXPRESSION(CAR(csl));
    if(!expression_reference_p(se)) {
      if(!unbounded_expression_p(se)) {
	free_expression(se);
	EXPRESSION_(CAR(csl)) = make_unbounded_expression();
      }
    }
  }
  return; // Useful for gdb?
}


/* To create the points-to "pt_to" between a cell "c" containing a
 * constant path reference based on a formal parameter or a global
 * variable or another stub on one hand, and another new stub on the
 * other. The type of the sink is argument "st".
 *
 * It is not clear why "st" is an argument. "st" is not modified by
 * side effect to take into account dimensions implied by the
 * reference in cell "c" because a copy, "nst" is used, but it may be
 * embedded in the definition of the sink stub in the scalar case...
 *
 * Argument "exact_p" specifies the approximation of the generated
 * points-to.
 *
 * Assumption: the reference in cell "c" is a constant memory
 * path. Should it be checked?
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
 * used in points-to analysis, points_to_cell_types_compatibility(nl, r).
 *
 * The global sink name of the generated stub is a concatenation of
 * the formal parameter, underscore, some number, and
 * POINTS_TO_MODULE_NAME as module name.
 *
 * Also, we have a choice: either points to the first element of an
 * implicit array or points towards the array itself. To be consistent
 * with the interpretation of "p=a;" and "p=&a[0]", we chose to points
 * towards the object itself. But Pass effects_with_points_to seems to
 * expect pointers to the first array element. A normalization
 * function could be used to switch from one convention to the
 * other. The current idea is that, as much as possibke, points-to
 * "c1->c2" implies that "c1==&c2;" holds.
 *
 * As regards Point 3, Beatrice Creusillet also suggests to check if
 * the source cell is an array and then to generate a special sink
 * with a list of unbounded expressions as first dimension to express
 * the fact that several targets are defined at once and that the set
 * of arcs is between any element of the source and any target. Of
 * course, several arrays may be used when structs of structs with
 * array fields are defined.
 *
 * Here, we needs lots of examples with constant memory path
 * references to prepare a precise specification of the desired
 * function. Let c stands for the reference hidden in cell "c":
 *
 * assignment13.c: "c=_t2_2_2[0][ip2];": here we have an array of structs
 * containing a pointer field. Beatrice would like us to infer
 * _t2_2_2[*][ip2] -> _t2_2_2_2[*]... Imagine the general case with
 * structs with arrays of structs with... We may also want an arc
 * _t2_2_2[*][ip2] -> NULL (see property
 * POINTS_TO_NULL_POINTER_INITIALIZATION), or even an arc
 * _t2_2_2[*][ip2] -> UNDEFINED (not implemented).
 * We may also want: _t2_2_2[*][ip2] -> _t2_2_2_2[*][0]
 *
 * * "void foo(int *p) c=p;": depending on property on strict typing,
 * POINTS_TO_STRICT_POINTER_TYPES, we want either p -> _p_1
 * or p -> _p_1[0]
 *
 * * "void foo(int * pa[10]) {int * c=pa[0];}": pa[*] -> NULL, pa[*]
 * -> _pa_1[*] but c->_pa_1[0]
 *
 * ptr_to_array01.c: "int ptr_to_array01(int * (*p)[10]) {int a; (*p)[3] = &a;}"
 * p->_p_1, p_1[3] -> a
 *
 *
 * FI: if we single out the NULL pointer value, this has to be a may
 * points-to. Another discussion about stubs is based on the fact that
 * they may represent several cells at the call site, although they
 * are only one cell at run-time and for any specific execution of the
 * called function.
 *
 * Singling out the NULL (and UNDEFINED?) pointer values is useful to
 * exploit conditions in tests and while loops. It may also lead to
 * more precise fix-points for the points-to graph. This issue is
 * dealt with in many different places. It impacts the approximation
 * of the generated points-to. The approximation itself is another
 * issue as pointed out by Beatrice Creusillet because we may have an
 * approximation on the source noce, on the sink node or on the
 * arc. Currently, an exact approximation seems to indicate that no
 * approximation at all is made.
 *
 * The cell "c" is not embedded in the generated points-to "pt_to". A
 * copy is allocated. The output has no sharing with the input
 * parameters.
 */
points_to create_stub_points_to(cell c, // source of the points-to
				type st, // expected type for the sink cell 
				// or the sink cell reference...
				bool exact_p)
{
  //points_to pt_to = points_to_undefined;
  //reference sink_ref = reference_undefined;
  cell source_cell = copy_cell(c);
  reference r = cell_any_reference(source_cell);
  entity v = reference_variable(r);
  list sl = reference_indices(r); // They may include fields as well
                                  // as usual array subscripts
  //int rd = (int) gen_length(sl); // FI: To be used later, too simple to handle fields

  // FI->AM: we do not resolve the typedef, nor the dimensions hidden by
  // the typedefs...
  // type vt = entity_type(v);
  bool to_be_freed;
  type vt = points_to_cell_to_type(c, &to_be_freed);
  cell sink_cell = cell_undefined;
  bool e_exact_p = true;

  if(type_variable_p(vt)) {
    variable vv = type_variable(vt);
    list dl = variable_dimensions(vv);
    int vd = (int) gen_length(dl);

    //pips_assert("source dimension is well known", source_dim==vd);

    // Cell array dimension count (some subscripts are fields)
    int cd = points_to_indices_to_array_index_number(sl);

    if(cd==0 && vd==0) {
      // "st" can be an array type
      // variable_entity_dimension();
      // variable_dimension_number();
      sink_cell = create_scalar_stub_sink_cell(v, st, vd);
      e_exact_p = exact_p;
    }
    else if(cd>0) {
      // In case subscript indices have constant values, replace them
      // with unbounded expressions
      points_to_indices_to_unbounded_indices(sl);
      // dimensions to be added to the dimensions of "st"
      list ndl = make_unbounded_dimensions(cd);
      type nst = copy_type(st);
      pips_assert("type_variable_p(nst)", type_variable_p(nst));
      variable nstv = type_variable(nst);
      variable_dimensions(nstv) = gen_nconc(ndl, variable_dimensions(nstv));
      sink_cell = create_scalar_stub_sink_cell(v, nst, vd);
      // FI: this should be performed by the previous function
      points_to_cell_add_unbounded_subscripts(sink_cell);
      e_exact_p = false;
    }
    else { //cd==0 && vd>0
      // The source is an array of pointers of you do not know what...
      // The processing is almost identical to the one above
      list ndl = gen_full_copy_list(dl);
      // Add these dimensions to "st"
      type nst = copy_type(st);
      // FI: quid of arrays of functions, type equivalent to pointers to
      // functions?
      pips_assert("type_variable_p(nst)", type_variable_p(nst));
      variable nstv = type_variable(nst);
      variable_dimensions(nstv) = gen_nconc(ndl, variable_dimensions(nstv));
      sink_cell = create_scalar_stub_sink_cell(v, nst, vd);
      // Adapt the source cell
      reference scr = cell_any_reference(source_cell);
      reference_indices(scr) = NIL; // memory leak, free_expressions()
      points_to_cell_add_unbounded_subscripts(source_cell);
      e_exact_p = false;
    }
  }
  else if(type_functional_p(vt)) {
    sink_cell = create_scalar_stub_sink_cell(v, copy_type(st), 0);
    e_exact_p = true;
  }
  else
    pips_internal_error("Unexpected case.\n");

  points_to_cell_types_compatibility(source_cell, sink_cell);
  approximation rel = e_exact_p? make_approximation_exact():
    make_approximation_may();
  points_to pt_to = make_points_to(source_cell, sink_cell, rel,
			 make_descriptor_none());
  pointer_index ++;

  if(to_be_freed) free_type(vt);
  
  return pt_to;
}

/* Take into account the POINTS_TO_STRICT_POINTER_TYPE to allocate a
 * sink cell of type "t" if the strictness is requested and of type
 * "array of t" if not.
 */
points_to create_advanced_stub_points_to(cell c, type t, bool exact_p)
{
  points_to pt = points_to_undefined;
  bool strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
  if(strict_p)
    pt = create_stub_points_to(c, t, exact_p);
  else {
    /* assume that pointers always points towards an array of
       unknown dimension. */
    type at = type_to_array_type(t);
    pt = create_stub_points_to(c, at, exact_p);
    // FI: I do not know if we should free t [and/or at]
  }
  return pt;
}

/* To create the points-to stub associated to the formal parameter,
 * the sink name is a concatenation of the formal parmater and the
 * POINTS_TO_MODULE_NAME.
 */
points_to create_pointer_to_array_stub_points_to(cell c, type t, bool exact_p)
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
  entity stub_entity = gen_find_entity(formal_name);
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

  if(entity_undefined_p(stub_entity)) {
    entity DummyTarget = FindOrCreateEntity(POINTER_DUMMY_TARGETS_AREA_LOCAL_NAME,
					    POINTER_DUMMY_TARGETS_AREA_LOCAL_NAME);
    // FI->AM: weird, it is redone when the entity already exists
    entity_kind(DummyTarget) = ENTITY_POINTER_DUMMY_TARGETS_AREA;
    storage rs = make_storage_ram(make_ram(get_current_module_entity(),
					   DummyTarget,
					   UNKNOWN_RAM_OFFSET, NIL));
    stub_entity = make_entity(formal_name,
				   pt,
				   // FI->AM: if it is made rom, then
				   // the entitY is no longer
				   // recognized by entity_stub_sink_p()
				   // make_storage_rom(),
				   rs,
				   make_value_unknown());
  }

  if(type_strict_p)
    sink_ref = make_reference(stub_entity, CONS(EXPRESSION, int_to_expression(0), NIL));
  else if((int)gen_length(l_dim)>1){
    sink_ref = make_reference(stub_entity,l_ind);
  }
  else {
    // sink_ref = make_reference(stub_entity, CONS(EXPRESSION, int_to_expression(0), NIL));
    // FI: no reason to index an array; see "p = &a;"
    sink_ref = make_reference(stub_entity, NIL);
  }

  cell sink_cell = make_cell_reference(sink_ref);
  approximation rel =
    exact_p? make_approximation_exact() : make_approximation_may();
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
      pt_to = create_pointer_to_array_stub_points_to(c, upt,
						     !null_initialization_p);
      pt_in = set_add_element(pt_in, pt_in,
			      (void*) pt_to );
    }
    else {
      switch(basic_tag(fpb)){
      case is_basic_int:{
	// type st = type_undefined; // sink type
	pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_float:{
	pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_logical:{
	pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_overloaded:{
	// FI: Oops, what are we doing here?
	pt_to = create_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_complex:{
	pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_pointer:{
	pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p); 
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
	pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_bit:
	pips_internal_error("Not implemented.\n");
	break;
      case is_basic_string:{
	// FI: I'm not too sure about what to do for strings...
	pt_to = create_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_typedef:{
	pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
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
    pt_to = create_stub_points_to(c, upt, !null_initialization_p);
    add_arc_to_pt_map(pt_to, pt_in);
}
  else if(type_void_p(upt)) {
    /* Create a target of unknown type */
    pt_to = create_stub_points_to(c, upt/* make_type_unknown() */, 
				  !null_initialization_p);
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
  bool exact_p = !get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");
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
		  pt_to = create_stub_points_to(source_cell, p_ent_type, exact_p);
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
  bool exact_p = !get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");

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
		  pt_to = create_stub_points_to(source_cell, ent_type, exact_p);
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


/* Type "t" is supposed to be a concrete type.
 *
 * Type "t" may be modified: no.
 *
 * Cell "c" is copied.
 *
 * The dimensions of type "t" are forgotten. They are retrieved later
 * from the type of the entity references in cell "c".
 */
set array_formal_parameter_to_stub_points_to(type t, cell c)
{
  set pt_in = set_generic_make(set_private,
			       points_to_equal_p,
			       points_to_rank);
  basic fpb = variable_basic(type_variable(t));

  if(basic_pointer_p(fpb)) {
    reference r = cell_any_reference(c);
    entity e = reference_variable(r);
    reference ref = make_reference(e, NULL);
    reference_consistent_p(ref);
    cell cel = make_cell_reference(ref);
    type pt = copy_type(basic_pointer(fpb));

    bool strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
    if(scalar_type_p(pt) && !strict_p) {
      /* Add an artificial dimension for pointer arithmetic */
      expression l = make_zero_expression();
      expression u = make_unbounded_expression();
      dimension d = make_dimension(l,u);
      variable ptv = type_variable(pt);
      variable_dimensions(ptv) = CONS(DIMENSION, d, NIL);
    }

    bool exact_p = !get_bool_property("POINTS_TO_NULL_POINTER_INITIALIZATION");
    points_to pt_to = create_stub_points_to(cel, pt, exact_p);
    //cell source = points_to_source(pt_to);
    pt_in = set_add_element(pt_in, pt_in,
			    (void*) pt_to );
  }

  return pt_in;

}
