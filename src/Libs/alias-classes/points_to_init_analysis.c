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

/* Allocate a stub entity "stub" for entity "e" and with type
 * "t". Abort if "stub" already exists.
 *
 * It seems that type "t" could be derived from "e" since it should be
 * the pointed type of "e"'s type, but it is not at all the case in
 * general. Variable e is used to build a reference and the type
 * pointed by the reference may be different when arrays of structs of
 * arrays of structs are involved.
 */
entity create_stub_entity(entity e, string fs, type t)
{
  // local name for the stub
  string s = string_undefined;
  string en = (string) entity_user_name(e);

  // FI: guarantee about *local* new name uniqueness?
  if(formal_parameter_p(e)) {
    // Naming for sinks of formal parameters: use their offsets
    formal f = storage_formal( entity_storage(e) );
    int off = formal_offset(f);
    s = strdup(concatenate("_", en, fs,"_", i2a(off), NULL));
  }
  else if(top_level_entity_p(e)){ // FI: global_variable_p()
    // Naming for sinks of global variable: use their offsets
    int off = ram_offset(storage_ram(entity_storage(e)));
    s = strdup(concatenate("_", en, fs,"_", i2a(off), NULL));
  }
  else if(static_global_variable_p(e)){ // "static int i;"
    // Naming for sinks of static global variable: use their offsets
    int off = ram_offset(storage_ram(entity_storage(e)));
    s = strdup(concatenate("_", en, fs,"_", i2a(off), NULL));
  }
  else if(entity_stub_sink_p(e)) {
    // Naming for sinks of stubs: repeat their last suffix
    char *suffix = strrchr(en,'_');
    s = strdup(concatenate( en, fs, suffix, NULL )); 
  }
  
  // FI: the stub entity already exists?
  string formal_name = strdup(concatenate(get_current_module_name(),
					  MODULE_SEP_STRING, s, NULL));
  entity m = get_current_module_entity();
  entity stub = gen_find_entity(formal_name);
  // FI: I expect here a pips_assert("The stub cannot exist",
  // entity_undefined_p(stub));
 
  // If entity "stub" does not already exist, create it.
  if(entity_undefined_p(stub)) {
    entity fa = FindOrCreateEntity(get_current_module_name(),
					    FORMAL_AREA_LOCAL_NAME);
    if(type_undefined_p(entity_type(fa))) {
      // entity a = module_to_heap_area(f);
      entity_type(fa) = make_type(is_type_area, make_area(0, NIL));
      
      //ram r = make_ram(f, a, DYNAMIC_RAM_OFFSET, NIL);
      entity_storage(fa) = make_storage_rom();
      entity_initial(fa) = make_value(is_value_unknown, UU);
      // FI: DO we want to declare it abstract location?
      entity_kind(fa) = ABSTRACT_LOCATION | ENTITY_FORMAL_AREA;
      //entity_kind(fa) = ENTITY_FORMAL_AREA;
      AddEntityToDeclarations(fa, m);
    }
    stub = make_entity(formal_name,
		       copy_type(t),
		       make_storage_ram(make_ram(m, fa, DYNAMIC_RAM_OFFSET, NIL)),
		       make_value_unknown());
    (void) add_C_variable_to_area(fa, stub);

    AddEntityToDeclarations(stub, m);
  }
  else {
    /* FI: we are in deep trouble because the stub entity has already
     * been created... but we have no idea if it is or not the entity we
     * wanted as it depends on field names in the reference. And the
     * reference is not available from this function.
     */
    type st = entity_basic_concrete_type(stub);
    if(array_pointer_type_equal_p(st, t)) {
      // This may happen when evaluating conditions on demand because
      // they are evaluated twice, once true and once false.
      ;
    }
    else
      // Should be an internal error...
      pips_internal_error("Type incompatible request for a stub.\n");
  }

  return stub;
}

/* Create a stub entity "se" for entity "v" with type "t" and return a
 * cell based on a reference to the stub entity "se" with "d"
 * unbounded subscripts to account for the dimension of the source and
 * a zero subscript for implicit array.
 *
 * Type "pt" is the theoretically expected type for the reference
 * "sink_ref" within the returned cell, "sink_cell". We assert
 * "pt==points_to_cell_to_type(sink_cell)".
 *
 * Type "t" must account for the extra "d" dimensions.
 *
 * The type strictness is handled by the caller.
 *
 * The name of the function is misleading. It should be
 * "create_stub_sink_cell"... but in fact is does not handle the
 * struct case because struct can contain may different pointers,
 * directly or indirectly, depending on fields. This function is ok if
 * v a pointer or an array of pointers, but not if v is a struct.
 *
 * The function is called four times from create_stub_points_to().
 */
cell create_scalar_stub_sink_cell(entity v, // source entity
				  type st, // stub type
				  type pt, // sink cell type, pointed type
				  int d, // number of preexisting subscripts
				  list sl, // pre-existing subscripts
				  string fs) // disambiguator for fields
{
  //type vt = entity_basic_concrete_type(v);
  //pips_assert("v is not a struct", !struct_type_p(vt));
  type t = type_undefined;

  if(type_void_p(st))
    t = MakeTypeOverloaded();
  else
    t = copy_type(st);

  entity stub_entity = create_stub_entity(v, fs, t);
  reference sink_ref = reference_undefined;

  ifdebug(1) {
    pips_debug(1, "Entity \"%s\"\n", entity_local_name(v));
    pips_debug(1, "Stub type: "); print_type(t);
    fprintf(stderr, "\n");
    pips_debug(1, "Pointed type: "); print_type(pt);
    fprintf(stderr, "\n");
    pips_debug(1, "Number of source dimensions: %d\n", d); 
    fprintf(stderr, "\n");
  }

  if(type_functional_p(t)) {
    pips_assert("The source dimension is zero if the target is not an array", d==0);
    sink_ref = make_reference(stub_entity,  NIL);
  }
  else if(type_variable_p(t)) {
    /* When scalars are used, we should have "d==0" and "td==0" and hence "sl==NIL" */
    int td = variable_dimension_number(type_variable(t));
    pips_assert("The target dimension is greater than or equal to the source dimension", d<=td);
    int i;
    //if(false) {
    if(true) {
      list tl = NIL;
      /* FI: use 0 for all proper target dimensions */
      /* FI: adding 0 subscripts is similar to a dereferencing. We do
	 not know at this level if the dereferencing has been
	 requested. See pointer_reference02. The handling fo eval_p must
	 be modified correspondingly by adding 0 subscripts when the
	 source is an array. Or evaluation must be skipped. */
      for(i=d;i<td;i++) {
	tl = CONS(EXPRESSION, make_zero_expression(), tl);
      }
      sl = gen_nconc(sl, tl);
      sink_ref = make_reference(stub_entity, sl);
    }
    else {
      sink_ref = make_reference(stub_entity, sl);
      bool e_to_be_freed;
      type ept = points_to_reference_to_type(sink_ref, &e_to_be_freed);
      i = d;
      while(!array_pointer_type_equal_p(pt, ept)
	    && !(type_void_p(pt) && overloaded_type_p(ept))
	    && i<td) {
	if(e_to_be_freed) free_type(ept);
	list tl = CONS(EXPRESSION, make_zero_expression(), NIL);
	reference_indices(sink_ref) =
	  gen_nconc(reference_indices(sink_ref), tl);
	ept = points_to_reference_to_type(sink_ref, &e_to_be_freed);
	i++;
      }
      if(!array_pointer_type_equal_p(pt, ept)
	 && !(type_void_p(pt) && overloaded_type_p(ept)))
	pips_internal_error("The stub and expected types are incompatible.\n");
      else
	;
      if(e_to_be_freed) free_type(ept);
    }
  }
  else if(type_void_p(t)) {
    pips_assert("Implemented", false);
  }

  cell sink_cell = make_cell_reference(sink_ref);

  ifdebug(1) {
    bool to_be_freed;
    type ept = points_to_cell_to_type(sink_cell, & to_be_freed);
    if(!array_pointer_type_equal_p(pt, ept)
       && !(type_void_p(pt) && overloaded_type_p(ept))) {
      bool ok_p = false;
      if(array_type_p(pt)) {
	if(!array_type_p(ept)) {
	  // FI: do not forget the [0] subscript added...
	  basic bpt = variable_basic(type_variable(pt));
	  basic bept = variable_basic(type_variable(ept));
	  if(basic_equal_p(bpt, bept))
	    ok_p = true; // to be able to breakpointx
	}
      }
      if(!ok_p) {
      pips_debug(1, "pt = "); print_type(pt);
      fprintf(stderr, "\n");
      pips_debug(1, "ept = "); print_type(ept);
      fprintf(stderr, "\n");
      pips_internal_error("Effective type of sink cell does not match its expected type\n");
      }
    }
    if(to_be_freed) free_type(ept);
    pips_debug(1, "source entity: \"%s\", sink_cell: ", entity_user_name(v));
    print_points_to_cell(sink_cell);
    fprintf(stderr, "\n");
    pips_assert("sink_cell is consistent", cell_consistent_p(sink_cell));
  }

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

/* Generate a new subscript list. References to fields are ignored,
 * constant and unbounded expressions are preserved, non-constant
 * expressions are replaced by unbounded expressions.
 */
list points_to_indices_to_subscript_indices(list ptsl)
{
  list csl = NIL;
  list sl = NIL;
  for(csl = ptsl; !ENDP(csl); POP(csl)) {
    expression se = EXPRESSION(CAR(csl));
    // FI: how many different kinds of expressions do we have?
    // This dichotomy between references and calls may be too simple
    if(!expression_reference_p(se)) {
      if(!unbounded_expression_p(se)) {
	if(extended_integer_constant_expression_p(se))
	  sl = CONS(EXPRESSION, copy_expression(se), sl);
	else
	// do not propagate store-dependent information
	sl = CONS(EXPRESSION, make_unbounded_expression(), sl);
      }
      else { // copy the unbounded expression
	sl = CONS(EXPRESSION, copy_expression(se), sl);
      }
    }
    else {
      reference r = expression_reference(se);
      entity v = reference_variable(r);
      if(entity_field_p(v))
	; // ignore fields
      else
	// do not propagate store-dependent information
	sl = CONS(EXPRESSION, make_unbounded_expression(), sl);
    }
  }
  sl = gen_nreverse(sl);
  return sl;
}

/* Build an ASCII string to disambiguate the different field paths
 * that may exist in similar references.
 *
 * If the variable referenced by "r" is not a struct, returns the
 * empty string.
 *
 * If it is a struct, derive a string that is unique to a particular
 * combination of fields and subfields.
 */
string reference_to_field_disambiguator(reference r)
{
  string fs = string_undefined;
  string ofs = string_undefined;
  list sl = reference_indices(r);
  FOREACH(EXPRESSION, s, sl) {
    if(expression_reference_p(s)) {
      entity f = reference_variable(expression_reference(s));
      if(entity_field_p(f)) {
	int n = entity_field_rank(f);
	if(string_undefined_p(fs))
	  asprintf(&fs, "_%d_", n);
	else {
	  ofs = fs;
	  asprintf(&fs, "%s%d_", ofs, n);
	  free(ofs);
	}
      }
    }
  }
  if(string_undefined_p(fs))
    fs = strdup("");
  return fs;
}


/* points_to create_stub_points_to(cell c, bool exact_p)
 *
 * To create the points-to arc "pt_to" between a cell "c" containing a
 * constant path reference based on a formal parameter or a global
 * variable or another stub on one hand, and another new points-to
 * stub reference on the other.
 *
 * Argument "exact_p" specifies the approximation of the generated
 * points-to arc. It is overriden when NULL pointers are distinguished
 * according to property POINTS_TO_NULL_POINTER_INITIALIZATION. This
 * simplifies the semantics of the stub entities: they cannot
 * represent/hide the NULL abstract cell.
 *
 * The arc approximation itself is another issue as pointed out
 * by Beatrice Creusillet because we may have an approximation on the
 * source node, on the sink node or on the arc. Currently, an exact
 * approximation seems to indicate that no approximation at all is
 * made. This is issue is not currently solved (13 August 2012).
 *
 * Assumption: the reference in source cell "c" is a constant memory
 * path. So is the new reference hidden in the sink cell.
 *
 * This function must be consistent with type compatibility checks
 * used in points-to analysis, points_to_cell_types_compatibility(l, r).
 *
 * The global sink name of the generated stub is a concatenation of
 * the formal parameter, underscore, some number, and
 * POINTS_TO_MODULE_NAME as module name.
 *
 * The type of cell "c" and the type of the sink cell that is
 * generated must fit in some complicated way:
 *
 * 1. Do we consider types to be strict or do we allow pointer
 * arithmetic, which implies that pointers to scalars or anything else
 * in fact points to arrays? This is controlled by property
 * POINTS_TO_STRICT_POINTER_TYPES.
 *
 * 2. When the source is an array of pointers, do we add its
 * dimensions to the type of the sink? Yes, to preserve the
 * independence of independent cells (i.e. to be ready for a future
 * version with descriptors, compatible with dependence testing).
 *
 * 3. When the source is a really subscripted reference to an array of
 * pointers, how do we generate the subscripts of the sink? Especially
 * if the source is partially subscripted? We try to copy the
 * subscript to preserve as much information as possible.
 *
 * 4. Also, we have a choice: either point toward the first element of
 * an implicit array or point towards the array itself. To be
 * consistent with the interpretation of "p=a;" and "p=&a[0]", we
 * chose to points towards the object itself. But Pass
 * effects_with_points_to seems to expect pointers to the first array
 * element. A normalization function could be used to switch from one
 * convention to the other. The current idea is that, as much as
 * possible, points-to "c1->c2" implies that "c1==&c2;" holds. So 0
 * subscript are added to fill the last dimensions of the sink
 * reference.
 *
 * Here, we needs lots of examples with constant memory path
 * references to prepare a precise specification of the desired
 * function. Let c stands for the reference hidden in cell "c":
 *
 * * assignment13.c: "c=_t2_2_2[0][ip2];": here we have an array of
 * structs containing a pointer field. Beatrice would like us to infer
 * _t2_2_2[*][ip2] -> _t2_2_2_2[*]... Imagine the general case with
 * structs with arrays of structs with... We may also want an arc
 * _t2_2_2[*][ip2] -> NULL (see property
 * POINTS_TO_NULL_POINTER_INITIALIZATION), or even an arc
 * _t2_2_2[*][ip2] -> UNDEFINED (not implemented because about useless
 * since parameters are passed by value).  We may also want:
 * _t2_2_2[*][ip2] -> _t2_2_2_2[*][0]
 *
 * * "void foo(int *p) c=p;": depending on property on strict typing,
 * POINTS_TO_STRICT_POINTER_TYPES, we want either p -> _p_1
 * or p -> _p_1[0]
 *
 * * "void foo(int * pa[10]) {int * c=pa[0];}": pa[*] -> NULL, pa[*]
 * -> _pa_1[*] but c->_pa_1[0]
 *
 * * ptr_to_array01.c: "int ptr_to_array01(int * (*p)[10]) {int a; (*p)[3] = &a;}"
 * p->_p_1, p_1[3] -> a
 *
 * The cell "c" is not embedded in the generated points-to "pt_to". A
 * copy is allocated. The output has no sharing with the input
 * parameters.
 */
points_to create_stub_points_to(cell c, // source of the points-to
				type unused_st __attribute__ ((__unused__)), // expected type for the sink cell 
				// or the sink cell reference...
				bool exact_p)
{
  cell source_cell = copy_cell(c);
  reference source_r = cell_any_reference(source_cell);
  // FI: The field disambiguator "fs" is derived from source_r
  string fs = reference_to_field_disambiguator(source_r);
  entity v = reference_variable(source_r);
  list source_sl = reference_indices(source_r); // subscript list
  // The indices of a points-to reference may include fields as well as
  // usual array subscripts
  list sl = gen_full_copy_list(reference_indices(source_r));
  bool to_be_freed;
  type c_t = points_to_cell_to_type(c, &to_be_freed);
  type source_t = compute_basic_concrete_type(c_t);
  cell sink_cell = cell_undefined;
  bool e_exact_p = true;

  if(type_variable_p(source_t)) {
    bool strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
    //variable source_tv = type_variable(source_t);
    //list source_dl = variable_dimensions(source_tv);
    //int source_vd = (int) gen_length(source_dl); // FI: seems useless
    int source_cd = points_to_indices_to_array_index_number(source_sl);

    if(source_cd==0 /* && vd==0*/ ) {
      /* You may have a pointer or an unbounded array for source_t... */
      type sink_t = C_type_to_pointed_type(source_t);
      type stub_t = (strict_p || !type_variable_p(sink_t))?
		     copy_type(sink_t) : type_to_array_type(sink_t);
      sink_cell = create_scalar_stub_sink_cell(v, stub_t, sink_t, 0, NIL, fs);
      e_exact_p = exact_p;
      free_type(sink_t);
    }
    else if(source_cd>0) {
      // If called for a formal parameter, the reference may not
      // contain indices although we are dealing with an array...
      if(ENDP(sl)) {
	sl = make_unbounded_dimensions(source_cd);
      }
      else {
      // In case subscript indices have non constant values, replace them
      // with unbounded expressions
	sl = points_to_indices_to_subscript_indices(sl);
      }
      // dimensions to be added to the dimensions of "st"
      list ndl = make_unbounded_dimensions(source_cd);
      type sink_t = copy_type(type_to_pointed_type(source_t));
      if(type_void_p(sink_t)) {
	free_type(sink_t);
	sink_t = make_type_variable(make_variable(make_basic_overloaded(), NIL, NIL));
      }
      pips_assert("type_variable_p(sink_t)", type_variable_p(sink_t));
      variable nstv = type_variable(sink_t);
      variable_dimensions(nstv) = gen_nconc(ndl, variable_dimensions(nstv));
      type stub_t = strict_p ? copy_type(sink_t) : type_to_array_type(sink_t);
      sink_cell = create_scalar_stub_sink_cell(v, stub_t, sink_t, source_cd, sl, fs);
      // FI: this should be performed by the previous function
      // points_to_cell_add_unbounded_subscripts(sink_cell);
      e_exact_p = false;
    }
  }
  else if(type_functional_p(source_t)) {
    pips_internal_error("Unexpected case.\n");
    // sink_cell = create_scalar_stub_sink_cell(v, copy_type(st), st, 0);
    e_exact_p = true;
  }
  else
    pips_internal_error("Unexpected case.\n");

  points_to_cell_types_compatibility(source_cell, sink_cell);
  approximation rel = e_exact_p? make_approximation_exact():
    make_approximation_may();
  points_to pt_to = make_points_to(source_cell, sink_cell, rel,
			 make_descriptor_none());
  pointer_index ++; // FI: is not used for formal parameters, is this the right place for the increment

  if(to_be_freed) free_type(c_t);
  
  return pt_to;
}

/* Take into account the POINTS_TO_STRICT_POINTER_TYPE to allocate a
 * sink cell of type "t" if the strictness is requested and of type
 * "array of t" if not.
 */
points_to create_advanced_stub_points_to(cell c, type t, bool exact_p)
{
  pips_internal_error("This function is no longer used. Functionality moved into create_stub_points_to directly...\n");
  points_to pt = points_to_undefined;
  bool strict_p = get_bool_property("POINTS_TO_STRICT_POINTER_TYPES");
  if(true || strict_p || array_type_p(t))
    pt = create_stub_points_to(c, t, exact_p);
  else {
    /* assume that pointers to scalars always points towards an array
       of unknown dimension. */
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
    pt_in = add_arc_to_simple_pt_map(npt, pt_in);
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
	// pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_to = create_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_float:{
	//pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_to = create_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_logical:{
	//pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_to = create_stub_points_to(c, upt, !null_initialization_p);
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
	//pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_to = create_stub_points_to(c, upt, !null_initialization_p);
	pt_in = set_add_element(pt_in, pt_in,
				(void*) pt_to );
	break;
      }
      case is_basic_pointer:{
	//pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p); 
	pt_to = create_stub_points_to(c, upt, !null_initialization_p); 
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
	//pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_to = create_stub_points_to(c, upt, !null_initialization_p);
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
	//pt_to = create_advanced_stub_points_to(c, upt, !null_initialization_p);
	pt_to = create_stub_points_to(c, upt, !null_initialization_p);
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
    add_arc_to_simple_pt_map(pt_to, pt_in);
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
		  type p_ent_type = compute_basic_concrete_type(type_to_pointed_type(ent_type));
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
