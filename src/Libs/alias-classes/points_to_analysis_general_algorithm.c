/*

  $Id: scalarization.c 14503 2009-07-10 15:11:52Z mensi $

  Copyright 1989-2009 MINES ParisTech

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
/*
This file is used to compute points-to sets intraprocedurally.
At first we get SUMMARY_POINTS_TO, then we use it as input set for the
function points_to_statement().
The function points_to_statement() calls
points_to_recursive_statement(). At the level of
points_to_recursive_statement() we dispatch according to the
instruction's type : if it's a test we call points_to_test(), if it's
a sequence we call points_to_sequence() ...
When we the instruction is  call we call points_to_call() and
according to the nature of the call we affect the appropriate
treatment.
If the call is an intrinsics, preciselly the operator "=", we call
points_to_assignment(). This latter dispatch the treatment according
to the nature of the left hand side and the right hand side.
To summarize this is a call graph simulating the points-to analysis :

points_to_statement
      |
      |-->points_to_recursive_statement
                 |
		 |-->points_to_call------>points_to_intrinsics
		 |-->points_to_sequence       |
		 |-->points_to_while          |-->points_to_expression
		 |-->points_to_expression     |-->points_to_general_assignment
		 |...                         |-->points_to_filter_with_effects
                                              |-->points_to_assignment
					           |
						   |->basic_ref_ref
						   |->basic_ref_addr
						   |->basic_ref_deref
						   |...



*/
#include <stdlib.h>
#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
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
#include "preprocessor.h"
#include "pipsdbm.h"
#include "resources.h"
#include "prettyprint.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"

/* operator fusion to calculate the relation of two points
   to. (should be at a higher level) */
/* static approximation fusion_approximation(approximation app1, */
/* 					  approximation app2) */
/* { */
/*   if(approximation_exact_p(app1) && approximation_exact_p(app2)) */
/*     return make_approximation_exact(); */
/*   else */
/*     return make_approximation_may(); */
/* } */


/* we get the type of the expression by calling expression_to_type()
 * which allocates a new one. Then we call ultimate_type() to have
 * the final type. Finally we test if it's a pointer by using pointer_type_p().*/
bool expression_pointer_p(expression e) {
  type et = expression_to_type(e);
  type t = ultimate_type(et);
  return pointer_type_p(t);

}

/* Same as previous function, but for double pointers. */
bool expression_double_pointer_p(expression e) {
  bool double_pointer_p = false;
  type et = expression_to_type(e);
  type t = ultimate_type(et);
  if (pointer_type_p(t)) {
    type pt = basic_pointer(variable_basic(type_variable(t)));
    double_pointer_p = pointer_type_p(pt);
  }

  return double_pointer_p;
}

/* given an expression, return the referenced entity in case exp is a
   reference */
entity argument_entity(expression exp) {
  syntax syn = expression_syntax(exp);
  entity var = entity_undefined;

  if (syntax_reference_p(syn)) {
    reference ref = syntax_reference(syn);
    var = reference_variable(ref);
  }
  return copy_entity(var);
}

bool points_to_compare_cell(cell c1, cell c2){

  int i = 0;
  reference r1 = cell_to_reference(c1);
  reference r2 = cell_to_reference(c2);
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);
  list sl1 = NIL, sl2 = NIL;
  extern const char* entity_minimal_user_name(entity);

  i = strcmp(entity_minimal_user_name(v1), entity_minimal_user_name(v2));
  if(i==0) {
    sl1 = reference_indices(r1);
    sl2 = reference_indices(r2);
    int i1 = gen_length(sl1);
    int i2 = gen_length(sl2);

    i = i2>i1? 1 : (i2<i1? -1 : 0);

    for(;i==0 && !ENDP(sl1); POP(sl1), POP(sl2)){
      expression se1 = EXPRESSION(CAR(sl1));
      expression se2 = EXPRESSION(CAR(sl2));
      if(expression_constant_p(se1) && expression_constant_p(se2)){
	int i1 = expression_to_int(se1);
	int i2 = expression_to_int(se2);
	i = i2>i1? 1 : (i2<i1? -1 : 0);
      }else{
	string s1 = words_to_string(words_expression(se1, NIL));
	string s2 = words_to_string(words_expression(se2, NIL));
	i = strcmp(s1, s2);
      }
    }
  }

  return (i== 0 ? true: false) ;
}



/* Order the two points-to relations according to the alphabetical
   order of the underlying variables. Return -1, 0, or 1. */
int points_to_compare_location(void * vpt1, void * vpt2) {
  int i = 0;
  points_to pt1 = *((points_to *) vpt1);
  points_to pt2 = *((points_to *) vpt2);

  cell c1so = points_to_source(pt1);
  cell c2so = points_to_source(pt2);
  cell c1si = points_to_sink(pt1);
  cell c2si = points_to_sink(pt2);

  // FI: bypass of GAP case
  reference r1so = cell_to_reference(c1so);
  reference r2so = cell_to_reference(c2so);
  reference r1si = cell_to_reference(c1si);
  reference r2si = cell_to_reference(c2si);

  entity v1so = reference_variable(r1so);
  entity v2so = reference_variable(r2so);
  entity v1si = reference_variable(r1si);
  entity v2si = reference_variable(r2si);
  list sl1 = NIL, sl2 = NIL;
  // FI: memory leak? generation of a new string?
  extern const char* entity_minimal_user_name(entity);

  i = strcmp(entity_minimal_user_name(v1so), entity_minimal_user_name(v2so));
  if(i==0) {
    i = strcmp(entity_minimal_user_name(v1si), entity_minimal_user_name(v2si));
    if(i==0) {
      /* list */ sl1 = reference_indices(r1so);
      /* list */ sl2 = reference_indices(r2so);
      int i1 = gen_length(sl1);
      int i2 = gen_length(sl2);

      i = i2>i1? 1 : (i2<i1? -1 : 0);

      if(i==0) {
	list sli1 = reference_indices(r1si);
	list sli2 = reference_indices(r2si);
	int i1 = gen_length(sli1);
	int i2 = gen_length(sli2);

	i = i2>i1? 1 : (i2<i1? -1 : 0);
	if(i==0) {
	  for(;i==0 && !ENDP(sl1); POP(sl1), POP(sl2)){
	    expression se1 = EXPRESSION(CAR(sl1));
	    expression se2 = EXPRESSION(CAR(sl2));
	    if(expression_constant_p(se1) && expression_constant_p(se2)){
	      int i1 = expression_to_int(se1);
	      int i2 = expression_to_int(se2);
	      i = i2>i1? 1 : (i2<i1? -1 : 0);
	      if(i==0){
		for(;i==0 && !ENDP(sli1); POP(sli1), POP(sli2)){
		  expression sei1 = EXPRESSION(CAR(sli1));
		  expression sei2 = EXPRESSION(CAR(sli2));
		  if(expression_constant_p(sei1) && expression_constant_p(sei2)){
		    int i1 = expression_to_int(sei1);
		    int i2 = expression_to_int(sei2);
		    i = i2>i1? 1 : (i2<i1? -1 : 0);
		  }else{
		    string s1 = words_to_string(words_expression(se1, NIL));
		    string s2 = words_to_string(words_expression(se2, NIL));
		    i = strcmp(s1, s2);
		  }
		}
	      }
	    }else{
	      string s1 = words_to_string(words_expression(se1, NIL));
	      string s2 = words_to_string(words_expression(se2, NIL));
	      i = strcmp(s1, s2);
	    }
	  }
	}
      }
    }
  }
  return i;
}

/* merge two points-to sets; required to compute
   the points-to set of the if control statements. */
set merge_points_to_set(set s1, set s2) {
  set Definite_set = set_generic_make(set_private, points_to_equal_p,
				      points_to_rank);
  set Possible_set = set_generic_make(set_private, points_to_equal_p,
				      points_to_rank);
  set Intersection_set = set_generic_make(set_private, points_to_equal_p,
					  points_to_rank);
  set Union_set = set_generic_make(set_private, points_to_equal_p,
				   points_to_rank);
  set Merge_set = set_generic_make(set_private, points_to_equal_p,
				   points_to_rank);

  Intersection_set = set_intersection(Intersection_set, s1, s2);
  Union_set = set_union(Union_set, s1, s2);

  SET_FOREACH(points_to, i, Intersection_set) {
    if(approximation_tag(points_to_approximation(i)) == 2)
      Definite_set = set_add_element(Definite_set,Definite_set,
				     (void*) i );
  }

  SET_FOREACH(points_to, j, Union_set) {
    if(! set_belong_p(Definite_set, (void*)j)) {
      points_to pt = make_points_to(points_to_source(j), points_to_sink(j),
				    make_approximation_may(),
				    make_descriptor_none());
      Possible_set = set_add_element(Possible_set, Possible_set,(void*) pt);
    }
  }

  Merge_set = set_clear(Merge_set);
  Merge_set = set_union(Merge_set, Possible_set, Definite_set);

  return Merge_set;
}

set exact_to_may_points_to_set(set s)
{
  SET_FOREACH(points_to, pt, s){
    if(approximation_exact_p(points_to_approximation(pt)))
      points_to_approximation(pt) = make_approximation_may();
  }
  return s;
}

/* storing the points to associate to a statement s, in the case of
   loops the field store is set to false to prevent from key
   redefenition's */
void points_to_storage(set pts_to_set, statement s, bool store) {
  list pt_list = NIL;
  if(statement_unstructured_p(s) || statement_goto_p(s)){
    /* store = false; */
    points_to_list new_pt_list = make_points_to_list(pt_list);
    store_or_update_pt_to_list(s, new_pt_list);
  }
  if (!set_empty_p(pts_to_set) && store == true) {
    instruction i = statement_instruction(s);
    if(instruction_sequence_p(i)){
      sequence seq = instruction_sequence(i);
	pt_list = set_to_sorted_list(pts_to_set,
				     (int(*)(const void*, const void*))
				     points_to_compare_cells);
	points_to_list new_pt_list = make_points_to_list(pt_list);
	store_or_update_pt_to_list(s, new_pt_list);
      FOREACH(statement, stm, sequence_statements(seq)){
	pt_list = set_to_sorted_list(pts_to_set,
				     (int(*)(const void*, const void*))
				     points_to_compare_cells);
	points_to_list new_pt_list = make_points_to_list(pt_list);
	store_or_update_pt_to_list(stm, new_pt_list);
      }
    }
    else{
    pt_list = set_to_sorted_list(pts_to_set,
				 (int(*)(const void*, const void*))
				 points_to_compare_cells);
    points_to_list new_pt_list = make_points_to_list(pt_list);
    store_or_update_pt_to_list(s, new_pt_list);
    }
  }
  else if(set_empty_p(pts_to_set)){
    points_to_list new_pt_list = make_points_to_list(pt_list);
    store_or_update_pt_to_list(s, new_pt_list);
  }
}


/* one basic case of Emami: < x = y[i] >. ne specific treatment is
   yet implemented, later we should make a difference between two
   cases : i =0 and i > 0.  */
set basic_ref_array(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference r1 = reference_undefined;
  reference r2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;

  ifdebug(1) printf("\n cas x = y[i] \n");

  effect e1 = effect_undefined, e2 = effect_undefined;
  /*init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs, 
								 &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
								 &e2, false);
  effects_free(l1);
  effects_free(l2);
  /* transform tab[i] into tab[*]. */
  list l3 = effect_to_store_independent_sdfi_list(e2, false);
  generic_effects_reset_all_methods();
  r1 = effect_any_reference(e1);
  source = make_cell_reference(r1);
  // add the points_to relation to the set generated
  // by this assignment
  e2 = EFFECT(CAR(l3));
  /* transform tab[*] into tab[*][0]. */
  effect_add_dereferencing_dimension(e2);
  r2 = effect_any_reference(copy_effect(e2));
  free(l3);
  sink = make_cell_reference(copy_reference(r2));
  new_sink = make_cell_reference(copy_reference(r2));
  rel = make_approximation_exact();
  pt_to = make_points_to(source, new_sink, rel,
			 make_descriptor_none());
  gen_pts_to = set_add_element(gen_pts_to, gen_pts_to, (void*) pt_to);

  // creation of the written set
  // search of all the points_to relations in the
  // alias set where the source is equal to the lhs
  SET_FOREACH(points_to, k, pts_to_set) {
    if(locations_equal_p(points_to_source(k), source))
      written_pts_to = set_add_element(written_pts_to,
				       written_pts_to, (void *)k);
  }

  pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
  ifdebug(1)
    print_points_to_set("Points to pour le cas 1 <x = y>\n",
			pts_to_set);
 
  return pts_to_set;
}

/* one basic case of Emami: < x = &y >, x is a pointer, y is a
   constant reference. */
set basic_ref_addr_emami(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1) {
    pips_debug(1, " case x = &y\n");
  }

  // creation of the source
  effect e1 = effect_undefined, e2 = effect_undefined;
  /*init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs, &e1,
								 true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs, &e2,
								 false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(copy_reference(ref1));
  // creation of the sink
  ref2 = effect_any_reference(e2);
  sink = make_cell_reference(copy_reference(ref2));
  // creation of the approximation
  rel = make_approximation_exact();
  // creation of the points_to relation
  pt = make_points_to(copy_cell(source), copy_cell(sink), rel,
		      make_descriptor_none());
  // add the points_to relation to the set generated
  //by this assignement
  gen_pts_to = set_add_element(gen_pts_to, gen_pts_to, (void*) pt);
  // creation of the written set
  // search of all the points_to relations in the
  // alias set where the source is equal to the lhs
  SET_FOREACH(points_to, i,pts_to_set) {
    if (locations_equal_p(points_to_source(i), source))
      written_pts_to = set_add_element(written_pts_to, written_pts_to,
				       (void *) i);
  }
  pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
  pts_to_set = set_union(pts_to_set, pts_to_set, gen_pts_to);
  ifdebug (1) {
    print_points_to_set("points-to for case 2 <x = &y> \n ", pts_to_set);
  }
 
  return pts_to_set;
}


/* to compute m.a = n.a where a is of pointer type*/
set struct_double_pointer(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  effect e1 = effect_undefined, e2 = effect_undefined;
  reference r1 = reference_undefined;
  reference r2 = reference_undefined;
  if (expression_reference_p(lhs) && expression_reference_p(rhs)) {
    /* init the effect's engine. */
    set_methods_for_proper_simple_effects();
    list l1 = generic_proper_effects_of_complex_address_expression(lhs,
								   &e1, true);
    list l2 = generic_proper_effects_of_complex_address_expression(rhs,
								   &e2, false);
    effects_free(l1);
    effects_free(l2);
    generic_effects_reset_all_methods();
    r1= effect_any_reference(e1);
    cell source = make_cell_reference(r1);
    r2 = effect_any_reference(e2);
    cell sink = make_cell_reference(r2);
    set s = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);

    SET_FOREACH(points_to, i, pts_to_set){
	if(locations_equal_p(points_to_source(i), sink))
	  s = set_add_element(s, s, (void*)i);
      }
    
    SET_FOREACH(points_to, j, s){
	cell new_sink = copy_cell(points_to_sink(j));
	// locations new_source = copy_access(source);
	approximation rel = points_to_approximation(j);
	points_to pt_to = make_points_to(source,
					 new_sink,
					 rel,
					 make_descriptor_none());
	gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
				     (void*) pt_to );
      }
    
    SET_FOREACH(points_to, k, pts_to_set){
      if(locations_equal_p(points_to_source(k), source))
	written_pts_to = set_add_element(written_pts_to,
					 written_pts_to, (void *)k);
    }

    pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
    pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
    ifdebug    (1)
      print_points_to_set("Points to for the case x is a double pointer to struct\n",
			  pts_to_set);
  } else
    pips_internal_error("Don't know how to handle this kind of expression");

  return pts_to_set;

}
/* to compute m.a = n.a where a is of pointer type*/
set struct_pointer(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  effect e1 = effect_undefined, e2 = effect_undefined;
  reference r1 = reference_undefined;
  reference r2 = reference_undefined;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs, &e1,
								 true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs, &e2,
								 false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  r1 = effect_any_reference(e1);
  // print_reference(ref1);
  cell source = make_cell_reference(r1);
  // add the points_to relation to the set generated
  // by this assignement
  r2 = effect_any_reference(e2);
  // print_reference(ref2);
  cell sink = make_cell_reference(r2);
  set s = set_generic_make(set_private, points_to_equal_p, points_to_rank);

  SET_FOREACH(points_to, i, pts_to_set){
      if(locations_equal_p(points_to_source(i), sink))
	s = set_add_element(s, s, (void*)i);
    }
  
  SET_FOREACH(points_to, j, s){
      cell new_sink = copy_cell(points_to_sink(j));
      // locations new_source = copy_access(source);
      approximation rel = points_to_approximation(j);
      points_to pt_to = make_points_to(source,
				       new_sink,
				       rel,
				       make_descriptor_none());

      gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
				   (void*) pt_to );
    }

  SET_FOREACH(points_to, k, pts_to_set) {
    if(locations_equal_p(points_to_source(k), source))
      written_pts_to = set_add_element(written_pts_to,
				       written_pts_to, (void *)k);
  }

  pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
  ifdebug(1)
    print_points_to_set("Points to for the case <x = y>\n",
			pts_to_set);

  return pts_to_set;

}

// to decompose the assignment m = n where  m and n are respectively
// of type struct
// the result should be m.field1 = n.field2... A.M
set struct_decomposition(expression lhs, expression rhs, set pt_in) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  //pt_out = set_assign(pt_out, pt_in);
  entity e1 = expression_to_entity(lhs);
  entity e2 = expression_to_entity(rhs);
  type t1 = entity_type(e1);
  type t2 = entity_type(e2);
  variable v1 = type_variable(t1);
  variable v2 = type_variable(t2);
  basic b1 = variable_basic(v1);
  basic b2 = variable_basic(v2);
  entity ent1 = basic_derived(b1);
  entity ent2 = basic_derived(b2);
  type tt1 = entity_type(ent1);
  type tt2 = entity_type(ent2);
  list l1 = type_struct(tt1);
  list l2 = type_struct(tt2);
  FOREACH(ENTITY, i, l1) {
    if(expression_double_pointer_p(entity_to_expression(i))
       || expression_pointer_p(entity_to_expression(i))) {
      ent2 = ENTITY (CAR(l2));
      expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
				      lhs,
				      entity_to_expression(i));
      expression ex2 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
				      rhs,
				      entity_to_expression(ent2));
      expression_consistent_p(ex1);
      expression_consistent_p(ex1);
      pt_out = set_union(pt_out, pt_out,
			 struct_pointer(pt_in,
					copy_expression(ex1),
					copy_expression(ex2)));
    }
    l2 = CDR(l2);
  }
  return pt_out;
}


/* Using pt_in, compute the new points-to set for any assignment lhs =
   rhs  that meets Emami's patterns.

   If the assignment cannot be analyzed according to Emami's rules,
   returns an empty set. So the assignment can be treated by
   points_to_general_assignment().

   To be able to apply Emami rules we have to test the lhs and the rhs:
   are they references, fields of structs, &operator...lhs and rhs can
   be one of Emami enum's fields.
*/
set points_to_assignment(statement current, expression lhs, expression rhs,
			 set in) {
  set cur = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set incur = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set in_may = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set in_must = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set kill_must = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set kill_may = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set gen_must = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set gen_may = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set out = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set kill = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set gen = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set out1 = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set out2 = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  list L = NIL, R = NIL, args = NIL;
  bool address_of_p = false;
   
  if (instruction_expression_p(statement_instruction(current))) {
    //expression e = instruction_expression(statement_instruction(current));
    ;
  }


  /* Compute side effects of rhs and lhs */
  set_assign(cur, points_to_expression(rhs, in, true));
  set_assign(incur, points_to_expression(lhs, cur, true));
  /*Extract MAY/MUST points to relations from the input set*/
  set_assign(in_may, points_to_may_filter(in));
  set_assign(in_must, points_to_must_filter(in));
  /*Change lhs into a contant memory path*/
  L = expression_to_constant_paths(lhs,incur);
  /* rhs should be a lvalue, if not we should transform it into a
     lvalue or call the adequate function according to its type */
  if(array_argument_p(rhs)){
    R = array_to_constant_paths(rhs,incur);
    if(!expression_pointer_p(rhs))
      address_of_p = true;
  }else if(expression_reference_p(rhs)){
    if (array_argument_p(rhs)){
      R = array_to_constant_paths(rhs, incur);
      if(!expression_pointer_p(rhs))
	address_of_p = true;
    }
    /* scalar case, rhs is already a lvalue */
    entity e = expression_to_entity(rhs);
    type tt = ultimate_type(entity_type(e));
    if(same_string_p(entity_local_name(e),"NULL")){
      entity ne = entity_null_locations();
      reference nr = make_reference(ne, NIL);
      cell nc = make_cell_reference(nr);
      R = CONS(CELL, nc, NIL);
      address_of_p = true;
    }else if(entity_variable_p(e)){
      R = expression_to_constant_paths(rhs,incur);
      basic b = variable_basic(type_variable(tt));
      if(basic_pointer_p(b)){
	basic bt = variable_basic(type_variable(basic_pointer(b)));
	if(basic_typedef_p(bt)){
	  basic ulb = basic_ultimate(bt);
	  if(basic_derived_p(ulb)){
	   
	    return points_to_pointer_to_derived_assignment(current, lhs, rhs, incur);
	  }
	}else if(basic_derived_p(bt))
	  return points_to_derived_assignment(current, lhs, rhs, in);
	/* else */
	/* 	  R = expression_to_constant_paths(rhs,incur); */
      }
    }
  
    /* (derived_entity_p(e) || typedef_entity_p(e)) */
    /*       return points_to_derived_assignment(current, lhs, rhs, in); */
    else
      R = expression_to_constant_paths(rhs,incur);
  }else if(expression_null_locations_p(rhs)){
    /* NULL or zero case*/

  }
  else if (expression_cast_p(rhs)){
    expression nrhs = cast_expression(expression_cast(rhs));
    return points_to_assignment(current, lhs, nrhs, incur);
  }
  else if(expression_equal_integer_p(rhs, 0)){
    entity ne = entity_null_locations();
    reference nr = make_reference(ne, NIL);
    cell nc = make_cell_reference(nr);
    R = CONS(CELL, nc, NIL);
    address_of_p = true;

  }
  else if(assignment_expression_p(rhs)){
    call c = expression_call(rhs);
    args = call_arguments(c);
    expression nrhs = EXPRESSION(CAR(args));
    return  points_to_assignment(current, lhs, nrhs, incur);
  }
  else if(comma_expression_p(rhs)){
    /* comma case, lhs should point to the same location as the last
       pointer which appears into comma arguments*/
    call c = expression_call(rhs);
    args = call_arguments(c);
    expression nrhs = expression_undefined;
    FOREACH(expression, ex, args){
      incur = set_assign(incur, points_to_expression(ex, incur, true));
      nrhs = copy_expression(ex);
    }
    return  points_to_assignment(current, lhs, nrhs, incur);

  }
 /*  else if(expression_field_p(rhs)){ */
/*     /\* case data structure's field  *\/ */

/*   } */
  else if(address_of_expression_p(rhs)){
    /* case & opeator */
    call c = expression_call(rhs);
    args = call_arguments(c);
    expression nrhs = EXPRESSION(CAR(args));
   /*  set_assign(R, expression_to_constant_paths(nrhs,incur)); */
    if(array_argument_p(nrhs))
      R = array_to_constant_paths(nrhs,incur);
    else
    R = expression_to_constant_paths(nrhs,incur);
    address_of_p = true;

  }
  else if(subscript_expression_p(rhs)){
    /* case [] */
  /*   syntax sy = expression_syntax(rhs); */
/*     subscript sb = syntax_subscript(sy); */
/*     rhs = subscript_array(sb); */
    R = expression_to_constant_paths(rhs, incur);
  }
  else if(operator_expression_p(rhs, POINT_TO_OPERATOR_NAME)){
    /* case -> operator */
    R = expression_to_constant_paths(rhs, incur);
  }
  else if(expression_field_p(rhs)){
    R = expression_to_constant_paths(rhs,incur);
  }
  else if(operator_expression_p(rhs, DEREFERENCING_OPERATOR_NAME)){
    R = expression_to_constant_paths(rhs,incur);
  }
  else if(operator_expression_p(rhs, C_AND_OPERATOR_NAME)){
    /* case && operator */

  }
  else if(operator_expression_p(rhs, C_OR_OPERATOR_NAME)){
    /* case || operator */

  }
  else if(operator_expression_p(rhs, CONDITIONAL_OPERATOR_NAME)){
    /* case ? operator is similar to an if...else instruction */
    call c = expression_call(rhs);
    args = call_arguments(c);
    expression cond = EXPRESSION(CAR(args));
    expression arg1 = EXPRESSION(CAR(CDR(args)));
    expression arg2 = EXPRESSION(CAR(CDR(CDR(args))));
    incur = points_to_expression(cond, incur, true);
    out1 = points_to_assignment(current, lhs, arg1, incur);
    out2 = points_to_assignment(current,lhs, arg2, incur);
    return merge_points_to_set(out1, out2);
  }
  else if(expression_call_p(rhs)){
    if(ENTITY_MALLOC_SYSTEM_P(expression_to_entity(rhs)) ||
       ENTITY_CALLOC_SYSTEM_P(expression_to_entity(rhs))){
      expression sizeof_exp = EXPRESSION (CAR(call_arguments(expression_call(rhs))));
      type t = expression_to_type(lhs);
      reference nr = original_malloc_to_abstract_location(lhs,
							  t,
							  type_undefined,
							  sizeof_exp,
							  get_current_module_entity(),
							  current);
      cell nc = make_cell_reference(nr);
      R = CONS(CELL, nc, NIL);
      address_of_p = true;
    }
    else if(user_function_call_p(rhs)){
      type t = entity_type(call_function(expression_call(rhs)));
      entity ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
     /*  entity ne = entity_all_locations(); */
      reference nr = make_reference(ne, NIL);
      cell nc = make_cell_reference(nr);
      R = CONS(CELL, nc, NIL);
      address_of_p = true;
    }
    else{
     /*  entity ne = entity_all_locations(); */
      type t = entity_type(call_function(expression_call(rhs)));
      entity ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
      reference nr = make_reference(ne, NIL);
      cell nc = make_cell_reference(nr);
      R = CONS(CELL, nc, NIL);
      address_of_p = true;
    }
  }
  else{
    type t = expression_to_type(rhs);
    entity ne = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
   /*  entity ne = entity_all_locations(); */
    reference nr = make_reference(ne, NIL);
    cell nc = make_cell_reference(nr);
    R = CONS(CELL, nc, NIL);
    address_of_p = true;
  }
  /* set_assign(kill_may, kill_may_set(L, in_may)); */
  set_assign(kill_may, kill_may_set(L, in_may));
  set_assign(kill_must, kill_must_set(L, in));
  set_assign(gen_may, gen_may_set(L, R, in_may, &address_of_p));
  set_assign(gen_must, gen_must_set(L, R, in_must, &address_of_p));
  set_union(kill, kill_may, kill_must);
  set_union(gen, gen_may, gen_must);
  if(set_empty_p(gen))
    set_assign(gen, points_to_anywhere_typed(L, incur));
  set_difference(in, in, kill);
  set_union(out, in, gen);
  return out;
}


set points_to_pointer_to_derived_assignment(statement current, expression lhs, expression rhs, set in){

  set out = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  set cur = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set incur = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set in_may = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set in_must = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set kill_must = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set kill_may = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set gen_must = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set gen_may = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set kill = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  set gen = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  list L = NIL, R = NIL;
  bool address_of_p = false;

  set_assign(in_may, points_to_may_filter(in));
  set_assign(in_must, points_to_must_filter(in));
 
  entity el = expression_to_entity(lhs);
  if(entity_variable_p(el)){
    basic b = variable_basic(type_variable(entity_type(el)));
    if(basic_pointer_p(b)){
      set_assign(cur, points_to_expression(rhs, in, true));
      set_assign(incur, points_to_expression(lhs, cur, true));
      R = expression_to_constant_paths(rhs,incur);
      L = expression_to_constant_paths(lhs,incur);
      set_assign(kill_may, kill_may_set(L, in_may));
      set_assign(kill_must, kill_must_set(L, in_must));
      set_assign(gen_may, gen_may_set(L, R, in_may, &address_of_p));
      set_assign(gen_must, gen_must_set(L, R, in_must, &address_of_p));
      set_union(kill, kill_may, kill_must);
      set_union(gen, gen_may, gen_must);
      if(set_empty_p(gen))
	set_assign(gen, points_to_anywhere_typed(L, incur));
      set_difference(in, in, kill);
      set_union(out, in, gen);

      basic bt = variable_basic(type_variable(basic_pointer(b)));
      if(basic_typedef_p(bt)){
	basic ulb = basic_ultimate(bt);
	if(basic_derived_p(ulb)){
	  entity e1  = basic_derived(ulb);
	  type t1 = entity_type(e1);
	  if(type_struct_p(t1)){
	    entity rl = expression_to_entity(rhs);
	    basic b1 = variable_basic(type_variable(entity_type(rl)));
	    basic bt1 = variable_basic(type_variable(basic_pointer(b1)));
	    basic ulb1 = basic_ultimate(bt1);
	    entity e2  = basic_derived(ulb1);
	    type t2 = entity_type(e2);
	    list l_lhs = type_struct(t1);
	    list l_rhs = type_struct(t2);
	    for (; !ENDP(l_lhs) &&!ENDP(l_rhs)  ; POP(l_lhs), POP(l_rhs)){
	      expression ex1 = entity_to_expression(ENTITY(CAR(l_lhs)));
	      expression nlhs = MakeBinaryCall(entity_intrinsic(POINT_TO_OPERATOR_NAME),
					       lhs,
					       ex1);
	      expression ex2 = entity_to_expression(ENTITY(CAR(l_rhs)));
	      expression nrhs = MakeBinaryCall(entity_intrinsic(POINT_TO_OPERATOR_NAME),
					       rhs,
					       ex2);
	      if(expression_pointer_p(nlhs))
		set_assign(out, points_to_assignment(current, nlhs, nrhs, out));
	    }
	  }
	  else if(type_union_p(t1))
	    pips_user_warning("union case not handled yet\n");
	  else if(type_enum_p(t1))
	    pips_user_warning("enum case not handled yet\n");
	}
      }
    }
  }
  return out;
}


set points_to_derived_assignment(statement current, expression lhs, expression rhs, set in){

  set out = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  set_assign(out, in);
  entity el = expression_to_entity(lhs);
  if(entity_variable_p(el)){
    basic b = variable_basic(type_variable(entity_type(el)));
    if(basic_typedef_p(b)){
      basic ulb = basic_ultimate(b);
      if(basic_derived_p(ulb)){
	entity e1  = basic_derived(ulb);
	type t1 = entity_type(e1);
	if(type_struct_p(t1)){
	  if(expression_reference_p(rhs) || expression_pointer_p(rhs)){
	    entity rl = expression_to_entity(rhs);
	    if(entity_variable_p(rl)){
	      basic b1 = variable_basic(type_variable(entity_type(rl)));
	      basic ulb1 = basic_ultimate(b1);
	      if(basic_derived_p(ulb1)){
		entity e2  = basic_derived(ulb1);
		type t2 = entity_type(e2);
		list l_lhs = type_struct(t1);
		list l_rhs = type_struct(t2);
		for (; !ENDP(l_lhs) &&!ENDP(l_rhs)  ; POP(l_lhs), POP(l_rhs)){
		  expression ex1 = entity_to_expression(ENTITY(CAR(l_lhs)));
		  expression nlhs = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
						   lhs,
						   ex1);
		  expression ex2 = entity_to_expression(ENTITY(CAR(l_rhs)));
		  expression nrhs = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
						   rhs,
						   ex2);
		  if(expression_pointer_p(nlhs))
		    set_assign(out, points_to_assignment(current, nlhs, nrhs, in));
		}
	      }
	    }
	  }
	}
	else if(type_union_p(t1))
	  pips_user_warning("union case not handled yet\n");
	else if(type_enum_p(t1))
	  pips_user_warning("enum case not handled yet\n");
      }
    }
  }

  return  out;
}


/* change tab[i] into tab[*] .*/
cell get_array_path(expression e)
{
  effect ef = effect_undefined;
  reference  r = reference_undefined;
  cell c = cell_undefined;

  /*init the effect's engine*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(e, &ef,
								 true);
  list l2 = effect_to_store_independent_sdfi_list(ef, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  r = effect_any_reference(ef);
  c = make_cell_reference(r);
  return c;

}


/* Input : a cell c
   Output : side effect on c
   This function changes array element b[i] into b[*],
   it takes care of initializing the effect engine.
*/
cell array_to_store_independent(cell c)
{
  reference r = cell_reference(c);
  expression e = reference_to_expression(r);
  effect e1 = effect_undefined;
  /*init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(e,
								 &e1, false);
  effects_free(l1);
  list l2 = effect_to_store_independent_sdfi_list(e1, false);
  e1 = EFFECT(CAR(l2));
  r = effect_any_reference(e1);
  effects_free(l2);
  cell c1 = make_cell_reference(r);
  generic_effects_reset_all_methods();
  cell_consistent_p(c1);
  return c1;
}

/* Input : a cell c
   Output : side effect on c
   This function changes array element b[i] into b[0],
   it takes care of initializing the effect engine.
*/
cell add_array_dimension(cell c)
{
  reference r = cell_reference(c);
  expression e = reference_to_expression(r);
  effect e1 = effect_undefined;
  /*init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(e,
								 &e1, true);
  effect_add_dereferencing_dimension(e1);
  effects_free(l1);
  generic_effects_reset_all_methods();
  reference r1 = effect_any_reference(e1);
  cell c1 = make_cell_reference(r1);
  return c1;
}




/* iterate over the lhs_set, if it contains more than an element
   approximations are set to MAY, otherwise it's set to EXACT
   Set the sink to null value .*/
set points_to_null_pointer(list lhs_list, set input)
{
  /* lhs_path matches the kill set.*/
  set kill = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set gen = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set res= set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set input_kill_diff = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  approximation a = make_approximation_exact();
  SET_FOREACH(points_to, p, input)
    {
      FOREACH(cell, c, lhs_list)
	{
	  if(cell_equal_p(points_to_source(p),c))
	    set_add_element(kill,kill,(void*) p);
	}
    }

  /* input - kill */
  set_difference(input_kill_diff, input, kill);

  /* if the lhs_set or the rhs set contains more than an element, we
     set the approximation to MAY. */
  if(gen_length(lhs_list) > 1)// || set_size(rhs_set)>1)
    a = make_approximation_may();

  /* Computing the gen set*/
  FOREACH(cell, c, lhs_list){
      /* we test if lhs is an array, so we change a[i] into a[*] */
      if(array_type_p(cell_to_type(c)))
	c = array_to_store_independent(c);

      /* create a new points to with as source the current
	 element of lhs_set and sink the null value .*/
      entity e =entity_all_xxx_locations_typed(NULL_POINTER_NAME,cell_to_type(c));
      reference r = make_reference(e, NIL);
      cell sink = make_cell_reference(r);
      points_to pt_to = make_points_to(c, sink, a, make_descriptor_none());
      set_add_element(gen, gen, (void*) pt_to);

    }
  /* gen + input_kill_diff*/
  set_union(res, gen, input_kill_diff);

  return res;
}





set points_to_general_assignment(__attribute__ ((__unused__)) statement s,
				 expression lhs,
				 expression rhs,
				 set pts_in,
				 __attribute__ ((__unused__)) list l)
{
  set res = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  list lhs_list = NIL, rhs_list = NIL;
  bool nowhere_lhs_p = false, nowhere_rhs_p = false;

/* we test the type of lhs by using expression_pointer_p(), I'm  not
   sure if it's cover all the possibilitie, to be checked later ... */
  if(expression_pointer_p(lhs)) {
     /* we call expression_to_constant_path() which calls
	1- get_memory_path() to change **p into p[0][0]
        2- eval_cell_with_points_to() to eval the memory path using
	points to
        3- possible_constant_path_list() which depending in
	eval_cell_with_points_to() result's computes a constant path
	or an nowhere points to*/
    lhs_list = expression_to_constant_paths(lhs, pts_in); //, &nowhere_lhs_p);
    if(nowhere_lhs_p) {
      //return lhs_list;
      ;
    }
     /* Now treat the rhs, the call to & requires a special treatment */
     syntax s = expression_syntax(rhs);
     if(syntax_call_p(s)){
       call c = syntax_call(s);
       /* if it's a call to &, replace rhs by the first argument of & */
       if (entity_an_operator_p(call_function(c), ADDRESS_OF)){
	 rhs = EXPRESSION(CAR(call_arguments(c)));
	 rhs_list = expression_to_constant_paths(rhs, pts_in);//, &nowhere_rhs_p);
	 if(nowhere_rhs_p)
	   set_assign(res, points_to_nowhere_typed(lhs_list, pts_in));
	 else{
	   /* change basic_ref_addr into basic_ref_addr_emami*/
	   //set_assign(res, basic_ref_addr_emami(lhs_set, rhs_set,
	   //pts_in));
	   ;
	 }
       }
     }else if(syntax_cast_p(s)){
       set_assign(res, points_to_null_pointer(lhs_list, pts_in));
     }else if(expression_reference_p(rhs)){
       rhs_list = expression_to_constant_paths(rhs, pts_in);//, &nowhere_rhs_p);
       if(nowhere_rhs_p)
	 set_assign(res, points_to_nowhere_typed(lhs_list, pts_in));
       else{
	 /* change basic_ref_ref() into basic_ref_ref_emami()*/
	 //set_assign(res, basic_ref_ref_emami(lhs_set, rhs_set,
	 //pts_in));
	 ;
       }
     }else if(array_argument_p(rhs)){
       ;
     }
  }
  return res;
}




/* compute the points to set associate to a sequence of statements*/
set points_to_sequence(sequence seq, set pt_in, bool store) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  list dl = NIL;
   set_assign(pt_out, pt_in);
   FOREACH(statement, st, sequence_statements(seq)) {
     statement_consistent_p(st);
     set_assign(pt_out, points_to_recursive_statement(st,pt_out,store));
     if(statement_block_p(st) && !ENDP(dl=statement_declarations(st)))
	pt_out = points_to_projection(pt_out, dl);
   }
  
  return pt_out;
}

/* compute the points-to set for an intrinsic call */
set points_to_intrinsic(statement s, call c __attribute__ ((__unused__)), entity e, list pc, set pt_in,
			list el) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set pt_cur = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  expression lhs = expression_undefined;
  expression rhs = expression_undefined;
  pips_debug(8, "begin\n");

  /* Recursive descent on subexpressions for cases such as "p=q=t=u;" */
  set_assign(pt_cur, pt_in);
  FOREACH(EXPRESSION, ex, pc) {
    pt_cur = set_assign(pt_cur, points_to_expression(copy_expression(ex),
						     pt_cur, true));
  }
  if (ENTITY_ASSIGN_P(e)) {
    lhs = EXPRESSION(CAR(pc));
    rhs = EXPRESSION(CAR(CDR(pc)));
    /* test if lhs is a pointer */
    set_assign(pt_out, pt_cur);
   
      if(expression_pointer_p(lhs))
	{
	set_assign(pt_out,points_to_assignment(s, copy_expression(lhs),
					       copy_expression( rhs), pt_cur));
	if (set_empty_p(pt_out))
	  /* Use effects to remove broken points-to relations and to link
	     broken pointers towards default sinks.
	     pt_out = points_to_filter_with_expression_effects(e, pt_cur);
	  */
	  pt_out = points_to_general_assignment(s, copy_expression(lhs),copy_expression(rhs), pt_cur, el);
	}
      else if(expression_reference_p(lhs))
	{
	
	entity e = expression_to_entity(lhs);
	type t = entity_type(e);
	if(entity_variable_p(e))
	  {
	  basic b = variable_basic(type_variable(t));
	  if(basic_typedef_p(b))
	    {
	    basic ulb = basic_ultimate(b);
	    if(basic_derived_p(ulb))
	      set_assign(pt_out,points_to_derived_assignment(s, lhs, rhs, pt_cur));
	    }
	  }
	}
  }
  else if (/* ENTITY_PLUS_UPDATE_P(e) ||  ENTITY_MINUS_UPDATE_P(e)
	      ||*/ ENTITY_MULTIPLY_UPDATE_P(e) || ENTITY_DIVIDE_UPDATE_P(e)
	   || ENTITY_MODULO_UPDATE_P(e) || ENTITY_LEFT_SHIFT_UPDATE_P(e)
	   || ENTITY_RIGHT_SHIFT_UPDATE_P(e) || ENTITY_BITWISE_AND_UPDATE_P(e)
	   || ENTITY_BITWISE_XOR_UPDATE_P(e) || ENTITY_BITWISE_OR_UPDATE_P(e)) {
    /* Look at the lhs with
       generic_proper_effects_of_complex_address_expression(). If the
       main effect is an effect on a pointer, occurences of this
       pointer must be removed from pt_cur to build pt_out.
    */

    pt_out = points_to_filter_with_effects(pt_cur, el);
  } else if (ENTITY_POST_INCREMENT_P(e) || ENTITY_POST_DECREMENT_P(e)
	     || ENTITY_PRE_INCREMENT_P(e) || ENTITY_PRE_DECREMENT_P(e)) {
    lhs = EXPRESSION(CAR(pc));
    pt_out = set_assign(pt_out, points_to_post_increment(s, lhs, pt_cur, el ));
  }
  else if(ENTITY_PLUS_UPDATE_P(e) || ENTITY_MINUS_UPDATE_P(e)) {
    lhs = EXPRESSION(CAR(pc));
    rhs = EXPRESSION(CAR(CDR(pc)));
    pt_out = set_assign(pt_out, points_to_plus_update(s, lhs, rhs, pt_cur, el));

  }
  else if (ENTITY_STOP_P(e) || ENTITY_ABORT_SYSTEM_P(e)
	     || ENTITY_EXIT_SYSTEM_P(e) || ENTITY_C_RETURN_P(e)) {
    /* The call is never returned from. No information is available
       for the dead code that follows. pt_out is already set to the
       empty set. */
    set_assign(pt_out, pt_cur);
  }else if(ENTITY_AND_P(e)) {
    FOREACH(EXPRESSION, exp, pc) {
      pt_out = set_assign(pt_out, points_to_expression(copy_expression(exp),
						       pt_cur,
						       true));
    }
  } else if(ENTITY_OR_P(e)) {
    FOREACH(EXPRESSION, exp, pc) {
      pt_out = set_assign(pt_out, points_to_expression(copy_expression(exp),
						       pt_cur,
						       true));
    }
  }else if (ENTITY_COMMA_P(e)) {
    FOREACH(EXPRESSION, exp, pc) {
      pt_out = set_assign(pt_out, points_to_expression(copy_expression(exp),
						       pt_cur,
						       true));
    }
  }
  else {
    /*By default, use the expression effects to filter cur_pt */
    //if(!set_empty_p(pt_out))
    //pt_out =
    //set_assign(pt_out,points_to_filter_with_expression_effects(pt_cur,
    //effects));
    pt_out = points_to_filter_with_effects(pt_cur, el);
  }
  /* if pt_out != pt_cur, do not forget to free pt_cur... */
  pips_debug(8, "end\n");
 
  return pt_out;
}

/* input:
 *  a set of points-to pts
 *  a list of effects el
 *
 * output
 *  a updated set of points-to pts (side effects)
 *
 * Any pointer written in el does not point to its old target anymore
 * but points to any memory location. OK, this is pretty bad, but it
 * always is correct.
 */
set points_to_filter_with_effects(set pts, list el) {
  list lhs_list = NIL;

  FOREACH(EFFECT, e, el) {
    if(effect_pointer_type_p(e) && effect_write_p(e)) {

      cell c = effect_cell(e);
      /* Theheap problem with the future extension to GAP is hidden
	 within cell_to_reference */
      reference r = cell_to_reference(c);
      cell nc = make_cell_reference(r);
      if(ENDP(reference_indices(r))) {
	/* Simple case: a scalar pointer is written */
	/* Remove previous targets */
	SET_FOREACH(points_to, pt, pts) {
	  cell ptc = points_to_source(pt);
	  if(points_to_compare_cell(nc, ptc)){
	    if(! cell_in_list_p(ptc, lhs_list))
	      lhs_list = gen_concatenate(lhs_list, CONS(CELL, ptc, NIL));
	  }
	}
	set_assign(pts, points_to_anywhere_typed(lhs_list, pts));
      }
      else {
	/* Complex case: is the reference usable with the current pts?
	 * If it uses an indirection, check that the indirection is
	 * not thru nowhere, not thru NULL, and maybe not thru
	 * anywhere...
	 */
	/* Remove previous targets */
	SET_FOREACH(points_to, pt, pts) {
	  cell ptc = points_to_source(pt);
	  if(points_to_compare_cell(ptc, nc)){
	    if(! cell_in_list_p(ptc, lhs_list))
	      lhs_list = gen_concatenate(lhs_list,CONS(CELL, ptc, NIL));
	  }
	}

	/* add the anywhere points-to*/
	set_assign(pts, points_to_anywhere_typed(lhs_list, pts));
	//pips_internal_error("Complex pointer write effect."
	//" Not implemented yet\n");
      }
    }
  }
  return pts;
}

set points_to_plus_update(statement s, expression lhs, expression rhs, set pts, list eff_list)
{
  set res = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  list new_inds = NIL;
  if( expression_integer_constant_p(rhs)) {
    if(expression_reference_p(lhs)) {
      reference r = expression_reference(lhs);
      cell c = make_cell_reference(r);
      SET_FOREACH(points_to, pt, pts) {
	cell pt_source = points_to_source(pt);
	if(points_to_compare_cell(c, pt_source)) {
	  cell pt_sink = points_to_sink(pt);
	  reference ref_sink = cell_to_reference(pt_sink);
	  if(array_reference_p(ref_sink)) {
	    list l_ind = reference_indices(ref_sink);
	    FOREACH(EXPRESSION, exp, l_ind){
	      if(expression_integer_constant_p(exp)) {
		expression new_ind = make_unbounded_expression();
		new_inds = gen_nconc(CONS(EXPRESSION, new_ind, NIL),new_inds);
	      }
	      else
		new_inds = gen_nconc(new_inds, l_ind);
	    }
	    reference new_ref = make_reference(reference_variable(ref_sink),new_inds);
	    expression new_rhs = reference_to_expression(new_ref);
	    res =  set_assign(res, points_to_assignment( s, copy_expression(lhs), copy_expression(new_rhs), pts ));
	    new_inds = NIL;
	  }
	  else
	    res = set_assign(res, points_to_filter_with_effects(pts, eff_list));
	}
	else
	  res = set_assign(res, points_to_filter_with_effects(pts, eff_list));
      }
    }
  }
  else
    res = set_assign(res, points_to_filter_with_effects(pts, eff_list));

  return res;
}


set points_to_post_increment(statement s, expression lhs, set pts, list eff_list)
{
  set res = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  res = set_assign(res, pts);
  list new_inds = NIL;
  if(expression_reference_p(lhs)) 
    {
    reference r = expression_reference(lhs);
    cell c = make_cell_reference(r);
    SET_FOREACH(points_to, pt, pts) 
      {
      cell pt_source = points_to_source(pt);
      if(points_to_compare_cell(c, pt_source)) 
	{
	cell pt_sink = points_to_sink(pt);
	reference ref_sink = cell_to_reference(pt_sink);
	if(array_reference_p(ref_sink)) 
	  {
	  list l_ind = reference_indices(ref_sink);
	  FOREACH(EXPRESSION, exp, l_ind)
	    {
	    if(expression_integer_constant_p(exp)) 
	      {
	      expression new_ind = make_unbounded_expression();
	      new_inds = gen_nconc(CONS(EXPRESSION, new_ind, NIL),new_inds);
	      }
	    else
	      new_inds = gen_nconc( l_ind, new_inds);
	    }
	  reference new_ref = make_reference(reference_variable(ref_sink),new_inds);
	  expression new_rhs = reference_to_expression(new_ref);
	  res =  set_assign(res, points_to_assignment( s, copy_expression(lhs), copy_expression(new_rhs), res ));
	  new_inds = NIL;
	  }
	else
		res = set_assign(res, points_to_filter_with_effects(res, eff_list));
	}
      }
    }
  else
    res = set_assign(res, points_to_filter_with_effects(res, eff_list));



  return res;

}

bool cell_in_list_p(cell c, const list lx)
{
  list l = (list) lx;
  for (; !ENDP(l); POP(l))
    if (points_to_compare_cell(CELL(CAR(l)), c)) return true; /* found! */

  return false; /* else no found */
}

/* computing the points-to set of a while loop by iterating over its
   body until reaching a fixed-point. For the moment without taking
   into account the condition's side effect. */
set points_to_whileloop(whileloop wl, set pt_in, bool store __attribute__ ((__unused__))) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set cur = set_generic_make(set_private, points_to_equal_p,
				 points_to_rank);
  statement while_body = whileloop_body(wl);
  expression cond = whileloop_condition(wl);
  int i = 0;
  int k = 10;

  pt_in = set_assign(pt_in, points_to_expression(cond, pt_in, true));

  for(i = 0; i< k; i++){
    cur = set_assign(cur, set_clear(cur));
    cur = set_assign(cur, pt_in);
    set_clear(pt_out);
    set_assign(pt_out, points_to_recursive_statement(while_body,
						     cur,
						     false));
    SET_FOREACH(points_to, pt, pt_out){
      cell sc = points_to_source(pt);
      reference sr = cell_to_reference(sc);
      list sl = reference_indices(sr);
      cell kc = points_to_sink(pt);
      reference kr = cell_to_reference(kc);
      list kl = reference_indices(kr);
      if((int)gen_length(sl)>k){
	entity anywhere =entity_all_xxx_locations_typed(ANYWHERE_LOCATION,
					 cell_to_type(sc));
	reference r = make_reference(anywhere,NIL);
	sc = make_cell_reference(r);
      }
      if((int)gen_length(kl)>k){
	entity anywhere =entity_all_xxx_locations_typed(ANYWHERE_LOCATION,
					 cell_to_type(kc));
	reference r = make_reference(anywhere,NIL);
	kc = make_cell_reference(r);
      }
      points_to npt = make_points_to(sc, kc, points_to_approximation(pt), make_descriptor_none());
      if(!points_to_equal_p(npt,pt)){
      pt_out = set_del_element(pt_out, pt_out, (void*)pt);
      pt_out = set_add_element(pt_out, pt_out, (void*)npt);
      }
    }
    set_assign(pt_in, set_clear(pt_in));
    set_assign(pt_in, merge_points_to_set(cur, pt_out));
    if(set_equal_p(cur, pt_in))
      break;
  }
  points_to_storage(pt_in, while_body , true);
  set_assign(pt_in, points_to_independent_store(pt_in));
  return pt_in;
}

/* computing the points to of a for loop, before processing the body,
   compute the points to of the initialization. */
set points_to_forloop(forloop fl, set pt_in, bool store __attribute__ ((__unused__))) {
  statement for_body = forloop_body(fl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set cur = set_generic_make(set_private, points_to_equal_p,
				 points_to_rank);
  int i = 0;
  int k = 10;
  expression exp_ini = forloop_initialization(fl);
  expression exp_cond = forloop_condition(fl);
  expression exp_inc = forloop_increment(fl);

  pt_in = points_to_expression(exp_ini, pt_in, true);
  pt_in = points_to_expression(exp_cond, pt_in, true);
  pt_in = points_to_expression(exp_inc, pt_in, true);

  for(i = 0; i< k; i++){
    cur = set_assign(cur, set_clear(cur));
    cur = set_assign(cur, pt_in);
    set_clear(pt_out);
    set_assign(pt_out, points_to_recursive_statement(for_body,
						     cur,
						     false));
    SET_FOREACH(points_to, pt, pt_out){
      cell sc = points_to_source(pt);
      reference sr = cell_to_reference(sc);
      list sl = reference_indices(sr);
      cell kc = points_to_sink(pt);
      reference kr = cell_to_reference(kc);
      list kl = reference_indices(kr);
      if((int)gen_length(sl)>k){
	entity anywhere =entity_all_xxx_locations_typed(ANYWHERE_LOCATION,
							cell_to_type(sc));
	reference r = make_reference(anywhere,NIL);
	sc = make_cell_reference(r);
      }
      if((int)gen_length(kl)>k){
	entity anywhere =entity_all_xxx_locations_typed(ANYWHERE_LOCATION,
							cell_to_type(kc));
	reference r = make_reference(anywhere,NIL);
	kc = make_cell_reference(r);
      }
      points_to npt = make_points_to(sc, kc, points_to_approximation(pt), make_descriptor_none());
      if(!points_to_equal_p(npt,pt)){
	pt_out = set_del_element(pt_out, pt_out, (void*)pt);
	pt_out = set_add_element(pt_out, pt_out, (void*)npt);
      }
    }
    set_assign(pt_in, set_clear(pt_in));
    set_assign(pt_in, merge_points_to_set(cur, pt_out));
    if(set_equal_p(cur, pt_in))
      break;
  }
  points_to_storage(pt_in, for_body , true);
  set_assign(pt_in, points_to_independent_store(pt_in));
  return pt_in;
}

/* computing the points to of a  loop. to have more precise
 * information, maybe should transform the loop into a do loop by
 * activating the property FOR_TO_DO_LOOP_IN_CONTROLIZER. */
set points_to_loop(loop fl, set pt_in, bool store __attribute__ ((__unused__))) {
  statement loop_body = loop_body(fl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set cur = set_generic_make(set_private, points_to_equal_p,
				 points_to_rank);
  int i = 0;
  int k = 10;
  for(i = 0; i< k; i++){
    cur = set_assign(cur, set_clear(cur));
    cur = set_assign(cur, pt_in);
    set_clear(pt_out);
    set_assign(pt_out, points_to_recursive_statement(loop_body,
						     cur,
						     false));
    SET_FOREACH(points_to, pt, pt_out){
      cell sc = points_to_source(pt);
      reference sr = cell_to_reference(sc);
      list sl = reference_indices(sr);
      cell kc = points_to_sink(pt);
      reference kr = cell_to_reference(kc);
      list kl = reference_indices(kr);
      if((int)gen_length(sl)>k){
	entity anywhere =entity_all_xxx_locations_typed(ANYWHERE_LOCATION,
							cell_to_type(sc));
	reference r = make_reference(anywhere,NIL);
	sc = make_cell_reference(r);
      }
      if((int)gen_length(kl)>k){
	entity anywhere =entity_all_xxx_locations_typed(ANYWHERE_LOCATION,
							cell_to_type(kc));
	reference r = make_reference(anywhere,NIL);
	kc = make_cell_reference(r);
      }
      points_to npt = make_points_to(sc, kc, points_to_approximation(pt), make_descriptor_none());
      if(!points_to_equal_p(npt,pt)){
	pt_out = set_del_element(pt_out, pt_out, (void*)pt);
	pt_out = set_add_element(pt_out, pt_out, (void*)npt);
      }
    }
    set_assign(pt_in, set_clear(pt_in));
    set_assign(pt_in, merge_points_to_set(cur, pt_out));
    if(set_equal_p(cur, pt_in))
      break;
  }
  points_to_storage(pt_in, loop_body, true);
   set_assign(pt_in, points_to_independent_store(pt_in));
  return pt_in;
}

/*Computing the points to of a do while loop, we have to process the
  body a least once, before iterating until reaching the fixed-point. */
set points_to_do_whileloop(whileloop fl, set pt_in, bool store __attribute__ ((__unused__))) {
  statement dowhile_body = whileloop_body(fl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set cur = set_generic_make(set_private, points_to_equal_p,
				 points_to_rank);
  int i = 0;
  int k = 10;
  expression cond = whileloop_condition(fl);
  set_assign(pt_in, points_to_recursive_statement(dowhile_body,
						   pt_in, false));
  for(i = 0; i< k; i++){
    cur = set_assign(cur, set_clear(cur));
    cur = set_assign(cur, pt_in);
    set_clear(pt_out);
    set_assign(pt_out, points_to_recursive_statement(dowhile_body,
						     cur,
						     false));
    SET_FOREACH(points_to, pt, pt_out){
      cell sc = points_to_source(pt);
      reference sr = cell_to_reference(sc);
      list sl = reference_indices(sr);
      cell kc = points_to_sink(pt);
      reference kr = cell_to_reference(kc);
      list kl = reference_indices(kr);
      if((int)gen_length(sl)>k){
	entity anywhere =entity_all_xxx_locations_typed(ANYWHERE_LOCATION,
							cell_to_type(sc));
	reference r = make_reference(anywhere,NIL);
	sc = make_cell_reference(r);
      }
      if((int)gen_length(kl)>k){
	entity anywhere =entity_all_xxx_locations_typed(ANYWHERE_LOCATION,
							cell_to_type(kc));
	reference r = make_reference(anywhere,NIL);
	kc = make_cell_reference(r);
      }
      points_to npt = make_points_to(sc, kc, points_to_approximation(pt), make_descriptor_none());
      if(!points_to_equal_p(npt,pt)){
	pt_out = set_del_element(pt_out, pt_out, (void*)pt);
	pt_out = set_add_element(pt_out, pt_out, (void*)npt);
      }
    }
    set_assign(pt_in, set_clear(pt_in));
    pt_out =  points_to_expression(cond, pt_out, false);
    set_assign(pt_in, merge_points_to_set(cur, pt_out));
    if(set_equal_p(cur, pt_in))
      break;
  }

  points_to_storage(pt_in, dowhile_body, true);
  set_assign(pt_in, points_to_independent_store(pt_in));
  return pt_in;

}

/*Computing the points to of a test, all the relationships are of type
  MAY, can be refined later by using preconditions. */
set points_to_test(test test_stmt, set pt_in, bool store) {
  statement true_stmt = statement_undefined;
  statement false_stmt = statement_undefined;
  set true_pts_to = set_generic_make(set_private, points_to_equal_p,
				     points_to_rank);
  set false_pts_to = set_generic_make(set_private, points_to_equal_p,
				      points_to_rank);
  set res = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  
  /* condition's side effect and information are taked into account :
     if(p==q) or if(*p++) */
  expression cond = test_condition(test_stmt);
  pt_in = set_assign(pt_in, points_to_expression(cond, pt_in, store));
  true_stmt = test_true(test_stmt);
  true_pts_to = set_assign(true_pts_to,
			   points_to_recursive_statement(true_stmt,
							 pt_in, true));
  false_stmt = test_false(test_stmt);
  if(empty_statement_p(false_stmt))
    false_pts_to = set_assign(false_pts_to, pt_in);
  else
  false_pts_to = set_assign(false_pts_to,
			    points_to_recursive_statement(false_stmt,
							 pt_in, true));
  
  res = merge_points_to_set(true_pts_to, false_pts_to);
  return res;
}

/* computing the points-to of a call, user_functions not yet implemented. */
set points_to_call(statement s, call c, set pt_in, bool store __attribute__ ((__unused__))) {
  entity e = call_function(c);
  cons *pc = call_arguments(c);
  tag tt;
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set_assign(pt_in, points_to_init(s, pt_in));
  switch (tt = value_tag(entity_initial(e))) {
  case is_value_code:{
    /* call to an external function; preliminary version*/
    pips_user_warning("The function call to \"%s\" is ignored\n"
		      "On going implementation...\n", entity_user_name(e));
    set_assign(pt_out, pt_in);
  }
    break;
  case is_value_symbolic:
    set_assign(pt_out, pt_in);
    break;
  case is_value_constant:
    pt_out = set_assign(pt_out, pt_in);
    break;
  case is_value_unknown:
    pips_internal_error("function %s has an unknown value\n",
			entity_name(e));
    break;
  case is_value_intrinsic: {
    set_methods_for_proper_simple_effects();
    //set_methods_for_simple_pointer_effects();
    list el = call_to_proper_effects(c);
    generic_effects_reset_all_methods();
    pips_debug(5, "intrinsic function %s\n", entity_name(e));
    set_assign(pt_out,
		   points_to_intrinsic(s, c, e, pc, pt_in, el));
  }
    break;
  default:
    pips_internal_error("unknown tag %d\n", tt);
    break;
  }

  return pt_out;
}

/*We need call effects, which is not implemented yet, so we call
 * expression_to_proper_effects after creating an expression from the
 * call. Will be later moved at effects-simple/interface.c
 */
list call_to_proper_effects(call c) {
  expression e = call_to_expression(c);
  list el = expression_to_proper_effects(e);

  syntax_call( expression_syntax( e)) = call_undefined;
  free_expression(e);

  return el;
}

/* Process an expression, test if it's a call or a reference*/
set points_to_expression(expression e, set pt_in, bool store) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  call c = call_undefined;
  statement st = statement_undefined;
  syntax s = expression_syntax(copy_expression(e));
  switch (syntax_tag(s)) {
  case is_syntax_call: {
    c = syntax_call(s);
    st = make_expression_statement(e);
    set_assign(pt_out, points_to_call(st, c, pt_in, store));
  }
    break;
  case is_syntax_cast: {
    /* The cast is ignored, although it may generate aliases across
       types */
    cast ct = cast_undefined;
    expression e = expression_undefined;
    ct = syntax_cast(s);
    e = cast_expression(ct);
    st = make_expression_statement(e);
    set_assign(pt_out, points_to_expression(e, pt_in, store));
    break;
  }
  case is_syntax_reference: {
    set_assign(pt_out, pt_in);
    break;
  }

  case is_syntax_range: {
    set_assign(pt_out, pt_in);
    break;
  }
  case is_syntax_sizeofexpression: {
     set_assign(pt_out, pt_in);
    break;
  }
  case is_syntax_subscript: {
    set_assign(pt_out, pt_in);
    break;
  }
  case is_syntax_application: {
    set_assign(pt_out, pt_in);
    break;
  }
  case is_syntax_va_arg: {
    set_assign(pt_out, pt_in);
    break;
  }

  default:
    pips_internal_error("unexpected syntax tag (%d)", syntax_tag(s));
  }
  return pt_out;
}


set points_to_unstructured(__attribute__ ((__unused__))unstructured uns, set pt_in,__attribute__ ((__unused__)) bool store){
  /* statement s = control_statement(unstructured_entry(uns)); */
/*   statement ss = control_statement(unstructured_exit(uns)); */
/*   set_assign( pt_in, points_to_recursive_statement(s, pt_in, true)); */
/*   set_assign( pt_in, points_to_recursive_statement(ss, pt_in, true)); */
  return pt_in;

}

set points_to_goto(__attribute__ ((__unused__))statement current, set pt_in,__attribute__ ((__unused__)) bool store){
 /*  set_assign( pt_in, points_to_recursive_statement(current, pt_in, true)); */
  return pt_in;

}

/* Process recursively statement "current". Use pt_list as the input
   points-to set. Save it in the statement mapping. Compute and return
   the new points-to set which holds after the execution of the statement. */
set points_to_recursive_statement(statement current, set pt_in, bool store) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  list dl = NIL;
  set_assign(pt_out, pt_in);
  instruction i = statement_instruction(current);
  points_to_storage(pt_in, current, store);
  
  ifdebug(1) print_statement(current);
  switch (instruction_tag(i)){
      /* instruction = sequence + test + loop + whileloop +
	 goto:statement +
	 call + unstructured + multitest + forloop  + expression ;*/

  case is_instruction_call: {
    set_assign(pt_out, points_to_call(current,instruction_call(i), pt_in, store));
  }
    break;
  case is_instruction_sequence: {
    set_assign(pt_out, points_to_sequence(instruction_sequence(i),pt_in, store));
  }
    break;
  case is_instruction_test: {
    set_assign(pt_out, points_to_test(instruction_test(i), pt_in,
				      store));
    statement true_stmt = test_true(instruction_test(i));
    statement false_stmt = test_false(instruction_test(i));
    if(statement_block_p(true_stmt) && !ENDP(dl=statement_declarations(true_stmt))){
      pt_out = points_to_projection(pt_out, dl);
      pt_out = merge_points_to_set(pt_out, pt_in);
    }
    if(statement_block_p(false_stmt) && !ENDP(dl=statement_declarations(false_stmt))){
      pt_out = points_to_projection(pt_out, dl);
      pt_out = merge_points_to_set(pt_out, pt_in);
     }
  }
    break;
  case is_instruction_whileloop: {
    store = false;
    if (evaluation_tag(whileloop_evaluation(instruction_whileloop(i))) == 0)
      set_assign(pt_out, points_to_whileloop(
					     instruction_whileloop(i), pt_in, store));
    else
      set_assign(pt_out, points_to_do_whileloop(
						instruction_whileloop(i), pt_in, store));
    
      statement ws = whileloop_body(instruction_whileloop(i));
      if(statement_block_p(ws) && !ENDP(dl=statement_declarations(ws)))
	pt_out = points_to_projection(pt_out, dl);
  }
    break;
  case is_instruction_loop: {
    store = false;
    set_assign(pt_out, points_to_loop(instruction_loop(i), pt_in, store));
    statement ls = loop_body(instruction_loop(i));
      if(statement_block_p(ls) && !ENDP(dl=statement_declarations(ls)))
	pt_out = points_to_projection(pt_out, dl);
  }
    break;
  case is_instruction_forloop: {
    store = false;
    set_assign(pt_out, points_to_forloop(instruction_forloop(i),
					 pt_in, store));
    statement ls = forloop_body(instruction_forloop(i));
    if(statement_block_p(ls) && !ENDP(dl=statement_declarations(ls)))
      pt_out = points_to_projection(pt_out, dl);
  }
    break;
  case is_instruction_expression: {
    set_assign(pt_out, points_to_expression(
					    instruction_expression(i), pt_in, store));
  }
    break;
  case is_instruction_unstructured: {
   /*  statement e =control_statement( unstructured_entry(instruction_unstructured(i))); */
/*     statement s =control_statement( unstructured_exit(instruction_unstructured(i))); */
/*     set_assign(pt_out, points_to_recursive_statement(e, pt_in, store)); */
/*     set_assign(pt_out, points_to_recursive_statement(s, pt_out, store)); */
    // pips_internal_error("Case unstructured not implemented yet\n");
    store = false;
    set_assign(pt_out, points_to_unstructured(instruction_unstructured(i), pt_in, store));
  }
    break;
  case is_instruction_goto:{
    store = false;
    set_assign(pt_out, points_to_goto(current, pt_in, false));
  }
    break;
  default:
    pips_internal_error("Unexpected instruction tag %d\n", instruction_tag(
									   i));
    break;
  }
  return pt_out;
}
/* Entry point: intialize the entry poitns-to set; intraprocedurally,
   it's an empty set. */
void points_to_statement(statement current, set pt_in) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set_assign(pt_out, points_to_recursive_statement(current, pt_in,
							    true));
}

set points_to_init(statement s, set pt_in)
{
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  list l = NIL;

  set_assign(pt_out, pt_in);
  /* test if the statement is a declaration. */
  if (c_module_p(get_current_module_entity()) &&
      (declaration_statement_p(s))) {
    list l_decls = statement_declarations(s);
    pips_debug(1, "declaration statement \n");

    FOREACH(ENTITY, e, l_decls){
      if(type_variable_p(entity_type(e))){
	if(!storage_rom_p(entity_storage(e))){
	value v_init = entity_initial(e);
	/* generate points-to due to the initialisation */
	if(value_expression_p(v_init)){
	  expression exp_init = value_expression(v_init);
	  expression lhs = entity_to_expression(e);
	  if(expression_pointer_p(lhs))
	    pt_out = set_assign(pt_out,points_to_assignment(s,
							    copy_expression(lhs),
							    copy_expression(exp_init),
							    pt_in));
	}
	else{
	  l = points_to_init_variable(e);
	  FOREACH(CELL, cl, l){
	    list l_cl = CONS(CELL, cl, NIL);
	    set_union(pt_out, pt_out, points_to_nowhere_typed(l_cl, pt_out));
	  }
	}
      }
    }
  }
  }
  return pt_out;
}

list points_to_init_variable(entity e){
  list l = NIL;
  //  effect ef = effect_undefined;
  expression ex = entity_to_expression(e);

  if(entity_variable_p(e)){
    basic b = variable_basic(type_variable(entity_type(e)));
    if(expression_pointer_p(ex)){
      reference r = make_reference(e, NIL);
      cell c = make_cell_reference(r);
      l = CONS(CELL, c, NIL);
    }else if(entity_array_p(e)){
      if(basic_pointer_p(b)){
	expression ind = make_unbounded_expression();
	reference r = make_reference(e, CONS(EXPRESSION, ind, NULL));
	cell c = make_cell_reference(r);
	 l = CONS(CELL, c, NIL);
      }
    }else if(basic_derived_p(b)){
      entity ee = basic_derived(b);
      type t = entity_type(ee);
      if(type_struct_p(t)){
	l = points_to_init_struct(e, ee);
      }
    }else if(basic_typedef_p(b)){
      l = points_to_init_typedef(e);
    }
  }
  return l;
}

list points_to_init_pointer_to_typedef(entity e){
  list l = NIL;
  type t1 = ultimate_type(entity_type(e));
 
  if(type_variable_p(t1)){
    basic b = variable_basic(type_variable(entity_type(e)));
    if(basic_pointer_p(b)){
      type t2 = basic_pointer(variable_basic(type_variable(t1)));
      if(typedef_type_p(t2)){
	basic b1 = variable_basic(type_variable(t2));
	  if(basic_typedef_p(b1)){
	    entity e2  = basic_typedef(b1);
	    if(entity_variable_p(e2)){
	      basic b2 = variable_basic(type_variable(entity_type(e2)));
	      if(basic_derived_p(b2)){
		entity e3 = basic_derived(b2);
		l = points_to_init_pointer_to_derived(e, e3);
	      }
	    }
	  }
	}
      else if(derived_type_p(t2)){
	entity e3 = basic_derived(variable_basic(type_variable(t2)));
	l = points_to_init_pointer_to_derived(e, e3);
      }
    }
  }

  return l;
}

list points_to_init_pointer_to_derived(entity e, entity ee){
  list l = NIL;
  type tt = entity_type(ee);

  if(type_struct_p(tt))
   l = points_to_init_pointer_to_struct(e, ee);
  else if(type_union_p(tt))
    pips_user_warning("union case not handled yet \n");
  else if(type_enum_p(tt))
    pips_user_warning("union case not handled yet \n");
  return l;
}



list  points_to_init_pointer_to_struct(entity e, entity ee){
  list l = NIL;
  bool  eval = true;
  type tt = entity_type(ee);
  expression ex = entity_to_expression(e);

  if(type_struct_p(tt)){
    list l1 = type_struct(tt);
    if(!array_argument_p(ex)){
      FOREACH(ENTITY, i, l1){

	expression ef = entity_to_expression(i);
	if(expression_pointer_p(ef)){
	  expression ex1 = MakeBinaryCall(entity_intrinsic(POINT_TO_OPERATOR_NAME),
					  ex,
					  ef);
	  cell c = get_memory_path(ex1, &eval);
	  l = gen_nconc(CONS(CELL, c, NIL),l);
	}
      }
    }
    else
      l = points_to_init_array_of_struct(e, ee);
  }

  return l;
}

list points_to_init_typedef(entity e){
  list l = NIL;
   type t1 = entity_type(e);
    
  if(type_variable_p(t1)){
    basic b = variable_basic(type_variable(entity_type(e)));
    tag tt = basic_tag(b);
    switch(tt){
    case is_basic_int:;
      break;
    case is_basic_float:;
      break;
    case is_basic_logical:;
      break;
    case is_basic_overloaded:;
      break;
    case is_basic_complex:;
      break;
    case is_basic_string:;
      break;
    case is_basic_bit:;
      break;
    case is_basic_pointer:;
      break;
    case is_basic_derived:{
      bool  eval = true;
      type t = entity_type(e);
      expression ex = entity_to_expression(e);
      if(type_struct_p(t)){
	list l1 = type_struct(t);
	FOREACH(ENTITY, i, l1){
	  expression ef = entity_to_expression(i);
	  if(expression_pointer_p(ef)){
	    expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
					    ex,
					    ef);
	    cell c = get_memory_path(ex1, &eval);
	    l = gen_nconc(CONS(CELL, c, NIL),l);
	  }else if(array_argument_p(ef)){
	    /* print_expression(ef); */;
	}
	}
      }else if(type_enum_p(t))
	pips_user_warning("enum case not handled yet \n");
      else if(type_union_p(t))
	pips_user_warning("union cas not handled yet \nx");
    }
      break;
    case is_basic_typedef:
      {
	entity e1  = basic_typedef(b);
	type t1 = entity_type(e1);
	if(entity_variable_p(e1)){
	  basic b2 =  variable_basic(type_variable(t1));
	  if(basic_derived_p(b2)){
	    entity e2  = basic_derived(b2);
	    l = points_to_init_derived(e, e2);
	  }
	}
      }

      break;
    default: pips_internal_error("unexpected tag %d\n", tt);
      break;
    }
  }
  return l;
}


list points_to_init_derived(entity e, entity ee){
  list l = NIL;
  type tt = entity_type(ee);

  if(type_struct_p(tt))
   l = points_to_init_struct(e, ee);
  return l;
}



list  points_to_init_struct(entity e, entity ee){
  list l = NIL;
  bool  eval = true;
  type tt = entity_type(ee);
  expression ex = entity_to_expression(e);

  if(type_struct_p(tt)){
    list l1 = type_struct(tt);
    if(!array_argument_p(ex)){
      FOREACH(ENTITY, i, l1){

	expression ef = entity_to_expression(i);
	if(expression_pointer_p(ef)){
	  expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
					  ex,
					  ef);
	  cell c = get_memory_path(ex1, &eval);
	  l = gen_nconc(CONS(CELL, c, NIL),l);
	}
	else if(array_argument_p(ef)){
	  basic b = variable_basic(type_variable(entity_type(i)));
	  /* arrays of pointers are changed into independent store arrays
	     and initialized to nowhere_b0 */
	  if(basic_pointer_p(b)){
	    effect eff = effect_undefined;
	    expression ind = make_unbounded_expression();
	    expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
					    ex,
					    ef);
	    set_methods_for_proper_simple_effects();
	    list l1 = generic_proper_effects_of_complex_address_expression(ex1, &eff,
								       true);
	    reference r2 = effect_any_reference(eff);
	    effects_free(l1);
	    generic_effects_reset_all_methods();
	    reference_indices(r2)= gen_nconc(reference_indices(r2), CONS(EXPRESSION, ind, NIL));
	    cell c = make_cell_reference(r2);
	    l = gen_nconc(CONS(CELL, c, NIL),l);
	  }
	}
      }
    }
    else
      l = points_to_init_array_of_struct(e, ee);
  }

  return l;
}

list points_to_init_array_of_struct(entity e, entity field){
  list l = NIL;
  bool eval = true;
  type tt = entity_type(field);
  if(type_struct_p(tt)){
    list l1 = type_struct(tt);
    FOREACH(ENTITY, i, l1) {
      expression ee = entity_to_expression(i);
      if(expression_pointer_p(ee)){
	expression ex = entity_to_expression(e);
	if(array_argument_p(ex)){
	  effect ef = effect_undefined;
	  reference  r = reference_undefined;
	  /*init the effect's engine*/
	  set_methods_for_proper_simple_effects();
	  list l1 = generic_proper_effects_of_complex_address_expression(ex, &ef,
									 true);
	  effect_add_dereferencing_dimension(ef);
	  r = effect_any_reference(ef);
	  list l_inds = reference_indices(r);
	  EXPRESSION_(CAR(l_inds)) = make_unbounded_expression();
	  ex = reference_to_expression(r);
	  effects_free(l1);
	  generic_effects_reset_all_methods();
	}
	expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
					ex,
					ee);
	cell c = get_memory_path(ex1, &eval);
	l = gen_nconc(CONS(CELL, c, NIL), l);
      }
    }
  }
  else
    pips_internal_error("type struct expected\n");
      
  return l;
}

bool points_to_analysis(char * module_name) {
  entity module;
  statement module_stat;
  set pt_in =
    set_generic_make(set_private, points_to_equal_p, points_to_rank);
  list pts_to_list = NIL;

  init_pt_to_list();
  module = module_name_to_entity(module_name);
  set_current_module_entity(module);
  make_effects_private_current_context_stack();

  debug_on("POINTS_TO_DEBUG_LEVEL");

  pips_debug(1, "considering module %s\n", module_name);
  set_current_module_statement((statement) db_get_memory_resource(DBR_CODE,
								  module_name, true));
  module_stat = get_current_module_statement();
  statement_consistent_p(module_stat);

  /* Get the summary_intraprocedural_points_to resource.*/
  points_to_list summary_pts_to_list =
    (points_to_list) db_get_memory_resource(DBR_SUMMARY_POINTS_TO_LIST,
					    module_name, true);
  /* Transform the list of summary_points_to in set of points-to.*/
    points_to_list_consistent_p(summary_pts_to_list);
  //pts_to_list = gen_points_to_list_cons(summary_pts_to_list, pts_to_list);
  pts_to_list = gen_full_copy_list(points_to_list_list(summary_pts_to_list));
  pt_in = set_assign_list(pt_in, pts_to_list);
  /* Compute the points-to relations using the summary_points_to as input.*/
  points_to_statement(module_stat, pt_in);
  statement_points_to_consistent_p(get_pt_to_list());
  DB_PUT_MEMORY_RESOURCE(DBR_POINTS_TO_LIST, module_name, get_pt_to_list());

  reset_pt_to_list();

  reset_current_module_entity();
  reset_current_module_statement();
  reset_effects_private_current_context_stack();
  debug_off();
   
  bool good_result_p = true;
  return (good_result_p);

}



