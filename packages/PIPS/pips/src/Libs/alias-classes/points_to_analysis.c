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
#include "points_to_private.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
//#include "parser_private.h"
#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "effects-generic.h"
#include "effects-simple.h"
//#include "effects-convex.h"
//#include "transformations.h"
#include "pipsdbm.h"
#include "resources.h"
#include "newgen_set.h"
#include "alias-classes.h"


/* operator fusion to calculate the relation of two points
   to. (should be at a higher level) */
static approximation fusion_approximation(approximation app1,
					  approximation app2)
{
  if(approximation_exact_p(app1) && approximation_exact_p(app2))
    return make_approximation_exact();
  else
    return make_approximation_may();
}

/* To distinguish between the different basic cases of Emami's
   assignement : involving x, &x, *x, m, m.x, m->x, a[]...*/
enum Emami {
  EMAMI_ERROR = -1,
  EMAMI_NEUTRAL = 0,
  EMAMI_ADDRESS_OF = 1,
  EMAMI_DEREFERENCING = 2,
  EMAMI_FIELD = 3,
  EMAMI_STRUCT = 4,
  EMAMI_ARRAY = 5,
  EMAMI_HEAP = 6,
  EMAMI_POINT_TO = 7
};

/* Find out Emami's types */
static int emami_expression_type(expression exp) {
  tag t;
  switch (t = syntax_tag(expression_syntax(exp)))
    {
      /*in case it's an EMAMI_NEUTRAL  x or an EMAMI_ARRAY or an EMAMI_STRUC */
    case is_syntax_reference:
      {
	if (array_argument_p(exp))
	  return EMAMI_ARRAY;
	syntax s = expression_syntax(exp);
	reference r = syntax_reference(s);
	entity e = reference_variable(r);
	type typ = entity_type(e);
	variable v = type_variable(typ);
	basic b = variable_basic(v);
	t = basic_tag(b);
	if (t == is_basic_derived) {
	  entity ee = basic_derived(b);
	  type tt = entity_type(ee);
	  if (type_struct_p(tt) == true)
	    return EMAMI_STRUCT;
	  else
	    return EMAMI_NEUTRAL;
	} else
	  return EMAMI_NEUTRAL;
	break;
      }
    case is_syntax_range:
      return EMAMI_ERROR;
      break;
      /* in case it's an EMAMI_ADDRESS_OF or an EMAMI_DEREFERENCING or an
	 EMAMI_FIELD or an an EMAMI_HEAP or an EMAMI_POINT_TO...  */
    case is_syntax_call:
      {
	call c = expression_call(exp);
	if (entity_an_operator_p(call_function(c), ADDRESS_OF))
	  return EMAMI_ADDRESS_OF;
	else if (entity_an_operator_p(call_function(c), DEREFERENCING))
	  return EMAMI_DEREFERENCING;
	else if (entity_an_operator_p(call_function(c), FIELD))
	  return EMAMI_FIELD;
	else if (entity_an_operator_p(call_function(c), POINT_TO))
	  return EMAMI_POINT_TO;
	else if (ENTITY_MALLOC_SYSTEM_P(call_function(c)))
	  return EMAMI_HEAP;
	else
	  return EMAMI_ERROR;
	break;
      }

    case is_syntax_cast:
      {
	cast ct = syntax_cast(expression_syntax(exp));
	expression e = cast_expression(ct);

	return emami_expression_type(e);
	break;
      }
    case is_syntax_sizeofexpression:
      return EMAMI_ERROR;
      break;
    case is_syntax_subscript:
      return EMAMI_ERROR;
      break;
    case is_syntax_application:
      return EMAMI_ERROR;
      break;
    case is_syntax_va_arg:
      return EMAMI_ERROR;
      break;
    default:
      {
	pips_internal_error("unknown tag %d", t);
	return EMAMI_ERROR;
	break;
      }
    }
}

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


/* Order the two points-to relations according to the alphabetical
   order of the underlying variables. Return -1, 0, or 1. */
int compare_points_to_location(void * vpt1, void * vpt2) {
  points_to pt1 = (points_to) vpt1;
  points_to pt2 = (points_to) vpt2;
  int null_1 = (pt1 == (points_to) NULL), null_2 = (pt2 == (points_to) NULL);

  if (points_to_domain_number(pt1) != points_to_domain
      || points_to_domain_number(pt2) != points_to_domain)
    return (null_2 - null_1);
  else {
    cell c1 = points_to_source(pt1);
    cell c2 = points_to_source(pt2);
    reference r1 = cell_to_reference(c1);
    reference r2 = cell_to_reference(c2);
    entity e1 = reference_variable(r1);
    entity e2 = reference_variable(r2);
    //return reference_equal_p(r1, r2);
    return compare_entities_without_scope(&e1, &e2);
  }
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
      Possible_set = set_add_element(Possible_set,
				     Possible_set,
				     pt);
    }
  }
  Merge_set = set_clear(Merge_set);
  Merge_set = set_union(Merge_set, Possible_set, Definite_set);

  return Merge_set;
}

/* storing the points to associate to a statement s, in the case of
   loops the field store is set to false to prevent from the key redefenition*/
void points_to_storage(set pts_to_set, statement s, bool store) {
  list pt_list = NIL;
  if (!set_empty_p(pts_to_set) && store == true) {
    //  print_points_to_set(stderr,"",pts_to_set);
    pt_list = set_to_sorted_list(pts_to_set,
				 (int(*)(const void*, const void*)) compare_points_to_location);
    points_to_list new_pt_list = make_points_to_list(pt_list);
    store_pt_to_list(s, new_pt_list);
  }
}

/* one basic case of Emami: < x = y > and < m.x = m.y >, x and y are pointers. */
set basic_ref_ref(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;

  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;

  // creation of the source
  effect e1 = effect_undefined, e2 = effect_undefined;
  /*init the effect's engine*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs, &e1,
								 true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs, &e2,
								 false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);

  // add the points_to relation to the set generated
  // by this assignment
  reference ref = copy_reference(effect_any_reference(e2));
  sink = make_cell_reference(ref);
  set s = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  SET_FOREACH(points_to, i, pts_to_set) {
    if (locations_equal_p(points_to_source(i), sink))
      s = set_add_element(s, s, (void*) i);
  }
  SET_FOREACH(points_to, j, s) {
    new_sink = copy_cell(points_to_sink(j));
    // access new_source = copy_access(source);
    rel = points_to_approximation(j);
    pt_to = make_points_to(source, new_sink, rel, make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to, gen_pts_to, (void*) pt_to);
  }

  /* the case we have an array at the right hand side should be implemented later.*/
  //    /* in case x = y[i]*/
  //    if (array_argument_p(rhs)) {
  //        new_sink = make_cell_reference(ref2);
  //        // access new_source = copy_access(source);
  //        rel = make_approximation_exact();
  //        pt_to = make_points_to(source, new_sink, rel, make_descriptor_none());
  //        gen_pts_to = set_add_element(gen_pts_to, gen_pts_to, (void*) pt_to);
  //    }
  // creation of the written set
  // search of all the points_to relations in the
  // alias set where the source is equal to the lhs
  SET_FOREACH(points_to, k, pts_to_set) {
    if (locations_equal_p(points_to_source(k), source))
      written_pts_to = set_add_element(written_pts_to, written_pts_to,
				       (void *) k);
  }
  pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
  ifdebug(1)
    print_points_to_set("Points to for the  case <x = y>\n", pts_to_set);
  return pts_to_set;
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
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
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
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);
  // add the points_to relation to the set generated
  // by this assignment
  e2 = EFFECT(CAR(l3));
  /* transform tab[*] into tab[*][0]. */
  effect_add_dereferencing_dimension(e2);
  ref2 = effect_any_reference(copy_effect(e2));
  free(l3);
  sink = make_cell_reference(copy_reference(ref2));
  new_sink = make_cell_reference(copy_reference(ref2));
  // access new_source = copy_access(source);
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
set basic_ref_addr(set pts_to_set, expression lhs, expression rhs) {
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

/* one basic case of Emami: < x = *y >, we affect to x all the sinks
   of y. */
set basic_ref_deref(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set s = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;

  ifdebug(1) {
    pips_debug(1, " case  x = *y\n");
  }

  // creation of the source
  effect e1 = effect_undefined, e2 = effect_undefined;
  /*init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
								 &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(
								 rhs, &e2, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = copy_reference(effect_any_reference(e1));
  source = make_cell_reference(ref1);
  // creation of the sink
  ref2 = copy_reference(effect_any_reference(e2));
  sink = make_cell_reference(ref2);
  // fetch the points to relations
  // where source = source 1
  // creation of the written set
  // search of all the points_to relations in the
  // alias set where the source is equal to the lhs
  SET_FOREACH(points_to, elt, pts_to_set) {
    if(locations_equal_p(points_to_source(elt),
			 source))
      written_pts_to = set_add_element(written_pts_to,
				       written_pts_to,(void*)elt);
  }
  SET_FOREACH(points_to, i, pts_to_set) {
    if(locations_equal_p(points_to_source((points_to)i), sink)) {
      s =set_add_element(s, s, (void *)i);
    }
  }
  SET_FOREACH(points_to, j,s ) {
    SET_FOREACH(points_to, p, pts_to_set) {
      if(locations_equal_p(points_to_sink(j),
			   points_to_source(p))) {
	new_sink = copy_cell(points_to_sink(p));
	rel = fusion_approximation(points_to_approximation(j),
				   points_to_approximation(i));
	pt_to = make_points_to(source, new_sink, rel,
			       make_descriptor_none());
	gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
				     (void *) pt_to );
      }
    }
  }

  pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
  ifdebug (1){
    print_points_to_set("Points To pour le cas 3 <x = *y> \n",
			pts_to_set);
  }
 
  return pts_to_set;
}

/* case *x = y   or *m.x = y, x is a double pointer, y is a pointer. */
set basic_deref_ref(set pts_to_set, expression lhs, expression rhs) {
  set s1 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s2 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s3 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set change_pts_to = set_generic_make(set_private, points_to_equal_p,
				       points_to_rank);
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_source = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1) {
    pips_debug(1, " case *x = y\n");
  }
  
  effect e1 = effect_undefined, e2 = effect_undefined;
  /*init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
								 &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
								 &e2, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);
  // recuperation of y
  ref2 = effect_any_reference(e2);
  // ent2 = reference_variable(copy_reference(ref2));
  sink = make_cell_reference(copy_reference(ref2));

  /* creation of the set written_pts_to =
     {(x1,x2,rel)| (x, x1, EXACT), (x1, x2, rel) /in pts_to_set}*/
  SET_FOREACH(points_to, i, pts_to_set) {
    if( locations_equal_p(points_to_source(i), source) &&
	approximation_exact_p(points_to_approximation(i))) {
      SET_FOREACH(points_to, j,pts_to_set ) {
	if( locations_equal_p(points_to_source(j) ,
			      points_to_sink(i)))
	  written_pts_to =
	    set_add_element(written_pts_to,
			    written_pts_to, (void *)j);
      }
    }
  }
  /* {(x1, x2,EXACT)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
  SET_FOREACH(points_to, k, pts_to_set) {
    if( locations_equal_p(points_to_source(i), source)
	&& approximation_may_p(points_to_approximation(k))) {
      SET_FOREACH(points_to, h,pts_to_set ) {
	if(locations_equal_p(points_to_source(h),points_to_sink(k))&&
	   approximation_exact_p(points_to_approximation(h)))
	  s2 = set_add_element(s2, s2, (void *)h);
      }
    }
  }

  /* {(x1, x2,MAY)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
  SET_FOREACH(points_to, l, pts_to_set) {
    if(locations_equal_p(points_to_source(l), source) &&
       approximation_may_p(points_to_approximation(l))) {
      SET_FOREACH(points_to, m,pts_to_set) {
	if(locations_equal_p( points_to_source(m),points_to_sink(l))&&
	   approximation_exact_p(points_to_approximation(m))) {
	  points_to_approximation(m) = make_approximation_may();
	  s3 = set_add_element(s3, s3, (void *)m);
	}
      }
    }
  }
  change_pts_to = set_difference(change_pts_to, pts_to_set, s3);
  change_pts_to = set_union(change_pts_to, change_pts_to, s3);
  SET_FOREACH(points_to, n, pts_to_set) {
    if(locations_equal_p(points_to_source(n), source)) {
      SET_FOREACH(points_to, o, pts_to_set) {
	if(locations_equal_p(points_to_source(o) , sink)) {
	  new_source = copy_cell(points_to_sink(n));
	  new_sink = copy_cell(points_to_sink(o));
	  rel = fusion_approximation(points_to_approximation(n),
				     points_to_approximation(o));
	  pt_to = make_points_to(new_source, new_sink, rel,
				 make_descriptor_none());
	  gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
				       (void *)pt_to);
	}
      }
    }
  }
  s1 = set_difference(s1, change_pts_to, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, s1);
  
  return pts_to_set;
}

// case *x = &y and *m.x = &y;
static set basic_deref_addr(set pts_to_set, expression lhs, expression rhs) {
  set s1 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s2 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s3 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set change_pts_to = set_generic_make(set_private, points_to_equal_p,
				       points_to_rank);
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;

  ifdebug(1) {
    pips_debug(1, " case *x = &y\n");
  }
    
  effect e1 = effect_undefined, e2 = effect_undefined;
  /*init the effect's engine*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
								 &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs, &e2, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(copy_reference(ref1));
  ref2 = effect_any_reference(e2);
  sink = make_cell_reference(copy_reference(ref2));
  SET_FOREACH(points_to, i, pts_to_set) {
    if(locations_equal_p(points_to_source(i), source) &&
       approximation_exact_p(points_to_approximation(i))) {
      SET_FOREACH(points_to, j,pts_to_set) {
	if(locations_equal_p(points_to_source(j),points_to_sink(i)))
	  written_pts_to = set_add_element(written_pts_to,
					   written_pts_to,(void*)j);
      }
    }
  }
  SET_FOREACH(points_to, k, pts_to_set) {
    if(locations_equal_p(points_to_source(i), source) &&
       approximation_may_p(points_to_approximation(k))) {
      SET_FOREACH(points_to, h, pts_to_set) {
	if(locations_equal_p(points_to_source(h),points_to_sink(k))&&
	   approximation_exact_p(points_to_approximation(h))) {
	  s2 = set_add_element(s2, s2, (void *) h );
	}
      }
    }
  }
  SET_FOREACH(points_to, l, pts_to_set) {
    if(locations_equal_p(points_to_source(l), source) &&
       approximation_may_p(points_to_approximation(l))) {
      SET_FOREACH(points_to, m,pts_to_set) {
	if(locations_equal_p(points_to_source(m),points_to_sink(l))&&
	   approximation_exact_p(points_to_approximation(m))) {
	  points_to_approximation(m) = make_approximation_may();
	  s3 = set_add_element(s3, s3, (void *)m);
	}
      }
    }
  }
  change_pts_to = set_difference(change_pts_to, pts_to_set, s2);
  change_pts_to = set_union(change_pts_to, change_pts_to, s3);
  SET_FOREACH(points_to, n, pts_to_set) {
    if(locations_equal_p(points_to_source(n), source)) {
      SET_FOREACH(points_to, o, pts_to_set) {
	if(locations_equal_p(points_to_source(o),
			     points_to_sink(n))) {
	  points_to pt = make_points_to(points_to_source(o),
					sink,
					points_to_approximation(n),
					make_descriptor_none());
	  gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
				       (void *) pt );
	}
      }
    }
  }
  s1 = set_difference(s1, change_pts_to, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, s1);

  ifdebug(1)
    print_points_to_set("Points pour le cas 5 <*x = &y> \n ", pts_to_set);
  return pts_to_set;
}

// case *x = &y[i] and *m.x = &y[i], special case where the right hand
// side is an array's element.
set basic_deref_array(set pts_to_set, expression lhs, expression rhs) {
  set s1 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s2 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s3 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set change_pts_to = set_generic_make(set_private, points_to_equal_p,
				       points_to_rank);
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;

  ifdebug(1) {
    pips_debug(1, " case *x = &y[i] or *m.x = &y[i]\n");
  }
 
  effect e1 = effect_undefined, e2 = effect_undefined;
  /*init the effect's engine*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(
								 lhs, &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
								 &e2, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(copy_effect(e1));
  source = make_cell_reference(ref1);
  ref2 = effect_any_reference(copy_effect(e2));
  sink = make_cell_reference(ref2);
  SET_FOREACH(points_to, i, pts_to_set) {
    if(locations_equal_p(points_to_source(i), source) &&
       approximation_exact_p(points_to_approximation(i))) {
      SET_FOREACH(points_to, j,pts_to_set) {
	if(locations_equal_p(points_to_source(j),points_to_sink(i)))
	  written_pts_to =
	    set_add_element(written_pts_to,
			    written_pts_to,(void*)j);
      }
    }
  }
  SET_FOREACH(points_to, k, pts_to_set) {
    if(locations_equal_p(points_to_source(i), source) &&
       approximation_may_p(points_to_approximation(k))) {
      SET_FOREACH(points_to, h, pts_to_set) {
	if(locations_equal_p(points_to_source(h),points_to_sink(k))&&
	   approximation_exact_p(points_to_approximation(h))) {
	  s2 = set_add_element(s2, s2, (void *) h );
	}
      }
    }
  }
  SET_FOREACH(points_to, l, pts_to_set) {
    if(locations_equal_p(points_to_source(l), source) &&
       approximation_may_p(points_to_approximation(l))) {
      SET_FOREACH(points_to, m,pts_to_set) {
	if(locations_equal_p(points_to_source(m),points_to_sink(l))&&
	   approximation_exact_p(points_to_approximation(m))) {
	  points_to_approximation(m) = make_approximation_may();
	  s3 = set_add_element(s3, s3, (void *)m);
	}
      }
    }
  }
  change_pts_to = set_difference(change_pts_to, pts_to_set, s2);
  change_pts_to = set_union(change_pts_to, change_pts_to, s3);
  SET_FOREACH(points_to, n, pts_to_set) {
    if(locations_equal_p(points_to_source(n), source)) {
      SET_FOREACH(points_to, o,pts_to_set) {
	if(locations_equal_p( points_to_source(o),
			      points_to_sink(n))) {
	  points_to pt =
	    make_points_to(points_to_source(o),
			   sink,
			   points_to_approximation(n),
			   make_descriptor_none());
	  gen_pts_to =
	    set_add_element(gen_pts_to, gen_pts_to,
			    (void *) pt );
	}
      }
    }
  }
  s1 = set_difference(s1, change_pts_to, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, s1);
  ifdebug(1)
    print_points_to_set("Points pour le cas 5 <*x = &y> \n ",
			pts_to_set);
  
  return pts_to_set;
}

//case *x = *y, x and y are double pointers.
set basic_deref_deref(set pts_to_set, expression lhs, expression rhs) {
  set s1 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s2 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s3 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s4 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set change_pts_to = set_generic_make(set_private, points_to_equal_p,
				       points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_source = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;

  pips_debug(1, " case *x = *y\n");

  effect e1 = effect_undefined, e2 = effect_undefined;
  /*init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs, &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs, &e2, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  // FI->AM: you should replicate the reference, not the whole effect
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);
  ref2 = effect_any_reference(e2);
  sink = make_cell_reference(ref2);
  SET_FOREACH(points_to, i, pts_to_set) {
    if(locations_equal_p(points_to_source(i), source) &&
       approximation_exact_p(points_to_approximation(i))) {
      SET_FOREACH(points_to, j, pts_to_set) {
	if(locations_equal_p( points_to_source(j),
			      points_to_sink(i))) {
	  written_pts_to = set_add_element(written_pts_to,
					   written_pts_to,(void*)j);
	}
      }
    }
  }
  SET_FOREACH(points_to, k, pts_to_set) {
    if(locations_equal_p(points_to_source(i), source) &&
       approximation_may_p(points_to_approximation(k))) {
      SET_FOREACH(points_to, h, pts_to_set) {
	if(locations_equal_p( points_to_source(h), points_to_sink(k))
	   && approximation_exact_p(points_to_approximation(h))) {
	  points_to_approximation(h) = make_approximation_may();
	  s3 = set_add_element(s3, s3, (void *)h);
	}
      }
    }
  }
  SET_FOREACH(points_to, l, pts_to_set) {
    if(locations_equal_p(points_to_source(l), source) &&
       approximation_may_p(points_to_approximation(l))) {
      SET_FOREACH(points_to, m, pts_to_set) {
	if(locations_equal_p(points_to_source(m),points_to_sink(l))
	   && approximation_exact_p(points_to_approximation(m))) {
	  points_to_approximation(m) = make_approximation_may();
	  s4 = set_add_element(s4, s4, (void*)m);
	}
      }
    }
  }
  change_pts_to = set_difference(change_pts_to, pts_to_set, s3);
  change_pts_to = set_union(change_pts_to, change_pts_to, s4);
  SET_FOREACH(points_to, n, pts_to_set) {
    if(locations_equal_p(points_to_source(n), source)) {
      SET_FOREACH(points_to, o, pts_to_set) {
	if(locations_equal_p(points_to_source(o), sink)) {
	  SET_FOREACH(points_to, f, pts_to_set) {
	    if(locations_equal_p(points_to_source(f),
				 points_to_sink(o))) {
	      rel = fusion_approximation((fusion_approximation(points_to_approximation(n),
							       points_to_approximation(o))),
					 points_to_approximation(f));
	      new_source = copy_cell(points_to_sink(n));
	      new_sink = copy_cell(points_to_sink(f));
	      pt_to = make_points_to(new_source, new_sink, rel,
				     make_descriptor_none());
	      gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
					   (void *) pt_to );
	    }
	  }
	}
      }
    }
  }
  s1 = set_difference(s1, change_pts_to, written_pts_to);
  s2 = set_union(s2, gen_pts_to, s1);
  pts_to_set = set_union(pts_to_set, pts_to_set, s2);
  ifdebug(1)
    print_points_to_set("Points To pour le cas6  <*x = *y> \n",
			pts_to_set);
  
  return pts_to_set;
}

/* one basic case of Emami: < x.a = &y > */
set basic_field_addr(set pts_to_set, expression lhs, expression rhs) {
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
  list l = NIL;
  ifdebug(1) {
    pips_debug(1, " case x.a = &y\n");
  }

  effect e1 = effect_undefined;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  l = generic_proper_effects_of_complex_address_expression(lhs, &e1,
							   true);
  effects_free(l);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(copy_reference(ref1));
  // creation of the sink
  effect e2 = effect_undefined;
  set_methods_for_proper_simple_effects();
  l = generic_proper_effects_of_complex_address_expression(rhs,
							   &e2, false);
  effects_free(l);
  generic_effects_reset_all_methods();
  ref2 = effect_any_reference(e2);
  sink = make_cell_reference(copy_reference(ref2));
  // creation of the approximation
  rel = make_approximation_exact();
  // creation of the points_to approximation
  pt = make_points_to(source, sink, rel, make_descriptor_none());
  // add the points_to relation to the set generated
  //by this assignement
  gen_pts_to = set_add_element(gen_pts_to, gen_pts_to, (void*) pt);
  // creation of the written set
  // search of all the points_to relations in the
  // alias set where the source is equal to the lhs
  SET_FOREACH(points_to, i,pts_to_set) {
    if(locations_equal_p(points_to_source(i),source))
      written_pts_to = set_add_element(written_pts_to,
				       written_pts_to,
				       (void *)i);
  }

  pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
  pts_to_set = set_union(pts_to_set, pts_to_set, gen_pts_to);

  ifdebug(1) {
    print_points_to_set("points To for the case <x.a = &y> \n ",
			pts_to_set);
  }
  
  return pts_to_set;
}

/* one basic case of Emami: < x = y.a > */
set basic_ref_field(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent2 = entity_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;
  list l = NIL, l1 = NIL;
  pips_debug(1, " case x = y.a \n");
 
  effect e1 = effect_undefined, e2;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  l = generic_proper_effects_of_complex_address_expression(lhs, &e1,
							   true);
  l1 = generic_proper_effects_of_complex_address_expression(rhs, &e2,
							    false);
  effects_free(l);
  effects_free(l1);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);
  // add the points_to relation to the set generated
  // by this assignement
  reference ref = effect_any_reference(copy_effect(e2));
  // print_reference(copy_reference(r));
  sink = make_cell_reference(ref);
  set s = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  SET_FOREACH(points_to, i, pts_to_set) {
    if(locations_equal_p(points_to_source(i), sink))
      s = set_add_element(s, s, (void*)i);
  }
  SET_FOREACH(points_to, j, s) {
    new_sink = copy_cell(points_to_sink(j));
    // locations new_source = copy_locations(source);
    rel = points_to_approximation(j);
    pt_to = make_points_to(source, new_sink, rel,
			   make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
				 (void*) pt_to );
  }
  /* in case x = y[i]*/
  if (array_entity_p(ent2)) {
    new_sink = make_cell_reference(copy_reference(ref2));
    // locations new_source = copy_access(source);
    rel = make_approximation_exact();
    pt_to = make_points_to(source, new_sink, rel,
			   make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
				 (void*) pt_to);
  }
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
    print_points_to_set("Points to for the case <x = y.a>\n",
			pts_to_set);
 
  return pts_to_set;
}

/* one basic case of Emami: < x = y->a > */
set basic_ref_ptr_to_field(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent1 = entity_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;
  list l = NIL, l1 = NIL;

  ifdebug(1) printf("\n cas x = y->a \n");
 
  effect e1 = effect_undefined, e2;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  l = generic_proper_effects_of_complex_address_expression(lhs, &e1,
							   true);
  l1 = generic_proper_effects_of_complex_address_expression(rhs, &e2,
							    false);
  effects_free(l);
  effects_free(l1);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  ent1 = reference_variable(copy_reference(ref1));
  source = make_cell_reference(copy_reference(ref1));
  // add the points_to relation to the set generated
  // by this assignement
  ref2 = effect_any_reference(e2);
  // print_reference(copy_reference(r));
  sink = make_cell_reference(copy_reference(ref2));
  set s = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  SET_FOREACH(points_to, i, pts_to_set) {
    if(locations_equal_p(points_to_source(i), sink))
      s = set_add_element(s, s, (void*)i);
  }
  SET_FOREACH(points_to, j, s) {
    new_sink = copy_cell(points_to_sink(j));
    // access new_source = copy_access(source);
    rel = points_to_approximation(j);
    pt_to = make_points_to(source, new_sink, rel,
			   make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
				 (void*) pt_to );
  }
  /* in case x = y[i]*/
  if (array_argument_p(rhs)) {
    new_sink = make_cell_reference(copy_reference(ref2));
    rel = make_approximation_exact();
    pt_to = make_points_to(source, new_sink, rel,
			   make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
				 (void*) pt_to);
  }
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

  ifdebug        (1)
    print_points_to_set("Points to pour le cas 1 <x = y.a>\n",
			pts_to_set);

   return pts_to_set;
}

/* one basic case of Emami: < *x = m.y > */
set basic_deref_field(set pts_to_set, expression lhs, expression rhs) {
  set s1 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s2 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s3 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set change_pts_to = set_generic_make(set_private, points_to_equal_p,
				       points_to_rank);
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_source = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1) {
    pips_debug(1, " case *x = m;y\n");
  }
  
  effect e1 = effect_undefined, e2 = effect_undefined;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs, &e1,
								 true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs, &e2,
								 false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);
  // recuperation of y
  ref2 = effect_any_reference(e2);
  sink = make_cell_reference(ref2);

  /* creation of the set written_pts_to =
     {(x1,x2,rel)| (x, x1, EXACT), (x1, x2, rel) /in pts_to_set}*/
  SET_FOREACH(points_to, i, pts_to_set) {
    if( locations_equal_p(points_to_source(i), source) &&
	approximation_exact_p(points_to_approximation(i))) {
      SET_FOREACH(points_to, j,pts_to_set ) {
	if( locations_equal_p(points_to_source(j) ,
			      points_to_sink(i)))
	  written_pts_to = set_add_element(written_pts_to,
					   written_pts_to, (void *)j);
      }
    }
  }
  /* {(x1, x2,EXACT)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
  SET_FOREACH(points_to, k, pts_to_set) {
    if( locations_equal_p(points_to_source(i), source)
	&& approximation_may_p(points_to_approximation(k))) {
      SET_FOREACH(points_to, h,pts_to_set ) {
	if(locations_equal_p(points_to_source(h),points_to_sink(k))&&
	   approximation_exact_p(points_to_approximation(h)))
	  s2 = set_add_element(s2, s2, (void *)h);
      }
    }
  }

  /* {(x1, x2,MAY)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
  SET_FOREACH(points_to, l, pts_to_set) {
    if(locations_equal_p(points_to_source(l), source) &&
       approximation_may_p(points_to_approximation(l))) {
      SET_FOREACH(points_to, m,pts_to_set) {
	if(locations_equal_p(points_to_source(m),points_to_sink(l))&&
	   approximation_exact_p(points_to_approximation(m))) {
	  points_to_approximation(m) = make_approximation_may();
	  s3 = set_add_element(s3, s3, (void *)m);
	}
      }
    }
  }
  change_pts_to = set_difference(change_pts_to, pts_to_set, s3);
  change_pts_to = set_union(change_pts_to, change_pts_to, s3);
  SET_FOREACH(points_to, n, pts_to_set) {
    if(locations_equal_p(points_to_source(n), source)) {
      SET_FOREACH(points_to, o, pts_to_set) {
	if(locations_equal_p(points_to_source(o) , sink)) {
	  new_source = copy_cell(points_to_sink(n));
	  new_sink = copy_cell(points_to_sink(o));
	  rel = fusion_approximation(points_to_approximation(n),
				     points_to_approximation(o));
	  pt_to = make_points_to(new_source, new_sink, rel,
				 make_descriptor_none());
	  gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
				       (void *)pt_to);
	}
      }
    }
  }
  s1 = set_difference(s1, change_pts_to, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, s1);
  ifdebug        (1)
    print_points_to_set("Points to pour le cas 1 <x = y.a>\n",
			pts_to_set);

  return pts_to_set;

}

/* one basic case of Emami: < m.x = y->a > */
set basic_field_ptr_to_field(set pts_to_set, expression lhs __attribute__ ((__unused__)), expression rhs __attribute__ ((__unused__))) {
  return pts_to_set;
}

/* one basic case of Emami: < m->x =&y > */
set basic_ptr_to_field_addr(set pts_to_set, expression lhs, expression rhs) {
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
  list l = NIL;
  ifdebug(1) {
    pips_debug(1, " case m->x = &y\n");
  }

  effect e1 = effect_undefined;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  l = generic_proper_effects_of_complex_address_expression(copy_expression(lhs), &e1,
							   true);
  effects_free(l);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);
  // creation of the sink
  effect e2 = effect_undefined;
  set_methods_for_proper_simple_effects();
  l = generic_proper_effects_of_complex_address_expression(copy_expression(rhs),
							   &e2, false);
  effects_free(l);
  generic_effects_reset_all_methods();
  ref2 = effect_any_reference(e2);
  sink = make_cell_reference(ref2);
  // creation of the approximation
  rel = make_approximation_exact();
  // creation of the points_to relation
  pt = make_points_to(source, sink, rel, make_descriptor_none());
  // add the points_to approximation to the set generated
  //by this assignement
  gen_pts_to = set_add_element(gen_pts_to, gen_pts_to, (void*) pt);
  // creation of the written set
  // search of all the points_to relations in the
  // alias set where the source is equal to the lhs
  SET_FOREACH(points_to, i,pts_to_set) {
    if(locations_equal_p(points_to_source(i),source))
      written_pts_to = set_add_element(written_pts_to,
				       written_pts_to,
				       (void *)i);
  }

  pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
  pts_to_set = set_union(pts_to_set, pts_to_set, gen_pts_to);

  ifdebug        (1) {
    print_points_to_set("points To pour le cas 2 <m->x = &y> \n ",
			pts_to_set);
  }

  return pts_to_set;
}

/* one basic case of Emami: <  m->x = m->y > */
set basic_ptr_to_field_ptr_to_field(set pts_to_set, expression lhs,
				    expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;
 
  effect e1 = effect_undefined, e2 = effect_undefined;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
								 &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
								 &e2, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);

  // add the points_to relation to the set generated
  // by this assignement
  reference ref = copy_reference(effect_any_reference(e2));
  sink = make_cell_reference(ref);
  set s = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  SET_FOREACH(points_to, i, pts_to_set) {
    if(locations_equal_p(points_to_source(i), sink))
      s = set_add_element(s, s, (void*)i);
  }
  SET_FOREACH(points_to, j, s) {
    new_sink = copy_cell(points_to_sink(j));
    // access new_source = copy_access(source);
    rel = points_to_approximation(j);
    pt_to = make_points_to(source, new_sink, rel,
			   make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
				 (void*) pt_to );
  }

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
  ifdebug        (1)
    print_points_to_set("Points to pour le cas 1 <m->x =m-> y>\n",
			pts_to_set);
  
  return pts_to_set;
}

/* one basic case of Emami: < *x =m->y > */
set basic_deref_ptr_to_field(set pts_to_set, expression lhs, expression rhs) {
  set s1 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s2 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set s3 = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set change_pts_to = set_generic_make(set_private, points_to_equal_p,
				       points_to_rank);
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_source = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1) {
    pips_debug(1, " case *x = m;y\n");
  }
 
 

  effect e1 = effect_undefined, e2 = effect_undefined;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
								 &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
								 &e2, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  //            ent1 = argument_entity(lhs_tmp);

  source = make_cell_reference(ref1);
  // recuperation of y
  ref2 = effect_any_reference(e2);
  sink = make_cell_reference(ref2);

  /* creation of the set written_pts_to =
     {(x1,x2,rel)| (x, x1, EXACT), (x1, x2, rel) /in pts_to_set}*/
  SET_FOREACH(points_to, i, pts_to_set) {
    if( locations_equal_p(points_to_source(i), source) &&
	approximation_exact_p(points_to_approximation(i))) {
      SET_FOREACH(points_to, j,pts_to_set ) {
	if( locations_equal_p(points_to_source(j) ,
			      points_to_sink(i)))
	  written_pts_to = set_add_element(written_pts_to,
					   written_pts_to, (void *)j);
      }
    }
  }
  /* {(x1, x2,EXACT)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
  SET_FOREACH(points_to, k, pts_to_set) {
    if( locations_equal_p(points_to_source(i), source)
	&& approximation_may_p(points_to_approximation(k))) {
      SET_FOREACH(points_to, h,pts_to_set ) {
	if(locations_equal_p(points_to_source(h),points_to_sink(k))&&
	   approximation_exact_p(points_to_approximation(h)))
	  s2 = set_add_element(s2, s2, (void *)h);
      }
    }
  }

  /* {(x1, x2,MAY)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
  SET_FOREACH(points_to, l, pts_to_set) {
    if(locations_equal_p(points_to_source(l), source) &&
       approximation_may_p(points_to_approximation(l))) {
      SET_FOREACH(points_to, m,pts_to_set) {
	if(locations_equal_p( points_to_source(m),points_to_sink(l))&&
	   approximation_exact_p(points_to_approximation(m))) {
	  points_to_approximation(m) = make_approximation_may();
	  s3 = set_add_element(s3, s3, (void *)m);
	}
      }
    }
  }
  change_pts_to = set_difference(change_pts_to, pts_to_set, s3);
  change_pts_to = set_union(change_pts_to, change_pts_to, s3);
  SET_FOREACH(points_to, n, pts_to_set) {
    if(locations_equal_p(points_to_source(n), source)) {
      SET_FOREACH(points_to, o, pts_to_set) {
	if(locations_equal_p(points_to_source(o) , sink)) {
	  new_source = copy_cell(points_to_sink(n));
	  new_sink = copy_cell(points_to_sink(o));
	  rel = fusion_approximation(points_to_approximation(n),
				     points_to_approximation(o));
	  pt_to = make_points_to(new_source, new_sink, rel,
				 make_descriptor_none());
	  gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
				       (void *)pt_to);
	}
      }
    }
  }
  s1 = set_difference(s1, change_pts_to, written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, s1);
  ifdebug        (1)
    print_points_to_set("Points to pour le cas 1 <x = y.a>\n",
			pts_to_set);

  return pts_to_set;
}

/* one basic case of Emami: < m->x = y.a > */
set basic_ptr_to_field_field(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent2 = entity_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;

  ifdebug(1) printf("\n cas x = y \n");
  
  effect e1 = effect_undefined, e2 = effect_undefined;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
								 &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
								 &e2, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);

  // add the points_to relation to the set generated
  // by this assignement
  ref2 = effect_any_reference(e2);
  sink = make_cell_reference(ref2);
  set s = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  SET_FOREACH(points_to, i, pts_to_set) {
    if(locations_equal_p(points_to_source(i), sink))
      s = set_add_element(s, s, (void*)i);
  }
  SET_FOREACH(points_to, j, s) {
    new_sink = copy_cell(points_to_sink(j));
    // locations new_source = copy_access(source);
    rel = points_to_approximation(j);
    pt_to = make_points_to(source, new_sink, rel,
			   make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
				 (void*) pt_to );
  }
  /* in case x = y[i]*/
  if (array_entity_p(ent2)) {
    new_sink = make_cell_reference(copy_reference(ref2));
    // locations new_source = copy_access(source);
    rel = make_approximation_exact();
    pt_to = make_points_to(source, new_sink, rel,
			   make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
				 (void*) pt_to);
  }

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
  ifdebug        (1)
    print_points_to_set("Points to pour le cas 1 <x = y>\n",
			pts_to_set);
  return pts_to_set;
}

/* one basic case of Emami: < m->x = m > */
set basic_ptr_to_field_struct(set pts_to_set, expression lhs __attribute__ ((__unused__)), expression rhs __attribute__ ((__unused__))) {
  pips_internal_error("<case m->x = *y> not implemented yet ");
  return pts_to_set;
}

/* one basic case of Emami: < m->x = *y > */

set basic_ptr_to_field_deref(set pts_to_set __attribute__ ((__unused__)),
			     expression lhs __attribute__ ((__unused__)),
			     expression rhs __attribute__ ((__unused__)))
{
	pips_internal_error("<case m->x = *y> not implemented yet ");
    return NULL;

}

/* one basic case of Emami: < m->x = y > */
set basic_ptr_to_field_ref(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  points_to pt_to = points_to_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  // entity ent2 = entity_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  cell new_sink = cell_undefined;
  approximation rel = approximation_undefined;
 
  ifdebug(1) printf("\n cas m->x = y \n");
  
  effect e1 = effect_undefined, e2 = effect_undefined;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
								 &e1, true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
								 &e2, false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  source = make_cell_reference(ref1);

  // add the points_to relation to the set generated
  // by this assignement
  ref2 = effect_any_reference(e2);
  sink = make_cell_reference(ref2);
  set s = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  SET_FOREACH(points_to, i, pts_to_set) {
    if(locations_equal_p(points_to_source(i), sink))
      s = set_add_element(s, s, (void*)i);
  }
  SET_FOREACH(points_to, j, s) {
    new_sink = copy_cell(points_to_sink(j));
    // locations new_source = copy_access(source);
    rel = points_to_approximation(j);
    pt_to = make_points_to(source, new_sink, rel,
			   make_descriptor_none());
    gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
				 (void*) pt_to );
  }
  /* in case x = y[i]*/
  /* if (array_entity_p(ent2)) { */
/*     new_sink = make_cell_reference(copy_reference(ref2)); */
/*     // access new_source = copy_access(source); */
/*     rel = make_approximation_exact(); */
/*     pt_to = make_points_to(source, new_sink, rel, */
/* 			   make_descriptor_none()); */
/*     gen_pts_to = set_add_element(gen_pts_to, gen_pts_to, */
/* 				 (void*) pt_to); */
/*   } */

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
  ifdebug        (1)
    print_points_to_set("Points to pour le cas 1 <x = y>\n",
			pts_to_set);
  return pts_to_set;
}

/* to compute m.a = n.a where a is of pointer type*/
set struct_pointer(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  effect e1 = effect_undefined, e2 = effect_undefined;
  /* init the effect's engine.*/
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs, &e1,
								 true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs, &e2,
								 false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  reference ref1 = effect_any_reference(e1);
  // print_reference(ref1);
  cell source = make_cell_reference(ref1);
  // add the points_to relation to the set generated
  // by this assignement
  reference ref2 = effect_any_reference(e2);
  // print_reference(ref2);
  cell sink = make_cell_reference(ref2);
  set s = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  SET_FOREACH(points_to, i, pts_to_set)
    {
      if(locations_equal_p(points_to_source(i), sink))
	s = set_add_element(s, s, (void*)i);
    }
  SET_FOREACH(points_to, j, s)
    {
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
    print_points_to_set("Points to pour le cas 1 <x = y>\n",
			pts_to_set);

  return pts_to_set;

}

/* to compute m.a = n.a where a is of pointer type*/
set struct_double_pointer(set pts_to_set, expression lhs, expression rhs) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);
  effect e1 = effect_undefined, e2 = effect_undefined;
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
    reference ref1 = effect_any_reference(e1);
    cell source = make_cell_reference(copy_reference(ref1));
    // add the points_to relation to the set generated
    // by this assignement
    reference ref2 = effect_any_reference(e2);
    cell sink = make_cell_reference(copy_reference(ref2));
    set
      s = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
    SET_FOREACH(points_to, i, pts_to_set)
      {
	if(locations_equal_p(points_to_source(i), sink))
	  s = set_add_element(s, s, (void*)i);
      }
    SET_FOREACH(points_to, j, s)
      {
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
    ifdebug    (1)
      print_points_to_set("Points to pour le cas 1 <x = y>\n",
			  pts_to_set);
  } else
    pips_internal_error("Don't know how to handle this kind of expression");

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

/* first version of the treatment of the heap : x = ()malloc(sizeof()) */
set basic_ref_heap(set pts_to_set, expression lhs, expression rhs __attribute__ ((__unused__)) ,
		   statement current) {
  set gen_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set written_pts_to = set_generic_make(set_private, points_to_equal_p,
					points_to_rank);

  points_to pt_to = points_to_undefined;
  syntax syn1 = expression_syntax(lhs);
  reference ref1 = syntax_reference(syn1);
  reference ref2 = reference_undefined;
  cell source = cell_undefined;
  cell sink = cell_undefined;
  approximation rel = approximation_undefined;
 
  ifdebug(1) {
    pips_debug(1, " case  x =()malloc(sizeof()) \n");
  }

  effect e1 = effect_undefined;
  type lhst = expression_to_type(lhs);
  type pt = type_to_pointed_type(lhst);

  //set_methods_for_proper_simple_effects();
  set_methods_for_simple_pointer_effects();

  //set_methods_for_proper_references();
  list l1 = generic_proper_effects_of_complex_address_expression(copy_expression(lhs), /*  */
								 &e1, true);
  effects_free(l1);
  generic_effects_reset_all_methods();
  ref1 = effect_any_reference(e1);
  ref2 = original_malloc_to_abstract_location(copy_reference(ref1), pt, type_undefined,
				     expression_undefined, get_current_module_entity(),
				     statement_number(current));

  source = make_cell_reference(copy_reference(ref1));

  // creation of the sink
  sink = make_cell_reference(copy_reference(ref2));
  // fetch the points to relations
  // where source = source 1
  // creation of the written set
  // search of all the points_to relations in the
  // alias set where the source is equal to the lhs
  SET_FOREACH(points_to, elt, pts_to_set)
    {
      if(locations_equal_p(points_to_source(elt),
			   source))
	written_pts_to = set_add_element(written_pts_to,
					 written_pts_to,(void*)elt);
    }

  rel = make_approximation_exact();
  pt_to = make_points_to(source, sink, rel, make_descriptor_none());
  points_to_consistent_p(pt_to);
  set_add_element(gen_pts_to, gen_pts_to, (void *) pt_to);
  // gen_consistent_p(gen_pts_to);
  set_difference(pts_to_set, pts_to_set, written_pts_to);
  set_union(pts_to_set, gen_pts_to, pts_to_set);
  ifdebug        (1) {
    print_points_to_set("Points To pour le cas 3 <x ==()malloc(sizeof()) > \n",
			pts_to_set);
  }
 
  return pts_to_set;
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
			 set pt_in) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  //pt_out = set_assign(pt_out,pt_in);
  int rlt1 = 0;
  int rlt2 = 0;
  expression er = expression_undefined;
  expression el = expression_undefined, ex = expression_undefined;
  call c1 = call_undefined, c2 = call_undefined;
  list args = NIL, l = NIL;
  if (instruction_expression_p(statement_instruction(current))) {
    //expression e = instruction_expression(statement_instruction(current));
    ;
  }
  // FI: lhs and rhs are going to be much more general than simple references...

  rlt1 = emami_expression_type(lhs);
  rlt2 = emami_expression_type(rhs);
  switch (rlt1) {
    /* cas x = y
       write = {(x, x1, rel)|(x, x1, rel) in input}
       gen  = {(x, y1, rel)|(y, y1, rel) in input }
       pts_to=gen+(input - kill)
    */
  case EMAMI_NEUTRAL: {
    switch (rlt2) {
    case EMAMI_NEUTRAL: {
      /* x and y are references : case x = y*/
      if (expression_field_p(lhs)) {
	l = call_arguments(expression_call(lhs));
	el = EXPRESSION(CAR(CDR(l)));
      } else
	el = copy_expression(lhs);

      if (expression_field_p(rhs)) {
	args = call_arguments(expression_call(er));
	rhs = EXPRESSION(CAR(CDR(args)));
      } else
	er = copy_expression(rhs);
      if (expression_reference_p(el) && expression_reference_p(er)
	  && ((expression_pointer_p(el) && expression_pointer_p(er))
	      || (expression_double_pointer_p(el)
		  && expression_double_pointer_p(er))))

	 set_assign(pt_out, basic_ref_ref(pt_in, el, copy_expression(rhs)));
    }
      break;
      
    case EMAMI_ADDRESS_OF: {
     /* x is a reference, y is & : case x = &y*/
      expression e =expression_undefined;
      c2 = expression_call(rhs);
      args = call_arguments(c2);
      er = EXPRESSION(CAR(args));
      e = copy_expression(er);
      /* we will take later care of array cases...*/

      /* if(array_argument_p(rhs_tmp)){ */
      /* pts_to_set = set_assign(pts_to_set, basic_ref_array(pts_to_set, */
      /* lhs, */
      /*rhs_tmp)); */
      /*       } */
      if(expression_call_p(er) && ENTITY_POINT_TO_P(call_function(expression_call(er)))){
	args = call_arguments(expression_call(er));
	er = EXPRESSION(CAR(CDR(args)));
      }
      if (expression_reference_p(lhs) && (expression_reference_p(er)||array_argument_p(er)) && expression_pointer_p(lhs))
	set_assign(pt_out, basic_ref_addr(pt_in, lhs,e));
      //pt_out = basic_ref_addr(pt_in, lhs, er);
    }
      break;
    case EMAMI_DEREFERENCING: {
    /* x is a reference, y is *, case x = *y */

      if (expression_call_p(rhs)) {
	c2 = expression_call(rhs);
	/*Maybe we should test if it's a call to & even if it's done
	  a first time by emami_expression_type().*/
	args = call_arguments(c2);
	er = EXPRESSION(CAR(args));
	if (expression_reference_p(lhs) && expression_pointer_p(lhs)
	    && expression_reference_p(er)
	    && expression_double_pointer_p(er))
	  set_assign(pt_out, basic_ref_deref(pt_in, lhs, er));
      }
    }
      break;
    case EMAMI_HEAP: {
     /* x is a reference, y is a call to malloc, x = ()malloc */
    
  if (expression_reference_p(lhs))
	// && (expression_pointer_p(lhs)|| array_argument_p(lhs)))
	 set_assign(pt_out, basic_ref_heap(pt_in, lhs, rhs,
						   current));
      break;
    }
    case EMAMI_FIELD: {
   /* x is a reference, y is m.y, case x = m.y */
      c2 = expression_call(rhs);
      args = call_arguments(c2);
      er = EXPRESSION(CAR(CDR(args)));

      if (expression_reference_p(lhs) && expression_reference_p(er)
	  && expression_pointer_p(lhs) && (expression_pointer_p(er)
					   || array_argument_p(er)))
	 set_assign(pt_out, basic_ref_field(pt_in, lhs, er));
    }
      break;
    case EMAMI_POINT_TO: {
      /* x is a reference y is m->, case x = m->y */
      c2 = expression_call(rhs);
      args = call_arguments(c2);
      er = EXPRESSION(CAR(CDR(args)));
      if (expression_reference_p(lhs) && expression_reference_p(er)
	  && expression_pointer_p(lhs) && (expression_pointer_p(er)
					   || array_argument_p(er)))
	 set_assign(pt_out, basic_ref_ptr_to_field(pt_in, lhs,
							   rhs));
    }
      break;

    case EMAMI_STRUCT:
      /* x is a reference y is a struct.
	 we will return an empty set, so the assignment will be
	 treated by points_to_general_assignment()*/
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini\n ");
      // should we add a function that just return the same input set ?
      break;
    case EMAMI_ARRAY: {
      /* x is a reference y is an array, case x = y[] */
      if (expression_reference_p(lhs) && array_argument_p(rhs)
	  && expression_pointer_p(lhs))
	 set_assign(pt_out, basic_ref_array(pt_in, lhs, rhs));
    }
      break;
    default:
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini \n");
      break;
    }
  }
    break;
  case EMAMI_DEREFERENCING: {
    switch (rlt2) {
    case EMAMI_NEUTRAL: {
      /* x is * y is a reference, *x = y.
	 we have to test if it's a simple dereferencing or a multiple
	 levels of dereferencing*/

      c1 = expression_call(lhs);
      args = call_arguments(c1);
      el = EXPRESSION(CAR(args));
      ex = copy_expression(el);
      if (expression_field_p(el)) {
	/* if we have *m.x = y */
	c1 = expression_call(el);
	l = call_arguments(c1);
	ex = EXPRESSION(CAR(CDR(l)));
      }
      if (expression_reference_p(ex) && expression_pointer_p(ex)
	  && expression_reference_p(rhs))
	 set_assign(pt_out, basic_deref_ref(pt_in, ex, rhs));

    }
      break;
    case EMAMI_ADDRESS_OF: {
      /* x is * y is &, case *x = &y.*/
      /* we test if we have a field at the left side. */
      c1 = expression_call(lhs);
      args = call_arguments(c1);
      expression el = EXPRESSION(CAR(args));
      expression ex = copy_expression(el);
      /* we test if we have a field at the left side. */
      if (expression_field_p(ex)) {

	call c = expression_call(ex);
	args = call_arguments(c);
	ex = EXPRESSION(CAR(CDR(args)));
      }
      /*Since we have &y at the right side we have to get the
	first argument and to test if it's a reference or an array.*/
      er = copy_expression(rhs);
      if (expression_call_p(rhs)) {
	c2 = expression_call(rhs);
	l = call_arguments(c2);
	er = EXPRESSION(CAR(l));
      }
      if (expression_reference_p(ex) && expression_reference_p(er)
	  && expression_double_pointer_p(ex) && array_argument_p(er)) {
	set_assign(pt_out, basic_deref_array(pt_in, el, er));

      }
      if (expression_reference_p(ex) && expression_reference_p(er)
	  && expression_double_pointer_p(ex) && !array_argument_p(er))
	set_assign(pt_out, basic_deref_addr(pt_in, ex, er));
    }
      break;
    case EMAMI_DEREFERENCING: {
      /* x is * y is *, case *x = *y. */
      c1 = expression_call(lhs);
      args = call_arguments(c1);
      el = EXPRESSION(CAR(args));
      // recuperation of y

      c2 = expression_call(rhs);
      l = call_arguments(c2);
      er = EXPRESSION(CAR(l));
      if (expression_reference_p(el) && expression_reference_p(er)
	  && expression_double_pointer_p(el) && expression_pointer_p(
								     er))
	set_assign(pt_out, basic_deref_deref(pt_in, el, er));
    }
      break;
    case EMAMI_FIELD: {
      /* x is * y is m.y, case *x = m.y */
      c1 = expression_call(lhs);
      args = call_arguments(c1);
      el = EXPRESSION(CAR(args));
      ex = copy_expression(el);
      /* if we have *m.x = y */
      if (expression_field_p(el)) {
	l = call_arguments(expression_call(el));
	ex = EXPRESSION(CAR(CDR(l)));
      }
      // recuperation of y

      c2 = expression_call(rhs);
      l = call_arguments(c2);
      er = EXPRESSION(CAR(l));
      expression e = copy_expression(er);

      if (expression_field_p(er)) {
	l = call_arguments(expression_call(er));
	e = EXPRESSION(CAR(CDR(l)));
      }
      if (expression_reference_p(ex) && expression_reference_p(e)
	  && expression_double_pointer_p(ex) && expression_pointer_p(
								     e))
	set_assign(pt_out, basic_deref_field(pt_in, ex, e));
    }
      break;
    case EMAMI_POINT_TO: {
      /* x is * y is m->y, case *x = m->y */
      c1 = expression_call(lhs);
      args = call_arguments(c1);
      el = EXPRESSION(CAR(args));
      ex = copy_expression(el);

      if (expression_field_p(el)) {
	l = call_arguments(expression_call(el));
	ex = EXPRESSION(CAR(CDR(l)));
      }
      // recuperation of y
      c2 = expression_call(rhs);
      l = call_arguments(c2);
      er = EXPRESSION(CAR(l));
      expression e = copy_expression(er);

      if (expression_field_p(er)) {
	l = call_arguments(expression_call(er));
	e = EXPRESSION(CAR(CDR(l)));
      }
      if (expression_reference_p(ex) && expression_reference_p(e)
	  && expression_double_pointer_p(ex) && expression_pointer_p(
								     e))
	set_assign(pt_out, basic_deref_ptr_to_field(pt_in,
							     ex, e));
    }
      break;
    case EMAMI_STRUCT:
      /* x is * y is a struct, case *x = m */
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini\n ");
      break;
    case EMAMI_ARRAY:
      /* x is * y is an array , case *x = y[i] */
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini\n ");
      break;
    default:
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini\n ");
      break;
    }
  }

    break;
  case EMAMI_FIELD: {
    switch (rlt2) {
    case EMAMI_NEUTRAL: {
      /* x is field y is a reference, case x.a = y */
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini\n ");
    }
      break;
    case EMAMI_ADDRESS_OF: {
      /* x is a field, y is &, case x.a = &y */
      
      c2 = expression_call(rhs);
      args = call_arguments(c2);
      er = EXPRESSION(CAR(args));
      c1 = expression_call(lhs);
      l = call_arguments(c1);
      el = EXPRESSION(CAR(CDR(l)));
      if (expression_reference_p(el) && expression_reference_p(er)
	  && expression_pointer_p(el))
	set_assign(pt_out, basic_field_addr(pt_in, lhs, er));
    }
      break;
    case EMAMI_DEREFERENCING: {
   /* x is a field, y is *, case x.a = *y */
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini\n ");
    }
      break;
    case EMAMI_FIELD: {
      /* x is a field, y is a field , case x.a = m.y */
      if (expression_field_p(lhs)) {
	l = call_arguments(expression_call(lhs));
	el = EXPRESSION(CAR(CDR(l)));
      }else
	el = copy_expression(lhs);

      if (expression_field_p(rhs)) {
	args = call_arguments(expression_call(rhs));
	er = EXPRESSION(CAR(CDR(args)));
      }else
	er = copy_expression(rhs);
      if (expression_reference_p(el) && expression_reference_p(er)
	  && ((expression_pointer_p(el) && expression_pointer_p(er))
	      || (expression_double_pointer_p(el)
		  && expression_double_pointer_p(er))))

	set_assign(pt_out, basic_ref_ref(pt_in, el, er));
    }
      break;
    case EMAMI_POINT_TO: {
      /* x is a field, y is a pointer to field , case x.a = m->y */
      c2 = expression_call(rhs);
      args = call_arguments(c2);
      er = EXPRESSION(CAR(args));
      c1 = expression_call(lhs);
      l = call_arguments(c1);
      el = EXPRESSION(CAR(CDR(l)));
      if (expression_reference_p(el) && expression_reference_p(er)
	  && expression_pointer_p(el))
	set_assign(pt_out, basic_field_ptr_to_field(pt_in,
							     el, er));
    }
      break;
    case EMAMI_STRUCT:
      /* x is a field, y is a struct , case x.a = m */
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini\n ");
      break;
    default:
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini\n ");
      break;
    }
  }
    break;
  case EMAMI_STRUCT:
    {
      switch (rlt2) {
      case EMAMI_NEUTRAL: {
	/* x is a struct, y is a reference */
	ifdebug(1)
	  fprintf(stderr, "\n aucun pattern defini\n ");
      }
	break;
      case EMAMI_ADDRESS_OF:
	/* x is a struct, y is a & */
	ifdebug(1)
	  fprintf(stderr, "\n aucun pattern defini\n ");
	break;
      case EMAMI_DEREFERENCING:
	/* x is a struct, y is a *. */
	ifdebug(1)
	  fprintf(stderr, "\n aucun pattern defini\n ");
	break;
      case EMAMI_FIELD:
	/* x is a struct, y is a field. */
	ifdebug(1)
	  fprintf(stderr, "\n aucun pattern defini\n ");
	break;
      case EMAMI_STRUCT:
	/* x is a struct, y is a struct. */
	pt_out = struct_decomposition(lhs, rhs, pt_in);
	break;
      default:
	ifdebug(1)
	  fprintf(stderr, "\n aucun pattern defini\n ");
	break;

      }
    }
    break;
  case EMAMI_POINT_TO: {
    switch (rlt2) {
    case EMAMI_NEUTRAL: {
      /* x is pointer to field, y is a reference, case m->x = y. */
      c1 = expression_call(lhs);
      l = call_arguments(c1);
      el = EXPRESSION(CAR(CDR(l)));
    
      if (expression_reference_p(el) && expression_reference_p(rhs))
	set_assign(pt_out, basic_ptr_to_field_ref(pt_in, lhs,
							   rhs));
    }
      break;
    case EMAMI_ADDRESS_OF:
      {	/* x is pointer to field, y is a &, case m->x = &y. */

	c2 = expression_call(rhs);
	args = call_arguments(c2);
	er = EXPRESSION(CAR(args));
	c1 = expression_call(lhs);
	l = call_arguments(c1);
	el = EXPRESSION(CAR(CDR(l)));
	if (expression_reference_p(el) && expression_reference_p(er)
	    && expression_pointer_p(el))
	  set_assign(pt_out, basic_ptr_to_field_addr(pt_in,
							      copy_expression(lhs), er));
      }
      break;
    case EMAMI_DEREFERENCING: {
      /* x is pointer to field, y is a *, case m->x = *y. */
      pt_out = set_assign(pt_out, basic_ptr_to_field_deref(pt_in,
							   lhs, rhs));
    }
      break;
    case EMAMI_FIELD: {
      /* x is pointer to field, y is a field, case m->x = m.y */
      if (expression_call_p(lhs)) {
	c1 = expression_call(lhs);
	if (entity_an_operator_p(call_function(c1), POINT_TO)) {
	  l = call_arguments(c1);
	  el = EXPRESSION(CAR(CDR(l)));
	}
      }
      if (expression_field_p(rhs)) {
	args = call_arguments(expression_call(rhs));
	er = EXPRESSION(CAR(CDR(args)));
      }
      /* May be we should later test if rhs is an array.*/
      if ((expression_reference_p(el) && expression_reference_p(er))
	  &&((expression_pointer_p(el) && expression_pointer_p(er))
	     || (expression_double_pointer_p(el)
		 && expression_double_pointer_p(er))))
	set_assign(pt_out, basic_ptr_to_field_field(pt_in,
							     el, er));
    }
      break;
    case EMAMI_STRUCT:
      /* x is pointer to field, y is a struct, case m->x = n */
      pt_out = set_assign(pt_out, basic_ptr_to_field_struct(pt_in,
							    lhs, rhs));

      break;
    case EMAMI_POINT_TO: {
      /* x is pointer to field, y is a pointer to field, case m->x = n->y */
      args = call_arguments(expression_call(lhs));
      el = EXPRESSION(CAR(CDR(args)));
      l = call_arguments(expression_call(rhs));
      er = EXPRESSION(CAR(CDR(l)));
      if (expression_reference_p(el) && expression_reference_p(er)
	  && ((expression_pointer_p(el) && expression_pointer_p(
								er)) || (expression_double_pointer_p(el)
									 && expression_double_pointer_p(er))))
	set_assign(pt_out, basic_ptr_to_field_ptr_to_field(
								    pt_in, copy_expression(lhs), copy_expression(rhs)));
    }
      break;
    default:
      ifdebug(1)
	fprintf(stderr, "\n aucun pattern defini\n ");
      break;
    }
  }
    break;
  default:
    ifdebug(1)
      fprintf(stderr, "\n aucun pattern defini\n ");
    break;
  }

  return pt_out;
}

/* Generate new points-to for any assignment of a pointer, side
 * effect on pt_cur.
 *
 * If lhs is not a pointer, return pt_cur.
 *
 * Else we test the type of rhs :
 *  -  if it's  a user call and/or contains a user call, since we are
 *     using an intraprocedural analysis (until we implement an
 *     interprocedural one...), generate (c, anyhwere, may);
 *     remove all points-to having lhs as source;
 *
 *  -   if it's an intrinsic call to & and if the argument of & is a
 *       reference that can be converted into a cell we call basic_ref_addr()
 *      else we generate (c, anywhere, MAY)
 *
 *  -   if rhs a reference that can be converted into a cell we call
 *       basic_ref_ref()
 *
 *  -  else we update pt_cur by  using  effects through the call to
 *    points_to_filter_with_effects()
 * The points-to triple generated by this algorithm are much more
 * tricky to handle than Emami's because they may involved
 * indirections. They are not constant references. They must be
 * checked against the write effects before they are propagated from
 * one statement to the next. This must be dealt with at the sequence
 * level at least.
 *
 * Note: anywhere may depend on the type is the proprety
 * ALIASING_ACROSS_TYPE is set to FALSE.
 */
set points_to_general_assignment(statement s __attribute__ ((__unused__)), expression lhs, expression rhs,
				 set pt_cur, list el) {
  points_to npt = points_to_undefined;

  /* lhs is not of type pointer, we return the input poins-to set */
  if (!expression_pointer_p(lhs)) {
    return pt_cur;
    /* lhs is of type pointer so a points-to analysis can be started*/
  } else if (expression_reference_p(lhs)) {
    /* lhs can be converted into a cell in aim to  generate
     * points-to relations*/
    reference r = expression_reference(lhs);
    entity p = reference_variable(r);
    cell c = make_cell_reference(r);
    npt = points_to_anywhere(c);
    /*rhs is a call toa user function */
    if (user_function_call_p(rhs)) {
      SET_FOREACH(points_to, pt, pt_cur) {
	cell ptc = points_to_source(pt);
	reference ptr = cell_to_reference(ptc);
	entity sp = reference_variable(ptr);
	/* remove all the old points-to in which lhs appears as a source*/
	if(sp==p)
	  pt_cur = set_del_element(pt_cur, pt_cur, (void*)pt);
      }
      /* add the anywhere points-to*/
      pt_cur = set_add_element(pt_cur, pt_cur, (void*) npt);
    } else if (expression_call_p(rhs)) {
      call cl = expression_call(rhs);
      if (entity_an_operator_p(call_function(cl), ADDRESS_OF)) {
	list args = call_arguments(cl);
	expression e = EXPRESSION(CAR(args));
	if (expression_reference_p(e)) {
	  pt_cur = set_assign(pt_cur,
			      basic_ref_addr(pt_cur,copy_expression(lhs),copy_expression(rhs)));
	} else {
	  SET_FOREACH(points_to, pt, pt_cur) {
	    cell ptc = points_to_source(pt);
	    reference ptr = cell_to_reference(ptc);
	    entity sp = reference_variable(ptr);
	    /* remove all the old points-to in which lhs appears as a source*/
	    if(sp==p)
	      pt_cur = set_del_element(pt_cur, pt_cur, (void*)pt);
	  }
	  /* add the anywhere points-to*/
	  pt_cur = set_add_element(pt_cur, pt_cur, (void*) npt);
	}
      } else if (expression_reference_p(rhs)) {
	/* May be we should also add a test if rhs and lhs are fields of a struct.*/
	pt_cur = set_assign(pt_cur, basic_ref_ref(pt_cur,copy_expression(lhs),copy_expression(rhs)));

      }
    } else{
      /* update points-to set by using effects. */
      pt_cur = points_to_filter_with_effects(pt_cur, el);
    }
  } else
    pt_cur = points_to_filter_with_effects(pt_cur, el);
  return pt_cur;
  //pips_internal_error("To be implemented!");
      return NULL;
}

/* compute the points to set associate to a sequence of statements*/
set points_to_sequence(sequence seq, set pt_in, bool store) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set_assign(pt_out, pt_in);
  FOREACH(statement, st, sequence_statements(seq)) {
    set_assign(pt_out, points_to_recursive_statement(st,pt_out,store));
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
    pt_cur = set_union(pt_cur, pt_cur, points_to_expression(copy_expression(ex), pt_cur, TRUE));
  }

  if (ENTITY_ASSIGN_P(e)) {
    lhs = EXPRESSION(CAR(pc));
    rhs = EXPRESSION(CAR(CDR(pc)));
    set_assign(pt_out,points_to_assignment(s, copy_expression(lhs),copy_expression( rhs), pt_cur));
    
    if (set_empty_p(pt_out)) {
      /* Use effects to remove broken points-to relations and to link
	 broken pointers towards default sinks.
	 pt_out = points_to_filter_with_expression_effects(e, pt_cur);
      */
      //pt_out = points_to_filter_with_effects(pt_cur, exp);
      //pt_out = points_to_filter_with_expression_effects(exp, pt_cur);
      pt_out = points_to_general_assignment(s, copy_expression(lhs),copy_expression(rhs), pt_cur, el);
    }
  } else if (ENTITY_PLUS_UPDATE_P(e) || ENTITY_MINUS_UPDATE_P(e)
	     || ENTITY_MULTIPLY_UPDATE_P(e) || ENTITY_DIVIDE_UPDATE_P(e)
	     || ENTITY_MODULO_UPDATE_P(e) || ENTITY_LEFT_SHIFT_UPDATE_P(e)
	     || ENTITY_RIGHT_SHIFT_UPDATE_P(e) || ENTITY_BITWISE_AND_UPDATE_P(e)
	     || ENTITY_BITWISE_XOR_UPDATE_P(e) || ENTITY_BITWISE_OR_UPDATE_P(e)) {
    /* Look at the lhs with
       generic_proper_effects_of_complex_address_expression(). If the
       main effect is an effect on a pointer, occurences of this
       pointer must be removed from pt_cur to build pt_out.
    */

    lhs = EXPRESSION(CAR(pc));
    pt_cur = set_union(pt_cur, pt_cur, points_to_expression(copy_expression(lhs), pt_cur,
							    TRUE));
    pt_out = points_to_filter_with_effects(pt_cur, el);
  } else if (ENTITY_POST_INCREMENT_P(e) || ENTITY_POST_DECREMENT_P(e)
	     || ENTITY_PRE_INCREMENT_P(e) || ENTITY_PRE_DECREMENT_P(e)) {
    /* same */
    lhs = EXPRESSION(CAR(pc));
    pt_cur = set_union(pt_cur, pt_cur, points_to_expression(copy_expression(lhs), pt_cur,
							    TRUE));
    pt_out = points_to_filter_with_effects(pt_cur, el);
  } else if (ENTITY_STOP_P(e) || ENTITY_ABORT_SYSTEM_P(e)
	     || ENTITY_EXIT_SYSTEM_P(e) || ENTITY_C_RETURN_P(e)) {
    /* The call is never returned from. No information is available
       for the dead code that follows. pt_out is already set to the
       empty set. */
    ;
  } else if (ENTITY_COMMA_P(e)) {
    /* FOREACH on the expression list, points_to_expressions() */
    FOREACH    (EXPRESSION, ex, pc) {
      pt_out = set_union(pt_out, pt_out, points_to_expression(copy_expression(ex), pt_cur, TRUE));
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
  FOREACH(EFFECT, e, el) {
    if(effect_pointer_type_p(e) && effect_write_p(e)) {

      cell c = effect_cell(e);
      /* Theheap problem with the future extension to GAP is hidden
	 within cell_to_reference */
      reference r = cell_to_reference(c);

      if(ENDP(reference_indices(r))) {
	/* Simple case: a scalar pointer is written */
	entity p = reference_variable(r);
	points_to npt = points_to_undefined;

	/* Remove previous targets */
	SET_FOREACH(points_to, pt, pts) {
	  cell ptc = points_to_source(pt);
	  reference ptr = cell_to_reference(ptc);
	  entity sp = reference_variable(ptr);

	  if(sp==p)
	    pts = set_del_element(pts, pts, (void*)pt);
	}

	/* add the anywhere points-to*/
	npt = points_to_anywhere(copy_cell(c));
	pts = set_add_element(pts, pts, (void*) npt);
      }
      else {
	/* Complex case: is the reference usable with the current pts?
	 * If it uses an indirection, check that the indirection is
	 * not thru nowhere, not thru NULL, and maybe not thru
	 * anywhere...
	 */
	points_to npt = points_to_undefined;

	/* Remove previous targets */
	SET_FOREACH(points_to, pt, pts) {
	  cell ptc = points_to_source(pt);
	  reference ptr = cell_to_reference(ptc);

	  if(reference_equal_p(r, ptr))
	    pts = set_del_element(pts, pts, (void*)pt);
	}

	/* add the anywhere points-to*/
	npt = points_to_anywhere(copy_cell(c));
	pts = set_add_element(pts, pts, (void*) npt);
	//pips_internal_error("Complex pointer write effect."
	//" Not implemented yet\n");
      }
    }
  }
  return pts;
}

/* computing the points-to set of a while loop by iterating over its
   body until reaching a fixed-point. For the moment without taking
   into account the condition's side effect. */
set points_to_whileloop(whileloop wl, set pt_in, bool store __attribute__ ((__unused__))) {
  /* get the condition,to be used later in aim to refine the points-to set
     expression cond = whileloop_condition(wl);*/

  statement while_body = whileloop_body(wl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set pt_body = set_generic_make(set_private, points_to_equal_p,
				 points_to_rank);

  do {
    set_assign(pt_body, pt_in);
    set_assign(pt_out, points_to_recursive_statement(while_body,
							      pt_body, false));
    set_assign(pt_in, set_clear(pt_in));
    set_assign(pt_in, merge_points_to_set(pt_body, pt_out));
    pt_out = set_clear(pt_out);
  } while (!set_equal_p(pt_body, pt_in));
  set_assign(pt_out, pt_in);
  points_to_storage(pt_out, while_body, true);

  return pt_out;
}

/* computing the points to of a for loop, before processing the body,
   compute the points to of the initialization. */
set points_to_forloop(forloop fl, set pt_in, bool store) {
  statement for_body = forloop_body(fl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set pt_body = set_generic_make(set_private, points_to_equal_p,
				 points_to_rank);
  expression exp = forloop_initialization(fl);
  pt_in = points_to_expression(exp, pt_in, true);
  do {
    set_assign(pt_body, pt_in);
    set_assign(pt_out, points_to_recursive_statement(for_body,
							      pt_body, store));
    pt_in = set_clear(pt_in);
    pt_in = set_assign(pt_in, merge_points_to_set(pt_body, pt_out));
    pt_out = set_clear(pt_out);
  } while (!set_equal_p(pt_body, pt_in));
  set_assign(pt_out, pt_in);
  points_to_storage(pt_out, for_body, true);

  return pt_out;
}

/* computing the points to of a  loop. to have more precise
 * information, maybe should transform the loop into a do loop by
 * activating the property FOR_TO_DO_LOOP_IN_CONTROLIZER. */
set points_to_loop(loop fl, set pt_in, bool store) {
  statement loop_body = loop_body(fl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set pt_body = set_generic_make(set_private, points_to_equal_p,
				 points_to_rank);

  do {
    set_assign(pt_body, pt_in);
    set_assign(pt_out, points_to_recursive_statement(loop_body,
							      pt_body, store));
    pt_in = set_clear(pt_in);
    set_assign(pt_in, merge_points_to_set(pt_body, pt_out));
    pt_out = set_clear(pt_out);
  } while (!set_equal_p(pt_body, pt_in));
  set_assign(pt_out, pt_in);
  points_to_storage(pt_out, loop_body, true);

  return pt_out;
}

/*Computing the points to of a do while loop, we have to process the
  body a least once, before iterating until reaching the fixed-point. */
set points_to_do_whileloop(whileloop fl, set pt_in, bool store) {
  statement dowhile_body = whileloop_body(fl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set pt_body = set_generic_make(set_private, points_to_equal_p,
				 points_to_rank);
  set_assign(pt_out, points_to_recursive_statement(dowhile_body,
							    pt_in, false));
  set_assign(pt_in, pt_out);
  do {
    set_assign(pt_body, pt_in);
    set_assign(pt_out, points_to_recursive_statement(dowhile_body,
							      pt_body, store));
    set_assign(pt_in, set_clear(pt_in));
    set_assign(pt_in, merge_points_to_set(pt_body, pt_out));
    pt_out = set_clear(pt_out);
  } while (!set_equal_p(pt_body, pt_in));
  pt_out = set_assign(pt_out, pt_in);
  points_to_storage(pt_out, dowhile_body, true);

  return pt_out;

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
  set rlt_pts_to = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  /* condition's side effect and information not taked into account :
     if(p==q) or if(*p++) */
  true_stmt = test_true(test_stmt);
  true_pts_to = set_union(true_pts_to, pt_in, points_to_recursive_statement(
									    true_stmt, pt_in, store));
  false_stmt = test_false(test_stmt);
  false_pts_to = set_union(false_pts_to, pt_in,
			   points_to_recursive_statement(false_stmt, pt_in, store));
  rlt_pts_to = merge_points_to_set(true_pts_to, false_pts_to);
  return rlt_pts_to;
}

/* computing the points-to of a call, user_functions not yet implemented. */
set points_to_call(statement s, call c, set pt_in, bool store __attribute__ ((__unused__))) {
  entity e = call_function(c);
  cons *pc = call_arguments(c);
  tag tt;
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set_methods_for_proper_simple_effects();
  set_methods_for_simple_pointer_effects();
  list el = call_to_proper_effects(c);
  generic_effects_reset_all_methods();
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
    pips_internal_error("function %s has an unknown value",
			entity_name(e));
    break;
  case is_value_intrinsic: {
    pips_debug(5, "intrinsic function %s\n", entity_name(e));
    set_assign(pt_out,
		   points_to_intrinsic(s, c, e, pc, pt_in, el));
  }
    break;
    
  default:
    pips_internal_error("unknown tag %d", tt);
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
    break;
  }
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
    break;
  }
  case is_syntax_sizeofexpression: {
    break;
  }
  case is_syntax_subscript: {
    break;
  }
  case is_syntax_application: {
    break;
  }
  case is_syntax_va_arg: {
    break;
  }

  default:
    pips_internal_error("unexpected syntax tag (%d)", syntax_tag(s));
  }
  return pt_out;
}



/* Process recursively statement "current". Use pt_list as the input
   points-to set. Save it in the statement mapping. Compute and return
   the new
   points-to set which holds after the execution of the statement. */
set points_to_recursive_statement(statement current, set pt_in, bool store) {
  set pt_out = set_generic_make(set_private, points_to_equal_p,
				points_to_rank);
  set_assign(pt_out, pt_in);
  instruction i = statement_instruction(current);
  /*  Convert the pt_in set into a sorted list for storage */
  /*  Store the current points-to list */

  /* test if the statement is a declaration. */
  if (c_module_p(get_current_module_entity()) &&
      (declaration_statement_p(current) ))
    {/* retrieve the list of declarations attatched to the module*/
      list l_decls = statement_declarations(current);
      pips_debug(1, "declaration statement \n");

      FOREACH(ENTITY, e, l_decls)
	{
	  if(type_variable_p(entity_type(e)))
	    {/* test if the dclaration is an initialization */
	      value v_init = entity_initial(e);
	      /* generate points-to due to the initialisation */
	      if (value_expression_p(v_init))
		{
		  expression exp_init = value_expression(v_init);
		  expression lhs = entity_to_expression(e);
		  /* get the rhs (exp_init) and create the lhs from
		     the entity. Then call points_to_assignment which
		     will identify the type of each hand side and
		     call the appropriate basic case. */
		  set_assign(pt_out,points_to_assignment(current,
							 lhs,
							 exp_init,
							 pt_in));
		}
	    }
	}
    }
  points_to_storage(pt_in, current, store);
  ifdebug(1) print_statement(current);

  switch (instruction_tag(i))
    {
      /* instruction = sequence + test + loop + whileloop +
	 goto:statement +
	 call + unstructured + multitest + forloop  + expression ;*/

    case is_instruction_call: {
      set_assign(pt_out, points_to_call(current,
						 instruction_call(i), pt_in, store));
    }
      break;
    
    case is_instruction_sequence: {
      set_assign(pt_out, points_to_sequence(instruction_sequence(i),
						     pt_in, store));
    }
      break;
    
    case is_instruction_test: {
      set_assign(pt_out, points_to_test(instruction_test(i), pt_in,
						 store));
      break;
    }
    case is_instruction_whileloop: {
      store = false;
      if (evaluation_tag(whileloop_evaluation(instruction_whileloop(i))) == 0) {
	set_assign(pt_out, points_to_whileloop(
							instruction_whileloop(i), pt_in, false));
      } else
	set_assign(pt_out, points_to_do_whileloop(
							   instruction_whileloop(i), pt_in, false));
    }
      break;
    case is_instruction_loop: {
      store = false;
      set_assign(pt_out, points_to_loop(instruction_loop(i), pt_in,
						 store));
    }
      break;
    
    case is_instruction_forloop: {
      store = false;
      set_assign(pt_out, points_to_forloop(instruction_forloop(i),
						    pt_in, store));
    }
      break;
    
    case is_instruction_expression: {
      set_assign(pt_out, points_to_expression(
						       instruction_expression(i), pt_in, store));
    }
      break;
    
    case is_instruction_unstructured: {
      set_assign(pt_out, pt_in);
      break;
    }

    default:
      pips_internal_error("Unexpected instruction tag %d", instruction_tag(
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
								  module_name, TRUE));
  module_stat = get_current_module_statement();

  /* Get the summary_intraprocedural_points_to resource.*/
  points_to_list summary_pts_to_list =
    (points_to_list) db_get_memory_resource(DBR_SUMMARY_POINTS_TO_LIST,
					    module_name, TRUE);
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
   
  bool good_result_p = TRUE;
  return (good_result_p);

}
