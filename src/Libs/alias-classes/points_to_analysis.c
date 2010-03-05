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
/* package ma_phase
 */

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

/* Function storing points to information attached to a statement
 */
GENERIC_GLOBAL_FUNCTION(pt_to_list, statement_points_to)


/* operator fusion to calculate the relation of two points
   to. (should be at a higher level) */
static approximation  fusion_approximation(approximation app1,
										   approximation app2)
{
  if(approximation_exact_p(app1) && approximation_exact_p(app2))
	return make_approximation_exact();
  else
	return  make_approximation_may();
}

/* To distinguish between the different basic cases if Emami's
   assignement : involving x, &x, *x, m, m.x, m->x, a[]...*/
enum Emami {EMAMI_ERROR = -1, EMAMI_NEUTRAL = 0,
			EMAMI_ADDRESS_OF = 1, EMAMI_DEREFERENCING = 2,
			EMAMI_FIELD = 3, EMAMI_STRUCT = 4, EMAMI_ARRAY = 5, EMAMI_HEAP = 6, EMAMI_POINT_TO = 7};


/* Find out Emami's types */
static int emami_expression_type(expression exp)
{
  tag t ;
  switch(t = syntax_tag(expression_syntax(exp))){
/*in case it's an EMAMI_NEUTRAL  x or an EMAMI_ARRAY or an EMAMI_STRUC */
  case  is_syntax_reference :{
	if(array_argument_p(exp))
	  return EMAMI_ARRAY;
	syntax s = expression_syntax(exp);
	reference r = syntax_reference(s);
	entity e = reference_variable(r);
	type typ = entity_type(e);
	variable v = type_variable(typ);
	basic b = variable_basic(v);
	t = basic_tag(b);
	if( t == is_basic_derived){
	  entity ee = basic_derived(b);
	  type tt = entity_type(ee);
	  if(type_struct_p(tt)== true)
		return EMAMI_STRUCT;
	  else
		return EMAMI_NEUTRAL;
	}
	else
	  return EMAMI_NEUTRAL;
	break;
  }
  case  is_syntax_range :
	return	EMAMI_ERROR;
	break;
/* in case it's an 	EMAMI_ADDRESS_OF or an EMAMI_DEREFERENCING or an
  EMAMI_FIELD or an an EMAMI_HEAP or an EMAMI_POINT_TO...  */
  case  is_syntax_call :{
	call c = expression_call(exp);
	if(entity_an_operator_p(call_function(c), ADDRESS_OF))
	  return EMAMI_ADDRESS_OF;
	else if(entity_an_operator_p(call_function(c), DEREFERENCING))
	  return EMAMI_DEREFERENCING;
	else if(entity_an_operator_p(call_function(c), FIELD))
	  return EMAMI_FIELD;
	else if (entity_an_operator_p(call_function(c), POINT_TO))
	  return EMAMI_POINT_TO;
	else if(ENTITY_MALLOC_SYSTEM_P(call_function(c)))
	  return EMAMI_HEAP;
	else
	  return  EMAMI_ERROR;
	break;
  }

  case  is_syntax_cast :{
	cast ct = syntax_cast(expression_syntax(exp));
	expression e = cast_expression(ct);

	return	emami_expression_type(e);
	break;
  }
  case  is_syntax_sizeofexpression :
	return	EMAMI_ERROR;
	break;
  case  is_syntax_subscript :
	return	EMAMI_ERROR;
	break;
  case  is_syntax_application :
	return	EMAMI_ERROR;
	break;
  case  is_syntax_va_arg :
	return	EMAMI_ERROR;
	break;
  default :{
	pips_internal_error("unknown tag %d\n", t);
	return	EMAMI_ERROR;
	break;
  }
  }
}

/* a pointer_type_p already exist but do not respond to our needs...*/
static bool type_pointer_p(expression e)
{
  bool rlt = false;
  syntax s = expression_syntax(e);
  reference r = syntax_reference(s);
  entity ent = reference_variable(r);
  type t = entity_type(ent);
  if(type_variable_p(t)){
	variable v= type_variable(t);
	if(basic_pointer_p(variable_basic(v)))
	  rlt = true;
  }
  return rlt;

}

/* Same as previous function, but for double pointers. */
static bool type_double_pointer_p(expression e)
{
  syntax s = expression_syntax(e);
  reference r = syntax_reference(s);
  entity ent = reference_variable(r);
  type t = entity_type(ent);
  bool rlt = false;
  if(type_variable_p(t)){
	variable v= type_variable(t);
	basic b = variable_basic(v);
	if(basic_pointer_p(b)){
	  type typ = basic_pointer(b);
	  if(type_variable(typ)){
		variable v1= type_variable(typ);
		basic b1 = variable_basic(v1);
		if (basic_pointer_p(b1))
		  rlt = true;
	  }
	}
  }
  return rlt;
}

/* given an expression, return the referenced entity in case exp is a
   reference */
entity argument_entity(expression exp)
{
  syntax syn = expression_syntax(exp);
  entity var = entity_undefined;

  if(syntax_reference_p(syn)){
	reference ref = syntax_reference(syn);
	var = reference_variable(copy_reference(ref));
  }
  return copy_entity(var);
}

/* Order the two points-to relations according to the alphabetical
   order of the underlying variables. Rteurn -1, 0, or 1. */
int compare_points_to_access(void * vpt1 , void * vpt2 )
{
  points_to pt1=(points_to)vpt1;
  points_to pt2=(points_to)vpt2;
  int
	null_1 = (pt1==(points_to)NULL),
	null_2 = (pt2==(points_to)NULL);

  if (points_to_domain_number(pt1)!= points_to_domain ||
	  points_to_domain_number(pt2)!= points_to_domain )
	return(null_2-null_1);
  else{
	access a1 = points_to_source(pt1);
	access a2 = points_to_source(pt2);
	points_to_path p1 = access_points_to_path(a1);
	points_to_path p2 = access_points_to_path(a2);
	reference r1 =  points_to_path_reference(p1);
	reference r2 =  points_to_path_reference(p2);
	return reference_equal_p(r1, r2);
  }
}

/* merge two points-to sets; required to compute
   the points-to set of the if control statements. */
set merge_points_to_set(set s1, set s2)
{
  set Definite_set = set_generic_make(set_private,
									  points_to_equal_p,points_to_rank);
  set Possible_set = set_generic_make(set_private,
									  points_to_equal_p,points_to_rank);
  set Intersection_set = set_generic_make(set_private,
										  points_to_equal_p,
										  points_to_rank);
  set Union_set =set_generic_make(set_private,
								  points_to_equal_p,points_to_rank);
  set Merge_set =set_generic_make(set_private,
								  points_to_equal_p,points_to_rank);

  Intersection_set = set_intersection(Intersection_set, s1, s2);
  Union_set= set_union(Union_set, s1, s2);
  SET_FOREACH(points_to, i, Intersection_set){
	if(approximation_tag(points_to_relation(i)) == 2)
	  Definite_set   = set_add_element(Definite_set,Definite_set,
									   (void*) i );
  }
  SET_FOREACH(points_to, j, Union_set){
	if(! set_belong_p(Definite_set, (void*)j)){
	  points_to pt = make_points_to(points_to_source(j), points_to_sink(j),
									make_approximation_may());
	  Possible_set   = set_add_element(Possible_set,Possible_set,
									   pt);
	}
  }
  Merge_set = set_clear(Merge_set);
  Merge_set = set_union(Merge_set, Possible_set, Definite_set);
 
  return Merge_set;
}

/* storing the points to associate to a statement s, in the case of
   loops the field store is set to false to prevent from the key redefenition*/
void points_to_storage(set pts_to_set, statement s, bool store)
{list pt_list = NIL;
  if(!set_empty_p(pts_to_set) && store == true){
	//  print_points_to_set(stderr,"",pts_to_set);
	pt_list = set_to_sorted_list(pts_to_set,
								 (int(*)
								  (const void*,const void*))
								 compare_points_to_access);
	points_to_list new_pt_list = make_points_to_list(pt_list);
	store_pt_to_list(s, new_pt_list);
  }
}

/* one basic case of Emami: < x = y > and < m.x = m.y > */
set basic_ref_ref(set pts_to_set,
				  expression lhs,
				  expression rhs)
{
  set gen_pts_to =set_generic_make(set_private,
				   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
					points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=expression_syntax(lhs);
  syntax syn2=expression_syntax(rhs);
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  expression lhs_tmp = expression_undefined;
  expression rhs_tmp = expression_undefined;
  ifdebug(1) printf("\n cas x = y \n");
  if(syntax_call_p(syn1) && syntax_call_p(syn2)){
	call c1 = expression_call(lhs);
	call c2 = expression_call(rhs);
	if(entity_an_operator_p(call_function(c1), FIELD) &&
	   entity_an_operator_p(call_function(c2),FIELD)){
	  list l = call_arguments(c1);
	  lhs_tmp = EXPRESSION (CAR(CDR(l)));
	  list l1 = call_arguments(c2);
	  rhs_tmp = EXPRESSION (CAR(CDR(l1)));
	  syn1=expression_syntax(lhs_tmp);
	  syn2=expression_syntax(rhs_tmp);
	  ref2=expression_reference(rhs_tmp);
	  ent2=reference_variable(ref2);
	}
  }else{
	lhs_tmp = copy_expression(lhs);
	rhs_tmp = copy_expression(rhs);
	syn1=expression_syntax(lhs);
	syn2=expression_syntax(rhs);
	ref2=expression_reference(rhs);
	ent2=reference_variable(ref2);
  }
  if(syntax_reference_p(syn1) && syntax_reference_p(syn2)){
	if((type_pointer_p(lhs_tmp)&&(type_pointer_p(rhs_tmp) || array_entity_p(ent2))) ||
	   (type_double_pointer_p(lhs_tmp)&& type_double_pointer_p(rhs_tmp))){
	  // creation of the source
	  effect e1 = effect_undefined, e2 = effect_undefined;
	  set_methods_for_proper_simple_effects();
	  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
									 &e1,
									 true);
	  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
									 &e2,
									 false);
	  effects_free(l1);
	  effects_free(l2);
	  generic_effects_reset_all_methods();
	  ref1 = effect_any_reference(copy_effect(e1));
		points_to_path p1 = make_points_to_path(ref1);
	  source = make_access_referencing(p1);

	  // add the points_to relation to the set generated
	  // by this assignement
	  ref2 = effect_any_reference(copy_effect(e2));
	  points_to_path p2 = make_points_to_path(ref2);
	  sink = make_access_referencing(p2);
	  set s = set_generic_make(set_private,
							   points_to_equal_p,points_to_rank);
	  SET_FOREACH(points_to, i, pts_to_set){
		if(access_equal_p(copy_access(points_to_source(i)), sink))
		  s = set_add_element(s, s, (void*)i);
	  }
	  SET_FOREACH(points_to, j, s){
		nouv_sink = copy_access(points_to_sink(j));
		// access nouv_source = copy_access(source);
		rel = points_to_relation(j);
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
					     (void*) pt_to );
	  }
	  /* in case x = y[i]*/
	  if(array_entity_p(ent2)){
		points_to_path p2 =make_points_to_path(ref2);
		nouv_sink = make_access_referencing(p2);
		// access nouv_source = copy_access(source);
		rel =make_approximation_exact();
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
					     (void*) pt_to );
	  }

	  // creation of the written set
	  // search of all the points_to relations in the
	  // alias set where the source is equal to the lhs
	  SET_FOREACH(points_to, k, pts_to_set){
		if(access_equal_p(points_to_source(k), source))
		  written_pts_to = set_add_element(written_pts_to,
						   written_pts_to, (void *)k);
	  }
	  pts_to_set = set_difference(pts_to_set,
				      pts_to_set,
				      written_pts_to);
	  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
	  ifdebug(1)
		print_points_to_set(stderr,"Points to pour le cas 1 <x = y>\n",
							pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* one basic case of Emami: < x = y[i] >. ne specific treatment is
   yet implemented, later we should make a difference between two
   cases : i =0 and i > 0.  */
set basic_ref_array(set pts_to_set,
					expression lhs,
					expression rhs)
{
  set gen_pts_to =set_generic_make(set_private,
				   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
					points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=expression_syntax(lhs);
  syntax syn2=expression_syntax(rhs);
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1) printf("\n cas x = y[i] \n");
  syn1 = expression_syntax(lhs);
  syn2 = expression_syntax(rhs);
  ref2 = expression_reference(rhs);
  ent2 = reference_variable(ref2);
  if(syntax_reference_p(syn1) && syntax_reference_p(syn2)){
	if(type_pointer_p(lhs)){
	  // creation of the source
	  effect e1 = effect_undefined, e2 = effect_undefined;
	  set_methods_for_proper_simple_effects();
	  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
																	 &e1,
																	 true);
	  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
																	 &e2,
																	 false);
	  effects_free(l1);
	  effects_free(l2);
	  generic_effects_reset_all_methods();
	  ref1 = effect_any_reference(copy_effect(e1));
	  points_to_path p1 = make_points_to_path(ref1);
	  source = make_access_referencing(p1);
	  // add the points_to relation to the set generated
	  // by this assignement
	  ref2 = effect_any_reference(copy_effect(e2));
	  points_to_path p2 = make_points_to_path(ref2);
	  sink = make_access_referencing(p2);
	  nouv_sink = make_access_referencing(p2);
	  // access nouv_source = copy_access(source);
	  rel =make_approximation_exact();
	  pt_to = make_points_to(source, nouv_sink, rel);
	  gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
								   (void*) pt_to );
	  // creation of the written set
	  // search of all the points_to relations in the
	  // alias set where the source is equal to the lhs
	  SET_FOREACH(points_to, k, pts_to_set)
	  {
		if(access_equal_p(points_to_source(k), source))
		  written_pts_to = set_add_element(written_pts_to,
										   written_pts_to, (void *)k);
	  }
	  pts_to_set = set_difference(pts_to_set,
								  pts_to_set,
								  written_pts_to);
	  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
	  ifdebug(1)
		print_points_to_set(stderr,"Points to pour le cas 1 <x = y>\n",
							pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}


/* one basic case of Emami: < x = &y > */
set basic_ref_addr(set pts_to_set,
				   expression lhs,
				   expression rhs)
{
  set gen_pts_to =set_generic_make(set_private,
				   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
					points_to_equal_p,points_to_rank);
  points_to pt = points_to_undefined;
  syntax syn2 = syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  call c2 = call_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1){
	pips_debug(1, " case x = &y\n");
  }
  
  syn2 = expression_syntax(rhs);
  c2 = syntax_call(syn2);
  list args = call_arguments(c2);
  expression  rhs_tmp = EXPRESSION (CAR(args));
  if(array_argument_p(rhs_tmp)){
	pts_to_set = set_assign(pts_to_set, basic_ref_array(pts_to_set,
														lhs,
														rhs_tmp));
	return pts_to_set;
  }
  ref2 = expression_reference(rhs_tmp);
  ent2 = reference_variable(ref2);
  if(type_pointer_p(lhs)){
	// creation of the source
	effect e1 = effect_undefined, e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	list l1 = generic_proper_effects_of_complex_address_expression(lhs,
																   &e1,
																   true);
	list l2 = generic_proper_effects_of_complex_address_expression(rhs_tmp,
																   &e2,
																   false);
	effects_free(l1);
	effects_free(l2);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_referencing(p1);
	// creation of the sink
	ref2 = effect_any_reference(copy_effect(e2));
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_addressing(p2);
	// creation of the approximation
	rel = make_approximation_exact();
	// creation of the points_to relation
	pt = make_points_to(source, sink, rel);
	// add the points_to relation to the set generated
	//by this assignement
	gen_pts_to=set_add_element(gen_pts_to, gen_pts_to,(void*) pt);
	// creation of the written set
	// search of all the points_to relations in the
	// alias set where the source is equal to the lhs
	SET_FOREACH(points_to, i,pts_to_set){
	  if(access_equal_p(points_to_source(i),source))
		written_pts_to = set_add_element(written_pts_to,
										 written_pts_to,
										 (void *)i);
	}
	pts_to_set = set_difference(pts_to_set, pts_to_set, written_pts_to);
	pts_to_set = set_union(pts_to_set, pts_to_set, gen_pts_to);
	ifdebug(1){
	  print_points_to_set(stderr,"points To pour le cas 2 <x = &y> \n ",
						  pts_to_set);
	}
  }else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* one basic case of Emami: < x = *y > */
set basic_ref_deref(set pts_to_set,
					expression lhs,
					expression rhs)
{
  set gen_pts_to =set_generic_make(set_private,
								   points_to_equal_p,points_to_rank);
  set s =set_generic_make(set_private,
						  points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=syntax_undefined;
  syntax syn2=syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  call c2 = call_undefined;
  entity ent1 = entity_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1){
	pips_debug(1, " case  x = *y\n");
  }
  syn1=expression_syntax(lhs);
  ref1=syntax_reference(syn1);
  ent1=reference_variable(ref1);
  syn2=expression_syntax(rhs);
  c2 = syntax_call(syn2);
  list  args = call_arguments(c2);
  expression rhs_tmp = EXPRESSION (CAR(args));
  ref2 = expression_reference(rhs_tmp);
  ent2 = argument_entity(rhs_tmp);
  if(type_pointer_p(lhs)&&
	 type_double_pointer_p(rhs_tmp)){
	// creation of the source
	effect e1 = effect_undefined, e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	list  l1 = generic_proper_effects_of_complex_address_expression(lhs,
																	&e1,
																	true);
	list  l2 = generic_proper_effects_of_complex_address_expression(rhs_tmp,
																	&e2,
																	false);
	effects_free(l1);
	effects_free(l2);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_referencing(p1);
	// creation of the sink
	ref2 = effect_any_reference(copy_effect(e2));
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_referencing(p2);
	// fetch the points to relations
	// where source = source 1
	// creation of the written set
	// search of all the points_to relations in the
	// alias set where the source is equal to the lhs
	SET_FOREACH(points_to, elt, pts_to_set)
	{
	  if(access_equal_p(points_to_source(elt),
						source))
		written_pts_to = set_add_element(written_pts_to,
										 written_pts_to,(void*)elt);
	}
	SET_FOREACH(points_to, i, pts_to_set)
	{
	  if(access_equal_p(points_to_source((points_to)i), sink))
	  {
		s =set_add_element(s, s, (void *)i);
	  }
	}
	SET_FOREACH(points_to, j,s )
	{
	  SET_FOREACH(points_to, p, pts_to_set)
	  {
		if(access_equal_p(points_to_sink(j),
						  points_to_source(p))){
		  nouv_sink = copy_access(points_to_sink(p));
		  rel =fusion_approximation(points_to_relation(j),
									points_to_relation(i));
		  pt_to = make_points_to(source, nouv_sink, rel);
		  gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
									   (void *) pt_to );
		}
	  }
	}

	pts_to_set = set_difference(pts_to_set, pts_to_set,written_pts_to);
	pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
	ifdebug(1){
	  print_points_to_set(stderr,"Points To pour le cas 3 <x = *y> \n",
						  pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* cas *x = y   or *m.x = y */
set basic_deref_ref(set pts_to_set,
					expression lhs,
					expression rhs)
{
  set s1=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s2=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s3= set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set change_pts_to= set_generic_make(set_private,
									  points_to_equal_p,points_to_rank);
  set gen_pts_to =set_generic_make(set_private,
								   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  call c1 = call_undefined;
  entity ent1 = entity_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_source = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1){
	pips_debug(1, " case *x = y\n");
  }
  // recuperation of x
  syn1 = expression_syntax(lhs);
  c1 = syntax_call(syn1);
  list  args1 = call_arguments(c1);
  expression lhs_tmp = EXPRESSION (CAR(args1));
  expression ex = copy_expression(lhs_tmp);
  /* if we have *m.x = y */
  syntax s = expression_syntax(lhs_tmp);
  if(syntax_call_p(s))
  {
	call c = expression_call(lhs_tmp);
	if(entity_an_operator_p(call_function(c), FIELD))
	{
	  list l = call_arguments(c);
	  lhs_tmp = EXPRESSION (CAR(CDR(l)));
	}
  }

  if(type_double_pointer_p(lhs_tmp)&&
	 type_pointer_p(rhs)){
	effect e1 = effect_undefined, e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	list l1 = generic_proper_effects_of_complex_address_expression(ex,
																   &e1,
																   true);
	list l2 = generic_proper_effects_of_complex_address_expression(rhs,
																   &e2,
																   false);
	effects_free(l1);
	effects_free(l2);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	ent1 = argument_entity(lhs_tmp);


	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_dereferencing(p1);
	// recuperation of y
		ref2 = effect_any_reference(copy_effect(e2));
	ent2 = reference_variable(ref2);
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_dereferencing(p2);

	/* creation of the set written_pts_to =
	   {(x1,x2,rel)| (x, x1, EXACT), (x1, x2, rel) /in pts_to_set}*/
	SET_FOREACH(points_to, i, pts_to_set){
	  if( access_equal_p(points_to_source(i), source) &&
		  approximation_exact_p(points_to_relation(i))){
		SET_FOREACH(points_to, j,pts_to_set ){
		  if( access_equal_p(points_to_source(j) ,
							 points_to_sink(i)))
			written_pts_to = set_add_element(written_pts_to,
											 written_pts_to, (void *)j);
		}
	  }
	}
	/* {(x1, x2,EXACT)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
	SET_FOREACH(points_to, k, pts_to_set){
	  if( access_equal_p(points_to_source(i), source)
		  && approximation_may_p(points_to_relation(k))){
		SET_FOREACH(points_to, h,pts_to_set ){
		  if(access_equal_p(points_to_source(h),points_to_sink(k))&&
			 approximation_exact_p(points_to_relation(h)))
			s2 = set_add_element(s2, s2, (void *)h);
		}
	  }
	}

	/* {(x1, x2,MAY)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
	SET_FOREACH(points_to, l, pts_to_set){
	  if(access_equal_p(points_to_source(l), source) &&
		 approximation_may_p(points_to_relation(l))){
		SET_FOREACH(points_to, m,pts_to_set){
		  if(access_equal_p( points_to_source(m),points_to_sink(l))&&
			 approximation_exact_p(points_to_relation(m))){
			points_to_relation(m) = make_approximation_may();
			s3 = set_add_element(s3, s3, (void *)m);
		  }
		}
	  }
	}
	change_pts_to = set_difference(change_pts_to,pts_to_set, s3);
	change_pts_to = set_union(change_pts_to,change_pts_to, s3);
	SET_FOREACH(points_to, n, pts_to_set){
	  if(access_equal_p(points_to_source(n), source)){
		SET_FOREACH(points_to, o, pts_to_set){
		  if(access_equal_p(points_to_source(o) , sink)){
			nouv_source = copy_access(points_to_sink(n));
			nouv_sink = copy_access(points_to_sink(o));
			rel = fusion_approximation(points_to_relation(n),
									   points_to_relation(o));
			pt_to = make_points_to(nouv_source, nouv_sink, rel);
			gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
										 (void *)pt_to);
		  }
		}
	  }
	}
	s1 = set_difference(s1, change_pts_to, written_pts_to);
	pts_to_set = set_union(pts_to_set, gen_pts_to, s1);
  }
  else {
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

// cas *x = &y and *m.x = &y;
static set basic_deref_addr(set pts_to_set,
							expression lhs,
							expression rhs)
{
  set s1=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s2=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s3= set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set change_pts_to= set_generic_make(set_private,
									  points_to_equal_p,points_to_rank);
  set gen_pts_to =set_generic_make(set_private,
								   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  syntax syn1=syntax_undefined;
  syntax syn2=syntax_undefined;
  call c1 = call_undefined;
  call c2 = call_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent1 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  ifdebug(1){
	pips_debug(1, " case *x = &y\n");
  }
  // recuperation of x
  syn1 = expression_syntax(lhs);
  c1 = syntax_call(syn1);
  list  args1 = call_arguments(c1);
  expression  lhs_tmp = EXPRESSION (CAR(args1));
  expression ex = copy_expression(lhs_tmp);
  syntax s = expression_syntax(lhs_tmp);
  if(syntax_call_p(s))
  {
	call c = expression_call(lhs_tmp);
	if(entity_an_operator_p(call_function(c), FIELD))
	{
	  list l = call_arguments(c);
	  lhs_tmp = EXPRESSION (CAR(CDR(l)));
	}
  }
  ref1 = expression_reference(lhs_tmp);
  ent1 = argument_entity(lhs_tmp);
  // recuperation of y
  syn2 = expression_syntax(rhs);
  c2 = syntax_call(syn2);
  list args2 = call_arguments(c2);
  expression  rhs_tmp = EXPRESSION (CAR(args2));
  if(array_argument_p(rhs_tmp))
  {
	pts_to_set = set_assign(pts_to_set, basic_deref_array(pts_to_set,
														  lhs_tmp,
														  rhs_tmp));
	return pts_to_set;
  }

  if(type_double_pointer_p(lhs_tmp)){
	effect e1 = effect_undefined, e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	list l1 = generic_proper_effects_of_complex_address_expression(ex,
																   &e1,
																   true);
	list l2 = generic_proper_effects_of_complex_address_expression(rhs_tmp,
																   &e2,
																   false);
	effects_free(l1);
	effects_free(l2);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_referencing(p1);
	ref2 = effect_any_reference(copy_effect(e2));
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_addressing(p2);
	SET_FOREACH(points_to, i, pts_to_set){
	  if(access_equal_p(points_to_source(i), source) &&
		 approximation_exact_p(points_to_relation(i))){
		SET_FOREACH(points_to, j,pts_to_set){
		  if(access_equal_p(points_to_source(j),points_to_sink(i)))
			written_pts_to = set_add_element(written_pts_to,
											 written_pts_to,(void*)j);
		}
	  }
	}
	SET_FOREACH(points_to, k, pts_to_set){
	  if(access_equal_p(points_to_source(i), source) &&
		 approximation_may_p(points_to_relation(k))){
		SET_FOREACH(points_to, h, pts_to_set){
		  if(access_equal_p(points_to_source(h),points_to_sink(k))&&
			 approximation_exact_p(points_to_relation(h))){
			s2 = set_add_element(s2, s2, (void *) h );
		  }
		}
	  }
	}
	SET_FOREACH(points_to, l, pts_to_set){
	  if(access_equal_p(points_to_source(l), source) &&
		 approximation_may_p(points_to_relation(l))){
		SET_FOREACH(points_to, m,pts_to_set){
		  if(access_equal_p(points_to_source(m),points_to_sink(l))&&
			 approximation_exact_p(points_to_relation(m))){
			points_to_relation(m) = make_approximation_may();
			s3 = set_add_element(s3, s3, (void *)m);
		  }
		}
	  }
	}
	change_pts_to = set_difference(change_pts_to,pts_to_set, s2);
	change_pts_to = set_union(change_pts_to,change_pts_to, s3);
	SET_FOREACH(points_to, n, pts_to_set){
	  if(access_equal_p(points_to_source(n), source)){
		SET_FOREACH(points_to, o,pts_to_set){
		  if(access_equal_p( points_to_source(o),
							 points_to_sink(n))){
			points_to pt = make_points_to(points_to_source(o),
										  sink, points_to_relation(n));
			gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
										 (void *) pt );
		  }
		}
	  }
	}
	s1 = set_difference(s1, change_pts_to, written_pts_to);
	pts_to_set = set_union(pts_to_set,gen_pts_to, s1);
	ifdebug(1)
	  print_points_to_set(stderr,"Points pour le cas 5 <*x = &y> \n ",
						  pts_to_set);
  } else{
	ifdebug(1){
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

// cas *x = &y[i] and *m.x = &y[i];
set basic_deref_array(set pts_to_set,
					  expression lhs,
					  expression rhs)
{
  set s1=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s2=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s3= set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set change_pts_to= set_generic_make(set_private,
									  points_to_equal_p,points_to_rank);
  set gen_pts_to =set_generic_make(set_private,
								   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  syntax syn1=syntax_undefined;
  syntax syn2=syntax_undefined;
  call c1 = call_undefined;
  call c2 = call_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent1 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  ifdebug(1){
	pips_debug(1, " case *x = &y[i] or *m.x = &y[i]\n");
  }
  // recuperation of x
  syn1 = expression_syntax(lhs);
  c1 = syntax_call(syn1);
  list  args1 = call_arguments(c1);
  expression  lhs_tmp = EXPRESSION (CAR(args1));
  expression ex = copy_expression(lhs_tmp);
  syntax s = expression_syntax(lhs_tmp);
  if(syntax_call_p(s))
  {
	call c = expression_call(lhs_tmp);
	if(entity_an_operator_p(call_function(c), FIELD))
	{
	  list l = call_arguments(c);
	  lhs_tmp = EXPRESSION (CAR(CDR(l)));
	}
  }
  ref1 = expression_reference(lhs_tmp);
  ent1 = argument_entity(lhs_tmp);
  // recuperation of y
  syn2 = expression_syntax(rhs);
  c2 = syntax_call(syn2);
  list args2 = call_arguments(c2);
  expression  rhs_tmp = EXPRESSION (CAR(args2));
  if(type_double_pointer_p(lhs_tmp)){
	effect e1 = effect_undefined, e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	list l1 = generic_proper_effects_of_complex_address_expression(ex,
																   &e1,
																   true);
	list l2 = generic_proper_effects_of_complex_address_expression(rhs_tmp,
																   &e2,
																   false);
	effects_free(l1);
	effects_free(l2);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_referencing(p1);
	ref2 = effect_any_reference(copy_effect(e2));
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_addressing(p2);
	SET_FOREACH(points_to, i, pts_to_set){
	  if(access_equal_p(points_to_source(i), source) &&
		 approximation_exact_p(points_to_relation(i))){
		SET_FOREACH(points_to, j,pts_to_set){
		  if(access_equal_p(points_to_source(j),points_to_sink(i)))
			written_pts_to = set_add_element(written_pts_to,
											 written_pts_to,(void*)j);
		}
	  }
	}
	SET_FOREACH(points_to, k, pts_to_set){
	  if(access_equal_p(points_to_source(i), source) &&
		 approximation_may_p(points_to_relation(k))){
		SET_FOREACH(points_to, h, pts_to_set){
		  if(access_equal_p(points_to_source(h),points_to_sink(k))&&
			 approximation_exact_p(points_to_relation(h))){
			s2 = set_add_element(s2, s2, (void *) h );
		  }
		}
	  }
	}
	SET_FOREACH(points_to, l, pts_to_set){
	  if(access_equal_p(points_to_source(l), source) &&
		 approximation_may_p(points_to_relation(l))){
		SET_FOREACH(points_to, m,pts_to_set){
		  if(access_equal_p(points_to_source(m),points_to_sink(l))&&
			 approximation_exact_p(points_to_relation(m))){
			points_to_relation(m) = make_approximation_may();
			s3 = set_add_element(s3, s3, (void *)m);
		  }
		}
	  }
	}
	change_pts_to = set_difference(change_pts_to,pts_to_set, s2);
	change_pts_to = set_union(change_pts_to,change_pts_to, s3);
	SET_FOREACH(points_to, n, pts_to_set){
	  if(access_equal_p(points_to_source(n), source)){
		SET_FOREACH(points_to, o,pts_to_set){
		  if(access_equal_p( points_to_source(o),
							 points_to_sink(n))){
			points_to pt = make_points_to(points_to_source(o),
										  sink, points_to_relation(n));
			gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
										 (void *) pt );
		  }
		}
	  }
	}
	s1 = set_difference(s1, change_pts_to, written_pts_to);
	pts_to_set = set_union(pts_to_set,gen_pts_to, s1);
	ifdebug(1)
	  print_points_to_set(stderr,"Points pour le cas 5 <*x = &y> \n ",
						  pts_to_set);
  }
  else{
	ifdebug(1){
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

//cas *x = *y
set basic_deref_deref(set pts_to_set,
					  expression lhs,
					  expression rhs)
{
  set s1 = set_generic_make(set_private,
							points_to_equal_p,points_to_rank);
  set s2 = set_generic_make(set_private,
							points_to_equal_p,points_to_rank);
  set s3 = set_generic_make(set_private,
							points_to_equal_p,points_to_rank);
  set s4 = set_generic_make(set_private,
							points_to_equal_p,points_to_rank);
  set change_pts_to = set_generic_make(set_private,
									   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  set gen_pts_to = set_generic_make(set_private,
									points_to_equal_p, points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=syntax_undefined;
  syntax syn2=syntax_undefined;
  call c1 = call_undefined;
  call c2 = call_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent1 = entity_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_source = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1){
	pips_debug(1, " case *x = *y\n");
  }
  // recuperation of x
  syn1 = expression_syntax(lhs);
  c1 = syntax_call(syn1);
  list  args1 = call_arguments(c1);
  expression lhs_tmp = EXPRESSION (CAR(args1));
  ref1 = expression_reference(lhs_tmp);
  ent1 = argument_entity(lhs);
  // recuperation of y
  syn2 = expression_syntax(rhs);
  c2 = syntax_call(syn2);
  list args2 = call_arguments(c2);
  expression rhs_tmp = EXPRESSION (CAR(args2));
  ref2 = expression_reference(rhs_tmp);
  ent2 = argument_entity(rhs_tmp);
  if(type_double_pointer_p(lhs_tmp)&&
	 type_pointer_p(rhs)){
	effect e1 = effect_undefined, e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	list l1 = generic_proper_effects_of_complex_address_expression(lhs_tmp,
																   &e1,
																   true);
	list l2 = generic_proper_effects_of_complex_address_expression(rhs_tmp,
																   &e2,
																   false);
	effects_free(l1);
	effects_free(l2);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_referencing(p1);
	ref2 = effect_any_reference(copy_effect(e2));
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_referencing(p2);
	SET_FOREACH(points_to, i, pts_to_set){
	  if(access_equal_p(points_to_source(i), source) &&
		 approximation_exact_p(points_to_relation(i))){
		SET_FOREACH(points_to, j, pts_to_set){
		  if(access_equal_p( points_to_source(j),
							 points_to_sink(i))){
			written_pts_to = set_add_element(written_pts_to,
											 written_pts_to,(void*)j);
		  }
		}
	  }
	}
	SET_FOREACH(points_to, k, pts_to_set){
	  if(access_equal_p(points_to_source(i), source) &&
		 approximation_may_p(points_to_relation(k))){
		SET_FOREACH(points_to, h, pts_to_set){
		  if(access_equal_p( points_to_source(h), points_to_sink(k))
			 && approximation_exact_p(points_to_relation(h))){
			points_to_relation(h) = make_approximation_may();
			s3 = set_add_element(s3, s3, (void *)h);
		  }
		}
	  }
	}
	SET_FOREACH(points_to, l, pts_to_set){
	  if(access_equal_p(points_to_source(l), source) &&
		 approximation_may_p(points_to_relation(l))){
		SET_FOREACH(points_to, m, pts_to_set){
		  if(access_equal_p(points_to_source(m),points_to_sink(l))
			 && approximation_exact_p(points_to_relation(m))){
			points_to_relation(m) = make_approximation_may();
			s4 = set_add_element(s4, s4, (void*)m);
		  }
		}
	  }
	}
	change_pts_to = set_difference(change_pts_to,pts_to_set, s3);
	change_pts_to = set_union(change_pts_to,change_pts_to, s4);
	SET_FOREACH(points_to, n, pts_to_set){
	  if(access_equal_p(points_to_source(n), source)){
		SET_FOREACH(points_to, o, pts_to_set){
		  if(access_equal_p(points_to_source(o), sink)){
			SET_FOREACH(points_to, f, pts_to_set){
			  if(access_equal_p(points_to_source(f),
								points_to_sink(o))){
				rel = fusion_approximation((fusion_approximation(
											  points_to_relation(n),
											  points_to_relation(o))),
										   points_to_relation(f));
				nouv_source = copy_access(points_to_sink(n));
				nouv_sink = copy_access(points_to_sink(f));
				pt_to = make_points_to(nouv_source, nouv_sink, rel);
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
	pts_to_set = set_union(pts_to_set,pts_to_set, s2);
	ifdebug(1)
	  print_points_to_set(stderr,"Points To pour le cas6  <*x = *y> \n",
						  pts_to_set);
  }

  else{
	ifdebug(1){
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* one basic case of Emami: < x.a = &y > */
set basic_field_addr( set pts_to_set,
					  expression lhs,
					  expression rhs)
{
  set gen_pts_to =set_generic_make(set_private,
								   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  points_to pt = points_to_undefined;
  syntax syn1 = syntax_undefined;
  syntax syn2 = syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  call c2 = call_undefined;
  entity ent1 = entity_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  approximation rel = approximation_undefined;
  list l = NIL;
  ifdebug(1){
	pips_debug(1, " case x.a = &y\n");
  }

  syn2 = expression_syntax(rhs);
  c2 = syntax_call(syn2);
  list args = call_arguments(c2);
  expression  rhs_tmp = EXPRESSION (CAR(args));
  ref2 = expression_reference(rhs_tmp);
  ent2 = argument_entity(rhs_tmp);
  syn1 =expression_syntax(copy_expression(lhs));
  call  c1 = syntax_call(syn1);
  list args1 = call_arguments(c1);
  expression  lhs_tmp = EXPRESSION (CAR(CDR(args1)));
  if(type_pointer_p(lhs_tmp)){
	// creation of the source
	//syntax ss = expression_syntax(lhs_tmp);
	effect e1 = effect_undefined;
	set_methods_for_proper_simple_effects();
	l = generic_proper_effects_of_complex_address_expression(lhs, &e1, true);
	effects_free(l);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	// ref1 = expression_reference(lhs_tmp);
	//ref1 = effect_reference(eff);
	// ref1 = syntax_reference(ss);
	ent1 = reference_variable(ref1);
	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_referencing(p1);
	// creation of the sink
	effect e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	l = generic_proper_effects_of_complex_address_expression(rhs_tmp, &e2, false);
	effects_free(l);
	generic_effects_reset_all_methods();
	ref2 = effect_any_reference(copy_effect(e2));
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_addressing(p2);
	// creation of the approximation
	rel = make_approximation_exact();
	// creation of the points_to relation
	pt = make_points_to(source, sink, rel);
	// add the points_to relation to the set generated
	//by this assignement
	gen_pts_to=set_add_element(gen_pts_to, gen_pts_to,(void*) pt);
	// creation of the written set
	// search of all the points_to relations in the
	// alias set where the source is equal to the lhs
	SET_FOREACH(points_to, i,pts_to_set){
	  if(access_equal_p(points_to_source(i),source))
		written_pts_to = set_add_element(written_pts_to,
										 written_pts_to,
										 (void *)i);
	}

	pts_to_set=set_difference(pts_to_set, pts_to_set, written_pts_to);
	pts_to_set = set_union(pts_to_set, pts_to_set, gen_pts_to);

	ifdebug(1){
	  print_points_to_set(stderr,"points To pour le cas 2 <x.a = &y> \n ",
						  pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* one basic case of Emami: < x = y.a > */
set basic_ref_field(set pts_to_set,
					expression lhs,
					expression rhs)
{
  set gen_pts_to =set_generic_make(set_private,
								   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=expression_syntax(lhs);
  syntax syn2=syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  reference r = reference_undefined;
  entity ent1 = entity_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  points_to_path p1 = points_to_path_undefined;
  list l=NIL, l1=NIL;
  ifdebug(1) printf("\n cas x = y.a \n");
  syn1 = expression_syntax(lhs);
  syn2 = expression_syntax(rhs);

  call  c1 = syntax_call(syn2);
  list args1 = call_arguments(c1);
  expression  rhs_tmp = EXPRESSION (CAR(CDR(args1)));
  r = expression_reference(rhs);
  syntax s = expression_syntax(rhs_tmp);
  ref2=expression_reference(rhs_tmp);
  if(syntax_reference_p(syn1) && syntax_reference_p(s)){
	ent2=reference_variable(ref2);
	if(type_pointer_p(lhs)
	   &&(type_pointer_p(rhs_tmp) || array_entity_p(ent2))){
	  // creation of the source
	  effect e1 = effect_undefined, e2;
	  set_methods_for_proper_simple_effects();
	  l = generic_proper_effects_of_complex_address_expression(copy_expression(lhs),
															   &e1,
															   true);
	  l1 = generic_proper_effects_of_complex_address_expression(rhs,&e2,false);
	  effects_free(l);
	  effects_free(l1);
	  generic_effects_reset_all_methods();
	  ref1 = effect_any_reference(copy_effect(e1));
	  ent1 = reference_variable(ref1);
	  p1 = make_points_to_path(ref1);
	  source = make_access_referencing(p1);
	  // add the points_to relation to the set generated
	  // by this assignement
	  reference  ref = effect_any_reference(copy_effect(e2));
	  points_to_path p2 = make_points_to_path(ref);
	  // print_reference(copy_reference(r));
	  sink = make_access_referencing(p2);
	  set s = set_generic_make(set_private,
							   points_to_equal_p,points_to_rank);
	  SET_FOREACH(points_to, i, pts_to_set){
		if(access_equal_p(points_to_source(i), sink))
		  s = set_add_element(s, s, (void*)i);
	  }
	  SET_FOREACH(points_to, j, s){
		nouv_sink = copy_access(points_to_sink(j));
		// access nouv_source = copy_access(source);
		rel = points_to_relation(j);
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
									 (void*) pt_to );
	  }
	  /* in case x = y[i]*/
	  if(array_entity_p(ent2)){
		points_to_path p2 =make_points_to_path(ref2);
		nouv_sink = make_access_referencing(p2);
		// access nouv_source = copy_access(source);
		rel =make_approximation_exact();
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
									 (void*) pt_to );
	  }
	  // creation of the written set
	  // search of all the points_to relations in the
	  // alias set where the source is equal to the lhs
	  SET_FOREACH(points_to, k, pts_to_set){
		if(access_equal_p(points_to_source(k), source))
		  written_pts_to = set_add_element(written_pts_to,
										   written_pts_to, (void *)k);
	  }
	  pts_to_set = set_difference(pts_to_set,
								  pts_to_set,
								  written_pts_to);
	  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);

	  ifdebug(1)
		print_points_to_set(stderr,"Points to pour le cas 1 <x = y.a>\n",
							pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* one basic case of Emami: < x = y->a > */
set basic_ref_ptr_to_field(set pts_to_set,
						   expression lhs,
						   expression rhs)
{
  set gen_pts_to =set_generic_make(set_private,
								   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=expression_syntax(lhs);
  syntax syn2=syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent1 = entity_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  points_to_path p1 = points_to_path_undefined;
  list l=NIL, l1=NIL;

  ifdebug(1) printf("\n cas x = y->a \n");
  syn1 = expression_syntax(lhs);
  syn2 = expression_syntax(rhs);

  call  c1 = syntax_call(syn2);
  list args1 = call_arguments(c1);
  expression  rhs_tmp = EXPRESSION (CAR(CDR(args1)));
  syntax s = expression_syntax(rhs_tmp);
  ref2 = expression_reference(rhs_tmp);
  if(syntax_reference_p(syn1) && syntax_reference_p(s)){
	ent2=reference_variable(ref2);
	if(type_pointer_p(lhs)
	   &&(type_pointer_p(rhs_tmp) || array_entity_p(ent2))){
	  // creation of the source
	  effect e1 = effect_undefined, e2;
	  set_methods_for_proper_simple_effects();
	  l = generic_proper_effects_of_complex_address_expression(copy_expression(lhs),
															   &e1,
															   true);
	  l1 = generic_proper_effects_of_complex_address_expression(rhs,&e2,false);
	  effects_free(l);
	  effects_free(l1);
	  generic_effects_reset_all_methods();
	  ref1 = effect_any_reference(copy_effect(e1));
	  ent1 = reference_variable(ref1);
	  p1 = make_points_to_path(ref1);
	  source = make_access_referencing(p1);
	  // add the points_to relation to the set generated
	  // by this assignement
	  reference  ref = effect_any_reference(copy_effect(e2));
	  points_to_path p2 = make_points_to_path(ref);
	  // print_reference(copy_reference(r));
	  sink = make_access_referencing(p2);
	  set s = set_generic_make(set_private,
							   points_to_equal_p,points_to_rank);
	  SET_FOREACH(points_to, i, pts_to_set){
		if(access_equal_p(points_to_source(i), sink))
		  s = set_add_element(s, s, (void*)i);
	  }
	  SET_FOREACH(points_to, j, s){
		nouv_sink = copy_access(points_to_sink(j));
		// access nouv_source = copy_access(source);
		rel = points_to_relation(j);
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
									 (void*) pt_to );
	  }
	  /* in case x = y[i]*/
	  if(array_entity_p(ent2)){
		points_to_path p2 =make_points_to_path(ref2);
		nouv_sink = make_access_referencing(p2);
		rel =make_approximation_exact();
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
									 (void*) pt_to );
	  }
	  // creation of the written set
	  // search of all the points_to relations in the
	  // alias set where the source is equal to the lhs
	  SET_FOREACH(points_to, k, pts_to_set){
		if(access_equal_p(points_to_source(k), source))
		  written_pts_to = set_add_element(written_pts_to,
										   written_pts_to, (void *)k);
	  }
	  pts_to_set = set_difference(pts_to_set,
								  pts_to_set,
								  written_pts_to);
	  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);

	  ifdebug(1)
		print_points_to_set(stderr,"Points to pour le cas 1 <x = y.a>\n",
							pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* one basic case of Emami: < *x = m.y > */
set basic_deref_field(set pts_to_set,
					  expression lhs,
					  expression rhs)
{
  set s1=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s2=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s3= set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set change_pts_to= set_generic_make(set_private,
				      points_to_equal_p,points_to_rank);
  set gen_pts_to =set_generic_make(set_private,
				   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
					points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  call c1 = call_undefined;
  entity ent1 = entity_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_source = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1){
	pips_debug(1, " case *x = m;y\n");
  }
  // recuperation of x
  syn1 = expression_syntax(lhs);
  c1 = syntax_call(syn1);
  list  args1 = call_arguments(c1);
  expression lhs_tmp = EXPRESSION (CAR(args1));
  expression ex = copy_expression(lhs_tmp);
  /* if we have *m.x = y */
  syntax s = expression_syntax(lhs_tmp);
  if(syntax_call_p(s))
  {
	call c = expression_call(lhs_tmp);
	if(entity_an_operator_p(call_function(c), FIELD))
	{
	  list l = call_arguments(c);
	  lhs_tmp = EXPRESSION (CAR(CDR(l)));
	}
  }
  // recuperation of y
  syntax syn2 = expression_syntax(rhs);
  call c2 = syntax_call(syn2);
  list  args2 = call_arguments(c2);
  expression rhs_tmp = EXPRESSION (CAR(args2));
  expression e = copy_expression(rhs_tmp);
  /* if we have *m.x = m.y */
  syntax ss = expression_syntax(rhs_tmp);
  if(syntax_call_p(ss))
  {
	call c = expression_call(rhs_tmp);
	if(entity_an_operator_p(call_function(c), FIELD))
	{
	  list l = call_arguments(c);
	  rhs_tmp = EXPRESSION (CAR(CDR(l)));
	}
  }
  
  if(type_double_pointer_p(lhs_tmp)&& type_pointer_p(rhs_tmp)){
	effect e1 = effect_undefined, e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	list l1 = generic_proper_effects_of_complex_address_expression(ex,
																   &e1,
																   true);
	list l2 = generic_proper_effects_of_complex_address_expression(e,
																   &e2,
																   false);
	effects_free(l1);
	effects_free(l2);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	ent1 = argument_entity(lhs_tmp);


	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_dereferencing(p1);
	// recuperation of y
	ref2 = effect_any_reference(copy_effect(e2));
	ent2 = reference_variable(ref2);
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_dereferencing(p2);

	/* creation of the set written_pts_to =
	   {(x1,x2,rel)| (x, x1, EXACT), (x1, x2, rel) /in pts_to_set}*/
	SET_FOREACH(points_to, i, pts_to_set){
	  if( access_equal_p(points_to_source(i), source) &&
		  approximation_exact_p(points_to_relation(i))){
		SET_FOREACH(points_to, j,pts_to_set ){
		  if( access_equal_p(points_to_source(j) ,
				     points_to_sink(i)))
			written_pts_to = set_add_element(written_pts_to,
							 written_pts_to, (void *)j);
		}
	  }
	}
	/* {(x1, x2,EXACT)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
	SET_FOREACH(points_to, k, pts_to_set){
	  if( access_equal_p(points_to_source(i), source)
		  && approximation_may_p(points_to_relation(k))){
		SET_FOREACH(points_to, h,pts_to_set ){
		  if(access_equal_p(points_to_source(h),points_to_sink(k))&&
			 approximation_exact_p(points_to_relation(h)))
			s2 = set_add_element(s2, s2, (void *)h);
		}
	  }
	}

	/* {(x1, x2,MAY)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
	SET_FOREACH(points_to, l, pts_to_set){
	  if(access_equal_p(points_to_source(l), source) &&
		 approximation_may_p(points_to_relation(l))){
		SET_FOREACH(points_to, m,pts_to_set){
		  if(access_equal_p( points_to_source(m),points_to_sink(l))&&
			 approximation_exact_p(points_to_relation(m))){
			points_to_relation(m) = make_approximation_may();
			s3 = set_add_element(s3, s3, (void *)m);
		  }
		}
	  }
	}
	change_pts_to = set_difference(change_pts_to,pts_to_set, s3);
	change_pts_to = set_union(change_pts_to,change_pts_to, s3);
	SET_FOREACH(points_to, n, pts_to_set){
	  if(access_equal_p(points_to_source(n), source)){
		SET_FOREACH(points_to, o, pts_to_set){
		  if(access_equal_p(points_to_source(o) , sink)){
			nouv_source = copy_access(points_to_sink(n));
			nouv_sink = copy_access(points_to_sink(o));
			rel = fusion_approximation(points_to_relation(n),
									   points_to_relation(o));
			pt_to = make_points_to(nouv_source, nouv_sink, rel);
			gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
										 (void *)pt_to);
		  }
		}
	  }
	}
	s1 = set_difference(s1, change_pts_to, written_pts_to);
	pts_to_set = set_union(pts_to_set, gen_pts_to, s1);
  }
  else {
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;

}

/* one basic case of Emami: < m.x = y->a > */
set basic_field_ptr_to_field(set pts_to_set,
			     expression lhs,
			     expression rhs)
{
  return pts_to_set;
}

/* one basic case of Emami: < m->x =&y > */
set basic_ptr_to_field_addr(set pts_to_set,
							expression lhs,
							expression rhs)
{
 set gen_pts_to =set_generic_make(set_private,
								   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  points_to pt = points_to_undefined;
  syntax syn1 = syntax_undefined;
  syntax syn2 = syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  call c2 = call_undefined;
  entity ent1 = entity_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  approximation rel = approximation_undefined;
  list l = NIL;
  ifdebug(1){
	pips_debug(1, " case m->x = &y\n");
  }

  syn2 = expression_syntax(rhs);
  c2 = syntax_call(syn2);
  list args = call_arguments(c2);
  expression  rhs_tmp = EXPRESSION (CAR(args));
  ref2 = expression_reference(rhs_tmp);
  ent2 = argument_entity(rhs_tmp);
  syn1 =expression_syntax(copy_expression(lhs));
  call  c1 = syntax_call(syn1);
  list args1 = call_arguments(c1);
  expression  lhs_tmp = EXPRESSION (CAR(CDR(args1)));
  if(type_pointer_p(lhs_tmp)){
	// creation of the source
	//syntax ss = expression_syntax(lhs_tmp);
	effect e1 = effect_undefined;
	set_methods_for_proper_simple_effects();
	l = generic_proper_effects_of_complex_address_expression(lhs, &e1, true);
	effects_free(l);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	ent1 = reference_variable(ref1);
	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_referencing(p1);
	// creation of the sink
	effect e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	l = generic_proper_effects_of_complex_address_expression(rhs_tmp, &e2, false);
	effects_free(l);
	generic_effects_reset_all_methods();
	ref2 = effect_any_reference(copy_effect(e2));
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_addressing(p2);
	// creation of the approximation
	rel = make_approximation_exact();
	// creation of the points_to relation
	pt = make_points_to(source, sink, rel);
	// add the points_to relation to the set generated
	//by this assignement
	gen_pts_to=set_add_element(gen_pts_to, gen_pts_to,(void*) pt);
	// creation of the written set
	// search of all the points_to relations in the
	// alias set where the source is equal to the lhs
	SET_FOREACH(points_to, i,pts_to_set){
	  if(access_equal_p(points_to_source(i),source))
		written_pts_to = set_add_element(written_pts_to,
										 written_pts_to,
										 (void *)i);
	}

	pts_to_set=set_difference(pts_to_set, pts_to_set, written_pts_to);
	pts_to_set = set_union(pts_to_set, pts_to_set, gen_pts_to);

	ifdebug(1){
	  print_points_to_set(stderr,"points To pour le cas 2 <m->x = &y> \n ",
						  pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* one basic case of Emami: < *x =m->y > */
set basic_deref_ptr_to_field(set pts_to_set,
							 expression lhs,
							 expression rhs)
{

 set s1=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s2=set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set s3= set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set change_pts_to= set_generic_make(set_private,
				      points_to_equal_p,points_to_rank);
  set gen_pts_to =set_generic_make(set_private,
				   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
					points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  call c1 = call_undefined;
  entity ent1 = entity_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_source = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  ifdebug(1){
	pips_debug(1, " case *x = m;y\n");
  }
  // recuperation of x
  syn1 = expression_syntax(lhs);
  c1 = syntax_call(syn1);
  list  args1 = call_arguments(c1);
  expression lhs_tmp = EXPRESSION (CAR(args1));
  expression ex = copy_expression(lhs_tmp);
  /* if we have *m.x = m->y */
  syntax s = expression_syntax(lhs_tmp);
  if(syntax_call_p(s))
  {
	call c = expression_call(lhs_tmp);
	if(entity_an_operator_p(call_function(c), FIELD))
	{
	  list l = call_arguments(c);
	  lhs_tmp = EXPRESSION (CAR(CDR(l)));
	}
  }
  // recuperation of y
  syntax syn2 = expression_syntax(rhs);
  call c2 = syntax_call(syn2);
  list  args2 = call_arguments(c2);
  expression rhs_tmp = EXPRESSION (CAR(args2));
  expression e = copy_expression(rhs_tmp);
  
  syntax ss = expression_syntax(rhs_tmp);
  if(syntax_call_p(ss))
  {
	call c = expression_call(rhs_tmp);
	if(entity_an_operator_p(call_function(c), POINT_TO))
	{
	  list l = call_arguments(c);
	  rhs_tmp = EXPRESSION (CAR(CDR(l)));
	}
  }
  
  if(type_double_pointer_p(lhs_tmp)&&
	 type_pointer_p(rhs_tmp)){
	effect e1 = effect_undefined, e2 = effect_undefined;
	set_methods_for_proper_simple_effects();
	list l1 = generic_proper_effects_of_complex_address_expression(ex,
																   &e1,
																   true);
	list l2 = generic_proper_effects_of_complex_address_expression(e,
																   &e2,
																   false);
	effects_free(l1);
	effects_free(l2);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(copy_effect(e1));
	ent1 = argument_entity(lhs_tmp);


	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_dereferencing(p1);
	// recuperation of y
	ref2 = effect_any_reference(copy_effect(e2));
	ent2 = reference_variable(ref2);
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_dereferencing(p2);

	/* creation of the set written_pts_to =
	   {(x1,x2,rel)| (x, x1, EXACT), (x1, x2, rel) /in pts_to_set}*/
	SET_FOREACH(points_to, i, pts_to_set){
	  if( access_equal_p(points_to_source(i), source) &&
		  approximation_exact_p(points_to_relation(i))){
		SET_FOREACH(points_to, j,pts_to_set ){
		  if( access_equal_p(points_to_source(j) ,
				     points_to_sink(i)))
			written_pts_to = set_add_element(written_pts_to,
							 written_pts_to, (void *)j);
		}
	  }
	}
	/* {(x1, x2,EXACT)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
	SET_FOREACH(points_to, k, pts_to_set){
	  if( access_equal_p(points_to_source(i), source)
		  && approximation_may_p(points_to_relation(k))){
		SET_FOREACH(points_to, h,pts_to_set ){
		  if(access_equal_p(points_to_source(h),points_to_sink(k))&&
			 approximation_exact_p(points_to_relation(h)))
			s2 = set_add_element(s2, s2, (void *)h);
		}
	  }
	}

	/* {(x1, x2,MAY)|(x, x1, MAY),(x1, x2, EXACT) /in pts_to_set}*/
	SET_FOREACH(points_to, l, pts_to_set){
	  if(access_equal_p(points_to_source(l), source) &&
		 approximation_may_p(points_to_relation(l))){
		SET_FOREACH(points_to, m,pts_to_set){
		  if(access_equal_p( points_to_source(m),points_to_sink(l))&&
			 approximation_exact_p(points_to_relation(m))){
			points_to_relation(m) = make_approximation_may();
			s3 = set_add_element(s3, s3, (void *)m);
		  }
		}
	  }
	}
	change_pts_to = set_difference(change_pts_to,pts_to_set, s3);
	change_pts_to = set_union(change_pts_to,change_pts_to, s3);
	SET_FOREACH(points_to, n, pts_to_set){
	  if(access_equal_p(points_to_source(n), source)){
		SET_FOREACH(points_to, o, pts_to_set){
		  if(access_equal_p(points_to_source(o) , sink)){
			nouv_source = copy_access(points_to_sink(n));
			nouv_sink = copy_access(points_to_sink(o));
			rel = fusion_approximation(points_to_relation(n),
									   points_to_relation(o));
			pt_to = make_points_to(nouv_source, nouv_sink, rel);
			gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
										 (void *)pt_to);
		  }
		}
	  }
	}
	s1 = set_difference(s1, change_pts_to, written_pts_to);
	pts_to_set = set_union(pts_to_set, gen_pts_to, s1);
  }
  else {
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* one basic case of Emami: < m->x = y.a > */
set basic_ptr_to_field_field(set pts_to_set,
							 expression lhs,
							 expression rhs)
{
 set gen_pts_to =set_generic_make(set_private,
				   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
					points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=expression_syntax(lhs);
  syntax syn2=expression_syntax(rhs);
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  expression lhs_tmp = expression_undefined;
  expression rhs_tmp = expression_undefined;
  ifdebug(1) printf("\n cas x = y \n");
  if(syntax_call_p(syn1)) {
    call c1 = expression_call(lhs);
    if(entity_an_operator_p(call_function(c1), POINT_TO)){
      list l = call_arguments(c1);
      lhs_tmp = EXPRESSION (CAR(CDR(l)));
      syn1=expression_syntax(lhs_tmp);
    }
  }else{
    syn1=expression_syntax(lhs);
  }
  if(syntax_call_p(syn2)){
    call c2 = expression_call(rhs);
    if(entity_an_operator_p(call_function(c2),FIELD)){
      list l1 = call_arguments(c2);
      rhs_tmp = EXPRESSION (CAR(CDR(l1)));
      syn2=expression_syntax(rhs_tmp);
      ref2=expression_reference(rhs_tmp);
      ent2=reference_variable(ref2);
	}
  }else{
    rhs_tmp = copy_expression(rhs);
    syn2=expression_syntax(rhs);
    ref2=expression_reference(rhs);
    ent2=reference_variable(ref2);
  }
  if(syntax_reference_p(syn1) && syntax_reference_p(syn2)){
    ent2=reference_variable(ref2);
	if((type_pointer_p(lhs_tmp)&&(type_pointer_p(rhs_tmp) || array_entity_p(ent2))) ||
	   (type_double_pointer_p(lhs_tmp)&& type_double_pointer_p(rhs_tmp))){
	  // creation of the source
	  effect e1 = effect_undefined, e2 = effect_undefined;
	  set_methods_for_proper_simple_effects();
	  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
									 &e1,
									 true);
	  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
									 &e2,
									 false);
	  effects_free(l1);
	  effects_free(l2);
	  generic_effects_reset_all_methods();
	  ref1 = effect_any_reference(copy_effect(e1));
	  points_to_path p1 = make_points_to_path(ref1);
	  source = make_access_referencing(p1);

	  // add the points_to relation to the set generated
	  // by this assignement
	  ref2 = effect_any_reference(copy_effect(e2));
	  points_to_path p2 = make_points_to_path(ref2);
	  sink = make_access_referencing(p2);
	  set s = set_generic_make(set_private,
							   points_to_equal_p,points_to_rank);
	  SET_FOREACH(points_to, i, pts_to_set){
		if(access_equal_p(points_to_source(i), sink))
		  s = set_add_element(s, s, (void*)i);
	  }
	  SET_FOREACH(points_to, j, s){
		nouv_sink = copy_access(points_to_sink(j));
		// access nouv_source = copy_access(source);
		rel = points_to_relation(j);
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
					     (void*) pt_to );
	  }
	  /* in case x = y[i]*/
	  if(array_entity_p(ent2)){
		points_to_path p2 =make_points_to_path(ref2);
		nouv_sink = make_access_referencing(p2);
		// access nouv_source = copy_access(source);
		rel =make_approximation_exact();
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
					     (void*) pt_to );
	  }

	  // creation of the written set
	  // search of all the points_to relations in the
	  // alias set where the source is equal to the lhs
	  SET_FOREACH(points_to, k, pts_to_set){
		if(access_equal_p(points_to_source(k), source))
		  written_pts_to = set_add_element(written_pts_to,
						   written_pts_to, (void *)k);
	  }
	  pts_to_set = set_difference(pts_to_set,
				      pts_to_set,
				      written_pts_to);
	  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
	  ifdebug(1)
		print_points_to_set(stderr,"Points to pour le cas 1 <x = y>\n",
							pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* one basic case of Emami: < m->x = m > */
set basic_ptr_to_field_struct(set pts_to_set,
							  expression lhs,
							  expression rhs)
{
  return pts_to_set;
}


/* one basic case of Emami: < m->x = *y > */
set basic_ptr_to_field_deref(set pts_to_set,
							 expression lhs,
							 expression rhs)
{
  return pts_to_set;
}

/* one basic case of Emami: < m->x = y > */
set basic_ptr_to_field_ref(set pts_to_set,
						   expression lhs,
						   expression rhs)
{
 set gen_pts_to =set_generic_make(set_private,
				   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
					points_to_equal_p,points_to_rank);
  points_to pt_to = points_to_undefined;
  syntax syn1=expression_syntax(lhs);
  syntax syn2=expression_syntax(rhs);
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent2 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  access nouv_sink = access_undefined;
  approximation rel = approximation_undefined;
  expression lhs_tmp = expression_undefined;
  expression rhs_tmp = expression_undefined;
  ifdebug(1) printf("\n cas m->x = y \n");
  if(syntax_call_p(syn1)){ 
    call c1 = expression_call(lhs);
  if(entity_an_operator_p(call_function(c1), POINT_TO)){
	  list l = call_arguments(c1);
	  lhs_tmp = EXPRESSION (CAR(CDR(l)));
	  syn1=expression_syntax(lhs_tmp);
  }
  }else{
	lhs_tmp = copy_expression(lhs);
 }
  if(syntax_reference_p(syn1) && syntax_reference_p(syn2)){
    ref2 = syntax_reference(syn2);
    ent2=reference_variable(ref2);
	if((type_pointer_p(lhs_tmp)&&(type_pointer_p(rhs) || array_entity_p(ent2))) ||
	   (type_double_pointer_p(lhs_tmp)&& type_double_pointer_p(rhs))){
	  // creation of the source
	  effect e1 = effect_undefined, e2 = effect_undefined;
	  set_methods_for_proper_simple_effects();
	  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
									 &e1,
									 true);
	  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
									 &e2,
									 false);
	  effects_free(l1);
	  effects_free(l2);
	  generic_effects_reset_all_methods();
	  ref1 = effect_any_reference(copy_effect(e1));
	  points_to_path p1 = make_points_to_path(ref1);
	  source = make_access_referencing(p1);

	  // add the points_to relation to the set generated
	  // by this assignement
	  ref2 = effect_any_reference(copy_effect(e2));
	  points_to_path p2 = make_points_to_path(ref2);
	  sink = make_access_referencing(p2);
	  set s = set_generic_make(set_private,
							   points_to_equal_p,points_to_rank);
	  SET_FOREACH(points_to, i, pts_to_set){
		if(access_equal_p(points_to_source(i), sink))
		  s = set_add_element(s, s, (void*)i);
	  }
	  SET_FOREACH(points_to, j, s){
		nouv_sink = copy_access(points_to_sink(j));
		// access nouv_source = copy_access(source);
		rel = points_to_relation(j);
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
					     (void*) pt_to );
	  }
	  /* in case x = y[i]*/
	  if(array_entity_p(ent2)){
		points_to_path p2 =make_points_to_path(ref2);
		nouv_sink = make_access_referencing(p2);
		// access nouv_source = copy_access(source);
		rel =make_approximation_exact();
		pt_to = make_points_to(source, nouv_sink, rel);
		gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
					     (void*) pt_to );
	  }

	  // creation of the written set
	  // search of all the points_to relations in the
	  // alias set where the source is equal to the lhs
	  SET_FOREACH(points_to, k, pts_to_set){
		if(access_equal_p(points_to_source(k), source))
		  written_pts_to = set_add_element(written_pts_to,
						   written_pts_to, (void *)k);
	  }
	  pts_to_set = set_difference(pts_to_set,
				      pts_to_set,
				      written_pts_to);
	  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
	  ifdebug(1)
		print_points_to_set(stderr,"Points to pour le cas 1 <x = y>\n",
							pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}

/* to compute m.a = n.a where a is of pointer type*/
set struct_pointer(set pts_to_set, expression lhs, expression rhs)
{
  set gen_pts_to = set_generic_make(set_private,
									points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  effect e1 = effect_undefined, e2 = effect_undefined;
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
																 &e1,
																 true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
																 &e2,
																 false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  reference  ref1 = effect_any_reference(copy_effect(e1));
  points_to_path p1 = make_points_to_path(ref1);
	// print_reference(ref1);
  access source = make_access_referencing(p1);
  // add the points_to relation to the set generated
  // by this assignement
  reference ref2 = effect_any_reference(copy_effect(e2));
	// print_reference(ref2);
  points_to_path p2 = make_points_to_path(ref2);
  access sink = make_access_referencing(p2);
  set s = set_generic_make(set_private,
						   points_to_equal_p,points_to_rank);
  SET_FOREACH(points_to, i, pts_to_set)
  {
	if(access_equal_p(points_to_source(i), sink))
	  s = set_add_element(s, s, (void*)i);
  }
  SET_FOREACH(points_to, j, s)
  {
	access  nouv_sink = copy_access(points_to_sink(j));
	// access nouv_source = copy_access(source);
	approximation rel = points_to_relation(j);
	points_to pt_to = make_points_to(source,
									 nouv_sink,
									 rel);
	gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
								 (void*) pt_to );
  }
  SET_FOREACH(points_to, k, pts_to_set){
	if(access_equal_p(points_to_source(k), source))
	  written_pts_to = set_add_element(written_pts_to,
									   written_pts_to, (void *)k);
  }
  pts_to_set = set_difference(pts_to_set,
							  pts_to_set,
							  written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
  ifdebug(1)
	print_points_to_set(stderr,"Points to pour le cas 1 <x = y>\n",
						pts_to_set);

  return pts_to_set;

}


/* to compute m.a = n.a where a is of pointer type*/
set struct_double_pointer(set pts_to_set, expression lhs, expression rhs)
{
  set gen_pts_to = set_generic_make(set_private,
									points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
										points_to_equal_p,points_to_rank);
  effect e1 = effect_undefined, e2 = effect_undefined;
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(lhs,
																 &e1,
																 true);
  list l2 = generic_proper_effects_of_complex_address_expression(rhs,
																 &e2,
																 false);
  effects_free(l1);
  effects_free(l2);
  generic_effects_reset_all_methods();
  reference  ref1 = effect_any_reference(copy_effect(e1));
  points_to_path p1 = make_points_to_path(ref1);
  access source = make_access_referencing(p1);
  // add the points_to relation to the set generated
  // by this assignement
  reference ref2 = effect_any_reference(copy_effect(e2));
  points_to_path p2 = make_points_to_path(ref2);
  access sink = make_access_referencing(p2);
  set s = set_generic_make(set_private,
						   points_to_equal_p,points_to_rank);
  SET_FOREACH(points_to, i, pts_to_set)
  {
	if(access_equal_p(points_to_source(i), sink))
	  s = set_add_element(s, s, (void*)i);
  }
  SET_FOREACH(points_to, j, s)
  {
	access nouv_sink = copy_access(points_to_sink(j));
	// access nouv_source = copy_access(source);
	approximation rel = points_to_relation(j);
	points_to pt_to = make_points_to(source,
																	 nouv_sink,
																	 rel);
	gen_pts_to = set_add_element(gen_pts_to,gen_pts_to,
				     (void*) pt_to );
  }
  SET_FOREACH(points_to, k, pts_to_set){
	if(access_equal_p(points_to_source(k), source))
	  written_pts_to = set_add_element(written_pts_to,
					   written_pts_to, (void *)k);
  }
  pts_to_set = set_difference(pts_to_set,
			      pts_to_set,
			      written_pts_to);
  pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
  ifdebug(1)
	print_points_to_set(stderr,"Points to pour le cas 1 <x = y>\n",
						pts_to_set);

  return pts_to_set;

}

// to decompose the assignment m = n where  m and n are respectively
// of type struct
// the result should be m.field1 = n.field2... A.M

set struct_decomposition(expression lhs,
			 expression rhs,
			 set pt_in)
{
  set pt_out = set_generic_make(set_private,
				points_to_equal_p,points_to_rank);
  pt_out = set_assign(pt_out, pt_in);
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
  FOREACH(ENTITY, i, l1)
  {
	if(type_double_pointer_p(entity_to_expression(i)) ||type_pointer_p(entity_to_expression(i)))
	{
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
set basic_ref_heap(set pts_to_set,
		   expression lhs,
		   expression rhs,
		   statement current)
{

  set gen_pts_to =set_generic_make(set_private,
				   points_to_equal_p,points_to_rank);
  set written_pts_to = set_generic_make(set_private,
					points_to_equal_p,points_to_rank);

  points_to pt_to = points_to_undefined;
  syntax syn1=syntax_undefined;
  syntax syn2=syntax_undefined;
  reference ref1 = reference_undefined;
  reference ref2 = reference_undefined;
  entity ent1 = entity_undefined;
  access source = access_undefined;
  access sink = access_undefined;
  approximation rel = approximation_undefined;
  string ss;

  ifdebug(1){
	pips_debug(1, " case  x =()malloc(sizeof()) \n");
  }
  syn1=expression_syntax(lhs);
  ref1=syntax_reference(syn1);
  ent1=reference_variable(ref1);
  entity en = entity_undefined;
  syn2 = expression_syntax(rhs);
  list l = NIL;
  if(syntax_call_p(syn2)){
	call c = syntax_call(syn2);
	l = call_arguments(c);
	en = call_function(c);
  }
  if(syntax_cast_p(syn2)){
	cast ct = syntax_cast(syn2);
	expression e = cast_expression(ct);
	syntax s = expression_syntax(e);
	if(syntax_call_p(s)){
	  call cc = syntax_call(s);
	  l = call_arguments(cc);
	  en = call_function(cc);
	  ss = entity_local_name(en);

	}
  }

  if(type_pointer_p(lhs)){
	// creation of the source
	effect e1 = effect_undefined;
	set_methods_for_proper_simple_effects();
	set_methods_for_proper_references();
	list  l1 = generic_proper_effects_of_complex_address_expression(lhs,
																	&e1,
																	true);
	effects_free(l1);
	generic_effects_reset_all_methods();
	ref1 = effect_any_reference(e1);
	ref2 =  malloc_to_abstract_location(ref1, type_undefined, type_undefined, expression_undefined,
																			entity_undefined, statement_number(current));
	
	points_to_path p1 = make_points_to_path(ref1);
	source = make_access_referencing(p1);

	// creation of the sink
	points_to_path p2 = make_points_to_path(ref2);
	sink = make_access_addressing(p2);
	// fetch the points to relations
	// where source = source 1
	// creation of the written set
	// search of all the points_to relations in the
	// alias set where the source is equal to the lhs
	SET_FOREACH(points_to, elt, pts_to_set)
	{
	  if(access_equal_p(points_to_source(elt),
						source))
		written_pts_to = set_add_element(written_pts_to,
										 written_pts_to,(void*)elt);
	}

	rel = make_approximation_may();
	pt_to = make_points_to(source,sink, rel);
	gen_pts_to = set_add_element(gen_pts_to, gen_pts_to,
								 (void *) pt_to );



	pts_to_set = set_difference(pts_to_set, pts_to_set,written_pts_to);
	pts_to_set = set_union(pts_to_set, gen_pts_to, pts_to_set);
	ifdebug(1){
	  print_points_to_set(stderr,"Points To pour le cas 3 <x ==()malloc(sizeof()) > \n",
						  pts_to_set);
	}
  }
  else{
	ifdebug(1) {
	  pips_debug(1, "Neither variable is a pointer\n");
	}
  }
  return pts_to_set;
}





/* Compute the new points-to set for any assignment lhs = rhs */
set points_to_assignment(statement current,
			 expression lhs,
			 expression rhs,
			 set pt_in)
{
  set pt_out = set_generic_make(set_private,
								points_to_equal_p,points_to_rank);
  pt_out = set_assign(pt_out,pt_in);
  int rlt1=0;
  int rlt2=0;
  if(instruction_expression_p(statement_instruction(current))){
	expression   e = instruction_expression(statement_instruction(current));
  }
  // FI: lhs qnd rhs qre going to be much more general than simple references...
  rlt1 = emami_expression_type(lhs);
  rlt2 = emami_expression_type(rhs);
  switch (rlt1){
	/* cas x = y
	   write = {(x, x1, rel)|(x, x1, rel) in input}
	   gen  = {(x, y1, rel)|(y, y1, rel) in input }
	   pts_to=gen+(input - kill)
	*/
  case EMAMI_NEUTRAL :
	switch(rlt2){
	case EMAMI_NEUTRAL :{
	  // cas x = y
	  /* if(! set_empty_p( basic_ref_ref(effects, pt_in, lhs, rhs))) */
	  pt_out = set_assign(pt_out,
			      basic_ref_ref(pt_in, copy_expression(lhs),
					    copy_expression(rhs)));
	  break;
	}
	case EMAMI_ADDRESS_OF:{
	  // cas x = &y
	  if(! set_empty_p(basic_ref_addr(pt_in,copy_expression(lhs),
					  copy_expression(rhs))))
		pt_out = set_assign(pt_out,
				    basic_ref_addr(pt_in,
						   copy_expression(lhs),
						   copy_expression(rhs)));
	  break;
	}
	case EMAMI_DEREFERENCING:{
	  // cas x = *y
	  /*  if(! set_empty_p(basic_ref_deref(effects, pt_in, lhs, rhs))) */
	  pt_out = set_assign(pt_out,
			      basic_ref_deref(pt_in,
					      copy_expression(lhs),
					      copy_expression(rhs)));
	  break;
	}
	case EMAMI_HEAP:{
	  // cas x = *y
	  /*  if(! set_empty_p(basic_ref_deref(effects, pt_in, lhs, rhs))) */
	  pt_out = set_assign(pt_out,
			      basic_ref_heap( pt_in,
					      copy_expression(lhs),
					      copy_expression(rhs),
					      current));
	  break;
	}
	case EMAMI_FIELD:{
	  // cas x = *y
	  /*  if(! set_empty_p(basic_ref_deref(effects, pt_in, lhs, rhs))) */
	  pt_out = set_assign(pt_out,
			      basic_ref_field(pt_in,
					      copy_expression(lhs),
					      copy_expression(rhs)));
	  break;
	}
	case EMAMI_POINT_TO:{
	  // cas x = *y
	  /*  if(! set_empty_p(basic_ref_deref(effects, pt_in, lhs, rhs))) */
	  pt_out = set_assign(pt_out,
						  basic_ref_ptr_to_field(pt_in,
												 copy_expression(lhs),
												 copy_expression(rhs)));
	  break;
	}
	case EMAMI_STRUCT :
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  // should we add a function that just return the same input set ?
	  break;
	case EMAMI_ARRAY :{
	  pt_out = set_assign(pt_out,
						  basic_ref_array(pt_in,
										  copy_expression(lhs),
										  copy_expression(rhs)));
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	}
	default:
	  ifdebug(1) fprintf(stderr,"\n aucun pattern defini \n");
	  break;
	}
	break;
  case EMAMI_ADDRESS_OF:
	ifdebug(1) fprintf(stderr," \n aucun pattern defini \n");
	break;
  case EMAMI_DEREFERENCING:
	switch(rlt2)
	{
	case EMAMI_NEUTRAL:{
	  // cas *x = y
	  /* if(! set_empty_p(basic_deref_ref(effects, pt_in, lhs, rhs))) */
	  pt_out = set_assign(pt_out, basic_deref_ref(pt_in,
												  copy_expression(lhs),
												  copy_expression(rhs)));
	  break;
	}
	case EMAMI_ADDRESS_OF:
	  // cas *x = &y
	  pt_out = set_assign(pt_out,
						  basic_deref_addr(pt_in,
										   copy_expression(lhs),
										   copy_expression(rhs)));
	  break;
	case EMAMI_DEREFERENCING:{
	  //cas *x = *y
	  /* if(! set_empty_p(basic_deref_deref(effects, pt_in, lhs, rhs))) */
	  pt_out = set_assign(pt_out,
						  basic_deref_deref(pt_in,
											copy_expression(lhs),
											copy_expression(rhs)));
	  break;
	}
	case EMAMI_FIELD :
	  pt_out = set_assign(pt_out,
						  basic_deref_field(pt_in,
											copy_expression(lhs),
											copy_expression(rhs)));
	  break;
	case EMAMI_POINT_TO:{
	  pt_out = set_assign(pt_out,
						  basic_deref_ptr_to_field(pt_in,
												   copy_expression(lhs),
												   copy_expression(rhs)));
	  break;
	}
	case EMAMI_STRUCT :
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	case EMAMI_ARRAY :

	  break;
	default:
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	}
	break;
  case EMAMI_FIELD:
	switch(rlt2)
	{
	case EMAMI_NEUTRAL:{
	  // cas x.a = y
	  /* if(! set_empty_p(basic_deref_ref(effects, pt_in, lhs, rhs))) */
	  //pt_out = set_assign(pt_out, basic_field_ref(effects, pt_in, lhs, rhs));
	  break;
	}
	case EMAMI_ADDRESS_OF:
	  // cas x.a = &y
	  pt_out = set_assign(pt_out,
						  basic_field_addr(pt_in,
										   copy_expression(lhs),
										   copy_expression(rhs)));
	  break;
	case EMAMI_DEREFERENCING:{
	  //cas *(x.a) = *y
	  break;
	}
	case EMAMI_FIELD :
	  pt_out = set_assign(pt_out,
						  basic_ref_ref(pt_in,
										copy_expression(lhs),
										copy_expression(rhs)));
	  break;
	case EMAMI_POINT_TO:{
	  pt_out = set_assign(pt_out,
						  basic_field_ptr_to_field(pt_in,
												   copy_expression(lhs),
												   copy_expression(rhs)));
	  break;
	}
	case EMAMI_STRUCT :
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	default:
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	}
  case EMAMI_STRUCT:
	switch(rlt2)
	{
	case EMAMI_NEUTRAL:{
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	}
	case EMAMI_ADDRESS_OF:
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	case EMAMI_DEREFERENCING:{
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	}
	case EMAMI_FIELD :
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	case EMAMI_STRUCT :{
	  //fprintf(stderr,"case struct\n");
	  pt_out = struct_decomposition(lhs, rhs, pt_in);
	  // ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	}
	default:
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	}
  default:
  {
	ifdebug(1)
	  fprintf(stderr,"\n  aucun pattern defini\n ");
	break;
  }
  case EMAMI_POINT_TO:
	switch(rlt2)
	{
	case EMAMI_NEUTRAL:{
	  pt_out = set_assign(pt_out,
						  basic_ptr_to_field_ref(pt_in,
												 copy_expression(lhs),
												 copy_expression(rhs)));
	  break;
	}
	case EMAMI_ADDRESS_OF:
	  pt_out = set_assign(pt_out,
						  basic_ptr_to_field_addr(pt_in,
												  copy_expression(lhs),
												  copy_expression(rhs)));
	  break;
	case EMAMI_DEREFERENCING:{
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  pt_out = set_assign(pt_out,
						  basic_ptr_to_field_deref(pt_in, copy_expression(lhs),
												   copy_expression(rhs)));
	  break;
	}
	case EMAMI_FIELD :
	  pt_out = set_assign(pt_out,
						  basic_ptr_to_field_field(pt_in, copy_expression(lhs),
												   copy_expression(rhs)));
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	case EMAMI_STRUCT :{
	  pt_out = set_assign(pt_out,
						  basic_ptr_to_field_struct(pt_in, copy_expression(lhs),
													copy_expression(rhs)));

	  break;
	}
	case EMAMI_POINT_TO:{
	  pt_out = set_assign(pt_out,
						  basic_ptr_to_field_field( pt_in, copy_expression(lhs),
													copy_expression(rhs)));
	  break;
	}
	default:
	  ifdebug(1)	fprintf(stderr,"\n aucun pattern defini\n ");
	  break;
	}

  }

  return pt_out;
}

/* compute the points to set associate to a sequence of statements*/
set points_to_sequence(sequence seq, set pt_in, bool store)
{
  set pt_out = set_generic_make(set_private, points_to_equal_p,points_to_rank);
  pt_out = set_assign(pt_out, pt_in);
  FOREACH(statement, st, sequence_statements(seq)){
	pt_out = set_assign(pt_out, recursive_points_to_statement(st,pt_out,store));
  }
  return pt_out;
}

/* compute the points-to set for intrinsic-call*/
set points_to_intrinsic(statement s,
						entity e,
						list pc,
						set pt_in)
{
  set pt_out = set_generic_make(set_private,
								points_to_equal_p, points_to_rank);

  pips_debug(8, "begin\n");
  if(ENTITY_ASSIGN_P(e)){
	expression lhs = EXPRESSION(CAR(pc));
	expression rhs = EXPRESSION(CAR(CDR(pc)));
	pt_out = points_to_assignment(s, copy_expression(lhs), copy_expression(rhs), pt_in);
  }
  else if(ENTITY_PLUS_UPDATE_P(e) || ENTITY_MINUS_UPDATE_P(e)
		  || ENTITY_MULTIPLY_UPDATE_P(e) || ENTITY_DIVIDE_UPDATE_P(e)
		  || ENTITY_MODULO_UPDATE_P(e) || ENTITY_LEFT_SHIFT_UPDATE_P(e)
		  || ENTITY_RIGHT_SHIFT_UPDATE_P(e) || ENTITY_BITWISE_AND_UPDATE_P(e)
		  || ENTITY_BITWISE_XOR_UPDATE_P(e) || ENTITY_BITWISE_OR_UPDATE_P(e)){
	pt_out = set_assign(pt_out, pt_in);
  }
  else if(ENTITY_POST_INCREMENT_P(e) || ENTITY_POST_DECREMENT_P(e)
		  || ENTITY_PRE_INCREMENT_P(e) || ENTITY_PRE_DECREMENT_P(e)) {
	pt_out = set_assign(pt_out, pt_in);
  }
  else if(ENTITY_C_RETURN_P(e)) {
	pt_out = set_assign(pt_out, pt_in);
  }
  else if(ENTITY_STOP_P(e)||ENTITY_ABORT_SYSTEM_P(e)||ENTITY_EXIT_SYSTEM_P(e))
	pt_out = set_assign(pt_out, pt_in);
  else if(ENTITY_COMMA_P(e)) {
	pt_out = set_assign(pt_out, pt_in);
  }
  else
	pt_out =  set_assign(pt_out, pt_in);
  pips_debug(8, "end\n");
  return pt_out;
}



/* compputing the points-to set of a while loop by iterating over its
   body until reaching a fixed-point. For the moment without taking
   into account the condition's side effect. */
set points_to_whileloop(whileloop wl, set pt_in, bool store)
{
  /* get the condition,to used laterly to refine the points-to set
	 expression cond = whileloop_condition(wl);*/

  statement while_body = whileloop_body(wl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set pt_body = set_generic_make(set_private, points_to_equal_p,points_to_rank);

  do{
	pt_body = set_assign(pt_body, pt_in);
	pt_out =set_assign(pt_out, recursive_points_to_statement(while_body,
															 pt_body, false));
	pt_in = set_assign(pt_in, set_clear(pt_in));
	pt_in = set_assign(pt_in, merge_points_to_set(pt_body, pt_out));
	pt_out = set_clear(pt_out);
  }
  while(!set_equal_p(pt_body, pt_in));
  pt_out = set_assign(pt_out, pt_in);
  points_to_storage(pt_out,while_body , true);

  return  pt_out;
}

/* computing the points to of a for loop, before processing the body,
   compute the points to of the initialization. */
set points_to_forloop(forloop fl,
					  set pt_in,
					  bool store)
{
  statement for_body = forloop_body(fl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set pt_body = set_generic_make(set_private, points_to_equal_p,points_to_rank);
  expression exp = forloop_initialization(fl);
  pt_in = points_to_expression(exp, pt_in, true);
  do{
	pt_body = set_assign(pt_body, pt_in);
	pt_out =set_assign(pt_out, recursive_points_to_statement(for_body,
															 pt_body, store));
	pt_in = set_clear(pt_in);
	pt_in = set_assign(pt_in,merge_points_to_set(pt_body, pt_out));
	pt_out = set_clear(pt_out);
  }
  while(!set_equal_p(pt_body, pt_in));
  pt_out =set_assign(pt_out, pt_in);
  points_to_storage(pt_out,for_body , true);

  return  pt_out;
}

/*Computing the points to of a do while loop, we have to process the
  body a least once, before iterating until reaching the fixed-point. */
set points_to_do_whileloop(whileloop fl, set pt_in, bool store)
{
  statement dowhile_body = whileloop_body(fl);
  set pt_out = set_generic_make(set_private, points_to_equal_p,points_to_rank);
  set pt_body = set_generic_make(set_private, points_to_equal_p,points_to_rank);
  pt_out =set_assign(pt_out, recursive_points_to_statement(dowhile_body,
														   pt_in, false));
  pt_in = set_assign(pt_in, pt_out);
  do{
	pt_body = set_assign(pt_body, pt_in);
	pt_out =set_assign(pt_out, recursive_points_to_statement(dowhile_body,
															 pt_body, store));
	pt_in =set_assign(pt_in, set_clear(pt_in));
	pt_in = set_assign(pt_in,merge_points_to_set(pt_body, pt_out));
	pt_out = set_clear(pt_out);
  }
  while(!set_equal_p(pt_body, pt_in));
  pt_out =set_assign(pt_out, pt_in);
  points_to_storage(pt_out,dowhile_body , true);

  return  pt_out;

}

/*Computing the points to of a test, all the relationships are of type
  MAY, can be refined later by using precontions. */
set points_to_test(test test_stmt, set pt_in, bool store)
{
  statement true_stmt = statement_undefined;
  statement false_stmt = statement_undefined;

  set true_pts_to= set_generic_make(set_private,
									points_to_equal_p,points_to_rank);
  set false_pts_to = set_generic_make(set_private,
									  points_to_equal_p,points_to_rank);
  set rlt_pts_to = set_generic_make(set_private,
									points_to_equal_p,points_to_rank);
  /* condition's side effect and information not taked into account :
	 if(p==q) or if(*p++) */
  true_stmt = test_true(test_stmt);
  true_pts_to = set_union(true_pts_to,pt_in,
						  recursive_points_to_statement(true_stmt,pt_in,store));
  false_stmt = test_false(test_stmt);
  false_pts_to = set_union(false_pts_to,pt_in,
						   recursive_points_to_statement(false_stmt,
														 pt_in,
														 store));
  rlt_pts_to= merge_points_to_set(true_pts_to,false_pts_to);
  return rlt_pts_to;
}

/* computing the poinst to of a call, user_functions not yet implemented. */
set points_to_call(statement s,
				   call c,
				   set pt_in,
				   bool store
  )
{
  entity e = call_function(c);
  cons *pc = call_arguments(c);
  tag tt;

  set pt_out = set_generic_make(set_private,
								points_to_equal_p, points_to_rank);
  switch (tt = value_tag(entity_initial(e))) {
  case is_value_code:
	/* call to an external function; preliminary version*/
	pt_out = set_assign(pt_out, pt_in);
	break;
  case is_value_symbolic:
	pt_out = set_assign(pt_out, pt_in);
	break;
  case is_value_constant:
	pt_out = set_assign(pt_out, pt_in);
	break;
  case is_value_unknown:
	pips_internal_error("function %s has an unknown value\n", entity_name(e));
	break;
  case is_value_intrinsic:{
	pips_debug(5, "intrinsic function %s\n", entity_name(e));
	pt_out = set_assign(pt_out, points_to_intrinsic(s, e, pc, pt_in));
	break;
  }
  default:
	pips_internal_error("unknown tag %d\n", tt);
  }
  return pt_out;
}

/* Process an expression, test if it's a call or a reference*/
set points_to_expression(expression e, set pt_in, bool store)
{
  set pt_out = set_generic_make(set_private,
								points_to_equal_p, points_to_rank);
  call c = call_undefined;
  statement st = statement_undefined;
  syntax s = expression_syntax(copy_expression(e));
  switch (syntax_tag(s))
  {
  case is_syntax_call:{
	c = syntax_call(s);
	st = make_expression_statement(e);
	pt_out = set_assign(pt_out,points_to_call(st, c, pt_in, store));
	break;
  }
  case is_syntax_cast:{
	cast ct = cast_undefined;
	expression e = expression_undefined;
	ct = syntax_cast(s);
	e = cast_expression(ct);
	st = make_expression_statement(e);
	pt_out = set_assign(pt_out,points_to_expression(e, pt_in, store));
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
set recursive_points_to_statement(statement current, set pt_in, bool store)
{
  set pt_out = set_generic_make(set_private,
								points_to_equal_p, points_to_rank);
  pt_out = set_assign(pt_out, pt_in);
  instruction i = statement_instruction(current);
/*  Convert the pt_in set into a sorted list for storage */
/*  Store the current points-to list */
  points_to_storage(pt_in, current, store);
  ifdebug(1)  print_statement(current);

  switch(instruction_tag(i)) {
	/* instruction = sequence + test + loop + whileloop +
	   goto:statement +
	   call + unstructured + multitest + forloop  + expression ;*/

  case is_instruction_call:{
	pt_out = set_assign(pt_out,
						points_to_call(current,instruction_call(i),
									   pt_in, store));
	break;
  }
  case is_instruction_sequence:{
	pt_out = set_assign(pt_out,
						points_to_sequence(instruction_sequence(i),
										   pt_in, store));
	break;
  }
  case is_instruction_test:{
	pt_out = set_assign(pt_out,points_to_test(instruction_test(i),
											  pt_in, store));
	break;
  }
  case is_instruction_whileloop:{
	store = false;
	if(evaluation_tag(whileloop_evaluation(instruction_whileloop(i))) == 0){
	  pt_out = set_assign(pt_out,
						  points_to_whileloop(instruction_whileloop(i),
											  pt_in, false));
	}else 
	  pt_out = set_assign(pt_out,
						  points_to_do_whileloop(instruction_whileloop(i),
												 pt_in, false));

	break;
  }
  case is_instruction_forloop:{
	store = false;
	pt_out = set_assign(pt_out,
						points_to_forloop(instruction_forloop(i),
										  pt_in, store));
	break;
  }
  default:
	pips_internal_error("Unexpected instruction tag %d\n", instruction_tag(i));
  }

  return pt_out;
}
/* Entry point: intialize the entry poitns-to set; intraprocedurally,
   it's an empty set. */
void points_to_statement(statement current)
{
  set pt_in = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  set pt_out = set_generic_make(set_private, points_to_equal_p, points_to_rank);
  pt_out = set_assign(pt_out,
					  recursive_points_to_statement(current, pt_in, true));
}

bool points_to_analysis(char * module_name)
{
  entity module;
  statement module_stat;
  list pt_list = NIL;

  init_pt_to_list();

  set_current_module_entity(module_name_to_entity(module_name));
  set_methods_for_proper_simple_effects();
  module = get_current_module_entity();

  (*effects_computation_init_func)(module_name);

  debug_on("POINTS_TO_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);
  set_current_module_statement( (statement)
								db_get_memory_resource(DBR_CODE,
													   module_name, TRUE) );
  module_stat = get_current_module_statement();
  init_proper_rw_effects();

  debug_on("POINTERS_DEBUG_LEVEL");
  pips_debug(1, "begin\n");
  points_to_statement(module_stat);
  statement_points_to_consistent_p(get_pt_to_list());
  DB_PUT_MEMORY_RESOURCE
	(DBR_POINTS_TO_LIST, module_name, get_pt_to_list());

  reset_pt_to_list();
  reset_current_module_entity();
  reset_current_module_statement();
  reset_proper_rw_effects();
  generic_effects_reset_all_methods();

  bool good_result_p = TRUE;
  return (good_result_p);

}
