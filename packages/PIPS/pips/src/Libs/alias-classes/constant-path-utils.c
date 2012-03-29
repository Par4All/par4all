
/* This files containes all th eoperators defining  constant paths :

CP = Mdodule * Name * Type *Vref.
To calculate the lattice PC operators we define these operators first
on each  of its dimensions.
Each dimension represents a lattice with a bottom and a top.


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
#include "genC.h"


/* Already exists in points_to_general_algorithm.c, to be removed later...
   iterate over the lhs_set, if it contains more than an element
   approximations are set to MAY, otherwise it's set to EXACT
   Set the sink to nowhere .*/
set points_to_nowhere_typed(list lhs_list, set input)
{
  set kill= set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set gen = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set res = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set input_kill_diff = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  approximation a = make_approximation_exact();

  SET_FOREACH(points_to, p, input) {
      FOREACH(cell, c, lhs_list) {
	if(points_to_compare_cell(points_to_source(p),c))
	    set_add_element(kill, kill, p);
	}
    }

  set_difference(input_kill_diff, input, kill);

  /* if the lhs_set or the rhs set contains more than an element, we
     set the approximation to MAY. */
  if((int)gen_length(lhs_list) > 1)// || set_size(rhs_set)>1)
    a = make_approximation_may();

  /* Computing the gen set*/
  FOREACH(cell, c, lhs_list) {
      /* we test if lhs is an array, so we change a[i] into a[*]
       is cell_reference_to_type() the right function to call ?*/
      /* if(cell_reference_to_type(c) == array) */
      /*c = array_to_store_independent(c); */

      /* create a new points to with as source the current
	 element of lhs_set and sink the null value .*/
    bool to_be_freed = false;
    type c_type = cell_to_type(c, &to_be_freed);
      entity e = entity_all_xxx_locations_typed(NOWHERE_LOCATION,
					      c_type);
      reference r = make_reference(e, NIL);
      cell sink = make_cell_reference(r);
      points_to pt_to = make_points_to(c, sink, a, make_descriptor_none());
      set_add_element(gen, gen, (void*)pt_to);
      if(to_be_freed) free_type(c_type);
    }
  /* gen + input_kill_diff*/
  set_union(res, gen, input_kill_diff);

  return res;
}

/* arg1:  list of cells
   arg2:  set of points-to
   Create a points-to set with elements of lhs_list as
   source and NOWHERE as sink.
   Iterate over input and kill all points-to relations
   where sinks are elements of lhs_list.
*/
set points_to_nowhere(list lhs_list, set input)
{
  set kill = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set gen = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set res = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set input_kill_diff = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  approximation a = make_approximation_exact();

  SET_FOREACH(points_to, p, input) {
      FOREACH(cell, c, lhs_list) {
	  if(points_to_source(p) == c)
	    set_add_element(kill, kill, p);
	}
    }
  set_difference(input_kill_diff, input, kill);
  /* if the lhs_set or the rhs set contains more than an element, we
     set the approximation to MAY. */
  if((int)gen_length(lhs_list) > 1)// || set_size(rhs_set)>1)
    a = make_approximation_may();

  /* Computing the gen set */
  FOREACH(cell, c, lhs_list) {
    entity e = entity_all_xxx_locations(NOWHERE_LOCATION);
    reference r = make_reference(e, NIL);
    cell sink = make_cell_reference(r);
    points_to pt_to = make_points_to(c, sink, copy_approximation(a), make_descriptor_none());
    set_add_element(gen, gen, (void*)pt_to);
  }
  free_approximation(a);
  set_union(res, gen, input_kill_diff);

  return res;
}


/* Already exists in points_to_general_algorithm.c, to be removed later...
   iterate over the lhs_set, if it contains more than an element
   approximations are set to MAY, otherwise it's set to EXACT
   Set the sink to anywhere .*/
set points_to_anywhere_typed(list lhs_list, set input)
{
  set kill= set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set gen = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set res = set_generic_make(set_private, points_to_equal_p,
				    points_to_rank);
  set input_kill_diff = set_generic_make(set_private, points_to_equal_p,
			   points_to_rank);
  approximation a = make_approximation_exact();

  SET_FOREACH(points_to, p, input) {
      FOREACH(cell, c, lhs_list) {
	if(points_to_compare_cell(points_to_source(p), c))
	    set_add_element(kill, kill, p);
	}
    }

  /* input - kill */
  set_difference(input_kill_diff, input, kill);

  /* if the lhs_set or the rhs set contains more than an element, we
     set the approximation to MAY. */
  if((int)gen_length(lhs_list) > 1)// || set_size(rhs_set)>1)
    a = make_approximation_may();

  /* Computing the gen set*/
  FOREACH(cell, c, lhs_list) {
      /* create a new points to with as source the current
	 element of lhs_set and sink the null value .*/
    reference cr = cell_any_reference(c);
    entity er = reference_variable(cr);
    type t = entity_type(er);
    entity e = entity_all_xxx_locations_typed(ANYWHERE_LOCATION,t);
    reference r = make_reference(e, NIL);
    cell sink = make_cell_reference(r);
    points_to pt_to = make_points_to(c, sink, a, make_descriptor_none());
    set_add_element(gen, gen, (void*)pt_to);
    }

  set_union(res, gen, input_kill_diff);

  return res;
}


set points_to_anywhere(list lhs_list, set input)
{
  /* lhs_path matches the kill set.*/
  set kill= set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  set gen = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  set res = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  set input_kill_diff = set_generic_make(set_private, points_to_equal_p,
					 points_to_rank);
  approximation a = make_approximation_exact();

  SET_FOREACH ( points_to, p, input ) {
    FOREACH ( cell, c, lhs_list ) {
      if ( points_to_compare_cell(points_to_source(p), c) )
	set_add_element(kill, kill, p);
    }
  }

  /* input - kill */
  set_difference(input_kill_diff, input, kill);

  /* if the lhs_set or the rhs set contains more than an element, we
     set the approximation to MAY. */
  if ( (int)gen_length(lhs_list) > 1 )// || set_size(rhs_set)>1)
    a = make_approximation_may();

  /* Computing the gen set*/
  FOREACH ( cell, c, lhs_list ) {
    /* create a new points to with as source the current
       element of lhs_set and sink the null value .*/
    entity e = entity_all_xxx_locations(ANYWHERE_LOCATION);
    reference r = make_reference(e, NIL);
    cell sink = make_cell_reference(r);
    points_to pt_to = make_points_to(c, sink, a, make_descriptor_none());
    set_add_element(gen, gen, (void*)pt_to);
  }
  /* gen + input_kill_diff*/
  set_union(res, gen, input_kill_diff);

  return res;
}


/* input : expression e and a set of points_to
   output : a set of  constant paths
   translates an expression into a set of constant paths
   by first changing operators like . and -> into
   p[0](get_memory_path()). Then evaluate this path by using
   points_toèrelations already computed (eval_cell_with_points_to()).
   Finaly construct a set of constant paths according to the list
   returned by eval_cell_with_points_to().
*/
list array_to_constant_paths(expression e, set in __attribute__ ((__unused__)))
{
  list l = NIL;
  bool changed = false;
  //effect ef = effect_undefined;
  //reference r = reference_undefined;
  cell c = cell_undefined;
  cell c_new = cell_undefined;
  //int i;
  reference er = expression_reference(e);
  set_methods_for_proper_simple_effects();
  if(array_reference_p(er)) {
    c = make_cell_reference(er);
    c_new = simple_cell_to_store_independent_cell(c, &changed);
  }
  else {
    c_new = make_cell_reference(er);
  }
  generic_effects_reset_all_methods();
  l = CONS(CELL, c_new, NIL);

  return l;
}




/*
  Change dereferenced pointers and filed access into a constant paths
  When no constant path is found the expression is uninitialized pointer
*/
/* list expression_to_constant_paths(statement s, expression e, set in) */
/* { */
/*   set cur = set_generic_make(set_private, points_to_equal_p, */
/* 					 points_to_rank); */
/*   cell c = cell_undefined; */
/*   entity nowhere = entity_undefined; */
/*   list l  = NIL, l_in = NIL, l_eval = NIL, l_cell = NIL; */
/*   bool exact_p = false, *nowhere_p = false; */
/*   bool eval_p = true; */
/*   c = get_memory_path(e, &eval_p); */
/*   reference cr = cell_any_reference(c); */
/*   bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES"); */
/*   /\* Take into account global variables which are initialized in demand*\/ */
/*   /\* entity ce = reference_variable(cr); *\/ */
/*   /\* cell cc = make_cell_reference(cr); *\/ */
/*   l_cell = CONS(CELL, c, NIL); */
/*   in = set_assign(cur, points_to_init_global(s, l_cell, in)); */
/*   if( eval_p ) { */
/*     l_cell = CONS(CELL, c, NIL); */
/*     set_methods_for_proper_simple_effects(); */
/*     l_in = set_to_sorted_list(in, */
/* 			      (int(*)(const void*, const void*)) */
/* 			      points_to_compare_location); */
/*     l_eval = eval_cell_with_points_to(c, l_in, &exact_p); */
/*     generic_effects_reset_all_methods(); */
/*     /\* in = points_to_init_global(s, l_cell, in); *\/ */
/*     l_cell = gen_nconc(l,possible_constant_paths(l_eval,c,nowhere_p)); */

/*     in =  set_assign(in, points_to_init_global(s, l_cell, in)); */
/*   } */
/*   else { */
/*     l_cell = CONS(CELL, c, NIL); */
/*     in =  set_assign(cur, points_to_init_global(s, l_cell, in)); */
/*   } */
  

/*   /\* if c is an anywhere or a reference we don't evaluate it*\/ */
/*   if (!eval_p) */
/*     l = CONS(CELL, c, NIL); */
/*   else if (set_empty_p(in)|| entity_null_locations_p(reference_variable(cr)) */
/* 			      || entity_nowhere_locations_p(reference_variable(cr))){ */
/*     if (get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING")){ */
/*       pips_user_error("Uninitialized pointer\n"); */
/*     } */
/*     else { */
/*       pips_user_warning("Uninitialized pointer"); */
/*       bool to_be_freed = false; */
/*       type c_type = cell_to_type(c, &to_be_freed); */
/*       if ( type_sensitive_p )  */
/* 	nowhere = entity_all_xxx_locations_typed( */
/* 						      NOWHERE_LOCATION, */
/* 						      c_type); */
/*       else */
/* 	nowhere = entity_all_xxx_locations( NOWHERE_LOCATION ); */

/*       reference r = make_reference(nowhere,NIL); */
/*       c = make_cell_reference(r); */
/*       l = CONS(CELL, c, NIL); */
/*       if (to_be_freed) free_type(c_type); */
/*     } */
/*   } */
/*   else { */
/*     set_methods_for_proper_simple_effects(); */
/*     l_in = set_to_sorted_list(in, */
/* 			      (int(*)(const void*, const void*)) */
/* 			      points_to_compare_location); */
/*     l_eval = eval_cell_with_points_to(c, l_in, &exact_p); */
/*     generic_effects_reset_all_methods(); */
/*     l = gen_nconc(l,possible_constant_paths(l_eval,c,nowhere_p)); */
/*   } */

/*   return l; */
/* } */

/*
 * Change dereferenced pointers and field access into a constant paths.
 *
 * FI: For instance, *p becomes p[0], s.a becomes s[a] and p->a
 * becomes p[a] (to be checked with AM).
 *
 * When no constant path is found, the expression is considered
 * equivalent to an uninitialized pointer. 
 *
 * FI: to be reviewed with AM, obvious memory leaks
 * FI: what is the meaning of eval_p? set by get_memory_path()?
 */
list expression_to_constant_paths(statement s, expression e, set in)
{
  set cur = set_generic_make(set_private, points_to_equal_p,
					 points_to_rank);
  cell c = cell_undefined;
  entity nowhere = entity_undefined;
  list l  = NIL, l_in = NIL, l_eval = NIL, l_cell = NIL;
  bool exact_p = false, *nowhere_p = false, changed = true;
  bool eval_p = true;

  // FI: it looks very complicated when e is a simple reference, but
  // it may be a general approach

  c = get_memory_path(e, &eval_p);

  c = simple_cell_to_store_independent_cell(c, &changed);

  reference cr = cell_any_reference(c);
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");

  /* Take into account global variables which are initialized on demand*/
  /* entity ce = reference_variable(cr); */
  /* cell cc = make_cell_reference(cr); */
  l_cell = CONS(CELL, c, NIL);
  set g_set = points_to_init_global(s, l_cell, in);
  in = set_assign(cur, g_set);
  // FI: free_set(g_set);

  if( eval_p ) {
    // FI: memory leak for l_cell
    l_cell = CONS(CELL, c, NIL);
    set_methods_for_proper_simple_effects();
    l_in = set_to_sorted_list(in,
			      (int(*)(const void*, const void*))
			      points_to_compare_location);
    l_eval = eval_cell_with_points_to(c, l_in, &exact_p);
    generic_effects_reset_all_methods();
    /* in = points_to_init_global(s, l_cell, in); */
    // FI: memory leak for l_cell
    l_cell = gen_nconc(l, possible_constant_paths(l_eval,c,nowhere_p));

    in =  set_assign(in, points_to_init_global(s, l_cell, in));
    }
  else {
    // FI: memory leak for l_cell
    l_cell = CONS(CELL, c, NIL);
    in =  set_assign(cur, points_to_init_global(s, l_cell, in));
  }
  

  /* if c is an anywhere or a reference we don't evaluate it*/
  if (eval_p
      && !(set_empty_p(in)
	   || entity_null_locations_p(reference_variable(cr))
	   || entity_nowhere_locations_p(reference_variable(cr)))
      && 
      !get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING")) 
    {
      set_methods_for_proper_simple_effects();
      l_in = set_to_sorted_list(in,
				(int(*)(const void*, const void*))
				points_to_compare_location);
      l_eval = eval_cell_with_points_to(c, l_in, &exact_p);
      generic_effects_reset_all_methods();
      l = gen_nconc(l,possible_constant_paths(l_eval,c,nowhere_p));
    }
  else if (get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING") && eval_p) {
      pips_user_warning("Uninitialized pointer");
      bool to_be_freed = false;
      type c_type = cell_to_type(c, &to_be_freed);
	if ( type_sensitive_p )
	nowhere = entity_all_xxx_locations_typed(
						      NOWHERE_LOCATION,
						      c_type);
      else
	nowhere = entity_all_xxx_locations( NOWHERE_LOCATION );

      reference r = make_reference(nowhere,NIL);
      c = make_cell_reference(r);
      l = CONS(CELL, c, NIL);
      if (to_be_freed) free_type(c_type);
    }
  else if(eval_p)
    pips_user_error("Uninitialized pointer dereferencing\n");
  else
    l = CONS(CELL, c, NIL);

  return l;
}




/* This function returns a cell
- we initialize the effect engine
- we call generic_proper_effects_of complex_address()
  which transform my_str->field into my_str[.field]
- we reset the effect engine
- we create a cell from the main effect returned by
  generic_proper_effects_of_complex_address()
- we can't evaluate user call neither translate them into an
  constant access path, so eval_p is set to false.
*/
cell get_memory_path(expression e, bool * eval_p)
{
  effect ef = effect_undefined;
  reference  r = reference_undefined;
  entity anywhere = entity_undefined;
  cell c = cell_undefined;
  bool exact_p = false;
  /* we assume that we don't need to cover all expression type's, we
     are only intereseted in user function call, reference or pointer
     dereferencing */

  if (expression_call_p(e)) {
    if (user_function_call_p(e)) {
      (*eval_p) = false;
      call cl = expression_call(e);
      bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
      if(type_sensitive_p)
	anywhere =entity_all_xxx_locations_typed(ANYWHERE_LOCATION,
						      call_to_type(cl));
      else
	anywhere =  entity_all_xxx_locations(ANYWHERE_LOCATION);
      reference r = make_reference(anywhere,NIL);
      /* FI: Should not it be a preference? */
      c = make_cell_reference(r);
    }
    else {
      entity op = call_function(expression_call(e));
      if(ENTITY_FIELD_P(op) ||
	 ENTITY_POINT_TO_P(op) ||
	 ENTITY_DEREFERENCING_P(op)) {

	/*init the effect's engine*/
	set_methods_for_proper_simple_effects();
	list l_ef = NIL;
	list l1 = generic_proper_effects_of_complex_address_expression(
								       e,
								       &l_ef,
								       true);
	ef = EFFECT(CAR(l_ef));
	/* In fact, there should be a FOREACH to scan all elements of l_ef */
	gen_free_list(l_ef); /* free the spine */
	r = effect_any_reference(ef);
	c = make_cell_reference(r);
	(*eval_p) = effect_reference_dereferencing_p(r, &exact_p);
	effects_free(l1);
	generic_effects_reset_all_methods();

      }
    }
  }
  else if (syntax_subscript_p(expression_syntax(e))) {
    set_methods_for_proper_simple_effects();
    list l = NIL;
    list l_ef = NIL;
    list l2 = generic_proper_effects_of_complex_memory_access_expression(e,
									&l_ef,
									&l,
									true);
    ef = EFFECT(CAR(l_ef));
    /* In fact, there should be a FOREACH to scan all elements of l_ef */
    gen_free_list(l_ef); /* free the spine */
    r = effect_any_reference(ef);
    c = make_cell_reference(r);
    effects_free(l2);
    generic_effects_reset_all_methods();
    (*eval_p) = false;
  }
  else if (array_argument_p(e)) {
    (*eval_p) = false;
    set_methods_for_proper_simple_effects();
    list l_ef = NIL;
    list l1 = generic_proper_effects_of_complex_address_expression(e, &l_ef,
								   true);

    ef = EFFECT(CAR(l_ef));
    /* In fact, there should be a FOREACH to scan all elements of l_ef */
    gen_free_list(l_ef); /* free the spine */
    r = effect_any_reference(ef);
    c = make_cell_reference(r);
    effects_free(l1);
    generic_effects_reset_all_methods();
  }
  else if (expression_reference_p(e)) {
    r = expression_reference(e);
    if (!ENDP(reference_indices(r)) && !array_argument_p(e)) {
      list l_ef = NIL;
      (*eval_p) = true;
      set_methods_for_proper_simple_effects();
      list l1 = generic_proper_effects_of_complex_address_expression(e, &l_ef,
								   true);
      ef = EFFECT(CAR(l_ef)); 
      /* In fact, there should be a FOREACH to scan all elements of l_ef */
      gen_free_list(l_ef); /* free the spine */
      r = effect_any_reference(ef);
      c = make_cell_reference(r);
      effects_free(l1);
      generic_effects_reset_all_methods();
    }
    else {
      (*eval_p) = false;
      c = make_cell_reference(r);
    }
  }

  return c;
}

/* This function iterates over the list already computed by
   eval_cell_with_points_to() and  depending on the length and
   elements  contained in the list create :
   - a set containing the constant paths if the list contains elements
   and there are not anywhere or nowhere paths.
   - a set containing anywhere paths of the list is empty or contains
   a nowhere path.
*/
list possible_constant_paths(
			     list l,
			     cell lhs,
			     bool *nowhere_p __attribute__ ((__unused__))
			     )
{
  list paths = NIL;
  entity nowhere = entity_undefined;
  entity anywhere = entity_undefined;
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
  if (ENDP(l)) {
    if (get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING"))
      pips_user_error("Uninitialized pointer \n");
    else {
      pips_user_warning("Uninitialized pointer \n");
      bool to_be_freed = false;
      type lhs_type = cell_to_type(lhs, &to_be_freed);
      if ( type_sensitive_p )
       nowhere =entity_all_xxx_locations_typed(
						     NOWHERE_LOCATION,
						     lhs_type);
      else
	nowhere = entity_all_xxx_locations(NOWHERE_LOCATION);

      reference r = make_reference(nowhere,NIL);
      cell c = make_cell_reference(r);
      paths = CONS(CELL, c, NIL);
      if (to_be_freed) free_type(lhs_type);
    }
  }
  else {
    bool to_be_freed = false;
    type lhs_type = cell_to_type(lhs, &to_be_freed);
    FOREACH(cell, c, l){
      reference r = cell_any_reference(c);
      entity e = reference_variable(r);
      if(entity_nowhere_locations_p(e))
	pips_user_warning("Uninitialized pointer \n");
      if(entity_all_locations_p(e)){
	if( type_sensitive_p )
	 anywhere =entity_all_xxx_locations_typed(
							ANYWHERE_LOCATION,
							lhs_type);
	else
	  anywhere = entity_all_xxx_locations(ANYWHERE_LOCATION);

	r = make_reference(anywhere,NIL);
	c = make_cell_reference(r);
      }
      paths = gen_nconc(paths, CONS(CELL, c, NIL));
      if (to_be_freed) free_type(lhs_type);
    }
  }

  return paths;
}

/* we define operator max fot the lattice Module which has any_module
   as top and a bottom which is not yet clearly defined (maybe
   no_module)
   max_module : Module * Module -> Module
   Side effects on m1 if we have an anywhere location to return.
*/
cell max_module(cell m1, cell m2)
{
  reference r1 = cell_any_reference(m1);
  reference r2 = cell_any_reference(m2);
  entity e1 = reference_variable(r1);
  entity e2 = reference_variable(r2);

  if (same_string_p(entity_name(e1),entity_name(e2)))
    return m1;
  else {
    e1 = entity_all_locations();
    r1 = make_reference(e1,NIL);
    m1 = make_cell_reference(r1);
  }
  return m1;
}
/* operator kill for the dimension Module:
   - modules should be different from any_module otherwise we return
   false
   - when modules are different from any_module we test the equality
   of their names
   opkill_may_module : Module * Module -> Bool
   opkill_must_module : Module * Module -> Bool

*/
/* test if a module is the any_module location, to be moved to
   anywhere_abstract_locations.c later...*/
bool entity_any_module_p(entity e)
{
  bool any_module_p;
  any_module_p =  same_string_p(entity_module_name(e), ANY_MODULE_NAME);

  return any_module_p;
}

bool opkill_may_module(cell m1, cell m2)
{
  reference r1 = cell_any_reference(m1);
  reference r2 = cell_any_reference(m2);
  entity e1 = reference_variable(r1);
  entity e2 = reference_variable(r2);
  bool kill_may_p;

  /* if the lhs or the rhs is a nowhere location or a null/0 value we
     generate a pips_user_warning */
  if (entity_any_module_p(e1) && entity_any_module_p(e2))
    kill_may_p = true;
  else
    kill_may_p = same_string_p(entity_name(e1),entity_name(e2));

  return kill_may_p;
}

bool opkill_must_module(cell m1, cell m2)
{
  reference r1 = cell_any_reference(m1);
  reference r2 = cell_any_reference(m2);
  entity e1 = reference_variable(r1);
  entity e2 = reference_variable(r2);
  bool kill_must_p;

  if (entity_any_module_p(e1) && entity_any_module_p(e2))
    kill_must_p = false;
  else
    kill_must_p = same_string_p(entity_name(e1),entity_name(e2));

  return kill_must_p;
}


/* Opertaor gen for modules:
   m1 is the sink, m2 the source (m2 points to m1)
   opkill_gen_module : Module * Module -> Module

*/
cell op_gen_module(cell m1,  __attribute__ ((__unused__))cell m2)
{
  /* we return the module m1 whatever is its type (#any_module,
     TOP-LEVEL, any_module) */
  return m1;

}

/* We define operators for the lattice Name which can be a:
   -variable of a the program
   -malloc
   -NULL /0
   -STATIC/STACK/DYNAMIC/HEAP/FORMAL
   -nowhere/anywhere
*/

/* We define the max between 2 names according to the order
   established by the lattice Name, already done by
   entity_locations_max() but we have to add a new abstract location
   for Formal area */

/* opkill for the lattice Name tests if 2 names are :
   -variables of the program then we return the result of the
   comparison their names.
   -abstract locations so we return FALSE.
   Name * Name-> Bool
*/

bool opkill_may_name(cell n1, cell n2)
{
 reference r1 = cell_any_reference(n1);
 reference r2 = cell_any_reference(n2);
 entity e1 = reference_variable(r1);
 entity e2 = reference_variable(r2);
 bool kill_may_p;

  if (entity_nowhere_locations_p(e1)||entity_nowhere_locations_p(e2)||
     entity_null_locations_p(e1) || entity_null_locations_p(e2))
    pips_user_error("NULL or ANYWHERE locations can't appear as an lvalue\n");

 if (entity_abstract_location_p(e2)) {
   if (entity_all_locations_p(e2))
     kill_may_p = true;
   if (entity_abstract_location_p(e1)) {
     if (entity_all_locations_p(e1))
       kill_may_p = true;
     else
       kill_may_p = same_string_p(entity_name(e1), entity_name(e2));
   }
   else {
       /* if(entity_malloc_p(e2)) kill_may_p = false// function
	  entity_malloc_p() have to be defined and different from
	  entity_heap_location_p() */
       e1 = variable_to_abstract_location(e1);
       r1 = make_reference(e1,NIL);
       n1 = make_cell_reference(r1);
       kill_may_p = opkill_may_name(n1, n2);
   }
 }
  else  if ( entity_abstract_location_p(e1) ) {
     if (entity_all_locations_p(e1))
       kill_may_p = true;
     else {
       /* if(entity_malloc_p(e1)) kill_may_p = true// function
	  entity_malloc_p() have to be defined and different from
	  entity_heap_location_p() */
       e2 = variable_to_abstract_location(e2);
       r2 = make_reference(e2,NIL);
       n2 = make_cell_reference(r2);
       kill_may_p = opkill_may_name(n1, n2);
     }
   }
   else
     kill_may_p = same_string_p(entity_name(e1), entity_name(e2));

 return kill_may_p ;
}


bool opkill_must_name(cell n1, cell n2)
{
 reference r1 = cell_any_reference(n1);
 reference r2 = cell_any_reference(n2);
 entity e1 = reference_variable(r1);
 entity e2 = reference_variable(r2);
 bool kill_must_p;
 
 if (entity_nowhere_locations_p(e1)||entity_nowhere_locations_p(e2)||
     entity_null_locations_p(e1) || entity_null_locations_p(e2))
    pips_user_error("NULL or ANYWHERE locations can't appear as an lvalue\n");

 if (entity_abstract_location_p(e2)) {
   if (entity_all_locations_p(e2))
     kill_must_p = false;
   if (entity_abstract_location_p(e1)) {
     if(entity_all_locations_p(e1))
       kill_must_p = false;
     else
	 kill_must_p = same_string_p(entity_name(e1), entity_name(e2));
   }
   else {
     /* if(entity_malloc_p(e2)) kill_may_p = false// function
	entity_malloc_p() have to be defined and different from
	entity_heap_location_p() */
     e1 = variable_to_abstract_location(e1);
     r1 = make_reference(e1,NIL);
     n1 = make_cell_reference(r1);
     kill_must_p = opkill_may_name(n1, n2);
   }
 }
  else if ( entity_abstract_location_p(e1) ) {
     if (entity_all_locations_p(e1))
       kill_must_p = false;
     else {
       /* if(entity_malloc_p(e1)) kill_may_p = true// function
	  entity_malloc_p() have to be defined and different from
	  entity_heap_location_p() */
       e2 = variable_to_abstract_location(e2);
       r2 = make_reference(e2,NIL);
       n2 = make_cell_reference(r2);
       kill_must_p = opkill_may_name(n1, n2);
     }
   }
   else
     kill_must_p = same_string_p(entity_name(e1), entity_name(e2));

 return kill_must_p ;
}

type max_type(type t1, type t2)
{
  if (!type_unknown_p(t1) && ! type_unknown_p(t2) && type_equal_p(t1,t2))
    return t1;
  else {
    type t = MakeTypeUnknown();
    return t;
  }
}


bool opkill_may_type(type t1, type t2)
{
  if (!type_unknown_p(t1) && ! type_unknown_p(t2))
    return type_equal_p(t1,t2);
  else
    return false;

}

/* opkill_must_type is the same as op_kill_may_type...*/
bool opkill_must_type(type t1, type t2)
{
  return opkill_may_type(t1,t2);
}

type opgen_may_type(type t1, type t2)
{
  if (!type_unknown_p(t1)&& ! type_unknown_p(t2))
    return t1;
  else {
    type t = MakeTypeUnknown();
    return t;
  }
}

/* the same as opgen_may_type*/
type opgen_must_type(type t1, type t2)
{
  return opgen_may_type(t1,t2);
}


bool opkill_may_reference(cell c1, cell c2)
{
  reference r1 = reference_undefined;
  reference r2 = reference_undefined;
  bool kill_may_p = true;

  if (cell_reference_p(c1))
    r1 = cell_reference(c1);
  else
    r1 = preference_reference(cell_preference(c1));
  if (cell_reference_p(c2))
    r2 = cell_reference(c2);
  else
    r2 = preference_reference(cell_preference(c2));

  if (store_independent_reference_p(r1) || store_independent_reference_p(r2))
    return kill_may_p;
  else {
    kill_may_p = reference_equal_p(r1,r2);
    return kill_may_p;
      }
}


bool opkill_must_reference(cell c1, cell c2)
{
  reference r1 = cell_any_reference(c1);
  reference r2 = cell_any_reference(c2);
  bool kill_must_p = false;

  if (store_independent_reference_p(r1) || store_independent_reference_p(r2))
    return kill_must_p;
  else {
    kill_must_p = reference_equal_p(r1,r2);
    return kill_must_p;
      }
}


bool opkill_may_vreference(cell c1, cell c2)
{
  int i = 0;
  reference r1 = cell_any_reference(c1);
  reference r2 = cell_any_reference(c2);
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);
  list sl1 = NIL, sl2 = NIL;
  // FI: memory leak? generation of a new string?
  extern const char* entity_minimal_user_name(entity);

  i = strcmp(entity_minimal_user_name(v1), entity_minimal_user_name(v2));
  if ( i==0 ) {
    sl1 = reference_indices(r1);
    sl2 = reference_indices(r2);
    for (;i==0 && !ENDP(sl1) && ! ENDP(sl2) ; POP(sl1), POP(sl2))
      {
	expression se1 = EXPRESSION(CAR(sl1));
	expression se2 = EXPRESSION(CAR(sl2));
	if (unbounded_expression_p(se2) && expression_constant_p(se1))
	  i = 0;
	else if ( expression_constant_p(se1) && expression_constant_p(se2) ) {
	  int i1 = expression_to_int(se1);
	  int i2 = expression_to_int(se2);
	  i = i2>i1? 1 : (i2<i1? -1 : 0);

	  if ( i==0 ) {
	    string s1 = words_to_string(words_expression(se1, NIL));
	    string s2 = words_to_string(words_expression(se2, NIL));
	    i = strcmp(s1, s2);
	    }
	  }

      }
    }

  return (i==0? true: false);
}

bool opkill_must_vreference(cell c1, cell c2)
{
  int i = 0;
  reference r1 = cell_any_reference(c1);
  reference r2 = cell_any_reference(c2);
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);
  list sl1 = NIL, sl2 = NIL;
  // FI: memory leak? generation of a new string?
  extern const char* entity_minimal_user_name(entity);

  i = strcmp(entity_minimal_user_name(v1), entity_minimal_user_name(v2));
  if (i==0) {
    sl1 = reference_indices(r1);
    sl2 = reference_indices(r2);
    for (;i==0 && !ENDP(sl1) && ! ENDP(sl2) ; POP(sl1), POP(sl2)){
	expression se1 = EXPRESSION(CAR(sl1));
	expression se2 = EXPRESSION(CAR(sl2));
	if(expression_constant_p(se2) && unbounded_expression_p(se1)){
	  i = 0;
	}
	else if (expression_constant_p(se1) && expression_constant_p(se2)){
	  int i1 = expression_to_int(se1);
	  int i2 = expression_to_int(se2);
	  i = i2>i1? 1 : (i2<i1? -1 : 0);
	  if (i==0){
	    string s1 = words_to_string(words_expression(se1, NIL));
	    string s2 = words_to_string(words_expression(se2, NIL));
	    i = strcmp(s1, s2);
	}
      }
	else {
	  string s1 = words_to_string(words_expression(se1, NIL));
	  string s2 = words_to_string(words_expression(se2, NIL));
	  i = strcmp(s1, s2);
    }
    }
  }

  return (i==0? true: false);
}




bool opkill_may_constant_path(cell c1, cell c2)
{
  bool kill_may_p;
  reference r1 = cell_any_reference(c1);
  reference r2 = cell_any_reference(c2);
  type t1 = type_undefined;
  type t2 = type_undefined;
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);
  bool type_equal_p = true;

  if (! type_area_p(entity_type(v1)) && !type_area_p(entity_type(v2))){
    t1 = simple_effect_reference_type(r1);
    t2 = simple_effect_reference_type(r2);
    type_equal_p = opkill_may_type(t1,t2);
  }
  kill_may_p =opkill_may_module(c1,c2) && opkill_may_name(c1,c2) &&
    type_equal_p && opkill_may_vreference(c1,c2);

  return kill_may_p;
}

bool opkill_must_constant_path(cell c1, cell c2)
{
  bool kill_must_p;
  reference r1 = cell_any_reference(c1);
  reference r2 = cell_any_reference(c2);
  type t1 = type_undefined;
  type t2 = type_undefined;
  //entity v1 = reference_variable(r1);
  //entity v2 = reference_variable(r2);
  bool equal_p = true;
  t1 = cell_reference_to_type(r1,&equal_p);
  t2 = cell_reference_to_type(r2, &equal_p);
  equal_p = type_equal_p(t1,t2);
  /* if (! type_area_p(entity_type(v1)) && !type_area_p(entity_type(v2))){ */
  /*   if (entity_abstract_location_p(v1)) */
  /*     t1 = entity_type(v1); */
  /*   else */
  /*     t1 = simple_effect_reference_type(r1); */
  /*   if (entity_abstract_location_p(v2)) */
  /*     t2 = entity_type(v2); */
  /*   else */
  /*     t2 = simple_effect_reference_type(r2); */
  /*   type_equal_p = opkill_must_type(t1,t2); */
  /* } */
  kill_must_p =opkill_must_module(c1,c2) && opkill_must_name(c1,c2) &&
    equal_p && opkill_must_vreference(c1,c2);

   return kill_must_p;
}


set kill_may_set(list L, set in_may)
{
  set kill_may = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  FOREACH(cell, l, L){
     SET_FOREACH(points_to, pt, in_may){
       if (opkill_may_constant_path(points_to_source(pt),l)) {
	 points_to npt = make_points_to(points_to_source(pt), points_to_sink(pt), 
					make_approximation_may(), make_descriptor_none());
	 set_add_element(kill_may, kill_may,(void*)npt);
     }
  }
  }
  return kill_may;
}


set kill_must_set(list L, set in_must)
{
  set kill_must = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);

  FOREACH(cell, l, L){
     SET_FOREACH(points_to, s, in_must){
       if(opkill_must_constant_path(points_to_source(s),l))
	 set_add_element(kill_must, kill_must,(void*)s);
     }
  }
  return kill_must;
}

/* returns a set which contains all the MAY points to */
set points_to_may_filter(set in)
{
  set in_may = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);

  SET_FOREACH(points_to, pt, in){
    if (approximation_may_p(points_to_approximation(pt)))
      set_add_element(in_may, in_may, (void*)pt);
  }
  return in_may;
}

/* returns a set which contains all the EXACT points to */
set points_to_must_filter(set in)
{
  set in_must = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  SET_FOREACH(points_to, pt, in){
    if (approximation_exact_p(points_to_approximation(pt)))
      set_add_element(in_must, in_must, (void*)pt);
  }
  return in_must;
}

/* shoud be moved to expression.c*/
bool address_of_expression_p(expression e)
{
  bool address_of_p = false;
  syntax s = expression_syntax(e);
  if (syntax_call_p(s)) {
    if (entity_an_operator_p(call_function(syntax_call(s)), ADDRESS_OF))
       address_of_p = true;
  }
    return address_of_p;
}

bool subscript_expression_p(expression e)
{
  return syntax_subscript_p(expression_syntax(e));
}

/* Should be moved to anywhere_abstract_locations.c */

bool expression_null_locations_p(expression e)
{
  if (expression_reference_p(e)) {
    entity v = expression_variable(e);
    return entity_null_locations_p(v);
  }
  else
    return false;
}

/*
 * create a set of points-to relations of the form:
 *
 * element_L -> & element_R, MAY
 *
 * FI: it would be nice to have the equation or a formula
 * I do not understand why gen_may1 is built from in_may
*/
set gen_may_set(list L, list R, set in_may, bool *address_of_p)
{
  set gen_may1 = set_generic_make(set_private, points_to_equal_p,
				  points_to_rank);
  set gen_may2 = set_generic_make(set_private, points_to_equal_p,
				  points_to_rank);
  int len = (int) gen_length(L);

  if(len > 1) {
    /* If the source is not precisely known */
    FOREACH(cell, l, L){
      SET_FOREACH(points_to, pt, in_may){
	if(points_to_compare_cell(points_to_source(pt),l)){
	  // FI: it would be much easier/efficient to modify the approximation of pt
	  points_to npt = make_points_to(l, points_to_sink(pt),
					 make_approximation_may(),
					 make_descriptor_none());
	  set_add_element(gen_may1, gen_may1, (void*)npt);
	  set_del_element(gen_may1, gen_may1, (void*)pt);
	}
      }
    }
  }

  FOREACH(cell, l, L){
    // FI: memory leak due to call to call to gen_may_constant_paths()
    set gen_l = gen_may_constant_paths(l, R, in_may, address_of_p, len);
    // FI: be careful, the union does not preserve consistency because
    // the same arc may appear with different approximations
    set_union(gen_may2, gen_may2, gen_l);
    // free_set(gen_l);
  }

  set_union(gen_may2, gen_may2, gen_may1);

  return gen_may2;
}


/*
 * create a set of points-to relations of the form:
 * element_L -> & element_R, EXACT
 *
 * FI: address_of_p does not seem to be updated in this function. Why
 * pass a pointer? My analysis is wrong if gen_must_constant_paths() updates it
 */
set gen_must_set(list L, list R, set in_must, bool *address_of_p)
{
  set gen_must1 = set_generic_make(set_private, points_to_equal_p,
				   points_to_rank);
  set gen_must2 = set_generic_make(set_private, points_to_equal_p,
				   points_to_rank);
  int len = (int) gen_length(L);

  /* if len > 1 we must iterate over in_must and change all points-to
     relations having L as lhs into may relations */
  if(len > 1){
    FOREACH(cell, l, L){
      SET_FOREACH(points_to, pt, in_must){
	if(points_to_compare_cell(points_to_source(pt),l)){
	  points_to npt = make_points_to(l, points_to_sink(pt),
					 make_approximation_may(),
					 make_descriptor_none());
	  set_add_element(gen_must1, gen_must1, (void*)npt);
	}
      }
    }
  }

  FOREACH(cell, l, L){
    set must_l = gen_must_constant_paths(l, R, in_must, address_of_p, len);
    set_union(gen_must2, gen_must2, must_l);
    // FI: shouldn't must_l be freed?
  }
  set_union(gen_must2, gen_must2,gen_must1);

  return gen_must2;
}

set gen_may_constant_paths(cell l,
			   list R,
			   set in_may,
			   bool* address_of_p,
			   int Lc)
{
  set gen_may_cps = set_generic_make(set_private, points_to_equal_p,
				     points_to_rank);
  points_to pt = points_to_undefined;
  if(!(*address_of_p)){
    /* here we have x = y, then we generate (x,y1,a)|(y,y1,a) as
       points to relation */
    FOREACH(cell, r, R){
      SET_FOREACH(points_to, i, in_may){
	if (/* locations_equal_p */equal_must_vreference(r, points_to_source(i)) /* &&  !entity_abstract_location_p(el) */ ){
	  pt = make_points_to(l, points_to_sink(i), make_approximation_may(), make_descriptor_none());
	}
	if(array_entity_p(reference_variable(cell_any_reference(r)))){
	  reference ref = cell_any_reference(r);
	  bool t_to_be_freed = false;
	  type t = cell_reference_to_type(ref, &t_to_be_freed);
	  if(pointer_type_p(t))
	    pt = make_points_to(l, r, make_approximation_may(), make_descriptor_none());
	  else
	    pt = make_points_to(l, points_to_sink(i), make_approximation_may(), make_descriptor_none());
	  if (t_to_be_freed) free_type(t);
	}
	if(!points_to_undefined_p(pt)) {
	  set_add_element(gen_may_cps, gen_may_cps, (void*) pt);
	}
      }
    }
  }
  else {
    int Rc = (int) gen_length(R);
    FOREACH(cell, r, R){
      approximation a = (Lc+Rc>2) ?
	make_approximation_may() : make_approximation_exact();
      /* Should be replaced by opgen_constant_path(l,r) */
      //reference ref = cell_any_reference(r);
      /* if(reference_unbounded_indices_p(ref)) */
      /*   a = make_approximation_may(); */
      pt = make_points_to(l, r, a, make_descriptor_none());
      set_add_element(gen_may_cps, gen_may_cps, (void*)pt);
    }
  }

  return gen_may_cps;
}

/* This function should be at expression.c. It already exist
   and is called reference_with_unbounded_indices_p() but 
   includes two cases that should be disjoint: constant indices 
   and unbounded ones.
*/
bool reference_unbounded_indices_p(reference r)
{
  list sel = reference_indices(r);
  bool unbounded_p = true;

  MAP(EXPRESSION, se, {
      if(!unbounded_expression_p(se)) {
	unbounded_p = false;
	break;
      }
    }, sel);
  return unbounded_p;
}


/* Build a set of arcs from cell l towards cells in list R if
 * *address_p is true, or towards cells pointed by cells in list R if
 * not.
 *
 * Approximation is must if Lc==1. Lc is the cardinal of L, a set
 * containing l.
 *
 * FI->AM: I do not understand why the cardinal of R is not used too
 * when deciding if the approximation is may or must. I decide to
 * change the semantics of this function although it is used by the
 * initial analysis.
 *
 * FI: since *address_of_p is not modified, I do not understand why a
 * pointer is passed.
 *
 * FI->AM: sharing of a... A new approximation must be generated for
 * each new arc
 */
set gen_must_constant_paths(cell l,
			    list R,
			    set in_must,
			    bool* address_of_p,
			    int Lc)
{
  set gen_must_cps = set_generic_make(set_private, points_to_equal_p,
				      points_to_rank);
  points_to pt = points_to_undefined;
  approximation a = approximation_undefined;
  bool changed = false;
  int Rc = (int) gen_length(R);

  // Rc = 0;
  if(*address_of_p){
    /* if we have x = &y then we generate (x,y,a) as points to relation*/
    FOREACH(cell, r, R){
      /* Should be replaced by opgen_constant_path(l,r) */
      //reference ref = cell_any_reference(r);
      /* if(reference_unbounded_indices_p(ref)) */
      /* 	a = make_approximation_may(); */
      approximation a = (Lc+Rc>2)?
	make_approximation_may(): make_approximation_exact();
      pt = make_points_to(l, r, a, make_descriptor_none());
      set_add_element(gen_must_cps, gen_must_cps, (void*)pt);
    }
  }
  else {
    /* here we have x = y, then we generate (x,y1,a)|(y,y1,a) as
       points to relation */
    FOREACH(cell, r, R){
      SET_FOREACH(points_to, i, in_must) {
	if (/* locations_equal_p */equal_must_vreference(r, points_to_source(i))/*  && !entity_abstract_location_p(el) */){
	  set_methods_for_proper_simple_effects();
	  l = simple_cell_to_store_independent_cell(l, &changed);
	  generic_effects_reset_all_methods();
	  approximation a = (Lc+Rc>2)?
	    make_approximation_may() : make_approximation_exact();
	  pt = make_points_to(l, points_to_sink(i), a, make_descriptor_none());


	  /* if(array_entity_p(reference_variable(cell_any_reference(r)))){ */
	  /*   reference ref = cell_any_reference(r); */
	  /*   expression ex = reference_to_expression(ref); */

	  /*   if(!expression_pointer_p(ex)) */
	  /*     pt = make_points_to(l, r, a, make_descriptor_none()); */
	  /*   else */
	  /*     pt = make_points_to(l, points_to_sink(i), a, make_descriptor_none()); */
	  /* } */

	  /* if(!points_to_undefined_p(pt)) */
	  /*   set_add_element(gen_must_cps, gen_must_cps, (void*) pt); */

	  if(array_entity_p(reference_variable(cell_any_reference(r)))){
	    reference ref = cell_any_reference(r);
	    bool t_to_be_freed = false;
	    type t = cell_reference_to_type(ref, &t_to_be_freed);
	    /* if(reference_unbounded_indices_p(ref)) */
	    /*   a = make_approximation_may(); */
	    if(!pointer_type_p(t))
	      pt = make_points_to(l, r, a, make_descriptor_none());
	    else
	      pt = make_points_to(l, points_to_sink(i), a, make_descriptor_none());

	    if (t_to_be_freed) free_type(t);
	  }
	  if(!points_to_undefined_p(pt))
	    set_add_element(gen_must_cps, gen_must_cps, (void*) pt);
	}
      }
    }
  }
  
  return gen_must_cps;
}

points_to opgen_may_constant_path(cell l __attribute__ ((__unused__)), cell r __attribute__ ((__unused__)))
{
  points_to pt = points_to_undefined;
  return pt;
}

points_to opgen_must_constant_path(cell l __attribute__ ((__unused__)), cell r __attribute__ ((__unused__)))
{
  points_to pt = points_to_undefined;
  return pt;
}

bool opgen_may_module(entity e1, entity e2)
{
  const char* m1 = entity_module_name(e1);
  const char* m2 = entity_module_name(e2);

  if(entity_any_module_p(e1) ||entity_any_module_p(e2))
    return true;
  else 
    return same_string_p(m1, m2);
}

bool opgen_must_module(entity e1, entity e2)
{
  const char* m1 = entity_module_name(e1);
  const char* m2 = entity_module_name(e2);

  if(entity_any_module_p(e1) || entity_any_module_p(e2))
    return false;
  else
    return same_string_p(m1, m2);
}

bool opgen_may_name(entity e1, entity e2)
{
  string n1 = entity_name(e1);
  string n2 = entity_name(e2);

  if(entity_abstract_location_p(e1) ||entity_abstract_location_p(e2))
    return true;
  else
    return same_string_p(n1, n2);
}

bool opgen_must_name(entity e1, entity e2)
{
  string n1 = entity_name(e1);
  string n2 = entity_name(e2);

  if(entity_abstract_location_p(e1) ||entity_abstract_location_p(e2))
    return false;
  else
    return same_string_p(n1, n2);
}

bool opgen_may_vreference(list vr1, list vr2)
{
  bool gen_may_p = true;

  if(ENDP(vr1) || ENDP(vr2))
    return gen_may_p;
  else{
    FOREACH(expression, e, vr1){
      if(extended_integer_constant_expression_p(e)) {
	gen_may_p = false;
	break;
      }
    }
  }

  return gen_may_p;
}

/* Could be replaced by abstract_location_p() but this later don't
   take into account the null location */
bool atomic_constant_path_p(cell cp)
{
  bool atomic_cp_p = true;
  reference r = cell_any_reference(cp);
  entity e = reference_variable(r);

  if(entity_abstract_location_p(e)|| entity_null_locations_p(e))
    atomic_cp_p = false;
  return atomic_cp_p;
}

set opgen_null_location(set L, cell r)
{
set gen_null =  set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  SET_FOREACH(cell, l, L){
    points_to pt = make_points_to(l, r, make_approximation_exact(), make_descriptor_none());
    set_add_element(gen_null, gen_null, (void*) pt);
  }

  return gen_null;
}


/*
  iterate over the points to relation, every tab[i] is changed
  into tab[*] in order to obtain points to relations independent of the store
*/
set points_to_independent_store(set s)
{
  set res =  set_generic_make(set_private, points_to_equal_p,
			      points_to_rank);
  bool changed = false;

  SET_FOREACH(points_to, pt, s){
    cell source = points_to_source(pt);
    cell sink = points_to_sink(pt);
    cell new_source = simple_cell_to_store_independent_cell(source, &changed);
    cell new_sink =  simple_cell_to_store_independent_cell(sink, &changed);
    points_to npt = make_points_to(new_source, new_sink, points_to_approximation(pt),points_to_descriptor(pt));
    set_add_element(res, res, (void*) npt);
  }

  return res;
}

/* change tab[i] into tab[*] .*/
cell get_array_path(expression e)
{
  effect ef = effect_undefined;
  reference  r = reference_undefined;
  cell c = cell_undefined;

  /*init the effect's engine*/
  list l_ef = NIL;
  set_methods_for_proper_simple_effects();
  list l1 = generic_proper_effects_of_complex_address_expression(e, &l_ef,
								 true);
  ef = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
  gen_free_list(l_ef); /* free the spine */

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
  list l_ef = NIL;
  list l1 = generic_proper_effects_of_complex_address_expression(e,
								 &l_ef, false);
  e1 = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
  gen_free_list(l_ef); /* free the spine */

  effects_free(l1);
  list l2 = effect_to_store_independent_sdfi_list(e1, false);
  e1 = EFFECT(CAR(l2));
  r = effect_any_reference(e1);
  effects_free(l2);
  cell c1 = make_cell_reference(r);
  generic_effects_reset_all_methods();

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
  list l_ef = NIL;
  list l1 = generic_proper_effects_of_complex_address_expression(e,
								 &l_ef, true);

  e1 = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
  gen_free_list(l_ef); /* free the spine */

  effect_add_dereferencing_dimension(e1);
  effects_free(l1);
  generic_effects_reset_all_methods();
  reference r1 = effect_any_reference(e1);
  cell c1 = make_cell_reference(r1);

  return c1;
}


bool equal_must_vreference(cell c1, cell c2)
{
  int i = 0;
  bool changed = false;
  set_methods_for_proper_simple_effects();
  c1 = simple_cell_to_store_independent_cell(c1, &changed);
  c2 = simple_cell_to_store_independent_cell(c2, &changed);
  generic_effects_reset_all_methods();
  reference r1 = cell_any_reference(c1);
  reference r2 = cell_any_reference(c2);
  entity v1 = reference_variable(r1);
  entity v2 = reference_variable(r2);
  list sl1 = NIL, sl2 = NIL;
  // FI: memory leak? generation of a new string?
  extern const char* entity_minimal_user_name(entity);

  i = strcmp(entity_minimal_user_name(v1), entity_minimal_user_name(v2));
  if (i==0) {
    sl1 = reference_indices(r1);
    sl2 = reference_indices(r2);
    if( (int)gen_length(sl1) == (int)gen_length(sl2) ) {
      for (;i==0 && !ENDP(sl1) && ! ENDP(sl2) ; POP(sl1), POP(sl2)){
	expression se1 = EXPRESSION(CAR(sl1));
	expression se2 = EXPRESSION(CAR(sl2));
	if( unbounded_expression_p(se1) ){
	  i = 0;
	}
	else if (expression_constant_p(se1) && expression_constant_p(se2)){
	  int i1 = expression_to_int(se1);
	  int i2 = expression_to_int(se2);
	  i = i2>i1? 1 : (i2<i1? -1 : 0);
	  if (i==0){
	    string s1 = words_to_string(words_expression(se1, NIL));
	    string s2 = words_to_string(words_expression(se2, NIL));
	    i = strcmp(s1, s2);
	  }
	}
	else {
	  string s1 = words_to_string(words_expression(se1, NIL));
	  string s2 = words_to_string(words_expression(se2, NIL));
	  i = strcmp(s1, s2);
	}
      }
    }
    else
      i = 1;
  }

  return (i==0? true: false);
}
