
/* This file contains all the operators defining  constant paths :

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
//#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
//#include "control.h"
#include "constants.h"
#include "misc.h"
//#include "parser_private.h"
//#include "syntax.h"
//#include "top-level.h"
#include "text-util.h"
#include "text.h"
#include "properties.h"
//#include "pipsmake.h"
//#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
//#include "effects-convex.h"
//#include "transformations.h"
//#include "preprocessor.h"
//#include "pipsdbm.h"
//#include "resources.h"
//#include "prettyprint.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"
#include "genC.h"


cell make_nowhere_cell()
{
  entity e = entity_all_xxx_locations(NOWHERE_LOCATION);
  reference r = make_reference(e, NIL);
  cell sink = make_cell_reference(r);
  return sink;
}

cell make_typed_nowhere_cell(type t)
{
  entity e = entity_all_xxx_locations_typed(NOWHERE_LOCATION, t);
  reference r = make_reference(e, NIL);
  cell sink = make_cell_reference(r);
  return sink;
}

/* assuming source is a reference to a pointer, build the
 * corresponding sink when the pointer is not initialized, i.e. is
 * undefined.
 */
cell cell_to_nowhere_sink(cell source)
{
  bool type_sensitive_p = !get_bool_property("ALIASING_ACROSS_TYPES");
  cell sink = cell_undefined;
  if(type_sensitive_p) {
    // FI: let's hope we create neither sharing nor memory leak
    bool to_be_freed_p = true;
    type t = type_to_pointed_type(cell_to_type(source, &to_be_freed_p));
    sink = make_typed_nowhere_cell(copy_type(t));
    if(to_be_freed_p) free_type(t);
  }
  else
    sink = make_nowhere_cell();
  return sink;
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


/* Operator gen for modules:
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


/* FI: really weird and unefficient. Also I asummed that vreference
   was limited to the subscript list... FI->AM: to be checked wrt your
   dissertation. */
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

  // FI: why not compare the entities v1==v2?
  i = strcmp(entity_minimal_user_name(v1), entity_minimal_user_name(v2));
  if ( i==0 ) {
    sl1 = reference_indices(r1);
    sl2 = reference_indices(r2);
    for (;i==0 && !ENDP(sl1) && ! ENDP(sl2) ; POP(sl1), POP(sl2)) {
      expression se1 = EXPRESSION(CAR(sl1));
      expression se2 = EXPRESSION(CAR(sl2));
      if (unbounded_expression_p(se2) && expression_constant_p(se1))
	i = 0;
      else if (unbounded_expression_p(se1) && expression_constant_p(se2))
	i = 0;
      else if (expression_constant_p(se1) && expression_constant_p(se2) ) {
	int i1 = expression_to_int(se1);
	int i2 = expression_to_int(se2);
	i = i2>i1? 1 : (i2<i1? -1 : 0);

	// FI: this piece of code seems out of place, if i==0, i==0
	if ( i==0 ) {
	  string s1 = words_to_string(words_expression(se1, NIL));
	  string s2 = words_to_string(words_expression(se2, NIL));
	  i = strcmp(s1, s2);
	}
      }
      else if(field_expression_p(se1) && field_expression_p(se2))
	i = expression_equal_p(se1,se2)? 0 : 1;
    }
  }

  return (i==0? true: false);
}

/* returns true if c2 must kills c1 because of the subscript expressions
 *
 * This function should be rewritten from scratch, with a defined
 * semantics for "*" as a subscript and possibly a larger use of
 * expression_equal_p(). Also, we need to specify if the scopes for
 * each rereference are equal or not.
 */
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

  // FI: this step could be assumed performed earlier
  i = strcmp(entity_minimal_user_name(v1), entity_minimal_user_name(v2));
  if (i==0) {
    sl1 = reference_indices(r1);
    sl2 = reference_indices(r2);
    for (;i==0 && !ENDP(sl1) && ! ENDP(sl2) ; POP(sl1), POP(sl2)){
      expression se1 = EXPRESSION(CAR(sl1));
      expression se2 = EXPRESSION(CAR(sl2));
      if(expression_constant_p(se2) && unbounded_expression_p(se1)){
	//i = 0;
	i = 1;
      }
      else if(expression_constant_p(se1) && unbounded_expression_p(se2)){
	//i = 0; could be true if * is interpreted as "forall"
	i = 1;
      }
      else if (expression_constant_p(se1) && expression_constant_p(se2)){
	int i1 = expression_to_int(se1);
	int i2 = expression_to_int(se2);
	i = i2>i1? 1 : (i2<i1? -1 : 0);
	if (i==0){ // FI->AM: I do not understand this step
	  string s1 = words_to_string(words_expression(se1, NIL));
	  string s2 = words_to_string(words_expression(se2, NIL));
	  i = strcmp(s1, s2);
	}
      }
      else {
	// FI->AM: very dangerous; only true if both references appear
	// exactly in the same scope; and "*" were not dealt with!
	if(unbounded_expression_p(se1)||unbounded_expression_p(se2))
	  i = 1;
	else {
	  //string s1 = words_to_string(words_expression(se1, NIL));
	  //string s2 = words_to_string(words_expression(se2, NIL));
	  //i = strcmp(s1, s2);
	  i = expression_equal_p(se1, se2)? 0 : 1;
	}
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
    bool to_be_freed1, to_be_freed2;
    t1 = points_to_reference_to_type(r1,&to_be_freed1);
    t2 = points_to_reference_to_type(r2,&to_be_freed2);
    type_equal_p = opkill_may_type(t1,t2);
    if(to_be_freed1) free_type(t1);
    if(to_be_freed2) free_type(t2);
  }
  kill_may_p = opkill_may_module(c1,c2) && opkill_may_name(c1,c2) &&
    type_equal_p && opkill_may_vreference(c1,c2);

  return kill_may_p;
}

/* returns true if c2 kills c1 */
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
  t1 = points_to_reference_to_type(r1,&equal_p);
  t2 = points_to_reference_to_type(r2, &equal_p);
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
  kill_must_p = opkill_must_module(c1,c2) && opkill_must_name(c1,c2) &&
    equal_p && opkill_must_vreference(c1,c2);

   return kill_must_p;
}


/* Compute the set of arcs in the input points-to relation "in" whose
 * approximation must be changed from "exact" to "may".
 *
 * This set is linked to set "gen_may1", although consistency would be
 * easier to maintain if only "kill_may" were used to generate the new arcs...
 *
 * kill_may = { pt in "in"| exact(pt) ^ \exists l in L conflict(l, source(pt))}
 *
 * The restriction to !atomic does not seem useful.
 */
set kill_may_set(list L, set in_may)
{
  set kill_may = set_generic_make(set_private, points_to_equal_p,
			     points_to_rank);
  FOREACH(cell, l, L) {
    SET_FOREACH(points_to, pt, in_may) {
      cell pt_source = points_to_source(pt);
      if (opkill_may_constant_path(pt_source,l)) {
	points_to npt = make_points_to(pt_source,
				       points_to_sink(pt), 
				       make_approximation_exact(),
				       make_descriptor_none());
	set_add_element(kill_may, kill_may, (void*)npt);
      }
    }
  }
  return kill_may;
}


/* Generate the subset of arcs that must be removed from the
 * points-to graph "in".
 *
 * Set "in_must" is the subset of set "in" with exact points-to arcs only.
 *
 * kill_1 = kill_must = {pt in "in" | source(pt) in L ^ |L|=1 ^ atomic(L) }
 *
 * where "atomic(L)" is a short cut for "atomic(l) forall l in L"
 *
 * Here, correctly, the atomicity is not checked directly, but
 * properly, using an operator of the lattice.
 */
set kill_must_set(list L, set in)
{
  set kill_must = new_simple_pt_map();
  int nL = (int) gen_length(L);

  if(nL==1) {
    cell l = CELL(CAR(L));
    SET_FOREACH(points_to, s, in) {
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

#if 0
bool expression_null_locations_p(expression e)
{
  if (expression_reference_p(e)) {
    entity v = expression_variable(e);
    return entity_null_locations_p(v);
  }
  else
    return false;
}
#endif

/*
 * Create a new set of points-to arcs "may" from an input points-to
 * set "in_may" (in fact, "in"), a list of assigned points-to cells,
 * L, and a list of value cells, R. In fact, "may" is the set "gen"
 * for the assignment of R to L.
 *
 * The arcs in "may" belongs either to subset "may1", which is derived
 * from "in_may", or to subset "may2", which is derived from couples
 * (l,r) in L x R, or to subset "may3", which is derived from couples
 * (l',r) in L' x R, where L' is the set or lower elements of L. In
 * other words:
 *
 * L'(l) = { l' | for all x<l in CP l'<l and x,=l' }
 *
 * This is used for element of pointer arrays.If a[1] is updated, then
 * a[*] must be updated too. This is generalized to a[*][*]..[*] for
 * multi-dimensional arrays. No distinction is made for a[1][*] and
 * a[*][2]. They are not considered and are both replaced by a[*][*].
 *
 * For "may1":
 *
 * may1 = {pt=(l,r,may) | exists pt'=(l,r,a') in in_may and l in L U L'}
 *
 * For "may2":
 *
 * may2 = {pt=(l,r,a) | exists l in L, exists r in R,
 *                      a=(|L|==1 and |R|==1 and atomic(l) and atomic(r) }
 *
 * For "may3":
 * 
 * may3 = {pt=(l',r,may) | exists l in L, l' in L'(l), exists r in R}
 *
 * Hence, may = may1 U may2 U may3.
 *
 * Note the disymetry between L and R as far as may3 is concerned. If
 * a[1] points toward b, then a[*] points toward b. If a points toward
 * b[1], then we do not generate an arc from a to b[*].
 *
 * Note also that "may1" must be consistent with "kill_may" in
 * list_assignment_to_points_to() in order to generate a consistent
 * pt_out.
 */
set gen_may_set(list L, list R, set in_may, bool *address_of_p)
{
  set gen_may1 = set_generic_make(set_private, points_to_equal_p,
				  points_to_rank);
  set gen_may2 = set_generic_make(set_private, points_to_equal_p,
				  points_to_rank);
  set gen_may3 = set_generic_make(set_private, points_to_equal_p,
				  points_to_rank);
  int len = (int) gen_length(L);

  //if(len > 1) {
  ///* If the source is not precisely known */
  /* It is easier not to have to maintain the consistency between
     gen_may1 and kill_may. */
  if(false) {
    FOREACH(cell, l, L) {
      SET_FOREACH(points_to, pt, in_may){
	if(approximation_exact_p(points_to_approximation(pt))) {
	  //if(!atomic_points_to_cell_p(points_to_source(pt))) {
	    //if(points_to_compare_cell(points_to_source(pt),l)) {
	    if(cells_may_conflict_p(points_to_source(pt),l)) {
	      // FI: it would be much easier/efficient to modify the approximation of pt
	      // But it is incompatible with the implementation of sets...
	      points_to npt = make_points_to(copy_cell(points_to_source(pt)),
					     copy_cell(points_to_sink(pt)),
					     make_approximation_may(),
					     make_descriptor_none());
	      set_add_element(gen_may1, gen_may1, (void*)npt);
	      //set_del_element(gen_may1, gen_may1, (void*)pt);
	    }
	    //	  }
	}
      }
    }
  }

  // Possibly generate an error for dereferencing an undefined pointer
  bool error_p =
    !get_bool_property("POINTS_TO_UNINITIALIZED_POINTER_DEREFERENCING");
  int lc = (int) gen_length(L);
  FOREACH(cell, l, L){
    reference lr = cell_any_reference(l);
    entity lv = reference_variable(lr);
    bool null_p = entity_null_locations_p(lv);
    bool nowhere_p = entity_typed_nowhere_locations_p(lv)
      || entity_nowhere_locations_p(lv);
    string bug = null_p? "a null" : "";
    bug = nowhere_p? "an undefined" : "";
    // Can the lhs be accessed?
    if(null_p || nowhere_p) {
      // No: two options; either a user_error() for dereferencing an
      // unitialized pointer or a conversion to anywhere, typed or not
      // FI: why be tolerant of NULL pointer dereferencing? For may information
      if(lc==1) {
	if(error_p)
	  pips_user_error("Dereferencing of %s pointer.\n", bug);
	else {
	  pips_user_warning("Dereferencing of %s pointer.\n", bug);
	}
      }
      else {
	pips_user_warning("Possible dereferencing of %s pointer.\n", bug);
      }
      type t = entity_type(lv);
      cell nl = make_anywhere_points_to_cell(t);
      set gen_l = gen_may_constant_paths(nl, R, in_may, address_of_p, len);
      set_union(gen_may2, gen_may2, gen_l);
    }
    else {
      // FI: memory leak due to call to call to gen_may_constant_paths()
      set gen_l = gen_may_constant_paths(l, R, in_may, address_of_p, len);
      // FI: be careful, the union does not preserve consistency because
      // the same arc may appear with different approximations
      set_union(gen_may2, gen_may2, gen_l);
      // free_set(gen_l);
    }
  }

  set_union(gen_may2, gen_may2, gen_may1);
  set_union(gen_may2, gen_may2, gen_may3);

  return gen_may2;
}


/*
 * create a set of points-to relations of the form:
 * element_L -> & element_R, EXACT
 *
 * FI: address_of_p does not seem to be updated in this function. Why
 * pass a pointer? My analysis is wrong if gen_must_constant_paths() updates it
 *
 * FI: lots of common points between gen_must_set() and
 * gen_may_set()... Possible unification?
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

  bool error_p = false; // generate an error for dereferencing an undefined pointer
  int lc = (int) gen_length(L);
  FOREACH(cell, l, L){
    // Can the lhs be accessed?
    reference lr = cell_any_reference(l);
    entity lv = reference_variable(lr);
    bool null_p = entity_null_locations_p(lv);
    bool nowhere_p = entity_typed_nowhere_locations_p(lv)
      || entity_nowhere_locations_p(lv);
    string bug = null_p? "a null" : "";
    bug = nowhere_p? "an undefined" : "";
    if(null_p || nowhere_p) {
      // No: two options; either a user_error() for dereferencing an
      // unitialized pointer or a conversion to anywhere, typed or not
      if(lc==1) {
	if(error_p)
	  pips_user_error("Dereferencing of %s pointer.\n", bug);
	else {
	  pips_user_warning("Dereferencing of %s pointer.\n", bug);
	}
      }
      else {
	pips_user_warning("Possible dereferencing of %s pointer.\n", bug);
      }
      type t = entity_type(lv);
      cell nl = make_anywhere_points_to_cell(t);
      set must_l = gen_must_constant_paths(nl, R, in_must, address_of_p, len);
      set_union(gen_must2, gen_must2, must_l);
    }
    else {
    set must_l = gen_must_constant_paths(l, R, in_must, address_of_p, len);
    set_union(gen_must2, gen_must2, must_l);
    // FI: shouldn't must_l be freed?
    }
  }
  set_union(gen_must2, gen_must2,gen_must1);

  return gen_must2;
}

/* Does cell "c" represent a unique memory location or a set of memory
 * locations?
 *
 * This is key to decide if a points-to arc is a must or a may arc.
 *
 * Is it always possible to decide when heap abstract locations are concerned?
 *
 * See also cell_abstract_location_p()
 */
bool unique_location_cell_p(cell c)
{
  bool unique_p = !anywhere_cell_p(c)
    && !cell_typed_anywhere_locations_p(c)
    // FI: typed or not?
    // FI: how do you know when heap cells are unique and when they
    // represent a set of cells?
    && !heap_cell_p(c);

  // FI: how do you deal with arrays of pointers?
  if(unique_p) {
    reference r = cell_any_reference(c);
    list sl = reference_indices(r);
    FOREACH(EXPRESSION, s, sl) {
      if(unbounded_expression_p(s)) {
	unique_p = false;
	break;
      }
    }
  }
  return unique_p;
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
    pips_internal_error("address_of_p should always be true in the new implementation/.\n");
    /* here we have x = y, then we generate (x,y1,a)|(y,y1,a) as
       points to relation */
    FOREACH(cell, r, R){
      SET_FOREACH(points_to, i, in_may){
	if (/* locations_equal_p */equal_must_vreference(r, points_to_source(i)) /* &&  !entity_abstract_location_p(el) */ ){
	  cell nl = copy_cell(l);
	  pt = make_points_to(nl,
			      copy_cell(points_to_sink(i)),
			      make_approximation_may(),
			      make_descriptor_none());
	}
	if(array_entity_p(reference_variable(cell_any_reference(r)))){
	  reference ref = cell_any_reference(r);
	  bool t_to_be_freed = false;
	  type t = points_to_reference_to_type(ref, &t_to_be_freed);
	  if(pointer_type_p(t)) {
	    cell nl = copy_cell(l);
	    pt = make_points_to(nl, r, make_approximation_may(), make_descriptor_none());
	  }
	  else {
	    cell nl = copy_cell(l);
	    pt = make_points_to(nl, points_to_sink(i), make_approximation_may(), make_descriptor_none());
	  }
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
      // FI: check the unicity of the locations
      // FI: relationship with atomic_points_to_cell_p()?
      approximation a = (Lc+Rc>2
			 || !unique_location_cell_p(l)
			 || !unique_location_cell_p(r)) ?
	make_approximation_may() : make_approximation_exact();
      /* Should be replaced by opgen_constant_path(l,r) */
      //reference ref = cell_any_reference(r);
      /* if(reference_unbounded_indices_p(ref)) */
      /*   a = make_approximation_may(); */
      cell nl = copy_cell(l);
      /* Make sure the types are compatible... */
      points_to_cell_types_compatibility(nl, r);
      pt = make_points_to(nl, copy_cell(r), a, make_descriptor_none());
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
  //approximation a = approximation_undefined;
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
      approximation a = (Lc+Rc>2
			 || !unique_location_cell_p(l)
			 || !unique_location_cell_p(r))?
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
	    type t = points_to_reference_to_type(ref, &t_to_be_freed);
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

  Used by statement.c
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
    cell sink = make_nowhere_cell();
    points_to pt_to = make_points_to(c, sink, copy_approximation(a), make_descriptor_none());
    set_add_element(gen, gen, (void*)pt_to);
  }
  free_approximation(a);
  set_union(res, gen, input_kill_diff);

  return res;
}

/*
 * want to test if r can only be a constant path and nothing else
 * WARNING : not totally tested
 * for instance
 *        a[0], a[1], a[i], i, j, ... have to return false
 *        a[*], var points by formal parameter (_p_0, ...), element of strut (s.id, ...), heap element have to return true
 *  * !effect_reference_dereferencing_p(r, &exact_p)
 *      can return true when it's not a constant path like a[i] (a[i] and not a[*])
 *      can make a side effect for the declaration of variable in parameter (only?), don't know why
 *      for instance with Semantics-New/Pointer.sub/memcopy01
 *        void memcopy01([...], char dst[size])  -->  void memcopy01([...], char dst[i])
 *  * store_independent_reference_p(r)
 *      can return false for some cp like a[0], can't permit to treat the array
 *          return false for the structure too, can't permit to treat the struct
 *      can return true for i, j, ... that can be a constant path but not strictly a constant path
 * param r          reference to analyze to see if it's a constant path
 * return           true if r is exactly a constant path
 *
 * This function is not used by the points-to analysis, but by semantics
 */
bool strict_constant_path_p(reference r)
{
  bool constant_path = false;
  entity v = reference_variable(r);
  list l_ind = reference_indices(r);

  // Test the different top and bottom area
  if (entity_all_locations_p(v)
      || entity_anywhere_locations_p(v) || entity_typed_anywhere_locations_p(v)
      || entity_nowhere_locations_p(v) || entity_typed_nowhere_locations_p(v)
      || entity_all_module_locations_p(v)
      ) {
    constant_path = true;
  }
  else if (entity_all_module_heap_locations_p(v)
      || entity_all_heap_locations_p(v)
      ) {
    constant_path = true;
  }
  else if (entity_all_module_stack_locations_p(v)
      || entity_all_stack_locations_p(v)
      ) {
    constant_path = true;
  }
  else if (entity_all_module_static_locations_p(v)
      || entity_all_static_locations_p(v)
      ) {
    constant_path = true;
  }
  else if (entity_all_module_dynamic_locations_p(v)
      || entity_all_dynamic_locations_p(v)
      ) {
    constant_path = true;
  }
  else if (entity_abstract_location_p(v)) { // Maybe this test permit to eliminate the 4 test just before?
    constant_path = true;
  }
  // Test if it's the constant NULL
  else if (entity_null_locations_p(v)) {
    constant_path = true;
  }
  // Test if it's a formal parameter
  else if (entity_stub_sink_p(v)) {
    constant_path = true;
  }
  // Test if it's a heap element
  else if (heap_area_p(v)) {
    constant_path = true;
  }
  // Maybe not efficient enough, for array of struct or struct of array?
  // Test if it's a structure
  else if (struct_type_p(entity_type(v)) && !ENDP(l_ind)) {
    constant_path = true;
  }
  // Test if it's a array with only *
  else if (!ENDP(l_ind)) {
    // see reference_unbounded_indices_p
    constant_path = reference_unbounded_indices_p(r);
  }

  return constant_path;
}

/* TODO
 * most of the time return same result that !effect_reference_dereferencing_p for the moment
 * want to test if r can be a constant path
 * for instance
 *        a[i] have to return false (something else?)
 *        a[0], a[1], i, j, a[*], var points by formal parameter (_p_0, ...), element of strut (s.id, ...), heap element
 *                  have to return true
 *  * !effect_reference_dereferencing_p(r, &exact_p)
 *      can return true when it's not a constant path like a[i] (a[i] and not a[*])
 *      can make a side effect for the declaration of variable in parameter (only?), don't know why
 *      for instance with Semantics-New/Pointer.sub/memcopy01
 *        void memcopy01([...], char dst[size])  -->  void memcopy01([...], char dst[i])
 *  * store_independent_reference_p(r)
 *      can return false for some cp like a[0], can't permit to treat the array
 *          return false for the structure too, can't permit to treat the struct
 *      can return true for i, j, ... that can be a constant path but not strictly a constant path
 * param r          reference to analyze to see if it's a constant path
 * return           true if r can be constant path
 *
 * This function is not used by the points-to analysis, but by semantics
 */
bool can_be_constant_path_p(reference r)
{
  bool constant_path = true;

  if (strict_constant_path_p(r))
    constant_path = true;
  else {
    bool exact_p = true;
    if (!effect_reference_dereferencing_p(r, &exact_p)) {
      constant_path = true;
    }
    else {
      constant_path = false;
    }
  }

  return constant_path;
}
