/*

  $Id$

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
 * This file contains functions used to compute all pointer locations
 * within an entity, if any. It is tricky for structures containing
 * structures and for arrays of structures or pointeurs and much
 * easier for scalar pointers.
 *
 * The similar functions in Amira's implementation are located in
 * points_to_analysis_general_algorithm.c.
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
#include "constants.h"
#include "misc.h"
#include "properties.h"
#include "effects-generic.h"
#include "effects-simple.h"
//#include "effects-convex.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"


/* When the declaration of "e" does not contain an initial value, find
 * all allocated pointers in entity e. We distinguish between scalar
 * pointers, array of pointers and struct containing pointers and
 * struct. Return a list of cells each containing a reference to aa
 * pointer. These cells are used later as sources to build a points-to
 * data structure pointing to nowhere/undefined.
 */
list variable_to_pointer_locations(entity e)
{
  list l = NIL;
  
  if (entity_variable_p(e)) {
    /*AM: missing recursive descent on variable_dimensions. int a[*(q=p)]*/

    type ut = ultimate_type(entity_type(e));
    if(pointer_type_p(ut) || array_of_pointers_type_p(ut)) {
      if (entity_array_p(e)) {
	expression ind = make_unbounded_expression();
	reference r = make_reference(e, CONS(EXPRESSION, ind, NULL));
	cell c = make_cell_reference(r);
	l = CONS(CELL, c, NIL);
      }
      else {
	reference r = make_reference(e, NIL);
	cell c = make_cell_reference(r);
	l = CONS(CELL, c, NIL);
      } 
    }
    else if(struct_type_p(ut) || array_of_struct_type_p(ut)) {
      basic b = variable_basic(type_variable(ut));
      entity ee = basic_derived(b);
      l = struct_variable_to_pointer_locations(e, ee);
    }
  }

  return l;
}

/* return list of cells for pointers declared directly or indirecltly in
 * variable "e" of type struct defined by entity "ee" and its type.
 *
 * Typedefs have already been taken care of by the caller (?).
 */
list struct_variable_to_pointer_locations(entity e, entity ee)
{
  list l = NIL;
  // bool  eval = true;
  type tt = entity_type(ee);
  pips_assert("entity ee has type struct", type_struct_p(tt));
  list fl = type_struct(tt); // list of fields, or field list

  FOREACH(ENTITY, f, fl) {
    type ft = ultimate_type(entity_type(f));
    if(pointer_type_p(ft) || array_of_pointers_type_p(ft)) {
      // FI: I wonder if we should not build the points-to right away
      // when we know the type of the nowehere/undefined cell;
      // reminder: this type is useful to switch to a anywhere
      // abstract location
      list l2 = NIL;
      if(array_type_p(entity_type(f))) {
	expression s2 = make_unbounded_expression();
	l2 = CONS(EXPRESSION, s2, NIL);
      }
      expression s = entity_to_expression(f);
      reference r = make_reference(e, CONS(EXPRESSION, s, l2));
      cell c = make_cell_reference(r);
      l = gen_nconc(l, CONS(CELL, c, NIL));
    }
    else if(struct_type_p(ft) || array_of_struct_type_p(ft)) {
      // The main data structure contains a secondary data structure
      // FI: build the prefix and go down
      list l2 = NIL;
      if(array_type_p(entity_type(f))) {
	expression s2 = make_unbounded_expression();
	l2 = CONS(EXPRESSION, s2, gen_full_copy_list(l2));
      }
      expression s = entity_to_expression(f);
      reference r = make_reference(e, CONS(EXPRESSION, s, l2));
      cell c = make_cell_reference(r);

      /* Find pointers downwards and build a list of cells. */
      list ll = struct_variable_to_pointer_subscripts(c, f);

      free_cell(c);
      l = gen_nconc(l, ll);
      //pips_internal_error("Not implemented yet.\n");
      //l = array_of_struct_to_pointer_location(e, ee);
    }
  }

#if 0
  /* AM: Do not build expressions ef and ex */
  if( !array_argument_p(ex) ) {
    FOREACH( ENTITY, i, l1 ){
      expression ef = entity_to_expression(i);
      if( expression_pointer_p(ef) ) 
        {
          expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
                                          ex,
                                          ef);
          cell c = get_memory_path(ex1, &eval);
          l = gen_nconc(CONS(CELL, c, NIL),l);
        }
      else if( array_argument_p(ef) ) 
        {
          basic b = variable_basic(type_variable(entity_type(i)));
          /* arrays of pointers are changed into independent store arrays
             and initialized to nowhere_b0 */
          if( basic_pointer_p(b) )
            {
              effect eff = effect_undefined;
              expression ind = make_unbounded_expression();
              expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
                                              ex,
                                              ef);
              set_methods_for_proper_simple_effects();
              list l_ef = NIL;
              list l1 = generic_proper_effects_of_complex_address_expression(ex1, &l_ef,
                                                                             true);
              eff = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
              gen_free_list(l_ef);
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
    l = array_of_struct_to_pointer_location(e, ee);
#endif  
  return l;
}

/* returns a list of cells to reach pointers depending
 * on field f. Cell c is the current prefix.
 */
list struct_variable_to_pointer_subscripts(cell c, entity f)
{
  list sl = NIL;
  type ft = ultimate_type(entity_type(f));
  pips_assert("We are dealing with a struct", struct_type_p(ft));
  basic fb = variable_basic(type_variable(ft));
  type st = entity_type(basic_derived(fb));
  list sfl = type_struct(st);

  FOREACH(ENTITY, sf, sfl) {
    type sft = ultimate_type(entity_type(sf));
    if(pointer_type_p(sft) || array_of_pointers_type_p(sft)) {
      /* copy cell c and add a subscript for f */
      cell nc = copy_cell(c);
      reference r = cell_any_reference(nc);
      expression se = entity_to_expression(sf);
      reference_indices(r) = gen_nconc(reference_indices(r),
				       CONS(EXPRESSION, se, NIL));
      if(array_entity_p(sf)) {
	expression ue = make_unbounded_expression();
	reference_indices(r) = gen_nconc(reference_indices(r),
					 CONS(EXPRESSION, ue, NIL));
      }
      sl = gen_nconc(sl, CONS(CELL, nc, NIL));
    }
    else if(struct_type_p(sft) || array_of_struct_type_p(sft)) {
      /* copy cell c and add a subscript for f */
      cell nc = copy_cell(c);
      reference r = cell_any_reference(nc);
      expression se = entity_to_expression(sf);
      reference_indices(r) = gen_nconc(reference_indices(r),
				       CONS(EXPRESSION, se, NIL));
      if(array_entity_p(sf)) {
	expression ue = make_unbounded_expression();
	reference_indices(r) = gen_nconc(reference_indices(r),
					 CONS(EXPRESSION, ue, NIL));
      }
      list nsl = struct_variable_to_pointer_subscripts(nc, sf);
      sl = gen_nconc(sl, nsl);
      free_cell(nc);
    }
  }

  return sl;
}
/* FI: to be revisited with Amira. Many objects seems to be allocated
   but never freed. Cut-and-paste used to handle effects. */
list array_of_struct_to_pointer_location(entity e, entity field)
{
  list l = NIL;
  bool eval = true;
  type tt = entity_type(field);

  pips_assert("argument \"field\" is a struct", type_struct_p(tt));
  list l1 = type_struct(tt);
  FOREACH( ENTITY, i, l1 ) {
    expression ee = entity_to_expression(i);
    if( expression_pointer_p(ee)){
      expression ex = entity_to_expression(e);
      if( array_argument_p(ex) ) {
          effect ef = effect_undefined;
          reference  r = reference_undefined;

          /*init the effect's engine*/
          list l_ef = NIL;
          set_methods_for_proper_simple_effects();
          list l1 = generic_proper_effects_of_complex_address_expression(ex, &l_ef,
                                                                         true);
          
	  generic_effects_reset_all_methods();
          ef = EFFECT(CAR(l_ef)); /* In fact, there should be a FOREACH to scan all elements of l_ef */
          /* gen_free_list(l_ef); */ /* free the spine */
          effect_add_dereferencing_dimension(ef);
          r = effect_any_reference(ef);
          list l_inds = reference_indices(r);
          EXPRESSION_(CAR(l_inds)) = make_unbounded_expression();
          ex = reference_to_expression(r);
          effects_free(l1);
        }
      expression ex1 = MakeBinaryCall(entity_intrinsic(FIELD_OPERATOR_NAME),
                                      ex,
                                      ee);
      cell c = get_memory_path(ex1, &eval);
      l = gen_nconc(CONS(CELL, c, NIL), l);
    }
  }
      
  return l;
}
