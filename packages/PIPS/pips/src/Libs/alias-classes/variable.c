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
 * Several pieces of code have been cut-and-pasted. Either more
 * functions should have been defined or array of structs are not
 * handled in a graceful way or both problems occur.
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

    type t =  entity_basic_concrete_type(e);
    if(pointer_type_p(t) || array_of_pointers_type_p(t)) {
      if (entity_array_p(e)) {
	/* variable v = type_variable(t); */
	/* int d = (int) gen_length(variable_dimensions(v)); */
	/* list sl = NIL; */
	/* int i; */
	/* for(i=0;i<d;i++) { */
	/*   expression ind = make_unbounded_expression(); */
	/*   sl = CONS(EXPRESSION, ind, sl); */
	/* } */
	/* reference r = make_reference(e, sl); */
	reference r = make_reference(e, NIL);
	cell c = make_cell_reference(r);
	points_to_cell_add_unbounded_subscripts(c);
	l = CONS(CELL, c, NIL);
      }
      else {
	// FI: could be unified with previous case using d==0
	reference r = make_reference(e, NIL);
	cell c = make_cell_reference(r);
	l = CONS(CELL, c, NIL);
      } 
    }
    else if(struct_type_p(t) || array_of_struct_type_p(t)) {
      basic b = variable_basic(type_variable(t));
      entity ee = basic_derived(b);
      l = struct_variable_to_pointer_locations(e, ee);
    }
    //free_type(t); ee is used above
  }

  return l;
}

/* return list of cells for pointers declared directly or indirecltly in
 * variable "e" of type struct defined by entity "ee" and its type.
 *
 * Typedefs have already been taken care of by the caller (?).
 *
 * Signature with e and ee inherited from Amira Mensi.
 */
list struct_variable_to_pointer_locations(entity e, entity ee)
{
  list l = NIL;
  // bool  eval = true;
  type tt = entity_type(ee);
  pips_assert("entity ee has type struct", type_struct_p(tt));
  list fl = type_struct(tt); // list of fields, or field list

  list sl = NIL;
  if(array_entity_p(e)) {
    int i;
    for(i=0; i< variable_entity_dimension(e); i++) {
      expression se = make_unbounded_expression();
      sl = CONS(EXPRESSION, se, sl);
    }
  }

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
      list fsl = CONS(EXPRESSION, s, l2); // field list
      list nsl = gen_full_copy_list(sl); // subscript list
      list fl = gen_nconc(nsl,fsl); // full list
      reference r = make_reference(e, fl);
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
      list fsl = CONS(EXPRESSION, s, l2); // field list
      list nsl = gen_full_copy_list(sl); // subscript list
      list fl = gen_nconc(nsl,fsl); // full list
      reference r = make_reference(e, fl);
      cell c = make_cell_reference(r);

      /* Find pointers downwards and build a list of cells. */
      list ll = struct_variable_to_pointer_subscripts(c, f);

      free_cell(c);
      l = gen_nconc(l, ll);
      //pips_internal_error("Not implemented yet.\n");
      //l = array_of_struct_to_pointer_location(e, ee);
    }
  }
  gen_full_free_list(sl);
  return l;
}

/* returns a list of cells to reach pointers depending
 * on field f. Cell c is the current prefix.
 */
list struct_variable_to_pointer_subscripts(cell c, entity f)
{
  list sl = NIL;
  type ft = ultimate_type(entity_type(f));
  pips_assert("We are dealing with a struct", struct_type_p(ft)
	      || array_of_struct_type_p(ft));
  basic fb = variable_basic(type_variable(ft));
  type st = entity_type(basic_derived(fb));
  list sfl = type_struct(st);

  /* In case we are dealing with an array of structs, add subscript
     expressions in mc, a modified copy of parameter c */
  cell mc = copy_cell(c); // modified cell c
  /*
  if(array_type_p(ft)) {
    list ssl = NIL;
    int i;
    for(i=0; i< variable_dimension_number(type_variable(ft)); i++) {
      expression se = make_unbounded_expression();
      ssl = CONS(EXPRESSION, se, ssl);
    }
    reference r = cell_any_reference(mc);
    reference_indices(r) = gen_nconc(reference_indices(r), ssl);
  }
  */

  /* Take care of each field in the structure. */
  FOREACH(ENTITY, sf, sfl) {
    type sft = ultimate_type(entity_type(sf));
    if(pointer_type_p(sft) || array_of_pointers_type_p(sft)) {
      /* copy cell c and add a subscript for f */
      cell nc = copy_cell(mc);
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

  free_cell(mc);

  return sl;
}
