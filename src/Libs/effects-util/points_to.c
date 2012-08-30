/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2009-2010 HPC Project

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"

#include "misc.h"

/***************************************/
/* Function storing points to information attached to a statement
 */
/* Generate a global variable holding a statement_points_to, a mapping
 * from statements to lists of points-to arcs. The variable is called
 * "pt_to_list_object".
 *
 * The macro also generates a set of functions used to deal with this global variables.
 *
 * The functions are defined in newgen_generic_function.h:
 *
 * pt_to_list_undefined_p()
 *
 * reset_pt_to_list()
 *
 * error_reset_pt_to_list()
 *
 * set_pt_to_list(o)
 *
 * get_pt_to_list()
 *
 * store_pt_to_list(k, v)
 *
 * update_pt_to_list(k, v)
 *
 * load_pt_to_list(k)
 *
 * delete_pt_to_list(k)
 *
 * bound_pt_to_list_p(k)
 *
 * store_or_update_pt_to_list(k, v)
*/
GENERIC_GLOBAL_FUNCTION(pt_to_list, statement_points_to)

/* Functions specific to points-to analysis
*/

/* */
cell make_anywhere_points_to_cell(type t __attribute__ ((unused)))
{
  // entity n = entity_all_locations();
  entity n = entity_all_xxx_locations_typed(ANYWHERE_LOCATION, t);
  reference r = make_reference(n, NIL);
  cell c = make_cell_reference(r);
  return c;
}

bool formal_parameter_points_to_cell_p(cell c)
{
  bool formal_p = true;
  reference r = cell_any_reference(c);
  entity v = reference_variable(r);
  formal_p = formal_parameter_p(v);
  return formal_p;
}

bool stub_points_to_cell_p(cell c)
{
  bool formal_p = true;
  reference r = cell_any_reference(c);
  entity v = reference_variable(r);
  formal_p = entity_stub_sink_p(v); // FI: can be a source too
  return formal_p;
}

bool points_to_cell_in_list_p(cell c, list L)
{
  bool found_p = false;
  FOREACH(CELL, lc, L) {
    if(cell_equal_p(c,lc)) {
      found_p =true;
      break;
    }
  }
  return found_p;
}

/* Two cells are related if they are based on the same entity */
bool related_points_to_cell_in_list_p(cell c, list L)
{
  bool found_p = false;
  reference rc = cell_any_reference(c);
  entity ec = reference_variable(rc);
  FOREACH(CELL, lc, L) {
    reference rlc = cell_any_reference(lc);
    entity elc = reference_variable(rlc);
    if(ec==elc) {
      found_p =true;
      break;
    }
  }
  return found_p;
}

 /* Debug: print a cell list for points-to. Parameter f is not useful
    in a debugging context. */
void fprint_points_to_cell(FILE * f __attribute__ ((unused)), cell c)
{
  int dn = cell_domain_number(c);

  // For debugging with gdb, dynamic type checking
  if(dn==cell_domain) {
    if(cell_undefined_p(c))
      fprintf(stderr, "cell undefined\n");
    else {
      reference r = cell_any_reference(c);
      print_reference(r);
    }
  }
  else
    fprintf(stderr, "Not a Newgen cell object\n");
}

/* Debug: use stderr */
void print_points_to_cell(cell c)
{
  fprint_points_to_cell(stderr, c);
}

/* Debug */
void print_points_to_cells(list cl)
{
  if(ENDP(cl))
    fprintf(stderr, "Empty cell list");
  else {
    FOREACH(CELL, c, cl) {
      print_points_to_cell(c);
      if(!ENDP(CDR(cl)))
	fprintf(stderr, ", ");
    }
  }
  fprintf(stderr, "\n");
}

/* Check if expression "e" is a reference to a struct field. */
bool field_reference_expression_p(expression e)
{
  bool field_p = false;
  syntax s = expression_syntax(e);
  if(syntax_reference_p(s)) {
    reference r = syntax_reference(s);
    entity f = reference_variable(r);
    field_p = entity_field_p(f);
  }
  return field_p;
}

/* Compute the number of array subscript at the end of a points_to_reference
 *
 * Look for the last field subscript and count the number of
 * subscripts after it. If no field susbcript is found, then all
 * subscripts are final array subscripts.
 *
 * To make thinks easier, the subscript list is reversed.
 */
int points_to_reference_to_final_dimension(reference r)
{
  list sl = reference_indices(r);
  sl = gen_nreverse(sl);
  int d = 0;
  FOREACH(EXPRESSION, e, sl) {
    if(field_reference_expression_p(e))
      break;
    else
      d++;
  }
  sl = gen_nreverse(sl);
  reference_indices(r) = sl;
  return d;
}

/* Substitute the subscripts "sl" in points-to reference "r" just after the
 * last field subscript by "nsl".
 *
 * "sl" must be broken into three parts, possibly empty:
 *
 *  1. The first part that ends up at the last field reference. It may
 *  be empty when no field is referenced.
 *
 *  2. The second part that starts just after the last field reference
 *  and that counts at most as many elements as the new subscript
 *  list, "nsl". This part must be substituted.
 *
 *  3. The third part that is left unchanged after substitution.
 *
 * Issue: how do you know that the initial array subscript must be
 * preserved because it is an implicit dimension added for pointer
 * arithmetics?
 */
void points_to_reference_update_final_subscripts(reference r, list nsl)
{
  list sl = reference_indices(r);
  list sl1 = NIL, sl3 = NIL, sl23 = NIL;

  sl = gen_nreverse(sl); // sl1 and sl23 are built in the right order
  bool found_p = false;
  bool skip_one_p = false; // to skip indices added for pointer arithmetic
  FOREACH(EXPRESSION, e, sl) {
    if(field_reference_expression_p(e)) {
      type et = expression_to_type(e);
      if(pointer_type_p(et))
	skip_one_p = true;
      found_p = true;
      free_type(et);
    }
    if(found_p) {
      /* build sl1 */
      sl1 = CONS(EXPRESSION, e , sl1);
    }
    else
      sl23 = CONS(EXPRESSION, e , sl23);
  }

  if(skip_one_p && ENDP(sl23) && !ENDP(nsl)) {
    pips_internal_error("We are not generating a memory access constant path.\n");
  }

  // FI: place the new indices as early as possible
#if 0
  int n = (int) gen_length(nsl);
  int i = 0;
  FOREACH(EXPRESSION, e, sl23) {
    if(skip_one_p) {
      sl1 = gen_nconc(sl1, CONS(EXPRESSION, e , NIL));
      skip_one_p = false;
    }
    else {
      if(i<n)
	free_expression(e);
      else
	sl3 = gen_nconc(sl3, CONS(EXPRESSION, e, NIL));
      i++;
    }
  }

  sl = gen_nconc(sl1, nsl);
  sl = gen_nconc(sl, sl3);
#endif

  // FI: place the new indices as late as possible
  int n = (int) gen_length(nsl);
  int n23 = (int) gen_length(sl23);
  int i = 0;
  FOREACH(EXPRESSION, e, sl23) {
    if(i>=n23-n)
      free_expression(e);
    else
      sl3 = gen_nconc(sl3, CONS(EXPRESSION, e, NIL));
    i++;
  }

  sl = gen_nconc(sl1, sl3);
  sl = gen_nconc(sl, nsl);

  gen_free_list(sl23);
  reference_indices(r) = sl;

  // We do not want to generate indirection in the reference
  // The exactitude information is not relevant here
  // bool exact_p;
  // pips_assert("The reference is a constant memory access path",
  //             !effect_reference_dereferencing_p( r, &exact_p));
  // Might only work for standard references and not for points-to
  // references: core dumps with points-to references.
}

/* Look for the index in "r" that corresponds to a pointer of type "t"
 * and return the corresponding element list. In other words, the type
 * of "&r" is "t".
 *
 * It is done in a very inefficient way
 */
list points_to_reference_to_typed_index(reference r, type t)
{
  bool to_be_freed;
  type rt = points_to_reference_to_type(r, &to_be_freed);
  list rsl = reference_indices(r); // reference subscript list
  pips_assert("t is a pointer type", C_pointer_type_p(t));
  type pt = C_type_to_pointed_type(t);
  list psl = list_undefined; // pointed subscript list

  if(array_pointer_type_equal_p(rt, pt))
    psl = gen_last(rsl);
  else {
    if(to_be_freed) free_type(rt);
    entity v = reference_variable(r);
    list nl = NIL;
    int i;
    int n = (int) gen_length(rsl);
    reference nr = make_reference(v, nl);
    bool found_p = false;

    for(i=0;i<n;i++) {
      nl = gen_nconc(nl, CONS(EXPRESSION,
			      copy_expression(EXPRESSION(gen_nth(i, rsl))),
			      NIL));
      rt = points_to_reference_to_type(nr, &to_be_freed);
      if(array_pointer_type_equal_p(rt, pt)) {
	found_p = true;
	break;
      }
    }

    free_reference(nr);

    if(found_p)
      psl = gen_nthcdr(i, rsl);
    else
      pips_internal_error("Type not found.\n");
  }

  free_type(pt);

  return psl;
}
