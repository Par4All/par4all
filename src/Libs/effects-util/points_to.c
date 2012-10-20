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
      reference_indices(nr) = nl;
      rt = points_to_reference_to_type(nr, &to_be_freed);
      if(array_pointer_type_equal_p(rt, pt)) {
	found_p = true;
	break;
      }
    }

    free_reference(nr);

    if(found_p)
      psl = gen_nthcdr(i, rsl);
    else {
      // The issue may be due to a user bug, as was the case in
      // Strict_typing.sub/malloc03.c
      if(entity_heap_location_p(v)) {
	// It would be nice to have a current statement stack...
	pips_user_error("The dynamic allocation of \"%s\" is likely "
			"to be inadequate with its use in the current "
			"statement.\n", entity_local_name(v));
      }
      else
	pips_internal_error("Type not found.\n");
    }
  }

  free_type(pt);

  return psl;
}

/* Is it a unique concrete memory location? */
bool atomic_points_to_cell_p(cell c)
{
  reference r = cell_any_reference(c);
  bool atomic_p = null_cell_p(c) || atomic_points_to_reference_p(r);

  return atomic_p;
}

/* Is it a unique concrete memory location?
 *
 * No, if it is a reference to an abstract location.
 *
 * No, if the subscripts included an unbounded expression.
 *
 * Very preliminary version. One of the keys to Amira mensi's work.
 *
 * More about stubs: a stub is not NULL but there is no information to
 * know if they represent one address or a set of addresses. Unless
 * the intraprocedural points-to analysis is performed for each
 * combination of atomic/non-atomic stub, safety implies that
 * stub-based references are not atomic.
 *
 * Note: it is assumed that the reference is a points-to
 * reference. All subscripts are constants, field references or
 * unbounded expressions.
 */
bool atomic_points_to_reference_p(reference r)
{
  bool atomic_p = false;
  entity v = reference_variable(r);

  if(!entity_null_locations_p(v) // FI: NULL is considered atomic
     && !entity_typed_nowhere_locations_p(v)
     && !entity_typed_anywhere_locations_p(v)
     && !entity_anywhere_locations_p(v)
     && !entity_heap_location_p(v)) {
    list sl = reference_indices(r);
    entity v = reference_variable(r);
    if(!entity_stub_sink_p(v)) {
      atomic_p = true;
      FOREACH(EXPRESSION, se, sl) {
	if(unbounded_expression_p(se)) {
	  atomic_p = false;
	  break;
	}
      }
    }
  }

  return atomic_p;
}

/* points-to cells use abstract addresses, hence the proper comparison
 * is an intersection. simple references are considered to be
 * singleton.
 *
 * Assume no aliasing between variables and within data structures.
 *
 * It is safe to assume intersection...
 */
bool points_to_cells_intersect_p(cell lc, cell rc)
{
  bool intersect_p = false;
  if(cell_equal_p(lc, rc)) {
    // FI: too simple... All the subscript should be checked.
    // unbounded expressions should be used to decide about a possible
    // intersection... Unless this is guarded by
    // atomic_points_to_reference_p(). To be investigated.
    intersect_p = true;
  }
  else {
    // Look for abstract domains
    // Probably pretty complex...
    // Simple first version...
    reference lr = cell_any_reference(lc);
    entity le = reference_variable(lr);
    reference rr = cell_any_reference(rc);
    entity re = reference_variable(rr);
    intersect_p = entities_may_conflict_p(le, re);
  }
  return intersect_p;
}

/* Allocate a cell that is the minimal upper bound of the cells in
 * list "cl" according to the points-to cell lattice...
 *
 * An over-approximation is always safe. So, an anywhere cell, typed
 * or not, can be returned in a first drat implementation.
 *
 * The points-to cell lattice is the product of three lattices, the
 * module lattice, the type lattice and the abstracct reference
 * lattice...
 */
cell points_to_cells_minimal_upper_bound(list cl __attribute__ ((unused)))
{
#if 0
  entity m = points_to_cells_minimal_module_upper_bound(cl);
  type t = points_to_cells_minimal_type_upper_bound(cl);
  reference r = points_to_cells_minimal_reference_upper_bound(m, t, cl);
  cell c = make_cell_reference(r);
#endif
  type t = make_scalar_overloaded_type();
  cell c = make_anywhere_points_to_cell(t);
  return c;
}

entity points_to_cells_minimal_module_upper_bound(list cl __attribute__ ((unused)))
{
  entity m = entity_undefined;
  return m;
}

type points_to_cells_minimal_type_upper_bound(list cl __attribute__ ((unused)))
{
  type t = type_undefined;
  return t;
}

reference points_to_cells_minimal_reference_upper_bound(entity m __attribute__ ((unused)), type t __attribute__ ((unused)), list cl __attribute__ ((unused)))
{
  reference r = reference_undefined;
  return r;
}

/* Is this a reference to an array or a reference to a pointer? This
   is not linked to the type of the reference, as a reference may be a
   pointer, such as "a[10]" when "a" is declared int "a[10][20]".*/
bool points_to_array_reference_p(reference r)
{
  bool array_p = false;
  list sl = reference_indices(r);
  entity v = reference_variable(r);

  if(ENDP(sl)) {
    type t = entity_basic_concrete_type(v);
    array_p = array_type_p(t);
  }
  else {
    /* Look for the last field among the subscript */
    list rsl = gen_nreverse(sl);
    type t = type_undefined;
    int i = 0;
    FOREACH(EXPRESSION, se, rsl) {
      if(field_reference_expression_p(se)) {
	entity f = reference_variable(expression_reference(se));
	t = entity_basic_concrete_type(f);
	break;
      }
      i++;
    }
    if(type_undefined_p(t)) {
      t = entity_basic_concrete_type(v);
      variable vt = type_variable(t);
      list dl = variable_dimensions(vt);
      int d = (int) gen_length(dl);
      int i = (int) gen_length(rsl);
      if(i<d)
	array_p = true;
    }
    else {
      if(i==0) { // FI: could be merged with the "else if" clause
	array_p = array_type_p(t);
      }
      else if(array_type_p(t)) {
	variable vt = type_variable(t);
	list dl = variable_dimensions(vt);
	int d = (int) gen_length(dl);
	if(i<d)
	  array_p = true;
      }
    }
    reference_indices(r) = gen_nreverse(rsl);
  }
  return array_p;
}

/* If this is an array reference, what is the type of the underlying array type?
 *
 * This information cannot be obtained by direct type information
 * because subarrays are typed as pointers to even smaller arrays.
 *
 * If it is not an array reference, the returned type is undefined.
 *
 * No new type is allocated.
 */
type points_to_array_reference_to_type(reference r)
{
  list sl = reference_indices(r);
  entity v = reference_variable(r);
  type t = type_undefined;

  if(ENDP(sl)) {
    t = entity_basic_concrete_type(v);
  }
  else {
    /* Look for the last field among the subscript */
    list rsl = gen_nreverse(sl);
    FOREACH(EXPRESSION, se, rsl) {
      if(field_reference_expression_p(se)) {
	entity f = reference_variable(expression_reference(se));
	t = entity_basic_concrete_type(f);
	break;
      }
    }
    if(type_undefined_p(t)) {
      t = entity_basic_concrete_type(v);
    }
    else {
      ;
    }
    reference_indices(r) = gen_nreverse(rsl);
  }
  return t;
}


/* Add a set of zero subscripts to a reference "r" by side effect.
 *
 * Used when a partial array reference must be converted into a
 * reference to the first array element (zero_p==true) or to any
 * element (zero_p==false).
 *
 * The difficulty lies with field subscripts...
 */
void complete_points_to_reference_with_fixed_subscripts(reference r, bool zero_p)
{
  type t = type_undefined;

  // FI: this assert makes sense within the ri-util framework but is
  // too strong for the kind of references used in effects-util
  // pips_assert("scalar type", ENDP(reference_indices(r)));

  /* Find the current number of effective subscripts: is there a field
     subscript somewhere? */
  list sl = reference_indices(r);
  entity v = reference_variable(r);
  list rsl = gen_nreverse(sl);
  int i = 0;
  bool field_found_p = false;

  FOREACH(EXPRESSION, se, rsl) {
    if(expression_field_p(se)) {
      reference fr = expression_reference(se);
      entity f = reference_variable(fr);
      t = entity_basic_concrete_type(f); 
      field_found_p = true;
      break;
    }
    i++;
  }

  if(!field_found_p)
    t = entity_basic_concrete_type(v);

  variable vt = type_variable(t);
  list dl = variable_dimensions(vt);
  int d = (int) gen_length(dl);

  pips_assert("Not Too many subscripts wrt the type.\n", i<=d);

  list nsl = NIL; // subscript list
  int j;
  for(j=i+1;j<=d;j++) {
    expression s = zero_p? int_to_expression(0) : make_unbounded_expression();
    // reference_indices(r) = CONS(EXPRESSION, s, reference_indices(r));
    nsl = CONS(EXPRESSION, s, nsl);
  }

  reference_indices(r) = gen_nreverse(rsl);
  reference_indices(r) = gen_nconc(reference_indices(r), nsl);
}

void complete_points_to_reference_with_zero_subscripts(reference r)
{
  complete_points_to_reference_with_fixed_subscripts(r, true);
}

bool cells_must_point_to_null_p(list cl)
{
  bool must_p = true;
  pips_assert("The input list is not empty", !ENDP(cl));
  FOREACH(CELL, c, cl) {
    if(!null_cell_p(c)) {
      must_p = false;
      break;
    }
  }
  return must_p;
}

bool cells_may_not_point_to_null_p(list cl)
{
  bool may_not_p = true;
  pips_assert("The input list is not empty", !ENDP(cl));
  FOREACH(CELL, c, cl) {
    if(null_cell_p(c) || nowhere_cell_p(c)) {
      may_not_p = false;
      break;
    }
  }
  return may_not_p;
}

/* Check if points-to arc "spt" belongs to points-to set "pts". */
bool arc_in_points_to_set_p(points_to spt, set pts)
{
  bool in_p = false;
  SET_FOREACH(points_to, pt, pts) {
    if(points_to_equal_p(spt, pt)) {
      in_p = true;
      break;
    }
  }
  return in_p;
}

/* Does cell "source" points toward a non null fully defined cell in
 * points-to set pts?
 *
 * The function name is not well chosen. Something like
 * cell_points_to_defined_cell_p()/
 */
bool cell_points_to_non_null_sink_in_set_p(cell source, set pts)
{
  bool non_null_p = false;
  SET_FOREACH(points_to, pt, pts) {
    cell pt_source = points_to_source(pt);
    if(cell_equal_p(pt_source, source)) {
      cell pt_sink = points_to_sink(pt);
      if(null_cell_p(pt_sink))
	;
      else if(nowhere_cell_p(pt_sink))
	;
      else {
	non_null_p = true;
	break;
      }
    }
  }
  return non_null_p;
}
bool cell_points_to_null_sink_in_set_p(cell source, set pts)
{
  bool null_p = false;
  SET_FOREACH(points_to, pt, pts) {
    cell pt_source = points_to_source(pt);
    if(cell_equal_p(pt_source, source)) {
      cell pt_sink = points_to_sink(pt);
      if(null_cell_p(pt_sink)) {
	null_p = true;
	break;
      }
    }
  }
  return null_p;
}

/* See if an arc like "spt" exists in set "in", regardless of its
 * approximation. If yes, returns the approximation of the arc found
 * in "in".
 *
 * See also arc_in_points_to_set_p(), which requires full identity
 */
bool similar_arc_in_points_to_set_p(points_to spt, set in, approximation * pa)
{
  bool in_p = false;
  cell spt_source = points_to_source(spt);
  cell spt_sink = points_to_sink(spt);
  SET_FOREACH(points_to, pt, in) {
    if(points_to_cell_equal_p(spt_source, points_to_source(pt))
       && points_to_cell_equal_p(spt_sink, points_to_sink(pt))) {
      *pa = points_to_approximation(pt);
      break;
    }
  }
  return in_p;
}
