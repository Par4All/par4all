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
