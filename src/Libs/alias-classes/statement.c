/*

  $Id: points_to_analysis_general_algorithm.c 21101 2012-03-04 20:38:17Z amini $

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
This file contains functions used to compute points-to sets at statement level.
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
//#include "control.h"
#include "constants.h"
#include "misc.h"
//#include "parser_private.h"
//#include "syntax.h"
//#include "top-level.h"
//#include "text-util.h"
//#include "text.h"
#include "properties.h"
//#include "pipsmake.h"
//#include "semantics.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"
//#include "transformations.h"
//#include "preprocessor.h"
#include "pipsdbm.h"
#include "resources.h"
//#include "prettyprint.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"

/* See points_to_statement() */
pt_map statement_to_points_to(statement s, pt_map pt_in)
{
  pt_map pt_out;

  if(declaration_statement_p(s))
    pt_out = declaration_statement_to_points_to(s, pt_in);
  else {
    instruction i = statement_instruction(s);
    pt_out = instruction_to_points_to(i, pt_in);
  }

  return pt_out;
}

/* See points_to_init() */
pt_map declaration_statement_to_points_to(statement s, pt_map pt_in)
{
  pt_map pt_out;
  return pt_out;
}

/* See points_to_statement() */
pt_map instruction_to_points_to(instruction i, pt_map pt_in)
{
  pt_map pt_out;
  tag it = instruction_tag(i);
  switch(it) {
  case is_instruction_sequence: {
    sequence seq = instruction_sequence(i);
    pt_out = sequence_to_points_to(seq, pt_in);
    break;
  }
  case is_instruction_test: {
    test t = instruction_test(i);
    pt_out = test_to_points_to(t, pt_in);
    break;
  }
  case is_instruction_loop: {
    loop l = instruction_loop(i);
    pt_out = loop_to_points_to(l, pt_in);
    break;
  }
  case is_instruction_whileloop: {
    whileloop wl = instruction_whileloop(i);
    pt_out = whileloop_to_points_to(wl, pt_in);
    break;
  }
  case is_instruction_goto: {
    pips_internal_error("Go to instructions should have been removed "
			"before the analysis is started\n");
    break;
  }
  case is_instruction_call: {
    call c = instruction_call(i);
    pt_out = call_to_points_to(c, pt_in);
    break;
  }
  case is_instruction_unstructured: {
    unstructured u = instruction_unstructured(i);
    pt_out = unstructured_to_points_to(u, pt_in);
    break;
  }
  case is_instruction_multitest: {
    pips_internal_error("Not implemented\n");
    break;
  }
  case is_instruction_forloop: {
    forloop fl = instruction_forloop(i);
    pt_out = forloop_to_points_to(fl, pt_in);
    break;
  }
  case is_instruction_expression: {
    expression e = instruction_expression(i);
    pt_out = expression_to_points_to(e, pt_in);
    break;
  }
  default:
    ;
  }
  return pt_out;
}

pt_map sequence_to_points_to(sequence seq, pt_map pt_in)
{
  pt_map pt_out;
  return pt_out;
}

pt_map test_to_points_to(test t, pt_map pt_in)
{
  pt_map pt_out;
  return pt_out;
}

pt_map loop_to_points_to(loop l, pt_map pt_in)
{
  pt_map pt_out;
  return pt_out;
}

pt_map whileloop_to_points_to(whileloop wl, pt_map pt_in)
{
  pt_map pt_out;
  return pt_out;
}

pt_map unstructured_to_points_to(unstructured u, pt_map pt_in)
{
  pt_map pt_out;
  return pt_out;
}

pt_map multitest_to_points_to(multitest mt, pt_map pt_in)
{
  pt_map pt_out;
  pips_internal_error("Not implemented yet\n");
  return pt_out;
}

pt_map forloop_to_points_to(forloop fl, pt_map pt_in)
{
  pt_map pt_out;
  return pt_out;
}
