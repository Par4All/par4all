/*

  $Id:loop_nest_annotate.c 15433 2009-09-22 06:49:48Z creusillet $

  Copyright 1989-2010 MINES ParisTech

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

/* loop nests transformation phase for par4all :

   gpu_loop_nest_annotate takes a module that is of the form :


typedef float float_t;
void p4a_kernel_launcher_1(float_t save[501][501], float_t space[501][501])
{
  int i;
  int j;
  extern void p4a_kernel_wrapper_1(float_t save[501][501], float_t space[501][501], int i, int j);
kernel2:
  for(i = 0; i <= 498; i += 1)
    for(j = 0; j <= 498; j += 1)
      p4a_kernel_wrapper_1(save, space, i+1, j+1);
}

and transforms it into :

typedef float float_t;
void p4a_kernel_launcher_1(float_t save[501][501], float_t space[501][501])
{
   int i;
   int j;
   extern void p4a_kernel_wrapper_1(float_t save[501][501], float_t space[501][501], int i, int j);
kernel2:
   // Loop nest P4A begin, 2D(498,498)
   for(i = 0; i <= 498; i += 1)
      for(j = 0; j <= 498; j += 1)
   // Loop nest P4A end
   if (i <= 498 && j <= 498)
         p4a_kernel_wrapper_1(save, space, i+1, j+1);
}

for further generation of CUDA code.
 */

/* Ansi includes	*/
#include <stdio.h>
#include <string.h>

/* Newgen includes	*/
#include "genC.h"

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes	*/
#include "linear.h"
#include "ri.h"

#include "database.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "transformations.h"

#define COMMA         ","
#define OPENPAREN     "("
#define CLOSEPAREN    ")"


static list l_enclosing_loops = NIL;
static list l_number_iter_exp = NIL;
static int loop_nest_depth = 0;
static bool inner_reached = FALSE;

/* Statement stack to walk on control flow representation */
DEFINE_GLOBAL_STACK(p4a_private_current_stmt, statement)

static bool loop_push(loop l)
{
  l_enclosing_loops = gen_nconc(l_enclosing_loops, CONS(LOOP, l, NIL));
  loop_nest_depth++;
  return TRUE;
}

static void loop_annotate(loop l)
{
  /* the first time we enter this function is when we reach the innermost 
     loop nest level.
  */
  if (inner_reached == FALSE)
    {
      expression guard_exp = expression_undefined;
      statement guard_s = statement_undefined;

      /* We are at the innermost loop nest level */
      inner_reached = TRUE;

      /* first we add a guard to the loop body statement 
	 using the enclosing loops upperbounds.
      */
      FOREACH(LOOP, c_loop, l_enclosing_loops)
	{
	  entity c_index = loop_index(c_loop);
	  range c_range = loop_range(c_loop);
	  expression c_lower = range_lower(c_range);
	  expression c_upper = range_upper(c_range);
	  expression c_number_iter_exp = expression_undefined;
	  expression c_guard;
	  
	  c_guard = 
	    MakeBinaryCall(entity_intrinsic(C_LESS_OR_EQUAL_OPERATOR_NAME),
			   reference_to_expression(make_reference(c_index, 
								  NIL)), 
			   copy_expression(c_upper));

	  if (expression_undefined_p(guard_exp))
	    guard_exp = c_guard;
	  else
	    guard_exp = MakeBinaryCall(entity_intrinsic(C_AND_OPERATOR_NAME),
				       guard_exp, 
				       c_guard);
	  /* keep the number of iterations for the generation of 
	     the outermost comment */
	  c_number_iter_exp =  make_op_exp(MINUS_OPERATOR_NAME, 
					   c_upper, 
					   c_lower);
	  c_number_iter_exp =  make_op_exp(PLUS_OPERATOR_NAME, 
					   c_number_iter_exp,
					   make_integer_constant_expression(1));
	  l_number_iter_exp = gen_nconc(l_number_iter_exp, 
				     CONS(EXPRESSION, c_number_iter_exp, NIL));
	}
      
      pips_debug(2, "guard expression : %s\n",
		 words_to_string(words_expression(guard_exp)));

      guard_s = test_to_statement(make_test(guard_exp,
					    loop_body(l),
					    make_empty_block_statement()));
      /* Then we add the comment : // Loop nest P4A end */
      statement_comments(guard_s) = strdup("// Loop nest P4A end\n");

      loop_body(l) = guard_s;

    }
  
  /* we are now on our way back in the recusrsion; we do nothing, unless 
     we are at the uppermost level. Then we add the outermost comment : 
     // Loop nest P4A begin, xD(upper_bound,..)
  */
  if (gen_length(l_enclosing_loops) == 1)
    {
      statement current_stat = p4a_private_current_stmt_head();
      string outer_s = "// Loop nest P4A begin,";      

      outer_s = strdup(concatenate(outer_s, i2a(loop_nest_depth),
			     "D", OPENPAREN,  NULL));

      FOREACH(EXPRESSION, upper_exp, l_number_iter_exp)
	{
	  outer_s = strdup(concatenate(outer_s, 
				 words_to_string(words_expression(upper_exp)),
				 NULL));
	  loop_nest_depth --;	  
	  if (loop_nest_depth > 0)
	    outer_s = strdup(concatenate(outer_s, COMMA));
	}

      outer_s = strdup(concatenate(outer_s, CLOSEPAREN, "\n", NULL)); 
      statement_comments(current_stat) = outer_s;

      /* clean up things for another loop nest */
      inner_reached = FALSE;
      loop_nest_depth = 0;
      gen_free_list(l_number_iter_exp);
      l_number_iter_exp = NIL;
    }
  
  POP(l_enclosing_loops);
  return;
}


static bool stmt_push(statement s)
{
  pips_debug(1, "Entering statement %03zd :\n", statement_ordering(s));
  p4a_private_current_stmt_push(s);
  return(TRUE);
}

static void stmt_pop(statement s)
{
    p4a_private_current_stmt_pop();
    pips_debug(1, "End statement%03zd :\n", statement_ordering(s));

}


/**
 * annotates loop nests in the following way :
 *
 * for(i=0; i<=498; i++)
 *    for(j=0; j<=498; j++)
 *       foo();
 *
 * ==>
 *
 * // Loop nest P4A begin,2D(498,498)
 * for(i=0; i<=498; i++)
 *    for(j=0; j<=498; j++)
 *       // Loop nest P4A end
 *       if (i<=498&&j<=498)
 *       foo();
 *
 * for loops must have been transformed into loops.
 *
 * @param mod_name name of the  module
 *
 * @return true
 */
bool gpu_loop_nest_annotate(char *module_name)
{
    /* prelude */
  statement module_statement = 
    PIPS_PHASE_PRELUDE(module_name,
		       "P4A_LOOP_NEST_ANOTATE_DEBUG_LEVEL");
  
  
  make_p4a_private_current_stmt_stack();
  
  /* Compute the loops normalization of the module. */
  gen_multi_recurse(module_statement, 
		    statement_domain, stmt_push, stmt_pop,
		    loop_domain, loop_push, loop_annotate,
		    NULL); 
  free_p4a_private_current_stmt_stack();
  
  
  /* postlude */
  PIPS_PHASE_POSTLUDE(module_statement);
  
  return true;
}

