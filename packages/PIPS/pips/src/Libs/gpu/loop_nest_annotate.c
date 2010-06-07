/*

  $Id:loop_nest_annotate.c 15433 2009-09-22 06:49:48Z creusillet $

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

/* Loop nests transformation phase for par4all :

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
}
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
#include "effects.h"

#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"
#include "transformations.h"

#define COMMA         ","
#define OPENPAREN     "("
#define CLOSEPAREN    ")"


/* In modern PIPS programming, all this should be passed through a context
   instead of having this global variable. This should allow some PIPS
   parallelization some days... */

typedef struct {
  list l_enclosing_loops;
  list l_number_iter_exp;
  int loop_nest_depth;
/* True only when we reached the inner annotated loop: */
  bool inner_reached;
} gpu_lna_context;
  

/* Push a loop that match the criterion for annotation */
static bool loop_push(loop l, gpu_lna_context * p)
{
  /* In the mode when we just annotate parallel outer loop nests, just
     stop when we encounter a sequential loop: */
  if (get_bool_property("GPU_LOOP_NEST_ANNOTATE_PARALLEL")
      && loop_sequential_p(l))
    // Stop recursion:
    return FALSE;

  p->l_enclosing_loops = gen_nconc(p->l_enclosing_loops, CONS(LOOP, l, NIL));
  p->loop_nest_depth++;
  // Go on recursing:
  return TRUE;
}


/* Do the real annotation work on previously marked loops bottom-up */
static void loop_annotate(loop l, gpu_lna_context * p)
{
  /* The first time we enter this function is when we reach the innermost
     loop nest level.
  */
  if (p->inner_reached == FALSE)
    {
      expression guard_exp = expression_undefined;
      statement guard_s = statement_undefined;

      /* We are at the innermost loop nest level */
      p->inner_reached = TRUE;

      /* First we add a guard to the loop body statement using the
	 enclosing loops upper bounds.
      */
      FOREACH(LOOP, c_loop, p->l_enclosing_loops)
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
	  /* Keep the number of iterations for the generation of the
	     outermost comment */
	  c_number_iter_exp =  make_op_exp(MINUS_OPERATOR_NAME,
					   c_upper,
					   c_lower);
	  c_number_iter_exp =  make_op_exp(PLUS_OPERATOR_NAME,
					   c_number_iter_exp,
					   make_integer_constant_expression(1));
	  p->l_number_iter_exp =
	    gen_nconc(p->l_number_iter_exp,
		      CONS(EXPRESSION, c_number_iter_exp, NIL));
	}

      /* FI: how about an expression_to_string() */
      pips_debug(2, "guard expression : %s\n",
		 words_to_string(words_expression(guard_exp, NIL)));

      guard_s = test_to_statement(make_test(guard_exp,
					    loop_body(l),
					    make_empty_block_statement()));
      /* Then we add the comment : // Loop nest P4A end */
      statement_comments(guard_s) = strdup("// Loop nest P4A end\n");

      loop_body(l) = guard_s;

    }


  /* We are now on our way back in the recursion; we do nothing, unless
     we are at the uppermost level. Then we add the outermost comment :
     // Loop nest P4A begin, xD(upper_bound,..)
  */
  if (gen_length(p->l_enclosing_loops) == 1)
    {
      statement current_stat = (statement) gen_get_ancestor(statement_domain, l);
#define LOOP_NEST_P4A_BEGIN "// Loop nest P4A begin,"
      string outer_s;
      (void) asprintf(&outer_s, LOOP_NEST_P4A_BEGIN "%dD" OPENPAREN , p->loop_nest_depth);

      FOREACH(EXPRESSION, upper_exp, p->l_number_iter_exp) {
        string buf;
	string buf1 = words_to_string(words_expression(upper_exp, NIL));
        (void) asprintf(&buf,"%s%s",outer_s,buf1);
        free(outer_s);
	free(buf1);
        outer_s=buf;
	p->loop_nest_depth --;
	if (p->loop_nest_depth > 0) {
          (void) asprintf(&buf,"%s"COMMA,outer_s);
          free(outer_s);
          outer_s=buf;
	}
      }

      (void) asprintf(&statement_comments(current_stat),"%s"CLOSEPAREN"\n",outer_s);
      free(outer_s);
      /* reset context*/
      p->inner_reached = FALSE;
      p->loop_nest_depth = 0; 
      gen_free_list(p->l_number_iter_exp); 
      p->l_number_iter_exp = NIL;     
    }

  POP(p->l_enclosing_loops);
  return;
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
  // Use this module name and this environment variable to set
  statement module_statement =
    PIPS_PHASE_PRELUDE(module_name, "P4A_LOOP_NEST_ANOTATE_DEBUG_LEVEL");

  /* Initialize context */
  gpu_lna_context c;
  c.l_enclosing_loops = NIL;
  c.l_number_iter_exp = NIL;
  c.loop_nest_depth = 0;
  c.inner_reached = FALSE;

  /* Annotate the loop nests of the module. */
  gen_context_recurse(module_statement, &c, loop_domain, loop_push, loop_annotate);

  /* Clean up things: (hasn't it been done previously in loop_annotate?) */
  gen_free_list(c.l_number_iter_exp);

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);
  // The macro above does a "return TRUE" indeed.
}
