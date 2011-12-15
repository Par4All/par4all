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
 void p4a_kernel_wrapper_1(float_t save[501][501], float_t space[501][501], int i, int j);
 void p4a_kernel_launcher_1(float_t save[501][501], float_t space[501][501])
 {
 int i;
 int j;
 kernel2:
 for(i = 0; i <= 100; i += 1)
 for(j = 0; j <= 200; j += 1)
 p4a_kernel_wrapper_1(save, space, i+1, j+1);
 }

 and transforms it into :

 typedef float float_t;
 void p4a_kernel_wrapper_1(float_t save[501][501], float_t space[501][501], int i, int j);
 void p4a_kernel_launcher_1(float_t save[501][501], float_t space[501][501])
 {
 int i;
 int j;
 kernel2:
 // Loop nest P4A begin, 2D(200, 100)
 for(i = 0; i <= 100; i += 1)
 for(j = 0; j <= 200; j += 1)
 // Loop nest P4A end
 if (i <= 100 && j <= 200)
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
#include "effects-simple.h"

#define COMMA         ","
#define OPENPAREN     "("
#define CLOSEPAREN    ")"

/* In modern PIPS programming, all is passed through a context instead of
 having a global variable. This should allow some PIPS parallelization
 some days... :-) */

typedef struct {
  list l_enclosing_loops;
  list l_number_iter_exp;
  int max_loop_nest_depth;
  int loop_nest_depth;
  /* True only when we reach the inner annotated loop: */
  bool inner_reached;
  /* True if we only deal with parallel loop nests */
  bool gpu_loop_nest_annotate_parallel_p;
  /* The generation may fail because of an unhandled case for isntance */
  bool fail_p;
  loop inner_loop;
  expression guard_expression;
} gpu_lna_context;

/* Push a loop that matches the criterion for annotation */
static bool loop_push(loop l, gpu_lna_context * p) {
  if(p->max_loop_nest_depth == -1) {
    /* This is the first loop we met in this loop nest */
    if(p->gpu_loop_nest_annotate_parallel_p && !loop_parallel_p(l)) {
      return true;
    }

    /* Let's compute the loop_nest_depth */
    statement current_stat = (statement)gen_get_ancestor(statement_domain, l);
    p->max_loop_nest_depth
        = p->gpu_loop_nest_annotate_parallel_p ? depth_of_parallel_perfect_loop_nest(current_stat)
                                               : depth_of_perfect_loop_nest(current_stat);
  }

  if(p->loop_nest_depth >= p->max_loop_nest_depth)
    /* this loop does not belong to the perfectly nested loops */
    return false;
  else {
    p->l_enclosing_loops = gen_nconc(p->l_enclosing_loops, CONS(LOOP, l, NIL));
    p->loop_nest_depth++;
    // Go on recursing:
    return true;
  }
  return false;
}

/* Do the real annotation work on previously marked loops bottom-up */
static void loop_annotate(loop l, gpu_lna_context * p) {
  /* We have to select the operators that are different in C and FORTRAN */
  string and_op =
      (get_prettyprint_language_tag() == is_language_c) ? C_AND_OPERATOR_NAME
                                                        : AND_OPERATOR_NAME;
  string
      less_op =
          (get_prettyprint_language_tag() == is_language_c) ? C_LESS_OR_EQUAL_OPERATOR_NAME
                                                            : LESS_OR_EQUAL_OPERATOR_NAME;
  /* The first time we enter this function is when we reach the innermost
   loop nest level.
   */
  if(p->inner_loop == loop_undefined) {
    expression guard_exp = expression_undefined;

    /* We are at the innermost loop nest level */
    p->inner_loop = l;
    /* First we build the guard to be added to the loop body statement using the
     enclosing loops upper bounds.
     And we push on l_number_iter_exp an expression representing the
     number of iteration of each loop;
     currently, we do not check that variables modified inside loops
     are not used in loop bounds expressions.
     but we take care of loop indices used in deeper loop bounds.
     */FOREACH(LOOP, c_loop, p->l_enclosing_loops) {
      entity c_index = loop_index(c_loop);
      range c_range = loop_range(c_loop);
      expression c_lower = copy_expression(range_lower(c_range));
      expression c_upper = copy_expression(range_upper(c_range));
      expression c_inc = range_increment(c_range);
      expression c_number_iter_exp = expression_undefined;
      expression c_guard;

      if(expression_constant_p(c_inc) && expression_to_int(c_inc) == 1) {
        /* first check if the lower bound depend on enclosing loop indices */
        list l_eff_lower_bound = proper_effects_of_expression(c_lower);
        list l_eff_upper_bound = proper_effects_of_expression(c_upper);

        /* We have to clean the list of effect from any "preference" since the
         * next loop modify the references and can make our "preference" invalid
         */
        void remove_preferences(void * obj);
        {
          FOREACH(effect,e,l_eff_lower_bound) {
            remove_preferences(e);
          }
        }
        {
          FOREACH(effect,e,l_eff_upper_bound) {
            remove_preferences(e);
          }
        }

        FOREACH(LOOP, other_loop, p->l_enclosing_loops) {
          if(other_loop != l) {
            range range_other_loop = loop_range(other_loop);
            expression lower_other_loop = range_lower(range_other_loop);
            expression upper_other_loop = range_upper(range_other_loop);

            if(effects_read_variable_p(l_eff_lower_bound,
                                       loop_index(other_loop))) {
              expression new_lower_1 = c_lower;
              expression new_lower_2 = copy_expression(c_lower);
              replace_entity_by_expression(new_lower_1,
                                           loop_index(other_loop),
                                           lower_other_loop);
              (void)simplify_expression(&new_lower_1);
              replace_entity_by_expression(new_lower_2,
                                           loop_index(other_loop),
                                           upper_other_loop);

              (void)simplify_expression(&new_lower_2);
              c_lower = make_min_expression(new_lower_1,
                                            new_lower_2,
                                            get_prettyprint_language_tag());
            }
            if(effects_read_variable_p(l_eff_upper_bound,
                                       loop_index(other_loop))) {
              expression new_upper_1 = c_upper;
              expression new_upper_2 = copy_expression(c_upper);
              replace_entity_by_expression(new_upper_1,
                                           loop_index(other_loop),
                                           lower_other_loop);
              (void)simplify_expression(&new_upper_1);
              replace_entity_by_expression(new_upper_2,
                                           loop_index(other_loop),
                                           upper_other_loop);
              (void)simplify_expression(&new_upper_2);

              c_upper = make_max_expression(new_upper_1,
                                            new_upper_2,
                                            get_prettyprint_language_tag());
            }
          }
        }

        c_guard
            = MakeBinaryCall(entity_intrinsic(less_op),
                             reference_to_expression(make_reference(c_index,
                                                                    NIL)),
                             copy_expression(range_upper(c_range)));

        if(expression_undefined_p(guard_exp))
          guard_exp = c_guard;
        else
          guard_exp = MakeBinaryCall(entity_intrinsic(and_op),
                                     guard_exp,
                                     c_guard);

        /* FI: how about an expression_to_string() */pips_debug(2, "guard expression : %s\n",
            words_to_string(words_expression(guard_exp, NIL)));

        /* Keep the number of iterations for the generation of the
         outermost comment */
        c_number_iter_exp = make_op_exp(MINUS_OPERATOR_NAME, c_upper, c_lower);
        c_number_iter_exp = make_op_exp(PLUS_OPERATOR_NAME,
                                        c_number_iter_exp,
                                        int_to_expression(1));
        /* We will have deepest loop size first: */
        p->l_number_iter_exp = CONS(EXPRESSION, c_number_iter_exp,
            p->l_number_iter_exp);

      } else {
        p->fail_p = true;
        pips_user_warning("case not handled: loop increment is not 1.\n");
      }
    }
    if(!p->fail_p) {
      p->guard_expression = guard_exp;
    }

  }

  /* We are now on our way back in the recursion; we do nothing, unless
   we are at the uppermost level.
   */
  if(gen_length(p->l_enclosing_loops) == 1) {
    if(!p->fail_p)
    // if the process has succeeded, we add the outermost comment :
    // Loop nest P4A begin, xD(upper_bound,..) and the inner guard.
    {
      statement current_stat = (statement)gen_get_ancestor(statement_domain, l);
      // Then we add the comment such as: '// Loop nest P4A begin,3D(200, 100)'
      string outer_s;
      (void)asprintf(&outer_s,
                     "%s Loop nest P4A begin,%dD" OPENPAREN,
                     get_comment_sentinel(),
                     p->loop_nest_depth);

      bool first_iteration = true;
      /* Output inner dimension first: */FOREACH(EXPRESSION, upper_exp, p->l_number_iter_exp) {
        string buf;
        string buf1 = words_to_string(words_expression(upper_exp, NIL));
        if(first_iteration)
          /* Concatenate the dimension of the innermost loop: */
          (void)asprintf(&buf, "%s%s", outer_s, buf1);
        else
          /* Idem for other dimensions, but do not forget to insert the ', ' */
          (void)asprintf(&buf, "%s%s%s", outer_s, COMMA" ", buf1);
        free(outer_s);
        free(buf1);
        outer_s = buf;
        first_iteration = false;
      }
      (void)asprintf(&statement_comments(current_stat),
                     "%s"CLOSEPAREN"\n",
                     outer_s);
      free(outer_s);
      statement guard_s = test_to_statement(make_test(p->guard_expression,
              loop_body(p->inner_loop),
              make_empty_block_statement()));
      /* Then we add the comment : // Loop nest P4A end */statement_comments(guard_s)
          = strdup(concatenate(get_comment_sentinel(),
                               " Loop nest P4A end\n",
                               NULL));
      loop_body(p->inner_loop) = guard_s;

      /* reset context */
      p->loop_nest_depth = 0;
      p->max_loop_nest_depth = -1;
      gen_free_list(p->l_number_iter_exp);
      p->l_number_iter_exp = NIL;
      p->fail_p = false;
      p->inner_loop = loop_undefined;
      p->guard_expression = expression_undefined;
    }

    else
    // the process has failed: we clean everything and reset context
    {
      p->loop_nest_depth = 0;
      p->max_loop_nest_depth = -1;
      gen_full_free_list(p->l_number_iter_exp);
      p->l_number_iter_exp = NIL;
      p->fail_p = false;
      p->inner_loop = loop_undefined;
      if(!expression_undefined_p(p->guard_expression)) {
        free_expression(p->guard_expression);
        p->guard_expression = expression_undefined;
      }
    }
  }
  if(gen_length(p->l_enclosing_loops)) {
    POP(p->l_enclosing_loops);
  }
  return;
}

/**
 * annotates loop nests in the following way :
 *
 * for(i=0; i<=100; i++)
 *    for(j=0; j<=200; j++)
 *       foo();
 *
 * ==>
 *
 * // Loop nest P4A begin,2D(200, 100)
 * for(i=0; i<=100; i++)
 *    for(j=0; j<=200; j++)
 *       // Loop nest P4A end
 *       if (i<=100&&j<=200)
 *       foo();
 *
 * for loops must have been transformed into loops.
 *
 * @param mod_name name of the  module
 *
 * @return true
 */
bool gpu_loop_nest_annotate_on_statement(statement s) {
  /* Initialize context */
  gpu_lna_context c;
  c.l_enclosing_loops = NIL;
  c.l_number_iter_exp = NIL;
  c.max_loop_nest_depth = -1;
  c.loop_nest_depth = 0;
  c.inner_reached = false;
  c.gpu_loop_nest_annotate_parallel_p
      = get_bool_property("GPU_LOOP_NEST_ANNOTATE_PARALLEL");
  c.fail_p = false;
  c.inner_loop = loop_undefined;
  c.guard_expression = expression_undefined;

  /* Annotate the loop nests of the module. */
  gen_context_recurse(s, &c, loop_domain, loop_push, loop_annotate);

  /* Clean up things: (hasn't it been done previously in loop_annotate?) */
  gen_free_list(c.l_number_iter_exp);

  return true;
}

bool gpu_loop_nest_annotate(const char* module_name) {
  // Use this module name and this environment variable to set
  statement module_statement =
      PIPS_PHASE_PRELUDE(module_name, "P4A_LOOP_NEST_ANOTATE_DEBUG_LEVEL");

  gpu_loop_nest_annotate_on_statement(module_statement);

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement)
  ;
  // The macro above does a "return TRUE" indeed.
}

/**
 * Callback for gen_recurse
 * Parallelize perfectly nested loop nest, till we reach the magic comment
 *
 * FIXME : should detect the beginning sentinel, but since we use it in launcher,
 * it has no importance at that time
 */
static bool parallelize_annotated_loop_nest(statement s) {
  char **comment=NULL;
  if(statement_loop_p(s)) {
    execution_tag(loop_execution(statement_loop(s))) = is_execution_parallel;
    // Check the inner comment to find out the sentinel and stop recursion
    comment = find_first_statement_comment(loop_body(statement_loop(s)));
  } else {
    comment = find_first_statement_comment(s);
  }

  // Check sentinel
  if(comment  && !empty_comments_p(*comment) && NULL != strstr(*comment, "Loop nest P4A end")) {
    // stop recursion
    return false;
  }
  return true;
}

/** Parallelize the launcher based on loop nest annotate sentinels */
bool gpu_parallelize_annotated_loop_nest(const string mod_name) {
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(mod_name,
      "GPU_IFY_DEBUG_LEVEL");

  // Parallelize loops
  gen_recurse(module_statement,
      statement_domain, parallelize_annotated_loop_nest, gen_identity);

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement)
  ;
  // The macro above does a "return TRUE" indeed.
}


/**
 * Callback for gen_recurse
 * Remove annotation on a loop nest
 */
bool clear_annotated_loop_nest(statement s) {
  string comment = statement_comments(s);
  if(comment  && !empty_comments_p(comment)
      && (NULL != strstr(comment, "Loop nest P4A end")|| NULL != strstr(comment, "Loop nest P4A begin"))) {
    // clear the comment
    // We may instead filter out only the annotation inside the comment
    statement_comments(s) = string_undefined;
  }
  return true;
}

/** Remove all annotations on a loop nest */
bool gpu_clear_annotations_on_loop_nest(const string mod_name) {
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(mod_name,
      "GPU_IFY_DEBUG_LEVEL");

  // Parallelize loops
  gen_recurse(module_statement,
      statement_domain, clear_annotated_loop_nest, gen_identity);

  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement)
  ;
  // The macro above does a "return TRUE" indeed.
}
