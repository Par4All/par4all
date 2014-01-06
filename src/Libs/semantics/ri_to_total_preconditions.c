/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
 /* semantical analysis
  *
  * phasis 3: propagate total preconditions from statement to
  * sub-statement, starting from the module unique return statement. Total
  * preconditions are over-approximation of the theoretical total
  * preconditions, i.e. the conditions that must be met by the store
  * before a statement is executed to reach the end of the module
  * (intra-procedural) or the end of the program (interprocedural). Since
  * over-approximations are computed, the end of the module or of the
  * program cannot be reached if it is not met.
  *
  * For (simple) interprocedural analysis, this phasis should be performed
  * top-down on the call tree.
  *
  * Most functions are called xxx_to_total_precondition. They receive a
  * total postcondition as input and use a transformer to convert it into
  * a total precondition.
  *
  * Total preconditions are *NEVER* shared. Sharing would introduce desastrous
  * side effects when they are updated as for equivalenced variables and
  * would make freeing them impossible. Thus on a recursive path from
  * statement_to_total_precondition() to itself the precondition must have been
  * reallocated even when its value is not changed as between a block
  * precondition and the first statement of the block precondition. In the
  * same way statement_to_total_precondition() should never returned a
  * total_precondition aliased with its precondition argument. Somewhere
  * in the recursive call down the data structures towards
  * call_to_total_precondition() some allocation must take place even if the
  * statement as no effect on preconditions.
  *
  * Ambiguity: the type "transformer" is used to abstract statement effects
  * as well as effects combined from the beginning of the module to just
  * before the current statement (precondition) to just after the current
  * statement (total_precondition). This is because it was not realized that
  * preconditions could advantageously be seen as transformers of the initial
  * state when designing the ri.
  */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "transformer.h"

#include "semantics.h"

transformer statement_to_total_precondition(transformer, statement);

static transformer 
block_to_total_precondition(
			    transformer t_post,
			    list b)
{
  statement s;
  transformer t_pre = transformer_undefined;
  transformer s_post = transformer_undefined;
  list ls = b;

  pips_debug(8,"begin t_post=%p\n", t_post);

  if(ENDP(ls))
    t_pre = transformer_dup(t_post);
  else {
    list rls = gen_nreverse(ls);
    list crls = rls;

    s = STATEMENT(CAR(rls));
    s_post = statement_to_total_precondition(t_post, s);
    for (POP(crls) ; !ENDP(crls); POP(crls)) {
      s = STATEMENT(CAR(crls));
      t_pre = statement_to_total_precondition(s_post, s);
      s_post = t_pre;
    }
    ls = gen_nreverse(rls);

    /* t_pre is already associated with a statement */
    t_pre = transformer_dup(t_pre);
  }

  pips_debug(8,"post=%p end\n", t_pre);
  return t_pre;
}

static transformer 
unstructured_to_total_precondition(
    transformer pre,
    unstructured u,
    transformer tf)
{
  transformer post;
  control c;

  pips_assert("Not implemented yet", false);

  pips_debug(8,"begin\n");

  pips_assert("unstructured is defined", u!=unstructured_undefined);

  c = unstructured_control(u);
  if(control_predecessors(c) == NIL && control_successors(c) == NIL) {
    /* there is only one statement in u; no need for a fix-point */
    pips_debug(8,"unique node\n");
    /* FI: pre should not be duplicated because
     * statement_to_total_precondition() means that pre is not
     * going to be changed, just post produced.
     */
    post = statement_to_total_precondition(transformer_dup(pre),
					   control_statement(c));
  }
  else {
    /* Do not try anything clever! God knows what may happen in
       unstructured code. Total_Precondition post is not computed recursively
       from its components but directly derived from u's transformer.
       Preconditions associated to its components are then computed
       independently, hence the name unstructured_to_total_preconditionS
       instead of unstructured_to_total_precondition */
    /* propagate as precondition an invariant for the whole unstructured u
       assuming that all nodes in the CFG are fully connected, unless tf
       is not feasible because the unstructured is never exited or exited
       thru a direct or indirect call to STOP which invalidates the
       previous assumption. */
    transformer tf_u = transformer_undefined;
    transformer pre_u = transformer_undefined;

    pips_debug(8, "complex: based on transformer\n");
    if(transformer_empty_p(tf)) {
      tf_u = unstructured_to_flow_insensitive_transformer(u);
    }
    else {
      tf_u = tf;
    }
    pre_u = invariant_wrt_transformer(pre, tf_u);
    ifdebug(8) {
      debug(8,"unstructured_to_total_precondition",
	    "filtered precondition pre_u:\n");
      (void) print_transformer(pre_u) ;
    }
    post = unstructured_to_flow_sensitive_total_preconditions(pre_u, pre, u);
    pips_assert("A valid total_precondition is returned",
		!transformer_undefined_p(post));
    if(transformer_undefined_p(post)) {
      post = transformer_apply(transformer_dup(tf), pre);
    }
    transformer_free(pre_u);
  }

  pips_debug(8,"end\n");

  return post;
}

static transformer 
test_to_total_precondition(
    transformer t_post,
    test t,
    transformer tf,
    transformer context)
{
#   define DEBUG_TEST_TO_TOTAL_PRECONDITION 7
  expression c = test_condition(t);
  statement st = test_true(t);
  statement sf = test_false(t);
  transformer t_pre;

  pips_debug(DEBUG_TEST_TO_TOTAL_PRECONDITION, "begin\n");

  if(pips_flag_p(SEMANTICS_FLOW_SENSITIVE)) {
    transformer t_pret = statement_to_total_precondition(t_post, st);
    transformer t_pref = statement_to_total_precondition(t_post, sf);

    t_pret = transformer_add_domain_condition(t_pret, c, context,
					       true);
/*     t_pret = transformer_normalize(t_pret, 4); */
    t_pret = transformer_normalize(t_pret, 2);

    t_pref = transformer_add_domain_condition(t_pref, c, context,
					      false);
/*     transformer_normalize(t_pref, 4); */
    transformer_normalize(t_pref, 2);

    ifdebug(DEBUG_TEST_TO_TOTAL_PRECONDITION) {
      pips_debug(DEBUG_TEST_TO_TOTAL_PRECONDITION,"t_pret=%p\n",t_pret);
      (void) print_transformer(t_pret);
      pips_debug(DEBUG_TEST_TO_TOTAL_PRECONDITION,"t_pref=%p\n",t_pref);
      (void) print_transformer(t_pref);
    }

    t_pre = transformer_convex_hull(t_pret,t_pref);
  }
  else {
    (void) statement_to_total_precondition(t_post, st);
    (void) statement_to_total_precondition(t_post, sf);
    t_pre = transformer_apply(t_post, tf);
  }

  ifdebug(DEBUG_TEST_TO_TOTAL_PRECONDITION) {
    pips_debug(DEBUG_TEST_TO_TOTAL_PRECONDITION, "end post=\n");
    (void) print_transformer(t_pre);
  }

  return t_pre;
}

static transformer 
call_to_total_precondition(
    transformer t_post,
    call c,
    transformer tf)
{
  transformer t_pre = transformer_undefined;
  entity e = call_function(c);
  tag tt;

  pips_debug(8,"begin\n");

  switch (tt = value_tag(entity_initial(e))) {
  case is_value_intrinsic:
    t_pre = transformer_inverse_apply(tf, t_post);
    /* memory leak */
    t_pre = transformer_to_domain(t_pre);
    break;
  case is_value_code:
    t_pre = transformer_inverse_apply(tf, t_post);
    break;
  case is_value_symbolic:
  case is_value_constant:
    pips_internal_error("call to symbolic or constant %s", 
			entity_name(e));
    break;
  case is_value_unknown:
    pips_internal_error("unknown function %s", entity_name(e));
    break;
  default:
    pips_internal_error("unknown tag %d", tt);
  }

  pips_debug(8,"end\n");

  return t_pre;
}

static transformer 
instruction_to_total_precondition(
    transformer t_post,
    instruction i,
    transformer tf,
    transformer context)
{
  transformer t_pre = transformer_undefined;
  test t = test_undefined;
  loop l = loop_undefined;
  call c = call_undefined;

  pips_debug(9,"begin t_post=%p tf=%p\n", t_post, tf);

  switch(instruction_tag(i)) {
  case is_instruction_block:
    t_pre = block_to_total_precondition(t_post, instruction_block(i));
    break;
  case is_instruction_test:
    t = instruction_test(i);
    t_pre = test_to_total_precondition(t_post, t, tf, context);
    break;
  case is_instruction_loop:
    l = instruction_loop(i);
    t_pre = loop_to_total_precondition(t_post, l, tf, context);
    break;
  case is_instruction_whileloop: {
    whileloop wl = instruction_whileloop(i);
    evaluation ev = whileloop_evaluation(wl);

    if(evaluation_before_p(ev)) {
      t_pre = whileloop_to_total_precondition(t_post, wl, tf, context);
    }
    else {
      pips_user_error("Use property ??? to eliminate C repeat loops, "
		      "which are not handled directly\n");
    }
    break;
  }
  case is_instruction_forloop:
    pips_user_error("Use properties FOR_TO_DO_LOOP_IN_CONTROLIZER and"
		    "FOR_TO_WHILE_LOOP_IN_CONTROLIZER to eliminate C for loops, which are"
		    "not (yet) handled directly\n");
    //fl = instruction_forloop(i);
    //t_pre = forloop_to_total_precondition(t_post, fl, tf, context);
    break;
  case is_instruction_goto:
    pips_internal_error("unexpected goto in semantics analysis");
    /* never reached: post = pre; */
    break;
  case is_instruction_call:
    c = instruction_call(i);
    t_pre = call_to_total_precondition(t_post, c, tf);
    break;
  case is_instruction_unstructured:
    t_pre = unstructured_to_total_precondition(t_post, instruction_unstructured(i),
					      tf);
    break ;
  default:
    pips_internal_error("unexpected tag %d",
	       instruction_tag(i));
  }
  pips_debug(9,"resultat t_pre, %p:\n", t_pre);
  ifdebug(9) (void) print_transformer(t_pre);
  return t_pre;
}

transformer 
statement_to_total_precondition(
    transformer t_post,
    statement s)
{
  transformer t_pre = transformer_undefined;
  instruction i = statement_instruction(s);
  transformer tf = load_statement_transformer(s);
  /* Preconditions may be useful to deal with tests and loops and to find
     out if some control paths do not exist */
  /* transformer context = transformer_undefined; */
  transformer pre = load_statement_precondition(s);
  transformer context = transformer_range(pre);

  pips_debug(1,"begin\n");

  pips_assert("The statement total postcondition is defined", t_post != transformer_undefined);

  ifdebug(1) {
    _int so = statement_ordering(s);
    (void) fprintf(stderr, "statement %03td (%td,%td), total postcondition %p:\n",
		   statement_number(s), ORDERING_NUMBER(so),
		   ORDERING_STATEMENT(so), t_post);
    (void) print_transformer(t_post) ;
  }

  pips_assert("The statement transformer is defined", tf != transformer_undefined);
  ifdebug(1) {
    _int so = statement_ordering(s);
    pips_debug(9,"statement %03td (%td,%td), transformer %p:\n",
	       statement_number(s), ORDERING_NUMBER(so),
	       ORDERING_STATEMENT(so), tf);
    (void) print_transformer(tf) ;
  }

  if (!statement_reachable_p(s))
    {
      /* If the code is not reachable, thanks to STOP or GOTO, which
       * is a structural information, the total precondition is just empty.
       */

      t_pre = transformer_empty();
    }

  if (load_statement_total_precondition(s) == transformer_undefined) {
    list non_initial_values = list_undefined;

    t_pre = instruction_to_total_precondition(t_post, i, tf, context);

    /* keep only global initial scalar integer values;
       else, you might end up giving the same xxx#old name to
       two different local values (?) */
    non_initial_values =
      arguments_difference(transformer_arguments(t_pre),
			   get_module_global_arguments());

    MAPL(cv,
    {
      entity v = ENTITY(CAR(cv));
      ENTITY_(CAR(cv)) = entity_to_old_value(v);
    },
	 non_initial_values);

    /* add equivalence equalities */
    t_pre = tf_equivalence_equalities_add(t_pre);

/*     t_pre = transformer_normalize(t_pre, 4); */
    t_pre = transformer_normalize(t_pre, 2);

    if(!transformer_consistency_p(t_pre)) {
      _int so = statement_ordering(s);
      fprintf(stderr, "statement %03td (%td,%td), precondition %p end:\n",
	      statement_number(s), ORDERING_NUMBER(so),
	      ORDERING_STATEMENT(so), t_pre);
      print_transformer(t_pre);
      pips_internal_error("Non-consistent precondition after update");
    }

    t_pre = transformer_filter(t_pre, non_initial_values);

    /* store the total precondition in the ri */
    store_statement_total_precondition(s, t_pre);

    gen_free_list(non_initial_values);
  }
  else {
    _int so = statement_ordering(s);
    pips_debug(8, "total precondition already available:\n");
    (void) print_transformer(t_pre);
    pips_debug(8, "for statement %03td (%td,%td), total precondition %p end:\n",
	    statement_number(s), ORDERING_NUMBER(so),
	    ORDERING_STATEMENT(so), load_statement_total_precondition(s));
    pips_internal_error("total precondition already computed");
  }

  ifdebug(1) {
    _int so = statement_ordering(s);
    fprintf(stderr, "statement %03td (%td,%td), total precondition %p end:\n",
	    statement_number(s), ORDERING_NUMBER(so),
	    ORDERING_STATEMENT(so), load_statement_total_precondition(s));
    print_transformer(load_statement_total_precondition(s)) ;
  }

  ifdebug(1) {
    _int so = statement_ordering(s);
    fprintf(stderr, "statement %03td (%td,%td), total_precondition %p:\n",
	    statement_number(s), ORDERING_NUMBER(so),
	    ORDERING_STATEMENT(so), t_pre);
    print_transformer(t_pre) ;
  }

  free_transformer(context);

  pips_assert("unexpected sharing",t_post!=t_pre);

  pips_debug(1,"end\n");

  return t_pre;
}
