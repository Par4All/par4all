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
  *
  * $Id$
  *
  * $Log: ri_to_total_preconditions.c,v $
  * Revision 1.1  2001/10/23 15:58:10  irigoin
  * Initial revision
  *
  * */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
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

/* another non recursive section used to filter out total preconditions */
static list module_global_arguments = NIL;

static list
get_module_global_arguments()
{
  return module_global_arguments;
}

static void 
set_module_global_arguments(args)
list args;
{
  module_global_arguments = args;
}
/* end of the non recursive section */

transformer statement_to_total_precondition(transformer, statement);

static transformer 
block_to_total_precondition(
			    transformer b_pre,
			    list b)
{
  statement s;
  transformer post;
  transformer s_pre = transformer_undefined;
  list ls = b;

  pips_debug(8,"begin pre=%x\n", b_pre);

  /* The first statement of the block must receive a copy
   * of the block precondition to avoid data sharing
   */

  if(ENDP(ls))
    /* to avoid accumulating equivalence equalities */
    post = transformer_dup(b_pre);
  else {
    s = STATEMENT(CAR(ls));
    s_pre = transformer_dup(b_pre);
    post = statement_to_total_precondition(s_pre, s);
    for (POP(ls) ; !ENDP(ls); POP(ls)) {
      s = STATEMENT(CAR(ls));
      /* the precondition has been allocated as post */
      s_pre = post;
      post = statement_to_total_precondition(s_pre, s);
    }
  }

  pips_debug(8,"post=%x end\n", post);
  return post;
}

static transformer 
unstructured_to_total_precondition(
    transformer pre,
    unstructured u,
    transformer tf)
{
  transformer post;
  control c;

  pips_debug(8,"begin\n");

  pips_assert("unstructured is deinfed", u!=unstructured_undefined);

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
    /* propagate as precondition an invariant for the whole
       unstructured u assuming that all nodes in the CFG are fully
       connected, unless tf is not feasible because the unstructured
       is never exited or exited thru a call to STOP which invalidates
       the previous assumption. */
    transformer tf_u = transformer_undefined;
    transformer pre_u = transformer_undefined;

    pips_debug(8, "complex: based on transformer\n");
    if(transformer_empty_p(tf)) {
      tf_u = unstructured_to_global_transformer(u);
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
    /* FI: I do not know if I should duplicate pre or not. */
    /* FI: well, dumdum, you should have duplicated tf! */
    /* FI: euh... why? According to comments about transformer_apply()
     * neither arguments are modified...
     */
    /* post = unstructured_to_total_preconditions(pre_u, pre, u); */
    post = unstructured_to_accurate_total_preconditions(pre_u, pre, u);
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
    transformer tf)
{
#   define DEBUG_TEST_TO_TOTAL_PRECONDITION 7
  expression e = test_condition(t);
  statement st = test_true(t);
  statement sf = test_false(t);
  transformer t_pre;

  pips_debug(DEBUG_TEST_TO_TOTAL_PRECONDITION,"begin\n");

  /* there are three levels of flow sensitivity and we have only a
     boolean flag! FI */

  /* test conditions are assumed to have no side effects; it might
     be somewhere in the standard since functions called in an expression e
     cannot (should not) modify any variable used in e */

  if(pips_flag_p(SEMANTICS_FLOW_SENSITIVE) /* && !transformer_identity_p(tf) */) {
    /* convex hull might avoided if it is not required or if it is certainly useless 
     * but test information should always be propagated 
     */
    transformer pret =
      precondition_add_condition_information(transformer_dup(pre),e, pre,
					     TRUE);
    transformer pref = transformer_undefined;

    transformer postt;
    transformer postf;

    /* "strong" transformer normalization to detect dead code generated by the
     * test condition
     */
    /* A normalization of degree 3 is fine */
    /* transformer_normalize(pret, 3); */
    transformer_normalize(pret, 7);

    /* FI, CA: the following "optimization" was added to avoid a call
     * to Chernikova convex hull that core dumps:-(. 8  September 1993
     *
     * From a theoretical point of view, pref could always be computed.
     *
     * FI: removed because it is mathematically wrong in many cases;
     * the negation of the test condition is lost! I keep the structure
     * just in case another core dump occurs (25 April 1997).
     */
    if(!empty_statement_p(sf)||TRUE) {
	  
      pref = precondition_add_condition_information(transformer_dup(pre),e,
						    pre, FALSE);
      /* transformer_normalize(pref, 3); */
      transformer_normalize(pref, 7);
    }
    else {
      /* do not try to compute a refined precondition for an empty block
       * keep the current precondition to store in the precondition statement mapping
       */
      pref = transformer_dup(pre);
    }

    ifdebug(DEBUG_TEST_TO_TOTAL_PRECONDITION) {
      debug(DEBUG_TEST_TO_TOTAL_PRECONDITION,"test_to_total_precondition","pret=\n");
      (void) print_transformer(pret);
      debug(DEBUG_TEST_TO_TOTAL_PRECONDITION,"test_to_total_precondition","pref=\n");
      (void) print_transformer(pref);
    }

    postt = statement_to_total_precondition(pret, st);
    postf = statement_to_total_precondition(pref, sf);
    post = transformer_convex_hull(postt, postf);
    transformer_free(postt);
    transformer_free(postf);
  }
  else {
    (void) statement_to_total_precondition(pre, st);
    (void) statement_to_total_precondition(pre, sf);
    post = transformer_apply(tf, pre);
  }

  ifdebug(DEBUG_TEST_TO_TOTAL_PRECONDITION) {
    debug(DEBUG_TEST_TO_TOTAL_PRECONDITION,"test_to_total_precondition",
	  "end post=\n");
    (void) print_transformer(post);
  }

  return post;
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
    /* there is room for improvement because assign is now the only 
       well handled intrinsic */
    pips_debug(5, "intrinsic function %s\n",
	       entity_name(e));
    if(get_bool_property("SEMANTICS_RECOMPUTE_EXPRESSION_TRANSFORMERS")
       && ENTITY_ASSIGN_P(call_function(c))) {
      entity f = call_function(c);
      list args = call_arguments(c);
      /* impredance problem: build an expression from call c */
      expression expr = make_expression(make_syntax(is_syntax_call, c),
					normalized_undefined);
      list ef = expression_to_proper_effects(expr);
      transformer pre_r = transformer_range(pre);
      transformer new_tf = intrinsic_to_transformer(f, args, pre_r, ef);

      post = transformer_apply(new_tf, pre);
      syntax_call(expression_syntax(expr)) = call_undefined;
      free_expression(expr);
      free_transformer(new_tf);
      free_transformer(pre_r);
    }
    else {
      post = transformer_apply(tf, pre);
    }
    /* propagate precondition pre as summary precondition 
       of user functions */
    /* FI: don't! Summary preconditions are computed independently*/
    /*
      if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      list args = call_arguments(c);
      expressions_to_summary_precondition(pre, args);
      }
    */
    break;
  case is_value_code:
    pips_debug(5, "external function %s\n", entity_name(e));
    if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      /*
	list args = call_arguments(c);

	transformer pre_callee = transformer_dup(pre);
	pre_callee = 
	add_formal_to_actual_bindings(c, pre_callee);
	add_module_call_site_precondition(e, pre_callee);
      */
      /*
	expressions_to_summary_precondition(pre, args);
      */
    }
    post = transformer_apply(tf, pre);
    break;
  case is_value_symbolic:
  case is_value_constant:
    pips_internal_error("call to symbolic or constant %s\n", 
			entity_name(e));
    break;
  case is_value_unknown:
    pips_internal_error("unknown function %s\n", entity_name(e));
    break;
  default:
    pips_internal_error("unknown tag %d\n", tt);
  }

  pips_debug(8,"end\n");

  return post;
}

static transformer 
instruction_to_total_precondition(
    transformer t_post,
    instruction i,
    transformer tf)
{
  transformer t_pre = transformer_undefined;
  test t;
  loop l;
  whileloop wl;
  call c;

  pips_debug(9,"begin pre=%p tf=%p\n", pre, tf);

  switch(instruction_tag(i)) {
  case is_instruction_block:
    t_pre = block_to_total_precondition(t_post, instruction_block(i));
    break;
  case is_instruction_test:
    t = instruction_test(i);
    t_pre = test_to_total_precondition(t_post, t, tf);
    break;
  case is_instruction_loop:
    l = instruction_loop(i);
    t_pre = loop_to_total_precondition(t_post, l, tf);
    break;
  case is_instruction_whileloop:
    wl = instruction_whileloop(i);
    t_pre = whileloop_to_total_precondition(t_post, wl, tf);
    break;
  case is_instruction_goto:
    pips_error("instruction_to_total_precondition",
	       "unexpected goto in semantics analysis");
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
    pips_error("instruction_to_total_precondition","unexpected tag %d\n",
	       instruction_tag(i));
  }
  pips_debug(9,"resultat post, %p:\n", t_pre);
  ifdebug(9) (void) print_transformer(t_pre);
  return post;
}

transformer 
statement_to_total_precondition(
    transformer t_post,
    statement s)
{
  transformer t_pre = transformer_undefined;
  instruction i = statement_instruction(s);
  transformer tf = load_statement_transformer(s);

  /* ACHTUNG! "pre" is likely to be misused! FI, Sept. 3, 1990 */

  debug(1,"statement_to_total_precondition","begin\n");

  pips_assert("The statement total postcondition is defined", t_post != transformer_undefined);

  ifdebug(1) {
    int so = statement_ordering(s);
    (void) fprintf(stderr, "statement %03d (%d,%d), precondition %p:\n",
		   statement_number(s), ORDERING_NUMBER(so),
		   ORDERING_STATEMENT(so), t_post);
    (void) print_transformer(t_post) ;
  }

  pips_assert("The statement transformer is defined", tf != transformer_undefined);
  ifdebug(1) {
    int so = statement_ordering(s);
    (void) fprintf(stderr, "statement %03d (%d,%d), transformer %p:\n",
		   statement_number(s), ORDERING_NUMBER(so),
		   ORDERING_STATEMENT(so), tf);
    (void) print_transformer(tf) ;
  }

  if (!statement_reachable_p(s))
    {
      /* FC: if the code is not reachable (thanks to STOP or GOTO), which
       * is a structural information, the total precondition is just identity.
       */

      t_pre = transformer_identity();
    }

  if (load_statement_total_precondition(s) == transformer_undefined) {
    /* keep only global initial scalar integer values;
       else, you might end up giving the same xxx#old name to
       two different local values */
    list non_initial_values =
      arguments_difference(transformer_arguments(pre),
			   get_module_global_arguments());

    MAPL(cv,
    {
      entity v = ENTITY(CAR(cv));
      ENTITY(CAR(cv)) = entity_to_old_value(v);
    },
	 non_initial_values);

    t_pre = instruction_to_total_precondition(t_post, i, tf);

    /* add equivalence equalities */
    t_pre = tf_equivalence_equalities_add(t_pre);

    t_pre = transformer_normalize(t_pre, 4);

    if(!transformer_consistency_p(t_pre)) {
      int so = statement_ordering(s);
      fprintf(stderr, "statement %03d (%d,%d), precondition %p end:\n",
	      statement_number(s), ORDERING_NUMBER(so),
	      ORDERING_STATEMENT(so), t_pre);
      print_transformer(t_pre);
      pips_internal_error("Non-consistent precondition after update\n");
    }

    /* store the precondition in the ri */
    store_statement_total_precondition(s,
				       transformer_filter(t_pre,
							  non_initial_values));

    gen_free_list(non_initial_values);
  }
  else {
    pips_debug(8,"total precondition already available");
    (void) print_transformer(t_pre);
    pips_error("statement_to_total_precondition",
	       "precondition already computed\n");
  }

  ifdebug(1) {
    int so = statement_ordering(s);
    fprintf(stderr, "statement %03d (%d,%d), total precondition %p end:\n",
	    statement_number(s), ORDERING_NUMBER(so),
	    ORDERING_STATEMENT(so), load_statement_total_precondition(s));
    print_transformer(load_statement_total_precondition(s)) ;
  }

  ifdebug(1) {
    int so = statement_ordering(s);
    fprintf(stderr, "statement %03d (%d,%d), total_precondition %p:\n",
	    statement_number(s), ORDERING_NUMBER(so),
	    ORDERING_STATEMENT(so), t_pre);
    print_transformer(t_pre) ;
  }

  pips_assert("unexpected sharing",t_post!=t_pre);

  pips_debug(1,"end\n");

  return t_pre;
}
