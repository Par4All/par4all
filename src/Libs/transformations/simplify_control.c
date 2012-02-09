/*

  $Id$

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
/*
  Dead loop elimination.
  Ronan Keryell, 12/1993 -> 1995.
  one trip loops fixed, FC 08/01/1998
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "properties.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "transformer.h"
#include "semantics.h"
#include "control.h"
#include "callgraph.h"

#include "transformations.h"

/* To avoid calling unspaghettify() if unnecessary: */
static bool some_unstructured_ifs_have_been_changed;


/**************************************************************** STATISTICS */

static int dead_code_if_removed;
static int dead_code_if_replaced_by_its_effect;
static int dead_code_if_false_branch_removed;
static int dead_code_if_true_branch_removed;
static int dead_code_loop_removed;
static int dead_code_loop_executed_once;
static int dead_code_statement_removed;
static int dead_code_unstructured_if_removed;
static int dead_code_unstructured_if_replaced_by_its_effect;
static int dead_code_unstructured_if_false_branch_removed;
static int dead_code_unstructured_if_true_branch_removed;
static void suppress_dead_code_statement(statement);

static transformer (*get_statement_precondition)(statement);

static void
initialize_dead_code_statistics()
{
    dead_code_if_removed = 0;
    dead_code_if_replaced_by_its_effect = 0;
    dead_code_if_false_branch_removed = 0;
    dead_code_if_true_branch_removed = 0;
    dead_code_loop_removed = 0;
    dead_code_loop_executed_once = 0;
    dead_code_statement_removed = 0;
    dead_code_unstructured_if_removed = 0;
    dead_code_unstructured_if_replaced_by_its_effect = 0;
    dead_code_unstructured_if_false_branch_removed = 0;
    dead_code_unstructured_if_true_branch_removed = 0;
    initialize_clean_up_sequences_statistics();
}

static void
display_dead_code_statistics()
{
    if (get_bool_property("DEAD_CODE_DISPLAY_STATISTICS")) {
	int elimination_count = dead_code_statement_removed
	    + dead_code_loop_removed
		+ dead_code_loop_executed_once
		    + dead_code_if_removed
			+ dead_code_if_replaced_by_its_effect;
	elimination_count += dead_code_unstructured_if_removed
	    + dead_code_unstructured_if_replaced_by_its_effect;

	if (elimination_count > 0)
	{
	  user_log("* %d dead code part%s %s been discarded. *\n",
		   elimination_count,
		   elimination_count > 1 ? "s" : "",
		   elimination_count > 1 ? "have" : "has");

	  user_log("Statements removed (directly dead): %d\n",
		   dead_code_statement_removed);

	  user_log("Loops: loops removed: %d, loops executed only once: %d\n",
		   dead_code_loop_removed, dead_code_loop_executed_once);

	  user_log("Structured tests: \"if\" removed: %d, "
		   "\"if\" replaced by side effects: %d\n"
		   "\t(\"then\" removed: %d, "
		   "\"else\" removed: %d)\n",
		   dead_code_if_removed, dead_code_if_replaced_by_its_effect,
		   dead_code_if_true_branch_removed,
		   dead_code_if_false_branch_removed);

	  user_log("Unstructured tests: \"if\" removed: %d, "
		   "\"if\" replaced by side effects: %d\n"
		   "\t(unstructured \"then\" removed: %d, "
		   "unstructured \"else\" removed: %d)\n",
		   dead_code_unstructured_if_removed,
		   dead_code_unstructured_if_replaced_by_its_effect,
		   dead_code_unstructured_if_true_branch_removed,
		   dead_code_unstructured_if_false_branch_removed);
	  /* Display also the statistics about clean_up_sequences
	     that is called in suppress_dead_code: */
	  display_clean_up_sequences_statistics();
	}
    }
}

/********************************************************************* DEBUG */

static void stdebug(int dl, string msg, statement s)
{
  ifdebug(dl) {
    pips_debug(dl, "statement %p: %s\n", s, msg);
    if (s) {
      print_statement(s);
    }
  }
  pips_assert("The statement is consistent",
	      statement_consistent_p(s));
}

/* Give an information on the liveness of the 2 if's branches: */
static dead_test
dead_test_filter(statement st_true, statement st_false)
{
  pips_debug(5, "Begin\n");

  stdebug(9, "dead_test_filter: then branch", st_true);
  stdebug(9, "dead_test_filter: else branch", st_false);

  ifdebug(8)
    {
      transformer pretrue = get_statement_precondition(st_true);
      transformer prefalse = get_statement_precondition(st_false);
      fprintf(stderr,"NN true and false branches");
      sc_fprint(stderr,
		predicate_system(transformer_relation(pretrue)),
		(char* (*)(Variable)) entity_local_name);
      sc_fprint(stderr,
		predicate_system(transformer_relation(prefalse)),
		(char* (*)(Variable)) entity_local_name);
    }

  if (get_statement_precondition==load_statement_precondition
      && !statement_strongly_feasible_p(st_true)) {
    pips_debug(5, "End: then_is_dead\n");
    return then_is_dead;
  }

  if (get_statement_precondition==load_statement_precondition
      && !statement_strongly_feasible_p(st_false)) {
    pips_debug(5, "End: else_is_dead\n");
    return else_is_dead;
  }

  pips_debug(5, "End: nothing_about_test\n");
  return nothing_about_test;
}

/* Replace an instruction by an empty one. If there is a label on the
   statement, put it on a continue to be coherent with the PIPS RI. */
static bool discard_statement_and_save_label_and_comment(statement s)
{

  /* NN -> Bug found : if we have two loops with the same label
     such as :

     DO 100 I=1,N
     DO 100 J=1,M
     ......

     100 CONTINUE

     and the inner loop is a dead statement, there is an error when
     compiling the generated file Fortran.  Because the label of the
     last statement in the inner loop might be used by an outer loop
     and, in doubt, should be preserved.

     SOLUTION : like in full_loop_unroll()*/

  if (statement_loop_p(s)) {
    loop l = instruction_loop(statement_instruction(s));
    entity flbl = find_final_statement_label(loop_body(l));

    if(!entity_empty_label_p(flbl)) {

      instruction block =  make_instruction_block(NIL);
      statement stmt = make_continue_statement(flbl);
      instruction_block(block)= gen_nconc(instruction_block(block),
					  CONS(STATEMENT, stmt, NIL ));
      free_instruction(statement_instruction(s));
      /* And put a new empty one: */
      statement_instruction(s) = block;
      /* Since the RI need to have no label on instruction block: */

      fix_sequence_statement_attributes(s);

    }

  }
   /* FI: Why do we want to avoid eliminating declarations? If they are
   * dead code, they should not be used anywhere? This is only useful
   * if the elimination decision is taken according to effects...
   */
  else if(declaration_statement_p(s)) {
    ; /* do not modify the declaration statement... in case it has no
	 memory effect */
  }
  /* "this avoids removing declarations" (previous comment)
   *
   * FI: If a whole block has no effect, its internal declarations are
   * useless too. But if it is the body of a function returning a
   * value, the return statement should be preserved for syntactic
   * correctness. This is not done here. The main test should be
   * if(statement_block_p(s)).
   *
   * Maybe also, we might want to preserve some comments. See
   * Transformations/no_effect_statement_00
   */
  else if(false && !ENDP(statement_declarations(s))) {
    if(statement_block_p(s)) {
      FOREACH(STATEMENT,st,statement_block(s))
	if(!declaration_statement_p(s))
	  discard_statement_and_save_label_and_comment(st);
    }
  }
  else {
    /* Discard the old instruction: */
    free_instruction(statement_instruction(s));
    /* And put a new empty one: */
    statement_instruction(s) = make_instruction_block(NIL);

    /* Since the RI need to have no label on instruction block: */
    fix_sequence_statement_attributes(s);
  }
  return false;
}


/* Use the precondition to know wether a loop is never executed
 *
 * The accuracy (and, hence, duration) of the test depends on
 * the precondition normalization. There is a trade-off between
 * the time spent in dead code elimination and the time spent
 * in precondition computation.
 */
static bool
dead_loop_p(loop l)
{
  statement s;
  bool feasible_p = true;
  intptr_t c = 0;

  if(range_count(loop_range(l), &c)) {
    feasible_p = (c>0);
  }
  else {
    s = loop_body(l);
    /* feasible_p = statement_feasible_p(s); */
    if(get_statement_precondition==load_statement_precondition)
      feasible_p = statement_strongly_feasible_p(s);
  }
  return !feasible_p;
}


/* Replace a loop statement with the statement inside the loop. */
static void
remove_loop_statement(statement s, instruction i, loop l)
{
  range lr = loop_range(l);
  expression rl = range_lower(lr);
  expression rincr = range_increment(lr);
  expression init_val = copy_expression(rl);
  expression last_val = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
				       copy_expression(init_val),
				       copy_expression(rincr));
  expression index;
  statement as1,as2;

  /* Assume here that index is a scalar variable... :-) */
  pips_assert("dead_statement_filter", entity_scalar_p(loop_index(l)));

  index = make_factor_expression(1, loop_index(l));

  /* If we can, we will replace the use of the loop index in the body with
   * the lower bound, we handle only pure simple scalar expression cases
   */
  if( (expression_integer_constant_p(rl) && expression_to_int(rl) >= 0)
      || (expression_reference_p(rl)
          && entity_scalar_p(reference_variable(expression_reference(rl))))) {

    entity up_bound;
    if(expression_integer_constant_p(rl)) {
      up_bound = int_to_entity(expression_to_int(rl));
    } else {
      up_bound = reference_variable(expression_reference(rl));
    }
    replace_entity(loop_body(l),loop_index(l),up_bound);

    // A property can prevent the assignment of upper bound to loop index
    if(get_bool_property("SIMPLIFY_CONTROL_DIRECTLY_PRIVATE_LOOP_INDICES")) {
      statement_instruction(s) =
          make_instruction_block(make_statement_list(loop_body(l)));
    } else {
      statement_instruction(s) =
          make_instruction_block(
              make_statement_list(loop_body(l),
                                  as2=make_assign_statement(copy_expression(index),
                                                            last_val))
          );
    }
  } else {
    /* Standard case, the loop header is removed, but we have to assign value to
     * the loop index:
     * for(i=low;i<upper;i+=inc)
     * =>
     * i=low;
     * // body...
     * i=low+inc;
     */
    as1 = make_assign_statement(index, init_val);
    as2 = make_assign_statement(copy_expression(index),last_val);
    statement_instruction(s) =
        make_instruction_block(make_statement_list(as1,
                                                   loop_body(l),
                                                   as2)
        );
    statement_label(as1) = statement_label(s);
  }
  statement_label(s) = entity_empty_label();
  fix_sequence_statement_attributes(s);

  /* fix some extra attributes */
  store_cumulated_rw_effects_list(as1,gen_full_copy_list(load_cumulated_rw_effects_list(s)));
  store_cumulated_rw_effects_list(as2,gen_full_copy_list(load_cumulated_rw_effects_list(s)));

  stdebug(4, "remove_loop_statement", s);

  loop_body(l) = make_empty_statement();
  free_instruction(i);
}

/* true if do i = x, x or equivalent.
 */
static bool loop_executed_once_p(statement s, loop l)
{
  range rg;
  expression m1, m2, m3;
  normalized n_m1, n_m2, n_m3;
  transformer pre;
  Psysteme precondition_ps;
  Pvecteur pv3;
  Pcontrainte pc3;
  bool m3_negatif = false, m3_positif = false, retour;
  /* Automatic variables read in a CATCH block need to be declared volatile as
   * specified by the documentation*/
  Psysteme volatile ps;

  retour = false;
  rg = loop_range(l);
  m1 = range_lower(rg);
  m2 = range_upper(rg);

  /* m1 == m2
   */
  /* Not necessarily true with side effects: DO i = inc(n), inc(n) */
  if (expression_equal_p(m1, m2))
    return true;

  pre = get_statement_precondition(s);
  precondition_ps = predicate_system(transformer_relation(pre));

  m3 = range_increment(rg);
  n_m1 = NORMALIZE_EXPRESSION(m1);
  n_m2 = NORMALIZE_EXPRESSION(m2);
  n_m3 = NORMALIZE_EXPRESSION(m3);

  if (normalized_linear_p(n_m1) && normalized_linear_p(n_m2))
  {
    /* m1 - m2 == 0 redundant ?
     */
    Pcontrainte eq = contrainte_make(vect_substract(normalized_linear(n_m1),
						    normalized_linear(n_m2)));

    if (eq_redund_with_sc_p(precondition_ps, eq))
      retour = true;

    contrainte_free(eq);

    if (retour) return true;
  }
  if (normalized_linear_p(n_m3)) {
    /* Teste le signe de l'incr�ment en fonction des pr�conditions : */
    pv3 = vect_dup(normalized_linear(n_m3));
    pc3 = contrainte_make(pv3);
    ps = sc_dup(precondition_ps);
    sc_add_ineg(ps, pc3);
    CATCH(overflow_error) {
      sc_rm(ps);
      return false;
    }
    TRY {
      m3_negatif = sc_rational_feasibility_ofl_ctrl(ps,FWD_OFL_CTRL,true);
      (void) vect_chg_sgn(pv3);
      m3_positif = sc_rational_feasibility_ofl_ctrl(ps,FWD_OFL_CTRL,true);
      UNCATCH(overflow_error);
    }
    pips_debug(2, "loop_increment_value positif = %d, negatif = %d\n",
	       m3_positif, m3_negatif);

    /* Vire aussi pv3 & pc3 : */
    sc_rm(ps);
  }
  if ((m3_positif ^ m3_negatif) && normalized_linear_p(n_m3) &&
      normalized_linear_p(n_m1) && normalized_linear_p(n_m2))
  {
    /* Si l'incr�ment a un signe � connu � et diff�rent de 0 et que
       les bornes sont connues : */
    Pvecteur pv1, pv2, pv3, pvx, pv;
    Pcontrainte ca, cb;

    pv1 = normalized_linear(n_m1);
    pv2 = normalized_linear(n_m2);
    pv3 = normalized_linear(n_m3);

    /* pv = m1 - m2, i.e. m1 - m2 <= 0 */
    pv = vect_substract(pv1, pv2);

    /* pvx = m1 - m2 + m3, i.e. m1 + m3 -m2 <=0 */
    pvx = vect_add(pv, pv3);

    if (m3_positif) {
      /* L'incr�ment est positif. */
       (void) vect_chg_sgn(pvx);
      /* m1 - m2 <= 0 && m2 - m1 - m3 <= -1 */
    }

    if (m3_negatif) {
      (void) vect_chg_sgn(pv);
      /* m2 - m1 >= 0 && -m2 + m1 + m3 <= -1 */
    }

    vect_add_elem(&pvx, TCST, VALUE_ONE);

    ca = contrainte_make(pvx);
    cb = contrainte_make(pv);

    /* ??? on overflows, should assume false...
     */
    retour = ineq_redund_with_sc_p(precondition_ps, ca) &&
             ineq_redund_with_sc_p(precondition_ps, cb);

    /* Vire du m�me coup pv et pvx : */
    contrainte_free(ca), contrainte_free(cb);
  }

  return retour;
}

/* Remplace une boucle vide par la seule initialisation de l'indice : */
static bool remove_dead_loop(statement s, instruction i, loop l)
{
  expression index, rl;
  range lr;
  statement as;
  expression init_val;
  instruction block =  make_instruction_block(NIL);
  entity flbl;

  /* On va remplacer la boucle par l'initialisation de l'indice a`
     sa valeur initiale seulement. */

  init_val = copy_expression(rl = range_lower(lr = loop_range(l)));
  /*pips_assert("remove_dead_loop", gen_defined_p(init_val));*/
  /*expression init_val = copy_expression(range_lower(loop_range(l)));*/

  /* Assume here that index is a scalar variable... :-) */
  pips_assert("remove_dead_loop", entity_scalar_p(loop_index(l)));

  index = make_factor_expression(1, loop_index(l));

  /* NN -> Bug found : if we have two loops with the same label
     such as :

     DO 100 I=1,N
        DO 100 J=1,M
     ......

     100 CONTINUE

     and the inner loop is a dead statement, there is an error when
     compiling the generated file Fortran.  Because the label of the
     last statement in the inner loop might be used by an outer loop
     and, in doubt, should be preserved.

     SOLUTION : like in full_loop_unroll()*/

  /*  *****OLD CODE***************
      statement_instruction(s) =
      make_instruction_block(CONS(STATEMENT,
      as = make_assign_statement(index , init_val),
      NIL));
      statement_label(as) = statement_label(s);
      statement_label(s) = entity_empty_label();

      fix_sequence_statement_attributes(s);*/

  /*  *****NEW CODE***************/


  instruction_block(block)=
    gen_nconc(instruction_block(block),
	      CONS(STATEMENT,
		   as = make_assign_statement(index , init_val),
		   NIL ));

  flbl = find_final_statement_label(loop_body(l));

  if(!entity_empty_label_p(flbl)) {
    statement stmt = make_continue_statement(flbl);
    instruction_block(block)= gen_nconc(instruction_block(block),
					CONS(STATEMENT, stmt, NIL ));
  }

  statement_instruction(s) = block;
  /* Since the RI need to have no label on instruction block: */
  statement_label(as) = statement_label(s);
  statement_label(s) = entity_empty_label();
  fix_sequence_statement_attributes(s);

  stdebug(9, "remove_dead_loop: New value of statement", s);

  free_instruction(i);
  return false;
}
/* Return true if a statement has at least one write effect in the
   effects list. */
static bool statement_write_effect_p(statement s)
{
  bool write_effect_found = false;
  list effects_list = load_proper_rw_effects_list(s);

  FOREACH(effect, an_effect, effects_list)
  {
    if (action_write_p(effect_action(an_effect))) {
      write_effect_found = true;
      break;
    }
  };

  return write_effect_found;
}


/* Remove an IF(x) statement (replace s with an empty statement) if x
   has no write proper effect. If x has a write effect, replace s with a
   statement as bool_var = x: (he', a french joke !)
   this_test_is_unstructured_p is a hint for the statistics.
   true means that you assert that the test is unstructured.
 */
static void remove_if_statement_according_to_write_effects
(statement s, bool this_test_is_unstructured_p)
{
  instruction i = statement_instruction(s);

  pips_assert("statement is consistent at entry", s);

  if (statement_write_effect_p(s)) {
    /* There is a write effect, so we cannot discard the IF
       expression. Keep it in a temporarily variable: */
    entity temp_var =
      make_new_scalar_variable(get_current_module_entity(),
			       MakeBasic(is_basic_logical));
    /* Create the bool_var = x: */
    statement_instruction(s) =
      make_assign_instruction(entity_to_expression(temp_var),
			      test_condition(instruction_test(i)));
    test_condition(instruction_test(i)) = expression_undefined;

    if (this_test_is_unstructured_p)
      dead_code_unstructured_if_replaced_by_its_effect++;
    else
      dead_code_if_replaced_by_its_effect++;

    pips_assert("statement is consistent after partial dead code removeal", s);
  }
  else {
    /* There is no write effect, the statement can be discarded: */
    statement_instruction(s) = make_instruction_block(NIL);
    fix_sequence_statement_attributes(s);

    if (this_test_is_unstructured_p)
      dead_code_unstructured_if_removed++;
    else
      dead_code_if_removed++;

  pips_debug(8, "let's use some heap\n");
    pips_assert("statement is consistent after dead code removal", s);
  }

  pips_debug(8, "let's use some heap\n");

  pips_assert("statement is consistent at exit", s);

  /* Discard the IF: */
  /* FI: I do not understand why, but this free breaks the
     unstructured containing s (case "no write effect". */
  /* For unknown reasons, the preconditions for the true and false
     branches are recomputed... because they are not reachable from
     the controls... however, statements are reachable from controls
     and statements are linked to preconditions... */
  // free_instruction(i);

  pips_debug(8, "let's use some heap\n");
  pips_assert("statement is consistent at exit", s);
}


static bool dead_deal_with_test(statement s,
				test t)
{
  statement st_true = test_true(t);
  statement st_false = test_false(t);
  expression c = test_condition(t);
  enum dead_test what_is_dead = nothing_about_test;

  if(true_expression_p(c))
    what_is_dead = else_is_dead;
  else if(false_expression_p(c))
    what_is_dead = then_is_dead;
  else
    what_is_dead = dead_test_filter(st_true, st_false);

  switch (what_is_dead) {

  case then_is_dead :
    /* Delete the test and the useless true : */
    test_false(t) = statement_undefined;
    test_true(t) = statement_undefined;
    remove_if_statement_according_to_write_effects(s,
						   false /* structured if */);
    /* Concatenate an eventual IF expression (if write effects) with
       the false branch: */
    statement_instruction(s) =
      make_instruction_block(
			     make_statement_list(instruction_to_statement(statement_instruction(s)),st_false)
			     );

    /* Go on the recursion on the remaining branch : */
    suppress_dead_code_statement(st_false);
    dead_code_if_true_branch_removed++;
    return false;
    break;

  case else_is_dead :
    /* Delete the test and the useless false : */
    test_false(t) = statement_undefined;
    test_true(t) = statement_undefined;
    remove_if_statement_according_to_write_effects(s,
						   false /* structured if */);
    /* Concatenate an eventual IF expression (if write effects) with
       the false branch: */
    statement_instruction(s) =
       make_instruction_block(
			      make_statement_list(
						  instruction_to_statement(statement_instruction(s)),st_true));
    /* Go on the recursion on the remaining branch : */
    suppress_dead_code_statement(st_true);
    dead_code_if_false_branch_removed++;
    return false;
    break;

  case nothing_about_test :
    break;

  default :
    pips_assert("dead_deal_with_test does not understand dead_test_filter()",
		true);
  }
  return true;
}


/* Give an information on the liveness of the 2 unstructured if's
   branches. Accept the statement that contains the if: */
static dead_test dead_unstructured_test_filter(statement st)
{
    /* In an unstructured test, we need to remove the dead control
       link according to preconditions. Unfortunately, preconditions
       are attached to statements and not to control vertice. Hence we
       need to recompute a precondition on these vertice: */
    dead_test test_status;
    transformer pre_true, pre_false;
    test t = instruction_test(statement_instruction(st));
    transformer pre = get_statement_precondition(st);
    expression cond = test_condition(t);

    pips_assert("Preconditions are defined for all statements",
		!transformer_undefined_p(pre));

    pips_assert("The statement is consistent",
		statement_consistent_p(st));

    ifdebug(6)
	sc_fprint(stderr,
		  predicate_system(transformer_relation(pre)),
		  (char* (*)(Variable)) entity_local_name);

    /* FI: this is the piece of code which may explain why the
       instruction cannot be freed in
       remove_if_statement_according_to_write_effects(). */
    /* Compute the precondition for each branch: */
    pre_true =
	precondition_add_condition_information(transformer_dup(pre),
					       cond,
					       transformer_undefined,
					       true);
    ifdebug(6)
	sc_fprint(stderr,
		  predicate_system(transformer_relation(pre_true)),
		  (char* (*)(Variable)) entity_local_name);

    pre_false =
	precondition_add_condition_information(transformer_dup(pre),
					       cond,
					       transformer_undefined,
					       false);
    ifdebug(6)
	sc_fprint(stderr,
		  predicate_system(transformer_relation(pre_false)),
		  (char* (*)(Variable)) entity_local_name);

    if (transformer_empty_p(pre_true)) {
	pips_debug(5, "then_is_dead\n");
	test_status = then_is_dead;
    }
    else if (transformer_empty_p(pre_false)) {
	pips_debug(5, "else_is_dead\n");
	test_status = else_is_dead;
    }
    else {
	pips_debug(5, "nothing_about_test\n");
	test_status = nothing_about_test;
    }

    free_transformer(pre_true);
    free_transformer(pre_false);

    pips_assert("The statement is consistent",
		statement_consistent_p(st));

    return test_status;
}

static void dead_recurse_unstructured(unstructured u)
{
  statement st = statement_undefined;
  //list blocs = NIL;
  list nodes = NIL;

  control_map_get_blocs(unstructured_control(u),&nodes);

  pips_assert("unstructured is consistent at the beginning",
	      unstructured_consistent_p(u));

  nodes = gen_nreverse(nodes);

  /* CONTROL_MAP removed for debugging */
  FOREACH(CONTROL, c, nodes) {
    int number_of_successors = gen_length(control_successors(c));

    pips_debug(3, "(gen_length(control_successors(c)) = %d)\n",
	       number_of_successors);
    st = control_statement(c);

    if (number_of_successors == 0 || number_of_successors == 1) {
      /* Ok, the statement is not an unstructured if, that
	 means that we can apply a standard elimination if
	 necessary. The statement is consistent on return. */
      suppress_dead_code_statement(st);
    }
    else if (number_of_successors == 2
	     && instruction_test_p(statement_instruction(st))) {
      /* In an unstructured test, we need to remove the
	 dead control link according to
	 preconditions. Unfortunately, preconditions
	 are attached to statements and not to control
	 vertice. Hence we need to recompute a
	 precondition on these vertice: */
      control true_control = CONTROL(CAR(control_successors(c)));
      control false_control = CONTROL(CAR(CDR(control_successors(c))));

      switch (dead_unstructured_test_filter(st)) {
      case then_is_dead :

	pips_assert("unstructured is consistent before then is dead",
	      unstructured_consistent_p(u));
	pips_debug(3, "\"Then\" is dead...\n");
	/* Remove the link to the THEN control
	   node. Rely on unspaghettify() to remove
	   down this path later: */
	gen_remove_once(&control_successors(c), true_control);
	gen_remove_once(&control_predecessors(true_control), c);

	pips_assert("unstructured is consistent after then_is_dead",
	      unstructured_consistent_p(u));
	/* Replace the IF with nothing or its expression: */
	remove_if_statement_according_to_write_effects
	  (control_statement(c), true /* unstructured if */);

	pips_assert("unstructured is consistent after then_is_dead",
	      unstructured_consistent_p(u));

	some_unstructured_ifs_have_been_changed = true;
	dead_code_unstructured_if_true_branch_removed++;

	pips_assert("unstructured is consistent after then_is_dead",
	      unstructured_consistent_p(u));
	break;

      case else_is_dead :

  pips_assert("unstructured is consistent before else_is_dead",
	      unstructured_consistent_p(u));
	pips_debug(3, "\"Else\" is dead...\n");
	/* Remove the link to the ELSE control
	   node. Rely on unspaghettify() to remove
	   down this path later: */
	gen_remove_once(&control_successors(c), false_control);
	gen_remove_once(&control_predecessors(false_control), c);
	/* Replace the IF with nothing or its expression: */
	remove_if_statement_according_to_write_effects
	  (control_statement(c), true /* unstructured if */);

	some_unstructured_ifs_have_been_changed = true;
	dead_code_unstructured_if_false_branch_removed++;

  pips_assert("unstructured is consistent after else_is_dead",
	      unstructured_consistent_p(u));
	break;

      case nothing_about_test :
	pips_debug(3, "Nothing about this test...");

	/* same successor in both branches... remove one!
	 * maybe unspaghettify should also be okay? it seems not.
	 */
	if (true_control==false_control)
	  {
	    gen_remove_once(&control_successors(c), false_control);
	    gen_remove_once(&control_predecessors(false_control), c);
	    /* Replace the IF with nothing or its expression: */
	    remove_if_statement_according_to_write_effects
	      (control_statement(c), true /* unstructured if */);

	    some_unstructured_ifs_have_been_changed = true;
	    dead_code_unstructured_if_false_branch_removed++;
	  }
	break;

      default :
	pips_internal_error("does not understand dead_test_filter()");
      }
    }
    else
      pips_internal_error("Unknown unstructured type");

    /* Let's hope the unstructured is still consistent */
    pips_assert("unstructured is consistent after some iterations",
		unstructured_consistent_p(u));
  }
  gen_free_list(nodes);

  pips_assert("unstructured is consistent at the end",
	      unstructured_consistent_p(u));
}

static bool one_iteration_while_loop_p(statement s)
{
  instruction i = statement_instruction(s);
  whileloop wl = instruction_whileloop(i);
  evaluation e = whileloop_evaluation(wl);
  bool one_iteration_p = false;
  statement wb = whileloop_body(wl);
  transformer prec = load_statement_precondition(s);
  transformer tf = load_statement_transformer(wb);
  expression c = whileloop_condition(wl);

  if(evaluation_after_p(e)) {
    // See if the repeat until is only executed once
    // Issue: the precondition holding before the condition is
    // re-evaluated because it has not been stored
    transformer tp = transformer_apply(tf, prec); // test precondition
    transformer ct = condition_to_transformer(c, tp, true);
    one_iteration_p = transformer_empty_p(ct);
    free_transformer(tp), free_transformer(ct);
  }
  else {
    // See if the loop condition is false when tested after the body execution
    // Should be useless since integrated in wb's precondition
    // transformer fct = condition_to_transformer(c, prec, true);
    transformer bprec = load_statement_precondition(wb);
    transformer tp = transformer_apply(tf, bprec); // test precondition
    transformer sct = condition_to_transformer(c, tp, true);
    one_iteration_p = transformer_empty_p(sct);
    free_transformer(tp), free_transformer(sct);
  }
  return one_iteration_p;
}

/* If it can be proven that the while loop is executed only once,
 * replace the while loop by its body in s.
 *
 * It is assumed that s contains a while loop.
 */ 
static void simplify_while_loop(statement s)
{
  instruction i = statement_instruction(s);
  whileloop wl = instruction_whileloop(i);
  evaluation e = whileloop_evaluation(wl);
  bool one_iteration_p = one_iteration_while_loop_p(s);

  if(one_iteration_p
     && (!evaluation_after_p(e) ||
	 // A quick fix for FREIA demonstrations
	 get_bool_property("SIMPLIFY_CONTROL_DO_WHILE"))
     // Same fix for while loops and the FREIA demonstration
     && get_bool_property("SIMPLIFY_CONTROL_DO_WHILE")) {

    // FI: this is buggy if the condition has a side effect
    // An expression statement should be added before the body and
    // another one after the body in case it is a do{}while() loop

    // Easiest solution: do not simplify while loops when there
    // conditions have side effects

    statement wb = whileloop_body(wl);
    statement_instruction(s) = statement_instruction(wb);
    // We should try to preserve labels, comments and statement numbers...
    statement_number(s) = statement_number(wb);
    // An issue if we have a label for s and a label for wb...
    ; // FI: to be seen later, wb is unlikely to have a label...
    // Concatenate comments...
    // FI: to be seen later... The parser seems to junk the loop comments
    string sc = statement_comments(s);
    string wbc = statement_comments(wb); // will be freed with wb
    string nc;
    if(empty_comments_p(sc)) {
      if(empty_comments_p(wbc))
	nc = empty_comments;
      else
	nc = strdup(wbc);
    }
    else {
      if(empty_comments_p(wbc))
	nc = strdup(sc);
      else
	nc = strdup(concatenate(sc, wbc, NULL)); // new comment
      free(sc);
    }
    statement_comments(s) = empty_comments;
    insert_comments_to_statement(s, nc);
    // Get rid of obsolete pieces of data structures
    whileloop_body(wl) = statement_undefined;
    free_instruction(i);
    statement_instruction(wb) = instruction_undefined;
    free_statement(wb);
  }
}

/* Essaye de faire le menage des blocs vides recursivement.
   En particulier, si les 2 branches d'un test sont vides on peut supprimer
   le test, si le corps d'une boucle est vide on peut supprimer la boucle.
*/
static void
dead_statement_rewrite(statement s)
{
   instruction i = statement_instruction(s);
   tag t = instruction_tag(i);

   pips_debug(2, "Begin for statement %td (%td, %td)\n",
	      statement_number(s),
	      ORDERING_NUMBER(statement_ordering(s)),
	      ORDERING_STATEMENT(statement_ordering(s)));

   stdebug(2, "dead_statement_rewrite: "
           "The current statement st at the beginning",
           s);

   switch(t) {
   case is_instruction_sequence: {
       /* It is now mostly dealt with clean_up_sequences_internal() at the
          end of the function: */
     /* Make sure the declaration list at the block level fits the
	declarations in the sequence in case some where eliminated */
      list dl = statement_declarations(s);
      list idl = statement_to_direct_declarations(s);
      gen_free_list(dl);
      statement_declarations(s) = idl;
      break;
   }
   case is_instruction_loop:
   case is_instruction_forloop:
     break;
   case is_instruction_whileloop:
     simplify_while_loop(s);
     break;
   case is_instruction_test:
   {
       statement st_true, st_false;
       test te;

       pips_debug(2, "is_instruction_test\n\n");
       stdebug(9, "dead_statement_rewrite: test", s);

       te = instruction_test(i);
       st_true = test_true(te);
       st_false = test_false(te);
       if (empty_statement_or_continue_p(st_true)
	   && empty_statement_or_continue_p(st_false)) {
	 /* Even if there is a label, it is useless since it is not
	    an unstructured. */
	 pips_debug(2, "test deletion\n");
	 stdebug(9, "dead_statement_rewrite: ", s);

	 remove_if_statement_according_to_write_effects
	   (s, false /* structured if */);
       }
       break;
   }

   case is_instruction_unstructured:
     /* Rely on the unspaghettify() at end: */
     break;

   case is_instruction_call:
     break;

   case is_instruction_expression:
     break;

   default:
       pips_internal_error("Unexpected instruction tag %d", t);
       break;
   }

   /* If we have now a sequence, clean it up, but be careful with
      declarations and user name conflicts... Bug here. See Transformations/until02.c */
   clean_up_sequences_internal(s);

   pips_debug(2, "End for statement %td (%td, %td)\n",
	      statement_number(s),
	      ORDERING_NUMBER(statement_ordering(s)),
	      ORDERING_STATEMENT(statement_ordering(s)));
   stdebug(2, "dead_statement_rewrite: The current statement st at the end", s);
}


static bool dead_statement_filter(statement s)
{
  instruction i;
  bool retour;
  effects crwe = load_cumulated_rw_effects(s);
  list crwl = effects_effects(crwe);
  transformer pre = get_statement_precondition(s);

  pips_assert("statement s is consistent", statement_consistent_p(s));

  i = statement_instruction(s);
  pips_debug(2, "Begin for statement %td (%td, %td)\n",
	     statement_number(s),
	     ORDERING_NUMBER(statement_ordering(s)),
	     ORDERING_STATEMENT(statement_ordering(s)));
  ifdebug(8)
    {
      sc_fprint(stderr,
		predicate_system(transformer_relation(pre)),
		(char* (*)(Variable)) entity_local_name);
    }

  stdebug(9, "dead_statement_filter: The current statement", s);

  pips_assert("statement s is consistent", statement_consistent_p(s));

  /* Pour permettre un affichage du code de retour simple : */
  for(;;) {
    if (statement_ordering(s) == STATEMENT_ORDERING_UNDEFINED) {
      /* Well, it is likely some unreachable code that should be
	 removed later by an unspaghettify: */
      pips_debug(2, "This statement is likely unreachable. Skip...\n");
      retour = false;
      break;
    }

    /* If a statement has no write effects on the store and if it
       cannot hides a control effect in a user-defined function*/
    if (ENDP(crwl) // No effects, store, declaration, type
	// Beware of Property MEMORY_EFFECTS_ONLY
	&& !statement_may_have_control_effects_p(s)
	&& !format_statement_p(s) // Fortran format
	&& ! declaration_statement_p(s) // a declaration
	/*&& ENDP(statement_declarations(s))*/) { // sequence with
						  // decl. only
						  // possible impact:
						  // stack space ovfl.
      pips_debug(2, "Ignored statement %td (%td, %td)\n",
		 statement_number(s),
		 ORDERING_NUMBER(statement_ordering(s)),
		 ORDERING_STATEMENT(statement_ordering(s)));
      retour = discard_statement_and_save_label_and_comment(s);
      dead_code_statement_removed++;
      break;
    }

    /* First remove (almost) all statements with an empty precondition */
    /* FI: This (very CPU expensive) test must be useless
     * because the control sequence
     * can only be broken by a test or by a loop or by a test in
     * an unstructured. These cases already are tested elsewhere.
     *
     * Furthermore, feasible statements take longer to test than
     * dead statement. And they are the majority in a normal piece
     * of code.
     *
     * But STOP statements can also introduce discontinuities...
     * Well, to please Ronan let's put a fast accurate enough
     * test for that last case.
     *
     * This can also happen when a function is never called, but
     * analyzed interprocedurally.
     */
    if (get_statement_precondition==load_statement_precondition
	&& !statement_weakly_feasible_p(s)) {
      pips_debug(2, "Dead statement %td (%td, %td)\n",
		 statement_number(s),
		 ORDERING_NUMBER(statement_ordering(s)),
		 ORDERING_STATEMENT(statement_ordering(s)));
      retour = discard_statement_and_save_label_and_comment(s);
      dead_code_statement_removed++;
      break;
    }

    if (instruction_sequence_p(i)
	&& get_statement_precondition==load_statement_precondition
	&& !statement_feasible_p(s)) {
      pips_debug(2, "Dead sequence statement %td (%td, %td)\n",
		 statement_number(s),
		 ORDERING_NUMBER(statement_ordering(s)),
		 ORDERING_STATEMENT(statement_ordering(s)));
      retour = discard_statement_and_save_label_and_comment(s);
      dead_code_statement_removed++;
      break;
    }

    if (instruction_loop_p(i)) {
      loop l = instruction_loop(i);
      if (dead_loop_p(l)) {
	pips_debug(2, "Dead loop %s at statement %td (%td, %td)\n",
		   label_local_name(loop_label(l)),
		   statement_number(s),
		   ORDERING_NUMBER(statement_ordering(s)),
		   ORDERING_STATEMENT(statement_ordering(s)));

	retour = remove_dead_loop(s, i, l);
	dead_code_loop_removed++;
	break;
      }
      else if (loop_executed_once_p(s, l)) {
        /* This piece of code is not ready yet */
        statement body = loop_body(l);
        ifdebug(2) {
          pips_debug(2,
                     "loop %s at %td (%td, %td) executed once and only once\n",
                     label_local_name(loop_label(l)),
                     statement_number(s),
                     ORDERING_NUMBER(statement_ordering(s)),
                     ORDERING_STATEMENT(statement_ordering(s)));

          stdebug(9, "dead_statement_filter", s);
        }

        remove_loop_statement(s, i, l);
        dead_code_loop_executed_once++;
        stdebug(9, "dead_statement_filter: out remove_loop_statement", s);

        suppress_dead_code_statement(body);
        retour = false;
        break;
      }
      else {
	/* Standard loop, proceed downwards */
	retour = true;
	break;
      }
    }

    /* FI: nothing for while loops, repeat loops and for loops;
     should always entered while loops be converted into repeat
     loops? Should while loop body postcondition be used to transform
     a while loop into a test? */

    if (instruction_test_p(i)) {
      test t = instruction_test(i);
      expression c = test_condition(t);
      (void) simplify_boolean_expression_with_precondition(c, pre);
      retour = dead_deal_with_test(s, t);
      break;
    }

    if (instruction_unstructured_p(i)) {
      /* Special case for unstructured: */
      pips_assert("the instruction is consistent", instruction_consistent_p(i));
      dead_recurse_unstructured(instruction_unstructured(i));
      pips_assert("the instruction is consistent", instruction_consistent_p(i));

      /* Stop going down since it has just been done in
       * dead_recurse_unstructured():
       */
      retour = false;
      break;
    }

    /* To deal with C conditional operator, at least, at the top
       level in the statement, i.e. no recursive descent in the call
       or the expression */
    if(instruction_call_p(i)) {
      call c = instruction_call(i);
      entity op = call_function(c);
      if(ENTITY_CONDITIONAL_P(op)) {
	list args = call_arguments(c);
	expression bool_exp = EXPRESSION(CAR(args));
	(void) simplify_boolean_expression_with_precondition(bool_exp, pre);
	if(true_expression_p(bool_exp)) {
	  expression e1 = EXPRESSION(CAR(CDR(args)));
	  instruction_tag(i) = is_instruction_expression;
	  instruction_expression(i) = copy_expression(e1);
	  free_call(c);
	  // Not really removed, just simplified...
	  dead_code_statement_removed++;
	}
	else if(false_expression_p(bool_exp)) {
	  expression e2 = EXPRESSION(CAR(CDR(CDR(args))));
	  instruction_tag(i) = is_instruction_expression;
	  instruction_expression(i) = copy_expression(e2);
	  free_call(c);
	  dead_code_statement_removed++;
	}
      }
      /* FI: no need to set retour nor to break*/
    }

    /* Well, else we are going on the inspection... */
    retour = true;
    break;
  }

  pips_assert("statement s is consistent before rewrite",
	      statement_consistent_p(s));

  if (retour == false) {
    /* Try to rewrite the code underneath. Useful for tests with
     * two empty branches
     */
    dead_statement_rewrite(s);
  }

  pips_debug(2, "End for statement %td (%td, %td): %s going down\n",
	     statement_number(s),
	     ORDERING_NUMBER(statement_ordering(s)),
	     ORDERING_STATEMENT(statement_ordering(s)),
	     retour? "" : "not");

  pips_assert("statement s is consistent at the end",
	      statement_consistent_p(s));

  return retour;
}

/* Suppress the dead code of the given module: */
static void suppress_dead_code_statement(statement mod_stmt)
{
  pips_assert("statement d is consistent", statement_consistent_p(mod_stmt));

  dead_statement_filter(mod_stmt);
  gen_recurse(mod_stmt, statement_domain,
	      dead_statement_filter, dead_statement_rewrite);
  dead_statement_rewrite(mod_stmt);

  pips_assert("statement d is consistent", statement_consistent_p(mod_stmt));
}


/*
 * Simply Control
 */
bool generic_simplify_control(string mod_name, bool use_precondition_p)
{
  statement mod_stmt;

  if(compilation_unit_p(mod_name)) {
    // bad idea to run simplify control on compilation unit
    pips_user_warning("Simplify control isn't intended to run on compilation"
        " unit ! Abort...");
    return false;
  }


  /* Get the true ressource, not a copy. */
  mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);

  set_current_module_statement(mod_stmt);

  set_current_module_entity(module_name_to_entity(mod_name));

  /* FI: RK used a false for db_get, i.e. an impur db_get...
   * I do not know why
   */
  set_proper_rw_effects((statement_effects)
      db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, true));

  if(use_precondition_p) {
    /* Tnsformers are useful to derive preconditions for control
       points where they are not stored, e.g. end of body in until
       loops */
    set_transformer_map((statement_mapping)
			 db_get_memory_resource(DBR_TRANSFORMERS,
						mod_name, true));
    set_precondition_map((statement_mapping)
			 db_get_memory_resource(DBR_PRECONDITIONS,
						mod_name, true));
  }

  set_cumulated_rw_effects((statement_effects)
      db_get_memory_resource(DBR_CUMULATED_EFFECTS, mod_name, true));

  debug_on("SIMPLIFY_CONTROL_DEBUG_LEVEL");

  ifdebug(1) {
    pips_debug(1, "Begin for %s\n", mod_name);
      pips_assert("Statements inconsistants...",
		  statement_consistent_p(mod_stmt));
  }

  // Necessary because of simplify_boolean_expression_with_precondition()
  // if(use_precondition_p)
    module_to_value_mappings(get_current_module_entity());
  initialize_dead_code_statistics();
  some_unstructured_ifs_have_been_changed = false;
  suppress_dead_code_statement(mod_stmt);
  if(!c_module_p(get_current_module_entity()))
    insure_return_as_last_statement(get_current_module_entity(), &mod_stmt);
  display_dead_code_statistics();
  // See above: if(use_precondition_p)
    free_value_mappings();
  reset_cumulated_rw_effects();

  debug_off();

  /* Reorder the module, because new statements have been generated. */
  module_reorder(mod_stmt);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name,
			 (char *) compute_callees(mod_stmt));

  reset_current_module_statement();
  reset_current_module_entity();
  if(use_precondition_p) {
    reset_transformer_map();
    reset_precondition_map();
  }
  reset_proper_rw_effects();
  ifdebug(1) {
    if(use_precondition_p)
      free_value_mappings();
  }

  pips_debug(1, "End for %s\n", mod_name);

  if (some_unstructured_ifs_have_been_changed)
     /* Now call the unspaghettify() function to remove some unreachable
        code after unstructured "if" elimination: */
     unspaghettify(mod_name);

  remove_useless_label(mod_name);

  ifdebug(1)
    pips_assert("statement is still consistent...",
		statement_consistent_p(mod_stmt));

  return true;
}

/* Use preconditions to simplify control */
bool simplify_control(string mod_name)
{
  get_statement_precondition = load_statement_precondition;
  return generic_simplify_control(mod_name, true);
}

static transformer load_identity_precondition(statement s __attribute__ ((__unused__)))
{
  static transformer id = transformer_undefined;
  if(transformer_undefined_p(id))
    id = transformer_identity();
  return id;
}

/* Do not use preconditions, only constant expressions */
bool simplify_control_directly(string mod_name)
{
  get_statement_precondition = load_identity_precondition;
  return generic_simplify_control(mod_name, false);
}

/*
 * Simply Control old alias
 */
bool suppress_dead_code(string mod_name)
{
  pips_user_warning("This phase has been renamed, please use 'simplify_control'"
      " from now. The old alias 'suppress_dead_code' might be removed soon.\n" );
  return simplify_control(mod_name);
}

/**
 * recursievly remove all labels from a module
 * only labels in unstructured are kept
 * @param module_name module considered
 *
 * @return true (never fails)
 */
bool
remove_useless_label(char* module_name)
{
   /* Get the module ressource */
   entity module = module_name_to_entity( module_name );
   statement module_statement =
       (statement) db_get_memory_resource(DBR_CODE, module_name, true);

   set_current_module_entity( module );
   set_current_module_statement( module_statement );

   gen_recurse(module_statement, statement_domain, gen_true, statement_remove_useless_label);
   clean_up_sequences(module_statement);

   module_reorder(module_statement);
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_statement);

   reset_current_module_entity();
   reset_current_module_statement();

   return true;
}



