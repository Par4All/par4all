/*
  $Id$

  Dead loop elimination.
  Ronan Keryell, 12/1993 -> 1995.

  one trip loops fixed, FC 08/01/1998

  $Log: dead_code_elimination.c,v $
  Revision 1.27  2000/12/11 15:41:45  ancourt
  call to compute_callees at the end of suppress_dead_code

  Revision 1.26  2000/12/11 13:39:55  nguyen
  Modified function remove_dead_loop to deal with bug caused by nested loops
  with same labels
  Modified  dead_test_filter : change statement_feasible_p by
  statement_strongly_feasible_p (see example )

  Revision 1.25  2000/11/28 15:10:46  nguyen
  *** empty log message ***

  Revision 1.24  2000/11/08 07:47:00  nguyen
  assert added in dead_unstructured_test_filter()

  Revision 1.23  2000/07/19 15:33:43  ancourt
  test linearity of loop range to deal with REAL range type

  Revision 1.22  2000/07/05 11:02:28  coelho
  same branches special case handled.

  Revision 1.21  2000/07/05 08:46:16  coelho
  bug fixed:
  unstructured if with dead then or else... when branches where equal.

  Revision 1.20  2000/07/05 07:11:02  coelho
  new debug stuff

  Revision 1.19  2000/07/04 16:09:51  coelho
  hop.

  Revision 1.18  2000/07/03 12:17:31  nguyen
  Modify function discard_statement_and_save_label_and_comment(statement s):
  to avoid bug if both inner and outer loops of a nested loop have the same
  label. This label must be preseved while eliminating the dead inner loop.

  Revision 1.17  1999/01/08 12:43:25  coelho
  rcs stuff...

*/

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
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
    if (s) print_statement(s);
  }
}

/* Give an information on the liveness of the 2 if's branches: */
static dead_test
dead_test_filter(statement true, statement false)
{
  pips_debug(5, "Begin\n");

  stdebug(9, "dead_test_filter: then branch", true);
  stdebug(9, "dead_test_filter: else branch", false);

  ifdebug(8)
    {
      transformer pretrue = load_statement_precondition(true);
      transformer prefalse = load_statement_precondition(false);
      fprintf(stderr,"NN true and false branches");
      sc_fprint(stderr,
		predicate_system(transformer_relation(pretrue)),
		entity_local_name);
      sc_fprint(stderr,
		predicate_system(transformer_relation(prefalse)),
		entity_local_name);
    }

  if (!statement_strongly_feasible_p(true)) {
    pips_debug(5, "End: then_is_dead\n");
    return then_is_dead;
  }
  
  if (!statement_strongly_feasible_p(false)) {
    pips_debug(5, "End: else_is_dead\n");
    return else_is_dead;
  }
  
  pips_debug(5, "End: nothing_about_test\n");
  return nothing_about_test;
}

/* Replace an instruction by an empty one. If there is a label on the
   statement, put it on a continue to be coherent with the PIPS RI. */
bool
discard_statement_and_save_label_and_comment(statement s)
{

  /* NN -> Bug found : if we have two loops with the same label 
     such as : 
     DO 100 I=1,N
        DO 100 J=1,M
     ......
     
     100 CONTINUE    
     and the inner loop is a dead statement, there is an error when 
     compiling the generated file Fortran. 
     Because the label of the last statement in the inner loop 
     might be used by an outer loop and, in doubt, should be preserved.

     SOLUTION : like in full_loop_unroll()*/

  
  if (instruction_loop_p(statement_instruction(s)))
    { 
      entity flbl = find_final_statement_label(loop_body(instruction_loop(statement_instruction(s))));

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
  else 
    {
      /* Discard the old instruction: */
      free_instruction(statement_instruction(s));
      /* And put a new empty one: */
      statement_instruction(s) = make_instruction_block(NIL);
      
      /* Since the RI need to have no label on instruction block: */
      fix_sequence_statement_attributes(s); 
    }
   return FALSE;
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
  bool feasible_p;

  s = loop_body(l);
  /* feasible_p = statement_feasible_p(s); */
  feasible_p = statement_strongly_feasible_p(s);

  return !feasible_p;
}


/* Replace a loop statement with the statement inside the loop. */
static void
remove_loop_statement(statement s, instruction i, loop l)
{
  expression index, rl;
  range lr;
  statement as;
  expression init_val;

  init_val = copy_expression (rl = range_lower(lr = loop_range(l))); 
  /* Assume here that index is a scalar variable... :-) */
  pips_assert("dead_statement_filter", entity_scalar_p(loop_index(l)));

  index = make_factor_expression(1, loop_index(l));
  statement_instruction(s) =
    make_instruction_block(CONS(STATEMENT,
				as = make_assign_statement(index , init_val),
                                CONS(STATEMENT, loop_body(l), NIL)));
  statement_label(as) = statement_label(s);
  statement_label(s) = entity_empty_label();
  fix_sequence_statement_attributes(s);

  stdebug(4, "remove_loop_statement", s);

  loop_body(l) = make_empty_statement();
  free_instruction(i);
}

/* TRUE if do i = x, x or equivalent.
 */
static bool loop_executed_once_p(statement s, loop l)
{
  entity ind;
  range rg;
  expression m1, m2, m3;
  normalized n_m1, n_m2, n_m3;
  transformer pre;
  Psysteme ps, precondition_ps;
  Pvecteur pv3;
  Pcontrainte pc3;
  bool m3_negatif, m3_positif, retour;

  retour = FALSE;
  ind = loop_index(l);
  rg = loop_range(l);
  m1 = range_lower(rg);
  m2 = range_upper(rg);

  /* m1 == m2
   */
  if (expression_equal_p(m1, m2))
    return TRUE;

  pre = load_statement_precondition(s);
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
      retour = TRUE;
    
    contrainte_free(eq);

    if (retour) return TRUE;
  }
  if (normalized_linear_p(n_m3)) { 
    /* Teste le signe de l'incrément en fonction des préconditions : */
    pv3 = vect_dup(normalized_linear(n_m3));
    pc3 = contrainte_make(pv3);
    ps = sc_dup(precondition_ps);
    sc_add_ineg(ps, pc3);
    m3_negatif = sc_faisabilite(ps);
    
    (void) vect_chg_sgn(pv3);
    m3_positif = sc_faisabilite(ps);
    
    pips_debug(2, "loop_increment_value positif = %d, negatif = %d\n",
	       m3_positif, m3_negatif);
    
    /* Vire aussi pv3 & pc3 : */
    sc_rm(ps);
  }
  if ((m3_positif ^ m3_negatif) && normalized_linear_p(n_m3) && 
      normalized_linear_p(n_m1) && normalized_linear_p(n_m2))
  {
    /* Si l'incrément a un signe « connu » et différent de 0 et que
       les bornes sont connues : */
    Pvecteur pv1, pv2, pv3, pvx, pv;
    Pcontrainte ca, cb;

    pv1 = normalized_linear(n_m1);
    pv2 = normalized_linear(n_m2);
    pv3 = normalized_linear(n_m3);
    
    /* pv = m1 - m2 */
    pv = vect_substract(pv1, pv2);

    /* pv = m1 - m2 + m3 */
    pvx = vect_add(pv, pv3);

    if (m3_positif) {
      /* L'incrément est positif. */
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

    /* ??? on overflows, should assume FALSE...
     */
    retour = ineq_redund_with_sc_p(precondition_ps, ca) &&
             ineq_redund_with_sc_p(precondition_ps, cb);

    /* Vire du même coup pv et pvx : */
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
  /*expression init_val = expression_dup(range_lower(loop_range(l)));*/

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
     compiling the generated file Fortran. 
     Because the label of the last statement in the inner loop 
     might be used by an outer loop and, in doubt, should be preserved.

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
  return FALSE;
}


/* Remove an IF(x) statement (replace s with an empty statement) if x
   has no write proper effect. If x has a write effect, replace s with a
   statement as bool_var = x: (he', a french joke !)
   this_test_is_unstructured_p is a hint for the statistics.
   TRUE means that you assert that the test is unstructured.
 */
void remove_if_statement_according_to_write_effects
(statement s, bool this_test_is_unstructured_p)
{
   instruction i = statement_instruction(s);

   if (statement_write_effect_p(s)) {
      /* There is a write effect, so we cannot discard the IF
         expression. Keep it in a temporarily variable: */
      entity temp_var = make_new_scalar_variable(get_current_module_entity(),
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
   }
   else {
      /* There is no write effect, the statement can be discarded: */
      statement_instruction(s) = make_instruction_block(NIL);
      fix_sequence_statement_attributes(s);
      
      if (this_test_is_unstructured_p)
	  dead_code_unstructured_if_removed++;
      else
	  dead_code_if_removed++;
   }

   /* Discard the IF: */
   free_instruction(i);
}


static bool
dead_deal_with_test(statement s,
                    test t) 
{  
  statement true = test_true(t);
  statement false = test_false(t);

  switch (dead_test_filter(true, false)) {
    
  case then_is_dead :
    /* Delete the test and the useless true : */ 
    test_false(t) = statement_undefined;
    test_true(t) = statement_undefined;
    remove_if_statement_according_to_write_effects(s,
						   FALSE /* structured if */);
    /* Concatenate an eventual IF expression (if write effects) with
       the false branch: */
    statement_instruction(s) =
       make_instruction_block(CONS(STATEMENT,
                                  make_stmt_of_instr(statement_instruction(s)),
                                  CONS(STATEMENT,
                                       false, NIL)));
    
    /* Go on the recursion on the remaining branch : */
    suppress_dead_code_statement(false);
    dead_code_if_true_branch_removed++;
    return FALSE;
    break;

  case else_is_dead :
    /* Delete the test and the useless false : */ 
    test_false(t) = statement_undefined;
    test_true(t) = statement_undefined;
    remove_if_statement_according_to_write_effects(s,
						   FALSE /* structured if */);
    /* Concatenate an eventual IF expression (if write effects) with
       the false branch: */
    statement_instruction(s) =
       make_instruction_block(CONS(STATEMENT,
                                  make_stmt_of_instr(statement_instruction(s)),
                                  CONS(STATEMENT,
                                       true, NIL)));
    /* Go on the recursion on the remaining branch : */
    suppress_dead_code_statement(true);
    dead_code_if_false_branch_removed++;
    return FALSE;
    break;

  case nothing_about_test :
    break;

  default :
    pips_assert("dead_deal_with_test does not understand dead_test_filter()",
                TRUE);
  }
  return TRUE;
}


/* Give an information on the liveness of the 2 unstructured if's
   branches. Accept the statement that contains the if: */
static dead_test
dead_unstructured_test_filter(statement st)
{
    /* In an unstructured test, we need to remove the dead control
       link according to preconditions. Unfortunately, preconditions
       are attached to statements and not to control vertice. Hence we
       need to recompute a precondition on these vertice: */
    dead_test test_status;
    transformer pre_true, pre_false;
    test t = instruction_test(statement_instruction(st));
    transformer pre = load_statement_precondition(st);
    expression cond = test_condition(t);
    
    pips_assert("Preconditions are defined for all statements",
		!transformer_undefined_p(pre));

    ifdebug(6)
	sc_fprint(stderr,
		  predicate_system(transformer_relation(pre)),
		  entity_local_name);

    /* Compute the precondition for each branch: */
    pre_true =
	precondition_add_condition_information(transformer_dup(pre),
					       cond,
					       TRUE);
    ifdebug(6)
	sc_fprint(stderr,
		  predicate_system(transformer_relation(pre_true)),
		  entity_local_name);

    pre_false =
	precondition_add_condition_information(transformer_dup(pre),
					       cond,
					       FALSE);
    ifdebug(6)
	sc_fprint(stderr,
		  predicate_system(transformer_relation(pre_false)),
		  entity_local_name);

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

    return test_status;
}

static void
dead_recurse_unstructured(unstructured u)
{
    statement st = statement_undefined;
    list blocs = NIL;
  
    CONTROL_MAP(c, {
	int number_of_successors = gen_length(control_successors(c));
                  
	pips_debug(3, "(gen_length(control_successors(c)) = %d)\n",
		   number_of_successors);     
	st = control_statement(c);
                  
	if (number_of_successors == 0 || number_of_successors == 1) {
	    /* Ok, the statement is no an unstructured if, that
	       means that we can apply a standard elimination if
	       necessary: */
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
	      pips_debug(3, "\"Then\" is dead...");
	      /* Remove the link to the THEN control
		 node. Rely on unspaghettify() to remove
		 down this path later: */
	      gen_remove_once(&control_successors(c), true_control);
	      gen_remove_once(&control_predecessors(true_control), c);
	      /* Replace the IF with nothing or its expression: */
	      remove_if_statement_according_to_write_effects
		(control_statement(c), TRUE /* unstructured if */);
	      
	      some_unstructured_ifs_have_been_changed = TRUE;
	      dead_code_unstructured_if_true_branch_removed++;
	      break;
                         
	    case else_is_dead :
	      pips_debug(3, "\"Else\" is dead...");
	      /* Remove the link to the ELSE control
		 node. Rely on unspaghettify() to remove
		 down this path later: */
	      gen_remove_once(&control_successors(c), false_control);
	      gen_remove_once(&control_predecessors(false_control), c);
	      /* Replace the IF with nothing or its expression: */
	      remove_if_statement_according_to_write_effects
		(control_statement(c), TRUE /* unstructured if */);
	      
	      some_unstructured_ifs_have_been_changed = TRUE;
	      dead_code_unstructured_if_false_branch_removed++;
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
		  (control_statement(c), TRUE /* unstructured if */);
	      
		some_unstructured_ifs_have_been_changed = TRUE;
		dead_code_unstructured_if_false_branch_removed++;
	      }
	      break;
	      
	    default :
	      pips_internal_error("does not understand dead_test_filter()");
	    }
	}
	else
	    pips_error("dead_deal_with_test",
		       "Unknown unstructured type");
    },
		unstructured_control(u),
		blocs);
    gen_free_list(blocs);
}




/* Essaye de faire le me'nage des blocs vides re'cursivement.
   En particulier, si les 2 branches d'un test sont vides on peut supprimer
   le test, si le corps d'une boucle est vide on peut supprimer la boucle.
*/
static void
dead_statement_rewrite(statement s)
{
   instruction i = statement_instruction(s);
   tag t = instruction_tag(i);

   pips_debug(2, "Begin for statement %d (%d, %d)\n",
	      statement_number(s),
	      ORDERING_NUMBER(statement_ordering(s)),
	      ORDERING_STATEMENT(statement_ordering(s)));

   stdebug(2, "dead_statement_rewrite: The current statement", s);

   switch(t) {
   case is_instruction_sequence:
       /* It is now dealt with clean_up_sequences_internal() at the
          end of the function: */
       break;

   case is_instruction_loop:   
   case is_instruction_whileloop:
       break;
    
   case is_instruction_test: 
   {
       statement true, false;
       test te;

       pips_debug(2, "is_instruction_test\n\n");
       stdebug(9, "dead_statement_rewrite: test", s);
    
       te = instruction_test(i);
       true = test_true(te);
       false = test_false(te);
       if (empty_statement_or_continue_p(true)
	   && empty_statement_or_continue_p(false)) {
	 /* Even if there is a label, it is useless since it is not
	    an unstructured. */
	 pips_debug(2, "test deletion\n");
	 stdebug(9, "dead_statement_rewrite: ", s);
	 
	 remove_if_statement_according_to_write_effects
	   (s, FALSE /* structured if */);
       }
       break;
   }

   case is_instruction_unstructured:
       /* Rely on the unspaghettify() at end: */
       break;

   case is_instruction_call:
       break;

   default:
       pips_error("", "Unexpected instruction tag %d\n", t);
       break;
   }

   /* If we have now a sequence, clean it up: */
   clean_up_sequences_internal(s);
   
   pips_debug(2, "End for statement %d (%d, %d)\n",
	      statement_number(s),
	      ORDERING_NUMBER(statement_ordering(s)),
	      ORDERING_STATEMENT(statement_ordering(s)));
}


static bool
dead_statement_filter(statement s)
{
   instruction i;
   bool retour;

   i = statement_instruction(s);
   pips_debug(2, "Begin for statement %d (%d, %d)\n",
	      statement_number(s),
	      ORDERING_NUMBER(statement_ordering(s)),
	      ORDERING_STATEMENT(statement_ordering(s)));
   ifdebug(8)
     {
       transformer pre = load_statement_precondition(s);
       sc_fprint(stderr,
		 predicate_system(transformer_relation(pre)),
		 entity_local_name);
     }

   stdebug(9, "dead_statement_filter: The current statement", s);

   /* Pour permettre un affichage du code de retour simple : */
   for(;;) {
       if (statement_ordering(s) == STATEMENT_ORDERING_UNDEFINED) {
	   /* Well, it is likely some unreachable code that should be
              removed later by an unspaghettify: */
	 pips_debug(2, "This statement is likely unreachable. Skip...\n");
	 retour = FALSE;
	 break;
       }

      /* Vire de'ja` (presque) tout statement dont la pre'condition est
         fausse : */
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
	*/
      if (!statement_weakly_feasible_p(s)) {
	pips_debug(2, "Dead statement %d (%d, %d)\n",
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
	   pips_debug(2, "Dead loop %d at statement %d (%d, %d)\n",
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
	       pips_debug(2, "loop %d at %d (%d, %d) executed once and only once\n",
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
	     retour = FALSE;
	     break;
	 }
	 else {
	     /* Standard loop, proceed downwards */
	     retour = TRUE;
	     break;
	 }
     }

      if (instruction_test_p(i)) {
         test t = instruction_test(i);
         retour = dead_deal_with_test(s, t);
         break;
      }

      if (instruction_unstructured_p(i)) {
         /* Special case for unstructured: */
         dead_recurse_unstructured(instruction_unstructured(i));
         
         /* Stop going down since it has just been done in
	  * dead_recurse_unstructured():
	  */
         retour = FALSE;
         break;
      }

      /* Well, else we are going on the inspection... */
      retour = TRUE;
      break;
   }

   if (retour == FALSE) {
       /* Try to rewrite the code underneath. Useful for tests with
	* two empty branches
	*/
      dead_statement_rewrite(s);
  }

   pips_debug(2, "End for statement %d (%d, %d): %s going down\n",
	 statement_number(s),
	 ORDERING_NUMBER(statement_ordering(s)),
	 ORDERING_STATEMENT(statement_ordering(s)),
	 retour? "" : "not");

   return retour;
}


/* Suppress the dead code of the given module: */
void
suppress_dead_code_statement(statement mod_stmt)
{
    dead_statement_filter(mod_stmt);
    gen_recurse(mod_stmt, statement_domain,
		dead_statement_filter, dead_statement_rewrite);
    dead_statement_rewrite(mod_stmt);	

}


/*
 * Dead code elimination     
 * mod_name : MODule NAME, nom du programme Fortran
 * mod_stmt : MODule STateMenT
 */

bool
suppress_dead_code(char * mod_name)  
{
  statement mod_stmt;

  /* Get the true ressource, not a copy. */
  mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);
  set_current_module_statement(mod_stmt);

  set_current_module_entity(local_name_to_top_level_entity(mod_name));

  /* FI: RK used a FALSE for db_get, i.e. an impur db_get...
   * I do not know why
   */
  set_proper_rw_effects((statement_effects)
			db_get_memory_resource(DBR_PROPER_EFFECTS,
					       mod_name,
					       TRUE));

  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,
					      mod_name,
					      TRUE));

  debug_on("DEAD_CODE_DEBUG_LEVEL");

  ifdebug(1) {
    pips_debug(1, "Begin for %s\n", mod_name);
      pips_assert("Statements inconsistants...", statement_consistent_p(mod_stmt));
  }
  
  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource(DBR_CUMULATED_EFFECTS, mod_name, TRUE));
  module_to_value_mappings(get_current_module_entity());
  initialize_dead_code_statistics();
  some_unstructured_ifs_have_been_changed = FALSE;
  suppress_dead_code_statement(mod_stmt);
  insure_return_as_last_statement(get_current_module_entity(), &mod_stmt);
  display_dead_code_statistics();
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
  reset_precondition_map();
  reset_proper_rw_effects();
  ifdebug(1) {
      free_value_mappings();
  }

  pips_debug(1, "End for %s\n", mod_name);
  
  if (some_unstructured_ifs_have_been_changed)
     /* Now call the unspaghettify() function to remove some unreachable
        code after unstructured "if" elimination: */
     unspaghettify(mod_name);

  ifdebug(1)
      pips_assert("Statements inconsistants...", 
		  statement_consistent_p(mod_stmt));
  
  return TRUE;
}


/* Return true if a statement has at least one write effect in the
   effects list. */
bool
statement_write_effect_p(statement s)
{
   bool write_effect_found = FALSE;
   list effects_list = load_proper_rw_effects_list(s);

   MAP(EFFECT, an_effect,
       {
          if (action_write_p(effect_action(an_effect))) {
             write_effect_found = TRUE;
             break;
          }
       },
       effects_list);

   return write_effect_found;
}
