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

#include "effects-generic.h"
#include "effects-simple.h"
#include "control.h"

#include "transformations.h"


/* Statistiques parametres */
static int trivial_test_removed;
static int trivial_test_unstructured_removed;



/*
  Afficher les résultats des statistiques de la Suppress Trivial Test.
 */
static void
display_trivial_test_statistics()
{
  int elimination_count = trivial_test_removed + trivial_test_unstructured_removed ;
  if (elimination_count > 0) {
    user_log("* %d trivial test part%s %s been discarded. *\n",
	     elimination_count,
	     elimination_count > 1 ? "s" : "",
	     elimination_count > 1 ? "have" : "has");
    user_log("Structured trivial tests: %d\n" ,
	     trivial_test_removed);
    
    user_log("Unstructured trivial tests: %d\n" ,
	     trivial_test_unstructured_removed);
    /* Display also the statistics about clean_up_sequences
       that is called in suppress_trivial_test: */
    display_clean_up_sequences_statistics();
  }
  else
    pips_debug(8, "There is not any trivial test in this program !\n");
}



/*
  Determiner l'expression inverse de l'expression e.
 */
static expression
MakeInvertExpression(expression e)
{
  expression nc = expression_undefined;
  expression tmpc = expression_undefined;

  if (logical_expression_p(e)) {
      entity op = call_function(syntax_call(expression_syntax(e)));
      list args = call_arguments(syntax_call(expression_syntax(e)));
      expression e1 = expression_undefined;
      expression e2 = expression_undefined;
      if (!ENDP(args)) {
	e1 =  EXPRESSION(CAR(args));
	if (!ENDP(CDR(args)))
	  e2 = EXPRESSION(CAR(CDR(args)));
      }

      if (relational_expression_p(e)) {
	normalized n1 = NORMALIZE_EXPRESSION(e1);
	normalized n2 = NORMALIZE_EXPRESSION(e2);
	pips_debug(8, "The expression %s is a relational expression\n",
		   words_to_string(words_syntax(expression_syntax(e), NIL)));

	if (normalized_linear_p(n1) && normalized_linear_p(n2)) {

	  if (ENTITY_NON_EQUAL_P(op)) {
	    nc = eq_expression(e1, e2);
	  }
	  else if (ENTITY_EQUAL_P(op)) {
	    nc = ne_expression(e1, e2);
	  }
	  else if (ENTITY_GREATER_OR_EQUAL_P(op)) {
	    nc = lt_expression(e1, e2);
	  }
	  else if (ENTITY_LESS_OR_EQUAL_P(op)) {
	    nc = gt_expression(e1, e2);
	  }
	  else if (ENTITY_LESS_THAN_P(op)) {
	    nc = ge_expression(e1, e2);
	  }
	  else if (ENTITY_GREATER_THAN_P(op)) {
	    nc = le_expression(e1, e2);
	  }
	}
      }
      else if (logical_operator_expression_p(e)) {
	pips_debug(8, "The expression %s is a logical expression\n",
		   words_to_string(words_syntax(expression_syntax(e), NIL)));

	if (ENTITY_NOT_P(op)) {
	  tmpc = copy_expression(e1);
	}
	else if (ENTITY_AND_P(op)) {
	  expression tmpc1 = MakeInvertExpression(e1);
	  expression tmpc2 = MakeInvertExpression(e2);
	  tmpc = or_expression(tmpc1, tmpc2);
	}
	else if (ENTITY_OR_P(op)) {
	  expression tmpc1 = MakeInvertExpression(e1);
	  expression tmpc2 = MakeInvertExpression(e2);
	  tmpc = and_expression(tmpc1, tmpc2);
	}
	else if (ENTITY_EQUIV_P(op)) {
	  // expression tmpc1 = MakeInvertExpression(e1);
	  // expression tmpc2 = MakeInvertExpression(e2);
	  tmpc = binary_intrinsic_expression(NON_EQUIV_OPERATOR_NAME, e1, e2);
	}	  
	else if (ENTITY_NON_EQUIV_P(op)) {
	  // expression tmpc1 = MakeInvertExpression(e1);
	  // expression tmpc2 = MakeInvertExpression(e2);
	  tmpc = binary_intrinsic_expression(EQUIV_OPERATOR_NAME, e1, e2);
	}

	if (true_expression_p(tmpc))
	  nc =  make_false_expression();
	else if (false_expression_p(tmpc))
	  nc = make_true_expression();
	else 
	  nc = copy_expression(tmpc);
      }
  }

  return (nc) ;
}



/*
  Construire une nouvelle conditionelle instruction à partir de la conditionelle instruction vide
  du statement s dans le cas elle est bien structurée ("structured instruction").
  La conditionelle instruction vient d'être crée sera remplacer l'ancienne dans s.
 */
static void
trivial_test_deal_with_test(statement s) 
{  
  instruction i = statement_instruction(s);
  test t = instruction_test(i);
  expression cond = test_condition(t);
  expression ncond = MakeInvertExpression(cond);

  test  new = make_test(copy_expression(ncond),
			copy_statement(test_false(t)),
			make_block_statement(NIL));
  statement_instruction(s) = make_instruction(is_instruction_test, new);

  free_instruction(i);    
  trivial_test_removed++;
}



/*
  Cette fonction sera appelée dans le cas où la conditionelle instruction vide du statement s n'est
  pas structurée ("unstructured instruction"). Donc, avant de la remplacer par une nouvelle conditionelle
  instruction, il faut enlever tous les successeurs et predecesseurs correspondants de se control.
 */
static void
trivial_test_deal_with_unstructured(persistant_statement_to_control m, statement s) 
{
  instruction i = statement_instruction(s);
  test t = instruction_test(i);
  expression cond = test_condition(t);
  expression ncond = MakeInvertExpression(cond);

  control c = apply_persistant_statement_to_control(m,s);
  control ctrue = CONTROL(CAR(control_successors(c)));
  // control false = CONTROL(CAR(CDR(control_successors(c))));
 
  // control tmp = copy_control(false);
  // false = copy_control(ctrue);
  // ctrue = copy_control(tmp);
  // free_control(tmp);

  // update_control_lists(c, m);
  // UPDATE_CONTROL(c, s, control_predecessors(c), control_successors(c));
  
  // statement new_ctrue = control_statement(false);
 
  // make new test with control
  // test  new = make_test(copy_expression(ncond),
  //		 make_continue_statement(statement_label(new_ctrue)),
  //		 copy_statement(new_ctrue),
  //             make_block_statement(NIL));


  // make new test with test_false()

  test  new = make_test(copy_expression(ncond),
			copy_statement(test_false(t)),
			make_block_statement(NIL));

  gen_remove(&control_successors(c), ctrue);
  gen_remove(&control_predecessors(ctrue), c);
   
  statement_instruction(s) = make_instruction(is_instruction_test, new);

  free_instruction(i);
  trivial_test_unstructured_removed++;
}



static void
trivial_test_statement_rewrite(statement s, persistant_statement_to_control m)
{
   instruction i = statement_instruction(s);
   tag t = instruction_tag(i);

   switch(t) {
   case is_instruction_test:
   {
     test t ;
     statement strue ;

     debug(2, "trivial_test_statement_rewrite", "is_instruction_test\n", "\n");
     ifdebug(9) {
       print_statement(s);
     }

     t = instruction_test(i) ;
     strue = test_true(t) ;
     if (empty_statement_or_continue_p(strue)) {
       pips_debug(8,"The branch true of this test instruction is empty!\n");
       if (bound_persistant_statement_to_control_p(m, s)) {
	 pips_debug(8, "This instruction is unstructured instruction\n");
	 trivial_test_deal_with_unstructured(m, s);        
       }       
       else { 
	 pips_debug(8, "This instruction is structured instruction\n");
	 trivial_test_deal_with_test(s);
       }
     }
     break;
   }
   case is_instruction_call:
   case is_instruction_whileloop:
   case is_instruction_sequence:
   case is_instruction_loop:
   case is_instruction_unstructured:
     break;
   default:
     pips_internal_error("Unexpected instruction tag %d", t);
     break;
   }
}



static bool store_mapping(control c, persistant_statement_to_control map)
{
  extend_persistant_statement_to_control(map, control_statement(c), c);
  return true;
}



static void
suppress_trivial_test_statement(statement mod_stmt)
{
  persistant_statement_to_control map;
  
  map = make_persistant_statement_to_control();
  gen_context_multi_recurse(mod_stmt, map, control_domain, store_mapping, gen_null,
			    statement_domain, gen_true, trivial_test_statement_rewrite, NULL);
  free_persistant_statement_to_control(map);
}



/*
 * Trivial test elimination     
 * mod_name : MODule NAME, nom du programme Fortran
 * mod_stmt : MODule STateMenT

 * Elimination des branches de conditionnelles vides du programme en simplifiant et améliorant des 
   expressions des tests logiques des conditionnelles dans le module mod_name.
 */
bool
suppress_trivial_test(char * mod_name)  
{
  statement mod_stmt;
  set_current_module_entity(module_name_to_entity(mod_name));
  mod_stmt= (statement) db_get_memory_resource(DBR_CODE, mod_name, true);
  set_current_module_statement(mod_stmt);
  set_ordering_to_statement(mod_stmt);	

  debug_on("TRIVIAL_TEST_DEBUG_LEVEL");

  ifdebug(1) {
      debug(1,"trivial_test_elimination", "Begin for %s\n", mod_name);
      pips_assert("Inconsistent statements ...", statement_consistent_p(mod_stmt));
  }
  
  trivial_test_removed = 0 ;
  trivial_test_unstructured_removed = 0;

  suppress_trivial_test_statement(mod_stmt);
  display_trivial_test_statistics();

  debug_off();

  module_reorder(mod_stmt);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);

  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();

  debug(1,"trivial_test_elimination", "End for %s\n", mod_name);
  
  ifdebug(1)
      pips_assert("Inconsistent statements ...", statement_consistent_p(mod_stmt));
  
  return true;
}












