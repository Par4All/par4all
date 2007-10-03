/* If conversion 
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "ri-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "transformer.h"
#include "semantics.h"
#include "control.h"
#include "transformations.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "properties.h"
#include "atomizer.h"

#include "expressions.h"

#include "ricedg.h"

#include "sac-local.h"

static graph dependence_graph;

/*
This function returns TRUE if the statement comment contains searched_string
 */
static bool check_if_statement_contains_comment(statement s, string searched_string)
{
  string comments;

  if (!statement_with_empty_comment_p(s)) {
    
    comments = strdup(statement_comments(s));
      
    if (strstr(comments, searched_string) != NULL) {
      return TRUE;
    }
  }

  return FALSE;
}

/*
This function returns the left reference of call c
 */
static reference get_left_ref_from_assign(call c)
{
   expression expr = EXPRESSION(CAR(call_arguments(c)));

   pips_assert("expr is a reference", syntax_reference_p(expression_syntax(expr)));

   return syntax_reference(expression_syntax(expr));
}

/*
This function returns a phi statement: lRef = PHI(cond, rRef1, rRef2)
(if(cond) then lRef = rRef1 else lRef = rRef2)
 */
static statement make_phi_assign_stat(reference lRef, expression cond,
				      reference rRef1, reference rRef2)
{
   list args = gen_make_list(expression_domain, 
      copy_expression(cond),
      reference_to_expression(copy_reference(rRef1)),
      reference_to_expression(copy_reference(rRef2)),
      NULL);

  expression phiExp = call_to_expression(make_call((entity)get_function_entity(SIMD_PHI_NAME),
			 args));

  statement newS = make_assign_statement(reference_to_expression(copy_reference(lRef)), phiExp);

  return newS;
}

/*
This function returns true if the right expression of an assign statement
has a write effect
 */
static bool expr_has_write_eff_p(expression expr)
{
   bool actionWrite = FALSE;

   list ef = expression_to_proper_effects(expr);

   MAP(EFFECT, f,
   {
      if(action_write_p(effect_action(f)))
      {
	 actionWrite = TRUE;
      }

   }, ef);

   gen_free_list(ef);

   return actionWrite;
}

/*
The if_conversion phase only supports assign statement whose right
expression does not have any write effect.

So, this function returns true if the statement stat is supported.
 */
bool simd_supported_stat(statement stat)
{
   if(!instruction_call_p(statement_instruction(stat)))
      return FALSE;

   call c = instruction_call(statement_instruction(stat));

   // Only the assign statements with no side effects are supported
   if(!ENTITY_ASSIGN_P(call_function(c)))
      return FALSE;

   expression rExp = EXPRESSION(CAR(CDR(call_arguments(c))));

   if(expr_has_write_eff_p(rExp))
      return FALSE;

   return TRUE;
}

static bool process_true_call_stat(expression cond, statement stat, list * outStat, list * postlude)
{
   call c = copy_call(instruction_call(statement_instruction(stat)));

   // Only the assign statements with no side effects are supported
   if(!simd_supported_stat(stat))
      return FALSE;

   // lRef is the left reference of the assign call
   reference lRef = get_left_ref_from_assign(c);

   entity e = reference_variable(lRef);

   statement assign = NULL;

   basic newBas = get_basic_from_array_ref(lRef);

   if(basic_undefined_p(newBas))
      return FALSE;

   // If lRef is a scalar, ...
   if(gen_length(reference_indices(lRef)) == 0)
   {
      // Create a new entity
      entity newVar = make_new_scalar_variable_with_prefix(entity_local_name(e),
					   get_current_module_entity(),
					   newBas);

      // Replace the left reference of the assign call by the new entity
      syntax_reference(expression_syntax(EXPRESSION(CAR(call_arguments(c))))) = make_reference(newVar, NIL);

      // Make an assign statement to add to the postlude
      assign = make_phi_assign_stat(
         lRef,
	 cond,
         make_reference(newVar, NIL),
	 lRef);

      *outStat = CONS(STATEMENT, call_to_statement(c), *outStat);
      *postlude = CONS(STATEMENT, assign, *postlude);
   }
   // If lRef is an array reference, ...
   else
   {
      // Create a new entity
      entity newVar = make_new_scalar_variable_with_prefix(entity_local_name(e),
					get_current_module_entity(),
					newBas);

      // Replace the left reference of the assign call by the new entity
      syntax_reference(expression_syntax(EXPRESSION(CAR(call_arguments(c))))) = make_reference(newVar, NIL);

      // Make an assign statement to add to the postlude
      assign = make_phi_assign_stat(
         lRef,
	 cond,
         make_reference(newVar, NIL),
	 lRef);

      *outStat = CONS(STATEMENT, call_to_statement(c), *outStat);
      *postlude = CONS(STATEMENT, assign, *postlude);

   }

   return TRUE;
}

/*
This function changes the true statement stat in two list.

For example, if the true statement stat is:

A(I) = I + 1
J = J + 1

then outStat will be:

A0 = I + 1
J0 = J + 1

and postlude will be:

A(I) = PHI(COND, A0, A(I))
J = J0
 */
static bool process_true_stat(expression cond, statement stat, list * outStat, list * postlude)
{
   *outStat = NIL;
   *postlude = NIL;

   // It must have been verified in the if_conversion_init phase
   pips_assert("stat is a call or a sequence statement", 
      (instruction_call_p(statement_instruction(stat)) ||
       instruction_sequence_p(statement_instruction(stat))));

   // If stat is a call statement, ...
   if(instruction_call_p(statement_instruction(stat)))
   {
      return process_true_call_stat(cond, stat, outStat, postlude);
   }
   // If stat is a sequence statement, ...
   else if(instruction_sequence_p(statement_instruction(stat)))
   {
      sequence seq = copy_sequence(instruction_sequence(statement_instruction(stat)));

      list pCur = NIL;

      list statDone = NIL;

      list saveStat = NIL;

      MAP(STATEMENT, cs,
      {
         // Only the assign statements with no side effects are supported
         if(!simd_supported_stat(cs))
	    return FALSE;

      }, sequence_statements(seq));

      MAPL(cs,
      {
         pips_assert("cs is a call statement", 
		     instruction_call_p(statement_instruction(STATEMENT(CAR(cs)))));

         call c = instruction_call(statement_instruction(STATEMENT(CAR(cs))));

         // Only the assign statements with no side effects are supported
         if(!simd_supported_stat(STATEMENT(CAR(cs))))
	    return FALSE;

	 if(gen_find_eq(STATEMENT(CAR(cs)), statDone) != statement_undefined)
	 {
	    continue;
	 }

         reference lRef = get_left_ref_from_assign(c);

         entity e = reference_variable(lRef);

	 statement assign = NULL;

         basic newBas = get_basic_from_array_ref(lRef);

         if(basic_undefined_p(newBas))
            return FALSE;

         // If the written reference is a scalar, ...
	 if(gen_length(reference_indices(lRef)) == 0)
	 {

	    // Create a new entity
            entity newVar = make_new_scalar_variable_with_prefix(entity_local_name(e),
					         get_current_module_entity(),
	    			                 newBas);

            // For each following statement, replace the old reference by
            // the new one
	    MAP(STATEMENT, rStat,
	    {
	       call rc = instruction_call(statement_instruction(rStat));

               reference lrRef = get_left_ref_from_assign(rc);

	       if(reference_equal_p(lRef, lrRef))
	       {
		  statDone = CONS(STATEMENT, rStat, statDone);
	       }

               saCallReplace(rc, lRef, newVar);

	    }, CDR(cs));

            syntax_reference(expression_syntax(EXPRESSION(CAR(call_arguments(c))))) = make_reference(newVar, NIL);

	    // Make the phi statement
            assign = make_phi_assign_stat(
               lRef,
	       cond,
               make_reference(newVar, NIL),
	       lRef);

	 }
         // If the written reference is not a scalar, ...
	 else
	 {
	    list lSaveInd = NIL;
	    list pSaveInd = NIL;

	    // For each reference index, 
	    MAP(EXPRESSION, ind,
	    {
	       // Create a new index variable used to store the current index
               // value
               entity newInd = make_new_scalar_variable_with_prefix("IND",
					         get_current_module_entity(),
					         make_basic(is_basic_int, (void *)4));

               // Save the current index value
	       statement indStat = make_assign_statement(entity_to_expression(newInd), copy_expression(ind));

	       // Add a comment to inform the if_conversion_compact phase 
               // that this statement must be compacted if possible
	       statement_comments(indStat) = strdup(IF_CONV_TO_COMPACT);

	       // Insert the created statement before the current one
	       sequence_statements(seq) = gen_insert_before(indStat,
		        STATEMENT(CAR(cs)),
	       	        sequence_statements(seq));

	       // Store the new index in lSaveInd
	       if(lSaveInd == NIL)
	       {
		  lSaveInd = pSaveInd = CONS(EXPRESSION, entity_to_expression(newInd), NIL);
	       }
	       else
	       {
		  CDR(pSaveInd) = CONS(EXPRESSION, entity_to_expression(newInd), NIL);
		  pSaveInd = CDR(pSaveInd);
	       }

	    }, reference_indices(lRef));

	    // Make a  new variable
            entity saveVar = make_new_scalar_variable_with_prefix(entity_local_name(e),
					         get_current_module_entity(),
					         newBas);

	    // Make a statement to save the current value of the reference
	    statement saveStat = make_assign_statement(entity_to_expression(saveVar), reference_to_expression(lRef));

	    // Add a comment to inform the if_conversion_compact phase 
            // that this statement must be compacted if possible
	    statement_comments(saveStat) = strdup(IF_CONV_TO_COMPACT);

	    // Insert the created statement before the current one
	    sequence_statements(seq) = gen_insert_before(saveStat,
		     STATEMENT(CAR(cs)),
	       	     sequence_statements(seq));

            entity newVar = make_new_scalar_variable_with_prefix(entity_local_name(e),
					         get_current_module_entity(),
					         newBas);

            syntax_reference(expression_syntax(EXPRESSION(CAR(call_arguments(c))))) = make_reference(newVar, NIL);

            assign = make_phi_assign_stat(
               make_reference(e, lSaveInd),
	       cond,
               make_reference(newVar, NIL),
	       make_reference(saveVar, NIL));
	 }

	 if(*postlude == NIL)
	 {
	    *postlude = pCur = CONS(STATEMENT, assign, *postlude);
	 }
	 else
	 {
	    CDR(pCur) = CONS(STATEMENT, assign, NIL);
	    pCur = CDR(pCur);
	 }

      }, sequence_statements(seq));

      *outStat = sequence_statements(seq);

      list old = *outStat;
      *outStat = gen_concatenate(*outStat, saveStat);
      gen_free_list(old);
   }

   return TRUE;
}

static list do_conversion(list tOutStats, list tPostlude)
{
   list outStats = NIL;
   list old = NIL;

   old = outStats;
   outStats = gen_concatenate(outStats, tOutStats);
   gen_free_list(old);

   old = outStats;
   outStats = gen_concatenate(outStats, tPostlude);
   gen_free_list(old);

   return outStats;
}

/*
This function is called for each code statement.
 */
static void if_conv_statement(statement cs)
{
   // If the statement is not a test statement, then nothing to do
   if(!instruction_test_p(statement_instruction(cs)))
      return;

   test t = instruction_test(statement_instruction(cs));

   pips_assert("statement is a test", instruction_test_p(statement_instruction(cs)));

   bool success;

   // If the statement comment contains the string IF_TO_CONVERT,
   // then it means that this statement must be converted ...
   success = check_if_statement_contains_comment(cs, IF_TO_CONVERT);

   // ... so let's convert.
   if(success)
   {
      list tOutStats = NIL;
      list tPostlude = NIL;

      // Process the "true statements" (test_false(t) is empty because if_conversion
      // phase is done after if_conversion_init phase).
      success = process_true_stat(test_condition(t), test_true(t), &tOutStats, &tPostlude);

      // If process_true_stat was a success, ...
      if(success)
      {
	 list newSeq;
	 instruction newInst;
	 instruction oldInst;

	 // Do the conversion and get the new statements sequence
	 newSeq = do_conversion(tOutStats, tPostlude);

	 // Replace the test instruction by the new sequence instruction
	 statement lastStat = cs;

         newInst = make_instruction_block(newSeq);

	 oldInst = statement_instruction(lastStat);
         statement_instruction(lastStat) = newInst;
	 free_instruction(oldInst);

         statement_label(lastStat) = entity_empty_label();
         statement_number(lastStat) = STATEMENT_NUMBER_UNDEFINED;
         statement_ordering(lastStat) = STATEMENT_ORDERING_UNDEFINED;
         statement_comments(lastStat) = empty_comments;
      }

      gen_free_list(tOutStats);
      gen_free_list(tPostlude);
   }
}

/*
This phase do the actual if conversion.
It changes:

c IF_TO_CONVERT
    if(L1) then
       A(I) = I + 1
       J = J + 1
    endif

into:

    A0 = I + 1
    J0 = J + 1
    A(I) = PHI(L1, A0, A(I))
    J = PHI(L1, J0, J)

This phase MUST be used after if_conversion_init phase

 */
boolean if_conversion(char * mod_name)
{
  // get the resources
   statement mod_stmt = (statement)
      db_get_memory_resource(DBR_CODE, mod_name, TRUE);

   set_current_module_statement(mod_stmt);
   set_current_module_entity(local_name_to_top_level_entity(mod_name));

   dependence_graph = 
      (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);

   debug_on("IF_CONVERSION_DEBUG_LEVEL");
   // Now do the job

   gen_recurse(mod_stmt, statement_domain,
	       gen_true, if_conv_statement);

   // Reorder the module, because new statements have been added 
   module_reorder(mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name, 
			  compute_callees(mod_stmt));
 
   // update/release resources
   reset_current_module_statement();
   reset_current_module_entity();

   debug_off();

   return TRUE;
}
