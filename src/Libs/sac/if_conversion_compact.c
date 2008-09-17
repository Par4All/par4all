/* If conversion compact
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

#include "sac-local.h"

/*
This function return TRUE if the call is:

... = PHI(....)
 */
static bool check_assign_phi_call(call c)
{
   entity func = call_function(c);

   if(ENTITY_ASSIGN_P(func))
   {
      expression rExp = EXPRESSION(CAR(CDR(call_arguments(c))));

      if(expression_call_p(rExp))
      {
	 call cc = syntax_call(expression_syntax(rExp));

	 if(!strcmp(SIMD_PHI_NAME,
	    entity_local_name(call_function(cc))))
	 {
	    return TRUE;
	 }
      }
   }

   return FALSE;
}

/*
This function returns true if the two expression are complement
and that the first call of the expression cond2 is .NOT.
 */
static bool check_first_arg(expression cond1, expression cond2)
{
   // If cond2 is not an expression or if the call operator is not .NOT., ...
   if(!expression_call_p(cond2) ||
      strcmp(NOT_OPERATOR_NAME,
         entity_local_name(call_function(syntax_call(expression_syntax(cond2))))))
      return FALSE;

   call c = syntax_call(expression_syntax(cond2));

   // Check that the two condition are the same
   if(same_expression_p(cond1, EXPRESSION(CAR(call_arguments(c)))))
   {
     return TRUE;
   }

   return FALSE;
}

static bool process_stat_to_compact(statement stat)
{
  string comments = NULL;
  char*  next_line;

  bool res = FALSE;

  if (!statement_with_empty_comment_p(stat)) 
  {
    comments = strdup(statement_comments(stat));
    next_line = strtok (comments, "\n");
    if (next_line != NULL) {
      do {
	if(!strncmp(next_line, IF_CONV_TO_COMPACT, 20))
        {
	   res = TRUE;
	   free(statement_comments(stat));
	   statement_comments(stat) = empty_comments;
	   break;
	}

	next_line = strtok(NULL, "\n");
      }
      while (next_line != NULL);
    }

    if(comments)
       free(comments);
  }

  return res;
}

/*
This function does the job for each sequence.
 */
static void if_conversion_compact_stats(statement stat)
{
   // Only look at the sequence statements
   if(!statement_block_p(stat))
      return;

   list seq = NIL;
   list newseq = NIL;

   seq = sequence_statements(instruction_sequence(statement_instruction(stat)));
   newseq = da_process_list(seq, FALSE, process_stat_to_compact);

   sequence_statements(instruction_sequence(statement_instruction(stat))) = newseq;
   gen_free_list(seq);

   list pStat = NIL;

   //Go through the statements of the sequence
   MAPL(lStat,
   {
      bool rep = FALSE;

      statement cStat1 = STATEMENT(CAR(lStat));

      // If it is a call to phi function,...
      if(statement_call_p(cStat1) &&
         check_assign_phi_call(statement_call(cStat1)))
      {

	//Go through the remaining statements of the sequence
	MAP(STATEMENT, cStat2,
	{
            // If it is a call to phi function,...
            if(statement_call_p(cStat2) && 
	       check_assign_phi_call(statement_call(cStat2)))
            {
	       call c1 = statement_call(cStat1);
	       list arg1 = call_arguments(c1);

	       expression phiExp1 = EXPRESSION(CAR(CDR(call_arguments(c1))));
	       list phiArg1 = call_arguments(syntax_call(expression_syntax(phiExp1)));

	       expression lArg1 = EXPRESSION(CAR(arg1));
	       expression cond1 = EXPRESSION(CAR(phiArg1));
	       expression tVal1 = EXPRESSION(CAR(CDR(phiArg1)));

	       call c2 = statement_call(cStat2);
	       list arg2 = call_arguments(c2);

	       expression phiExp2 = EXPRESSION(CAR(CDR(call_arguments(c2))));
	       list phiArg2 = call_arguments(syntax_call(expression_syntax(phiExp2)));

	       expression lArg2 = EXPRESSION(CAR(arg2));
	       expression cond2 = EXPRESSION(CAR(phiArg2));
	       expression tVal2 = EXPRESSION(CAR(CDR(phiArg2)));

	       // If the phi condition are complement and 
	       // that they share the same lValue
	       if(check_first_arg(cond1, cond2) &&
		  same_expression_p(lArg1, lArg2) )
	       {
		  // The argument list of the new phi call
	          list args = gen_make_list(expression_domain, 
	             copy_expression(cond1),
	             copy_expression(tVal1),
	             copy_expression(tVal2),
	             NULL);

		  // The new phi call
                  expression phiExp = call_to_expression(
                     make_call((entity)get_function_entity(SIMD_PHI_NAME), args));

		  // The new assignment-phi call
                  call newCall = make_call(entity_intrinsic(ASSIGN_OPERATOR_NAME),
                     CONS(EXPRESSION, copy_expression(lArg1), CONS(EXPRESSION, phiExp, NIL)));

		  // Free the old call of the statement
		  free_call(c2);

		  // Insert the new call
		  instruction_call(statement_instruction(cStat2)) = newCall;

		  pips_assert("cStat1 not the first stat in sequence", 
                     pStat != NIL);

		  // Delete the first statement
		  CDR(pStat) = CDR(lStat);
		  free_statement(cStat1);
		  rep = TRUE;
		  break;
	       }
	    }
	 }, lStat);
      }

      if(!rep)
         pStat = lStat;

   }, statement_block(stat));
}

/*
This phase is applied after if_conversion phase and will changes:

.
.
.
I = PHI(L, I1, I)
.
.
.
I = PHI(.NOT.L, I2, I)
.
.
.

into:

.
.
.
I = PHI(L, I1, I2)
.
.
.
 */
boolean if_conversion_compact(char * mod_name)
{
  // get the resources
   statement mod_stmt = (statement)
      db_get_memory_resource(DBR_CODE, mod_name, TRUE);

   set_current_module_statement(mod_stmt);
   set_current_module_entity(module_name_to_entity(mod_name));

   graph dg = (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);

   init_dep_graph(dg);

   set_proper_rw_effects((statement_effects) 
      db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, TRUE));

   debug_on("IF_CONVERSION_COMPACT_DEBUG_LEVEL");
   // Now do the job

   gen_recurse(mod_stmt, statement_domain,
	       gen_true, if_conversion_compact_stats);

   // Reorder the module, because new statements have been added 
   module_reorder(mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name, 
			  compute_callees(mod_stmt));
 
   // update/release resources
   reset_current_module_statement();
   reset_current_module_entity();
   reset_proper_rw_effects();

   debug_off();

   return TRUE;
}
