
#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "semantics.h"
#include "effects-generic.h"
#include "transformations.h"

#include "sac.h"


static bool should_unroll_p(instruction i)
{
   switch(instruction_tag(i))
   {
      case is_instruction_call:
	 return TRUE;

      case is_instruction_sequence:
      {
	 cons * j;

	 for( j = sequence_statements(instruction_sequence(i));
	      j != NIL;
	      j = CDR(j) )
	 {
	    statement s = STATEMENT(CAR(j));
	    if (!should_unroll_p(statement_instruction(s)))
	       return FALSE;
	 }
	 return TRUE;
      }

      case is_instruction_test:
      case is_instruction_loop:
      case is_instruction_whileloop:
      case is_instruction_goto:
      case is_instruction_unstructured:
      default:
	 return FALSE;
   }
}

static bool simd_unroll_loop_filter(statement s)
{
   int varwidth;
   instruction i;
   loop l;
   instruction iBody;

   /* If this is not a loop, keep on recursing */
   i = statement_instruction(s);
   if (!instruction_loop_p(i))
      return TRUE;
   l = instruction_loop(i);

   /* Can only simdize certain loops */
   iBody = statement_instruction(loop_body(l));
   if (!should_unroll_p(iBody))
      return TRUE;  /* can't do anything */

   /* Compute variable size */
   varwidth = effective_variables_width(iBody);

   /* Unroll as many times as needed by the variables width */
   if ((varwidth > 32) || (varwidth <= 0)) 
      return FALSE;
   else if (varwidth <= 8) 
      varwidth = 8;
   else if (varwidth <= 16) 
      varwidth = 16;
   else 
      varwidth = 32;
   loop_unroll(s, 64 / varwidth);

   /* Do not recursively analyse the loop */
   return FALSE;
}

static void simd_unroll_loop_rewrite(statement s)
{
   return;
}

void simd_unroll_as_needed(statement module_stmt)
{
   gen_recurse(module_stmt, statement_domain, 
	       simd_unroll_loop_filter, simd_unroll_loop_rewrite);
}

bool simdizer_auto_unroll(char * mod_name)
{
   // get the resources
   statement mod_stmt = (statement)
      db_get_memory_resource(DBR_CODE, mod_name, TRUE);

   set_current_module_statement(mod_stmt);
   set_current_module_entity(local_name_to_top_level_entity(mod_name));

   debug_on("SIMDIZER_DEBUG_LEVEL");

   simd_unroll_as_needed(mod_stmt);

   pips_assert("Statement is consistent after SIMDIZER_AUTO_UNROLL", 
	       statement_consistent_p(mod_stmt));

   // Reorder the module, because new statements have been added
   module_reorder(mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
 
   // update/release resources
   reset_current_module_statement();
   reset_current_module_entity();

   debug_off();

   return TRUE;
}
