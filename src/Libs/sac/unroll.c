
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

#include "control.h"

#include "sac.h"

#include <limits.h>

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
	 return TRUE;

      case is_instruction_loop:
      case is_instruction_whileloop:
      case is_instruction_goto:
      case is_instruction_unstructured:
      default:
	 return FALSE;
   }
}

typedef struct {
      int min;
      int max;
} MinMaxVar;

static void compute_variable_size(statement s, MinMaxVar* varwidth)
{
   int width = effective_variables_width(statement_instruction(s));

   if (width > varwidth->max)
      varwidth->max = width;

   if (width < varwidth->min)
      varwidth->min = width;
}

static bool simple_simd_unroll_loop_filter(statement s)
{
   MinMaxVar varwidths;
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
   varwidths.min = INT_MAX;
   varwidths.max = 0;
   gen_context_recurse(iBody, &varwidths, statement_domain, gen_true, 
		       compute_variable_size);

   /* Decide between min and max unroll factor */
   if (get_bool_property("SIMD_AUTO_UNROLL_MINIMIZE_UNROLL"))
      varwidth = varwidths.max;
   else
      varwidth = varwidths.min;

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

static void compute_parallelism_factor(statement s, MinMaxVar* factor)
{
   int varwidth = effective_variables_width(statement_instruction(s));

   /* see if the statement can be SIMDized */
   MAP(MATCH,
       m,
   {
      /* and if so, to what extent it may benefit from unrolling */
      MAP(OPCODE,
	  o,
      {
	 if (opcode_subwordSize(o) >= varwidth) //opcode may be used
	 {
	    if (opcode_vectorSize(o) > factor->max)
	       factor->max = opcode_vectorSize(o);
	    if (opcode_vectorSize(o) < factor->min)
	       factor->min = opcode_vectorSize(o);
	 }
      },
	  opcodeClass_opcodes(match_type(m)));
   },
       match_statement(s));
}

static bool full_simd_unroll_loop_filter(statement s)
{
   MinMaxVar factor;
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

   /* look at each of the statements in the body */
   factor.min = INT_MAX;
   factor.max = 1;
   gen_context_recurse(iBody, &factor, statement_domain, gen_true, 
		       compute_parallelism_factor);

   /* Decide between min and max unroll factor, and unroll */
   if (get_bool_property("SIMD_AUTO_UNROLL_MINIMIZE_UNROLL"))
      loop_unroll(s, factor.min);
   else
      loop_unroll(s, factor.max);

   /* Do not recursively analyse the loop */
   return FALSE;
}

void simd_unroll_as_needed(statement module_stmt)
{
   /* Choose algorithm to use, and use it */
   if (get_bool_property("SIMD_AUTO_UNROLL_SIMPLE_CALCULATION"))
   {
      gen_recurse(module_stmt, statement_domain, 
		  simple_simd_unroll_loop_filter, gen_null);
   }
   else
   {
      init_tree_patterns();
      init_operator_id_mappings();
      gen_recurse(module_stmt, statement_domain, 
		  full_simd_unroll_loop_filter, gen_null);
   }
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
