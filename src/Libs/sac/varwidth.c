
#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "sac.h"

int effective_reference_width(reference r)
{
   basic b;
   type t = entity_type(reference_variable(r));

   if (!type_variable_p(t))
      return 1000; /* big number, to ensure no parallelization */
   b = variable_basic(type_variable(t));

   switch(basic_tag(b))
   {
      case is_basic_int:
	 return basic_int(b);

      case is_basic_float:
	 return basic_float(b);

      case is_basic_logical:
	 return basic_logical(b);

      default:
	 return 1000; /* big number, to ensure no parallelization */
   }
}

int effective_call_variables_width(call c)
{
   cons * args;
   int maxArgsWidth;

   maxArgsWidth = 0;

   /* Look at each argument */
   for( args = call_arguments(c);
        args != NIL;
	args = CDR(args) )
   {
      int curArgWidth;
      syntax s;

      /* make sure the argument is a reference */
      s = expression_syntax(EXPRESSION(CAR(args)));
      if (!syntax_reference_p(s))
      {
	 /* we are supposed to be in 3-form, so this should not happen */
	 printf("WARNING: effective_variable_width called on non 3-form"
		"call statemnent.\n");
	 return 1000; /* big number, to ensure no parallelization */
      }
      
      /* compute the actual width of the parameter */
      curArgWidth = effective_reference_width(syntax_reference(s));

      /* infer the maximum args width */
      if (curArgWidth > maxArgsWidth)
	 maxArgsWidth = curArgWidth;
   }

   return maxArgsWidth;
}

int effective_variables_width(instruction i)
{
   switch(instruction_tag(i))
   {
      case is_instruction_sequence:
      {
	 cons * j;
	 int maxVarWidth = 0;

	 for( j = sequence_statements(instruction_sequence(i));
	      j != NIL;
	      j = CDR(j) )
	 {
	    int varwidth = 
	       effective_variables_width(statement_instruction(STATEMENT(CAR(j))));

	    if (varwidth > maxVarWidth)
	       maxVarWidth = varwidth;
	 }
	 
	 return maxVarWidth;
      }

      case is_instruction_test:
      case is_instruction_loop:
      case is_instruction_whileloop:
      case is_instruction_goto:
      case is_instruction_unstructured:
      default:
	 printf("WARNING: complex loop !\n");
	 return 1000; /* big number, to ensure no parallelization */

      case is_instruction_call:
	 return effective_call_variables_width(instruction_call(i));
   }
}


