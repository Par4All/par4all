/* -- utils.c
 *
 * package atomizer :  Alexis Platonoff, juillet 91
 * --
 *
 * Miscellaneous functions
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "graph.h"
#include "dg.h"
/* #include "database.h" */

#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "list.h"

#include "graph.h"
#include "dg.h"
#include "loop_normalize.h"
#include "atomizer.h"


/*============================================================================*/
/* bool expression_intrinsic_operation_p(expression exp): Returns TRUE
 * if "exp" is an expression with a call to an intrinsic operation.
 */
bool expression_intrinsic_operation_p(exp)
expression exp;
{
entity e;
syntax syn = expression_syntax(exp);
if (syntax_tag(syn) != is_syntax_call)
  return (FALSE);

e = call_function(syntax_call(syn));

return(value_tag(entity_initial(e)) == is_value_intrinsic);
}



/*============================================================================*/
/* bool call_constant_p(call c): Returns TRUE if "c" is a call to a constant,
 * that is, a constant number or a symbolic constant.
 */
bool call_constant_p(c)
call c;
{
value cv = entity_initial(call_function(c));

return( (value_tag(cv) == is_value_constant) ||
        (value_tag(cv) == is_value_symbolic)   );
}



/*============================================================================*/
/* bool instruction_in_list_p(instruction inst, list l): Returns TRUE if "inst"
 * is in "l".
 *
 * Note: "l" must be a list of instructions.
 */
bool instruction_in_list_p(inst, l)
instruction inst;
list l;
{
bool not_found = TRUE;
 
while((not_found) && (l != NIL))
  {
  instruction current_inst = INSTRUCTION(CAR(l));
  if (inst == current_inst)
    not_found = FALSE;
  else
    l = CDR(l);
  }   
return (! not_found);
}



/*============================================================================*/
/* bool nlc_linear_expression_p(expression exp): returns TRUE if "exp" is an
 * integer linear expression with only NLCs variables.
 *
 * NLC means Normalized Loop Counter.
 *
 * Called functions :
 *       _ unnormalize_expression() : loop_normalize/utils.c
 */
bool nlc_linear_expression_p(exp)
expression exp;
{
Pvecteur vect;
bool ONLY_NLCs;

debug(7, "nlc_linear_expression_p", "exp : %s\n",
      words_to_string(words_expression(exp)));

if(normalized_tag(NORMALIZE_EXPRESSION(exp)) == is_normalized_complex)
  ONLY_NLCs = FALSE;
else
  {
  vect = (Pvecteur) normalized_linear(expression_normalized(exp));
  ONLY_NLCs = TRUE;

  for(; !VECTEUR_NUL_P(vect) && ONLY_NLCs ; vect = vect->succ)
    {
    entity var = (entity) vect->var;

    if( ! term_cst(vect) )
      if( ! (ENTITY_NLC_P(var)) )
        ONLY_NLCs = FALSE;
    }
  }

/* We unnormalize the expression, otherwise, it causes an error when we
 * normalize an expression with sub-expressions already normalized.
 * (cf. NormalizedExpression() in normalize/normalize.c).
 *
 * This unnormalization is done recursively upon all the sub-expressions of
 * "exp".
 */
unnormalize_expression(exp);

debug(7, "nlc_linear_expression_p", "   result : %d\n", ONLY_NLCs);

return(ONLY_NLCs);
}

