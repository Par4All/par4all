/* -- utils.c
 *
 * package atomizer :  Alexis Platonoff, juillet 91
 * --
 *
 * Miscellaneous functions
 */

#include "local.h"

#define NLC_PREFIX 			"NLC"
#define ENTITY_NLC_P(e) (strncmp(entity_local_name(e), NLC_PREFIX, 3) == 0)




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

