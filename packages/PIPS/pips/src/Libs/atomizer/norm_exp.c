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
/* -- norm_exp.c
 *
 * package atomizer :  Alexis Platonoff, aout 91
 * --
 *
 * Functions for the normalization of expressions.
 *
 * Before the atomization, all the expression of the Code are put
 * in a normal form. This normal form gathers the NLCs
 * variables in the innermost parenthesis with the constant term.
 * The NLCs are sorted in order to have the innerloop counter in the
 * inner parenthesis.
 *
 * Thus, the following expression:
 *      (4*S + ( NLC1 + ((T + 7) + 3*NLC2))) + C) + 8*NLC3
 * is normalized in:
 *      4*S + (T + (C + (8*NLC1 + (3*NLC2 + (NLC3 + 7)))))
 *
 * For more information about NLCs, see loop_normalize package.
 */
/* #include "loop_normalize.h" */
#define NLC_PREFIX 			"NLC"
#define ENTITY_NLC_P(e) (strncmp(entity_local_name(e), NLC_PREFIX, 3) == 0)

#include "local.h"


/*============================================================================*/
/* void normal_expression_of_expression(expression exp): normalizes "exp".
 *
 * There are three cases:
 *       1. "exp" is a call to an intrinsic or external function.
 *       2. "exp" is a reference.
 *	 3. "exp" is a range
 *
 * In the case (1), if "exp" is linear (ie normalized), we call
 * reconfig_expression(); it computes the normalization of "exp" with its
 * Pvecteur. If "exp" is not integer linear, this function is called
 * recursively upon the arguments of the call.
 *
 * In case (2), we call this function upon each of the indices of "exp".
 *
 * In case (3), we do nothing. Such a case may occur with a range argument in
 * a call to a write or read procedure.
 */
void normal_expression_of_expression(exp)
expression exp;
{
syntax sy = expression_syntax(exp);
if(syntax_tag(sy) == is_syntax_call)
  {
  call c = syntax_call(sy);
  if(! call_constant_p(c))
    {
    NORMALIZE_EXPRESSION(exp);
    if(normalized_tag(expression_normalized(exp)) == is_normalized_linear)
      {
      pips_debug(5, "Expression Linear : %s\n",
	    words_to_string(words_expression(exp, NIL)));

      reconfig_expression(exp);
      }
    else
      {
      list args = call_arguments(c);

      pips_debug(5, "Expression Complex : %s\n",
		 words_to_string(words_expression(exp, NIL)));

      for(; args != NIL; args = CDR(args))
	normal_expression_of_expression(EXPRESSION(CAR(args)));
      }
    expression_normalized(exp) = normalized_undefined;
    }
  }
else if(syntax_tag(sy) == is_syntax_reference)
  {
  reference ref = syntax_reference(sy);
  list inds = reference_indices(ref);
  for(; inds != NIL; inds = CDR(inds))
    normal_expression_of_expression(EXPRESSION(CAR(inds)));
  }
else if(syntax_tag(sy) == is_syntax_range)
  {
   pips_debug(6, "Expression Range : %s\n",
	      words_to_string(words_expression(exp, NIL)));
  }
else
  pips_internal_error("Bad expression tag");
}



/*============================================================================*/
/* void normal_expression_of_statement(statement s): normalizes the
 * expressions contained in "s".
 */
void normal_expression_of_statement(s)
statement s;
{
instruction inst = statement_instruction(s);

debug(4, "normal_expression_of_statement", "begin STATEMENT\n");

switch(instruction_tag(inst))
  {
  case is_instruction_block :
  { list block = instruction_block(inst);
    for(; block != NIL ; block = CDR(block))
      normal_expression_of_statement(STATEMENT(CAR(block)));
    break; }
  case is_instruction_test :
  { test t = instruction_test(inst);
    normal_expression_of_expression(test_condition(t));
    normal_expression_of_statement(test_true(t));
    normal_expression_of_statement(test_false(t));
    break; }
  case is_instruction_loop :
  { loop l = instruction_loop(inst);
    range lr = loop_range(l);
    normal_expression_of_expression(range_lower(lr));
    normal_expression_of_expression(range_upper(lr));
    normal_expression_of_expression(range_increment(lr));
    normal_expression_of_statement(loop_body(l));
    break; }
  case is_instruction_call : 
  { call c = instruction_call(inst);
    list args = call_arguments(c);

    debug(4, "normal_expression_of_statement", "Stmt CALL: %s\n",
             entity_local_name(call_function(c)));

    for(; args != NIL; args = CDR(args))
      normal_expression_of_expression(EXPRESSION(CAR(args)));
    break; }
  case is_instruction_goto : break;
  case is_instruction_unstructured :
  { normal_expression_of_unstructured(instruction_unstructured(inst));
    break; }
  default : pips_internal_error("Bad instruction tag");
  }
debug(4, "normal_expression_of_statement", "end STATEMENT\n");
}



/*============================================================================*/
/* void normal_expression_of_unstructured(unstructured u): normalizes the
 * expressions of an unstructured instruction "u".
 */
void normal_expression_of_unstructured(u)
unstructured u;
{
list blocs = NIL;

debug(3, "normal_expression_of_unstructured", "begin UNSTRUCTURED\n");

CONTROL_MAP(c, { normal_expression_of_statement(control_statement(c)) ; },
                 unstructured_control( u ), blocs);

gen_free_list(blocs);

debug(3, "normal_expression_of_unstructured", "end UNSTRUCTURED\n");
}



/*============================================================================*/
/* int get_nlc_number(entity nlc_ent): returns the number ending "nlc_ent"
 * name.
 * The local name is "NLC#", so we have to get the "#".
 */
int get_nlc_number(nlc_ent)
entity nlc_ent;
{
const char* nlc_name = entity_local_name(nlc_ent);
const char* num = nlc_name+3;

return(atoi(num));
}



/*============================================================================*/
/* static Pvecteur config_vecteur(Pvecteur Vvar): returns a Pvecteur resulting
 * of the configuration of the Pvecteur "Vvar".
 *
 * Firstly, we put into three different Pvecteurs the constant term, the NLCs
 * variables and the others variables (not NLCs).
 * The NLCs variables are ordered from the greater to the smaller number.
 *
 * Secondly, we concatenate these three Pvecteurs in the order:
 * constant_term, nlc, not_nlc.
 *
 * For example, with Pvecteur:
 *   2*I  3*NLC2  1*NLC1  4*T  7
 * we could obtain:
 *   7  3*NLC2  1*NLC1  4*T  2*I
 */
static Pvecteur config_vecteur(Vvar)
Pvecteur Vvar;
{
Pvecteur Vnot_nlc, Vnlc, Vterm_cst, newV, Vc, Vaux;

Vterm_cst = VECTEUR_NUL;
Vnlc = VECTEUR_NUL;
Vnot_nlc = VECTEUR_NUL;
for(Vc = Vvar; !VECTEUR_NUL_P(Vc); Vc = Vc->succ)
  {
  /* "Vc" is the constant term. */
  if(term_cst(Vc))
    Vterm_cst = vect_new(Vc->var, Vc->val);
  else
    {
    entity var = (entity) Vc->var;
    Pvecteur new_vect = vect_new(Vc->var, Vc->val);

    /* "Vc" is a NLC. */
    if(ENTITY_NLC_P(var))
      {
      int num, crt_num = 0;

      if(VECTEUR_NUL_P(Vnlc))
        Vnlc = new_vect;
      else
        {
        num = get_nlc_number(var);

        crt_num = get_nlc_number((entity) Vnlc->var);
        if(num > crt_num)
          {
          new_vect->succ = Vnlc;
          Vnlc = new_vect;
          }
        else
          {
          bool not_fin = true;
          Pvecteur Vs = Vnlc, Vp;
          while(not_fin)
            {
            Vp = Vs;
            Vs = Vs->succ;
            if(!VECTEUR_NUL_P(Vs))
              {
              crt_num = get_nlc_number((entity) Vs->var);
              if(num > crt_num)
                {
                new_vect->succ = Vs;
                Vp->succ = new_vect;
                not_fin = false;
                }
              }
            else
              {
              Vp->succ = new_vect;
              not_fin = false;
              }
            }
          }
        }
      }

    /* "Vc" is not a NLC. */
    else
      {
      new_vect->succ = Vnot_nlc;
      Vnot_nlc = new_vect;
      }
    }
  }

newV = Vnot_nlc;

if(!VECTEUR_NUL_P(Vnlc))
  {
  Vc = Vnlc;
  while(!VECTEUR_NUL_P(Vc))
    {
    Vaux = Vc;
    Vc = Vc->succ;
    }
  Vaux->succ = newV;
  newV = Vnlc;
  }

if(!VECTEUR_NUL_P(Vterm_cst))
  {
  Vterm_cst->succ = newV;
  newV = Vterm_cst;
  }

return(newV);
}



/*============================================================================*/
/* void reconfig_expression(expression exp): "exp" is reordered so as to gather
 * all the NLCs in the innermost parenthesis. More, the NLC of the inner loop
 * is in the innermost parenthesis with the TCST (constant term).
 * For example, if we have:
 *      (4*S + ( NLC1 + ((T + 7) + 3*NLC2))) + C) + 8*NLC3
 * it is reordered in:
 *      4*S + (T + (C + (8*NLC1 + (3*NLC2 + (NLC3 + 7)))))
 *
 * Called functions:
 *       _ Pvecteur_to_expression() : loop_normalize/utils.c
 */
void reconfig_expression(exp)
expression exp;
{
Pvecteur vect, new_vect;
syntax sy = expression_syntax(exp);
normalized nor = NORMALIZE_EXPRESSION(exp);
expression new_exp;

if(normalized_tag(nor) != is_normalized_linear)
  return;
if(syntax_tag(sy) == is_syntax_reference)
  return;
if(call_constant_p(syntax_call(sy)))
  return;

vect = (Pvecteur) normalized_linear(nor);

/* We configurate the Pvecteur of "exp". */
new_vect = config_vecteur(vect);

/* We build a new expression with the configurated Pvecteur. */
new_exp = Pvecteur_to_expression(new_vect);

/* We change the syntax of "exp". */
if(new_exp != expression_undefined)
  expression_syntax(exp) = expression_syntax(new_exp);
}
