/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: system_to_code.c,v $ ($Date: 1994/04/11 17:01:24 $, ) version $Revision$, 
 * got on %D%, %T%
 * $Id$
 */

/*
 * Standard includes
 */
 
#include <stdio.h>
#include <string.h> 
extern fprintf();

/*
 * Psystems stuff
 */

#include "types.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/*
 * Newgen stuff
 */

#include "genC.h"

#include "ri.h" 
#include "hpf.h" 
#include "hpf_private.h"

/*
 * PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "control.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"

/* in syntax.h */
entity CreateIntrinsic(string name);

/* 
 * my own local includes
 */

#include "hpfc.h"
#include "defines-local.h"

/*
 * ONE ARRAY REFERENCES MODIFICATIONS
 */


/*
 * TEST GENERATION
 */

/*
 * expression Psysteme_to_expression(Psysteme systeme)
 *
 * From a Psysteme, a logical expression that checks for
 * the constraints is generated.
 */
expression Psysteme_to_expression(systeme)
Psysteme systeme;
{
    entity
	equ = local_name_to_top_level_entity(EQUAL_OPERATOR_NAME),
	leq = local_name_to_top_level_entity(LESS_OR_EQUAL_OPERATOR_NAME);
    list
	conjonction = 
	    gen_nconc
		(Pcontrainte_to_expression_list(sc_egalites(systeme), equ),
		 Pcontrainte_to_expression_list(sc_inegalites(systeme), leq));
    expression
	result = expression_list_to_conjonction(conjonction);

    gen_free_list(conjonction);
    return(result);
}

/*
 * 
 */
list Pcontrainte_to_expression_list(constraint, operator)
Pcontrainte constraint;
entity operator;
{
    list
	result = NIL;
    Pcontrainte
	c = NULL;
    Pvecteur
      vneg = VECTEUR_NUL,
      vpos = VECTEUR_NUL;

    for(c=constraint;
	!CONTRAINTE_UNDEFINED_P(c);
	c=c->succ)
      if (entity_undefined_p(operator))
	/* 
	 * a simple expression is generated.
	 */
	result = 
	  CONS(EXPRESSION, 
	       make_vecteur_expression(contrainte_vecteur(c)),
	       result);
      else
	/*
	 * a binary operator should be given, as eg, ne, lt, le, gt, ge.
	 */
	Pvecteur_separate_on_sign(contrainte_vecteur(c), &vpos, &vneg),
	result = 
	  CONS(EXPRESSION,
	       MakeBinaryCall(operator,
			      make_vecteur_expression(vpos),
			      make_vecteur_expression(vneg)),
	       result),
	vect_rm(vpos), vpos=VECTEUR_NUL,
        vect_rm(vneg), vneg=VECTEUR_NUL;
    
    return(result);
}

/* expression contrainte_to_loop_bound(pc, var, is_lower)
 * Pcontrainte *pc;
 * Variable var;
 * bool is_lower;
 * 
 * the is_lower (lower/upper) loop bound for variable var relative
 * to Pcontrainte *pc is generated, and the used constraints are
 * removed from the set.
 */
expression contrainte_to_loop_bound(pc, var, is_lower)
Pcontrainte *pc;
Variable var;
bool is_lower;
{
  int 
    len = 0,
    sign = is_lower? +1: -1;
  entity
    operator = is_lower? CreateIntrinsic("MAX"): CreateIntrinsic("MIN");
  Pcontrainte
    new = CONTRAINTE_UNDEFINED,
    goods = CONTRAINTE_UNDEFINED,
    others = CONTRAINTE_UNDEFINED,
    c = CONTRAINTE_UNDEFINED;
  entity
    divide = CreateIntrinsic("DIVIDE");
  expression
    result = expression_undefined;
  list
    le = NIL;
  
  /* constraints to be considered are first put in goods or others.
   */
  for(c=*pc;
      c!=(Pcontrainte) NULL;
      c=c->succ)
    if (sign*vect_coeff(c->vecteur, var)>=0)
      new = contrainte_make(vect_dup(c->vecteur)),
      new->succ = goods,
      goods = new;
    else
      new = contrainte_make(vect_dup(c->vecteur)),
      new->succ = others,
      others = new;

  *pc = (contrainte_rm(*pc), others);

  /*
   * 'goods' are used to generate the bounds
   */
  for(c=goods;
      c!=(Pcontrainte) NULL;
      c=c->succ)
    {
      Value 
	val = vect_coeff(c->vecteur);
      Pvecteur
	v = vect_del_var(c->vecteur, var);
      expression
	e = expression_undefined;

      pips_assert("contrainte_to_loop_bound", val!=0);

      if (val<0) vect_chg_sgn(v), val=-val;

      e = make_vecteur_expression(v);
      if (val!=1) e = MakeBinaryCall(divide, e, int_to_expression(val));

      le = CONS(EXPRESSION, e, NIL);
    }

  /*
   * final operation
   */
  len = gen_length(le);  pips_assert("contrainte_to_loop_bound", len!=0);

  /* NO, MIN and MAX should take any number of arguments in fortran!
   * // result = expression_list_to_binay_operator_call(le, operator); 
   */
  result = len==1? EXPRESSION(CAR(le)): make_call_expression(operator, le);

  if (len==1) gen_free_list(le);

  return(result);
}

/*
 * sc is used to generate the loop nest bounds for variables vars.
 * the loop statement is returned, plus a pointer to the body statement.
 */
statement systeme_to_loop_nest(sc, vars, pinner)
Psysteme sc;
list vars;
statement *pinner;
{
  Pcontrainte
    c = sc_inegalites(sc);
  list 
    reverse = gen_nreverse(gen_copy_seq(vars));
  statement 
    inner = statement_undefined,
    current = statement_undefined;

  pips_assert("Psysteme_to_loop_nest", 
	      (sc_nbre_egalites(sc)==0) && (gen_length(vars)>0));

  MAPL(ce,
    {
       Variable
	 var = (Variable) ENTITY(CAR(ce));
       range
	 rg = make_range(contrainte_to_loop_bound(&c, var, TRUE),
			 contrainte_to_loop_bound(&c, var, FALSE),
			 int_to_expression(1));

       current = 
	 mere_statement
	   (make_instruction
	    (is_instruction_loop,
	     make_loop((entity) var,
		       rg, 
		       current,
		       entity_empty_label(),
		       make_execution(is_execution_sequential, UU),
		       NIL)));

       if (statement_undefined_p(inner)) inner=current;
    },
       reverse);

  gen_free_list(reverse);

  *pinner=inner;
  return(current);
}


/*
 * that's all
 */
