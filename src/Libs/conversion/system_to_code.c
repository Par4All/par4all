/*
 * HPFC module by Fabien COELHO,
 *    moved to conversion on 15 May 94
 *
 * SCCS stuff:
 * $RCSfile: system_to_code.c,v $ ($Date: 1994/06/03 13:50:33 $, ) version $Revision$, 
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

#include "conversion.h"

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
 * entity divide;
 * 
 * the is_lower (lower/upper) loop bound for variable var relative
 * to Pcontrainte *pc is generated, and the used constraints are
 * removed from the set.
 */
expression contrainte_to_loop_bound(pc, var, is_lower, divide)
Pcontrainte *pc;
Variable var;
bool is_lower;
entity divide;
{
  int 
    len = 0,
    sign = is_lower? -1: +1;
  entity
    operator = is_lower? CreateIntrinsic("MAX"): CreateIntrinsic("MIN");
  Pcontrainte
    new = CONTRAINTE_UNDEFINED,
    goods = CONTRAINTE_UNDEFINED,
    others = CONTRAINTE_UNDEFINED,
    c = CONTRAINTE_UNDEFINED;
  expression
    result = expression_undefined;
  list
    le = NIL;
  
  debug(5, "contrainte_to_loop_bound",
	"computing %ser bound for variable %s\n",
	(is_lower?"low":"upp"), entity_local_name((entity) var));

  ifdebug(6)
  {
      fprintf(stderr, "[contrainte_to_loop_bound] constraints are:\n");
      inegalites_fprint(stderr, *pc, entity_local_name);
  }

  /* constraints to be considered are first put in goods or others.
   */
  for(c=*pc;
      c!=(Pcontrainte) NULL;
      c=c->succ)
    if (sign*vect_coeff(var, c->vecteur)>0)
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
	val = vect_coeff(var, c->vecteur);
      Pvecteur
	v = vect_del_var(c->vecteur, var);
      expression
	e = expression_undefined;

      pips_assert("contrainte_to_loop_bound", val!=0);

      if (val>0) 
	  vect_chg_sgn(v);
      else
	  val=-val;

      e = make_vecteur_expression(v);
      if (val!=1) e = MakeBinaryCall(divide, e, int_to_expression(val));

      le = CONS(EXPRESSION, e, le);
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
 * vars may be empty. the loop statement is returned.
 *
 * sc is not touched...
 */
statement systeme_to_loop_nest(sc, vars, body, divide)
Psysteme sc;
list vars;
statement body;
entity divide; /* I have to give the divide entity to be called */
{
    Psysteme
	t = sc_dup(sc);
    Pcontrainte
	c = sc_inegalites(t);
    list 
	reverse = gen_nreverse(gen_copy_seq(vars));
    statement 
	current = body;
    
    pips_assert("Psysteme_to_loop_nest", (sc_nbre_egalites(t)==0));

    sc_inegalites(t)=CONTRAINTE_UNDEFINED;
    
    MAPL(ce,
     {
	 Variable
	     var = (Variable) ENTITY(CAR(ce));
	 range
	     rg = range_undefined;
	 
	 debug(5, "systeme_to_loop_nest",
	       "variable %s loop\n", entity_name((entity) var));
	 
	 rg = make_range(contrainte_to_loop_bound(&c, var, TRUE, divide),
			 contrainte_to_loop_bound(&c, var, FALSE, divide),
			 int_to_expression(1));
	 
	 current = 
	     make_stmt_of_instr
		 (make_instruction
		  (is_instruction_loop,
		   make_loop((entity) var,
			     rg, 
			     current,
			     entity_empty_label(),
			     make_execution(is_execution_sequential, UU),
			     NIL)));
	 
     },
	 reverse);
    
    gen_free_list(reverse);
    contraintes_free(c);
    sc_rm(t);
    
    return(current);
}

/*
 * statement generate_optional_if(sc, stat)
 *
 * if sc is Z^n then no if is required,
 * if sc is empty, then statement is nop,
 * else an if is required
 */
statement generate_optional_if(sc, stat)
Psysteme sc;
statement stat;
{
    if (sc_rn_p(sc)) return(stat);
    if (sc_empty_p(sc)) return(make_empty_statement());

    return(st_make_nice_test(Psysteme_to_expression(sc),
			     CONS(STATEMENT, stat, NIL),
			     NIL));
}

/*
 * that's all
 */
