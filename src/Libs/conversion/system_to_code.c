/*
 * HPFC module by Fabien COELHO,
 *    moved to conversion on 15 May 94
 *
 * SCCS stuff:
 * $RCSfile: system_to_code.c,v $ ($Date: 1994/11/29 08:31:54 $, ) version $Revision$, 
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

/* expression constraints_to_loop_bound(c, var, is_lower)
 * Pcontrainte c;
 * Variable var;
 * bool is_lower;
 * entity divide;
 * 
 * the is_lower (lower/upper) loop bound for variable var relative
 * to Pcontrainte c is generated. All the constraints in c are used,
 * and they must be ok. 
 */
expression constraints_to_loop_bound(c, var, is_lower, divide)
Pcontrainte c;
Variable var;
bool is_lower;
entity divide;
{
  int 
    len = 0,
    sign = is_lower? -1: +1;
  entity
    operator = is_lower? CreateIntrinsic("MAX"): CreateIntrinsic("MIN");
  expression
    result = expression_undefined;
  list
    le = NIL;
  
  debug(5, "constraints_to_loop_bound",
	"computing %ser bound for variable %s\n",
	(is_lower?"low":"upp"), entity_local_name((entity) var));

  ifdebug(6)
  {
      fprintf(stderr, "[constraints_to_loop_bound] constraints are:\n");
      inegalites_fprint(stderr, c, entity_local_name);
  }

  pips_assert("constraints_to_loop_bound", !CONTRAINTE_UNDEFINED_P(c));

  /*  each contraint is considered in turn to generate the bound
   */
  for(;
      c!=(Pcontrainte) NULL;
      c=c->succ)
    {
      Value 
	val = vect_coeff(var, c->vecteur);
      Pvecteur
	v = vect_del_var(c->vecteur, var);
      expression
	e = expression_undefined;

      pips_assert("constraints_to_loop_bound", sign*val>0);

      if (val>0) 
	  vect_chg_sgn(v);
      else
	  /*  ax+b <= 0 and a<0 => x >= (b+(-a-1))/(-a)
	   */
	  val=-val, vect_add_elem(&v, TCST, val-1);

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

/* this function checks whether the lower and upper constraints
 * are going to generate the same bound on variable var.
 */
bool bounds_equal_p(var, lower, upper)
Variable var;
Pcontrainte lower, upper;
{
    Pvecteur
	v_lower,
	v_upper,
	sum;
    Value
	val_lower, 
	val_upper,
	the_ppcm;
    bool
	result;

    if (nb_elems_list(lower)!=1 || nb_elems_list(upper)!=1) return(FALSE);

    val_upper = vect_coeff(var, upper->vecteur);
    val_lower = vect_coeff(var, lower->vecteur);
    
    /* ??? the arithmetic ppcm version is on int instead of values 
     */
    the_ppcm = ppcm(-val_lower,val_upper);

    v_lower = vect_dup(lower->vecteur);
    v_lower = vect_multiply(v_lower, -the_ppcm/val_lower);

    v_upper = vect_dup(upper->vecteur);
    v_upper = vect_multiply(v_upper, the_ppcm/val_upper);

    sum = vect_add(v_lower, v_upper);
    vect_add_elem(&sum, TCST, the_ppcm-1);
    vect_normalize(sum);

    result = VECTEUR_NUL_P(sum) || 
	(sum->succ==NULL && var_of(sum)==TCST && val_of(sum)==0);

    vect_rm(v_lower), vect_rm(v_upper), vect_rm(sum);

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
    range rg;
    Variable var;
    Pcontrainte	
	c, lower, upper;
    list reverse;
    statement 
	assign,
	current = body;
    
    if (ENDP(vars)) return(body);

    c = contraintes_dup(sc_inegalites(sc));
    reverse = gen_nreverse(gen_copy_seq(vars));

    pips_assert("Psysteme_to_loop_nest", (sc_nbre_egalites(sc)==0));
    
    MAPL(ce,
     {
	 var = (Variable) ENTITY(CAR(ce));
	 
	 debug(5, "systeme_to_loop_nest",
	       "variable %s loop\n", entity_name((entity) var));
	 
	 constraints_for_bounds(var, &c, &lower, &upper);

	 if (bounds_equal_p(var, lower, upper))
	 {
	     /*   VAR = LOWER
	      *   body
	      */
	     assign = 
		 make_assign_statement(entity_to_expression((entity) var),
				       constraints_to_loop_bound(lower, var, 
								 TRUE, divide));
	     current = 
		 make_block_statement(CONS(STATEMENT, assign,
				      CONS(STATEMENT, current,
					   NIL)));

	 }
	 else
	 {
	     /*   DO VAR = LOWER, UPPER, 1
	      *     body
	      *   ENDDO
	      */
	     rg = make_range(constraints_to_loop_bound(lower, var, 
						       TRUE, divide),
			     constraints_to_loop_bound(upper, var, 
						       FALSE, divide),
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
	 }

	 contraintes_free(lower);
	 contraintes_free(upper);
     },
	 reverse);
    
    gen_free_list(reverse);
    contraintes_free(c);

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
