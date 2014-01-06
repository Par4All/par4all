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
/*
 * HPFC module by Fabien COELHO,
 *    moved to conversion on 15 May 94
 */

/* Standard includes
 */
 
#include <stdio.h>
#include <string.h> 
#include <limits.h>

/* Psystems stuff
 */

#include "linear.h"

/* Newgen stuff
 */
#include "genC.h"
#include "ri.h" 
#include "effects.h"
/* PIPS stuff
 */

#include "misc.h"
#include "ri-util.h" 
#include "effects-util.h" 
#include "effects-generic.h"
#include "effects-convex.h"
#include "properties.h"

#include "conversion.h"

/********************************************************** TEST GENERATION */

/* expression Psysteme_to_expression(Psysteme systeme)
 *
 * From a Psysteme, a logical expression that checks for
 * the constraints is generated.
 */
expression 
Psysteme_to_expression(
    Psysteme systeme)
{
    entity
	equ = entity_intrinsic(EQUAL_OPERATOR_NAME),
	leq = entity_intrinsic(LESS_OR_EQUAL_OPERATOR_NAME);
    list conjonction;
    expression result;

    conjonction = 
	gen_nconc
	    (Pcontrainte_to_expression_list(sc_egalites(systeme), equ),
	     Pcontrainte_to_expression_list(sc_inegalites(systeme), leq));

    result = expression_list_to_conjonction(conjonction);

    gen_free_list(conjonction);
    return result;
}

list 
Pcontrainte_to_expression_list(
    Pcontrainte constraint,
    entity operator)
{
    list result = NIL;
    Pcontrainte c = NULL;
    Pvecteur vneg = VECTEUR_NUL, vpos = VECTEUR_NUL;

    for(c=constraint; c; c=c->succ)
      if (entity_undefined_p(operator))

	/* a simple expression is generated.
	 */
	result = 
	    CONS(EXPRESSION, 
	       make_vecteur_expression(contrainte_vecteur(c)),
	       result);
      else

	/* a binary operator should be given, as eg, ne, lt, le, gt, ge.
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
    
    return result;
}

/************************************************ BOUNDS FOR OPTIMIZATIONS */

/* here I store simple bounds on variables appearing in the systems
 * to allow the code generation phase to use them to improve the code.
 */
/* returns the lower and upper bounds of var if available
 * the concept could be transfered to SC ?
 */
static bool
vect_simple_definition_p(
    Pvecteur v,
    Variable *pvar,
    Value *pcoe,
    Value *pcst)
{
    Variable var;
    int size = vect_size(v);
    if (size>2 || size<1) return false;

    var = var_of(v);
    if (var==TCST && size<=1) return false;

    if (var==TCST) /* size == 2 */
    {
	*pvar = var_of(v->succ);
	*pcoe = val_of(v->succ);
	*pcst = val_of(v);
	return true;
    }
    else
    {
	*pvar = var;
	*pcoe = val_of(v);
	if (!v->succ) 
	{
	    *pcst = 0;
	    return true;
	}

	if (var_of(v->succ)!=TCST) return false;
	*pcst = val_of(v->succ);
	return true;
    }
}

GENERIC_LOCAL_FUNCTION(lowers, entity_int)
GENERIC_LOCAL_FUNCTION(uppers, entity_int)

static void
store_an_upper(
    entity var,
    int val)
{
    pips_debug(9, "%s <= %d\n", entity_local_name(var), val);

    if (bound_uppers_p(var))
    {
	int old = load_uppers(var);
	if (old>val) update_uppers(var, val);
    }
    else
	store_uppers(var, val);
}

static void
store_a_lower(
    entity var,
    int val)
{
    pips_debug(9, "%s >= %d\n", entity_local_name(var), val);

    if (bound_lowers_p(var))
    {
	int old = load_lowers(var);
	if (old<val) update_lowers(var, val);
    }
    else
	store_lowers(var, val);
}

/* I could keep the system for further optimizations...
 * Here, only direct lower and upper bounds are extracted and kept.
 */
void
set_information_for_code_optimizations(
    Psysteme s)
{
    Pcontrainte c;
    Variable var; 
    Value coe, cst, val;

    init_lowers();
    init_uppers();

    /* first look thru equalities
     */
    for (c=sc_egalites(s); c; c=c->succ)
    {
	if (vect_simple_definition_p(contrainte_vecteur(c), &var, &coe, &cst))
	{
	    val = value_div(value_uminus(cst), coe);
	    store_an_upper((entity) var, VALUE_TO_INT(val));
	    store_a_lower((entity) var, VALUE_TO_INT(val));
	}
    }

    /* then thru inequalities
     */
    for (c=sc_inegalites(s); c; c=c->succ)
    {
	if (vect_simple_definition_p(contrainte_vecteur(c), &var, &coe, &cst))
	{
	    if (value_pos_p(coe)) /* UPPER */
	    {
		Value n = value_uminus(cst),
		      r = value_pdiv(n, coe);
		store_an_upper((entity) var, VALUE_TO_INT(r));
	    }
	    else /* LOWER */
	    {
		Value n = value_minus(cst,value_plus(coe,VALUE_ONE)),
		      d = value_uminus(coe),
		      r = value_pdiv(n,d);
		store_a_lower((entity) var, VALUE_TO_INT(r));
	    }
	}
    }
}

void
reset_information_for_code_optimizations()
{
    close_lowers();
    close_uppers();
}


/* this functions returns bounds for variable var if both are available.
 */
static bool    /* whether bounds were found */
range_of_variable(
    Variable var, /* the VARiable */
    Value * lb,   /* Lower Bound */
    Value * ub)   /* Upper Bound */
{
    if (lowers_undefined_p() || uppers_undefined_p()) 
	return false; /* no information available, that's for sure */

    if (!bound_lowers_p((entity) var) || !bound_uppers_p((entity) var))
	return false;

    *lb = int_to_value(load_lowers((entity) var));
    *ub = int_to_value(load_uppers((entity) var));

    return true;
}

/* returns v lower bound if found, or INT_MIN.
 */
static Value vecteur_lower_bound(
    Pvecteur v)
{
    Value bound = 0, val;
    Variable var;

    if (lowers_undefined_p() || uppers_undefined_p()) 
	return VALUE_MIN; /* no information available, that's for sure */

    for(; v; v=v->succ)
    {
	var = var_of(v);
	val = val_of(v);

	if (var==TCST) 
	    value_addto(bound,val) ;
	else
	{
	    int il;
	    Value vl,p;
	    if (value_pos_p(val))
	    {
		if (!bound_lowers_p((entity) var)) 
		    return VALUE_MIN;
		else
		    il = load_lowers((entity) var);
	    }
	    else /* val < 0, I guess */
	    {
		if (!bound_uppers_p((entity) var))
		    return VALUE_MIN;
		else
		    il = load_uppers((entity) var);
	    }
	    
	    vl = int_to_value(il);
	    p = value_mult(val,vl);
	    value_addto(bound,p);
	}
    }

    return bound;
}

static bool
evaluate_divide_if_possible(
    Pvecteur v,
    Value denominator,
    Value *result)
{
    Value min=0, max=0;

    for(; v; v=v->succ)
    {
	Variable var = var_of(v);
	Value coef = val_of(v), lb, ub;

	if (var==TCST)
	    value_addto(min,coef), value_addto(max,coef);
	else
	{
	    Value cu,cl;
	    if (!range_of_variable(var, &lb, &ub))
		return false;
	    cu = value_mult(coef,ub);
	    cl = value_mult(coef,lb);
	    
	    if (value_pos_p(coef))
		value_addto(min,cl), value_addto(max,cu);
	    else
		value_addto(min,cu), value_addto(max,cl);
	}
    }

    value_pdivision(min, denominator);
    value_pdivision(max, denominator);

    *result = min;
    return value_eq(min,max);
}

/* expression constraints_to_loop_bound(c, var, is_lower)
 * 
 * the is_lower (lower/upper) loop bound for variable var relative
 * to Pcontrainte c is generated. All the constraints in c are used,
 * and they must be ok. 
 */
expression 
constraints_to_loop_bound(
    Pcontrainte c,  /* the constraints of the bound */
    Variable var,   /* the index variable */
    bool is_lower,  /* lower or upper bound */
    entity divide)  /* integer division to be called */
{
  int len = 0, sign = is_lower? -1: +1;
  expression result;
  list le = NIL;
  Psysteme s;
  
  pips_debug(5, "computing %ser bound for variable %s\n",
	     (is_lower?"low":"upp"), entity_local_name((entity) var));

  ifdebug(6)
  {
      fprintf(stderr, "[constraints_to_loop_bound] constraints are:\n");
      inegalites_fprint(stderr, c, (get_variable_name_t) entity_local_name);
  }

  message_assert("some constraints", !CONTRAINTE_UNDEFINED_P(c));

  /*  the constraints are sorted first, to ensure a deterministic result
   *  ??? the sorting criterion is rather strange:-)
   */
  s = sc_make(NULL, contraintes_dup(c));
  vect_sort(sc_base(s), compare_Pvecteur);
  sc_sort_constraints(s, sc_base(s));
  
  /*  each constraint is considered in turn to generate a bound
   */
  for(c=sc_inegalites(s); c; c=c->succ)
  {
      Value val = vect_coeff(var, c->vecteur), computed;
      Pvecteur vdiv = vect_del_var(c->vecteur, var), vadd = VECTEUR_NUL, v;
      expression ediv, eadd, e;

      message_assert("coherent value and sign", sign*value_sign(val)>0);

      if (value_pos_p(val)) 
	  vect_chg_sgn(vdiv);
      else
	  /*  ax+b <= 0 and a<0 => x >= (b+(-a-1))/(-a)
	   */
	  value_oppose(val), 
	  vect_add_elem(&vdiv, TCST, value_minus(val,VALUE_ONE));

      if (value_one_p(val))
      {
	  le = CONS(EXPRESSION, make_vecteur_expression(vdiv), le);
	  continue;
      }

      /* extract coefficients that are dividable by val...
       * x = (ay+bz)/a -> x = y + bz/a
       */
      for (v=vdiv; v; v=v->succ)
      {
	  Variable va = var_of(v);
	  Value vl = val_of(v);

	  if (value_zero_p(value_mod(vl,val))) 
	      vect_add_elem(&vadd, va, value_div(vl,val));
      }

      for (v=vadd; v; v=v->succ)
	  vect_erase_var(&vdiv, var_of(v));

      /* assert. no a.i=0 should have reached this point...
       */
      message_assert("some expression", vdiv || vadd);

      /* I perform some other optimizations here, by looking at
       * the extent of the numerator, that may result in a constant after
       * division by the denominator. For instance, x = y/a and 0 <= y < a
       * would lead to x = 0, which is quite simpler... I need a hook
       * from the callers of this function to retrieve the constant lower
       * and upper bounds of each variable in order to perform this.
       * ??? this optimizations should/could be perform earlier on 
       * the original system... but the implementation in a general context
       * does not seems obvious to me...
       */
      if (evaluate_divide_if_possible(vdiv, val, &computed))
      {
	  vect_rm(vdiv), vdiv=VECTEUR_NUL;
	  vect_add_elem(&vadd, TCST, computed);
      }

      if (vdiv)
      {
	  ediv = make_vecteur_expression(vdiv);

	  /* use / instead of the provided idiv if operand >=0
	   */
	  ediv = MakeBinaryCall(value_posz_p(vecteur_lower_bound(vdiv)) ? 
				entity_intrinsic(DIVIDE_OPERATOR_NAME) : 
				divide, ediv, Value_to_expression(val));
	  
	  if (vadd)
	  {
	      eadd = make_vecteur_expression(vadd);
	      e = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
				 eadd, ediv);
	  }
	  else
	      e = ediv;
      }
      else 
	  e = make_vecteur_expression(vadd);

      vect_rm(vdiv), vect_rm(vadd);
      le = CONS(EXPRESSION, e, le);
  }

  /* final operation: MAX or MIN if more than one bound
   */
  // Added an option to automatically take the constant value over the other expressions
  // Should not affect the function unless the property is set to True (False by default)
  len = gen_length(le);  message_assert("some expressions", len!=0);

  if (len==1)
  {
      result = EXPRESSION(CAR(le));
      gen_free_list(le);
  }
  else if (get_bool_property("PSYSTEME_TO_LOOPNEST_FOR_RSTREAM")) {
    int nb_constant = 0;
    FOREACH(expression, exp, le) {
      if (expression_constant_p(exp)) {
	result = exp;
	nb_constant++;
      }
    }
    gen_free_list(le);
    pips_assert("More than one constant expression in loop bounds", nb_constant == 1);
  }
  else
  {
      entity operator = entity_intrinsic(is_lower ? "MAX" : "MIN");
      result = make_call_expression(operator, le);
  }

  sc_rm(s);

  return result;
}

/* this function checks whether the lower and upper constraints
 * are going to generate the same bound on variable var.
 */
bool 
bounds_equal_p(
    Variable var,
    Pcontrainte lower,
    Pcontrainte upper)
{
    Pvecteur v_lower, v_upper, sum;
    Value val_lower, val_upper, the_ppcm;
    bool result;

    if (nb_elems_list(lower)!=1 || nb_elems_list(upper)!=1) return(false);

    val_upper = vect_coeff(var, upper->vecteur); /*coeff for var in the constraint*/
    val_lower = vect_coeff(var, lower->vecteur);
    
    /* ??? the arithmetic ppcm version is on int instead of values 
     */
    the_ppcm = ppcm(value_uminus(val_lower), val_upper);

    v_lower = vect_dup(lower->vecteur);
    v_lower = vect_multiply(v_lower, 
			    value_div(value_uminus(the_ppcm),val_lower));

    v_upper = vect_dup(upper->vecteur);
    v_upper = vect_multiply(v_upper, 
			    value_div(the_ppcm,val_upper));

    sum = vect_add(v_lower, v_upper);
    vect_add_elem(&sum, TCST, value_minus(the_ppcm,VALUE_ONE));
    vect_normalize(sum);

    result = VECTEUR_NUL_P(sum) || 
	(sum->succ==NULL && var_of(sum)==TCST && val_of(sum)==0);

    vect_rm(v_lower), vect_rm(v_upper), vect_rm(sum);

    return result;
}

/* sc is used to generate the loop nest bounds for variables vars.
 * vars may be empty. the loop statement is returned.
 *
 * sc is not touched...
 */
statement 
systeme_to_loop_nest(
    Psysteme sc,
    list /* of entity */ vars,
    statement body,
    entity divide) /* I have to give the divide entity to be called */
{
    range rg;
    Pcontrainte	c, lower, upper;
    list reverse;
    statement assign, current = body;
    Psysteme s;

    if (ENDP(vars)) return body;

    s = sc_dup(sc);   /* duplicate sc*/
    sc_transform_eg_in_ineg(s);  /* ??? could do a better job with = */
    c = sc_inegalites(s);    

    reverse = gen_nreverse(gen_copy_seq(vars));  /* reverse the list of vars*/

    message_assert("no equalities, now", sc_nbre_egalites(s)==0);

    FOREACH(ENTITY,e,reverse)
    {
        Variable var = (Variable) e;

        pips_debug(5, "variable %s loop\n", entity_name((entity) var));

        constraints_for_bounds(var, &c, &lower, &upper);
        if( !CONTRAINTE_UNDEFINED_P(lower) && !CONTRAINTE_UNDEFINED_P(upper) )
        {

            if (bounds_equal_p(var, lower, upper))
            {
                /*   VAR = LOWER
                 *   body
                 */
                assign = 
                    make_assign_statement(entity_to_expression((entity) var),
                            constraints_to_loop_bound(lower, var, 
                                true, divide));
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
                            true, divide),
                        constraints_to_loop_bound(upper, var, 
                            false, divide),
                        int_to_expression(1));

                current = 
                    instruction_to_statement
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
        }
    }

    gen_free_list(reverse);
    sc_inegalites(s)=c, sc_rm(s);

    return current;
}

/* statement generate_optional_if(sc, stat)
 *
 * if sc is Z^n then no if is required,
 * if sc is empty, then statement is nop,
 * else an if is required
 */
statement 
generate_optional_if(
    Psysteme sc,
    statement stat)
{
    if (sc_rn_p(sc)) return(stat);
    if (sc_empty_p(sc)) return(make_empty_statement());

    return st_make_nice_test(Psysteme_to_expression(sc),
			     CONS(STATEMENT, stat, NIL),
			     NIL);
}

/* that is all
 */
