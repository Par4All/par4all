/*
 * HPFC module by Fabien COELHO,
 *    moved to conversion on 15 May 94
 *
 * SCCS stuff:
 * $RCSfile: system_to_code.c,v $ ($Date: 1995/10/03 16:28:10 $, ) version $Revision$, 
 * got on %D%, %T%
 * $Id$
 */

/* Standard includes
 */
 
#include <stdio.h>
#include <string.h> 

/* Psystems stuff
 */

#include "types.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Newgen stuff
 */

#include "genC.h"

#include "ri.h" 

/* PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "control.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"

entity CreateIntrinsic(string name);/* in syntax.h */

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
	equ = local_name_to_top_level_entity(EQUAL_OPERATOR_NAME),
	leq = local_name_to_top_level_entity(LESS_OR_EQUAL_OPERATOR_NAME);
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
static boolean
vect_simple_definition_p(
    Pvecteur v,
    Variable *pvar,
    Value *pcoe,
    Value *pcst)
{
    Variable var;
    int size = vect_size(v);
    if (size>2 || size<1) return FALSE;

    var = var_of(v);
    if (var==TCST && size<=1) return FALSE;

    if (var==TCST) /* size == 2 */
    {
	*pvar = var_of(v->succ);
	*pcoe = val_of(v->succ);
	*pcst = val_of(v);
	return TRUE;
    }
    else
    {
	*pvar = var;
	*pcoe = val_of(v);
	if (!v->succ) 
	{
	    *pcst = 0;
	    return TRUE;
	}

	if (var_of(v->succ)!=TCST) return FALSE;
	*pcst = val_of(v->succ);
	return TRUE;
    }
}

GENERIC_LOCAL_FUNCTION(lowers, entity_int)
GENERIC_LOCAL_FUNCTION(uppers, entity_int)

static void
store_an_upper(
    entity var,
    int val)
{
    pips_debug(5, "%s <= %d\n", entity_local_name(var), val);

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
    pips_debug(5, "%s >= %d\n", entity_local_name(var), val);

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
	    val = DIVIDE(cst, coe);
	    store_an_upper((entity) var, (int) val);
	    store_a_lower((entity) var, (int) val);
	}
    }

    /* then thru inequalities
     */
    for (c=sc_inegalites(s); c; c=c->succ)
    {
	if (vect_simple_definition_p(contrainte_vecteur(c), &var, &coe, &cst))
	{
	    if (coe>0) /* UPPER */
		store_an_upper((entity) var, 
			       (int) POSITIVE_DIVIDE(-cst, coe));
	    else /* LOWER */
		store_a_lower((entity) var, 
			      (int) POSITIVE_DIVIDE(cst-coe-1, -coe));
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
 * used thru a hook provided in Psysteme_to_code. This allows to provide
 * the code generation with some information that allow to improve the
 * generated code.
 * ??? Whether this should be here or not is another question. 
 */
static boolean
range_of_variable(
    Variable var, 
    Value * lb,
    Value * ub)
{
    if (lowers_undefined_p() || uppers_undefined_p()) 
	return FALSE; /* no information available, that's for sure */

    if (!bound_lowers_p(var) || !bound_uppers_p(var))
	return FALSE;

    *lb = (Value) load_lowers((entity) var);
    *ub = (Value) load_uppers((entity) var);

    return TRUE;
}

static boolean
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
	    min+=coef, max+=coef;
	else
	{
	    if (!range_of_variable(var, &lb, &ub))
		return FALSE;

	    if (coef>0)
		min+=coef*lb, max+=coef*ub;
	    else
		min+=coef*ub, max+=coef*lb;
	}
    }

    *result = DIVIDE(min, denominator);

    return *result==DIVIDE(max, denominator);
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
      inegalites_fprint(stderr, c, entity_local_name);
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

      message_assert("coherent value and sign", sign*val>0);

      if (val>0) 
	  vect_chg_sgn(vdiv);
      else
	  /*  ax+b <= 0 and a<0 => x >= (b+(-a-1))/(-a)
	   */
	  val=-val, vect_add_elem(&vdiv, TCST, val-1);

      if (val==1)
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

	  if (vl%val==0) vect_add_elem(&vadd, va, vl/val);
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
	  ediv = MakeBinaryCall(divide, ediv, int_to_expression(val));	  
	  
	  if (vadd)
	  {
	      eadd = make_vecteur_expression(vadd);
	      e = MakeBinaryCall(CreateIntrinsic(PLUS_OPERATOR_NAME),
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
  len = gen_length(le);  message_assert("some expressions", len!=0);

  if (len==1)
  {
      result = EXPRESSION(CAR(le));
      gen_free_list(le);
  }
  else
  {
      entity operator = is_lower ?
	  CreateIntrinsic("MAX"): CreateIntrinsic("MIN");

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

    if (nb_elems_list(lower)!=1 || nb_elems_list(upper)!=1) return(FALSE);

    val_upper = vect_coeff(var, upper->vecteur);
    val_lower = vect_coeff(var, lower->vecteur);
    
    /* ??? the arithmetic ppcm version is on int instead of values 
     */
    the_ppcm = ppcm(-val_lower, val_upper);

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

    s = sc_dup(sc);
    sc_transform_eg_in_ineg(s);  /* ??? could do a better job with = */
    c = sc_inegalites(s);

    reverse = gen_nreverse(gen_copy_seq(vars));

    message_assert("no equalities, now", sc_nbre_egalites(s)==0);
    
    MAP(ENTITY, e,
    {
	Variable var = (Variable) e;
	
	pips_debug(5, "variable %s loop\n", entity_name((entity) var));
	
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
