 /* semantic analysis: processing of expressions
  *
  * $Id$
  *
  * $Log: expression.c,v $
  * Revision 1.1  2001/06/19 09:21:57  irigoin
  * Initial revision
  *
  *
  */
#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
/* #include "database.h" */
#include "linear.h"
#include "ri.h"
/*
#include "text.h"
#include "text-util.h"
*/
#include "ri-util.h"
/* #include "constants.h" */
/* #include "control.h" */
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

#include "vecteur.h"
#include "contrainte.h"
/*
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
*/

#include "transformer.h"

#include "semantics.h"

static transformer 
affine_to_transformer(entity e, Pvecteur a, bool assignment)
{
    transformer tf = transformer_undefined;
    Pvecteur ve = vect_new((Variable) e, VALUE_ONE);
    entity e_new = entity_to_new_value(e);
    entity e_old = entity_to_old_value(e);
    cons * tf_args = CONS(ENTITY, e_new, NIL);
    /* must be duplicated right now  because it will be
       renamed and checked at the same time by
       value_mappings_compatible_vector_p() */
    Pvecteur vexpr = vect_dup(a);
    Pcontrainte c;
    Pvecteur eq = VECTEUR_NUL;

    debug(8, "affine_to_transformer", "begin\n");

    ifdebug(9) {
	pips_debug(9, "\nLinearized expression:\n");
	vect_dump(vexpr);
    }

    if(!assignment) {
	vect_add_elem(&vexpr, (Variable) e, (Value) 1);

	ifdebug(8) {
	    pips_debug(8, "\nLinearized expression for incrementation:\n");
	    vect_dump(vexpr);
	}
    }

    if(value_mappings_compatible_vector_p(ve) &&
       value_mappings_compatible_vector_p(vexpr)) {
	ve = vect_variable_rename(ve,
				  (Variable) e,
				  (Variable) e_new);
	(void) vect_variable_rename(vexpr,
				    (Variable) e_new,
				    (Variable) e_old);
	eq = vect_substract(ve, vexpr);
	vect_rm(ve);
	vect_rm(vexpr);
	c = contrainte_make(eq);
	tf = make_transformer(tf_args,
		      make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));
    }
    else {
	vect_rm(eq);
	vect_rm(ve);
	vect_rm(vexpr);
	tf = transformer_undefined;
    }

    debug(8, "affine_to_transformer", "end\n");

    return tf;
}

transformer 
affine_increment_to_transformer(entity e, Pvecteur a)
{
    transformer tf = transformer_undefined;

    tf = affine_to_transformer(e, a, FALSE);

    return tf;
}

static transformer modulo_to_transformer(entity e,
					 expression expr,
					 transformer pre)
{
    transformer tf = transformer_undefined;
    expression arg2 = expression_undefined;
    call c = syntax_call(expression_syntax(expr));

    debug(8, "modulo_to_transformer", "begin\n");

    pips_assert("Precondition is unused", transformer_undefined_p(pre));
    
    arg2 = find_ith_argument(call_arguments(c), 2);

    if(integer_constant_expression_p(arg2)) {
	int d = integer_constant_expression_value(arg2);
	entity e_new = entity_to_new_value(e);
	Pvecteur ub = vect_new((Variable) e_new, VALUE_ONE);
	Pvecteur lb = vect_new((Variable) e_new, VALUE_MONE);
	Pcontrainte clb = contrainte_make(lb);
	Pcontrainte cub = CONTRAINTE_UNDEFINED;
	cons * tf_args = CONS(ENTITY, e_new, NIL);

	vect_add_elem(&ub, TCST, int_to_value(1-d));
	vect_add_elem(&lb, TCST, int_to_value(d-1));
	cub = contrainte_make(ub);
	clb->succ = cub;
	tf = make_transformer(tf_args,
		make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }

    ifdebug(8) {
	debug(8, "modulo_to_transformer", "result:\n");
	print_transformer(tf);
	debug(8, "modulo_to_transformer", "end\n");
    }

   return tf;
}

static transformer iabs_to_transformer(entity e, 
				       expression expr,
				       transformer pre)
{
    transformer tf = transformer_undefined;
    call c = syntax_call(expression_syntax(expr));
    expression arg = EXPRESSION(CAR(call_arguments(c)));
    normalized n = NORMALIZE_EXPRESSION(arg);

    debug(8, "iabs_to_transformer", "begin\n");

    pips_assert("Precondition is unused", transformer_undefined_p(pre));

    if(normalized_linear_p(n)) {
	entity e_new = entity_to_new_value(e);
	entity e_old = entity_to_old_value(e);
	Pvecteur vlb1 = vect_dup((Pvecteur) normalized_linear(n));
	Pvecteur vlb2 = vect_multiply(vect_dup((Pvecteur) normalized_linear(n)), VALUE_MONE);
	Pcontrainte clb1 = CONTRAINTE_UNDEFINED;
	Pcontrainte clb2 = CONTRAINTE_UNDEFINED;
	cons * tf_args = CONS(ENTITY, e_new, NIL);

	(void) vect_variable_rename(vlb1,
				    (Variable) e_new,
				    (Variable) e_old);

	(void) vect_variable_rename(vlb2,
				    (Variable) e_new,
				    (Variable) e_old);

	vect_add_elem(&vlb1, (Variable) e_new, VALUE_MONE);
	vect_add_elem(&vlb2, (Variable) e_new, VALUE_MONE);
	clb1 = contrainte_make(vlb1);
	clb2 = contrainte_make(vlb2);
	clb1->succ = clb2;
	tf = make_transformer(tf_args,
		make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb1)));
    }

    ifdebug(8) {
	debug(8, "iabs_to_transformer", "result:\n");
	print_transformer(tf);
	debug(8, "iabs_to_transformer", "end\n");
    }

   return tf;
}

static transformer 
integer_divide_to_transformer(entity e, expression expr, transformer pre)
{
    transformer tf = transformer_undefined;
    call c = syntax_call(expression_syntax(expr));
    expression arg1 = expression_undefined;
    normalized n1 = normalized_undefined;
    expression arg2 = expression_undefined;

    debug(8, "integer_divide_to_transformer", "begin\n");
 
    pips_assert("Precondition is unused", transformer_undefined_p(pre));
   
    arg1 = find_ith_argument(call_arguments(c), 1);
    n1 = NORMALIZE_EXPRESSION(arg1);
    arg2 = find_ith_argument(call_arguments(c), 2);

    if(integer_constant_expression_p(arg2) && normalized_linear_p(n1)) {
	int d = integer_constant_expression_value(arg2);
	entity e_new = entity_to_new_value(e);
	entity e_old = entity_to_old_value(e);
	cons * tf_args = CONS(ENTITY, e, NIL);
	/* must be duplicated right now  because it will be
	   renamed and checked at the same time by
	   value_mappings_compatible_vector_p() */
	Pvecteur vlb =
	    vect_multiply(vect_dup(normalized_linear(n1)), VALUE_MONE); 
	Pvecteur vub = vect_dup(normalized_linear(n1));
	Pcontrainte clb = CONTRAINTE_UNDEFINED;
	Pcontrainte cub = CONTRAINTE_UNDEFINED;

	(void) vect_variable_rename(vlb,
				    (Variable) e_new,
				    (Variable) e_old);
	(void) vect_variable_rename(vub,
				    (Variable) e_new,
				    (Variable) e_old);

	vect_add_elem(&vlb, (Variable) e_new, int_to_value(d));
	vect_add_elem(&vub, (Variable) e_new, int_to_value(-d));
	vect_add_elem(&vub, TCST, int_to_value(1-d));
	clb = contrainte_make(vlb);
	cub = contrainte_make(vub);
	clb->succ = cub;
	tf = make_transformer(tf_args,
	       make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }

    ifdebug(8) {
	debug(8, "integer_divide_to_transformer", "result:\n");
	print_transformer(tf);
	debug(8, "integer_divide_to_transformer", "end\n");
    }

    return tf;
}

static transformer 
integer_power_to_transformer(entity e, expression expr, transformer pre)
{
  transformer tf = transformer_undefined;
  call c = syntax_call(expression_syntax(expr));
  expression arg1 = expression_undefined;
  normalized n1 = normalized_undefined;
  expression arg2 = expression_undefined;
  normalized n2 = normalized_undefined;

  debug(8, "integer_power_to_transformer", "begin\n");

    pips_assert("Precondition is unused", transformer_undefined_p(pre));
    
  arg1 = find_ith_argument(call_arguments(c), 1);
  n1 = NORMALIZE_EXPRESSION(arg1);
  arg2 = find_ith_argument(call_arguments(c), 2);
  n2 = NORMALIZE_EXPRESSION(arg2);

  if(signed_integer_constant_expression_p(arg2) && normalized_linear_p(n1)) {
    int d = signed_integer_constant_expression_value(arg2);

    if(d%2==0) {
      entity e_new = entity_to_new_value(e);
      entity e_old = entity_to_old_value(e);
      cons * tf_args = CONS(ENTITY, e, NIL);

      if(d==0) {
	/* 1 is assigned unless arg1 equals 0... which is neglected */
	Pvecteur v = vect_new((Variable) e_new, VALUE_ONE);

	vect_add_elem(&v, TCST, VALUE_MONE);
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(contrainte_make(v),
						     CONTRAINTE_UNDEFINED)));
      }
      else if(d>0) {
	/* Does not work because unary minus is not seen as part of a constant */
	/* The expression value must be greater or equal to arg2 and positive */
	/* must be duplicated right now  because it will be
	   renamed and checked at the same time by
	   value_mappings_compatible_vector_p() */
	Pvecteur vlb1 = vect_dup(normalized_linear(n1));
	Pvecteur vlb2 = vect_multiply(vect_dup(normalized_linear(n1)), VALUE_MONE);
	Pcontrainte clb1 = CONTRAINTE_UNDEFINED;
	Pcontrainte clb2 = CONTRAINTE_UNDEFINED;

	(void) vect_variable_rename(vlb1,
				    (Variable) e_new,
				    (Variable) e_old);

	vect_add_elem(&vlb1, (Variable) e_new, VALUE_MONE);
	vect_add_elem(&vlb2, (Variable) e_new, VALUE_MONE);
	clb1 = contrainte_make(vlb1);
	clb2 = contrainte_make(vlb2);
	clb1->succ = clb2;
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb1)));
      }
      else {
	/* d is negative and even */
	entity e_new = entity_to_new_value(e);
	cons * tf_args = CONS(ENTITY, e, NIL);
	Pvecteur vub = vect_new((Variable) e_new, VALUE_ONE);
	Pvecteur vlb = vect_new((Variable) e_new, VALUE_MONE);
	Pcontrainte clb = CONTRAINTE_UNDEFINED;
	Pcontrainte cub = CONTRAINTE_UNDEFINED;

	vect_add_elem(&vub, TCST, VALUE_MONE);
	clb = contrainte_make(vlb);
	cub = contrainte_make(vub);
	clb->succ = cub;
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
      }
    }
    else if(d<0) {
      /* d is negative, arg1 cannot be 0, expression value is -1, 0
	 or 1 */
      entity e_new = entity_to_new_value(e);
      cons * tf_args = CONS(ENTITY, e, NIL);
      Pvecteur vub = vect_new((Variable) e_new, VALUE_MONE);
      Pvecteur vlb = vect_new((Variable) e_new, VALUE_ONE);
      Pcontrainte clb = CONTRAINTE_UNDEFINED;
      Pcontrainte cub = CONTRAINTE_UNDEFINED;

      vect_add_elem(&vub, TCST, VALUE_MONE);
      vect_add_elem(&vlb, TCST, VALUE_MONE);
      clb = contrainte_make(vlb);
      cub = contrainte_make(vub);
      clb->succ = cub;
      tf = make_transformer(tf_args,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }
    else if(d==1) {
	entity e_new = entity_to_new_value(e);
	cons * tf_args = CONS(ENTITY, e, NIL);
	Pvecteur v = vect_dup(normalized_linear(n1));

	vect_add_elem(&v, (Variable) e_new, VALUE_MONE);
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(contrainte_make(v),
						     CONTRAINTE_UNDEFINED)));
    }
  }
  else if(signed_integer_constant_expression_p(arg1)) {
    int d = signed_integer_constant_expression_value(arg1);
    entity e_new = entity_to_new_value(e);

    if(d==0||d==1) {
      /* 0 or 1 is assigned unless arg2 equals 0... which is neglected */
      cons * tf_args = CONS(ENTITY, e, NIL);
      Pvecteur v = vect_new((Variable) e_new, VALUE_ONE);

      vect_add_elem(&v, TCST, int_to_value(-d));
      tf = make_transformer(tf_args,
			    make_predicate(sc_make(contrainte_make(v),
						   CONTRAINTE_UNDEFINED)));
    }
    else if(d > 1) {
      /* the assigned value is positive */
      cons * tf_args = CONS(ENTITY, e, NIL);
      Pvecteur v1 = vect_new((Variable) e_new, VALUE_MONE);
      Pcontrainte c1 = contrainte_make(v1);

      if(normalized_linear_p(n2)) {
	Pvecteur v2 = vect_dup(normalized_linear(n2));
	Pcontrainte c2 = CONTRAINTE_UNDEFINED;
	Pvecteur v3 = vect_multiply(vect_dup(normalized_linear(n2)), (Value) d);
	Pcontrainte c3 = CONTRAINTE_UNDEFINED;

	vect_add_elem(&v2, TCST, VALUE_ONE);
	vect_add_elem(&v2, (Variable) e_new, VALUE_MONE);
	c2 = contrainte_make(v2);
	contrainte_succ(c1) = c2;
	vect_add_elem(&v3, (Variable) e_new, VALUE_MONE);
	c3 = contrainte_make(v3);
	contrainte_succ(c2) = c3;
      }

      tf = make_transformer(tf_args,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, c1)));
    }
    else if(d == -1) {
      /* The assigned value is 1 or -1 */
      entity e_new = entity_to_new_value(e);
      cons * tf_args = CONS(ENTITY, e, NIL);
      Pvecteur vub = vect_new((Variable) e_new, VALUE_MONE);
      Pvecteur vlb = vect_new((Variable) e_new, VALUE_ONE);
      Pcontrainte clb = CONTRAINTE_UNDEFINED;
      Pcontrainte cub = CONTRAINTE_UNDEFINED;

      vect_add_elem(&vub, TCST, VALUE_MONE);
      vect_add_elem(&vlb, TCST, VALUE_MONE);
      clb = contrainte_make(vlb);
      cub = contrainte_make(vub);
      clb->succ = cub;
      tf = make_transformer(tf_args,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }
  }

  ifdebug(8) {
    debug(8, "integer_power_to_transformer", "result:\n");
    print_transformer(tf);
    debug(8, "integer_power_to_transformer", "end\n");
  }

  return tf;
}

static transformer 
minmax_to_transformer(entity e, expression expr, transformer pre, bool minmax)
{
    transformer tf = transformer_undefined;
    call c = syntax_call(expression_syntax(expr));
    expression arg = expression_undefined;
    normalized n = normalized_undefined;
    list cexpr;
    cons * tf_args = CONS(ENTITY, e, NIL);
    Pcontrainte cl = CONTRAINTE_UNDEFINED;

    debug(8, "minmax_to_transformer", "begin\n");

    pips_assert("Precondition is unused", transformer_undefined_p(pre));

    for(cexpr = call_arguments(c); !ENDP(cexpr); POP(cexpr)) {
	arg = EXPRESSION(CAR(cexpr));
	n = NORMALIZE_EXPRESSION(arg);

	if(normalized_linear_p(n)) {
	    Pvecteur v = vect_dup((Pvecteur) normalized_linear(n));
	    Pcontrainte cv = CONTRAINTE_UNDEFINED;
	    entity e_new = entity_to_new_value(e);
	    entity e_old = entity_to_old_value(e);

	    (void) vect_variable_rename(v,
					(Variable) e,
					(Variable) e_old);
	    vect_add_elem(&v, (Variable) e_new, VALUE_MONE);

	    if(minmax) {
		v = vect_multiply(v, VALUE_MONE);
	    }

	    cv = contrainte_make(v);
	    cv->succ = cl;
	    cl = cv;

	}
    }

    if(CONTRAINTE_UNDEFINED_P(cl) || CONTRAINTE_NULLE_P(cl)) {
	Psysteme sc = sc_make(CONTRAINTE_UNDEFINED, cl);
	entity oldv = entity_to_old_value(e);
	entity newv = entity_to_new_value(e);

	sc_base(sc) = base_add_variable(base_add_variable(BASE_NULLE,
							  (Variable) oldv),
					(Variable) newv);
	sc_dimension(sc) = 2;
	tf = make_transformer(tf_args,
			      make_predicate(sc));
    }
    else {
	/* A miracle occurs and the proper basis is derived from the
	   constraints ( I do not understand why the new and the old value
	   of e both appear... so it may not be necessary for the
	   consistency check... I'm lost, FI, 6 Jan. 1999) */
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, cl)));
    }


    ifdebug(8) {
	debug(8, "minmax_to_transformer", "result:\n");
	print_transformer(tf);
	debug(8, "minmax_to_transformer", "end\n");
    }

    return tf;
}

static transformer 
min0_to_transformer(entity e, expression expr, transformer pre)
{
    return minmax_to_transformer(e, expr, pre, TRUE);
}

static transformer 
max0_to_transformer(entity e, expression expr, transformer pre)
{
    return minmax_to_transformer(e, expr, pre, FALSE);
}

/* */
transformer 
integer_expression_to_transformer(
    entity e,
    expression expr,
    transformer pre)
{
  transformer tf = transformer_undefined;
  normalized n = NORMALIZE_EXPRESSION(expr);

  pips_debug(8, "begin\n");

  /* Assume: e is a value */

  if(normalized_linear_p(n)) {
    tf = affine_assignment_to_transformer(e,
					  (Pvecteur) normalized_linear(n));
  }
  else if(modulo_expression_p(expr)) {
    tf = modulo_to_transformer(e, expr, pre);
  }
  else if(divide_expression_p(expr)) {
    tf = integer_divide_to_transformer(e, expr, pre);
  }
  else if(power_expression_p(expr)) {
    tf = integer_power_to_transformer(e, expr, pre);
  }
  else if(iabs_expression_p(expr)) {
    tf = iabs_to_transformer(e, expr, pre);
  }
  else if(min0_expression_p(expr)) {
    tf = min0_to_transformer(e, expr, pre);
  }
  else if(max0_expression_p(expr)) {
    tf = max0_to_transformer(e, expr, pre);
  }
  else if(user_function_call_p(expr) 
	  && get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
    tf = user_function_call_to_transformer(e, expr, ef);
  }
  else {
    /* vect_rm(ve); */
    tf = transformer_undefined;
  }
}

    pips_debug(8, "end with tf=%p\n", tf);

    return tf;
}

transformer 
expression_to_transformer(
    entity e,
    expression expr,
    transformer pre)
{
    transformer tf = transformer_undefined;

    pips_debug(8, "begin\n");

    if(entity_has_values_p(e)) {
        /* Pvecteur ve = vect_new((Variable) e, VALUE_ONE); */
	normalized n = NORMALIZE_EXPRESSION(expr);

	if(normalized_linear_p(n)) {
	    tf = affine_assignment_to_transformer(e,
		       (Pvecteur) normalized_linear(n));
	}
	else if(modulo_expression_p(expr)) {
	    tf = modulo_to_transformer(e, expr);
	}
	else if(divide_expression_p(expr)) {
	    tf = integer_divide_to_transformer(e, expr);
	}
	else if(power_expression_p(expr)) {
	    tf = integer_power_to_transformer(e, expr);
	}
	else if(iabs_expression_p(expr)) {
	    tf = iabs_to_transformer(e, expr);
	}
	else if(min0_expression_p(expr)) {
	    tf = min0_to_transformer(e, expr);
	}
	else if(max0_expression_p(expr)) {
	    tf = max0_to_transformer(e, expr);
	}
	else if(user_function_call_p(expr) 
		&& get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
	    tf = user_function_call_to_transformer(e, expr, ef);
	}
	else {
	    /* vect_rm(ve); */
	    tf = transformer_undefined;
	}
    }

    pips_debug(8, "end with tf=%p\n", tf);

    return tf;
}

/* transformer assigned_expression_to_transformer(entity e, expression expr, list ef):
 * returns a transformer abstracting the effect of assignment e = expr
 * if entity e and entities referenced in expr are accepted for
 * semantics analysis anf if expr is affine; else returns
 * transformer_undefined
 *
 * Note: it might be better to distinguish further between e and expr
 * and to return a transformer stating that e is modified when e
 * is accepted for semantics analysis.
 */
transformer 
assigned_expression_to_transformer(
    entity e,
    expression expr,
    list ef)
{
    transformer tf = transformer_undefined;

    pips_debug(8, "begin\n");

    if(entity_has_values_p(e)) {
        /* Pvecteur ve = vect_new((Variable) e, VALUE_ONE); */
	normalized n = NORMALIZE_EXPRESSION(expr);

	if(normalized_linear_p(n)) {
	    tf = affine_assignment_to_transformer(e,
		       (Pvecteur) normalized_linear(n));
	}
	else if(modulo_expression_p(expr)) {
	    tf = modulo_to_transformer(e, expr);
	}
	else if(divide_expression_p(expr)) {
	    tf = integer_divide_to_transformer(e, expr);
	}
	else if(power_expression_p(expr)) {
	    tf = integer_power_to_transformer(e, expr);
	}
	else if(iabs_expression_p(expr)) {
	    tf = iabs_to_transformer(e, expr);
	}
	else if(min0_expression_p(expr)) {
	    tf = min0_to_transformer(e, expr);
	}
	else if(max0_expression_p(expr)) {
	    tf = max0_to_transformer(e, expr);
	}
	else if(user_function_call_p(expr) 
		&& get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
	    tf = user_function_call_to_transformer(e, expr, ef);
	}
	else {
	    /* vect_rm(ve); */
	    tf = transformer_undefined;
	}
    }

    pips_debug(8, "end with tf=%p\n", tf);

    return tf;
}

/* PROCESSING OF LOGICAL EXPRESSIONS */

transformer transformer_logical_inequalities_add(transformer tf, entity v)
{
  /* the values of v are between 0 and 1 */
  Pvecteur ineq1 = vect_new((Variable) v, VALUE_ONE);
  Pvecteur ineq2 = vect_new((Variable) v, VALUE_MONE);

  vect_add_elem(&ineq1, TCST, VALUE_MONE);

  tf = transformer_inequality_add(tf, ineq1);
  tf = transformer_inequality_add(tf, ineq2);

  return tf;
}

static transformer logical_constant_to_transformer(entity v,
						   entity f)
{
  transformer tf = transformer_undefined;
  Pvecteur eq = vect_new((Variable) v, VALUE_ONE);
  Pcontrainte c;

  if(ENTITY_TRUE_P(f)) {
    vect_add_elem(&eq, TCST , VALUE_MONE);
  }
  else if(ENTITY_FALSE_P(f)) {
    ;
  }
  else {
    pips_error("logical_constant_to_transformer",
	       "Unknown logical constant %s\n",
	       entity_name(f));
  }
  c = contrainte_make(eq);
  tf = make_transformer(NIL,
			make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));

  return tf;
}

static transformer logical_unary_operation_to_transformer(entity v,
							  call c)
{
  transformer tf = transformer_undefined;
  entity op = call_function(c);
  expression arg = EXPRESSION(CAR(call_arguments(c)));
  Pvecteur eq = vect_new((Variable) v, VALUE_ONE);

  /* pips_assert("A unique argument", ENDP(CDR(call_arguments(c)))); */

  if(ENTITY_NOT_P(op)) {
    entity tmp = make_local_temporary_value_entity(entity_type(v));
    tf = logical_expression_to_transformer(tmp, arg);
    vect_add_elem(&eq, (Variable) tmp , VALUE_ONE);
    vect_add_elem(&eq, TCST , VALUE_MONE);
  }
  else {
    pips_error("logical_constant_to_transformer",
	       "Unknown logical constant %s\n",
	       entity_name(op));
  }
  tf = transformer_equality_add(tf, eq);
  tf = transformer_logical_inequalities_add(tf, v);

  return tf;
}

static transformer logical_binary_operation_to_transformer(entity v,
							   call c)
{
  entity op = call_function(c);
  Pvecteur eq1 = VECTEUR_NUL;
  Pvecteur eq2 = VECTEUR_NUL;
  Pvecteur eq3 = VECTEUR_NUL;
  expression arg1 = EXPRESSION(CAR(call_arguments(c)));
  expression arg2 = EXPRESSION(CAR(CDR(call_arguments(c))));
  entity tmp1 = make_local_temporary_value_entity(entity_type(v));
  entity tmp2 = make_local_temporary_value_entity(entity_type(v));
  transformer tf1 = logical_expression_to_transformer(tmp1, arg1);
  transformer tf2 = logical_expression_to_transformer(tmp2, arg2);
  transformer tf = transformer_intersection(tf1, tf2);

  ifdebug(9) {
    pips_debug(9, "Begin for value %s with subtransformers:\n",
	       entity_name(v));
    dump_transformer(tf1);
    dump_transformer(tf2);
  }

  free_transformer(tf1);
  free_transformer(tf2);

  ifdebug(9) {
    pips_debug(9, "Union of subtransformers:\n");
    dump_transformer(tf);
  }

  if(ENTITY_AND_P(op)) {
    /* v <= tmp1, v <= tmp2, v >= tmp1+tmp2-1 */
    eq1 = vect_new((Variable) v, VALUE_ONE);
    vect_add_elem(&eq1, (Variable) tmp1, VALUE_MONE);
    eq2 = vect_new((Variable) v, VALUE_ONE);
    vect_add_elem(&eq2, (Variable) tmp2, VALUE_MONE);
    eq3 = vect_new((Variable) v, VALUE_MONE);
    vect_add_elem(&eq3, (Variable) tmp1, VALUE_ONE);
    vect_add_elem(&eq3, (Variable) tmp2, VALUE_ONE);
    vect_add_elem(&eq3, TCST, VALUE_MONE);
  }
  else if(ENTITY_OR_P(op)) {
    /* v >= tmp1, v>= tmp2, v <= tmp1+tmp2 */
    eq1 = vect_new((Variable) v, VALUE_MONE);
    vect_add_elem(&eq1, (Variable) tmp1, VALUE_ONE);
    eq2 = vect_new((Variable) v, VALUE_MONE);
    vect_add_elem(&eq2, (Variable) tmp2, VALUE_ONE);
    eq3 = vect_new((Variable) v, VALUE_ONE);
    vect_add_elem(&eq3, (Variable) tmp1, VALUE_MONE);
    vect_add_elem(&eq3, (Variable) tmp2, VALUE_MONE);
  }
  else if(ENTITY_EQUAL_P(op)) {
    /* v >= 1-tmp1-tmp2, v >= tmp1+tmp2-1 */
    eq1 = vect_new((Variable) v, VALUE_MONE);
    vect_add_elem(&eq1, (Variable) tmp1, VALUE_MONE);
    vect_add_elem(&eq1, (Variable) tmp2, VALUE_MONE);
    vect_add_elem(&eq1, TCST , VALUE_ONE);
    eq2 = vect_new((Variable) v, VALUE_MONE);
    vect_add_elem(&eq2, (Variable) tmp1, VALUE_ONE);
    vect_add_elem(&eq2, (Variable) tmp2, VALUE_ONE);
    vect_add_elem(&eq2, TCST , VALUE_MONE);
  }
  else if(ENTITY_NON_EQUAL_P(op)) {
    /* v <= tmp1+tmp2, v <= 2-tmp1-tmp2 */
    eq1 = vect_new((Variable) v, VALUE_ONE);
    vect_add_elem(&eq1, (Variable) tmp1, VALUE_MONE);
    vect_add_elem(&eq1, (Variable) tmp2, VALUE_MONE);
    eq2 = vect_new((Variable) v, VALUE_ONE);
    vect_add_elem(&eq2, (Variable) tmp1, VALUE_ONE);
    vect_add_elem(&eq2, (Variable) tmp2, VALUE_ONE);
    vect_add_elem(&eq2, TCST , VALUE_MONE+VALUE_MONE);
  }
  else {
    pips_error("logical_binary_operation_to_transformer",
	       "Unexpected binary logical operator %s\n",
	       entity_name(op));
  }
  tf = transformer_inequality_add(tf, eq1);
  tf = transformer_inequality_add(tf, eq2);
  if(!VECTEUR_NUL_P(eq3)) {
    tf = transformer_inequality_add(tf, eq3);
  }
  tf = transformer_logical_inequalities_add(tf, v);

  ifdebug(9) {
    pips_debug(9, "End with transformer:\n");
    dump_transformer(tf);
  }

  return tf;
}

static transformer logical_binary_function_to_transformer(entity v,
							  call c)
{
  transformer tf = transformer_undefined;
  entity op = call_function(c);

  if(ENTITY_AND_P(op)||ENTITY_OR_P(op)) {
    tf = logical_binary_operation_to_transformer(v, c);
  }
  else {
    expression expr1 = EXPRESSION(CAR(call_arguments(c)));
    expression expr2 = EXPRESSION(CAR(CDR(call_arguments(c))));
    basic b1 = basic_of_expression(expr1);
    basic b2 = basic_of_expression(expr2);

    if(basic_logical_p(b1) && basic_logical_p(b2)) {
      tf = logical_binary_operation_to_transformer(v, c);
    }
    else {
      /* Non internal functions such as EQ, NEQ, GE, GT, LE, LT,... */
      /* For the time being, no information is derived, even if EQ and NEQ
	 were applied to logical subexpressions (or is it EQV and NEQV?) */
      tf = transformer_identity();
    }

    free_basic(b1);
    free_basic(b2);
  }

  return tf;
}

/* v is assumed to be a temporary value and r a logical program variable */
static transformer logical_reference_to_transformer(entity v,
						    entity r)
{
  transformer tf = transformer_undefined;
  entity r_new = entity_to_new_value(r);


  tf = simple_equality_to_transformer(v, r_new, FALSE);
  tf = transformer_logical_inequalities_add(tf, r_new);

  return tf;
}

/* Could be used to compute preconditions too. v is assumed to be a new
   value or a temporary value. */
transformer logical_expression_to_transformer(entity v,
					      expression rhs)
{
  transformer tf = transformer_undefined;
  syntax srhs = expression_syntax(rhs);

    switch(syntax_tag(srhs)) {
    case is_syntax_call:
      switch(gen_length(call_arguments(syntax_call(srhs)))) {
      case 0:
	tf = logical_constant_to_transformer(v, call_function(syntax_call(srhs)));
	break;
      case 1:
	tf = logical_unary_operation_to_transformer(v, syntax_call(srhs));
	break;
      case 2:
	tf = logical_binary_function_to_transformer(v, syntax_call(srhs));
	break;
      default:
	pips_error("logical_expression_to_transformer",
		   "Too many logical arguments, %d, for operator %s\n",
		   gen_length(call_arguments(syntax_call(srhs))),
		   entity_name(call_function(syntax_call(srhs))));
      }
      break;
    case is_syntax_reference:
	tf = logical_reference_to_transformer(v, reference_variable(syntax_reference(srhs)));
      break;
    case is_syntax_range:
      pips_error("logical_expression_to_transformer", "Unexpected tag %d\n", syntax_tag(srhs));
      break;
    default:
      pips_error("logical_expression_to_transformer", "Illegal tag %d\n", syntax_tag(srhs));
    }
    return tf;
}
