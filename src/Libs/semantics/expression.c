 /* semantic analysis: processing of expressions, with or without
  * preconditions
  *
  * $Id$
  *
  * $Log: expression.c,v $
  * Revision 1.3  2001/07/19 17:54:31  irigoin
  * Lots of additional reformatting with an indentation factor of two instead
  * of four
  *
  * Revision 1.2  2001/07/13 15:01:06  irigoin
  * First multitype version
  *
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

/* TYPE INDEPENDENT OPERATIONS */

static transformer generic_reference_to_transformer(entity v, entity r, bool is_internal)
{
  transformer tf = transformer_undefined;

  if((!is_internal && analyzable_scalar_entity_p(r)) || entity_has_values_p(r)) {
    entity r_new = is_internal? entity_to_new_value(r) : external_entity_to_new_value(r);
    tf =  simple_equality_to_transformer(v, r_new, FALSE);
  }
  return tf;
}

static transformer 
generic_minmax_to_transformer(entity e,
			      list args,
			      transformer pre,
			      bool minmax,
			      bool is_internal)
{
  transformer tf = transformer_undefined;
  transformer tf_acc = transformer_undefined;
  list cexpr;
  Pcontrainte cl = CONTRAINTE_UNDEFINED;

  pips_debug(8, "begin\n");

  pips_assert("Precondition is unused", transformer_undefined_p(pre));

  for(cexpr = args; !ENDP(cexpr); POP(cexpr)) {
    expression arg = EXPRESSION(CAR(cexpr));
    type t = entity_type(e);
    entity tmp = make_local_temporary_value_entity(t);
    transformer tfe = any_expression_to_transformer(tmp, arg, pre, is_internal);
	
    Pvecteur v = vect_new((Variable) tmp, VALUE_ONE);
    Pcontrainte cv = CONTRAINTE_UNDEFINED;

    vect_add_elem(&v, (Variable) e, VALUE_MONE);

    if(minmax) {
      v = vect_multiply(v, VALUE_MONE);
    }

    cv = contrainte_make(v);
    cv->succ = cl;
    cl = cv;

    /* memory leak! */
    tf_acc = transformer_safe_image_intersection(tf_acc, tfe);
  }


  if(CONTRAINTE_UNDEFINED_P(cl) || CONTRAINTE_NULLE_P(cl)) {
    Psysteme sc = sc_make(CONTRAINTE_UNDEFINED, cl);

    sc_base(sc) = base_add_variable(BASE_NULLE, (Variable) e);
    sc_dimension(sc) = 1;
    tf = make_transformer(NIL,
			  make_predicate(sc));
  }
  else {
    /* A miracle occurs and the proper basis is derived from the
       constraints ( I do not understand why the new and the old value
       of e both appear... so it may not be necessary for the
       consistency check... I'm lost, FI, 6 Jan. 1999) */
    tf_acc = transformer_inequalities_add(tf_acc, cl);
    tf = tf_acc;
  }

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

/* Type independent. Most of it should be factored out for unary operations. */
static transformer unary_minus_operation_to_transformer(entity v,
							expression e1,
							transformer pre,
							bool is_internal)
{
  transformer tf = transformer_undefined;
  entity tmp1 = make_local_temporary_value_entity(entity_type(v));
  /* sub-expressions are not necessarily integer? Then the expression
     should not be integer? */
  transformer tf1 = any_expression_to_transformer(tmp1, e1, pre, is_internal);
  transformer tf_op = simple_unary_minus_to_transformer(v, tmp1);

  pips_debug(9, "Begin for %s\n", entity_local_name(v));

  tf = transformer_safe_intersection(tf1, tf_op);

  /* Much too early: you are in between recursive calls! */
  /* tf = transformer_temporary_value_projection(tf); */
    
  free_transformer(tf1);
  free_transformer(tf_op);

  pips_debug(9, "End with %p\n", tf);
  ifdebug(9) dump_transformer(tf);

  return tf;
}

static transformer 
generic_unary_operation_to_transformer(
    entity e,
    entity op,
    expression e1,
    transformer pre,
    bool is_internal)
{
  transformer tf = transformer_undefined;

  if(ENTITY_UNARY_MINUS_P(op)) {
    tf = unary_minus_operation_to_transformer(e, e1, pre, is_internal);
  }

  return tf;
}

/* Type independent. Most of it should be factored out for binary
   operations. v= e1 + e2 or v = e1 - e2 */
static transformer addition_operation_to_transformer(entity v,
						     expression e1,
						     expression e2,
						     transformer pre,
						     bool addition_p,
						     bool is_internal)
{
  transformer tf = transformer_undefined;
  entity tmp1 = make_local_temporary_value_entity(entity_type(v));
  entity tmp2 = make_local_temporary_value_entity(entity_type(v));
  /* sub-expressions are not necessarily integer? Then the expression
     should not be integer? */
  transformer tf1 = any_expression_to_transformer(tmp1, e1, pre, is_internal);
  transformer tf2 = any_expression_to_transformer(tmp2, e2, pre, is_internal);
  transformer tf12 = transformer_undefined;
  transformer tf_op = simple_addition_to_transformer(v, tmp1, tmp2, addition_p);

  pips_debug(9, "Begin officialy... but a lot has been done already!\n");

  /* Intersection is not powerful enough to cope with side effects. But
     side effects can be dealt with only if the operation order is known
     when the standard is violated. We assume here a right to left evaluation. */
  if(transformer_safe_affect_transformer_p(tf1, tf2)) {
    pips_debug(9, "Side effects of tf2 on tf1\n");
    pips_user_warning("Non standard compliant code: side effect in part\n"
		      "of an expression affect variables used in a later part\n");
    tf12 = transformer_combine(tf1, tf2);
  }
  else if (transformer_safe_affect_transformer_p(tf2, tf1)){
    pips_debug(9, "Side effects of tf2 on tf1\n");
    pips_user_warning("Non standard compliant code: side effect in part\n"
		      "of an expression affect variables used in an earlier part\n");
    tf12 = transformer_combine(tf1, tf2);
  }
  else {
    pips_debug(9, "No adversary side effects\n");
    if(transformer_undefined_p(tf1)
       || transformer_undefined_p(tf2)
       || (ENDP(transformer_arguments(tf1)) && ENDP(transformer_arguments(tf2)))) {
      /* No side effects at all */
      pips_debug(9, "No side effects at all\n");
      tf12 = transformer_safe_intersection(tf1, tf2);
      free_transformer(tf1);
    }
    else {
    pips_debug(9, "Side effects on other variables\n");
      tf12 = transformer_combine(tf1, tf2);
    }
  }

  tf = transformer_safe_image_intersection(tf12, tf_op);

  /*
  ifdebug(9) {
    pips_debug(9, "before projection, with temporaries v=%s, tmp1=%s, tmp2=%s,"
	       " transformer rf=%p\n",
	       entity_local_name(v), entity_local_name(tmp1),
	       entity_local_name(tmp2), tf);
    dump_transformer(tf);
  }
  */

  /* Too early: you are projecting v and loosing all useful information
     within an expression! */
  /* tf = transformer_temporary_value_projection(tf); */
    
  free_transformer(tf2);
  free_transformer(tf12);
  free_transformer(tf_op);

  pips_debug(9, "End with transformer tf=%p\n", tf);
  ifdebug(9) dump_transformer(tf);

  return tf;
}

static transformer constant_to_transformer(entity v,
					   expression rhs)
{
  transformer tf = transformer_undefined;
  syntax srhs = expression_syntax(rhs);

  pips_debug(9, "Begin for entity %s",
	     entity_local_name(v));

  /* the rhs is a call to a constant function */
  if(syntax_call_p(srhs)) {
    entity f = call_function(syntax_call(srhs));

    pips_debug(9, " and constant %s\n", module_local_name(f));

    if(entity_constant_p(f)) {
      basic fb = entity_basic(f);

      if(basic_string_p(fb)) {
	basic vb = entity_basic(v);
	if(basic_string_p(vb)) {
	  int llhs = string_type_size(vb);
	  int lrhs = string_type_size(fb);

	  if(llhs==-1 || llhs >= lrhs) {
	    /* Implicitly sufficient length or implicit padding with SPACEs */
	    tf = simple_equality_to_transformer(v, f, FALSE);
	  }
	  else {
	    /* Take the prefix of the constant. Expect problems with PIPS quotes... */
	    string n = malloc(llhs+3);
	    entity f1 = entity_undefined;

	    n = strncpy(n, module_local_name(f), llhs+1);
	    *(n+llhs+1) = '"';
	    *(n+llhs+2) = 0;
	    f1 = make_constant_entity(n, is_basic_string, llhs);
	    tf = simple_equality_to_transformer(v, f1, FALSE);
	  }
	}
	else {
	  pips_user_error("CHARACTER variable assigned a non CHARACTER constant, \"%s\"\n",
			  module_local_name(f));
	}
      }
      else {
	/* This is not a constant string */
	tf = simple_equality_to_transformer(v, f, FALSE);
      }
    }
  }

  pips_debug(9, "End for entity %s\n",
	     entity_local_name(v));

  return tf;
}

/* HANDLING OF CONDITIONS */

#   define DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN 7

/* call transformer_add_condition_information_updown() recursively
 * on both sub-expressions if veracity == TRUE; else
 * try your best...
 *
 */

static transformer 
transformer_add_anded_conditions_updown(
    transformer pre,
    expression c1,
    expression c2,
    bool veracity,
    bool upwards)
{
  static transformer transformer_add_condition_information_updown(
								  transformer,
								  expression,
								  bool,
								  bool);
  transformer newpre = transformer_undefined;

  pips_debug(9,"Begin with pre=%p, veracity=%s, upwards=%s\n,",
	     pre, bool_to_string(veracity),
	     bool_to_string(upwards));

  if(veracity) {
    /* FI: this is a bit confusing because of aliasing conditions.
     * The second condition is added after the first one was analyzed
     * and the "and" effect is enforced by side effect
     *
     * This is equivalent to allocating copies of pre and then computing
     * their intersection.
     */
    transformer pre1 = transformer_undefined;
    ifdebug(9) {
      pips_debug(9, "pre=%p:\n", pre);
      dump_transformer(pre);
    }
	  
    pre1 = transformer_add_condition_information_updown
      (pre, c1, veracity, upwards);

    ifdebug(9) {
      pips_debug(9, "pre1=%p:\n", pre1);
      dump_transformer(pre1);
    }
	  
    newpre = transformer_add_condition_information_updown
      (pre1, c2, veracity, upwards);

    ifdebug(9) {
      pips_debug(9, "newpre=%p:\n", newpre);
      dump_transformer(newpre);
    }
  }
  else if(!upwards) {
    /* use precondition pre and hope than the convex hull won't destroy
       all information */
    /* compute !(c1&&c2) as !c1 || !c2 */
    transformer pre1 = transformer_dup(pre);
    transformer pre2 = pre;

    pre1 = transformer_add_condition_information_updown
      (pre1, c1, FALSE, upwards);
    pre2 = transformer_add_condition_information_updown
      (pre2, c2, FALSE, upwards);
    newpre = transformer_convex_hull(pre1, pre2);
		  
    ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
		 "pre1 =\n");
      dump_transformer(pre1);
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
		 "pre2 =\n");
      dump_transformer(pre2);
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
		 "newpre =\n");
      dump_transformer(newpre);
    }
  }
  else {
    /* might be possible to add a convex hull on the initial values
     * as in !upwards
     */
    newpre = pre;
  }

  ifdebug(9) {
    pips_debug(9, "end with newpre=%p\n", newpre);
    dump_transformer(newpre);
  }

  return newpre;
}

/* call transformer_add_condition_information_updown() recursively on both
 * sub-expressions if veracity == FALSE; else try to do your best...
 *
 * Should be fused with the .AND. case?!? Careful with veracity...
 *
 */

static transformer transformer_add_ored_conditions_updown(
    transformer pre,
    expression c1,
    expression c2,
    bool veracity,
    bool upwards)
{
  static transformer transformer_add_condition_information_updown(
    transformer,
    expression,
    bool,
    bool);
  transformer newpre = transformer_undefined;

  pips_debug(9,"Begin with pre=%p, veracity=%s, upwards=%s\n,",
	     pre, bool_to_string(veracity),
	     bool_to_string(upwards));

  if(!veracity) {
    /* compute !(c1||c2) as !c1 && !c2 */
    newpre = transformer_add_condition_information_updown
      (pre, c1, FALSE, upwards);
    newpre = transformer_add_condition_information_updown
      (newpre, c2, FALSE, upwards);
  }
  else if(!upwards) {
    /* compute (c1||c2) as such */
    transformer pre1 = transformer_dup(pre);
    transformer pre2 = pre;

    pre1 = transformer_add_condition_information_updown
      (pre1, c1, TRUE, upwards);
    pre2 = transformer_add_condition_information_updown
      (pre2, c2, TRUE, upwards);
    newpre = transformer_convex_hull(pre1, pre2);
		      
    /* free_transformer(pre1); */
    /* free_transformer(pre2); */
		      
    ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	    "pre1 =\n");
      (void) (void) dump_transformer(pre1);
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	    "pre2 =\n");
      (void) dump_transformer(pre2);
      pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	    "newpre =\n");
      (void) (void) dump_transformer(newpre);
    }
  }
  else {
    /* might be possible to add a convex hull on the initial values
     * as in !upwards
     */
    newpre = pre;
  }

  ifdebug(9) {
    pips_debug(9, "end with newpre=%p\n", newpre);
    dump_transformer(newpre);
  }

  return newpre;
}

static transformer 
transformer_add_call_condition_information_updown(
    transformer pre,
    entity op,
    list args,
    bool veracity,
    bool upwards)
{
  static transformer transformer_add_condition_information_updown(
    transformer,
    expression,
    bool,
    bool);
  transformer newpre = transformer_undefined;
  expression c1 = expression_undefined;
  expression c2 = expression_undefined;

  pips_debug(9,"Begin with pre=%p, op=%s, veracity=%s, upwards=%s\n,",
	     pre, module_local_name(op), bool_to_string(veracity),
	     bool_to_string(upwards));

  if(!ENDP(args)) {
    c1 = EXPRESSION(CAR(args));
    if(!ENDP(CDR(args))) {
      c2 = EXPRESSION(CAR(CDR(args)));
    }
  }

  if(ENTITY_RELATIONAL_OPERATOR_P(op)) {
    cons * ef1 = expression_to_proper_effects(c1);
    cons * ef2 = expression_to_proper_effects(c2);
	
    if(integer_scalar_read_effects_p(ef1) && integer_scalar_read_effects_p(ef2)) {
      newpre = transformer_add_integer_relation_information(pre, op, c1, c2, 
							    veracity, upwards);
    }
    else {
      newpre = transformer_add_any_relation_information(pre, op, c1, c2, 
							veracity);
    }

    /* Help! I need a free_effects(ef) here */
    //    free_effect(ef1);
    //    free_effect(ef2);
  }
  else if(ENTITY_AND_P(op)) {
    newpre = transformer_add_anded_conditions_updown
      (pre, c1, c2, veracity, upwards);
  }
  else if(ENTITY_OR_P(op)) {
    newpre = transformer_add_ored_conditions_updown
      (pre, c1, c2, veracity, upwards);
  }
  else if(ENTITY_NOT_P(op)) {
    newpre = transformer_add_condition_information_updown
      (pre, c1, !veracity, upwards);
  }
  else if((ENTITY_TRUE_P(op) && !veracity) ||
	  (ENTITY_FALSE_P(op) && veracity)) {
    free_transformer(pre);
    newpre = transformer_empty();
  }
  else
    /* do not know what to do with other logical operators, for the time being! 
     * keep pre unmodified
     */
    newpre = pre;

  ifdebug(9) {
    pips_debug(9, "end with newpre=%p\n", newpre);
    dump_transformer(newpre);
  }

  return newpre;
}

/* Non-convex information can be made convex when moving postcondition downwards
 * because some of the parts introducing non-convexity may be made empty by
 * the pre-existing precondition.
 *
 * This is not true for transformers computed upwards, although conditions on
 * initial values could also be added. Let's wait for a case which would prove
 * this useful.
 *
 * Argument pre is modified and may or not be returned as newpre. It may also be
 * freed and newpre allocated. In other words, argument "pre" should not be used
 * after a call to this function.
 */
static transformer 
transformer_add_condition_information_updown(
    transformer pre,
    expression c,
    bool veracity,
    bool upwards)
{
  /* default option: no condition can be added */
  transformer newpre = pre;
  syntax s = expression_syntax(c);

  ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
    pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	  "begin upwards=%s veracity=%s c=", 
	  bool_to_string(upwards), bool_to_string(veracity));
    print_expression(c);
    (void) fprintf(stderr,"pre=%p\n", pre);
    dump_transformer(pre);
  }

  switch(syntax_tag(s)){
  case is_syntax_call: 
    {
      entity f = call_function(syntax_call(s));
      list args = call_arguments(syntax_call(s));

      newpre = transformer_add_call_condition_information_updown
	(pre, f, args, veracity, upwards);
      break;
    }
  case is_syntax_reference:
    {
      /* A logical variable must be referenced */
      entity l = reference_variable(syntax_reference(s));
      if(entity_has_values_p(l)) {
	entity l_new = entity_to_new_value(l);
	Pvecteur eq = vect_new((Variable) l_new, VALUE_ONE);

	if(veracity) {
	  vect_add_elem(&eq, TCST, VALUE_MONE);
	}

	newpre = transformer_equality_add(pre, eq);
      }
      break;
    }
  case is_syntax_range:
    {
      pips_internal_error("range used as test condition!\n");
      break;
    }
  default:
      pips_internal_error("ill. expr. as test condition\n");
  }

  ifdebug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN) {
    pips_debug(DEBUG_TRANSFORMER_ADD_CONDITION_INFORMATION_UPDOWN,
	  "end newpre=%p\n", newpre);
    dump_transformer(newpre);
  }

  return newpre;
}

transformer 
transformer_add_condition_information(
    transformer pre,
    expression c,
    bool veracity)
{
    transformer post = 
	transformer_add_condition_information_updown(pre, c, veracity, TRUE);

    post = transformer_temporary_value_projection(post);
    reset_temporary_value_counter();

    return post;
}

transformer
precondition_add_condition_information(
    transformer pre,
    expression c,
    bool veracity)
{
    transformer post = 
	transformer_add_condition_information_updown(pre, c, veracity, FALSE);

    post = transformer_temporary_value_projection(post);
    reset_temporary_value_counter();

    return post;
}

/* INTEGER EXPRESSIONS */
transformer 
simple_affine_to_transformer(entity e, Pvecteur a, bool is_internal)
{
  transformer tf = transformer_undefined;
  Pvecteur ve = vect_new((Variable) e, VALUE_ONE);
  Pvecteur vexpr = vect_dup(a);
  Pcontrainte c;
  Pvecteur eq = VECTEUR_NUL;

  pips_debug(8, "begin with is_internal=%s\n", bool_to_string(is_internal));

  ifdebug(9) {
    pips_debug(9, "\nLinearized expression:\n");
    vect_dump(vexpr);
  }

  if(!is_internal
     || (value_mappings_compatible_vector_p(ve) &&
	 value_mappings_compatible_vector_p(vexpr))) {
    /* The renaming used to eliminate equivalenced variables cannot be
       performed because e may be a temporary value. It is better to
       perform it at a higher level, for instance at the assignement
       level. */
    eq = vect_substract(ve, vexpr);
    vect_rm(ve);
    vect_rm(vexpr);
    c = contrainte_make(eq);
    tf = make_transformer(NIL,
			  make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));
  }
  else {
    vect_rm(eq);
    vect_rm(ve);
    vect_rm(vexpr);
    tf = transformer_undefined;
  }

  pips_debug(8, "end with tf=%p\n", tf);
  ifdebug(8) dump_transformer(tf);

  return tf;
}

transformer 
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

  pips_debug(8, "begin\n");

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

  pips_debug(8, "end\n");

  return tf;
}

transformer 
affine_increment_to_transformer(entity e, Pvecteur a)
{
    transformer tf = transformer_undefined;

    tf = affine_to_transformer(e, a, FALSE);

    return tf;
}

static transformer modulo_to_transformer(entity e, /* assumed to be a value */
					 expression arg1,
					 expression arg2,
					 transformer pre,
					 bool is_internal)
{
  transformer tf = transformer_undefined;

  debug(8, "modulo_to_transformer", "begin\n");

  pips_assert("Precondition is unused", pre==pre);
  pips_assert("arg1 is unused, shut up the compiler", arg1==arg1);
  pips_assert("arg1 is unused, shut up the compiler", is_internal==is_internal);
    
  if(integer_constant_expression_p(arg2)) {
    int d = integer_constant_expression_value(arg2);
    Pvecteur ub = vect_new((Variable) e, VALUE_ONE);
    Pvecteur lb = vect_new((Variable) e, VALUE_MONE);
    Pcontrainte clb = contrainte_make(lb);
    Pcontrainte cub = CONTRAINTE_UNDEFINED;

    vect_add_elem(&ub, TCST, int_to_value(1-d));
    vect_add_elem(&lb, TCST, int_to_value(d-1));
    cub = contrainte_make(ub);
    clb->succ = cub;
    tf = make_transformer(NIL,
			  make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
  }

  ifdebug(8) {
    debug(8, "modulo_to_transformer", "result:\n");
    dump_transformer(tf);
    debug(8, "modulo_to_transformer", "end\n");
  }

  return tf;
}

static transformer iabs_to_transformer(entity e, /* assumed to be a value */
				       expression expr,
				       transformer pre)
{
  transformer tf = transformer_undefined;
  normalized n = NORMALIZE_EXPRESSION(expr);

  pips_debug(8, "begin\n");

  pips_assert("Precondition is unused", transformer_undefined_p(pre));

  if(normalized_linear_p(n)) {
    Pvecteur vlb1 = vect_dup((Pvecteur) normalized_linear(n));
    Pvecteur vlb2 = vect_multiply(vect_dup((Pvecteur) normalized_linear(n)), VALUE_MONE);
    Pcontrainte clb1 = CONTRAINTE_UNDEFINED;
    Pcontrainte clb2 = CONTRAINTE_UNDEFINED;
    cons * tf_args = NIL;

    vect_add_elem(&vlb1, (Variable) e, VALUE_MONE);
    vect_add_elem(&vlb2, (Variable) e, VALUE_MONE);
    clb1 = contrainte_make(vlb1);
    clb2 = contrainte_make(vlb2);
    clb1->succ = clb2;
    tf = make_transformer(tf_args,
			  make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb1)));
  }

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

static transformer 
integer_divide_to_transformer(entity e,
			      expression arg1,
			      expression arg2,
			      transformer pre,
			      bool is_internal)
{
  transformer tf = transformer_undefined;
  normalized n1 = NORMALIZE_EXPRESSION(arg1);

  pips_debug(8, "begin with is_internal=%s\n",
	     bool_to_string(is_internal));
 
  /* pips_assert("Precondition is unused", transformer_undefined_p(pre)); */
  pips_assert("Shut up the compiler", pre==pre);

  if(integer_constant_expression_p(arg2) && normalized_linear_p(n1)) {
    int d = integer_constant_expression_value(arg2);
    /* must be duplicated right now  because it will be
       renamed and checked at the same time by
       value_mappings_compatible_vector_p() */
    Pvecteur vlb =
      vect_multiply(vect_dup(normalized_linear(n1)), VALUE_MONE); 
    Pvecteur vub = vect_dup(normalized_linear(n1));
    Pcontrainte clb = CONTRAINTE_UNDEFINED;
    Pcontrainte cub = CONTRAINTE_UNDEFINED;

    vect_add_elem(&vlb, (Variable) e, int_to_value(d));
    vect_add_elem(&vub, (Variable) e, int_to_value(-d));
    vect_add_elem(&vub, TCST, int_to_value(1-d));
    clb = contrainte_make(vlb);
    cub = contrainte_make(vub);
    clb->succ = cub;
    tf = make_transformer(NIL,
			  make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
  }

  ifdebug(8) {
    debug(8, "integer_divide_to_transformer", "result:\n");
    dump_transformer(tf);
    debug(8, "integer_divide_to_transformer", "end\n");
  }

  return tf;
}

static transformer 
integer_multiply_to_transformer(entity v,
				expression e1,
				expression e2,
				transformer pre,
				bool is_internal)
{
  transformer tf = transformer_undefined;

  pips_assert("Shut up the compiler",
	      v==v && e1==e1 && e2==e2 && pre==pre && is_internal==is_internal);

  return tf;
}

static transformer 
integer_power_to_transformer(entity e,
			     expression arg1,
			     expression arg2,
			     transformer pre,
			     bool is_internal)
{
  transformer tf = transformer_undefined;
  normalized n1 = NORMALIZE_EXPRESSION(arg1);
  normalized n2 = NORMALIZE_EXPRESSION(arg2);

  debug(8, "integer_power_to_transformer", "begin\n");

  pips_assert("Precondition is unused", transformer_undefined_p(pre));
  pips_assert("Shut up the compiler", is_internal==is_internal);

  if(signed_integer_constant_expression_p(arg2) && normalized_linear_p(n1)) {
    int d = signed_integer_constant_expression_value(arg2);

    if(d%2==0) {

      if(d==0) {
	/* 1 is assigned unless arg1 equals 0... which is neglected */
	Pvecteur v = vect_new((Variable) e, VALUE_ONE);

	vect_add_elem(&v, TCST, VALUE_MONE);
	tf = make_transformer(NIL,
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

	vect_add_elem(&vlb1, (Variable) e, VALUE_MONE);
	vect_add_elem(&vlb2, (Variable) e, VALUE_MONE);
	clb1 = contrainte_make(vlb1);
	clb2 = contrainte_make(vlb2);
	clb1->succ = clb2;
	tf = make_transformer(NIL,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb1)));
      }
      else {
	/* d is negative and even */
	Pvecteur vub = vect_new((Variable) e, VALUE_ONE);
	Pvecteur vlb = vect_new((Variable) e, VALUE_MONE);
	Pcontrainte clb = CONTRAINTE_UNDEFINED;
	Pcontrainte cub = CONTRAINTE_UNDEFINED;

	vect_add_elem(&vub, TCST, VALUE_MONE);
	clb = contrainte_make(vlb);
	cub = contrainte_make(vub);
	clb->succ = cub;
	tf = make_transformer(NIL,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
      }
    }
    else if(d<0) {
      /* d is negative, arg1 cannot be 0, expression value is -1, 0
	 or 1 */
      Pvecteur vub = vect_new((Variable) e, VALUE_MONE);
      Pvecteur vlb = vect_new((Variable) e, VALUE_ONE);
      Pcontrainte clb = CONTRAINTE_UNDEFINED;
      Pcontrainte cub = CONTRAINTE_UNDEFINED;

      vect_add_elem(&vub, TCST, VALUE_MONE);
      vect_add_elem(&vlb, TCST, VALUE_MONE);
      clb = contrainte_make(vlb);
      cub = contrainte_make(vub);
      clb->succ = cub;
      tf = make_transformer(NIL,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }
    else if(d==1) {
      Pvecteur v = vect_dup(normalized_linear(n1));
      
      vect_add_elem(&v, (Variable) e, VALUE_MONE);
      tf = make_transformer(NIL,
			    make_predicate(sc_make(contrainte_make(v),
						   CONTRAINTE_UNDEFINED)));
    }
  }
  else if(signed_integer_constant_expression_p(arg1)) {
    int d = signed_integer_constant_expression_value(arg1);

    if(d==0||d==1) {
      /* 0 or 1 is assigned unless arg2 equals 0... which is neglected */
      Pvecteur v = vect_new((Variable) e, VALUE_ONE);

      vect_add_elem(&v, TCST, int_to_value(-d));
      tf = make_transformer(NIL,
			    make_predicate(sc_make(contrainte_make(v),
						   CONTRAINTE_UNDEFINED)));
    }
    else if(d > 1) {
      /* the assigned value is positive */
      Pvecteur v1 = vect_new((Variable) e, VALUE_MONE);
      Pcontrainte c1 = contrainte_make(v1);

      if(normalized_linear_p(n2)) {
	Pvecteur v2 = vect_dup(normalized_linear(n2));
	Pcontrainte c2 = CONTRAINTE_UNDEFINED;
	Pvecteur v3 = vect_multiply(vect_dup(normalized_linear(n2)), (Value) d);
	Pcontrainte c3 = CONTRAINTE_UNDEFINED;

	vect_add_elem(&v2, TCST, VALUE_ONE);
	vect_add_elem(&v2, (Variable) e, VALUE_MONE);
	c2 = contrainte_make(v2);
	contrainte_succ(c1) = c2;
	vect_add_elem(&v3, (Variable) e, VALUE_MONE);
	c3 = contrainte_make(v3);
	contrainte_succ(c2) = c3;
      }

      tf = make_transformer(NIL,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, c1)));
    }
    else if(d == -1) {
      /* The assigned value is 1 or -1 */
      Pvecteur vub = vect_new((Variable) e, VALUE_MONE);
      Pvecteur vlb = vect_new((Variable) e, VALUE_ONE);
      Pcontrainte clb = CONTRAINTE_UNDEFINED;
      Pcontrainte cub = CONTRAINTE_UNDEFINED;

      vect_add_elem(&vub, TCST, VALUE_MONE);
      vect_add_elem(&vlb, TCST, VALUE_MONE);
      clb = contrainte_make(vlb);
      cub = contrainte_make(vub);
      clb->succ = cub;
      tf = make_transformer(NIL,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }
  }

  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

static transformer 
integer_minmax_to_transformer(entity e,
			      list args,
			      transformer pre,
			      bool is_min,
			      bool is_internal)
{
  transformer tf = transformer_undefined;
  expression arg = expression_undefined;
  normalized n = normalized_undefined;
  list cexpr;
  Pcontrainte cl = CONTRAINTE_UNDEFINED;

  pips_debug(8, "begin for %s %s\n",
	     is_min? "minimum" : "maximum",
	     is_internal? "intraprocedural" : "interprocedural");

  pips_assert("Precondition is unused", transformer_undefined_p(pre));

  for(cexpr = args; !ENDP(cexpr); POP(cexpr)) {
    arg = EXPRESSION(CAR(cexpr));
    n = NORMALIZE_EXPRESSION(arg);

    if(normalized_linear_p(n)) {
      Pvecteur v = vect_dup((Pvecteur) normalized_linear(n));
      Pcontrainte cv = CONTRAINTE_UNDEFINED;

      vect_add_elem(&v, (Variable) e, VALUE_MONE);

      if(is_min) {
	v = vect_multiply(v, VALUE_MONE);
      }

      cv = contrainte_make(v);
      cv->succ = cl;
      cl = cv;

    }
  }

  if(CONTRAINTE_UNDEFINED_P(cl) || CONTRAINTE_NULLE_P(cl)) {
    Psysteme sc = sc_make(CONTRAINTE_UNDEFINED, cl);

    sc_base(sc) = base_add_variable(BASE_NULLE, (Variable) e);
    sc_dimension(sc) = 1;
    tf = make_transformer(NIL,
			  make_predicate(sc));
  }
  else {
    /* A miracle occurs and the proper basis is derived from the
       constraints ( I do not understand why the new and the old value
       of e both appear... so it may not be necessary for the
       consistency check... I'm lost, FI, 6 Jan. 1999) */
    tf = make_transformer(NIL,
			  make_predicate(sc_make(CONTRAINTE_UNDEFINED, cl)));
  }


  ifdebug(8) {
    pips_debug(8, "result:\n");
    dump_transformer(tf);
    pips_debug(8, "end\n");
  }

  return tf;
}

static transformer 
min0_to_transformer(entity e, list args, transformer pre, bool is_internal)
{
  return integer_minmax_to_transformer(e, args, pre, TRUE, is_internal);
}

static transformer 
max0_to_transformer(entity e, list args, transformer pre, bool is_internal)
{
  return integer_minmax_to_transformer(e, args, pre, FALSE, is_internal);
}

/* */
static transformer 
integer_unary_operation_to_transformer(
    entity e,
    entity op,
    expression e1,
    transformer pre,
    bool is_internal)
{
  transformer tf = transformer_undefined;

  tf = generic_unary_operation_to_transformer(e, op, e1, pre, is_internal);

  if(transformer_undefined_p(tf)) {
    if(ENTITY_IABS_P(op)) {
      tf = iabs_to_transformer(e, e1, pre);
    }
  }

  return tf;
}

static transformer 
integer_binary_operation_to_transformer(
    entity e,
    entity op,
    expression e1,
    expression e2,
    transformer pre,
    bool is_internal)
{
  transformer tf = transformer_undefined;

  if(ENTITY_PLUS_P(op) || ENTITY_MINUS_P(op)) {
    tf = addition_operation_to_transformer(e, e1, e2, pre, ENTITY_PLUS_P(op), is_internal);
  }
  else if(ENTITY_MULTIPLY_P(op)) {
    tf = integer_multiply_to_transformer(e, e1, e2, pre, is_internal);
  }
  else if(ENTITY_MODULO_P(op)) {
    tf = modulo_to_transformer(e, e1, e2, pre, is_internal);
  }
  else if(ENTITY_DIVIDE_P(op)) {
    tf = integer_divide_to_transformer(e, e1, e2, pre, is_internal);
  }
  else if(ENTITY_POWER_P(op)) {
    tf = integer_power_to_transformer(e, e1, e2, pre, is_internal);
  }

  return tf;
}

static transformer 
integer_call_expression_to_transformer(
    entity e,
    expression expr, /* needed to compute effects for user calls */
    transformer pre,
    bool is_internal)
{
  syntax sexpr = expression_syntax(expr);
  call c = syntax_call(sexpr);
  entity f = call_function(c);
  list args = call_arguments(c);
  transformer tf = transformer_undefined;
  int arity = gen_length(args);

  pips_debug(9, "Begin\n");

  /* tests are organized to trap 0-ary user-defined functions as well as
     binary min and max */

  if(ENTITY_MIN0_P(f)||ENTITY_MIN_P(f)) {
    tf = min0_to_transformer(e, args, pre, is_internal);
  }
  else if(ENTITY_MAX0_P(f)||ENTITY_MAX_P(f)) {
    tf = max0_to_transformer(e, args, pre, is_internal);
  }
  else if(value_code_p(entity_initial(f))) {
    if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      tf = user_function_call_to_transformer(e, expr);
    }
    else {
      /* No need to use effects here. It will be done at a higher level */
      ;
    }
  }
  else if(arity==0) {
    /* integer constant must have been detected earlier as linear (affine)
       expressions */
    pips_user_error("Call to %s with no arguments", entity_name(f));
  }
  else if(arity==1) {
    expression e1 = EXPRESSION(CAR(args));
    tf = integer_unary_operation_to_transformer(e, f, e1, pre, is_internal);
  }
  else if(arity==2) {
    expression e1 = EXPRESSION(CAR(args));
    expression e2 = EXPRESSION(CAR(CDR(args)));
    tf = integer_binary_operation_to_transformer(e, f, e1, e2, pre, is_internal);
  }

  pips_debug(9, "End with tf=%p\n", tf);

  return tf;
}

transformer 
integer_expression_to_transformer(
    entity v,
    expression expr,
    transformer pre,
    bool is_internal) /* Do check wrt to value mappings... if you are
                           not dealing with interprocedural issues */
{
  transformer tf = transformer_undefined;
  normalized n = NORMALIZE_EXPRESSION(expr);
  syntax sexpr = expression_syntax(expr);

  pips_debug(8, "begin for expression: ");
  ifdebug(8) print_expression(expr);

  /* Assume: e is a value */

  if(normalized_linear_p(n)) {
    tf = simple_affine_to_transformer(v, (Pvecteur) normalized_linear(n), is_internal);
  }
  else if(syntax_call_p(sexpr)) {
    tf = integer_call_expression_to_transformer(v, expr, pre, is_internal);
  }
  else if(syntax_reference_p(sexpr)) {
      entity e = reference_variable(syntax_reference(sexpr));
      tf = generic_reference_to_transformer(v, e, is_internal);
  }
  else {
    /* vect_rm(ve); */
    tf = transformer_undefined;
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
    pips_internal_error("Unknown logical constant %s\n",
	       entity_name(f));
  }
  c = contrainte_make(eq);
  tf = make_transformer(NIL,
			make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));

  return tf;
}

static transformer logical_unary_operation_to_transformer(entity v,
							  call c,
							  transformer pre,
							  bool is_internal)
{
  transformer tf = transformer_undefined;
  entity op = call_function(c);
  expression arg = EXPRESSION(CAR(call_arguments(c)));
  Pvecteur eq = vect_new((Variable) v, VALUE_ONE);

  /* pips_assert("A unique argument", ENDP(CDR(call_arguments(c)))); */

  if(ENTITY_NOT_P(op)) {
    entity tmp = make_local_temporary_value_entity(entity_type(v));
    tf = logical_expression_to_transformer(tmp, arg, pre, is_internal);
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
							   call c,
							   transformer pre,
							   bool is_internal)
{
  entity op = call_function(c);
  Pvecteur eq1 = VECTEUR_NUL;
  Pvecteur eq2 = VECTEUR_NUL;
  Pvecteur eq3 = VECTEUR_NUL;
  expression arg1 = EXPRESSION(CAR(call_arguments(c)));
  expression arg2 = EXPRESSION(CAR(CDR(call_arguments(c))));
  entity tmp1 = make_local_temporary_value_entity(entity_type(v));
  entity tmp2 = make_local_temporary_value_entity(entity_type(v));
  transformer tf1 = logical_expression_to_transformer(tmp1, arg1, pre, is_internal);
  transformer tf2 = logical_expression_to_transformer(tmp2, arg2, pre, is_internal);
  transformer tf = transformer_safe_intersection(tf1, tf2);

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

  if(!transformer_undefined_p(tf)) {
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
      pips_internal_error("Unexpected binary logical operator %s\n",
			  entity_name(op));
    }
    tf = transformer_inequality_add(tf, eq1);
    tf = transformer_inequality_add(tf, eq2);
    if(!VECTEUR_NUL_P(eq3)) {
      tf = transformer_inequality_add(tf, eq3);
    }
    tf = transformer_logical_inequalities_add(tf, v);
  }

  ifdebug(9) {
    pips_debug(9, "End with transformer:\n");
    dump_transformer(tf);
  }

  return tf;
}

static transformer logical_binary_function_to_transformer(entity v,
							  call c,
							  transformer pre,
							  bool is_internal)
{
  transformer tf = transformer_undefined;
  entity op = call_function(c);

  if(ENTITY_AND_P(op)||ENTITY_OR_P(op)) {
    tf = logical_binary_operation_to_transformer(v, c, pre, is_internal);
  }
  else {
    expression expr1 = EXPRESSION(CAR(call_arguments(c)));
    expression expr2 = EXPRESSION(CAR(CDR(call_arguments(c))));
    basic b1 = basic_of_expression(expr1);
    basic b2 = basic_of_expression(expr2);

    if(basic_logical_p(b1) && basic_logical_p(b2)) {
      tf = logical_binary_operation_to_transformer(v, c, pre, is_internal);
    }
    else if(!transformer_undefined_p(pre)) {
      /* Non internal overloaded functions such as EQ, NEQ, GE, GT, LE, LT,... */
      entity tmp = make_local_temporary_value_entity(entity_type(v));
      transformer tfe = transformer_add_any_relation_information(copy_transformer(pre),
								 op,
								 expr1,
								 expr2,
								 TRUE);

      /* if the transformer is not feasible, return FALSE */
      if(!transformer_empty_p(tfe)) {
	Pvecteur eq = vect_new((Variable) tmp, VALUE_ONE);
	Pcontrainte c;

	c = contrainte_make(eq);
	tf = make_transformer(NIL,
			      make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));
      }
      free_transformer(tfe);
    }
    else {
      ;
    }

    free_basic(b1);
    free_basic(b2);
  }

  return tf;
}

/* v is assumed to be a temporary value and r a logical program variable */
static transformer logical_reference_to_transformer(entity v,
						    entity r,
						    bool is_internal)
{
  transformer tf = transformer_undefined;

  if(!is_internal || entity_has_values_p(r)) {
    entity r_new = is_internal? entity_to_new_value(r) : external_entity_to_new_value(r);


    tf = simple_equality_to_transformer(v, r_new, FALSE);
    tf = transformer_logical_inequalities_add(tf, r_new);
  }

  return tf;
}

/* Could be used to compute preconditions too. v is assumed to be a new
   value or a temporary value. */
transformer logical_expression_to_transformer(entity v,
					      expression rhs,
					      transformer pre,
					      bool is_internal)
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
      tf = logical_unary_operation_to_transformer(v, syntax_call(srhs), pre, is_internal);
      break;
    case 2:
      tf = logical_binary_function_to_transformer(v, syntax_call(srhs), pre, is_internal);
      break;
    default:
      pips_internal_error("Too many logical arguments, %d, for operator %s\n",
			  gen_length(call_arguments(syntax_call(srhs))),
			  entity_name(call_function(syntax_call(srhs))));
    }
    break;
  case is_syntax_reference:
    tf = logical_reference_to_transformer(v, reference_variable(syntax_reference(srhs)),
					  is_internal);
    break;
  case is_syntax_range:
    pips_internal_error("Unexpected tag %d\n", syntax_tag(srhs));
    break;
  default:
    pips_internal_error("Illegal tag %d\n", syntax_tag(srhs));
  }
  return tf;
}

transformer string_expression_to_transformer(entity v,
					     expression rhs)
{
  transformer tf = transformer_undefined;
  syntax srhs = expression_syntax(rhs);

  switch(syntax_tag(srhs)) {
  case is_syntax_call:
    switch(gen_length(call_arguments(syntax_call(srhs)))) {
    case 0:
      /* tf = constant_to_transformer(v, call_function(syntax_call(srhs))); */
      tf = constant_to_transformer(v, rhs);
      break;
    case 1:
      break;
    case 2:
      break;
    case 3:
      break;
    default:
      pips_internal_error("Too many arguments, %d, for operator %s\n",
			  gen_length(call_arguments(syntax_call(srhs))),
			  entity_name(call_function(syntax_call(srhs))));
    }
    break;
  case is_syntax_reference:
    {
      entity r = reference_variable(syntax_reference(srhs));

      if(entity_has_values_p(r)) {
	basic bv = variable_basic(type_variable(entity_type(v)));
	basic br = variable_basic(type_variable(entity_type(r)));

	if(basic_string_p(bv) && basic_string_p(br)) {
	  int llhs = string_type_size(bv);
	  int lrhs = string_type_size(br);

	  if(llhs == -1 || llhs >= lrhs) {
	    entity r_new = entity_to_new_value(r);
	    tf =  simple_equality_to_transformer(v, r_new, FALSE);
	  }
	}
	else {
	  pips_user_error("Incompatible type in CHARACTER assignment: %s\n",
			  basic_to_string(br));
	}
      }
    break;
    }
  case is_syntax_range:
    pips_internal_error("Unexpected tag %d\n", syntax_tag(srhs));
    break;
  default:
    pips_internal_error("Illegal tag %d\n", syntax_tag(srhs));
  }

  return tf;
}

static transformer 
float_unary_operation_to_transformer(
    entity e,
    entity op,
    expression e1,
    transformer pre,
    bool is_internal)
{
  transformer tf = transformer_undefined;

  tf = generic_unary_operation_to_transformer(e, op, e1, pre, is_internal);

  return tf;
}

static transformer 
float_binary_operation_to_transformer(
    entity e,
    entity op,
    expression e1,
    expression e2,
    transformer pre,
    bool is_internal)
{
  transformer tf = transformer_undefined;

  if(ENTITY_PLUS_P(op) || ENTITY_MINUS_P(op)) {
    tf = addition_operation_to_transformer(e, e1, e2, pre, ENTITY_PLUS_P(op), is_internal);
  }

  return tf;
}

static transformer 
float_call_expression_to_transformer(
    entity v,
    expression expr, /* needed to compute effects for user calls */
    transformer pre,
    bool is_internal)
{
  syntax sexpr = expression_syntax(expr);
  call c = syntax_call(sexpr);
  entity op = call_function(c);
  list args = call_arguments(c);
  transformer tf = transformer_undefined;
  int arity = gen_length(args);

  if(ENTITY_AMIN1_P(op)||ENTITY_DMIN1_P(op)||ENTITY_MIN_P(op)) {
    tf = generic_minmax_to_transformer(v, args, pre, TRUE, is_internal);
  }
  else if(ENTITY_AMAX1_P(op)||ENTITY_DMAX1_P(op)||ENTITY_MAX_P(op)) {
    tf = generic_minmax_to_transformer(v, args, pre, FALSE, is_internal);
  }
  else if(value_code_p(entity_initial(op))
	  && get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
    /* Just in case some integer variables were modified by the function
       call, e.g. the count of its dynamic calls. */
    tf = user_function_call_to_transformer(v, expr);
  }
  else {
    switch(arity) {
    case 0:
      /* tf = constant_to_transformer(v, call_function(syntax_call(srhs))); */
      tf = constant_to_transformer(v, expr);
      break;
    case 1:
      {
	expression e1 = EXPRESSION(CAR(args));
	tf = float_unary_operation_to_transformer(v, op, e1, pre, is_internal);
	break;
      }
    case 2:
      {
	expression e1 = EXPRESSION(CAR(args));
	expression e2 = EXPRESSION(CAR(CDR(args)));
	tf = float_binary_operation_to_transformer(v, op, e1, e2, pre, is_internal);
	break;
      }
    default:
      /* min, max,... */
      pips_user_warning("Operator %s not analyzed\n", entity_name(op));
    }
  }

  return tf;
}

transformer float_expression_to_transformer(entity v,
					    expression rhs,
					    transformer pre,
					    bool is_internal)
{
  transformer tf = transformer_undefined;
  syntax srhs = expression_syntax(rhs);

  /* pre can be used if some integer variables with constant values have
     to be promoted to float. */

  /* pips_assert("Precondition is undefined", transformer_undefined_p(pre)); */

  switch(syntax_tag(srhs)) {
  case is_syntax_call:
    tf = float_call_expression_to_transformer(v, rhs, pre, is_internal);
    break;
  case is_syntax_reference:
    {
      entity e = reference_variable(syntax_reference(srhs));
      tf = generic_reference_to_transformer(v, e, is_internal);
      break;
    }
  case is_syntax_range:
    pips_internal_error("Unexpected tag %d\n", syntax_tag(srhs));
    break;
  default:
    pips_internal_error("Illegal tag %d\n", syntax_tag(srhs));
  }

  return tf;
}

transformer transformer_add_any_relation_information(
	transformer pre, /* precondition */
	entity op,
	expression e1, 
	expression e2, 
	bool veracity)   /* the relation is true or not */
{
  basic b1 = basic_of_expression(e1);
  basic b2 = basic_of_expression(e2);

  pips_debug(8, "begin with pre=%p\n", pre);

  if(basic_tag(b1)==basic_tag(b2)) {

    switch(basic_tag(b1)) {
    case is_basic_logical:
      /* Logical are represented by integer values*/
    case is_basic_float:
      /* PIPS does not represent negative constants: call to unary_minus */
    case is_basic_complex:
      /* PIPS does not represent complex constants: call to CMPLX */
    case is_basic_string:
      /* Only constant string are processed */
    case is_basic_int:
      {
	/* This is not correct if side effects occur in e1 or e2 */
	/* This is very complicated to add constraints from tf1, tf2 and
           rel in pre! */
	entity tmp1 = make_local_temporary_value_entity_with_basic(b1);
	entity tmp2 = make_local_temporary_value_entity_with_basic(b2);
	transformer tf1 = any_expression_to_transformer(tmp1, e1, pre, TRUE);
	transformer tf2 = any_expression_to_transformer(tmp2, e2, pre, TRUE);
	transformer rel = relation_to_transformer(op, tmp1, tmp2, veracity);

	/* Memory leak with successive pre values */
	pre = transformer_safe_image_intersection(pre, tf1);
	pre = transformer_safe_image_intersection(pre, tf2);
	pre = transformer_safe_image_intersection(pre, rel);

	pips_assert("Pre is still consistent with tf1 and tf2 and rel",
		    transformer_argument_consistency_p(pre));

	free_transformer(tf1);
	free_transformer(tf2);
	free_transformer(rel);
	break;
      }
    case is_basic_overloaded:
      pips_internal_error("illegal overloaded type for operator %s\n",
		 entity_name(op));
      break;
    default:
      pips_internal_error("unknown basic b=%d\n", basic_tag(b1));
    }
  }

  free_basic(b1);
  free_basic(b2);

  /* pre may be unchanged when no information is derived */
  pips_debug(8, "end with pre=%p\n", pre);

  return pre;
}


transformer 
any_expression_to_transformer(
			      entity v,
			      expression expr,
			      transformer pre,
			      bool is_internal)
{
  transformer tf = transformer_undefined;
  basic be = basic_of_expression(expr);
  basic bv = variable_basic(type_variable(entity_type(v)));

  ifdebug(8) {
    pips_debug(8, "begin for entity %s of type %s and %s expression ", entity_local_name(v),
	       basic_to_string(bv), basic_to_string(be));
    print_expression(expr);
  }

  /* Assume v is a value */
  if( (basic_tag(bv)==basic_tag(be))
      || (basic_float_p(bv) && basic_int_p(be))) {
    switch(basic_tag(be)) {
    case is_basic_int:
      if(basic_int_p(bv)) {
	tf = integer_expression_to_transformer(v, expr, pre, is_internal);
      }
      else {
	pips_user_warning("Integer expression assigned to float value\n"
			  "Apply PIPS type checker for better results\n");
	/* Constants, at least, could be typed coerced */
	/* Redundant with explicit type coercion also available in PIPS */
	/* To be done later */
      }
      break;
    case is_basic_logical:
      tf = logical_expression_to_transformer(v, expr, pre, is_internal);
      break;
    case is_basic_float:
      /* PIPS does not represent negative constants: call to unary_minus */
      tf = float_expression_to_transformer(v, expr, pre, is_internal);
      break;
    case is_basic_complex:
      /* PIPS does not represent complex constants: call to CMPLX */
      break;
    case is_basic_string:
      /* Only constant string are processed */
      tf = string_expression_to_transformer(v, expr);
      break;
    case is_basic_overloaded:
      /* The overloading is supposed to have been lifted by
	 basic_of_expression() */
      pips_internal_error("illegal overloaded type for an expression\n");
      break;
    default:
      pips_internal_error("unknown basic b=%d\n", basic_tag(be));
    }
  }
  else {
    pips_user_warning("Implicit type coercion between %s variable and %s "
		      "expression may reduce semantic analysis accuracy.\n"
		      "Apply 'type_checker' to explicit all type coercions.\n",
		      basic_to_string(bv), basic_to_string(be));
  }

  /* tf may be transformer_undefined when no information is derived */
  pips_debug(8, "end with tf=%p\n", tf);

  return tf;
}

bool false_condition_wrt_precondition_p(expression c, transformer pre)
{
  bool result = FALSE;

  result = eval_condition_wrt_precondition_p(c, pre, FALSE);

  return result;
}

bool true_condition_wrt_precondition_p(expression c, transformer pre)
{
  bool result = FALSE;

  result = eval_condition_wrt_precondition_p(c, pre, TRUE);

  return result;
}

bool eval_condition_wrt_precondition_p(expression c, transformer pre, bool veracity)
{
  bool result = FALSE;
  transformer f = transformer_dup(pre);

  if(veracity) {
    f = precondition_add_condition_information(f, c, FALSE);
  }
  else {
    f = precondition_add_condition_information(f, c, TRUE);
  }

  result = transformer_empty_p(f);

  return result;
}

/* It is supposed to be obsolete but is still called. Maybe, it's only
   partly obsolete... If upwards is false, it is worth performing more
   convex hulls because the precondition on entry may restrain the space. */
transformer 
transformer_add_integer_relation_information(
    transformer pre,
    entity relop,
    expression e1,
    expression e2,
    bool veracity,
    bool upwards) /* upwards = transformer, !upwards = precondition */
{
#   define DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION 7
  /* default: no change */
  transformer newpre = pre;
  /* both expressions e1 and e2 must be affine */
  normalized n1 = NORMALIZE_EXPRESSION(e1);
  normalized n2 = NORMALIZE_EXPRESSION(e2);

  ifdebug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION) {
    pips_debug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION,
	       "begin upwards=%s veracity=%s relop=%s e1=", 
	       bool_to_string(upwards), bool_to_string(veracity),
	       entity_local_name(relop));
    print_expression(e1);
    (void) fprintf(stderr,"e2=");
    print_expression(e2);
    (void) fprintf(stderr,"pre=");
    print_transformer(pre);
  }

  if(normalized_linear_p(n1) && normalized_linear_p(n2)
     && value_mappings_compatible_vector_p((Pvecteur) normalized_linear(n1))
     && value_mappings_compatible_vector_p((Pvecteur) normalized_linear(n2))) {
    Pvecteur v1 = vect_dup((Pvecteur) normalized_linear(n1));
    Pvecteur v2 = vect_dup((Pvecteur) normalized_linear(n2));

    /* Make sure that new values only are used in v1 and v2 */
    variables_to_new_values(v1);
    variables_to_new_values(v2);

    if((ENTITY_EQUAL_P(relop) && veracity) ||
       (ENTITY_NON_EQUAL_P(relop) && !veracity)) {
      /* v1 - v2 == 0 */
      Pvecteur v = vect_substract(v1, v2);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_equality_add(pre, v);
    }
    else if((ENTITY_EQUAL_P(relop) && !veracity) ||
	    (ENTITY_NON_EQUAL_P(relop) && veracity)) {
      /* v2 - v1 + 1 <= 0 ou v1 - v2 + 1 <= 0 */
      /* FI: I do not know if this is valid when your are moving upwards
       * variables in v2 and v1 may have to be renamed as #init values (i.e. old values)
       */
      transformer prea = transformer_dup(pre);
      transformer preb = pre;
      Pvecteur va = vect_substract(v2, v1);
      Pvecteur vb = vect_substract(v1, v2);
      vect_add_elem(&va, TCST, VALUE_ONE);
      vect_add_elem(&vb, TCST, VALUE_ONE);
      /* FI: I think that this should be programmed (see comment above)
       * but I'm waiting for a bug to occur... (6 July 1993)
       *
       * FI: Well, the bug was eventually seen:-) (8 November 1995)
       */
      if(upwards) {
	upwards_vect_rename(va, pre);
	upwards_vect_rename(vb, pre);
      }
      prea = transformer_inequality_add(prea, va);
      preb = transformer_inequality_add(preb, vb);
      newpre = transformer_convex_hull(prea, preb);
      /* free_transformer(prea); */
      /* free_transformer(preb); */
    }
    else if ((ENTITY_GREATER_THAN_P(relop) && veracity) ||
	     (ENTITY_LESS_OR_EQUAL_P(relop) && !veracity)) {
      /* v2 - v1 + 1 <= 0 */
      Pvecteur v = vect_substract(v2, v1);
      vect_add_elem(&v, TCST, VALUE_ONE);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_inequality_add(pre, v);
    }
    else if ((ENTITY_GREATER_THAN_P(relop) && !veracity) ||
	     (ENTITY_LESS_OR_EQUAL_P(relop) && veracity)) {
      /* v1 - v2 <= 0 */
      Pvecteur v = vect_substract(v1, v2);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_inequality_add(pre, v);
    }
    else if ((ENTITY_GREATER_OR_EQUAL_P(relop) && veracity) ||
	     (ENTITY_LESS_THAN_P(relop) && !veracity)) {
      /* v2 - v1 <= 0 */
      Pvecteur v = vect_substract(v2, v1);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_inequality_add(pre, v);
    }
    else if ((ENTITY_GREATER_OR_EQUAL_P(relop) && !veracity) ||
	     (ENTITY_LESS_THAN_P(relop) && veracity)) {
      /* v1 - v2 + 1 <= 0 */
      Pvecteur v = vect_substract(v1, v2);
      vect_add_elem(&v, TCST, VALUE_ONE);
      if(upwards) {
	upwards_vect_rename(v, pre);
      }
      newpre = transformer_inequality_add(pre, v);
    }
    else {
      /* do nothing... although Malik may have tried harder! */
      newpre = pre;
    }
    vect_rm(v1);
    vect_rm(v2);
  }
  else
    /* do nothing, although MODULO and INTEGER DIVIDE could be handled */
    newpre = pre;

  ifdebug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION) {
    pips_debug(DEBUG_TRANSFORMER_ADD_RELATION_INFORMATION,
	       "end newpre=\n");
    print_transformer(newpre);
    pips_assert("Transformer is internally consistent", 
		transformer_internal_consistency_p(newpre));
  }

  return newpre;
}
