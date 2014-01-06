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
 /* semantical analysis
  *
  * phasis 2: propagate preconditions from statement to sub-statement,
  * starting from the module 1st statement
  *
  * For (simple) interprocedural analysis, this phasis should be performed
  * top-down on the call tree.
  *
  * Most functions are called xxx_to_postcondition although the purpose is
  * to compute preconditions. However transformer are applied to preconditions
  * to produce postconditions. Thus these modules store the preconditions
  * and then compute an independent (i.e. no sharing) postcondition which
  * is returned to be used by the caller.
  *
  * Preconditions are *NEVER* shared. Sharing would introduce desastrous
  * side effects when they are updated as for equivalenced variables and
  * would make freeing them impossible. Thus on a recursive path from
  * statement_to_postcondition() to itself the precondition must have been
  * reallocated even when its value is not changed as between a block
  * precondition and the first statement of the block precondition. In the
  * same way statement_to_postcondition() should never returned a
  * postcondition aliased with its precondition argument. Somewhere
  * in the recursive call down the data structures towards
  * call_to_postcondition() some allocation must take place even if the
  * statement as no effect on preconditions.
  *
  * Preconditions can be used to evaluate sub-expressions because Fortran
  * standard prohibit side effects within an expression. For instance, in:
  *  J = I + F(I)
  * function F cannot modify I.
  *
  * Ambiguity: the type "transformer" is used to abstract statement effects
  * as well as effects combined from the beginning of the module to just
  * before the current statement (precondition) to just after the current
  * statement (postcondition). This is because it was not realized that
  * preconditions could advantageously be seen as transformers of the initial
  * state when designing the ri.
  */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "transformer.h"

#include "semantics.h"

/* another non recursive section used to filter out preconditions */
static list module_global_arguments = NIL;

list
get_module_global_arguments()
{
    return module_global_arguments;
}

void set_module_global_arguments(list args)
{
    module_global_arguments = args;
}
/* end of the non recursive section */

transformer statement_to_postcondition(transformer, statement);

static transformer block_to_postcondition(transformer b_pre,
					  list b)
{
    statement s;
    transformer post;
    transformer s_pre = transformer_undefined;
    list ls = b;

    debug(8,"block_to_postcondition","begin pre=%x\n", b_pre);

    /* The first statement of the block must receive a copy
     * of the block precondition to avoid data sharing
     */

    if(ENDP(ls))
	/* to avoid accumulating equivalence equalities */
	post = transformer_dup(b_pre);
    else {
	s = STATEMENT(CAR(ls));
	s_pre = transformer_dup(b_pre);
	post = statement_to_postcondition(s_pre, s);
	for (POP(ls) ; !ENDP(ls); POP(ls)) {
	    s = STATEMENT(CAR(ls));
	    /* the precondition has been allocated as post */
	    s_pre = post;
	    post = statement_to_postcondition(s_pre, s);
	    // FI: within a long loop body, do not replicate the loop
	    // bound constraints
	    post = transformer_normalize(post, 2);
	}
    }

    pips_debug(8,"post=%p end\n", post);
    return post;
}

static transformer test_to_postcondition(transformer pre,
					 test t,
					 transformer tf)
{
#   define DEBUG_TEST_TO_POSTCONDITION 7
    expression e = test_condition(t);
    statement st = test_true(t);
    statement sf = test_false(t);
    transformer post;

    debug(DEBUG_TEST_TO_POSTCONDITION,"test_to_postcondition","begin\n");

    /* there are three levels of flow sensitivity and we have only a
       bool flag! FI */

    /* test conditions are assumed to have no side effects; it might
       be somewhere in the standard since functions called in an expression e
       cannot (should not) modify any variable used in e */

    if(pips_flag_p(SEMANTICS_FLOW_SENSITIVE) /* && !transformer_identity_p(tf) */) {
	/* convex hull might avoided if it is not required or if it is certainly useless 
	 * but test information should always be propagated 
	 */
      /* True and false condition transformers. The nest three variables should be freed. */
      transformer tct = condition_to_transformer(e, pre, true);
      transformer fct = condition_to_transformer(e, pre, false);
	  /* To obtain the best results, transformer_normalize(x,7) should
             be applied to subtransformers generated by
             precondition_add_condition_information() in case of equality
             condition and in case of string or float analysis. */
      //	transformer pret =
      //  precondition_add_condition_information(transformer_dup(pre),e, pre,
      //					  true);
      transformer pret = transformer_apply(tct, pre);

	transformer pref = transformer_undefined;

	transformer postt;
	transformer postf;

	/* "strong" transformer normalization to detect dead code generated by the
	 * test condition
	 */
	/* A normalization of degree 3 is fine */
	/* transformer_normalize(pret, 3); */
	transformer_normalize(pret, 7);
	  /* Just to get a stronger normalization with
	     sc_safe_normalize()... which is sc_normalize(), a weak
	     normalization function. FI: I do not understand what's
	     going on. */
	  pret = transformer_temporary_value_projection(pret);  

	/* FI, CA: the following "optimization" was added to avoid a call
	 * to Chernikova convex hull that core dumps:-(. 8  September 1993
	 *
	 * From a theoretical point of view, pref could always be computed.
	 *
	 * FI: removed because it is mathematically wrong in many cases;
	 * the negation of the test condition is lost! I keep the structure
	 * just in case another core dump occurs (25 April 1997).
	 */
	if(!empty_statement_p(sf)||true) {
	  /* To obtain the best results, transformer_normalize(x,7) should
             be applied to subtransformers generated by
             precondition_add_condition_information() in case of equality
             condition and in case of string or float analysis. */
	  /*
	    pref = precondition_add_condition_information(transformer_dup(pre),e,
							  pre, false);
	  */
	  pref = transformer_apply(fct, pre);
	  /* transformer_normalize(pref, 3); */
	  transformer_normalize(pref, 7);
	  /* Just to get a stronger normalization with sc_safe_normalize() */
	  pref = transformer_temporary_value_projection(pref);  
	}
	else {
	    /* do not try to compute a refined precondition for an empty block
	     * keep the current precondition to store in the precondition statement mapping
	     */
	    pref = transformer_dup(pre);
	}

	ifdebug(DEBUG_TEST_TO_POSTCONDITION) {
	    pips_debug(DEBUG_TEST_TO_POSTCONDITION, "pret=%p\n", pret);
	    (void) print_transformer(pret);
	    pips_debug(DEBUG_TEST_TO_POSTCONDITION, "pref=%p\n", pref);
	    (void) print_transformer(pref);
	}

	postt = statement_to_postcondition(pret, st);
	postf = statement_to_postcondition(pref, sf);
	post = transformer_convex_hull(postt, postf);
	free_transformer(postt);
	free_transformer(postf);
    }
    else {
      /* Be careful, pre is updated by statement_to_postcondition! */
	(void) statement_to_postcondition(pre, st);
	(void) statement_to_postcondition(pre, sf);
	post = transformer_apply(tf, pre);
    }

    ifdebug(DEBUG_TEST_TO_POSTCONDITION) {
	debug(DEBUG_TEST_TO_POSTCONDITION,"test_to_postcondition",
	      "end post=\n");
	(void) print_transformer(post);
    }

    return post;
}

static transformer 
expression_to_postcondition(
    transformer pre,
    expression exp,
    transformer tf)
{
  transformer post = transformer_undefined;

  if(get_bool_property("SEMANTICS_RECOMPUTE_EXPRESSION_TRANSFORMERS")) {
    /* Wild guess. See what should be done in call_to_postcondition() */
    //list el = expression_to_proper_effects(exp);
    list el = expression_to_proper_constant_path_effects(exp);
    transformer new_tf = expression_to_transformer(exp, pre, el);
    post = transformer_apply(new_tf, pre);
    free_transformer(new_tf);
  }
  else
    post = transformer_apply(tf, pre);

  return post;
}

static transformer call_to_postcondition(transformer pre,
					 call c,
					 transformer tf)
{
  transformer post = transformer_undefined;
  entity e = call_function(c);
  tag tt;

  pips_debug(8,"begin\n");

  switch (tt = value_tag(entity_initial(e))) {
  case is_value_intrinsic:
    /* there is room for improvement because assign is now the
       only well handled intrinsic */
    pips_debug(5, "intrinsic function %s\n",
	       entity_name(e));
    if(get_bool_property("SEMANTICS_RECOMPUTE_EXPRESSION_TRANSFORMERS")
       && ENTITY_ASSIGN_P(call_function(c))) {
      entity f = call_function(c);
      list args = call_arguments(c);
      /* impedance problem: build an expression from call c */
      expression expr = make_expression(make_syntax(is_syntax_call, c),
					normalized_undefined);
      //list ef = expression_to_proper_effects(expr);
      list ef = expression_to_proper_constant_path_effects(expr);
      transformer pre_r = transformer_range(pre);
      transformer new_tf = intrinsic_to_transformer(f, args, pre_r, ef);

      post = transformer_apply(new_tf, pre);
      syntax_call(expression_syntax(expr)) = call_undefined;
      free_expression(expr);
      free_transformer(new_tf);
      free_transformer(pre_r);
    }
    else {
      post = transformer_apply(tf, pre);
    }
    /* propagate precondition pre as summary precondition
       of user functions */
    /* FI: don't! Summary preconditions are computed independently*/
    /*
      if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      list args = call_arguments(c);
      expressions_to_summary_precondition(pre, args);
      }
    */
    break;
  case is_value_code:
    pips_debug(5, "external function %s\n", entity_name(e));
    if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      /*
	list args = call_arguments(c);

	transformer pre_callee = transformer_dup(pre);
	pre_callee =
	add_formal_to_actual_bindings(c, pre_callee);
	add_module_call_site_precondition(e, pre_callee);
      */
      /*
	expressions_to_summary_precondition(pre, args);
      */
    }
    post = transformer_apply(tf, pre);
    break;
  case is_value_symbolic:
  case is_value_constant: {
    /* Declared in preprocessor.h */
    /* This cannot occur in Fortran, but is possible in C. */
    if(c_module_p(get_current_module_entity())) {
      post = transformer_apply(tf, pre);
    }
    else {
      pips_internal_error("call to symbolic or constant %s",
			  entity_name(e));
    }
    break;
  }
  case is_value_unknown:
    pips_internal_error("unknown function %s", entity_name(e));
    break;
  default:
    pips_internal_error("unknown tag %d", tt);
  }

  pips_debug(8,"end\n");

  return post;
}

/******************************************************** DATA PRECONDITIONS */

/* a remake that works in all cases. FC. Isn'it a bit presumptuous? FI.
 *
 * This function works for C as well as Fortran. Its name should be
 * initial_value_to_preconditions.
 *
 * FI: It remains to be enhanced to handle more cases for non-integer
 * types. EvalExpression() should be extended to non-integer
 * types. The new fields of data structure "constant" should be
 * exploited.
 *
 */
static transformer fortran_data_to_prec_for_variables(entity m, list /* of entity */le)
{
  transformer pre = transformer_identity();
  transformer pre_r = transformer_undefined; // range of pre
  linear_hashtable_pt b = linear_hashtable_make(); /* already seen */
  //list ce = list_undefined;

  pips_debug(8, "begin for %s\n", module_local_name(m));

  /* look for entities with an initial value. */
  FOREACH(ENTITY, e, le) {
    value val = entity_initial(e);

    pips_debug(8, "begin for variable %s\n", entity_name(e));

    if (entity_has_values_p(e) && !linear_hashtable_isin(b, e)) {
      if(value_constant_p(val)) {
	constant c = value_constant(val);
	if (constant_int_p(c)) {
	  int int_val = constant_int(value_constant(val));

	  Pvecteur v = vect_make(VECTEUR_NUL,
				 (Variable) e, VALUE_ONE,
				 TCST, int_to_value(-int_val));
	  pre = transformer_equality_add(pre, v);
	  linear_hashtable_put_once(b, e, e);
	}
	else if (constant_call_p(c)) {
	  entity f = constant_call(c);
	  basic bt = variable_basic(type_variable(functional_result
						  (type_functional(entity_type(f)))));

	  if((basic_float_p(bt) && float_analyzed_p())
	     || (basic_string_p(bt) && string_analyzed_p())
	     || (basic_logical_p(bt) && boolean_analyzed_p()) ) {
	    Pvecteur v = vect_make(VECTEUR_NUL,
				   (Variable) e, VALUE_ONE,
				   (Variable) constant_call(c), VALUE_MONE,
				   TCST, VALUE_ZERO);
	    pre = transformer_equality_add(pre, v);
	    linear_hashtable_put_once(b, e, e);
	  }
	}
      }
      else if(value_expression_p(val)) {
	expression expr = value_expression(val);
	transformer npre = safe_any_expression_to_transformer(e, expr, pre, false);

	pre = transformer_combine(pre, npre);
	pre = transformer_safe_normalize(pre, 2);
	free_transformer(npre);
      }
    }
  }

  pre = transformer_temporary_value_projection(pre);
  pre_r = transformer_range(pre);
  free_transformer(pre);

  linear_hashtable_free(b);
  pips_assert("some transformer", pre_r != transformer_undefined);

  ifdebug(8) {
    dump_transformer(pre_r);
    pips_debug(8, "end for %s\n", module_local_name(m));
  }

  return pre_r;
}

static transformer c_data_to_prec_for_variables(entity m, list /* of entity */le)
{
  transformer pre = transformer_identity();
  transformer pre_r = transformer_undefined; // range of pre
  //linear_hashtable_pt b = linear_hashtable_make(); /* already seen */
  //list ce = list_undefined;

  pips_debug(8, "begin for %s\n", module_local_name(m));

  /* look for static entities with an initial value. */
  FOREACH(ENTITY, e, le) {
    if(variable_static_p(e) && entity_has_values_p(e)) {
      /* e may not have values but its initialization expression may
	 refer other variables which have values, however they are not
	 likely to be static initializations. */
      expression ie = variable_initial_expression(e);
      transformer pre_r = transformer_range(pre);
      transformer npre = transformer_undefined;
      if( !expression_undefined_p(ie) )
          npre = any_expression_to_transformer(e, ie, pre_r, false);
      /* any_expression_to_transformer may end up with an undefined transformer ... */
      if(transformer_undefined_p(npre))
          npre=transformer_identity();

      pips_debug(8, "begin for variable %s\n", entity_name(e));


      pre = transformer_combine(pre, npre);
      pre = transformer_safe_normalize(pre, 2);
      free_transformer(npre);
      free_transformer(pre_r);
    }
  }

  pre = transformer_temporary_value_projection(pre);
  pre_r = transformer_range(pre);
  free_transformer(pre);

  ifdebug(8) {
    dump_transformer(pre_r);
    pips_debug(8, "end for %s\n", module_local_name(m));
  }

  return pre_r;
}

static transformer data_to_prec_for_variables(entity m, list /* of entity */le)
{
  transformer tf = transformer_undefined;

  if(c_language_module_p(m))
    tf = c_data_to_prec_for_variables(m, le);
  else if(fortran_language_module_p(m) || fortran95_language_module_p(m))
    tf = fortran_data_to_prec_for_variables(m, le);
  else
    pips_internal_error("Unexpected language");

  return tf;
}

/* returns an allocated list of entities that appear in lef.
 * an entity may appear several times.
 */
list effects_to_entity_list(list lef)
{
  list le = NIL;
  MAP(EFFECT, e,
      le = CONS(ENTITY, reference_variable(effect_any_reference(e)), le),
      lef);
  return gen_nreverse(le);
}

/* restricted to variables with effects. */
transformer data_to_precondition(entity m)
{
  list lef = load_module_intraprocedural_effects(m);
  list le = effects_to_entity_list(lef);
  transformer pre = data_to_prec_for_variables(m, le);
  gen_free_list(le);
  return pre;
}

/* any variable is included. */
transformer all_data_to_precondition(entity m)
{
  /* FI: it would be nice, if only for debugging, to pass a more
     restricted list...

     This assumes the all variables declared in a statement is also
     declared at the module level. */
  //  transformer pre =
  // data_to_prec_for_variables(m, code_declarations(entity_code(m)));
  list dl = module_to_all_declarations(m);
  transformer pre = data_to_prec_for_variables(m, dl);

  gen_free_list(dl);

  return pre;
}

static transformer 
instruction_to_postcondition(
    transformer pre,
    instruction i,
    transformer tf)
{
    transformer post = transformer_undefined;
    test t;
    loop l;
    whileloop wl;
    forloop fl;
    call c;
    expression exp;

    pips_debug(9,"begin pre=%p tf=%p\n", pre, tf);

    switch(instruction_tag(i)) {
      case is_instruction_block:
	post = block_to_postcondition(pre, instruction_block(i));
	break;
      case is_instruction_test:
	t = instruction_test(i);
	post = test_to_postcondition(pre, t, tf);
	break;
      case is_instruction_loop:
	l = instruction_loop(i);
	post = loop_to_postcondition(pre, l, tf);
	break;
    case is_instruction_whileloop: {
	wl = instruction_whileloop(i);
	if(evaluation_before_p(whileloop_evaluation(wl)))
	  post = whileloop_to_postcondition(pre, wl, tf);
	else
	  post = repeatloop_to_postcondition(pre, wl, tf);
	break;
    }
    case is_instruction_forloop: {
	fl = instruction_forloop(i);
	post = forloop_to_postcondition(pre, fl, tf);
	break;
    }
      case is_instruction_goto:
	pips_internal_error("unexpected goto in semantics analysis");
	/* never reached: post = pre; */
	break;
      case is_instruction_call:
	c = instruction_call(i);
	post = call_to_postcondition(pre, c, tf);
	break;
      case is_instruction_unstructured:
	post = unstructured_to_postcondition(pre, instruction_unstructured(i),
					     tf);
	break ;
      case is_instruction_expression:
	exp = instruction_expression(i);
	post = expression_to_postcondition(pre, exp, tf);
	break ;
      case is_instruction_multitest:
	pips_internal_error("Should have been removed by the controlizer?");
	break ;
      default:
	pips_internal_error("unexpected tag %d", instruction_tag(i));
    }
    pips_debug(9,"resultat post, %p:\n", post);
    ifdebug(9) (void) print_transformer(post);
    return post;
}

/* Assume that all references are legal. Assume that variables used in
   array declarations are not modified in the module. */
static void add_reference_information(transformer pre, statement s, bool renaming)
{
  list efs = load_proper_rw_effects_list(s);

  FOREACH(EFFECT, e, efs) {
    reference r = effect_any_reference(e);
    list li = reference_indices(r);

    if(!ENDP(li)){
      entity v = reference_variable(r);
      variable tv = type_variable(ultimate_type(entity_type(v)));
      basic b = variable_basic(tv);
      list ld = NIL;

      pips_assert("Variable must be of type 'variable'",
		  type_variable_p(entity_type(v)));
      if(!basic_pointer_p(b)) {
	ld = variable_dimensions(tv);
	/* This assert is too strong in argument lists in Fortran and
	   everywhere in C */
	//pips_assert("Reference dimension = array dimension",
	//	  gen_length(li)==gen_length(ld));
	pips_assert("Reference dimension = array dimension",
		    gen_length(li)<=gen_length(ld));
	FOREACH(EXPRESSION, i, li) {
	  normalized ni = NORMALIZE_EXPRESSION(i);
	  if(normalized_linear_p(ni)) {
	    Pvecteur vi = normalized_linear(ni);
	    dimension d = DIMENSION(CAR(ld));
	    normalized nl = NORMALIZE_EXPRESSION(dimension_lower(d));
	    normalized nu = NORMALIZE_EXPRESSION(dimension_upper(d));
	    if(normalized_linear_p(nl) && normalized_linear_p(nu)) {
	      Pvecteur vl = normalized_linear(nl);
	      Pvecteur vu = normalized_linear(nu);

	      if(value_mappings_compatible_vector_p(vi)) {
		if(value_mappings_compatible_vector_p(vl)) {
		  Pvecteur cv = vect_substract(vl, vi);

		  if(renaming)
		    upwards_vect_rename(cv, pre);
		  if(!vect_constant_p(cv) || vect_coeff(TCST, cv) > 0) {
		    transformer_inequality_add(pre, cv);
		  }
		  else {
		    vect_rm(cv);
		  }
		}

		if(value_mappings_compatible_vector_p(vu)) {
		  Pvecteur cv = vect_substract(vi, vu);

		  if(renaming)
		    upwards_vect_rename(cv, pre);
		  if(!vect_constant_p(cv) || vect_coeff(TCST, cv) > 0) {
		    transformer_inequality_add(pre, cv);
		  }
		  else {
		    vect_rm(cv);
		  }
		}
	      }

	    }
	  }
	  POP(ld);
	}
      }
    }
  }
}

void precondition_add_reference_information(transformer pre, statement s)
{
  add_reference_information(pre, s, false);
}

void transformer_add_reference_information(transformer tf, statement s)
{
  add_reference_information(tf, s, true);
}

// FI: these constants are not (yet) defined in ri-util-local.h
#define unsigned_char          11
#define unsigned_short_int     12
#define unsigned_int           14
#define unsigned_long_int      16
#define unsigned_long_long_int 18

/* Add some of the constraints linked to the type of a variable */
static void add_type_information(transformer tf)
{
  Psysteme ps = predicate_system(transformer_relation(tf));
  Pbase b = vect_copy(sc_base(ps)); // FI: the basis may evolve when
                                    // new constraints are added
  Pbase cb = BASE_NULLE;

  for(cb=b; !BASE_NULLE_P(cb); cb = vecteur_succ(cb)) {
    Variable v = vecteur_var(cb);
    //entity ev = (entity) v;
    if(v!=TCST && !local_old_value_entity_p(v)) { // FI: TCST check should be useless in a basis
      entity e = value_to_variable(v);
      type t = ultimate_type(entity_type(e));
      if(unsigned_type_p(t)) {
	basic tb = variable_basic(type_variable(t));
	int s = basic_int(tb);
	// FI: the lower bound, lb, could also be defined, e.g. for "char"
	long long int ub = -1/*, lb = 0*/;
	long long int period = 0;
	switch(s) {
	case unsigned_char:
	  ub = 256-1, period = 256;
	  break;
	case unsigned_short_int:
	  ub = 256*256-1, period = 256*256;
	  break;
	case unsigned_int:
	  // We go straight to the overflows!
	  // ub = 256L*256L*256L*256L-1;
	  ub = 0, period = 256L*256L*256L*256L;
	  break;
	case unsigned_long_int:
	  // We go straight to the overflows!
	  // ub = 256L*256L*256L*256L-1;
	  ub = 0, period = 256L*256L*256L*256L;
	  break;
	case unsigned_long_long_int:
	  // We go straight to the overflows!
	  //ub = 256*256*256*256*256*256*256*256-1;
	  //ub = 0; // FI: too dangerous
	  break;
	default:
	  ub=-1; // do nothing
	  break;
	}
	if(ub>=0) { 
	  /* We assume exists lambda s.t. e = ep + period*lambda */
	  entity ep = make_local_temporary_value_entity(t);
	  entity lambda = make_local_temporary_integer_value_entity();
	  tf = transformer_add_inequality_with_integer_constraint(tf, ep, 0, false);
	  if(ub>0) {
	    // Without this dangerous upperbound, we lose information
	    // but avoid giving wrong results.
	    // FI: The upperbound is dangerous because linear does not
	    // expect such large constants, especially my bounded normalization
	    // that should be itself bounded! See
	    // check_coefficient_reduction() and the stack allocation
	    // of arrays a, b and c!
	    tf = transformer_add_inequality_with_integer_constraint(tf, ep, ub, true);
	  }
	  Pvecteur eq =
	    vect_make(vect_new(e, VALUE_MONE),
		      ep, VALUE_ONE,
		      lambda, (Value) period, TCST, VALUE_ZERO, NULL);
	  tf = transformer_equality_add(tf, eq);
	  /* Compute the value of lambda */
	  Value pmin, pmax;
	  if(precondition_minmax_of_value(lambda, tf, (intptr_t*) &pmin, (intptr_t*) &pmax)) {
	    if(pmin==pmax) 
	      tf = transformer_add_equality_with_integer_constant(tf, lambda, pmin);
	  }
	  /* Now we must get rid of lambda and e and possibly e's old value */
	  list p =
	    CONS(ENTITY, e, CONS(ENTITY, lambda, NIL));
	  pips_assert("ep is in tf basis",
		      base_contains_variable_p(sc_base(ps), ep));
	  pips_assert("lambda is in tf basis",
		      base_contains_variable_p(sc_base(ps), lambda));
	  if(entity_is_argument_p(e, transformer_arguments(tf))) {
	    entity old_e = entity_to_old_value(e);
	    p = CONS(ENTITY, old_e, p);
	    pips_assert("old_e is in tf basis",
			base_contains_variable_p(sc_base(ps), ep));
	  }
	  tf = safe_transformer_projection(tf, p);
	  gen_free_list(p);
	  /* Now we must substitute ep by e */
	  tf = transformer_value_substitute(tf, ep, e);
	}
      }
    }
  }
  vect_rm(b);
}

void precondition_add_type_information(transformer pre)
{
  add_type_information(pre);
}

void transformer_add_type_information(transformer tf)
{
  add_type_information(tf);
}

/* Refine the precondition pre of s using side effects and compute its
   postcondition post. Postcondition post is returned. */
transformer statement_to_postcondition(
    transformer pre, /* postcondition of predecessor */
    statement s)
{
    transformer post = transformer_undefined;
    instruction i = statement_instruction(s);
    /* FI: if the statement s is a loop, the transformer tf is not the
       statement transformer but the transformer T* which maps the
       precondition pre onto the loop body precondition. The real
       statement transformer is obtained by executing the loop till
       it is exited. See complete_any_loop_transformer() */
    transformer tf = load_statement_transformer(s);

    /* ACHTUNG! "pre" is likely to be misused! FI, Sept. 3, 1990 */

    pips_debug(1,"begin\n");

    pips_assert("The statement precondition is defined",
		pre != transformer_undefined);

    ifdebug(1) {
	int so = statement_ordering(s);
	(void) fprintf(stderr, "statement %03td (%d,%d), precondition %p:\n",
		       statement_number(s), ORDERING_NUMBER(so),
		       ORDERING_STATEMENT(so), pre);
	(void) print_transformer(pre) ;
    }

    pips_assert("The statement transformer is defined",
		tf != transformer_undefined);
    ifdebug(1) {
	int so = statement_ordering(s);
	(void) fprintf(stderr, "statement %03td (%d,%d), transformer %p:\n",
		       statement_number(s), ORDERING_NUMBER(so),
		       ORDERING_STATEMENT(so), tf);
	(void) print_transformer(tf) ;
    }

    if (!statement_reachable_p(s))
    {
	/* FC: if the code is not reachable (thanks to STOP or GOTO), which
	 * is a structural information, the precondition is just empty.
	 */
      /* Psysteme s = predicate_system(transformer_relation(pre)); */
      pre = empty_transformer(pre);
    }

    if (load_statement_precondition(s) == transformer_undefined) {
	/* keep only global initial scalar integer values;
	   else, you might end up giving the same xxx#old name to
	   two different local values */
	list non_initial_values =
	    arguments_difference(transformer_arguments(pre),
				 get_module_global_arguments());
	list dl = declaration_statement_p(s) ? statement_declarations(s) : NIL;

	/* FI: OK, to be fixed when the declaration representation is
	   frozen. */
	if(!ENDP(statement_declarations(s)) && !statement_block_p(s)
	   && !declaration_statement_p(s)) {
	  // FI: Just to gain some time before dealing with controlizer and declarations updates
	  //pips_internal_error("Statement %p carries declarations");
	  pips_user_warning("Statement %p with instruction carries declarations\n",
			    instruction_identification(statement_instruction(s)));
	}

	MAPL(cv,
	 {
	   entity v = ENTITY(CAR(cv));
	   ENTITY_(CAR(cv)) = entity_to_old_value(v);
	 },
	     non_initial_values);

	/* add array references information */
	if(get_bool_property("SEMANTICS_TRUST_ARRAY_REFERENCES")) {
	    precondition_add_reference_information(pre, s);
	}

	/* add type information */
	if(get_bool_property("SEMANTICS_USE_TYPE_INFORMATION")) {
	  transformer_add_type_information(pre);
	}

	/* Add information from declarations when useful */
	if(declaration_statement_p(s) && !ENDP(dl)) {
	  /* FI: it might be better to have another function,
	     declarations_to_postconditions(), which might be
	     slightly more accurate? Note that
	     declarations_to_transformer() does compute the
	     postcondition, but free it before returning the transformer */
	  transformer dt = declarations_to_transformer(dl, pre);
	  transformer dpre = transformer_apply(dt, pre);

	  //post = instruction_to_postcondition(dpre, i, tf);
	  //free_transformer(dpre);
	  // FI: Let's assume that declaration statement do not
	  //require further analysis
	  post = dpre;
	}
	else {
	  post = instruction_to_postcondition(pre, i, tf);
	}

	/* Remove information when leaving a block */
	if(statement_block_p(s) && !ENDP(statement_declarations(s))) {
	  list vl = variables_to_values(statement_declarations(s));

	  if(!ENDP(vl))
	    post = safe_transformer_projection(post, vl);
	}
	else if(statement_loop_p(s)
		&& !get_bool_property("SEMANTICS_KEEP_DO_LOOP_EXIT_CONDITION")) {
	  loop l = statement_loop(s);
	  entity i = loop_index(l);
	  list vl = variable_to_values(i);
	  post = safe_transformer_projection(post, vl);
	}

	/* add equivalence equalities */
	pre = tf_equivalence_equalities_add(pre);

	/* eliminate redundancy, rational redundancy but not integer redundancy.  */

	/* FI: nice... but time consuming! */
	/* Version 3 is OK. Equations are over-used and make
	 * inequalities uselessly conplex
	 */
	/* pre = transformer_normalize(pre, 3); */

	/* pre = transformer_normalize(pre, 6); */
	/* pre = transformer_normalize(pre, 7); */

	/* The double normalization could be avoided with a non-heuristics
           approach. For ocean, its overhead is 34s out of 782.97 to give
           816.62s: 5 %. The double normalization is also useful for some
           exit conditions of WHILE loops (w05, w06, w07). It is not
           powerful enough for preconditions containing equations with
           three or more variables such as fraer01,...*/

	if(!transformer_consistency_p(pre)) {
	  ;
	}
	/* BC: pre = transformer_normalize(pre, 4); */
	/* FI->BC: why keep a first normalization before the next
	   one? FI: Because a level 2 normalization does things that
	   a level 4 does not perform! Although level 2 is much
	   faster... */
	pre = transformer_normalize(pre, 2);

	if(!transformer_consistency_p(pre)) {
	  ;
	}
	/* pre = transformer_normalize(pre, 2); */
	if(get_int_property("SEMANTICS_NORMALIZATION_LEVEL_BEFORE_STORAGE")
	   == 4)
	  // FI HardwareAccelerator/freia_52: this level does not
	  // handle the signs properly for simple equations like -i==0
	  // and it does not sort constraints lexicographically!
	  pre = transformer_normalize(pre, 4);
	else
	  pre = transformer_normalize(pre, 2);

	if(!transformer_consistency_p(pre)) {
	    int so = statement_ordering(s);
	    fprintf(stderr, "statement %03td (%d,%d), precondition %p end:\n",
			   statement_number(s), ORDERING_NUMBER(so),
			   ORDERING_STATEMENT(so), pre);
	    print_transformer(pre);
	    pips_internal_error("Non-consistent precondition after update");
	}

	/* Do not keep too many initial variables in the
	 * preconditions: not so smart? invariance01.c: information is
	 * lost... Since C passes values, it is usually useless to
	 * keep track of the initial values of arguments because
	 * programmers usually do not modify them. However, when they
	 * do modify the formal parameter, information is lost.
	 *
	 * See character01.c, but other counter examples above about
	 * non_initial_values.
	 *
	 * FI: redundancy possibly added. See asopt02. Maybe this
	 * should be moved up before the normalization step.
	 */
	if(get_bool_property("SEMANTICS_FILTER_INITIAL_VALUES")) {
	  pre = transformer_filter(pre, non_initial_values);
	  pre = transformer_normalize(pre, 2);
	}

	/* store the precondition in the ri */
	store_statement_precondition(s, pre);

	gen_free_list(non_initial_values);
    }
    else {
	pips_debug(8,"precondition already available\n");
	/* pre = statement_precondition(s); */
	(void) print_transformer(pre);
	pips_internal_error("precondition already computed");
    }

    /* post = instruction_to_postcondition(pre, i, tf); */

    ifdebug(1) {
	int so = statement_ordering(s);
	fprintf(stderr, "statement %03td (%d,%d), precondition %p end:\n",
		statement_number(s), ORDERING_NUMBER(so),
		ORDERING_STATEMENT(so), load_statement_precondition(s));
	print_transformer(load_statement_precondition(s)) ;
    }

    ifdebug(1) {
	int so = statement_ordering(s);
	fprintf(stderr, "statement %03td (%d,%d), postcondition %p:\n",
		statement_number(s), ORDERING_NUMBER(so),
		ORDERING_STATEMENT(so), post);
	print_transformer(post) ;
    }

    pips_assert("no sharing",post!=pre);

    pips_debug(1, "end\n");

    return post;
}

/* This function is mostly copied from
   declarations_to_transformer(). It is used to recompute
   intermediate preconditions and to process the initialization
   expressions with the proper precondition. For instance, in:

   int i = 10, j = i+1, a[i], k = foo(i);

   you need to collect information about i's value to initialize j,
   dimension a and compute the summary precondition of function foo().

   We assume that the precondition does not change within the
   expression as in:

   int k = i++ + foo(i);

   I do not remember if the standard prohibits this or not, but it may
   well forbid such expressions or state that the result is undefined.

   But you can also have:

   int a[i++][foo(i)];

   or

   int a[i++][j=foo(i)];

   and the intermediate steps are overlooked by
   declaration_to_transformer() but can be checked with a proper
   process_dimensions() function.

   This function can be called from ri_to_preconditions.c to propagate
   preconditions or from interprocedural.c to compute summary
   preconditions. In the second case, the necessary side effects are
   provided by the two functional parameters.
*/
transformer propagate_preconditions_in_declarations
(list dl,
 transformer pre,
 void (*process_initial_expression)(expression, transformer)
 // FI: ongoing implementation
 //, transformer (*process_dimensions)(entity, transformer),
)
{
  //entity v = entity_undefined;
  //transformer btf = transformer_undefined;
  //transformer stf = transformer_undefined;
  transformer post = transformer_undefined;
  list l = dl;

  pips_debug(8,"begin\n");

  if(ENDP(l))
    post = copy_transformer(pre);
  else {
    entity v = ENTITY(CAR(l));
    expression ie = variable_initial_expression(v);
    transformer stf = declaration_to_transformer(v, pre);
    // FI: ongoing implementation
    //transformer stf = (*process_dimensions)(v, pre);
    transformer btf = transformer_dup(stf);
    transformer next_pre = transformer_undefined;

    if(!expression_undefined_p(ie)) {
      (*process_initial_expression)(ie, pre);
      free_expression(ie);
    }

    post = transformer_safe_apply(stf, pre);
/*     post = transformer_safe_normalize(post, 4); */
    post = transformer_safe_normalize(post, 2);

    for (POP(l) ; !ENDP(l); POP(l)) {
      v = ENTITY(CAR(l));
      ie = variable_initial_expression(v);
      if(!expression_undefined_p(ie)) {
	(*process_initial_expression)(ie, post);
	free_expression(ie);
      }

      if(!transformer_undefined_p(next_pre))
	free_transformer(next_pre);
      next_pre = post;
      stf = declaration_to_transformer(v, next_pre);
      post = transformer_safe_apply(stf, next_pre);
/*       post = transformer_safe_normalize(post, 4); */
      post = transformer_safe_normalize(post, 2);
      btf = transformer_combine(btf, stf);
/*       btf = transformer_normalize(btf, 4); */
      btf = transformer_normalize(btf, 2);

      ifdebug(1)
	pips_assert("btf is a consistent transformer",
		    transformer_consistency_p(btf));
	pips_assert("post is a consistent transformer if pre is defined",
		    transformer_undefined_p(pre)
		    || transformer_consistency_p(post));
    }
    free_transformer(btf);
  }

  pips_debug(8, "end\n");
  return post;
}
