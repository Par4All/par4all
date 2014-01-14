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
/* Postpone convex hulls by keeping transformer lists instead
 *
 * This development was prompted by the last example found in the
 * paper by Schrammel and Jeannet at NSAD 2010. See test cases
 * schrammel04, 05 and 06. The minimal goal is to avoid the indentity
 * transformer when performing the convex hull of several
 * transformers.
 *
 * This could also be useful to automatize the handling of tests
 * within a loop using the technique presented at NSAD 2010 by Ancourt
 * & al. The control structure
 *
 * "while(c) if(t) T; else F;"
 *
 * is replaced by
 *
 * "while(c) {while(c&&t) T; while(c&& !t) F;}".
 *
 * This replacement could be performed on the equations instead of
 * requiring a program transformation.
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
 /* semantical analysis
  *
  * phasis 3: compute transformer lists from statements and statement
  * transformers
  *
  * For refined precondition analysis. Keep track of all control paths
  * within sequences
  *
  * Francois Irigoin, September 2010
  */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
#include "database.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

//#include "vecteur.h"
//#include "contrainte.h"
//#include "ray_dte.h"
//#include "sommet.h"
//#include "sg.h"
//#include "polyedre.h"

#include "transformer.h"

#include "semantics.h"




/* Note: initializations of static variables are not used as
   transformers but to initialize the program precondition. */
/* It is not assumed that entity_has_values_p(v)==true */
/* A write effect on the declared variable is assumed as required by
   Beatrice Creusillet for region computation. */
list declaration_to_transformer_list(entity v, transformer pre)
{
  list tfl = NIL;
  transformer tf = transformer_undefined;

  pips_internal_error("Not implemented yet.");

  pips_debug(8, "Transformer for declaration of \"%s\"\n", entity_name(v));

  if(false && !entity_has_values_p(v)) {
    /* FI: the initialization expression might have relevant
       side-effects? This could ba handled by generalizing
       variable_to_initial_expression() and by returning
       expression_undefined incase of failure instead of aborting. */
    tf = transformer_identity();
  }
  else if(variable_static_p(v)) {
    if(get_bool_property("SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT"))
      tf = transformer_range(pre);
    else
      tf = transformer_identity();
  }
  else {
    /* Use the dimension expressions and the initial value */
    transformer dt = dimensions_to_transformer(v, pre);
    transformer npre = transformer_apply(dt, pre);
    expression ie = variable_initial_expression(v);
    transformer itf = safe_assigned_expression_to_transformer(v, ie, npre);
    tf = dt;
    tf = transformer_combine(tf, itf);
    free_expression(ie);
  }

  pips_assert("tf is defined", !transformer_undefined_p(tf));

  /* FI: I preserve the code below in case I have problems with
     integer typing in the future*/
  /*
  if(entity_has_values_p(v) && !variable_static_p(v)) {
    value vv = entity_initial(v);
    if(value_unknown_p(vv)) {
      tf = transformer_identity();
    }
    else if (value_symbolic_p(vv)) {
      pips_internal_error("Unexpected value tag: symbolic");
    }
    else if (value_constant_p(vv)) {
      tf = transformer_identity();
      //  SG: quickly fix this, unsure about the meaning
      //pips_internal_error("Unexpected value tag: constant");
    }
    else if (value_expression_p(vv)) {
      expression e = value_expression(vv);
      basic eb = basic_of_expression(e);
      type vt = ultimate_type(entity_type(v));
      basic vb = variable_basic(type_variable(vt));

      if(same_basic_p(eb, vb)) {
	tf = safe_any_expression_to_transformer(v, e, pre, false);
	tf = transformer_temporary_value_projection(tf);
      }
      else {
	if(basic_int_p(eb) && basic_int_p(vb)) {
	  int i1 = basic_int(eb);
	  int i2 = basic_int(vb);
	  if(ABS(i1-i2)==10) {
	    tf = safe_any_expression_to_transformer(v, e, pre, false);
	    tf = transformer_temporary_value_projection(tf);
	    pips_user_warning("Possible conversion issue between signed and"
			      " unsigned int\n");
	  }
	  else {
	    tf = safe_any_expression_to_transformer(v, e, pre, false);
	    tf = transformer_temporary_value_projection(tf);
	    pips_user_warning("Possible conversion issue between diffent kinds"
			      " of  ints and/or char (%dd and %d)\n", i1, i2);
	  }
	}
	else {
	  //list el = expression_to_proper_effects(e);
          list el = expression_to_proper_constant_path_effects(e);

	  pips_user_warning("Type mismatch detected in initialization expression."
			    " May be due to overloading and/or implicit confusion"
			    " between logical and integer in C\n");
	  tf = effects_to_transformer(el);
	}
      }
    }
    else {
      pips_internal_error("Unexpected value tag");
    }
  }
  else {
    tf = transformer_identity();
  }
  */

  ifdebug(8) {
    pips_debug(8, "Ends with:\n");
    (void) print_transformers(tfl);
  }

  return tfl;
}

/* For C declarations. Very close to a block_to_transformer() as
   declarations can be seen as a sequence of assignments.

   Note: initialization of static variables are not taken into
   account. They must be used for summary preconditions.
 */
list declarations_to_transformer_list(list dl, transformer pre)
{
  list tfl = list_undefined;
  entity v = entity_undefined;
  transformer btf = transformer_undefined;
  transformer stf = transformer_undefined;
  transformer post = transformer_undefined;
  transformer next_pre = transformer_undefined;
  list l = dl;

  // May never be useful
  pips_internal_error("Not implemented yet.");

  pips_debug(8,"begin\n");

  if(ENDP(l))
    btf = transformer_identity();
  else {
    v = ENTITY(CAR(l));
    stf = declaration_to_transformer(v, pre);
    post = transformer_safe_apply(stf, pre);
/*     post = transformer_safe_normalize(post, 4); */
    post = transformer_safe_normalize(post, 2);
    btf = transformer_dup(stf);
    for (POP(l) ; !ENDP(l); POP(l)) {
      v = ENTITY(CAR(l));
      if(!transformer_undefined_p(next_pre))
	free_transformer(next_pre);
      next_pre = transformer_range(post);
      stf = declaration_to_transformer(v, next_pre);
      post = transformer_safe_apply(stf, next_pre);
      free_transformer(next_pre);
      next_pre = transformer_undefined; // FI: works even without this...
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
    free_transformer(post);
  }

  pips_debug(8, "end\n");
  return tfl;
}

/* Generate all possible linear control paths
 *
 * We start with a unique precondition.
 *
 * For each statement, we start with a transformer list btfl_i-1 and
 * with a postcondition list postl_i-1
 *
 * To each statement s_i, we associate a transformer list stfl_i.
 *
 * For each transformer stf_j in stfl_i, for each transformer btf_k
 * and postcondition post_k in btfl_i-1 and postl_i-1, we compute a
 * new transformer nbtf_j,k=stf_j o btf_k and a new postcondition
 * npost_j,k=stf_j(post_k).
 *
 * btfl_i = {nbtf_j,k} and postl_i = {npost_j,k}
 *
 * The first statement must be handled in a specific way or btfl_0 be
 * initialized to a singleton list with the identity transformer and
 * postl_0 be initialized as a singleton list with a single element
 * {pre}.
 *
 * The number of transformers returned increase exponentially with the
 * block size n. Assume that each statement returns two transformers:
 * 2**n transformers are returned by this function.
 *
 * More thinking about memory management probably useful. Also
 * postconditions are automatically available in the transformers. The
 * simultaneous computation of postconditions seem redundant.
 */
static list block_to_transformer_list(list b, transformer pre)
{
  list btfl = NIL; //block transformer list
  transformer btf = transformer_undefined;
  //transformer stf = transformer_undefined;
  transformer post = transformer_undefined;
  //transformer next_pre = transformer_undefined;
  list l = b;

  pips_debug(8,"begin\n");

  if(ENDP(l)) {
    //btf = transformer_identity();
    // pre must be a range, hence a subset of identity
    btf = copy_transformer(pre);
    btfl = CONS(TRANSFORMER, btf, NIL);
  }
  else {
    /* Handle the first statement */
    statement s = STATEMENT(CAR(l));
    list postl = NIL;
    btfl = statement_to_transformer_list(s, pre);
    postl = transformers_safe_apply(btfl, pre);
    postl = transformers_range(postl);
    postl = transformers_safe_normalize(postl, 2);
    // Useless if no storage occurs; stfl should replace btfl above
    //btfl = gen_full_sequence_copy(stfl);
    pips_assert("One postcondition for each transformer",
		gen_length(btfl) == gen_length(postl));

    /* For the next statements */
    for (POP(l) ; !ENDP(l); POP(l)) {
      list precl = postl;
      list npostl = NIL;
      list nbtfl = NIL;
      s = STATEMENT(CAR(l));
      FOREACH(TRANSFORMER, btf, btfl) {
	// Get the corresponding postcondition, now a precondition
	transformer next_pre = TRANSFORMER(CAR(precl));
	list stfl = statement_to_transformer_list(s, next_pre);
	FOREACH(TRANSFORMER, stf, stfl) {
	  post = transformer_safe_apply(stf, next_pre);
	  post = transformer_safe_normalize(post, 2);
	  transformer post_r = transformer_range(post);
	  free_transformer(post);
	  transformer nbtf = copy_transformer(btf); // preserve outer
						    // loop variable
	  nbtf = transformer_combine(nbtf, stf);
	  nbtf = transformer_normalize(nbtf, 2);
	  ifdebug(1) {
	    pips_assert("btf is a consistent transformer",
			transformer_consistency_p(btf));
	    pips_assert("post_r is a consistent transformer if pre is defined",
			transformer_undefined_p(pre)
			|| transformer_consistency_p(post_r));
	  }
	  if(!transformer_empty_p(nbtf)) {
	    nbtfl = gen_nconc(nbtfl, CONS(TRANSFORMER, nbtf, NIL));
	    npostl = gen_nconc(npostl, CONS(TRANSFORMER, post_r, NIL));
	  }
	}
	gen_full_free_list(stfl);
	POP(precl);
      }
      /* Now, switch from btfl and postl to nbtfl and npostl */
      // The transformers are updated and moved into the new list nbtfl
      //gen_full_free_list(btfl);
      //gen_full_free_list(postl);
      btfl = nbtfl;
      postl = npostl;
    }
    /* Postconditions are discarded */
    gen_full_free_list(postl);

    /* Clean up the transformer list: reduce identity transformers to
       one if any and move it ahead of the list */

    btfl = clean_up_transformer_list(btfl);
    btfl = transformers_safe_normalize(btfl, 2);
  }

  pips_debug(8, "ends with %zd transformers\n", gen_length(btfl));
  ifdebug(8)
    print_transformers(btfl);
  return btfl;
}


static list test_to_transformer_list(test t,
				     transformer pre,
				     list ef) /* effects of t */
{
  list tl = NIL;
  statement st = test_true(t);
  statement sf = test_false(t);
  transformer tf;

  /* EXPRESSION_TO_TRANSFORMER() SHOULD BE USED MORE EFFECTIVELY */

  pips_debug(8,"begin\n");

  /* Clearly, this must be true for transformer list to be
     computed. However, if test_to_transformer_list() is more general
     than test_to_transformer(), the later should disappear and be
     merged here, with a convex hull to reduce the list */
  if(pips_flag_p(SEMANTICS_FLOW_SENSITIVE)) {
    expression e = test_condition(t);
    /* Ideally, they should be initialized with the current best
       precondition, intraprocedural if nothing else better is
       available. This function's profile as well as most function
       profiles in ri_to_transformers should be modifed. */
    transformer tftwc = transformer_undefined_p(pre)?
      transformer_identity() :
      precondition_to_abstract_store(pre);
    transformer context = transformer_dup(tftwc);
    transformer tffwc = transformer_dup(tftwc);
    list post_tftwc = NIL;
    list post_tffwc = NIL;
    list ta = NIL;
    list fa = NIL;
    /* True condition transformer, TCT */
    transformer tct = condition_to_transformer(e, context, true);
    /* False condition transformer, FCT */
    transformer fct = condition_to_transformer(e, context, false);

    tftwc = transformer_apply(tct, context);

    ifdebug(8) {
      pips_debug(8, "tftwc before transformer_temporary_value_projection %p:\n", tftwc);
      (void) print_transformer(tftwc);
    }
    tftwc = transformer_temporary_value_projection(tftwc);
    reset_temporary_value_counter();
    ifdebug(8) {
      pips_debug(8, "tftwc before transformer_apply %p:\n", tftwc);
      (void) print_transformer(tftwc);
    }
    list ttl = statement_to_transformer_list(st, tftwc);
    // FI: why should we compute postconditions for ttl?!?
    post_tftwc = transformer_apply_map(ttl, tftwc);
    ifdebug(8) {
      pips_debug(8, "tftwc after transformer_apply %p:\n", tftwc);
      (void) print_transformer(tftwc);
      pips_debug(8, "post_tftwc after transformer_apply %p:\n", post_tftwc);
      (void) print_transformers(post_tftwc);
    }

    /* tffwc = precondition_add_condition_information(tffwc, e, context, false); */
    tffwc = transformer_apply(fct, context);
    tffwc = transformer_temporary_value_projection(tffwc);
    reset_temporary_value_counter();
    list ftl = statement_to_transformer_list(sf, tffwc);
    // FI: why should we compute postconditions for ftl?!?
    post_tffwc = transformer_apply_map(ftl, tffwc);

    ifdebug(8) {
      pips_debug(8, "post_tftwc before transformer_convex_hull %p:\n", post_tftwc);
      (void) print_transformers(post_tftwc);
      pips_debug(8, "post_tffwc after transformer_apply %p:\n", post_tffwc);
      (void) print_transformers(post_tffwc);
    }

    //tf = transformer_convex_hull(post_tftwc, post_tffwc);
    //tl = merge_transformer_lists(ttl, ftl); condition and its side
    //effects are lost
    tl = merge_transformer_lists(post_tftwc, post_tffwc);
    transformer_free(context);
    transformer_free(tftwc);
    transformer_free(tffwc);
    //transformer_free(post_tftwc);
    //transformer_free(post_tffwc);
    free_arguments(ta);
    free_arguments(fa);
  }
  else {
    transformer id = transformer_identity();
    (void) statement_to_transformer(st, id);
    (void) statement_to_transformer(sf, id);
    tf = effects_to_transformer(ef);
    tl = CONS(TRANSFORMER, tf, NIL);
    free_transformer(id);
  }

  pips_debug(8,"end\n");
  return tl;
}

/* because of the conditional and the comma C intrinsics at least */
list  intrinsic_to_transformer_list(entity e,
				    list pc,
				    transformer pre,
				    list ef) /* effects of intrinsic call */
{
  list tl = NIL;
  transformer tf = transformer_undefined;

  pips_internal_error("Not implemented yet.");

  pips_debug(8, "begin\n");

  if(ENTITY_ASSIGN_P(e)) {
    tf = any_assign_to_transformer(pc, ef, pre);
  }
 else if(ENTITY_PLUS_UPDATE_P(e) || ENTITY_MINUS_UPDATE_P(e)
	 || ENTITY_MULTIPLY_UPDATE_P(e) || ENTITY_DIVIDE_UPDATE_P(e)
	 || ENTITY_MODULO_UPDATE_P(e) || ENTITY_LEFT_SHIFT_UPDATE_P(e)
	 || ENTITY_RIGHT_SHIFT_UPDATE_P(e) || ENTITY_BITWISE_AND_UPDATE_P(e)
	 || ENTITY_BITWISE_XOR_UPDATE_P(e) || ENTITY_BITWISE_OR_UPDATE_P(e)) {
    //tf = update_addition_operation_to_transformer(pc, ef, pre);
   tf = any_update_to_transformer(e, pc, ef, pre);
  }
 else if(ENTITY_POST_INCREMENT_P(e) || ENTITY_POST_DECREMENT_P(e)
	 || ENTITY_PRE_INCREMENT_P(e) || ENTITY_PRE_DECREMENT_P(e)) {
   tf = any_basic_update_to_transformer(e, pc, ef, pre);
  }
 else if(ENTITY_C_RETURN_P(e)) {
   tf = c_return_to_transformer(e, pc, ef, pre);
  }
  else if(ENTITY_STOP_P(e)||ENTITY_ABORT_SYSTEM_P(e)||ENTITY_EXIT_SYSTEM_P(e)
	  || ENTITY_ASSERT_FAIL_SYSTEM_P(e))
    tf = transformer_empty();
  else if(ENTITY_COMMA_P(e)) {
    tf = expressions_to_transformer(pc, pre);
  }
  else if(ENTITY_CONDITIONAL_P(e)) {
    /* FI: this may happen, for instance with the macro definition of
       assert() or because the programmer writes "i>1? (i = 2): (i =
       3);" instead of "i = i>1? 2 : 3;" */
    expression cond = EXPRESSION(CAR(pc));
    expression e1 = EXPRESSION(CAR(CDR(pc)));
    expression e2 = EXPRESSION(CAR(CDR(CDR(pc))));
    tf = conditional_to_transformer(cond, e1, e2, pre, ef);
  }
  else if(ENTITY_ASSERT_SYSTEM_P(e)) {
    /* FI: the condition should be evaluated and considered true on
       exit, but this is sometimes captured by a macro definition and the code
       below is then useless */
    expression cond = EXPRESSION(CAR(pc));
    tf = condition_to_transformer(cond, pre, true);
  }
  else
    tf = effects_to_transformer(ef);

  pips_debug(8, "end\n");

  tl=CONS(TRANSFORMER,tf,NIL);
  return tl;
}


/* transformer assigned_expression_to_transformer(entity e, expression
 * expr, list ef): returns a transformer abstracting the effect of
 * assignment "e = expr" when possible, transformer_undefined otherwise.
 *
 * Note: it might be better to distinguish further between e and expr
 * and to return a transformer stating that e is modified when e
 * is accepted for semantics analysis.
 *
 */
list assigned_expression_to_transformer_list(entity v,
					     expression expr,
					     transformer pre)
{
  list tl = NIL;
  transformer tf = transformer_undefined;

  pips_internal_error("Not implemented yet.");

  pips_debug(8, "begin\n");

  if(entity_has_values_p(v)) {
    entity v_new = entity_to_new_value(v);
    entity v_old = entity_to_old_value(v);
    entity tmp = make_local_temporary_value_entity(entity_type(v));
    //list tf_args = CONS(ENTITY, v, NIL);

    tf = any_expression_to_transformer(tmp, expr, pre, true);
    // The assignment may be part of a more complex expression
    // This should be guarded by "is_internal==FALSE" if is_internal were an argument
    //reset_temporary_value_counter();
    if(!transformer_undefined_p(tf)) {
      /* if the assigned variable is also assigned by the expression
       as in i = (i = 2) + 1, transformer_value_substitute() cannot be
       used right away. The previous store must be projected out. */
      if(entity_is_argument_p(v, transformer_arguments(tf))) {
	/* v must be assigned */
	transformer teq = simple_equality_to_transformer(v, tmp, true);
	tf = transformer_combine(tf, teq);
	free_transformer(teq);

      }
      else { /* subcase of previous aternative */
	tf = transformer_value_substitute(tf, v_new, v_old);
	tf = transformer_value_substitute(tf, tmp, v_new);
	// v cannot be a temporary variable
	//transformer_arguments(tf) =
	//arguments_add_entity(transformer_arguments(tf), v);
	tf = transformer_add_value_update(tf, v);
      }
      tf = transformer_temporary_value_projection(tf);
    }
  }
  else {
    /* vect_rm(ve); */
    tf = transformer_undefined;
  }

  pips_debug(8, "end with tf=%p\n", tl);

  return tl;
}

/* Always returns a fully defined transformer.
 *
 * FI: The property to compute transformers in context is not taken
 * into account to add information from pre within tf. pre is used to
 * evaluate expr, but is not made part of tf.
 */
list safe_assigned_expression_to_transformer_list(entity v,
						    expression expr,
						    transformer pre)
{
  list tl = NIL;
  transformer tf = transformer_undefined;

  pips_internal_error("Not implemented yet.");

  if(expression_undefined_p(expr)) {
    ; // That is fixed below
  }
  else
    tf = assigned_expression_to_transformer(v, expr, pre);

  if(transformer_undefined_p(tf)) {
    if(get_bool_property("SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT"))
      tf = transformer_range(pre);
    else
      tf = transformer_identity();

    if(entity_has_values_p(v)) {
      tf = transformer_add_modified_variable_entity(tf, v);
    }
  }

  //pips_assert("tf is defined", !transformer_undefined_p(tf));
  pips_assert("tl is consistent", transformers_consistency_p(tl));

  return tl;
}


/* This function never returns an undefined transformer. It is used
   for an assignment statement, not for an assignment operation. */
list integer_assign_to_transformer_list(expression lhs,
					expression rhs,
					transformer pre,
					list ef) /* effects of assign */
{
  /* algorithm: if lhs and rhs are linear expressions on scalar integer
     variables, build the corresponding equation; else, use effects ef

     should be extended to cope with constant integer division as in
     N2 = N/2
     because it is used in real program; inequalities should be
     generated in that case 2*N2 <= N <= 2*N2+1

     same remark for MOD operator

     implementation: part of this function should be moved into
     transformer.c
  */

  list tl = NIL;
  transformer tf = transformer_undefined;
  normalized n = NORMALIZE_EXPRESSION(lhs);

  pips_internal_error("Not implemented yet.");

  pips_debug(8,"begin\n");

  if(normalized_linear_p(n)) {
    Pvecteur vlhs = (Pvecteur) normalized_linear(n);
    entity e = (entity) vecteur_var(vlhs);

    if(entity_has_values_p(e) /* && integer_scalar_entity_p(e) */) {
      /* FI: the initial version was conservative because
       * only affine scalar integer assignments were processed
       * precisely. But non-affine operators and calls to user defined
       * functions can also bring some information as soon as
       * *some* integer read or write effect exists
       */
      /* check that *all* read effects are on integer scalar entities */
      /*
	if(integer_scalar_read_effects_p(ef)) {
	tf = assigned_expres`sion_to_transformer(e, rhs, ef);
	}
      */
      /* Check that *some* read or write effects are on integer
       * scalar entities. This is almost always true... Let's hope
       * assigned_expression_to_transformer() returns quickly for array
       * expressions used to initialize a scalar integer entity.
       */
      if(some_integer_scalar_read_or_write_effects_p(ef)) {
	tf = assigned_expression_to_transformer(e, rhs, pre);
      }
    }
  }
  /* if some condition was not met and transformer derivation failed */
  if(tf==transformer_undefined)
    tf = effects_to_transformer(ef);

  pips_debug(6,"return tl=%lx\n", (unsigned long)tl);
  ifdebug(6) (void) print_transformers(tl);
  pips_debug(8,"end\n");
  return tl;
}

list any_scalar_assign_to_transformer_list(entity v,
					   expression rhs,
					   list ef, /* effects of assign */
					   transformer pre) /* precondition */
{
  list tl = NIL;
  transformer tf = transformer_undefined;

  pips_internal_error("Not implemented yet.");

  if(entity_has_values_p(v)) {
    entity v_new = entity_to_new_value(v);
    entity v_old = entity_to_old_value(v);
    entity tmp = make_local_temporary_value_entity(ultimate_type(entity_type(v)));

    tf = any_expression_to_transformer(tmp, rhs, pre, true);

    if(!transformer_undefined_p(tf)) {

      pips_debug(9, "A transformer has been obtained:\n");
      ifdebug(9) dump_transformer(tf);

      if(entity_is_argument_p(v, transformer_arguments(tf))) {
	/* Is it standard compliant? The assigned variable is modified by the rhs. */
	transformer teq = simple_equality_to_transformer(v, tmp, true);
	string s = words_to_string(words_syntax(expression_syntax(rhs),NIL));

	pips_user_warning("Variable %s in lhs is uselessly updated by the rhs '%s'\n",
			  entity_local_name(v), s);

	free(s);

	tf = transformer_combine(tf, teq);
	free_transformer(teq);
      }
      else {
	/* Take care of aliasing */
	entity v_repr = value_to_variable(v_new);

	/* tf = transformer_value_substitute(tf, v_new, v_old); */
	tf = transformer_value_substitute(tf, v_new, v_old);

	pips_debug(9,"After substitution v_new=%s -> v_old=%s\n",
	      entity_local_name(v_new), entity_local_name(v_old));

	tf = transformer_value_substitute(tf, tmp, v_new);

	pips_debug(9,"After substitution tmp=%s -> v_new=%s\n",
	      entity_local_name(tmp), entity_local_name(v_new));

	transformer_add_modified_variable(tf, v_repr);
      }
    }
    if(!transformer_undefined_p(tf)) {
      tf = transformer_temporary_value_projection(tf);
      pips_debug(9, "After temporary value projection, tf=%p:\n", tf);
      ifdebug(9) dump_transformer(tf);
    }
    reset_temporary_value_counter();
  }

  if(transformer_undefined_p(tf))
    tf = effects_to_transformer(ef);

  return tl;
}

list any_assign_to_transformer_list(list args, /* arguments for assign */
				    list ef, /* effects of assign */
				    transformer pre) /* precondition */
{
  list tl = NIL;
  transformer tf = transformer_undefined;
  expression lhs = EXPRESSION(CAR(args));
  expression rhs = EXPRESSION(CAR(CDR(args)));
  syntax slhs = expression_syntax(lhs);

  pips_internal_error("Not implemented yet.");

  pips_assert("2 args to assign", CDR(CDR(args))==NIL);

  /* The lhs must be a scalar reference to perform an interesting analysis */
  if(syntax_reference_p(slhs)) {
    reference rlhs = syntax_reference(slhs);
    if(ENDP(reference_indices(rlhs))) {
      entity v = reference_variable(rlhs);
      tf = any_scalar_assign_to_transformer(v, rhs, ef, pre);
    }
  }

  /* if some condition was not met and transformer derivation failed */
  if(tf==transformer_undefined)
    tf = effects_to_transformer(ef);

  pips_debug(6,"return tl=%p\n", tl);
  ifdebug(6) (void) print_transformers(tl);
  pips_debug(8,"end\n");
  return tl;
}

list any_update_to_transformer_list(entity op,
				    list args, /* arguments for update */
				    list ef, /* effects of assign */
				    transformer pre) /* precondition */
{
  list tl = NIL;
  transformer tf = transformer_undefined;
  expression lhs = EXPRESSION(CAR(args));
  expression rhs = EXPRESSION(CAR(CDR(args)));
  syntax slhs = expression_syntax(lhs);

  pips_internal_error("Not implemented yet.");

  pips_assert("2 args for regular update", CDR(CDR(args))==NIL);

  /* The lhs must be a scalar reference to perform an interesting analysis */
  if(syntax_reference_p(slhs)) {
    reference rlhs = syntax_reference(slhs);
    if(ENDP(reference_indices(rlhs))) {
      entity v = reference_variable(rlhs);
      expression ve = entity_to_expression(v);
      entity sop = update_operator_to_regular_operator(op);
      expression n_rhs = MakeBinaryCall(sop, ve, copy_expression(rhs));

      tf = any_scalar_assign_to_transformer(v, n_rhs, ef, pre); 
      free_expression(n_rhs);
    }
  }

  /* if some condition was not met and transformer derivation failed */
  if(tf==transformer_undefined)
    tf = effects_to_transformer(ef);

  pips_debug(6,"return tl=%p\n", tl);
  ifdebug(6) (void) print_transformers(tl);
  pips_debug(8,"end\n");
  return tl;
}

list any_basic_update_to_transformer_list(entity op,
					  list args, /* arguments for update */
					  list ef, /* effects of assign */
					  transformer pre) /* precondition */
{
  list tl = NIL;
  transformer tf = transformer_undefined;
  expression lhs = EXPRESSION(CAR(args));
  syntax slhs = expression_syntax(lhs);

  pips_internal_error("Not implemented yet.");

  pips_assert("1 arg for basic_update", CDR(args)==NIL);

  /* The lhs must be a scalar reference to perform an interesting analysis */
  if(syntax_reference_p(slhs)) {
    reference rlhs = syntax_reference(slhs);
    if(ENDP(reference_indices(rlhs))) {
      entity v = reference_variable(rlhs);
      expression ve = expression_undefined;
      expression n_rhs = expression_undefined;
      entity plus = entity_intrinsic(PLUS_C_OPERATOR_NAME);

      if(ENTITY_POST_INCREMENT_P(op) || ENTITY_PRE_INCREMENT_P(op))
	ve = int_to_expression(1);
      else
	ve = int_to_expression(-1);

      n_rhs = MakeBinaryCall(plus, ve, copy_expression(lhs));

      tf = any_scalar_assign_to_transformer(v, n_rhs, ef, pre);
      free_expression(n_rhs);
    }
  }

  /* if some condition was not met and transformer derivation failed */
  if(tf==transformer_undefined)
    tf = effects_to_transformer(ef);

  pips_debug(6,"return tl=%p\n", tl);
  ifdebug(6) (void) print_transformers(tl);
  pips_debug(8,"end\n");
  return tl;
}

static list instruction_to_transformer_list(instruction i,
					    transformer tf,
					    transformer pre,
					    list e) /* effects associated to instruction i */
{
  list tl = NIL;
  //transformer tf = transformer_undefined;
  test t;
  loop l;
  //call c;
  whileloop wl;
  forloop fl;

  pips_debug(8,"begin\n");

  switch(instruction_tag(i)) {
  case is_instruction_block:
    tl = block_to_transformer_list(instruction_block(i), pre);
    break;
  case is_instruction_test:
    t = instruction_test(i);
    tl = test_to_transformer_list(t, pre, e);
    break;
  case is_instruction_loop:
    l = instruction_loop(i);
    tl = complete_loop_transformer_list(tf, pre, l);
    break;
  case is_instruction_whileloop: {
    wl = instruction_whileloop(i);
    // FI: the complete_xxx functions require a transformer as argument
    tl = complete_whileloop_transformer_list(tf, pre, wl);
    break;
  }
  case is_instruction_forloop:
    fl = instruction_forloop(i);
    tl = complete_forloop_transformer_list(tf, pre, fl);
    break;
  case is_instruction_goto:
    pips_internal_error("unexpected goto in semantics analysis");
    tl = NIL; // never executed
    break;
  case is_instruction_call: {
    /* Nothing fancy yet in spite of the C ? and , operators*/
    call c = instruction_call(i);
    tf = call_to_transformer(c, pre, e);
    if(!transformer_empty_p(tf))
      tl = CONS(TRANSFORMER, tf, NIL);
    // Reuse the existing transformer: tl==NIL has a semantic, not feasible
    // tl = NIL;
  }
    break;
  case is_instruction_unstructured:
    /* Bourdoncle's to be replaced */
    //tf = unstructured_to_transformer(instruction_unstructured(i), pre, e);
    //tl = CONS(TRANSFORMER, tf, NIL);
    // Reuse the existing transformer
    tl = list_undefined;
    break ;
  case is_instruction_expression:
    /* Nothing fancy yet in spite of the C ? and , operators*/
    //tl = expression_to_transformer_list(instruction_expression(i), pre, e);
    //tf = expression_to_transformer(instruction_expression(i), pre, e);
    //tl = CONS(TRANSFORMER, tf, NIL);
    // Reuse the existing transformer
    tl = list_undefined;
    break;
  default:
    pips_internal_error("unexpected tag %d", instruction_tag(i));
  }
  pips_debug(9, "resultat:\n");
  ifdebug(9)
    {
      if(!list_undefined_p(tl))
	(void) print_transformers(tl);
      else
	(void) fprintf(stderr, "undefined list\n");
    }
  pips_debug(8, "end\n");
  return tl;
}

/* A transformer is already available for statement s, but it is
   going to be refined into a list of transformers to isolate at
   least the identity transformer from effective transformers. */
list statement_to_transformer_list(statement s,
				   transformer spre) /* stmt precondition */
{
  transformer tf = load_statement_transformer(s);
  instruction i = statement_instruction(s);
  list tl = NIL;
  list e = NIL;
  //transformer t = transformer_undefined;
  //transformer ot = transformer_undefined; /* old transformer for s */
  //transformer nt = transformer_undefined; /* new transformer for s under spre */
  //transformer te = transformer_undefined; /* nt updated with loop exit information */
  transformer pre = transformer_undefined;

  pips_debug(8,"begin for statement %03td (%td,%td) with precondition %p:\n",
	     statement_number(s), ORDERING_NUMBER(statement_ordering(s)),
	     ORDERING_STATEMENT(statement_ordering(s)), spre);
  ifdebug(8) {
    pips_assert("The statement and its substatements are fully defined",
		all_statements_defined_p(s));
    (void) print_transformer(spre);
  }

  pips_assert("spre is a consistent precondition",
	      transformer_consistent_p(spre));

  if(get_bool_property("SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT")) {
    pre = transformer_undefined_p(spre)? transformer_identity() :
      transformer_range(spre);
    if(refine_transformers_p) {
      /* Transformation REFINE_TRANSFORMERS is being executed: add
	 information available from the statement precondition */
      transformer srpre = load_statement_precondition(s);
      transformer srpre_r = transformer_range(srpre);

      pre = transformer_domain_intersection(pre, srpre_r);
      free_transformer(srpre_r);
    }
  }
  else {
    // Avoid lots of test in the callees
    // pre = transformer_undefined;
    pre = transformer_identity();
  }

  pips_assert("pre is a consistent precondition",
	      transformer_consistent_p(pre));

  pips_debug(8,"Range precondition pre:\n");
  ifdebug(8) {
    (void) print_transformer(pre);
  }

  // Probably not useful
  e = load_cumulated_rw_effects_list(s);

  /* it would be nicer to control warning_on_redefinition */
  if (true) {
    list dl = declaration_statement_p(s) ? statement_declarations(s) : NIL;

    if(!ENDP(dl)) {
      transformer dt = declarations_to_transformer(dl, pre);
      tl = CONS(TRANSFORMER, dt, NIL);
    }
    else {
      tl = instruction_to_transformer_list(i, tf, pre, e);
      /* FI: an empty tl means that s does not return. An undefined
	 tl means that instruction_to_transformer() is not
	 implemented in a satisfactory way. So the existing
	 transformer is used by default. */
      if(list_undefined_p(tl)) {
	transformer tf = load_statement_transformer(s);
	tl = CONS(TRANSFORMER, copy_transformer(tf), NIL);
      }
    }

    FOREACH(TRANSFORMER, nt, tl) {
      /* add array references information using proper effects */
      if(get_bool_property("SEMANTICS_TRUST_ARRAY_REFERENCES")) {
	transformer_add_reference_information(nt, s);
	/* nt = transformer_normalize(nt, 0); */
      }

      /* When we leave a block the local stack allocated variables
	 disappear */
      if(statement_block_p(s) && !ENDP(dl=statement_declarations(s))) {
	/* Get rid of the dynamic and stack variables declared in this block
	   statement. No stack variable should be analyzed as the stack
	   area is used only for dependent types. */
	list vl = dynamic_variables_to_values(dl);
	if(!ENDP(vl))
	  nt = safe_transformer_projection(nt, vl);
      }
      /* nt = transformer_normalize(nt, 7); */
      /*     nt = transformer_normalize(nt, 4); */
      nt = transformer_normalize(nt, 2);

      if(!transformer_consistency_p(nt)) {
	_int so = statement_ordering(s);
	(void) fprintf(stderr, "statement %03td (%td,%td):\n",
		       statement_number(s),
		       ORDERING_NUMBER(so), ORDERING_STATEMENT(so));
	/* (void) print_transformer(load_statement_transformer(s)); */
	(void) print_transformer(nt);
	dump_transformer(nt);
	pips_internal_error("Inconsistent transformer detected");
      }
      ifdebug(1) {
	pips_assert("Transformer is internally consistent",
		    transformer_internal_consistency_p(nt));
      }
    }

#if 0
    /* When the statement is virtually replicated via control nodes, the
       statement transformer is the convex hull of all its replicate
       transformers. */
    if(transformer_undefined_p(ot)) {
      if (get_int_property("SEMANTICS_NORMALIZATION_LEVEL_BEFORE_STORAGE") == 4)
	nt = transformer_normalize(nt, 4);
      t = copy_transformer(nt);
    }
    else {
      /* This implies that transformers are computed in context and that
	 we are dealing with a non-deterministic unstructured with several
	 nodes for a unique statement. */
      t = transformer_convex_hull(nt, ot);

      ifdebug(1) {
	pips_assert("transformers are computed in context",
		    get_bool_property("SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT"));
	pips_debug(1, "Convex hull for transformer of statement  %03td (%td,%td)\n",
		   statement_number(s), ORDERING_NUMBER(statement_ordering(s)),
		   ORDERING_STATEMENT(statement_ordering(s)));
	pips_debug(1, "Previous transformer:\n");
	(void) print_transformer(ot);
	pips_debug(1, "New transformer:\n");
	(void) print_transformer(nt);
	pips_debug(1, "Resulting transformer:\n");
	(void) print_transformer(t);
      }
    }

    /* store or update the statement transformer */
    if(transformer_undefined_p(ot)) {
      store_statement_transformer(s, t);
    }
    else {
      transformer_free(ot);
      update_statement_transformer(s, t);
      /* delete_statement_transformer(s); */
      /* store_statement_transformer(s, t); */
    }
#endif
  }
  else {
    pips_user_warning("redefinition for statement %03d (%d,%d)\n",
		      statement_number(s),
		      ORDERING_NUMBER(statement_ordering(s)),
		      ORDERING_STATEMENT(statement_ordering(s)));
    pips_internal_error("transformer redefinition");
  }

  /* The transformer returned for the statement is not always the
     transformer stored for the statement. This happens for loops and for
     context sensitive transformers for replicated statements in
     CFG/unstructured. See comments in loop.c */

  // FI: Should not be useful for transformer lists? Why?
  // te = complete_statement_transformer(nt, pre, s);

  free_transformer(pre);

  ifdebug(8) {
    pips_assert("The statement and its substatements are still fully defined",
		all_statements_defined_p(s));
  }

  //pips_debug(8,"end for statement %03td (%td,%td) with t=%p, nt=%p and te=%p\n",
  //     statement_number(s), ORDERING_NUMBER(statement_ordering(s)),
  //     ORDERING_STATEMENT(statement_ordering(s)), t, nt, te);

  ifdebug(1) {
    pips_debug(1, "Transformer list has %zd elements:\n", gen_length(tl));
    print_transformers(tl);
  }

  return tl;
}
