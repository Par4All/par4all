/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
  * phasis 1: compute transformers from statements and statements effects
  *
  * For (simple) interprocedural analysis, this phasis should be performed
  * bottom-up on the call tree.
  *
  * Francois Irigoin, April 1990
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

#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "transformer.h"

#include "semantics.h"

/**
 * \details         Apply or add an "concrete" effect e on a transformer tf
 *                  we must verify that the effect e is a write effect and
 *                  the effect of e is on abstract variable
 * \param args      arguments for assign (*p=v), list_undefined if no args
 * \param tf        transformer
 * \param e         write effects
 * \return          transformer with the effect
 */
static transformer apply_concrete_effect_to_transformer(list args, transformer tf, effect e)
{
  ifdebug(8) {
    pips_debug(8, "Begin\n");
    (void) print_transformer(tf);
  }
  entity v = reference_variable(effect_any_reference(e));

  if (!approximation_may_p(effect_approximation(e))) // NL : a may effect don't permit to know anything
  {
    expression rhs = EXPRESSION(CAR(CDR(args)));
    pips_assert("2 args to assign", CDR(CDR(args))==NIL);

    ifdebug(8) {
      (void) print_expression(rhs);
    }

    transformer post = any_scalar_assign_to_transformer_without_effect(v, rhs, tf);
    if(!transformer_undefined_p(post))
    {
      free_transformer(tf);
      tf = post;
    }
  }
  else {
    // NL : if we had a may effect, we make a projection for the variable
    // NL : need to make a function to make transformer_cylinder_base_projection(tf, v)
    //TODO : transformer_cylinder_base_projection(tf, v);
    transformer_add_variable_update(tf, v);

    //TODO : delete this part when transformer_cylinder_base_projection done
    // NL : reset all the sc can be to much,
    //      but don't see how to do otherwise without a projection function
    {
    Psysteme sc = predicate_system(transformer_relation(tf));
    sc_egalites(sc) = contraintes_free(sc_egalites(sc));
    sc_nbre_egalites(sc) = 0;
    sc_inegalites(sc) = contraintes_free(sc_inegalites(sc));
    sc_nbre_inegalites(sc) = 0;
    }
  }


  ifdebug(8) {
    (void) print_transformer(tf);
    pips_debug(8, "Ends\n");
  }
  return tf;
}

/**
 * \details         Apply or add an abstract effect e on a transformer tf
 *                  we must verify that the effect e is a write effect and
 *                  the effect of e is on abstract variable
 * \param tf        transformer
 * \param e         abstract write effects
 * \param apply_p   if we apply or add an effect, true if we apply
 * \return          transformer with the effect
 */
static transformer apply_abstract_effect_to_transformer(transformer tf, effect e, bool apply_p)
{
  ifdebug(8) {
    pips_debug(8, "Begin\n");
    (void) print_transformer(tf);
  }
  entity v = reference_variable(effect_any_reference(e));

  /* All analyzed variables conflicting with v must be considered
   *  written.
   *
   * This should depend on the abstract location, its type when
   * anywhere effects are typed, and its scope when abstract
   * locations can be restricted to a module or a compilation
   * unit. See the lattice defined by Amira Mensi.
   */
  list wvl = modified_variables_with_values();

  FOREACH(ENTITY, wv, wvl)
  {
    ifdebug(8) {
      (void) print_entity_variable(wv);
    }
    //NL : only modified variable with conflict with the effect is treat
    if(entities_may_conflict_p(v,wv) && entity_has_values_p(wv)) {
      if (apply_p) {
        //NL : if we apply effect, we need to check where the effect take place
        if (entity_all_module_locations_p(v)) {
          // NL : need to make a function to make transformer_cylinder_base_projection(tf, wv)
          //TODO : transformer_cylinder_base_projection(tf, wv);
          transformer_add_variable_update(tf, wv);
        }
        else {
          // NL : v and not wv, don't know why (come from effect with pointer values)
          // for testcase Effects/Effects_With_Pointer_Values.sub/dereferencing04(_2)
          // NL : transformer_add_variable_update need to be complete
          //      not sure if we have to use transformer_add_variable_update or transformer_add_value_update
          transformer_add_variable_update(tf, v);
          //transformer_add_value_update(tf, v);
        }
      }
      //NL : else we add effect, we just need to add the variable
      else {
        // NL : transformer_add_variable_update need to be complete
        //      not sure if we have to use transformer_add_variable_update or transformer_add_value_update
        transformer_add_variable_update(tf, wv);
        //transformer_add_value_update(tf, wv);
      }
    }
  }

  //TODO : delete this IF when transformer_cylinder_base_projection done
  // NL : if the effect is to write *ANYWHERE*, we lose all the constraint
  if (apply_p && entity_all_module_locations_p(v))
  {
    Psysteme sc = predicate_system(transformer_relation(tf));
    sc_egalites(sc) = contraintes_free(sc_egalites(sc));
    sc_nbre_egalites(sc) = 0;
    sc_inegalites(sc) = contraintes_free(sc_inegalites(sc));
    sc_nbre_inegalites(sc) = 0;
  }

  ifdebug(8) {
    (void) print_transformer(tf);
    pips_debug(8, "Ends\n");
  }
  return tf;
}

/**
 * \details         Apply or add an effect e on a transformer tf
 *                  only the write effects is compute
 * \param args      arguments for assign (*p=v), list_undefined if no args
 * \param tf        transformer
 * \param e         effect
 * \param apply_p   if we apply or add an effect, true if we apply
 * \return          transformer with the effect
 */
//must be static because of the dependence (type effect only declare on effects.h)
static transformer apply_effect_to_transformer(list args, transformer tf, effect e, bool apply_p)
{
  ifdebug(8) {
    pips_debug(8, "Begin\n");
    (void) print_transformer(tf);
  }
  entity v = reference_variable(effect_any_reference(e));

  //action a = effect_action(e);
  //normally this test is already done by the caller function
  //if(action_write_p(a))
  {
    /* The check on static should be useless because already taken
       into account when effects are computed. And it is harmful here
       since most effects on static variables cannot be ignored. */
    if(entity_has_values_p(v)
        && store_effect_p(e) // FI: we should have action_memory_write_p()
        //&& !variable_static_p(v)
    ) {
      pips_debug(8, "\"concrete\" effect\n");
      if (apply_p
          //&& !list_undefined_p(args) // NL : apply_p=false <-> args=list_undefined
      ) {
        // NL : case *p=v
        tf = apply_concrete_effect_to_transformer(args, tf, e);
      }
      else {
        // NL : general case, don't understand what exactly we do in this case (what kind of case it is)
        // NL : transformer_add_variable_update need to be complete
        //      not sure if we have to use transformer_add_variable_update or transformer_add_value_update
        transformer_add_variable_update(tf, v);
        //transformer_add_value_update(tf, v);
      }
    }
    else if(entity_abstract_location_p(v)) {
      pips_debug(8, "abstract effect\n");
      tf = apply_abstract_effect_to_transformer(tf, e, apply_p);
    }
  }

  ifdebug(8) {
    (void) print_transformer(tf);
    pips_debug(8, "Ends\n");
  }
  return tf;
}

/**
 * old name add_effects_to_transformer
 *
 * \details
 * Make sure that all variables modified according to the effect list
 * e and taken into account by the semantics analysis is an argument
 * of tf and that the corresponding variables are declared in the
 * basis.
 *
 * Non allocation, just a side effect on tf.
 *
 * \brief           Apply or add a list of effect el on a transformer tf
 *                  only the write effects is compute
 * \param args      arguments for assign (*p=v), list_undefined if no args
 * \param tf        transformer
 * \param el        list of effects
 * \param apply_p   if we apply or add the list of effect, true if we apply
 * \return          transformer with the effects
 */
transformer apply_effects_to_transformer(list args, transformer tf, list el, bool apply_p)
{
  /* algorithm: keep only memory write effects on variables with values */
  FOREACH(EFFECT, e, el) {
    action a = effect_action(e);

    if(action_write_p(a))
      tf = apply_effect_to_transformer(args, tf, e, apply_p);
  }
  return tf;
}


transformer effects_to_transformer(list e) /* list of effects */
{
  transformer tf = transformer_identity();
  tf =  apply_effects_to_transformer(list_undefined, tf, e, false);
  return tf;
}

/* Previous version of effects_to_transformer()
transformer effects_to_transformer(list e)
{
  list args = NIL;
  Pbase b = VECTEUR_NUL;
  Psysteme s = sc_new();

  s->base = b;
  s->dimension = vect_size(b);

  return make_transformer(args, make_predicate(s));
}
*/

transformer filter_transformer(transformer t, list e)
{
  /* algorithm: keep only information about scalar variables with values
   * appearing in effects e and store it into a newly allocated transformer
   */
  Pbase b = VECTEUR_NUL;
  Psysteme s = SC_UNDEFINED;
  Psysteme sc = predicate_system(transformer_relation(t));
  list args = NIL;
  Psysteme sc_restricted_to_variables_transitive_closure(Psysteme, Pbase);

  FOREACH(EFFECT, ef, e) {
    reference r = effect_any_reference(ef);
    /* action a = effect_action(ef); */
    entity v = reference_variable(r);

    if(/* action_write_p(a) && */ entity_has_values_p(v)) {
      /* I do not know yet if I should keep old values... */
      entity new_val = entity_to_new_value(v);
      b = vect_add_variable(b, (Variable) new_val);

      if(entity_is_argument_p(v, transformer_arguments(t))) {
	args = arguments_add_entity(args, v);
      }
    }
  }

  /* FI: I should check if sc is sc_empty but I haven't (yet) found a
     cheap syntactic test */
  s = sc_restricted_to_variables_transitive_closure(sc, b);

  return make_transformer(args, make_predicate(s));
}


/* Recursive Descent in Data Structure Statement */

/* SHARING : returns the transformer stored in the database. Make a
 * copy before using it. The copy is not made here because the result
 * is not always used after a call to this function, and it would
 * create non reachable structures. Another solution would be to store
 * a copy and free the unused result in the calling function but
 * transformer_free does not really free the transformer. Not very
 * clean.  BC, oct. 94
 */

/* Assumes that entity_has_values_p(v) holds. */
transformer dimensions_to_transformer(entity v, transformer pre)
{
  transformer dt = transformer_identity();
  type vt = entity_type(v); // Do not use ultimate_type or you'll miss
			    // the dimensions.

  if(type_variable_p(vt)) {
    list dl = variable_dimensions(type_variable(vt)); // dimension list
    if(!ENDP(dl)) { // to save a copy and to simplify debugging
      transformer cpre = copy_transformer(pre);

      FOREACH(DIMENSION, d, dl) {
	expression l = dimension_lower(d);
	expression u = dimension_upper(d);
	transformer lt = safe_expression_to_transformer(l, cpre);
	transformer lpre = transformer_apply(lt, pre);
	transformer lpre_r = transformer_range(lpre);
	transformer ut = safe_expression_to_transformer(u, lpre_r);
	transformer upre = transformer_apply(ut, lpre);

	free_transformer(cpre);
	cpre = transformer_range(upre);
	free_transformer(upre);
	free_transformer(lpre);
	free_transformer(lpre_r);

	dt = transformer_combine(transformer_combine(dt, lt), ut);
	free_transformer(lt);
	free_transformer(ut);
      }
      free_transformer(cpre);
    }
  }

  return dt;
}

/* Note: initializations of static variables are not used as
   transformers but to initialize the program precondition. */
/* It is not assumed that entity_has_values_p(v)==TRUE */
/* A write effect on the declared variable is assumed as required by
   Beatrice Creusillet for region computation. */
transformer declaration_to_transformer(entity v, transformer pre)
{
  transformer tf = transformer_undefined;

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
    transformer nr = transformer_range(npre);
    expression ie = variable_initial_expression(v);
    transformer itf = safe_assigned_expression_to_transformer(v, ie, nr);
    tf = dt;
    tf = transformer_combine(tf, itf);
    free_expression(ie);
    free_transformer(npre);
    free_transformer(nr);
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
    (void) print_transformer(tf);
  }

  return tf;
}

/* For C declarations. Very close to a block_to_transformer() as
   declarations can be seen as a sequence of assignments.

   Note: initialization of static variables are not taken into
   account. They must be used for summary preconditions.
 */
transformer declarations_to_transformer(list dl, transformer pre)
{
  entity v = entity_undefined;
  transformer btf = transformer_undefined;
  transformer stf = transformer_undefined;
  transformer post = transformer_undefined;
  transformer next_pre = transformer_undefined;
  list l = dl;

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
  return btf;
}

/* Compute the transformer of a block under precondition pre
 *
 * Precondition pre may be undefined to compute transformers purely
 * upwards or be defined if the transformers are refined (apply
 * REFINE_TRANSFORMERS) or if the transformers are computed in context.
 *
 * FI: it is not clear if postconditions should be propagated or if
 * the range of the current transformer is exactly what is needed to
 * compute the transformer of the next statement.
 *
 * When precondition pre is undefined, this piece of code is supposed
 * to behave as if preconditions were never calculated nor used. The
 * complexiy problem encountered with Semantics/mpeg2enc even with the
 * option SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT false seems to
 * indicate that we end up with usable preconditions even when they
 * are not needed.
 *
 * FI: more investigation is needed to control the execution time. The
 * spontaneous computation of preconditions would lead to a time
 * increase at all levels, for instance when non-affine operators are
 * approximated. The behavior of block_to_transformer() seems OK with
 * Semantics-New/block01. It should be checked again with
 * Semantics-New mpeg2enc.
 */
static transformer block_to_transformer(list b, transformer pre)
{
  statement s;
  transformer btf = transformer_undefined;
  transformer stf = transformer_undefined;
  transformer post = transformer_undefined;
  transformer next_pre = transformer_undefined;
  list l = b;

  pips_debug(8,"begin\n");

  if(ENDP(l))
    btf = transformer_identity();
  else {
    s = STATEMENT(CAR(l));
    stf = statement_to_transformer(s, pre);
    post = transformer_safe_apply(stf, pre);
/*     post = transformer_safe_normalize(post, 4); */
    post = transformer_safe_normalize(post, 2);
    btf = transformer_dup(stf);
    for (POP(l) ; !ENDP(l); POP(l)) {
      s = STATEMENT(CAR(l));
      if(!transformer_undefined_p(next_pre))
	free_transformer(next_pre);

      // In case "ocean", this is a performance bug due to r18644
      // next_pre = transformer_range(post);
      // free_transformer(post);
      next_pre = post;

      stf = statement_to_transformer(s, next_pre);
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
    free_transformer(post);
  }

  // FI: I add a stronger normalization at the end of the block
  //
  // The lighter normalization (level 2) in the loop was introduced to deal with very long
  // basic blocks generated by Scilab (I believe). This light
  // normalization does not detect equations split into two
  // inequalities, not identical constraints... What does it deal with?
  //
  // The stronger normalization (level 4) is added for putnonintrablk() from
  // mpeg2, although it's probably too late to recover from the very
  // large coefficients introduced earlier
  //
  // FI: does not seem to do much good because the normalization may
  // increase the complexity of the constraints
  // btf = transformer_normalize(btf, 4);

  pips_debug(8, "end\n");
  return btf;
}

list effects_to_arguments(list fx) /* list of effects */
{
  /* algorithm: keep only write effects on scalar variable with values */
  list args = NIL;

  FOREACH(EFFECT, ef, fx) {
    reference r = effect_any_reference(ef);
    action a = effect_action(ef);
    entity e = reference_variable(r);

    if(action_write_p(a) && entity_has_values_p(e)) {
      args = arguments_add_entity(args, e);
    }
  }

  return args;
}


static transformer test_to_transformer(test t,
				       transformer pre,
				       list ef) /* effects of t */
{
  statement st = test_true(t);
  statement sf = test_false(t);
  transformer tf;

  /* EXPRESSION_TO_TRANSFORMER() SHOULD BE USED MORE EFFECTIVELY */

  pips_debug(8,"begin\n");

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
    transformer post_tftwc = transformer_undefined;
    transformer post_tffwc = transformer_undefined;
    list ta = NIL;
    list fa = NIL;
    /* True condition transformer */
    transformer tct = condition_to_transformer(e, context, true);
    /* False condition transformer */
    transformer fct = condition_to_transformer(e, context, false);

    /*
    tftwc = transformer_dup(statement_to_transformer(st));
    tffwc = transformer_dup(statement_to_transformer(sf));
    */


    /* tftwc = precondition_add_condition_information(tftwc, e, context, true); */
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
    post_tftwc = transformer_apply(statement_to_transformer(st, tftwc), tftwc);
    //post_tftwc = transformer_normalize(post_tftwc, 2);

    ifdebug(8) {
      pips_debug(8, "tftwc after transformer_apply %p:\n", tftwc);
      (void) print_transformer(tftwc);
      pips_debug(8, "post_tftwc after transformer_apply %p:\n", post_tftwc);
      (void) print_transformer(post_tftwc);
    }

    /* tffwc = precondition_add_condition_information(tffwc, e, context, false); */
    tffwc = transformer_apply(fct, context);
    tffwc = transformer_temporary_value_projection(tffwc);
    reset_temporary_value_counter();
    post_tffwc = transformer_apply(statement_to_transformer(sf, tffwc), tffwc);
    //post_tffwc = transformer_normalize(post_tffwc, 2);

    ifdebug(8) {
      pips_debug(8, "post_tftwc before transformer_convex_hull %p:\n", post_tftwc);
      (void) print_transformer(post_tftwc);
      pips_debug(8, "post_tffwc after transformer_apply %p:\n", post_tffwc);
      (void) print_transformer(post_tffwc);
    }
    tf = transformer_convex_hull(post_tftwc, post_tffwc);
    transformer_free(context);
    transformer_free(tftwc);
    transformer_free(tffwc);
    transformer_free(post_tftwc);
    transformer_free(post_tffwc);
    free_arguments(ta);
    free_arguments(fa);
  }
  else {
    transformer id = transformer_identity();
    (void) statement_to_transformer(st, id);
    (void) statement_to_transformer(sf, id);
    tf = effects_to_transformer(ef);
    free_transformer(id);
  }

  pips_debug(8,"end\n");
  return tf;
}

transformer intrinsic_to_transformer(entity e,
				     list pc,
				     transformer pre,
				     list ef) /* effects of intrinsic call */
{
  transformer tf = transformer_undefined;

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
  else if(ENTITY_RAND_P(e)) {
    /* The result is positive and less than RAND_MAX, but it is
       ignored by the semantics anaysis */
    pips_user_warning("Value returned by intrinsic \"rand\" is ignored.\n");
    //tf = transformer_add_inequality_with_integer_constraint(transformer_identity(),
    // e, 0, true);
    tf = transformer_identity();
  }
  else
    tf = effects_to_transformer(ef);

  pips_debug(8, "end\n");

  return tf;
}

/* The transformer returned for a call site may be too accurate for
   the caller. Information about specific variables available at the
   callee level may be lost at the caller level because some abstract
   locations wraps up together independent variables. For instance,
   as soon as *any_module*:*any_where* appears, information loss
   seems inevitable.

   So, values in tf related to variables with no values in the current
   module must be projected, except for the values of the return
   variable and for the constants used to represent floating point
   numbers and strings.
*/
static transformer transformer_filter_subsumed_variables(transformer tf)
{
  Psysteme sc = predicate_system(transformer_relation(tf));
  Pbase b = sc_base(sc);
  Pbase cb = b;
  list svl = NIL; // subsumed value list
  list pvl = NIL; // preserved variable lis
  entity rv = entity_undefined;
  entity orv = entity_undefined; // should be useless in C, but is not
				 // in Fortran: unification

  /* This should be moved down to library transformer or vecteur using
     value_entity_p() as an argument, up to typing issues... */

  for(cb = b; !BASE_NULLE_P(cb); cb = vecteur_succ(cb)) {
    entity v = (entity) vecteur_var(cb);
    if(variable_return_p(v)) {
      string orvn = strdup(concatenate(entity_name(v),
				       OLD_VALUE_SUFFIX,
				       NULL));
      rv = v;
      orv = gen_find_tabulated(orvn, entity_domain);
      free(orvn);
    }
  }

  for(cb = b; !BASE_NULLE_P(cb); cb = vecteur_succ(cb)) {
    entity v = (entity) vecteur_var(cb);
    if(!value_entity_p(v) && v!=rv && v!=orv && !entity_constant_p(v)) {
      pips_user_warning("Value \"%s\" is projected because of "
			"imprecise effects\n", entity_name(v));
      svl = CONS(ENTITY, v, svl);
    }
  }

  tf = transformer_projection(tf, svl);

  gen_free_list(svl);

  /* Clean up argument */
  FOREACH(ENTITY, var, transformer_arguments(tf)) {
    if(entity_has_values_p(var))
      pvl = CONS(ENTITY, var, pvl);
  }

  pvl = gen_nreverse(pvl);
  gen_free_list(transformer_arguments(tf));
  transformer_arguments(tf) = pvl;

  return tf;
}

// FI: I use it in expression.c...
//static transformer user_call_to_transformer(entity, list, transformer, list);

/* Use to be static, but may be called from expressions in C. */
transformer call_to_transformer(call c,
				transformer pre,
				list ef) /* effects of call c */
{
  transformer tf = transformer_undefined;
  entity e = call_function(c);
  cons *pc = call_arguments(c);
  tag tt;

  pips_debug(8,"begin with precondition %p\n", pre);

  switch (tt = value_tag(entity_initial(e))) {
  case is_value_code:
    /* call to an external function; preliminary version:
       rely on effects */
    pips_debug(5, "external function \"%s\"\n", entity_name(e));
    if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      type et = ultimate_type(entity_type(e));
      type rt = ultimate_type(functional_result(type_functional(et)));

      if(type_void_p(rt)) {
	tf = user_call_to_transformer(e, pc, pre, ef);
	reset_temporary_value_counter(); // might not be a good idea
					 // with expression lists?
	// Get rid of variables that have been subsumed by abstract
	// locations
	tf = transformer_filter_subsumed_variables(tf);
      }
      else {
	if(analyzable_type_p(rt)) {
	  /* A temporary variable should be allocated and
	     user_function_call_to_transformer() be used. The variable
	     should then be projected to keep only the side effects of
	     the call. */
	  entity trv = make_local_temporary_value_entity(rt);
	  expression expr = call_to_expression(c);

	  tf = user_function_call_to_transformer(trv, expr, pre);
	  tf = transformer_temporary_value_projection(tf);
	  reset_temporary_value_counter();
	  pips_user_warning("Analyzable result of function \"%s\" ignored.\n",
			    entity_user_name(e));
	  // Get rid of variables that have been subsumed by abstract
	  // locations
	  tf = transformer_filter_subsumed_variables(tf);
	}
	else {
	  tf = user_call_to_transformer(e, pc, pre, ef);
	  reset_temporary_value_counter(); // might not be a good idea
	  pips_user_warning("Result of function \"%s\" ignored.\n",
			    entity_user_name(e));
	  // Get rid of variables that have been subsumed by abstract
	  // locations
	  tf = transformer_filter_subsumed_variables(tf);
	}
      }
    }
    else
      tf = effects_to_transformer(ef);
    break;
  case is_value_symbolic:
  case is_value_constant:
    tf = transformer_identity();
    break;
  case is_value_unknown:
    pips_internal_error("function %s has an unknown value", entity_name(e));
    break;
  case is_value_intrinsic:
    pips_debug(5, "intrinsic function %s\n", entity_name(e));
    tf = intrinsic_to_transformer(e, pc, pre, ef);
    break;
  default:
    pips_internal_error("unknown tag %d", tt);
  }
  pips_assert("transformer tf is consistent",
	      transformer_consistency_p(tf));

  pips_debug(8,"Transformer before intersection with precondition, tf=%p\n",
	     tf);
  ifdebug(8) {
    (void) print_transformer(tf);
  }

  /* Add information from pre. Invariant information is easy to
     use. Information about initial values, that is final values in pre,
     can also be used. */
  tf = transformer_safe_domain_intersection(tf, pre);
  pips_debug(8,"After intersection and before normalization with tf=%p\n", tf);
  ifdebug(8) {
    (void) print_transformer(tf);
  }
  pips_debug(8,"with precondition pre=%p\n", pre);
  ifdebug(8) {
    (void) print_transformer(pre);
  }
/*   tf = transformer_normalize(tf, 4); */
  tf = transformer_normalize(tf, 2);

  pips_debug(8,"end after normalization with tf=%p\n", tf);
  ifdebug(8) {
    (void) print_transformer(tf);
  }

  return(tf);
}

/* The Fortran and C versions are about the same. Should I revert and
   unify them, except for t_calle? This could be unified too by
   calling user_call_to_transformer()? */
static transformer
c_user_function_call_to_transformer(
				    entity e, /* a value */
				    expression expr, /* a call to a function */
				    transformer pre) /* its precondition */
{
  transformer t_caller = transformer_undefined;
  syntax s = expression_syntax(expr);
  call c = syntax_call(s);
  entity f = call_function(c);
  list pc = call_arguments(c);
  basic rbt = basic_of_call(c, true, true);
  //list ef = expression_to_proper_effects(expr);
  list ef = expression_to_proper_constant_path_effects(expr);

  pips_debug(8, "begin\n");
  pips_assert("s is a call", syntax_call_p(s));

  /* if there is no implicit cast */
  if(same_basic_p(rbt, entity_basic(e))) {
    const char* fn = module_local_name(f);
    entity rv = FindEntity(fn, fn);
    entity orv = entity_undefined;
    transformer t_equal = simple_equality_to_transformer(e, rv, false);

    pips_assert("rv is defined",
		!entity_undefined_p(rv));

    /* Build a transformer reflecting the call site */
    /* Too bad the precondition is not passed down to evaluate the
     actual argument expressions...  To be changed in the C version*/
    t_caller = c_user_call_to_transformer(f, pc, pre, ef);

    ifdebug(8) {
      pips_debug(8, "Transformer %p for callee %s:\n",
		 t_caller, entity_local_name(f));
      dump_transformer(t_caller);
    }

    /* Consistency cannot be checked on a non-local transformer */
    /* pips_assert("t_equal is consistent",
       transformer_consistency_p(t_equal)); */

    ifdebug(8) {
      pips_debug(8,
		 "Transformer %p for equality of %s with %s:\n",
		 t_equal, entity_local_name(e), entity_name (rv));
      dump_transformer(t_equal);
    }

    /* Combine the effect of the function call and of the equality */
    t_caller = transformer_combine(t_caller, t_equal);
    free_transformer(t_equal);

    /* Get rid of the temporary representing the function's value */
    orv = global_new_value_to_global_old_value(rv);
    if(entity_undefined_p(orv))
      t_caller = transformer_filter(t_caller, CONS(ENTITY, rv, NIL));
    else
      t_caller = transformer_filter(t_caller,
				    CONS(ENTITY, rv, CONS(ENTITY, orv, NIL)));

    ifdebug(8) {
      pips_debug(8,
		 "Final transformer %p for assignment of %s with %s:\n",
		 t_caller, entity_local_name(e), entity_name(rv));
      dump_transformer(t_caller);
    }
  }
  else {
    t_caller = effects_to_transformer(ef);
  }

  gen_free_list(ef);

  pips_debug(8, "end with t_caller=%p\n", t_caller);

  return t_caller;
}

static transformer 
fortran_user_function_call_to_transformer(
					  entity e, /* a value */
					  expression expr, /* a call to a function */
					  transformer __attribute__ ((unused)) pre) /* its precondition */
{
  syntax s = expression_syntax(expr);
  call c = syntax_call(s);
  entity f = call_function(c);
  list pc = call_arguments(c);
  transformer t_caller = transformer_undefined;
  basic rbt = basic_of_call(c, true, true);
  //list ef = expression_to_proper_effects(expr);
  list ef = expression_to_proper_constant_path_effects(expr);

  pips_debug(8, "begin\n");
  pips_assert("s is a call", syntax_call_p(s));

  /* if(basic_int_p(rbt)) { */
  if(basic_equal_p(rbt, variable_basic(type_variable(entity_type(e))))) {
    const char* fn = module_local_name(f);
    entity rv = FindEntity(fn, fn);
    entity orv = entity_undefined;
    Psysteme sc = SC_UNDEFINED;
    Pcontrainte c = CONTRAINTE_UNDEFINED;
    Pvecteur eq = VECTEUR_NUL;
    transformer t_equal = transformer_undefined;

    pips_assert("rv is defined",
		!entity_undefined_p(rv));

    /* Build a transformer reflecting the call site */
    /* Too bad the precondition is not passed down to evaluate the
     actual argument expressions...  To be changed in the C version*/
    t_caller = fortran_user_call_to_transformer(f, pc, ef);

    ifdebug(8) {
      pips_debug(8, "Transformer %p for callee %s:\n",
		 t_caller, entity_local_name(f));
      dump_transformer(t_caller);
    }

    /* Build a transformer representing the equality of
     * the function value to e
     */
    eq = vect_make(eq,
		   (Variable) e, VALUE_ONE,
		   (Variable) rv, VALUE_MONE,
		   TCST, VALUE_ZERO);
    c = contrainte_make(eq);
    sc = sc_make(c, CONTRAINTE_UNDEFINED);
    t_equal = make_transformer(NIL,
			       make_predicate(sc));

    /* Consistency cannot be checked on a non-local transformer */
    /* pips_assert("t_equal is consistent",
       transformer_consistency_p(t_equal)); */

    ifdebug(8) {
      pips_debug(8,
		 "Transformer %p for equality of %s with %s:\n",
		 t_equal, entity_local_name(e), entity_name (rv));
      dump_transformer(t_equal);
    }

    /* Combine the effect of the function call and of the equality */
    t_caller = transformer_combine(t_caller, t_equal);
    free_transformer(t_equal);

    /* Get rid of the temporary representing the function's value */
    orv = global_new_value_to_global_old_value(rv);
    if(entity_undefined_p(orv))
      t_caller = transformer_filter(t_caller, CONS(ENTITY, rv, NIL));
    else
      t_caller = transformer_filter(t_caller,
				    CONS(ENTITY, rv, CONS(ENTITY, orv, NIL)));


    ifdebug(8) {
      pips_debug(8,
		 "Final transformer %p for assignment of %s with %s:\n",
		 t_caller, entity_local_name(e), entity_name(rv));
      dump_transformer(t_caller);
    }

    /* FI: e is added in arguments because user_call_to_transformer()
     * uses effects to make sure arrays and non scalar integer variables
     * impact is taken into account
     */
    /*
      transformer_arguments(t_caller) =
      arguments_rm_entity(transformer_arguments(t_caller), e);
    */

    /* FI, FI: il vaudrait mieux ne pas eliminer e d'abord1 */
    /* J'ai aussi des free a decommenter */
    /*
      if(ENDP(transformer_arguments(t_caller))) {
      transformer_arguments(t_caller) =
      gen_nconc(transformer_arguments(t_caller), CONS(ENTITY, e, NIL));
      }
      else {
      t_caller = transformer_value_substitute(t_caller, rv, e);
      }
    */
    /* Not checkable with temporary variables
       pips_assert("transformer t_caller is consistent",
       transformer_consistency_p(t_caller));
    */
  }
  else {
    pips_assert("transformer t_caller is undefined",
		transformer_undefined_p(t_caller));
  }

  gen_free_list(ef);

  pips_debug(8, "end with t_caller=%p\n", t_caller);


  return t_caller;
}

/* a function call is a call to a non void function in C and to a
   FUNCTION in Fortran */
transformer
user_function_call_to_transformer(
				  entity e, /* a value */
				  expression expr, /* a call to a function */
				  transformer pre) /* its precondition */
{
  transformer tf = transformer_undefined;
  call c = expression_call(expr);
  entity f = call_function(c);

  if(c_module_p(f))
    tf = c_user_function_call_to_transformer(e, expr, pre);
  else
    tf = fortran_user_function_call_to_transformer(e, expr, pre);

  return tf;
}

/* transformer translation
 */
transformer transformer_intra_to_inter(transformer tf,
				       list le)
{
  cons * lost_args = NIL;
  /* Filtered TransFormer ftf */
  transformer ftf = transformer_dup(tf);
  /* cons * old_args = transformer_arguments(ftf); */
  Psysteme sc = SC_UNDEFINED;
  Pbase b = BASE_UNDEFINED;
  Pbase eb = BASE_UNDEFINED;

  pips_debug(8,"begin\n");
  pips_debug(8,"argument tf=%p\n",ftf);
  ifdebug(8) (void) dump_transformer(ftf);

  /* get rid of tf's arguments that do not appear in effects le */

  /* build a list of arguments to suppress */
  /* FI: I do not understand anymore why corresponding old values do not have
   * to be suppressed too (6 July 1993)
   *
   * FI: because only read arguments are eliminated, non? (12 November 1995)
   *
   * FI: the resulting intermediate transformer is not consistent (18 July 2003)
   */
  /*
  MAPL(ca,
  {entity e = ENTITY(CAR(ca));
  if(!effects_write_entity_p(le, e) &&
     !storage_return_p(entity_storage(e)))
    lost_args = arguments_add_entity(lost_args, e);
  },
       old_args);
  */
  /* get rid of them */
  /* ftf = transformer_projection(ftf, lost_args); */

  /* free the temporary list of entities */
  /*
  gen_free_list(lost_args);
  lost_args = NIL;

  pips_debug(8,"after first filtering ftf=%x\n",ftf);
  ifdebug(8) (void) dump_transformer(ftf);
  */

  /* get rid of local read variables */

  /* FI: why not use this loop to get rid of *all* local variables, read or written? */

  sc = (Psysteme) predicate_system(transformer_relation(ftf));
  b = sc_base(sc);
  for(eb=b; !BASE_UNDEFINED_P(eb); eb = eb->succ) {
    entity e = (entity) vecteur_var(eb);

    if(e != (entity) TCST) {
      entity v = value_to_variable(e);

      /* Variables with no impact on the caller world are eliminated.
       * However, the return value associated to a function is preserved.
       */
      if( !effects_must_read_or_write_scalar_entity_p(le, v) &&
	  !storage_return_p(entity_storage(v)) &&
	  !entity_constant_p(v)) {
	lost_args = arguments_add_entity(lost_args, e);
      }
    }
  }

  /* get rid of them */
  ftf = transformer_projection(ftf, lost_args);

  /* free the temporary list of entities */
  gen_free_list(lost_args);
  lost_args = NIL;

  pips_debug(8,"return ftf=%p\n",ftf);
  ifdebug(8) (void) dump_transformer(ftf);
  pips_debug(8,"end\n");

  return ftf;
}

/* Number of parameters in pl before a vararg is reached. The varargs
   are not analyzed. */
unsigned int number_of_usable_functional_parameters(list pl)
{
  int n = 0;
  FOREACH(PARAMETER, p ,pl) {
	  type lpt = ultimate_type(parameter_type(p));
	  if(type_varargs_p(lpt))
	    break;
	  else
	    n++;
  }
  return n;
}

/* FI: the handling of varargs had to be modified for
   Semantics/va_arg.c. The code should probably now be refactored. */
transformer any_user_call_site_to_transformer(entity f,
					      list pc,
					      transformer pre,
					      list __attribute__ ((unused)) ef)
{
  transformer cpre = transformer_undefined_p(pre)?
    transformer_identity() : copy_transformer(pre);
  transformer tf = transformer_identity();
  type ft = ultimate_type(entity_type(f));
  functional fft = type_functional(ft); // proper assert checked earlier
  list pl = functional_parameters(fft);
  unsigned int pll = number_of_usable_functional_parameters(pl);
  list cpl = pl; // to simplify debugging
  int n = 1; /* Formal parameters are counted 1, 2, 3,...*/
  int mn = 1000 /*MAX_INT*/; /* Maximal numer of actual arguments that can be
		       used. */
  list fpvl = NIL; // list of formal parameter values

  if(pll != gen_length(pc)) {
    /* This may happen with a void declaration */
    if(pll==1 && gen_length(pc)==0
       && type_void_p(parameter_type(PARAMETER(CAR(pl))))) {
      mn = -1;
    }
    else if(pll < gen_length(pc)) {
      /* This may happen with a varargs: the number of actual
	 arguments is greater than or equal to the number of formal
	 parameters */
      if(pll==0) {
	/* FI: this case could be processed... */
	pips_user_warning("Different numbers of actual and formal parameters"
			  "(%d and %d) for function \"%s\"\n",
			  gen_length(pc), gen_length(pl), entity_user_name(f));
	mn = -1;
      }
      else {
	parameter lp = PARAMETER(CAR(gen_last(pl)));
	type lpt = ultimate_type(parameter_type(lp));
	if(type_varargs_p(lpt)) {
	  /* The first actual arguments can be used */
	  mn = pll;
	}
	else {
	  pips_user_error("Different numbers of actual and formal parameters"
			  "(%d and %d) for function \"%s\"\n",
			  gen_length(pc), gen_length(pl), entity_user_name(f));
	}
      }
    }
    else {
      pips_user_error("Incompatible numbers of actual and formal parameters"
		      "(%d and %d) for function \"%s\"\n",
		      gen_length(pc), pll, entity_user_name(f));
    }
  }
  else
    mn = pll;

  if(mn>=0) {
  /* Evaluate actual arguments from left to right linking it to a
     functional parameter when possible */
  FOREACH(EXPRESSION,e, pc) {
    type fpt = ultimate_type(parameter_type(PARAMETER(CAR(cpl))));  // formal parameter type
    type apt = expression_to_type(e); // actual parameter type
    entity fpv = find_ith_parameter(f, n); // formal parameter variable (and value)
    /* Because we are using the caller's framework, we cannot use the
       new/old value naming in the callee's framework

       entity fpvv = entity_to_new_value(fpv);

       Let's cheat: we know fpvv would be fpv. Furthermore, the formal
       parameter cannot be updated in C because of the value passing
       mode.
    */
    basic ab = variable_basic(type_variable(apt));
    basic fb = variable_basic(type_variable(fpt));
    basic b = basic_of_expression(e);
    transformer ctf = transformer_undefined;

    if(entity_undefined_p(fpv)) {
      /* FI: we could accept a number of actual parameters greater
	 than the number of formal parameters and move on by ignoring
	 expression e?!? Side effects in e should be taken into
	 account anyway... */
      pips_user_error("Cannot find formal parameter %d for function \"%s\"."
		      " Mismatch between function declaration and call site."
		      " Check the source code with flint, gcc or gfortran.\n",
		      n, entity_user_name(f));
    }

    if(analyzable_scalar_entity_p(fpv)) {
      if(type_equal_p(fpt, apt)) {
	/* Keep track of fpv to project it later */
	fpvl = CONS(ENTITY, fpv, fpvl);
	ctf =
	  // FI: I'm at a lost with this flag
	  //safe_any_expression_to_transformer(fpv, e, cpre, true);
	  safe_any_expression_to_transformer(fpv, e, cpre, false);
      }
      else if(basic_int_p(ab) && basic_int_p(fb)) {
	int as = basic_int(ab);
	int fs = basic_int(fb);

	if(as-fs==10 || fs-as==10)
	  pips_user_warning("Signed/unsigned integer type conversion.\n");
	else
	  pips_user_warning("Integer type conversion: actual %d and formal %d\n", as, fs);
	fpvl = CONS(ENTITY, fpv, fpvl);
	ctf = safe_any_expression_to_transformer(fpv, e, cpre, false);
      }
      else if((basic_int_p(ab) && derived_type_p(fpt))
	      ||(basic_int_p(fb) && derived_type_p(apt)) ) {
	/* FI: Let's assume it is an enum derived type... Too late
	   for a better job... */

	pips_user_warning("Integer/enum or enum/integer type conversion"
			  " for argument \"%s\" (rank %d) of function \"%s\" "
			  "called from function \"%s\".\n",
			  entity_user_name(fpv), n, entity_user_name(f),
			  entity_user_name(get_current_module_entity()));

	fpvl = CONS(ENTITY, fpv, fpvl);
	ctf = safe_any_expression_to_transformer(fpv, e, cpre, false);
      }
      else {
	/* Should be an error or a warning? */
	//list el = expression_to_proper_effects(e);
	list el = expression_to_proper_constant_path_effects(e);
	/*
	pips_user_error("Type incompatibility between call site and declaration"
			" for %d argument of function %s\n", n, entity_user_name(f));
	*/
	pips_user_warning("Type incompatibility between call site and declaration"
			  " for argument \"%s\" (rank %d) of function \"%s\" "
			  "called from function \"%s\": %s/%s\n",
			  entity_user_name(fpv), n, entity_user_name(f),
			  entity_user_name(get_current_module_entity()),
			  basic_to_string(fb), basic_to_string(ab));

	ctf = effects_to_transformer(el);
      }
    }
    else {
      /* The associated transformer may nevertheless carry useful/necessary information */
      //list el = expression_to_proper_effects(e);
      list el = expression_to_proper_constant_path_effects(e);

      ctf = effects_to_transformer(el);
    }

    transformer npre = transformer_undefined;

    tf = transformer_combine(tf, ctf);
    npre = transformer_apply(ctf, cpre);
    npre = transformer_normalize(npre, 2);
    free_transformer(cpre);
    cpre = npre;
    POP(cpl);
    n++;
    free_type(apt);
    free_basic(b);
    /* Can we still use the next actual parameter? Maybe not because
       of a vararg. */
    if(n>mn) break;
  }
  }

  gen_free_list(fpvl);

  return tf;
}

transformer c_user_call_to_transformer(entity f,
				       list pc,
				       transformer pre,
				       list ef)
{
  transformer tf = transformer_undefined;
  transformer t_callee = load_summary_transformer(f);

  pips_assert("t_callee is consistent", transformer_weak_consistency_p(t_callee));

  /* Compute the call site transformer */
  tf = any_user_call_site_to_transformer(f, pc, pre, ef);

  /* Combine tf with the summary transformer */
  tf = transformer_combine(tf, t_callee);

  /* Project the former parameters and the temporary values. */
  tf = transformer_temporary_value_projection(tf);
  tf = transformer_formal_parameter_projection(f, tf);

  tf = transformer_filter_subsumed_variables(tf);

  return tf;
}

/* Effects are necessary to clean up the transformer t_caller. For
   instance, an effect on variable X may not be taken into account in
   t_callee but it may be equivalenced thru a common to a variable i which
   is analyzed in the caller. If X is written, I value is lost. See
   Validation/equiv02.f. */

transformer fortran_user_call_to_transformer(entity f,
					     list pc,
					     list ef)
{
  transformer t_callee = transformer_undefined;
  transformer t_caller = transformer_undefined;
  transformer t_effects = transformer_undefined;
  entity caller = entity_undefined;
  list all_args = list_undefined;

  pips_debug(8, "begin\n");

  /* add equations linking formal parameters to argument expressions
     to transformer t_callee and project along the formal parameters */
  /* for performance, it  would be better to avoid building formals
     and to inline entity_to_formal_parameters */
  /* it wouls also be useful to distinguish between in and out
     parameters; I'm not sure the information is really available
     in a field ??? */
  list formals = module_to_formal_analyzable_parameters(f);
  list formals_new = NIL;

  t_callee = load_summary_transformer(f);

  ifdebug(8) {
    Psysteme s =
      (Psysteme) predicate_system(transformer_relation(t_callee));
    pips_debug(8, "Transformer for callee %s:\n",
	       entity_local_name(f));
    dump_transformer(t_callee);
    sc_fprint(stderr, s, (char * (*)(Variable)) dump_value_name);
  }

  t_caller = transformer_dup(t_callee);

  /* take care of analyzable formal parameters */

  FOREACH(ENTITY, fp, formals) {
    int r = formal_offset(storage_formal(entity_storage(fp)));
    expression expr = find_ith_argument(pc, r);

    if(expr == expression_undefined)
      pips_user_error("not enough args for %d formal parm."
		      " %s in call to %s from %s\n",
		      r, entity_local_name(fp), entity_local_name(f),
		      get_current_module_entity());
    else {
      /* type checking. You already know that fp is a scalar variable */
      type tfp = entity_type(fp);
      basic bfp = variable_basic(type_variable(tfp));
      basic bexpr = basic_of_expression(expr);
      //list l_eff = expression_to_proper_effects(expr);
      list l_eff = expression_to_proper_constant_path_effects(expr);

      if(effects_write_at_least_once_p(l_eff)) {
	pips_user_warning("Side effects in actual arguments are not yet taken into account\n."
			  "Meanwhile, atomize the call site to avoid the problem.\n");
      }
      gen_free_list(l_eff);

      if(!basic_equal_p(bfp, bexpr)) {
	pips_user_warning("Type incompatibility\n(formal %s/actual %s)"
			  "\nfor formal parameter %s (rank %d)"
			  "\nin call to %s from %s\n",
			  basic_to_string(bfp), basic_to_string(bexpr),
			  entity_local_name(fp), r, module_local_name(f),
			  module_local_name(get_current_module_entity()));
	continue;
      }
    }

    if(entity_is_argument_p(fp, transformer_arguments(t_callee))) {
      /* formal parameter e is modified. expr must be a reference */
      syntax sexpr = expression_syntax(expr);

      if(syntax_reference_p(sexpr)) {
	entity ap = reference_variable(syntax_reference(sexpr));

	if(entity_has_values_p(ap)) {
	  Psysteme s = (Psysteme) predicate_system(transformer_relation(t_caller));
	  entity ap_new = entity_to_new_value(ap);
	  entity ap_old = entity_to_old_value(ap);

	  if(base_contains_variable_p(s->base, (Variable) ap_new)) {
	    pips_user_error(
			    "Variable %s seems to be aliased thru variable %s"
			    " at a call site to %s in %s\n"
			    "PIPS semantics analysis assumes no aliasing as"
			    " imposed by the Fortran standard.\n",
			    entity_name(fp),
			    entity_name(value_to_variable(ap_new)),
			    module_local_name(f),
			    get_current_module_name());
	  }
	  else { /* normal case: ap_new==fp_new, ap_old==fp_old */
	    entity fp_new = external_entity_to_new_value(fp);
	    entity fp_old = external_entity_to_old_value(fp);

	    t_caller = transformer_value_substitute
	      (t_caller, fp_new, ap_new);
	    t_caller = transformer_value_substitute
	      (t_caller, fp_old, ap_old);
	  }
	}
	else { /* Variable ap is not analyzed. The information about fp
		  will be lost. */
	  ;
	}
      }
      else {
	/* Attemps at modifying a value: expr is call, fp is modified */
	/* Actual argument is not a reference: it might be a user error!
	 * Transformers do not carry the may/must information.
	 * A check with effect list ef should be performed...
	 *
	 * FI: does effect computation emit a MUST/MAYwarning?
	 */
	entity fp_new = external_entity_to_new_value(fp);
	entity fp_old = external_entity_to_old_value(fp);
	list args = arguments_add_entity(arguments_add_entity(NIL, fp_new), fp_old);

	pips_user_warning("value (!) might be modified by call to %s\n"
			  "%dth formal parameter %s\n",
			  entity_local_name(f), r, entity_local_name(fp));
	t_caller = transformer_filter(t_caller, args);
	free_arguments(args);
      }
    }
    else {
      /* Formal parameter fp is not modified. Add fp == expr, if possible. */
      /* We should evaluate expr under a precondition pre... which has
	 not been passed down. We set pre==tf_undefined. */
      entity fp_new = external_entity_to_new_value(fp);
      transformer t_expr = any_expression_to_transformer(fp_new, expr,
							 transformer_undefined,
							 false);

      if(!transformer_undefined_p(t_expr)) {
	t_expr = transformer_temporary_value_projection(t_expr);
	/* temporary value counter cannot be reset because other
	   temporary values may be in use in a case the user call is a
	   user function call */
	/* reset_temporary_value_counter(); */
	t_caller = transformer_safe_image_intersection(t_caller, t_expr);
	free_transformer(t_expr);
      }
    }
  }

  pips_debug(8, "Before formal new values left over are eliminated\n");
  ifdebug(8)   dump_transformer(t_caller);

  /* formal new and old values left over are eliminated */
  FOREACH(ENTITY, e, formals) {
    entity e_new = external_entity_to_new_value(e);
    formals_new = CONS(ENTITY, e_new, formals_new);
    /* test to insure that entity_to_old_value exists */
    if(entity_is_argument_p(e_new,
			    transformer_arguments(t_caller))) {
      entity e_old = external_entity_to_old_value(e);
      formals_new = CONS(ENTITY, e_old, formals_new);
    }
  }

  t_caller = transformer_filter(t_caller, formals_new);

  free_arguments(formals_new);
  free_arguments(formals);

  ifdebug(8) {
    Psysteme s = predicate_system(transformer_relation(t_caller));
    pips_debug(8,
	       "After binding formal/real parameters and eliminating formals\n");
    dump_transformer(t_caller);
    sc_fprint(stderr, s, (char * (*)(Variable)) dump_value_name);
  }

  /* take care of global variables */
  caller = get_current_module_entity();
  translate_global_values(caller, t_caller);

  /* FI: are invisible variables taken care of by translate_global_values()?
   * Yes, now...
   * A variable may be invisible because its location is reached
   * thru an array or thru a non-integer scalar variable in the
   * current module, for instance because a COMMON is defined
   * differently. A variable whose location is not reachable
   * in the current module environment is considered visible.
   */

  ifdebug(8) {
    pips_debug(8, "After replacing global variables\n");
    dump_transformer(t_caller);
  }

  if(!transformer_empty_p(t_caller)) {
    /* Callee f may have read/write effects on caller's scalar
     * integer variables thru an array and/or non-integer variables.
     */
    t_effects = effects_to_transformer(ef);
    all_args = arguments_union(transformer_arguments(t_caller),
			       transformer_arguments(t_effects));
    /*
      free_transformer(t_effects);
      gen_free_list(transformer_arguments(t_caller));
    */
    transformer_arguments(t_caller) = all_args;
    /* The relation basis must be updated too */
    FOREACH(ENTITY, v, transformer_arguments(t_effects)) {
	Psysteme sc = (Psysteme) predicate_system(transformer_relation(t_caller));
	sc_base_add_variable(sc, (Variable) v);
      }
  }
  else {
    pips_user_warning("Call to %s seems never to return."
		      " This may be due to an infinite loop in %s,"
		      " or to a systematic exit in %s,"
		      " or to standard violations (see previous messages)\n",
		      module_local_name(f),
		      module_local_name(f),
		      module_local_name(f));
  }

  ifdebug(8) {
    pips_debug(8,
	       "End: after taking all scalar effects in consideration %p\n",
	       t_caller);
    dump_transformer(t_caller);
  }

  /* The return value of a function is not yet projected. */
  pips_assert("transformer t_caller is consistent",
	      transformer_weak_consistency_p(t_caller));

  return t_caller;
}

transformer user_call_to_transformer(entity f,
				     list pc,
				     transformer pre,
				     list ef)
{
  transformer t_caller = transformer_undefined;

  pips_debug(8, "begin\n");
  pips_assert("f is a module", entity_module_p(f));

  if(!get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
    t_caller = effects_to_transformer(ef);
  }
  else {
    if(c_module_p(f))
      t_caller = c_user_call_to_transformer(f, pc, pre, ef);
    else
      t_caller = fortran_user_call_to_transformer(f, pc, ef);
  }
  return t_caller;
}

/* FI: This transformer carry the information about the value
   returned, but not the fact that the next statement is not
   reached. */
transformer c_return_to_transformer(entity e __attribute__ ((__unused__)),
				    list pc, list ef, transformer pre)
{
  transformer tf = transformer_undefined;
  entity m = get_current_module_entity();
  const char* mn = entity_local_name(m);
  entity rv = FindEntity(mn, mn);

  if(ENDP(pc))
    tf = transformer_identity();
  else {
    type umt = ultimate_type(entity_type(m));
    type rt = functional_result(type_functional(umt));

    pips_assert("A module has a functional type", type_functional_p(umt));

    if(!type_void_p(rt)) {
      //type urt = ultimate_type(rt);
      //basic b = variable_basic(type_variable(urt));

      /* FI: Are we sure the return value entity has values? No... */
      /* if(entity_has_values_p(rv)) {*/
      if(analyzable_scalar_entity_p(rv)) {
	expression expr = EXPRESSION(CAR(pc));
	//entity rvv = entity_to_new_value(rv);

	tf = any_expression_to_transformer(rv, expr, pre, false);
	if(transformer_undefined_p(tf))
	  tf = effects_to_transformer(ef);
	else
	  tf = transformer_temporary_value_projection(tf);
	tf = transformer_add_value_update(tf, rv);
      }
    }
    else {
      pips_user_error("value returned from a void function\n");
    }

    if(transformer_undefined_p(tf))
      tf = effects_to_transformer(ef);
  }

  return tf;
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
transformer assigned_expression_to_transformer(entity v,
					       expression expr,
					       transformer pre)
{
  transformer tf = transformer_undefined;

  pips_debug(8, "begin\n");

  if(entity_has_values_p(v)) {
    entity v_new = entity_to_new_value(v);
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
	entity v_old = entity_to_old_value(v);
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

  pips_debug(8, "end with tf=%p\n", tf);

  return tf;
}

/* Always returns a fully defined transformer.
 *
 * FI: The property to compute transformers in context is not taken
 * into account to add information from pre within tf. pre is used to
 * evaluate expr, but is not made part of tf.
 */
transformer safe_assigned_expression_to_transformer(entity v,
						    expression expr,
						    transformer pre)
{
  transformer tf = transformer_undefined;

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

    // FI: need to investigate interplay between typedef and
    // qualifier?
    // FI: Issue: a static variable qualified with const and accessed thru
    // a function. See hs_list_smoothing() in hyantes
    // Or you see the issue as using assignment analysis for static definition...
    if(entity_has_values_p(v) && !type_with_const_qualifier_p(entity_type(v))) {
      tf = transformer_add_modified_variable_entity(tf, v);
    }
  }

  pips_assert("tf is defined", !transformer_undefined_p(tf));
  pips_assert("tf is consistent", transformer_consistency_p(tf));

  return tf;
}


/* This function never returns an undefined transformer. It is used
   for an assignment statement, not for an assignment operation. */
transformer integer_assign_to_transformer(expression lhs,
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

  transformer tf = transformer_undefined;
  normalized n = NORMALIZE_EXPRESSION(lhs);

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

  pips_debug(6,"return tf=%lx\n", (unsigned long)tf);
  ifdebug(6) (void) print_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}



/**
 * \brief           assign to the variable v the expression rhs
 *                  WARNING : this function can return transformer_undefined
 * \param v         entity/variable to be assign
 * \param rhs       expression/value to assign
 * \param pre       transformer already present
 * \return          transformer_undefined or transformer with the scalar_assign
 */
transformer any_scalar_assign_to_transformer_without_effect(entity v,
                                                       expression rhs,
                                                       transformer pre) /* precondition */
{
  transformer tf = transformer_undefined;

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

  return tf;
}

transformer any_scalar_assign_to_transformer(entity v,
    expression rhs,
    list ef, /* effects of assign */
    transformer pre) /* precondition */
{
  transformer tf = any_scalar_assign_to_transformer_without_effect(v, rhs, pre);

  if(transformer_undefined_p(tf))
    tf = effects_to_transformer(ef);

  return tf;
}

transformer any_assign_to_transformer(list args, /* arguments for assign */
				      list ef, /* effects of assign */
				      transformer pre) /* precondition */
{
  transformer tf = transformer_undefined;
  expression lhs = EXPRESSION(CAR(args));
  expression rhs = EXPRESSION(CAR(CDR(args)));
  syntax slhs = expression_syntax(lhs);

  pips_assert("2 args to assign", CDR(CDR(args))==NIL);

  /* The lhs must be a scalar reference to perform an interesting
     analysis in Fortran. In C, the condition can be relaxed to take
     into account side effects in sub-expressions. */
  if(syntax_reference_p(slhs)) {
    reference rlhs = syntax_reference(slhs);
    if(ENDP(reference_indices(rlhs))) {
      entity v = reference_variable(rlhs);
      tf = any_scalar_assign_to_transformer(v, rhs, ef, pre);
    }
    else {
      /* check scalar side effects in the subscript expressions and
	 in the rhs (specific to C) */
      // FI: I assume that the value is never useful because of the
      // above condition
      transformer st // subscript transformer
	= generic_reference_to_transformer(entity_undefined, rlhs, pre, false);
      /* FI: not clear why this happens in Fortran and not in C */
      if(!transformer_undefined_p(st)) {
	transformer post = transformer_apply(st, pre);
	transformer npre = transformer_range(post);
	transformer rt = safe_expression_to_transformer(rhs, npre); // rhs
	tf = transformer_combine(st, rt); // st is absorbed into tf
	free_transformer(rt);
	free_transformer(npre);
	free_transformer(post);
      }
    }
  }

  /* if some condition was not met and transformer derivation failed */
  if(tf==transformer_undefined) {
    transformer tf1 = safe_expression_to_transformer(lhs, pre);
    transformer tf2 = safe_expression_to_transformer(rhs, pre);
    tf = transformer_combine (tf1, tf2);
    free_transformer(tf2); // tf1 is exported in tf
    // FI: previous solution, only based on effects, was much safer
    // and simpler !
    tf = apply_effects_to_transformer(args, tf, ef, true);
  }

  pips_debug(6,"return tf=%p\n", tf);
  ifdebug(6) (void) print_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

transformer any_update_to_transformer(entity op,
				      list args, /* arguments for update */
				      list ef, /* effects of assign */
				      transformer pre) /* precondition */
{
  transformer tf = transformer_undefined;
  expression lhs = EXPRESSION(CAR(args));
  expression rhs = EXPRESSION(CAR(CDR(args)));
  syntax slhs = expression_syntax(lhs);

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

  pips_debug(6,"return tf=%p\n", tf);
  ifdebug(6) (void) print_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

transformer any_basic_update_to_transformer(entity op,
					    list args, /* arguments for update */
					    list ef, /* effects of assign */
					    transformer pre) /* precondition */
{
  transformer tf = transformer_undefined;
  expression lhs = EXPRESSION(CAR(args));
  syntax slhs = expression_syntax(lhs);

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

  pips_debug(6,"return tf=%p\n", tf);
  ifdebug(6) (void) print_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

static transformer instruction_to_transformer(instruction i,
					      transformer pre,
					      list e) /* effects associated to instruction i */
{
  transformer tf = transformer_undefined;
  test t;
  loop l;
  call c;
  whileloop wl;
  forloop fl;

  pips_debug(8,"begin\n");

  switch(instruction_tag(i)) {
  case is_instruction_block:
    tf = block_to_transformer(instruction_block(i), pre);
    break;
  case is_instruction_test:
    t = instruction_test(i);
    tf = test_to_transformer(t, pre, e);
    break;
  case is_instruction_loop:
    l = instruction_loop(i);
    tf = loop_to_transformer(l, pre, e);
    break;
  case is_instruction_whileloop: {
    wl = instruction_whileloop(i);
    tf = whileloop_to_transformer(wl, pre, e);
    break;
  }
  case is_instruction_forloop:
    fl = instruction_forloop(i);
    tf = forloop_to_transformer(fl, pre, e);
    break;
  case is_instruction_goto:
    pips_internal_error("unexpected goto in semantics analysis");
    tf = transformer_identity();
    break;
  case is_instruction_call:
    c = instruction_call(i);
    tf = call_to_transformer(c, pre, e);
    break;
  case is_instruction_unstructured:
    tf = unstructured_to_transformer(instruction_unstructured(i), pre, e);
    break ;
  case is_instruction_expression:
    tf = expression_to_transformer(instruction_expression(i), pre, e);
    break;
  default:
    pips_internal_error("unexpected tag %d",
	       instruction_tag(i));
  }
  pips_debug(9, "resultat:\n");
  ifdebug(9) (void) print_transformer(tf);
  pips_debug(8, "end\n");
  return tf;
}

/* Returns the effective transformer ct for a given statement s. t is
 * the stored transformer. For loops, t is useful to compute the body
 * preconditions but not to compute the loop postcondition. ct can be
 * used to compute the statement s postcondition, no matter what kind
 * of statement s is, and to compute the transformer of a higher-level
 * statement enclosing s.
 *
 * In other words, load_statement_transformer(s) does not always
 * return a transformer which can be composed with another transformer
 * or applied to a precondition. But statement_to_transformer() always
 * returns such a transformer.
 *
 * Always allocates a new transformer. This probably creates a memory
 * leak when going up the internal representation because it was
 * originally assumed that the transformer returned recursively was
 * also the transformer stored at a lower level. This is changed
 * because this function calls itself recursively. So now
 * statement_to_transformer() returns a transformer which is not the
 * transformer stored for the statement.
 */
transformer complete_statement_transformer(transformer t,
					   transformer pre,
					   statement s)
{
  return generic_complete_statement_transformer(t, pre, s, true);
}

/* FI: only implemented for while loops */
transformer complete_non_identity_statement_transformer(transformer t,
							transformer pre,
							statement s)
{
  return generic_complete_statement_transformer(t, pre, s, false);
}


/* Loops, do, for, while or repeat, have transformers linked to their
   body preconditions so as to compute those. But the real loop
   transformer includes also the possible loop skip and the possible
   loop exit. This function completes transformer t, which is linked
   to the loop body precondition, and use additional information
   carried by statement s, analyzed with precondition pre to return
   the global loop transformer. If statement s is not a loop, a copy
   of t is returned.

   Parameter identity_p is likely to be useless. It was added to track
   identity transformers before they were dealt with by transformer
   lists.
 */
transformer generic_complete_statement_transformer(transformer t,
						   transformer pre,
						   statement s,
						   bool identity_p)
{
  /* If i is a loop, the expected transformer can be more complex (see
     nga06) because the stores transformer is later used to compute
     the loop body precondition. It cannot take into account the exit
     condition. So the exit condition is added by the complete_xxx
     functions. */
  transformer ct = transformer_undefined;
  instruction i = statement_instruction(s);

  if(instruction_loop_p(i)) {
    /* likely memory leak:-(. ct should be allocated in both test
       branches and freed at call site but I program everything under
       the opposite assumption */
    /* The refined transformer may be lost or stored as a block
       transformer is the loop is directly surrounded by a bloc or used to
       compute the transformer of the surroundings blokcs */
    ct = complete_loop_transformer(t, pre, instruction_loop(i));
  }
  else if(instruction_whileloop_p(i)) {
    whileloop w = instruction_whileloop(i);
    evaluation e = whileloop_evaluation(w);
    // This test could be deported by a call to
    // complete_whileloop_transformer()
    if(evaluation_before_p(e)) {
      ct = new_complete_whileloop_transformer(t, pre, w, !identity_p);
    }
    else {
      ct = complete_repeatloop_transformer(t, pre, w);
    }
  }
  else if(instruction_forloop_p(i)) {
    ct = complete_forloop_transformer(t, pre, instruction_forloop(i));
  }
  else {
    if(identity_p) {
      /* No need to complete it */
      ct = copy_transformer(t);
    }
    else {
      /* The search for an non identity execution path must be propagated
	 downwards */
      if(instruction_sequence_p(i)) {
	/* Each component may or not update the state... */
      }
    }
  }
  return ct;
}

transformer statement_to_transformer(
				     statement s,
				     transformer spre) /* stmt precondition */
{
  instruction i = statement_instruction(s);
  list e = NIL;
  transformer t = transformer_undefined;
  transformer ot = transformer_undefined; /* old transformer for s */
  transformer nt = transformer_undefined; /* new transformer for s under spre */
  transformer te = transformer_undefined; /* nt updated with loop exit information */
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
      transformer_range(spre); // FI: transformer_range() implies
			       // projection(s) and a possibly strong
			       // increase of the number of
			       // constraints. Many constraints may be
			       // redundant as far as integer points
			       // are concerned, but not redundant for
			       // rational numbers. We could use an
			       // approximate projection, elimination
			       // of all constraints containing an old value...
    if(refine_transformers_p) {
      /* Transformation REFINE_TRANSFORMERS is being executed: add
         information available from the statement precondition */
      transformer srpre = load_statement_precondition(s);
      transformer srpre_r = transformer_range(srpre);

      pre = transformer_domain_intersection(pre, srpre_r);
      pre = transformer_normalize(pre, 2); // FI: redundancy
					       // elimination required
      free_transformer(srpre_r);
    }
  }
  else {
    // Avoid lots of test in the callees ot statement_to_transformer
    // pre = transformer_undefined;
    pre = transformer_identity();
  }

  pips_assert("pre is a consistent precondition",
	      transformer_consistent_p(pre));

  pips_debug(8,"Range precondition pre:\n");
  ifdebug(8) {
    (void) print_transformer(pre);
  }

  e = load_cumulated_rw_effects_list(s);
  ot = load_statement_transformer(s);

  /* it would be nicer to control warning_on_redefinition */
  if (transformer_undefined_p(ot)
      || get_bool_property("SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT")) {
    //list dl = declaration_statement_p(s) ? statement_declarations(s) : NIL;
    list dl = declaration_statement_p(s) ? statement_declarations(s) : NIL;

    /* FI: OK, we will have to switch to the new declaration
       representation some day, but the old representation is still
       fine.*/
    if(!ENDP(statement_declarations(s)) && !statement_block_p(s)
       && !declaration_statement_p(s)) {
      // FI: Just to gain some time before dealing with controlizer and declarations updates
      //pips_internal_error("Statement %p carries declarations");
      pips_user_warning("Statement %d (%p) carries declarations\n",
			statement_number(s), s);
    }

    if(!ENDP(dl)) {
      transformer dt = declarations_to_transformer(dl, pre);
      /* not very smart because declarations_to_transformer() computes post and free it...*/
      transformer post = transformer_apply(dt, pre);
      transformer ipre = transformer_range(post);
      transformer it = transformer_undefined;

      // FI: this should not be duplicated. Temporary fix for modulo07.c
      /* add type information */
      /*
      if(get_bool_property("SEMANTICS_USE_TYPE_INFORMATION")) {
	transformer_add_type_information(nt);
      }
      */

      ifdebug(8) {
	pips_debug(8, "Statement local preconditions due to declarations:");
	print_transformer(ipre);
      }

      /* FI: how do we want to handle declarations:
      *
      * int i = 1; => T() {i==1}
      *
      * or
      *
      * int i = 1; => T(i) {i==1}
      *
      * What is the impact of this choice? BC prefers the second one
      * because it it consistent for convex effect computation.
      *
      * Note: this issue could be dealt with earlier in
      * declarations_to_transformer()
      */
      /* FI: the code below might be useful again when declarations
	 are carried by any kind of statement */
      //it = instruction_to_transformer(i, ipre, e);
      //nt = transformer_combine(dt, it);
      //free_transformer(it);
      if(false) {
	/* Option 1 */
	nt = dt;
      }
      else if(false) {
	/* Option 2, currently bugged */
	/* Currently, the preconditions is useless as only the
	   effects will be used to compute the CONTINUE transformer. */
	it = instruction_to_transformer(i, pre, e);
	nt = transformer_image_intersection(it, dt);
	free_transformer(it);
	free_transformer(dt);
      }
      else {
	nt = dt; // Do nothing because everything has been taken care of by
	  // declaration_to_transformer()
      }
      free_transformer(post);
      // free_transformer(ipre);
    }
    else {
      nt = instruction_to_transformer(i, pre, e);
    }

    /* add array references information using proper effects */
    if(get_bool_property("SEMANTICS_TRUST_ARRAY_REFERENCES")) {
      transformer_add_reference_information(nt, s);
      /* nt = transformer_normalize(nt, 0); */
    }

    /* add type information */
    if(get_bool_property("SEMANTICS_USE_TYPE_INFORMATION")) {
      transformer_add_type_information(nt);
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
  
  /* Written abstract locations may require some information destruction 
   *
   * The current implementation is too crude. A new function is
   * needed, abstract_effects_to_transformer(). The current
   * implementation is OK for anywhere effects.
   */

  if(effects_abstract_location_p(e)) {
    transformer etf = effects_to_transformer(e);
    /* This is mathematically correct but very inefficient (see ticket
       644) and useless as long anymodule:anywhere is the only
       abstract effect we have to deal with.
     */
    t = transformer_combine(t, etf);
    free_transformer(etf);
    /* Not a sufficient solution:
    free_transformer(t);
    t = etf;
    */
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
  }
  else {
    pips_user_warning("redefinition for statement %03d (%d,%d)\n",
		 statement_number(s), ORDERING_NUMBER(statement_ordering(s)), 
		 ORDERING_STATEMENT(statement_ordering(s)));
    pips_internal_error("transformer redefinition");
  }

  ifdebug(1) {
    _int so = statement_ordering(s);
    transformer stf = load_statement_transformer(s);

    (void) fprintf(stderr, "statement %03td (%td,%td), transformer %p:\n",
		   statement_number(s),
		   ORDERING_NUMBER(so), ORDERING_STATEMENT(so),
		   stf);
    (void) print_transformer(stf);
    pips_assert("same pointer", stf==t);
  }

  /* The transformer returned for the statement is not always the
     transformer stored for the statement. This happens for loops and for
     context sensitive transformers for replicated statements in
     CFG/unstructured. See comments in loop.c */

  te = complete_statement_transformer(nt, pre, s);

  free_transformer(pre);

  ifdebug(8) {
    pips_assert("The statement and its substatements are still fully defined",
		all_statements_defined_p(s));
  }

  pips_debug(8,"end for statement %03td (%td,%td) with t=%p, nt=%p and te=%p\n",
	     statement_number(s), ORDERING_NUMBER(statement_ordering(s)), 
	     ORDERING_STATEMENT(statement_ordering(s)), t, nt, te);

  return te;
}
