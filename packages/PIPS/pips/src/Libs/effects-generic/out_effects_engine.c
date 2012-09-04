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
/* package generic effects :  Be'atrice Creusillet 6/97
 *
 * File: out_effects_engine.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the generic functions necessary for the computation of
 * all types of out effects.
 *
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"

#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"

#include "properties.h"

#include "transformer.h"
#include "pipsdbm.h"
#include "resources.h"
#include "preprocessor.h"

#include "effects-generic.h"

typedef struct {
  bool memory_effects_only;
  bool memory_in_out_effects_only;
  effects_representation_val representation;
} out_effects_context;


static void init_out_effects_context(out_effects_context *ctxt,
			     effects_representation_val representation)
{
  ctxt->memory_effects_only = get_bool_property("MEMORY_EFFECTS_ONLY");
  ctxt->memory_in_out_effects_only = get_bool_property("MEMORY_IN_OUT_EFFECTS_ONLY");

  if (!  ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    // if the MEMORY_EFFECTS_ONLY is FALSE on input, and
    // we don't want to compute OUT non-memory effects,
    // we have to temporarily set MEMORY_EFFECTS_ONLY to
    // TRUE so that functions computing R/W effects
    // do not compute non memory effects which would have to be filtered.
    set_bool_property("MEMORY_EFFECTS_ONLY", true);
  ctxt->representation = representation;
}

static void reset_out_effects_context(out_effects_context *ctxt)
{
  if (!  ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    // if the MEMORY_EFFECTS_ONLY is FALSE on input, and
    // we did not want to compute IN non-memory effects,
    // we have to reset MEMORY_EFFECTS_ONLY to FALSE.
    set_bool_property("MEMORY_EFFECTS_ONLY", false);
}


static list convert_rw_effects(list lrw, out_effects_context *ctxt)
{
  if (!ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    lrw = effects_store_effects(lrw);

  return lrw;
}


static void reset_converted_rw_effects(list *lrw, out_effects_context *ctxt)
{
  if (!ctxt->memory_effects_only
      && ctxt->memory_in_out_effects_only)
    gen_free_list(*lrw);
  *lrw = NIL;
}

/**************************** GENERIC INTERPROCEDURAL OUT EFFECTS ANALYSIS */


static statement current_stmt = statement_undefined;
static entity current_callee = entity_undefined;
static list l_sum_out_eff = list_undefined;

void
reset_out_summary_effects_list()
{
    l_sum_out_eff = list_undefined;
}

void
update_out_summary_effects_list(list l_out)
{
    if (list_undefined_p(l_sum_out_eff))
	l_sum_out_eff = l_out;
    else
	l_sum_out_eff = (*effects_test_union_op)(l_sum_out_eff, l_out,
						 effects_same_action_p);
}

list
get_out_summary_effects_list()
{
    return(l_sum_out_eff);
}

/* void out_effects_from_call_site_to_callee(call c)
 * input    : a potential call site for current_callee.
 * output   : nothing
 * modifies : l_sum_out_reg becomes the may union of l_sum_out_reg and
 *            the translated out regions of the current call site.
 * comment  :
 */
static void
out_effects_from_call_site_to_callee(call c)
{
    transformer context;
    list l_out = NIL, l_tmp = NIL;

    if (call_function(c) != current_callee)
	return;

    context = (*load_context_func)(current_stmt);
    l_out = load_out_effects_list(current_stmt);

    l_tmp = generic_effects_forward_translation(current_callee,
					      call_arguments(c), l_out,
					      context);

    set pre_out_set = set_make(set_pointer);
    FOREACH(EFFECT, eff, l_tmp)
      {
	entity e = reference_variable(effect_any_reference(eff));
	if (!entity_pointer_p(e))
	  set_add_element(pre_out_set, pre_out_set, e);
      }

    list l_sum_rw_eff = (*db_get_summary_rw_effects_func)(module_local_name(current_callee));

    /* It's necessary to take the intersection with the summary regions of the
   * callee to avoid problems due to multiple usages of the same actual
   * parameter for different formal ones :
   *
   *      <a(PHI1)-OUT-MUST-{PHI1==i}
   *      foo(a, a, i)
   *
   *      <tab1-R-MUST-{PHI1==i}>, <tab2-W-MUST-{PHI1==i}
   *      void foo(int tab1[], int tab2[], int i)
   *
   * Without the intersection, we would obtain :
   *
   *      <tab1-OUT-MUST-{PHI1==i}>, <tab2-OUT-MUST-{PHI1==i}
   */
    pips_debug_effects(2, "R/W effects : \n", l_sum_rw_eff);
    l_tmp = (*effects_intersection_op)(l_tmp, effects_dup(l_sum_rw_eff),
					  effects_same_action_p);
    pips_debug_effects(2, "l_tmp after intersection : \n", l_tmp);

    /*
      AM : Trying to handle Effects/missing_out.tpips
      effects WRITE <array[PHI1]-write-MAY-{0<=PHI1}> and "pre" OUT <array[PHI1][PHI2]-write-EXACT-{0<=PHI1, PHI1+1<=n, 0<=PHI2, PHI2+1<=m}>
      are not combinable (same entity but different dims). Over-approximate OUT by WRITE
    */
    set out_set = set_make(set_pointer);
    FOREACH(EFFECT, eff, l_tmp)
      {
	entity e = reference_variable(effect_any_reference(eff));
	set_add_element(out_set, out_set, e);
      }
    FOREACH(EFFECT, eff, l_sum_rw_eff)
      {
	if(action_write_p(effect_action(eff)))
	  {
	    entity e = reference_variable(effect_any_reference(eff));

	    if(!set_belong_p(out_set, e) && set_belong_p(pre_out_set, e))
	      {
		effect new = copy_effect(eff);
		effect_approximation_tag(new) = is_approximation_may;
		l_tmp = CONS(EFFECT, new, l_tmp);
	      }
	  }
      }
    set_free(pre_out_set);
    set_free(out_set);

    update_out_summary_effects_list(l_tmp);
}


static bool
summary_out_effects_stmt_filter(statement s)
{
    pips_debug(1, "statement %03zd\n", statement_number(s));
    current_stmt = s;
    return(true);
}


static list
out_effects_from_caller_to_callee(entity caller, entity callee)
{
    const char *caller_name;
    statement caller_statement;

    reset_current_module_entity();
    set_current_module_entity(caller);
    caller_name = module_local_name(caller);
    pips_debug(2, "begin for caller: %s\n", caller_name);

    /* All we need to perform the translation */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, caller_name, true) );
    caller_statement = get_current_module_statement();

    (*effects_computation_init_func)(caller_name);

    set_out_effects( (*db_get_out_effects_func)(caller_name));

    current_callee = callee;
    gen_multi_recurse(caller_statement,
		      statement_domain, summary_out_effects_stmt_filter, gen_null,
		      call_domain, out_effects_from_call_site_to_callee, gen_null,
		      NULL);

    reset_current_module_entity();
    set_current_module_entity(callee);
    reset_current_module_statement();
    reset_out_effects();

    (*effects_computation_reset_func)(caller_name);

    pips_debug(2, "end\n");
    return(l_sum_out_eff);
}



/* bool summary_out_effects_engine(const char* module_name)
 * input    : the name of the current module.
 * output   : the list of summary out effects
 * modifies : nothing.
 * comment  : computes the summary out effects of the current module, using the
 *            out effects at all its call sites.
 */
bool
summary_out_effects_engine(const char* module_name)
{

    list l_eff = NIL;


    /* Look for all call sites in the callers */
    callees callers = (callees) db_get_memory_resource(DBR_CALLERS,
						       module_name,
						       true);
    entity callee = module_name_to_entity(module_name);


    set_current_module_entity(callee);
    make_effects_private_current_context_stack();

    debug_on("OUT_EFFECTS_DEBUG_LEVEL");
    ifdebug(1)
    {
	pips_debug(1, "begin for %s with %td callers\n",
	      module_name,
	      gen_length(callees_callees(callers)));
	MAP(STRING, caller_name, {fprintf(stderr, "%s, ", caller_name);},
	    callees_callees(callers));
	fprintf(stderr, "\n");
    }

    reset_out_summary_effects_list();
    MAP(STRING, caller_name,
    {
	entity caller = module_name_to_entity(caller_name);
	(void)out_effects_from_caller_to_callee(caller,callee);
    },
	callees_callees(callers));

    l_eff = get_out_summary_effects_list();
    if (list_undefined_p(l_eff))
	l_eff = NIL;

    (*db_put_summary_out_effects_func)(module_name, l_eff);

    ifdebug(1)
    {
      set_current_module_statement( (statement)
				    db_get_memory_resource(DBR_CODE, module_local_name(callee), true) );

	pips_debug(1, "summary out_effects for module %s:\n", module_name);
	(*effects_prettyprint_func)(l_eff);
	pips_debug(1, "end\n");
	reset_current_module_statement();
    }

    reset_current_module_entity();
    free_effects_private_current_context_stack();

    debug_off();
    return true;
}



/***************************** GENERIC INTRAPROCEDURAL OUT EFFECTS ANALYSIS */

/* void out_regions_from_unstructured_to_nodes(unstructured u,
 *                                             transformer c_trans, list l_out)
 * input    : an unstructured piece of code, the corresponding transformer, and
 *            the corresponding out regions.
 * output   : nothing.
 * modifies : nothing.
 * comment  : computes the out regions of each node of the unstructured.
 *     For a one node unstructured, it is trivial.
 *     In the usual case, If OUT_0 is the out region for the whole structure,
 *     IN_0 the in region for the whole structure and W_i the write region
 *     at node i, then OUT_i = (OUT_0 union IN_0) inter W_i. And the result
 *     is a may region.
 */
static bool
out_effects_from_unstructured_to_nodes(unstructured u, out_effects_context *ctxt)
{
    statement unst_stat = effects_private_current_stmt_head();
    list blocs = NIL;
    control ct;
    list
	l_out_unst, /* OUT effects of the unstructured */
	l_in;       /* IN effects of the unstructured */

    pips_debug(1, "begin\n");

    /* First, we get the out regions of the statement corresponding to the
     * unstructured.
     */
    l_out_unst = effects_dup(load_out_effects_list(unst_stat));

    ct = unstructured_control( u );

    if(control_predecessors(ct) == NIL && control_successors(ct) == NIL)
    {
	/* there is only one statement in u; */
	pips_debug(6,"unique node\n");
	store_out_effects_list(control_statement(ct), l_out_unst);
    }
    else
    {
	transformer unst_trans = (*load_transformer_func)(unst_stat);
	(*effects_transformer_inverse_composition_op)(l_out_unst, unst_trans);

	l_in = load_in_effects_list(unst_stat);
	l_out_unst = (*effects_union_op)(l_out_unst, effects_dup(l_in),
					 w_r_combinable_p);
	effects_to_write_effects(l_out_unst);

	CONTROL_MAP(c,
	{
	    statement node_stat = control_statement(c);
	    transformer node_prec = (*load_context_func)(node_stat);
	    list l_out_node = effects_dup(l_out_unst);
	    list l_rw = convert_rw_effects(load_rw_effects_list(node_stat), ctxt);
	    list l_w_node =
		effects_write_effects_dup(l_rw);
	    reset_converted_rw_effects(&l_rw, ctxt);

	    l_out_node = (*effects_precondition_composition_op)
		(l_out_node, node_prec);
	    l_out_node = (*effects_intersection_op)
		(l_out_node, l_w_node, w_w_combinable_p);
	    effects_to_may_effects(l_out_node);
	    store_out_effects_list(node_stat, l_out_node);
	    /* effects_free(l_w_node); */ /* ??? seems not a good idea... FC */

	}, ct, blocs) ;

	gen_free_list(blocs) ;
    }

    effects_free(l_out_unst);

    debug(1,"out_regions_from_unstructured_to_nodes","end\n");
    return(true);
}


/* void out_regions_from_loop_to_body(loop l, transformer c_trans, list l_out)
 * input    : a loop, its transformer, and its out regions.
 * output   : nothing.
 * modifies : nothing.
 * comment  : computes the out regions of the loop body. See report E/185
 *            for more details.
 */
static bool
out_effects_from_loop_to_body(loop l, out_effects_context *ctxt)
{
    statement
	loop_stat = effects_private_current_stmt_head(),
	body_stat = loop_body(l);
    list
	l_out_loop,
	l_body_out = NIL,
	l_body_w = NIL;
    entity i = loop_index(l);
    range r = loop_range(l);
    descriptor range_descriptor = descriptor_undefined;
    transformer loop_trans = (*load_transformer_func)(loop_stat);
    transformer loop_proper_context = transformer_undefined;

    pips_debug(1, "begin\n");
    pips_debug(1, "loop index %s.\n", entity_minimal_name(i));

    /* OUT effects of loop */
    l_out_loop = effects_dup(load_out_effects_list(loop_stat));
    ifdebug(1){
	debug(1,"","OUT_L = \n");
	(*effects_prettyprint_func)(l_out_loop);
    }

    /* Write effects of loop body */
    list l_rw = convert_rw_effects(load_rw_effects_list(body_stat), ctxt);
    l_body_w = effects_write_effects_dup(l_rw);
    reset_converted_rw_effects(&l_rw, ctxt);
    ifdebug(1){
	debug(1,"","W(i) = \n");
	(*effects_prettyprint_func)(l_body_w);
    }

    /* l_out = append(T_B(l_out), loop_proper_preconditions) */
    l_out_loop =
	(*effects_transformer_inverse_composition_op)(l_out_loop, loop_trans);
    range_descriptor = (*loop_descriptor_make_func)(l);
    if(descriptor_defined_p(range_descriptor))
      loop_proper_context = descriptor_to_context(range_descriptor);
    l_out_loop =
	(*effects_precondition_composition_op)(l_out_loop, loop_proper_context);

    if (! normalizable_and_linear_loop_p(i, r) || ! get_descriptor_range_p())
    {
	l_body_out =
	    (*effects_intersection_op)(l_out_loop, l_body_w, w_w_combinable_p);
	array_effects_to_may_effects(l_body_out);
    }
    else
    {
	expression e_incr = range_increment(loop_range(l));
	Value incr = vect_coeff
	    (TCST, (Pvecteur) normalized_linear(NORMALIZE_EXPRESSION(e_incr)));
	list
	    l_body_in = load_cumulated_in_effects_list(body_stat),
	    l_next_w,
	    l_prev_w,
	    l_next_in,
	    l_out_glob,
	    l_out_loc;
	add_intermediate_value(i);
	add_old_value(i);
	entity
	    i_prime = entity_to_intermediate_value(i),
	    i_d_prime = entity_to_old_value(i);
	Pvecteur
	    i_prime_gt_i,
	    i_d_prime_gt_i,
	    i_prime_gt_i_d_prime;
	descriptor
	    d_i_i_prime,
	    d_i_i_prime_i_d_prime;
	transformer
	    c_i_i_prime,
	    c_i_i_prime_i_d_prime;

	/* we will need to add several systems of constraints to the effects:
	 * d_i_i_prime = {i'> i, lb<=i<=ub}
	 * d_i_i_prime_i_d_prime = {i<i''<i', lb<=i<=ub, lb<=i'<=ub} */
	i_prime_gt_i =
	    vect_make(
		VECTEUR_NUL,
		(Variable) (value_one_p(incr)? i : i_prime), VALUE_ONE,
		(Variable) (value_one_p(incr)? i_prime : i), VALUE_MONE,
		TCST, VALUE_ONE);
	d_i_i_prime = copy_descriptor(range_descriptor);
	d_i_i_prime = descriptor_inequality_add(d_i_i_prime, i_prime_gt_i);
	c_i_i_prime = descriptor_to_context(d_i_i_prime);
	free_descriptor(d_i_i_prime);


	i_d_prime_gt_i =
	    vect_make(
		VECTEUR_NUL,
		(Variable) (value_one_p(incr)? i : i_d_prime), VALUE_ONE,
		(Variable) (value_one_p(incr)? i_d_prime : i), VALUE_MONE,
		TCST, VALUE_ONE);
	i_prime_gt_i_d_prime =
	    vect_make(
		VECTEUR_NUL,
		(Variable) (value_one_p(incr)? i_d_prime : i_prime), VALUE_ONE,
		(Variable) (value_one_p(incr)? i_prime : i_d_prime), VALUE_MONE,
		TCST, VALUE_ONE);
	d_i_i_prime_i_d_prime = copy_descriptor(range_descriptor);
	descriptor_variable_rename(d_i_i_prime_i_d_prime, i, i_prime);
	d_i_i_prime_i_d_prime =
	    descriptor_append(d_i_i_prime_i_d_prime, range_descriptor);
	d_i_i_prime_i_d_prime =
	    descriptor_inequality_add(d_i_i_prime_i_d_prime, i_d_prime_gt_i);
	d_i_i_prime_i_d_prime =
	    descriptor_inequality_add(d_i_i_prime_i_d_prime, i_prime_gt_i_d_prime);
	c_i_i_prime_i_d_prime = descriptor_to_context(d_i_i_prime_i_d_prime);
	free_descriptor(d_i_i_prime_i_d_prime);


	/* l_next_w = proj_i'(W(i', i'>i)) */
	l_next_w = effects_dup(l_body_w);
	l_next_w =
	    (*effects_descriptors_variable_change_func)(l_next_w, i, i_prime);
	l_next_w = (*effects_precondition_composition_op)(l_next_w, c_i_i_prime);

	ifdebug(1){
	    debug(1,"","W(i', i'>i) = \n");
	    (*effects_prettyprint_func)(l_next_w);
	}
	(*effects_union_over_range_op)(l_next_w, i_prime, r, descriptor_undefined);
	ifdebug(1){
	    debug(1,"","proj_i'(W(i', i'>i)) = \n");
	    (*effects_prettyprint_func)(l_next_w);
	}

	/* l_out_glob = ( W(i) inter l_out_loop) -_{sup} l_next_w */
	l_out_glob = (*effects_sup_difference_op)
	    ((*effects_intersection_op)(effects_dup(l_body_w),
					l_out_loop,
					w_w_combinable_p),
	     l_next_w,
	     w_w_combinable_p);

	ifdebug(1){
	    debug(1,"","l_out_glob = \n");
	    (*effects_prettyprint_func)(l_out_glob);
	}

	/* l_prev_w = proj_i''(W(i'', i<i''<i')) */
	l_prev_w = effects_dup(l_body_w);
	l_prev_w =
	    (*effects_descriptors_variable_change_func)(l_prev_w, i, i_d_prime);
	l_prev_w =
	    (*effects_precondition_composition_op)(l_prev_w, c_i_i_prime_i_d_prime);

	ifdebug(1){
	    debug(1,"","W(i'', i<i''<i') = \n");
	    (*effects_prettyprint_func)(l_prev_w);
	}

	l_prev_w = (*effects_union_over_range_op)
	    (l_prev_w, i_d_prime, r, descriptor_undefined);

	ifdebug(1){
	    debug(1,"","proj_i''(W(i'', i<i''<i')) = \n");
	    (*effects_prettyprint_func)(l_prev_w);
	}

	/* l_next_in = proj_i'( IN(i', i'>i) -_{sup} l_prev_w) */
	l_next_in = effects_dup(l_body_in);
	l_next_in =
	    (*effects_descriptors_variable_change_func)(l_next_in, i, i_prime);
	l_next_in = (*effects_precondition_composition_op)(l_next_in, c_i_i_prime);

	ifdebug(1){
	    debug(1,"","IN(i', i<i') = \n");
	    (*effects_prettyprint_func)(l_next_in);
	}
	l_next_in = (*effects_sup_difference_op)
	    (l_next_in, l_prev_w, r_w_combinable_p);

	ifdebug(1){
	    debug(1,"","IN(i', i<i') - proj_i''(W(i'')) = \n");
	    (*effects_prettyprint_func)(l_next_in);
	}
	l_next_in = (*effects_union_over_range_op)
	    (l_next_in, i_prime, r, descriptor_undefined);

	/* l_out_loc = W(i) inter l_next_in */
	l_out_loc = (*effects_intersection_op)(effects_dup(l_body_w),
					       l_next_in,
					       w_r_combinable_p);

	ifdebug(1){
	    debug(1,"","l_out_loc = \n");
	    (*effects_prettyprint_func)(l_out_loc);
	}

	/* l_body_out = l_out_glob Umust l_out_loc */
	l_body_out = (*effects_union_op)(l_out_glob, l_out_loc, w_w_combinable_p);
    }

    store_out_effects_list(body_stat, l_body_out);
    return(true);
}


/* static bool out_effects_from_test_to_branches(test t)
 * input    : a test.
 * output   : the true boolean.
 * modifies : .
 * comment  : computes the out regions of each branch of the test.
 */
static bool
out_effects_from_test_to_branches(test t, out_effects_context *ctxt)
{
    statement
	test_stat = effects_private_current_stmt_head(),
	branche;
    list
	l_out_test,
	l_out_branche = NIL,
	l_w_branche;
    transformer prec_branche;
    int i;

    pips_debug(1,"begin\n");

    /* First, we get the out regions of the statement corresponding to the test
     */
    l_out_test = load_out_effects_list(test_stat);

    /* Then we compute the out regions for each branch */
    for(i=1; i<=2; i++)
    {
	branche = (i==1)? test_true(t) : test_false(t);
	list l_rw_branche = convert_rw_effects(load_rw_effects_list(branche), ctxt);
	l_w_branche = effects_write_effects_dup(l_rw_branche);
	reset_converted_rw_effects(&l_rw_branche, ctxt);
	prec_branche = (*load_context_func)(branche);

	l_out_branche = effects_dup(l_out_test);
	l_out_branche = (*effects_precondition_composition_op)
	    (l_out_branche, prec_branche);
	l_out_branche = (*effects_intersection_op)
	    (l_out_branche, l_w_branche, w_w_combinable_p);
	store_out_effects_list(branche, l_out_branche);
    }

    pips_debug(1,"end\n");
    return(true);
}

/* Rout[s in for(c)s] = Rw[s] * MAY ?
 */
static bool out_effects_from_for_to_body(forloop f, out_effects_context *ctxt)
{
  statement body;
  list /* of effect */ lout;

  body = forloop_body(f);
  list l_rw = convert_rw_effects(load_rw_effects_list(body), ctxt);
  lout = effects_write_effects_dup(l_rw);
  reset_converted_rw_effects(&l_rw, ctxt);
  MAP(EFFECT, e,
      approximation_tag(effect_approximation(e)) = is_approximation_may,
      lout);
  store_out_effects_list(body, lout);

  return true;
}

/* Rout[s in while(c)s] = Rw[s] * MAY ?
 */
static bool out_effects_from_while_to_body(whileloop w, out_effects_context *ctxt)
{
  statement body;
  list /* of effect */ lout;

  body = whileloop_body(w);
  list l_rw = convert_rw_effects(load_rw_effects_list(body), ctxt);
  lout = effects_write_effects_dup(l_rw);
  reset_converted_rw_effects(&l_rw, ctxt);
  MAP(EFFECT, e,
      approximation_tag(effect_approximation(e)) = is_approximation_may,
      lout);
  store_out_effects_list(body, lout);

  return true;
}

/* void out_regions_from_block_to_statements(list l_stat, list l_out, ctrans)
 * input    : a list of statement contituting a linear sequence, the
 *            correponding out regions and the transformer of the block.
 * output   : nothing.
 * modifies : nothing.
 * comment  : computes the out regions of each statement in the sequence.
 *            uses the algorithm described in report E/185.
 */
static bool
out_effects_from_block_to_statements(sequence seq, out_effects_context *ctxt)
{
    statement seq_stat = effects_private_current_stmt_head();
    list l_out_seq = load_out_effects_list(seq_stat);
    list l_stat = sequence_statements(seq);


    ifdebug(1){
	pips_debug(1,"begin\n");
	pips_debug(1,"OUT effects of the current block :\n");
	(*effects_prettyprint_func)(l_out_seq);
    }

    if (ENDP(l_stat))
    {
	/* empty block of statements. Nothing to do. */
	if (get_bool_property("WARN_ABOUT_EMPTY_SEQUENCES"))
	    pips_user_warning("empty sequence\n");
	return true;
    }

    if (gen_length(l_stat) == 1)
    {
	/* if there is only one instruction in the block of statement,
         * its out effects are the effects of the block */

	ifdebug(1)
	{
	    pips_debug(1,"only one statement\n");
	}

	store_out_effects_list(STATEMENT(CAR(l_stat)), effects_dup(l_out_seq));
    }
    else
    {
	list l_stat_reverse = gen_nreverse(gen_copy_seq(l_stat));
	list l_next_stat_w_eff = NIL;
	list l_out_prime = NIL;
	list l_out_stat = NIL;
	list l_in_prime = NIL;
	list l_tmp = NIL;
	// FI: no need for a completed transformer when dealing with a
	// sequence
	transformer seq_trans = (*load_transformer_func)(seq_stat);

	l_out_seq = effects_dup(l_out_seq);

	/* OUT'_{n+1} = T_B(OUT(B)) */
	(*effects_transformer_inverse_composition_op)(l_out_seq, seq_trans);
	l_out_prime = l_out_seq;

	ifdebug(1)
	{
	    pips_debug(1,"OUT effects of block after translation into store"
		       " after block:\n");
	    (*effects_prettyprint_func)(l_out_prime);
	}

	/* We go through each statement (S_k) in reverse order */
	MAP(STATEMENT, c_stat,
	{
	    transformer c_stat_trans = (*load_completed_transformer_func)(c_stat);
	    list l_c_stat_rw_eff = convert_rw_effects(load_rw_effects_list(c_stat), ctxt);
	    list l_c_stat_w_eff = effects_write_effects(l_c_stat_rw_eff);
	    reset_converted_rw_effects(&l_c_stat_rw_eff, ctxt);

	    ifdebug(1){
		pips_debug(1,"intermediate effects\n");
		debug(1,"","W(k) = \n");
		(*effects_prettyprint_func)(l_c_stat_w_eff);
		debug(1,"","W(k+1) = \n");
		(*effects_prettyprint_func)(l_next_stat_w_eff);
	    }

	    /* OUT'_k = T_k^{-1} [ OUT'_{k+1} -_{sup} W_{k+1} ] */
	    l_out_prime =
		(*effects_sup_difference_op)(l_out_prime,
					     effects_dup(l_next_stat_w_eff),
					     w_w_combinable_p);
	    (*effects_transformer_composition_op)(l_out_prime, c_stat_trans);

	    ifdebug(1){
		debug(1,"","OUT'_k = \n");
		(*effects_prettyprint_func)(l_out_prime);
	    }

	    /* OUT_k = W_k inter [ OUT'_k Umust T_k^{-1} (IN'_{k+1})] */
	    ifdebug(1){
		debug(1,"","IN'_(k+1) = \n");
		(*effects_prettyprint_func)(l_in_prime);
	    }
	    (*effects_transformer_composition_op)(l_in_prime, c_stat_trans);

	    ifdebug(1){
		debug(1,"","IN'_(k+1) =  "
		      "(after elimination of modified variables)\n");
		(*effects_prettyprint_func)(l_in_prime);
	    }

	    l_tmp = (*effects_union_op)(effects_dup(l_out_prime),
					l_in_prime, w_w_combinable_p);

	    ifdebug(1){
		debug(1,"","OUT'_k Umust IN'_(k+1) = \n");
		(*effects_prettyprint_func)(l_tmp);
	    }

	     l_out_stat = (*effects_intersection_op)(effects_dup(l_c_stat_w_eff),
						     l_tmp,
						     w_w_combinable_p);

	     ifdebug(1){
		 debug(1,"","OUT_k = \n");
		 (*effects_prettyprint_func)(l_out_stat);
	     }

	     /* store the out effects of the current statement */
	     store_out_effects_list(c_stat, l_out_stat);

	     /* keep some information about the current statement, which will be
	      * the next statement. */
	     l_next_stat_w_eff = l_c_stat_w_eff;
	     l_in_prime = effects_dup(load_cumulated_in_effects_list(c_stat));
	     effects_to_write_effects(l_in_prime);
	},
	    l_stat_reverse);

    }
    pips_debug(1,"end\n");
    return(true);
}


static bool
out_effects_statement_filter(statement s, out_effects_context *ctxt)
{
    pips_debug(1, "Entering statement %03zd :\n", statement_ordering(s));
    effects_private_current_stmt_push(s);
    return true;
}

static void
out_effects_statement_leave(statement s, out_effects_context *ctxt)
{
    effects_private_current_stmt_pop();
    pips_debug(1, "End statement %03zd :\n", statement_ordering(s));
}


static void
out_effects_of_module_statement(statement module_stat, out_effects_context *ctxt)
{

    make_effects_private_current_stmt_stack();
    pips_debug(1,"begin\n");

    gen_context_multi_recurse(
			      module_stat,ctxt,
  statement_domain, out_effects_statement_filter, out_effects_statement_leave,
      sequence_domain, out_effects_from_block_to_statements, gen_null,
      test_domain, out_effects_from_test_to_branches, gen_null,
      loop_domain, out_effects_from_loop_to_body, gen_null,
      whileloop_domain, out_effects_from_while_to_body, gen_null,
      forloop_domain, out_effects_from_for_to_body, gen_null,
       unstructured_domain, out_effects_from_unstructured_to_nodes, gen_null,
      call_domain, gen_false, gen_null, /* calls are treated in another phase*/
      NULL);

    pips_debug(1,"end\n");
    free_effects_private_current_stmt_stack();

}


/* bool out_effects_engine(const char* module_name):
 * input    : the name of the current module.
 * requires : that transformers and precondition maps be set if needed.
 *            (it depends on the chosen instanciation of *load_context_func
 *             and *load_transformer_func).
 * output   : nothing !
 * modifies :
 * comment  : computes the out effects of the current module.
 */
bool
out_effects_engine(const char* module_name, effects_representation_val representation)
{

    statement module_stat;
    list l_sum_out = NIL;

    debug_on("OUT_EFFECTS_DEBUG_LEVEL");
    debug(1, "out_effects", "begin\n");

    make_effects_private_current_context_stack();

    set_current_module_entity(module_name_to_entity(module_name));

    /* Get the code of the module. */
    set_current_module_statement( (statement)
	db_get_memory_resource(DBR_CODE, module_name, true) );
    module_stat = get_current_module_statement();

    (*effects_computation_init_func)(module_name);

    /* Get the various effects and in_effects of the module. */

    set_rw_effects((*db_get_rw_effects_func)(module_name));
    set_invariant_rw_effects((*db_get_invariant_rw_effects_func)(module_name));

    set_cumulated_in_effects((*db_get_cumulated_in_effects_func)(module_name));
    set_in_effects((*db_get_in_effects_func)(module_name));
    set_invariant_in_effects((*db_get_invariant_in_effects_func)(module_name));

    /* initialise the map for out effects */
    init_out_effects();

    /* Get the out_summary_effects of the current module */
    l_sum_out = (*db_get_summary_out_effects_func)(module_name);

    /* And stores them as the out regions of the module statement */
    store_out_effects_list(module_stat, effects_dup(l_sum_out));

    /* Compute the out_effects of the module. */
    out_effects_context ctxt;
    init_out_effects_context(&ctxt, representation);
    out_effects_of_module_statement(module_stat, &ctxt);
    reset_out_effects_context(&ctxt);

    pips_debug(1, "end\n");

    debug_off();

    (*db_put_out_effects_func)(module_name, get_out_effects());

    reset_current_module_entity();
    reset_current_module_statement();

    reset_rw_effects();
    reset_invariant_rw_effects();
    reset_in_effects();
    reset_cumulated_in_effects();
    reset_invariant_in_effects();
    reset_out_effects();
    (*effects_computation_reset_func)(module_name);

    free_effects_private_current_context_stack();

    return true;
}
