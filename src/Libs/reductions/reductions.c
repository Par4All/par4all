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
 * detection of simple reductions.
 * debug driven by REDUCTIONS_DEBUG_LEVEL
 *
 * FC, June 1996.
 */

#include "local-header.h"
#include "control.h"      /* for CONTROL_MAP() */
#include "semantics.h"    /* for load_summary_effects() */

/******************************************** SETTINGS IN GENERIC EFFECTS */

static void
set_generic_effects_as_needed(void)
{
    effect_dup_func = simple_effect_dup;
}

/****************************************************** SUMMARY REDUCTIONS */

/* Fortran 77 anti aliasing rules implies that sg that
 * looks like a reduction within a subroutine can be perceived as so
 * from outside because no aliasing may cause the accumulator to be
 * linked to the rhs of the accumulation...
 * thus summary reductions can be propagated with no harm...
 * just the usual conditions must be checked (no other effect on the variable)
 */
reductions load_summary_reductions(entity f)
{
    pips_assert("is a module", entity_module_p(f));
    return (reductions) db_get_memory_resource
	(DBR_SUMMARY_REDUCTIONS, module_local_name(f), true);
}

static reduction
compute_one_summary_reduction(reduction model, list /* of effect */ le)
{
  reduction r = copy_reduction(model);

  /* keep the entities that are exported... */
  FOREACH(ENTITY, e,reduction_dependences(model)) {
    if (!effects_may_read_or_write_memory_paths_from_entity_p(le,e))
	    remove_variable_from_reduction(r, e);
  }

  gen_free_list(reduction_dependences(r));
  reduction_dependences(r) = NIL;

  DEBUG_REDUCTION(3, "result\n", r);
  return r;
}

static reductions
compute_summary_reductions(entity f)
{
    list /* of effect */ le = load_summary_effects(f);
    list /* of reduction */ lr = NIL, lc;

    lc = reductions_list
	(load_cumulated_reductions(get_current_module_statement()));

    pips_debug(3, "module %s: %td cumulated reductions\n",
	       entity_name(f), gen_length(lc));

    FOREACH(REDUCTION, r,lc) {
      DEBUG_REDUCTION(4, "considering\n", r);
      if (effects_may_read_or_write_memory_paths_from_entity_p(le, reduction_variable(r)))
        lr = CONS(REDUCTION, compute_one_summary_reduction(r, le), lr);
    }

    return make_reductions(lr);
}

/* handler for pipsmake
 * input: module name
 * output: TRUE
 * side effects: stores the summary reductions to pipsdbm
 */
bool summary_reductions(const char* module_name)
{
    reductions red;

    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module_name);

    set_generic_effects_as_needed();
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, true));
    set_cumulated_reductions((pstatement_reductions)
        db_get_memory_resource(DBR_CUMULATED_REDUCTIONS, module_name, true));

    red = compute_summary_reductions(get_current_module_entity());

    DB_PUT_MEMORY_RESOURCE(DBR_SUMMARY_REDUCTIONS, module_name, red);

    reset_cumulated_reductions();
    reset_current_module_statement();
    reset_current_module_entity();
    generic_effects_reset_all_methods();

    debug_off();
    return true;
}

/******************************************************* PROPER REDUCTIONS */

/* Function storing Proper Reductions
 */
GENERIC_GLOBAL_FUNCTION(proper_reductions, pstatement_reductions)

/************************************************ LIST OF REDUCED ENTITIES */

/* list of entities that may be reduced
 */
static list /* of entity */
add_reduced_variables(
    list /* of entity */ le,
    reductions rs)
{
  FOREACH (REDUCTION, r, reductions_list(rs)) {
    le = gen_once(reduction_variable(r), le);
  }
  return le;
}

static list /* of entity */
list_of_reduced_variables(
    statement node,
    list /* of statement */ls)
{
    list /* of entity */ le = NIL;
    le = add_reduced_variables(le, load_proper_reductions(node));
    FOREACH (STATEMENT, s, ls) {
      if (bound_cumulated_reductions_p(s) )
	le = add_reduced_variables(le, load_cumulated_reductions(s));
      else {
	pips_debug(5, "stat %s %p\n", note_for_statement(s), s);
	pips_assert ("should not happen, all statements should have been visited for reduction", false);
      }
    }
    return le;
}

/*********************************************************** CHECK PROPERS */
/* Returns NULL if not ok */
static reduction compatible_reduction_of_var(entity var, reductions rs) {
  reduction rnew = make_none_reduction(var);

  FOREACH(REDUCTION, r,reductions_list(rs)) {
    if (!update_compatible_reduction_with(&rnew, var, r)) {
      free_reduction(rnew);
      return NULL;
    }
  }
  return rnew;
}

/* returns NIL on any problem */
static list list_of_compatible_reductions(reductions rs) {
  list lnr=NIL, le = add_reduced_variables(NIL, rs);

  FOREACH(ENTITY, var,le) {
    reduction r = compatible_reduction_of_var(var, rs);
    if (r) {
      lnr = CONS(REDUCTION, r, lnr);
    } else {
      gen_free_list(le);
      gen_full_free_list(lnr);
      return NIL;
    }
  }
  gen_free_list(le);

  return lnr;
}

static list list_of_trusted_references(reductions rs) {
  list lr = NIL;
  FOREACH(REDUCTION, r,reductions_list(rs)) {
    FOREACH(PREFERENCE, p,reduction_trusted(r)) {
      lr = CONS(REFERENCE, preference_reference(p), lr);
    }
    lr = CONS(REFERENCE, reduction_reference(r), lr); /* ??? */
  }
  return lr;
}

/* argh... what about side effect related reductions ???
 * There are no relevant pointer to trust in such a case...
 * What I can do as a (temporary) fix is not to check
 * direct side effects (that is "call foo" ones) because
 * they do not need to be checked...
 */
static bool safe_effects_for_reductions(statement s, reductions rs) {
  list /* of effect */ le = effects_effects(load_proper_references(s)),
      /* of reference */ lr = list_of_trusted_references(rs);

  FOREACH(EFFECT, e,le) {
    if ((effect_write_p(e) && store_effect_p(e) && !gen_in_list_p(effect_any_reference(e), lr)) ||
        io_effect_entity_p(effect_variable(e)))	{
      pips_debug(8, "effect on %s (ref %p) not trusted\n",
                 entity_name(effect_variable(e)),
                 effect_any_reference(e));

      gen_free_list(lr);
      return false;
    }
  }

  gen_free_list(lr);
  return true;
}

/* must check that the found reductions are
 * (1) without side effects (no W on any other than accumulators),
 *     MA: why? If the W doesn't conflict with the accumulators it should be
 *     safe ! safe_effects_for_reductions seems overkill to me
 * (2) compatible one with respect to the other.
 * (3) not killed by other proper effects on accumulators.
 * to avoid these checks, I can stop on expressions...
 */
static void check_proper_reductions(statement s) {
  reductions rs = load_proper_reductions(s);
  list /* of reduction */ lr = reductions_list(rs), lnr;
  if(ENDP(lr))
    return;

  /* all must be compatible, otherwise some side effect!
   */
  lnr = list_of_compatible_reductions(rs); /* checks (2) */

  /* now lnr is the new list of reductions.
   */
  if(lnr && !safe_effects_for_reductions(s, rs)) { /* checks (1) and (3) */
    gen_full_free_list(lnr);
    lnr = NIL;
  }

  gen_full_free_list(lr);
  // update the reduction list
  reductions_list(rs) = lnr;
}

DEFINE_LOCAL_STACK(crt_stat, statement)

/* hack: no check for direct translations ("call foo")
 * thus in this case effects reductions will be okay...
 * the reason for the patch is that I do not know how to preserve easily
 * such "invisible" reductions against proper effects. FC.
 */
static call last_translated_module_call = call_undefined;

/*
 * This function simply add statement to the newgen struct generated by
 * GENERIC_GLOBAL_FUNCTION macro.
 * @return TRUE
 * @param s, the statement to add to the generic map with the default reductions
 *
 */
static bool pr_statement_flt(statement s)
{
    store_proper_reductions(s, make_reductions(NIL));
    // crt_stat_filter simply push on the stack and return true
    return crt_stat_filter(s);
}

/*
 *@param s, the statement to check for reductions
 */
static void pr_statement_wrt(statement s)
{
    instruction i = statement_instruction(s);
    if (instruction_call_p(i) &&
        instruction_call(i)!=last_translated_module_call)
        check_proper_reductions(s);
    // crt_stat_rewrite pops from the stack and chek that s
    // was the latest pushed object
    crt_stat_rewrite(s);
}

static bool pr_call_flt(call c)
{
    statement head = crt_stat_head();
    reductions reds = load_proper_reductions(head);
    reduction red;

    pips_debug(9, "considering call to %s\n", entity_name(call_function(c)));

    if (call_proper_reduction_p(head, c, &red)) {
      // direct proper reduction
      reductions_list(reds) =
	CONS(REDUCTION, red, reductions_list(reds));
    }
    else if (entity_module_p(call_function(c)))
    {
      last_translated_module_call = c;
      reductions_list(reds) =
	    gen_nconc(translate_reductions(c), reductions_list(reds));
    } else {
      ifdebug(4) {
        pips_debug(4,"Reductions for statement are:\n");
        FOREACH(REDUCTION, r,reductions_list(reds)) {
          DEBUG_REDUCTION(0, "considering\n", r);
        }
      }
    }

    return true;
}

/* performs the computation of proper reductions for statement s.
 * this is a direct computation, throught gen_multi_recurse.
 */
static void compute_proper_reductions(statement s)
{
    make_crt_stat_stack();
    gen_multi_recurse(s,
		      statement_domain, pr_statement_flt, pr_statement_wrt,
		      call_domain, pr_call_flt, gen_null,
		      NULL);
    free_crt_stat_stack();
}

/* handler for pipsmake
 * input: module name
 * output: TRUE
 * side effects: some
 * - requires CODE PROPER_EFFECTS and callees' SUMMARY_{EFFECTS,REDUCTIONS}
 * - returns PROPER_REDUCTIONS to pips dbm
 */
bool proper_reductions(const char* module_name)
{
    entity module;

    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module_name);

    init_proper_reductions();
    set_generic_effects_as_needed();

    /* gets what is needed from PIPS DBM
     */
    module = local_name_to_top_level_entity(module_name);
    set_current_module_entity(module);
    set_current_module_statement
	((statement) db_get_memory_resource(DBR_CODE, module_name, true));
    set_proper_references((statement_effects)
        db_get_memory_resource(DBR_PROPER_REFERENCES, module_name, true));

    /* do the job
     */
    compute_proper_reductions(get_current_module_statement());

    /* returns the result to the DBM...
     */
    DB_PUT_MEMORY_RESOURCE
	(DBR_PROPER_REDUCTIONS, module_name, get_proper_reductions());

    reset_proper_reductions();
    reset_proper_references();
    reset_current_module_entity();
    reset_current_module_statement();
    generic_effects_reset_all_methods();

    debug_off();
    return true;
}

/**************************************************** CUMULATED REDUCTIONS */

/* Function storing Cumulated Reductions
 */
GENERIC_GLOBAL_FUNCTION(cumulated_reductions, pstatement_reductions)

/************************************** CUMULATED REDUCTIONS OF STATEMENT */

/* returns a r reduction of any compatible with { node } u ls
 * input: var, node and ls
 * output: true and some *pr, or FALSE
 */
static bool
build_reduction_of_variable(
    entity var,
    statement node,
    list /* of statement */ ls,
    reduction *pr)
{
    *pr = make_reduction
	(reference_undefined,
	 make_reduction_operator_none(),
	 NIL, NIL);

    if (!update_compatible_reduction
	(pr, var, effects_effects(load_proper_references(node)),
	 load_proper_reductions(node)))
    {
	free_reduction(*pr);
	return false;
    }

    FOREACH (STATEMENT, s, ls) {
      if (bound_cumulated_reductions_p(s) ) {
	if (!update_compatible_reduction (pr, var, load_rw_effects_list(s),
					  load_cumulated_reductions(s)))
	  {
	    free_reduction(*pr);
	    return false;
	  }
      }
      else {
	pips_debug(5, "stat %s %p\n", note_for_statement(s), s);
	pips_assert ("should not happen, all statements should have been visited for reduction", false);
      }
    }

    return true;
}

/* builds cumulated reductions for node, depending on node and
 * list of statement ls.
 */
static void
build_creductions_of_statement(
    statement node,
    list /* of statement */ ls)
{
    list /* of entity */ le;
    list /* of reduction */ lr=NIL;
    reduction r;

    /* list of candidate entities */
    le = list_of_reduced_variables(node, ls);

    pips_debug(5, "stat %s %p: %td candidate(s)\n",
	       note_for_statement(node), node, gen_length(le));

    /* for each candidate, extract the reduction if any */
    FOREACH (ENTITY, var, le)
      if (build_reduction_of_variable(var, node, ls, &r))
	lr = CONS(REDUCTION, r, lr);

    /* store the result */
    pips_debug(5, "stat %s %p -> %td reductions\n",
	       note_for_statement(node), node, gen_length(lr));

    store_cumulated_reductions(node, make_reductions(lr));
    gen_free_list(le);
}

/* Cumulated Reduction propagation functions for each possible instructions.
 * Statement s cumulated reduction computation involves :
 * - its own proper reductions and effects
 * - the cumulated reductions and effects of its sons
 * the computation is performed by build_reductions_of_statement.
 * the current statement is retrieved thru the crt_stat stack.
 * Perform the bottom-up propagation of cumulated reductions
 */
static void compute_cumulated_reductions(instruction i)
{
    statement parent = (statement)gen_get_ancestor(statement_domain,i);
    list l = NIL;
    bool tofree = true;
    switch(instruction_tag(i)) {
        case is_instruction_sequence:
            tofree=false;
            l=instruction_block(i);break;
        case is_instruction_loop:
            l=make_statement_list(loop_body(instruction_loop(i)));break;
        case is_instruction_forloop:
            l=make_statement_list(forloop_body(instruction_forloop(i)));break;
        case is_instruction_whileloop:
            l=make_statement_list(whileloop_body(instruction_whileloop(i)));break;
        case is_instruction_test:
            l=make_statement_list(test_true(instruction_test(i)),test_false(instruction_test(i)));break;
        case is_instruction_unstructured:
            {
                list /* of control */ lc = NIL;
                CONTROL_MAP( c, lc=CONS(CONTROL,c,lc), unstructured_control(instruction_unstructured(i)), lc);
                l = control_list_to_statement_list(lc);
                gen_free_list(lc);
            } ; break ;
        case is_instruction_call:
        case is_instruction_expression:
            store_cumulated_reductions(parent,copy_reductions(load_proper_reductions(parent)));
            pips_debug(5, "stat %s %p\n", note_for_statement(parent), parent);
            return ; /* it is important to return here */
        default:
            pips_internal_error("should not happen");
    };
    build_creductions_of_statement(parent,l);
    if(tofree) gen_free_list(l);
}

/* handler for pipsmake
 * input: the module name
 * output: TRUE
 * side effects: some
 * - requires CODE, PROPER_{EFFECTS,REDUCTIONS} and CUMULATED_EFFECTS
 * - returns CUMULATED_REDUCTIONS to pips dbm
 */
bool cumulated_reductions(const char* module_name)
{
    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module_name);

    init_cumulated_reductions();
    set_generic_effects_as_needed();

    /* gets what is needed from PIPS DBM
     */
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, true));
    set_proper_references((statement_effects)
        db_get_memory_resource(DBR_PROPER_REFERENCES, module_name, true));
    set_rw_effects((statement_effects)
        db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
    set_proper_reductions((pstatement_reductions)
	db_get_memory_resource(DBR_PROPER_REDUCTIONS, module_name, true));

    /* do the job here
     */
    gen_recurse(get_current_module_statement(),instruction_domain,gen_true,compute_cumulated_reductions);

    /* returns the result to the DBM...
     */
    DB_PUT_MEMORY_RESOURCE
	(DBR_CUMULATED_REDUCTIONS, module_name, get_cumulated_reductions());

    reset_cumulated_reductions();
    reset_proper_reductions();
    reset_proper_references();
    reset_rw_effects();
    reset_current_module_entity();
    reset_current_module_statement();
    generic_effects_reset_all_methods();

    debug_off();
    return true;
}

/** 
 * match a reduction operator against operator entity
 * 
 * @param op reduction operator
 * 
 * @return entity representing corresponding operator
 */
entity reduction_operator_entity(reduction_operator op)
{
    string opname = string_undefined;
    switch( reduction_operator_tag(op) ) {
        case is_reduction_operator_sum:
            opname=PLUS_OPERATOR_NAME;break;
        case is_reduction_operator_and:
            opname=AND_OPERATOR_NAME;break;
        case is_reduction_operator_bitwise_and:
            opname=BITWISE_AND_OPERATOR_NAME;break;
        case is_reduction_operator_bitwise_or:
            opname=BITWISE_OR_OPERATOR_NAME;break;
        case is_reduction_operator_bitwise_xor:
            opname=BITWISE_XOR_OPERATOR_NAME;break;
        case is_reduction_operator_csum:
            opname=PLUS_C_OPERATOR_NAME;break;
        case is_reduction_operator_eqv:
            opname=EQUIV_OPERATOR_NAME;break;
        case is_reduction_operator_max:
            opname=MAX_OPERATOR_NAME;break;
        case is_reduction_operator_min:
            opname=MIN_OPERATOR_NAME;break;
        case is_reduction_operator_neqv:
            opname=NON_EQUIV_OPERATOR_NAME;break;
        case is_reduction_operator_or:
            opname=OR_OPERATOR_NAME;break;
        case is_reduction_operator_prod:
            opname=MULTIPLY_OPERATOR_NAME;break;
        default:
            pips_internal_error("unhandled case");
    }
    return entity_intrinsic(opname);
}

bool same_reduction_p(reduction r1, reduction r2)
{
    return ( (reference_equal_p(reduction_reference(r1),reduction_reference(r2))) &&
            (reduction_operator_tag(reduction_op(r1)) == reduction_operator_tag(reduction_op(r2))) );
}

/* end of it!
 */
