/* $RCSfile: reductions.c,v $ (version $Revision$)
 * $Date: 1996/06/18 15:57:09 $, 
 *
 * detection of simple reductions.
 * debug driven by REDUCTIONS_DEBUG_LEVEL
 *
 * FC, June 1996.
 */

#include "local-header.h"
#include "control.h" /* for CONTROL_MAP() */
#include "semantics.h" /* for load_summary_effects() */

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
	(DBR_SUMMARY_REDUCTIONS, module_local_name(f), TRUE);
}

/* should rather check conflicts ? */
static bool entity_in_effect_list_p(entity v, list /* of effect */ le)
{
    MAP(EFFECT, e, if (effect_variable(e)==v) return TRUE, le);
    return FALSE;
}

static reduction 
compute_one_summary_reduction(reduction model, list /* of effect */ le)
{
    reduction r = copy_reduction(model);
    entity var = reduction_variable(model);

    /* keep the entities that are exported... 
     */
    MAP(ENTITY, e,
	if (!entity_in_effect_list_p(e, le))
	    remove_variable_from_reduction(r, e),
	reduction_dependences(model));

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

    pips_debug(3, "module %s: %d cumulated reductions\n",
	       entity_name(f), gen_length(lc));

    MAP(REDUCTION, r,
    {
	DEBUG_REDUCTION(4, "considering\n", r);
	if (entity_in_effect_list_p(reduction_variable(r), le))
	    lr = CONS(REDUCTION, compute_one_summary_reduction(r, le), lr);
    },
	lc);

    return make_reductions(lr);
}

/* handler for pipsmake
 * input: module name
 * output: TRUE
 * side effects: stores the summary reductions to pipsdbm
 */
bool summary_reductions(string module_name)
{
    reductions red;

    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module_name);
    pips_user_warning("not implemented yet\n");

    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, TRUE));
    set_cumulated_reductions((pstatement_reductions)
        db_get_memory_resource(DBR_CUMULATED_REDUCTIONS, module_name, TRUE));

    red = compute_summary_reductions(get_current_module_entity());

    DB_PUT_MEMORY_RESOURCE(DBR_SUMMARY_REDUCTIONS, module_name, red);

    reset_cumulated_reductions();
    reset_current_module_statement();
    reset_current_module_entity();

    debug_off();
    return TRUE;
}

/******************************************************* PROPER REDUCTIONS */
/* Function storing Proper Reductions
 */
GENERIC_GLOBAL_FUNCTION(proper_reductions, pstatement_reductions)

DEFINE_LOCAL_STACK(crt_stat, statement)

static bool pr_statement_flt(statement s)
{
    store_proper_reductions(s, make_reductions(NIL));
    return crt_stat_filter(s);
}

/* static reductions reductions_from_callees */

static void pr_call_rwt(call c)
{
    statement head = crt_stat_head();
    reductions reds = load_proper_reductions(head);
    reduction red;

    if (call_proper_reduction_p(head, c, &red))
    {
	pips_debug(6, "stat 0x%x -> reduction on %s\n",
		   (unsigned int) head, entity_name(reduction_variable(red)));
	reductions_list(reds) = CONS(REDUCTION, red, reductions_list(reds));
    }
    else
    {
	/* very basic and false (commons forgotten) 
	 * at the time, just to test it 
	 * should be for all local calls inside, plus
	 * to be checked afterwards... (references, compatibilities)
	 */
	list lr = translate_reductions(c);

	if (lr)
	{
	    pips_user_warning("reduction translation basic and false!\n");
	    reductions_list(reds) = gen_nconc(lr, reductions_list(reds));
	}
    }
}

/* performs the computation of proper reductions for statement s.
 * this is a direct computation, throught gen_multi_recurse.
 * only call instructions must be visited, hence it is stopped on expressions
 */
static void compute_proper_reductions(statement s)
{
    make_crt_stat_stack();
    gen_multi_recurse(s,
		      statement_domain, pr_statement_flt, crt_stat_rewrite,
		      call_domain, gen_true, pr_call_rwt,
		      expression_domain, gen_false, gen_null,
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
bool proper_reductions(string module_name)
{
    entity module;

    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module_name);

    pips_user_warning("being implemented, keep cool...\n");

    init_proper_reductions();

    /* gets what is needed from PIPS DBM
     */
    module = local_name_to_top_level_entity(module_name);
    set_current_module_entity(module);
    set_current_module_statement
	((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE));
    set_proper_effects_map(effectsmap_to_listmap((statement_mapping) 
        db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE)));

    /* do the job 
     */
    compute_proper_reductions(get_current_module_statement());

    /* returns the result to the DBM...
     */
    DB_PUT_MEMORY_RESOURCE
	(DBR_PROPER_REDUCTIONS, module_name, get_proper_reductions());
    reset_proper_reductions();
    reset_proper_effects_map();
    reset_current_module_entity();
    reset_current_module_statement();

    debug_off();
    return TRUE;
}

/**************************************************** CUMULATED REDUCTIONS */

/* Function storing Cumulated Reductions
 */
GENERIC_GLOBAL_FUNCTION(cumulated_reductions, pstatement_reductions)

/************************************************ LIST OF REDUCED ENTITIES */

/* list of entities that may be reduced
 */
static list /* of entity */
add_reduced_variables(
    list /* of entity */ le,
    reductions rs)
{
    MAP(REDUCTION, r, 
	le = gen_once(reduction_variable(r), le), 
	reductions_list(rs));
    return le;
}
 
static list /* of entity */ 
list_of_reduced_variables(
    statement node, 
    list /* of statement */ls)
{
    list /* of entity */ le = NIL;
    le = add_reduced_variables(le, load_proper_reductions(node));
    MAP(STATEMENT, s, 
	le = add_reduced_variables(le, load_cumulated_reductions(s)),
	ls);
    return le;
}

/************************************** CUMULATED REDUCTIONS OF STATEMENT */

/* returns a r reduction of any compatible with { node } u ls
 * input: var, node and ls
 * output: TRUE and some *pr, or FALSE
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
	 make_reduction_operator(is_reduction_operator_none, UU),
	 NIL);

    if (!update_compatible_reduction
	(pr, var, load_statement_proper_effects(node),
	 load_proper_reductions(node)))
    {
	free_reduction(*pr);
	return FALSE;
    }

    MAP(STATEMENT, s,
    {
	if (!update_compatible_reduction
	    (pr, var, load_statement_cumulated_effects(s), 
	     load_cumulated_reductions(s)))
	{
	    free_reduction(*pr);
	    return FALSE;
	}
    },
	ls);

    return TRUE;
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

    pips_debug(5, "stat %s 0x%x: %d candidate(s)\n",
	       note_for_statement(node), (unsigned int) node, gen_length(le));

    /* for each candidate, extract the reduction if any */
    MAP(ENTITY, var,
	if (build_reduction_of_variable(var, node, ls, &r))
	    lr = CONS(REDUCTION, r, lr),
	le);

    /* store the result */
    pips_debug(5, "stat %s 0x%x -> %d reductions\n", 
	       note_for_statement(node), (unsigned int) node, gen_length(lr));

    store_cumulated_reductions(node, make_reductions(lr)); 
    gen_free_list(le);
}

/* Cumulated Reduction propagation functions for each possible instructions.
 * Statement s cumulated reduction computation involves :
 * - its own proper reductions and effects
 * - the cumulated reductions and effects of its sons
 * the computation is performed by build_reductions_of_statement.
 * the current statement is retrieved thru the crt_stat stack.
 */
static void cr_sequence_rwt(sequence s)
{
    build_creductions_of_statement(crt_stat_head(), sequence_statements(s));
}

static void cr_loop_rwt(loop l)
{
    list /* of statement */ ls = CONS(STATEMENT, loop_body(l), NIL);
    build_creductions_of_statement(crt_stat_head(), ls);
    gen_free_list(ls);
}

static void cr_test_rwt(test t)
{
    list /* of statement */ ls = 
	CONS(STATEMENT, test_true(t), CONS(STATEMENT, test_false(t), NIL));
    build_creductions_of_statement(crt_stat_head(), ls);
    gen_free_list(ls);
}

static void cr_unstructured_rwt(unstructured u)
{
    list /* of control */ lc = NIL, /* of statement */ ls;
    CONTROL_MAP(c, {}, unstructured_control(u), lc);
    ls = control_list_to_statement_list(lc);
    build_creductions_of_statement(crt_stat_head(), ls);
    gen_free_list(ls); gen_free_list(lc);
}

static void cr_call_rwt(call c)
{
    statement s = crt_stat_head();
    store_cumulated_reductions(s, copy_reductions(load_proper_reductions(s)));
}

/* Perform the bottom-up propagation of cumulated reductions 
 * for statement s, through gen_multi_recurse.
 * only call instructions must be walked thru,
 *   hence it is stoped on expressions.
 * the crt_stat stack is used.
 */
static void compute_cumulated_reductions(statement s)
{
    make_crt_stat_stack();
    gen_multi_recurse(s,
		      statement_domain, crt_stat_filter, crt_stat_rewrite,
		      sequence_domain, gen_true, cr_sequence_rwt,
		      loop_domain, gen_true, cr_loop_rwt,
		      test_domain, gen_true, cr_test_rwt,
		      unstructured_domain, gen_true, cr_unstructured_rwt,
		      call_domain, gen_true, cr_call_rwt,
		      expression_domain, gen_false, gen_null,
		      NULL);
    free_crt_stat_stack();
}

/* handler for pipsmake
 * input: the module name
 * output: TRUE
 * side effects: some
 * - requires CODE, PROPER_{EFFECTS,REDUCTIONS} and CUMULATED_EFFECTS
 * - returns CUMULATED_REDUCTIONS to pips dbm
 */
bool cumulated_reductions(string module_name)
{
    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module_name);

    pips_user_warning("being implemented, keep cool\n");

    init_cumulated_reductions();

    /* gets what is needed from PIPS DBM
     */
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, TRUE));
    set_proper_effects_map(effectsmap_to_listmap((statement_mapping) 
        db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, TRUE)));
    set_cumulated_effects_map(effectsmap_to_listmap((statement_mapping) 
        db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE)));
    set_proper_reductions((pstatement_reductions)
	db_get_memory_resource(DBR_PROPER_REDUCTIONS, module_name, TRUE));

    /* do the job here
     */
    compute_cumulated_reductions(get_current_module_statement());

    /* returns the result to the DBM...
     */
    DB_PUT_MEMORY_RESOURCE
	(DBR_CUMULATED_REDUCTIONS, module_name, get_cumulated_reductions());
    reset_cumulated_reductions();
    reset_proper_reductions();
    reset_proper_effects_map();
    reset_cumulated_effects_map();
    reset_current_module_entity();
    reset_current_module_statement();
    
    debug_off();
    return TRUE;
}

/* end of $RCSfile: reductions.c,v $
 */
