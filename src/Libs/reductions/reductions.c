/* $RCSfile: reductions.c,v $ (version $Revision$)
 * $Date: 1996/06/15 18:30:33 $, 
 *
 * detection of simple reductions.
 * debug driven by REDUCTIONS_DEBUG_LEVEL
 *
 * FC, June 1996.
 */

#include "local-header.h"
#include "control.h" /* for CONTROL_MAP() */

/****************************************************** SUMMARY REDUCTIONS */

reductions load_summary_reductions(entity f)
{
    pips_assert("is a module", entity_module_p(f));
    return (reductions) db_get_memory_resource
	(DBR_SUMMARY_REDUCTIONS, module_local_name(f), TRUE);
}

bool summary_reductions(string module)
{
    reductions red;

    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module);
    pips_user_warning("not implemented yet\n");

    /* that is indeed a first implementation */
    red = make_reductions(NIL);
    DB_PUT_MEMORY_RESOURCE(DBR_SUMMARY_REDUCTIONS, module, red);

    debug_off();
    return TRUE;
}

/******************************************************* PROPER REDUCTIONS */
/* Function storing Proper Reductions
 */
GENERIC_GLOBAL_FUNCTION(proper_reductions, pstatement_reductions)

reductions sload_proper_reductions(statement s)
{
    if (bound_proper_reductions_p(s))
	return load_proper_reductions(s);
    /* else */
    pips_user_warning("prop reduction not found for 0x%x\n", (unsigned int) s);
    return make_reductions(NIL); /* ??? memory leak! */
}

DEFINE_LOCAL_STACK(crt_stat, statement)

static bool pr_statement_flt(statement s)
{
    store_proper_reductions(s, make_reductions(NIL));
    return crt_stat_filter(s);
}

static void pr_call_rwt(call c)
{
    statement head = crt_stat_head();
    reduction red;

    if (call_proper_reduction_p(head, c, &red))
    {
	reductions reds = load_proper_reductions(head);
	pips_debug(6, "stat 0x%x -> reduction on %s\n",
		   (unsigned int) head, entity_name(reduction_variable(red)));
	reductions_list(reds) = CONS(REDUCTION, red, reductions_list(reds));
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

reductions sload_cumulated_reductions(statement s)
{
    if (bound_cumulated_reductions_p(s))
	return load_cumulated_reductions(s);
    /* else */
    pips_user_warning("cumu reduction not found for 0x%x\n", (unsigned int) s);
    return make_reductions(NIL); /* ??? memory leak */
}

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
	le = add_reduced_variables(le, sload_cumulated_reductions(s)),
	ls);
    return le;
}

/* returns a r reduction of any compatible with { node } u ls
 */
static bool 
build_reduction_of_variable(
    entity var,
    statement node,
    list /* of statement */ ls,
    reduction *pr)
{
    *pr = NULL; /* starting point */
    
    if (!update_compatible_reduction
	(pr, var, load_statement_proper_effects(node),
	 load_proper_reductions(node)))
	return FALSE;

    MAP(STATEMENT, s,
	if (!update_compatible_reduction
	    (pr, var, load_statement_cumulated_effects(s), 
	     load_cumulated_reductions(s)))
	    return FALSE,
	ls);

    return TRUE;
}

static reductions
build_reductions_of_variables(
    list /* of entity */ le,
    statement node,
    list /* of statement */ ls)
{
    list /* of reduction */ lr=NIL;
    reduction r;

    MAP(ENTITY, var,
	if (build_reduction_of_variable(var, node, ls, &r))
	    lr = CONS(REDUCTION, r, lr),
	le);

    return make_reductions(lr); /* gonna be of node */
}

static void 
build_reductions_of_statement(
    statement node,
    list /* of statement */ ls)
{
    list /* of entity */ le;
    le = list_of_reduced_variables(node, ls);
    store_cumulated_reductions
	(node, build_reductions_of_variables(le, node, ls));
    gen_free_list(le);
}

/* Cumulated Reduction propagation functions for each possible instructions.
 * Statement s cumulated reduction computation involves :
 * - its own proper reductions and effects
 * - the cumulated reductions and effects of its sons
 * the computation is performed by build_reductions_of_statement.
 */
static void cr_sequence_rwt(sequence s)
{
    build_reductions_of_statement(crt_stat_head(), sequence_statements(s));
}

static void cr_loop_rwt(loop l)
{
    list /* of statement */ ls = CONS(STATEMENT, loop_body(l), NIL);
    build_reductions_of_statement(crt_stat_head(), ls);
    gen_free_list(ls);
}

static void cr_test_rwt(test t)
{
    list /* of statement */ ls = 
	CONS(STATEMENT, test_true(t), CONS(STATEMENT, test_false(t), NIL));
    build_reductions_of_statement(crt_stat_head(), ls);
    gen_free_list(ls);
}

static void cr_unstructured_rwt(unstructured u)
{
    list /* of statements */ ls = NIL;
    CONTROL_MAP(c, {}, unstructured_control(u), ls);
    build_reductions_of_statement(crt_stat_head(), ls);
    gen_free_list(ls);
}

static void cr_call_rwt(call c)
{
    statement s = crt_stat_head();
    store_cumulated_reductions(s, copy_reductions(load_proper_reductions(s)));
}

/* Perform the bottom-up propagation of cumulated reductions 
 * for statement s, through gen_multi_recurse.
 * only call instructions must be walked thru,
 * hence it is stoped on expressions.
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
