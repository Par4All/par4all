/* $RCSfile: reductions.c,v $ (version $Revision$)
 * $Date: 1996/06/15 13:22:09 $, 
 *
 * detection of simple reductions.
 * debug driven by REDUCTIONS_DEBUG_LEVEL
 *
 * FC, June 1996.
 */

#include "local-header.h"

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

GENERIC_GLOBAL_FUNCTION(proper_reductions, pstatement_reductions)

DEFINE_LOCAL_STACK(crt_stat, statement)

static bool statement_flt(statement s)
{
    store_proper_reductions(s, make_reductions(NIL));
    return crt_stat_filter(s);
}

static void call_rwt(call c)
{
    statement head = crt_stat_head();
    reduction red;

    if (call_proper_reduction_p(head, c, &red))
    {
	reductions reds = load_proper_reductions(head);
	reductions_list(reds) = CONS(REDUCTION, red, reductions_list(reds));
    }
}

static void compute_proper_reductions(statement s)
{
    make_crt_stat_stack();
    gen_multi_recurse(s,
		      statement_domain, statement_flt, crt_stat_rewrite,
		      call_domain, gen_true, call_rwt,
		      NULL);
    free_crt_stat_stack();
}

bool proper_reductions(string module_name)
{
    entity module;

    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module_name);

    pips_user_warning("being implemented...\n");

    init_proper_reductions();

    /* gets what is needed from PIPS DBM
     */
    module = local_name_to_top_level_entity(module_name);
    set_current_module_entity(module);
    set_current_module_statement
	((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE));
    
    /* do the job 
     */
    compute_proper_reductions(get_current_module_statement());

    /* return the result to the DBM...
     */
    DB_PUT_MEMORY_RESOURCE
	(DBR_PROPER_REDUCTIONS, module_name, get_proper_reductions());
    reset_proper_reductions();

    debug_off();
    return TRUE;
}

/**************************************************** CUMULATED REDUCTIONS */

bool cumulated_reductions(string module)
{
    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module);

    pips_user_warning("not implemented yet\n");

    debug_off();
    return TRUE;
}

/* end of $RCSfile: reductions.c,v $
 */
