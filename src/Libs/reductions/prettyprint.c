/* $RCSfile: prettyprint.c,v $ (version $Revision$)
 * $Date: 1996/06/15 13:22:08 $, 
 *
 * prettyprint of reductions.
 *
 * FC, June 1996.
 */

#include "local-header.h"
#include "text.h"
#include "semantics.h"
#include "prettyprint.h"

#define PROP_SUFFIX ".preds"
#define CUMU_SUFFIX ".creds"

#define PROP_DECO "proper reductions"
#define CUMU_DECO "cumulated reductions"

GENERIC_LOCAL_FUNCTION(printed_reductions, pstatement_reductions)
static string reduction_decoration = NULL;

/* allocates and returns the name of the operator
 */
static string reduction_operator_name(reduction_operator o)
{
    string name = NULL;

    switch(reduction_operator_tag(o))
    {
    case is_reduction_operator_none: name = "none"; break;
    case is_reduction_operator_sum:  name = "sum";  break;
    case is_reduction_operator_prod: name = "prod"; break;
    case is_reduction_operator_min:  name = "min";  break;
    case is_reduction_operator_max:  name = "max";  break;
    case is_reduction_operator_and:  name = "and";  break;
    case is_reduction_operator_or:   name = "or";   break;
    default: pips_internal_error("unexpected reduction operator tag!");
    }

    return strdup(name);
}

static list /* of string */ words_reduction(reduction r)
{
    return CONS(STRING, reduction_operator_name(reduction_op(r)),
	   CONS(STRING, "[",
            gen_nconc( words_reference(reduction_reference(r)),
	   CONS(STRING, "],", NIL))));
}

static list /* of string */ words_reductions(reductions rs)
{
    list /* of string */ ls = NIL;
    MAP(REDUCTION, r, 
	ls = gen_nconc(words_reduction(r), ls),
	reductions_list(rs));
    return ls;
}

static text text_reductions(statement s)
{
    if (!bound_printed_reductions_p(s)) /* unreachable statements */
	return make_text(NIL);
    /* else */

    return words_predicate_to_commentary
	(words_reductions(load_printed_reductions(s)), "!");
}

static void 
print_reductions(
    string module_name, 
    string resource_name, 
    string decoration_name,
    string file_suffix)
{
    text t;

    set_current_module_entity
	(local_name_to_top_level_entity(module_name));
    set_printed_reductions((pstatement_reductions) 
	 db_get_memory_resource(resource_name, module_name, TRUE));
    set_current_module_statement
	((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE));
    reduction_decoration = decoration_name;

    init_prettyprint(text_reductions);

    t = text_module(get_current_module_entity(),
		    get_current_module_statement());

    (void) make_text_resource(module_name, DBR_PRINTED_FILE, file_suffix, t);

    free_text(t);
    close_prettyprint();

    reset_printed_reductions();
    reduction_decoration = NULL;
}
    

bool print_code_proper_reductions(string module_name)
{
    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module_name);

    print_reductions(module_name, DBR_PROPER_REDUCTIONS, 
		     PROP_DECO, PROP_SUFFIX);

    debug_off();
    return TRUE;
}

bool print_code_cumulated_reductions(string module_name)
{
    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", module_name);

    print_reductions(module_name, DBR_CUMULATED_REDUCTIONS, 
		     CUMU_DECO, CUMU_SUFFIX);
			      

    debug_off();
    return TRUE;
}

/* end of $RCSfile: prettyprint.c,v $
 */
