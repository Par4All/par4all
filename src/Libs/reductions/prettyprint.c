/* $RCSfile: prettyprint.c,v $ (version $Revision$)
 * $Date: 1996/06/17 11:32:50 $, 
 *
 * (pretty)print of reductions.
 *
 * FC, June 1996.
 */

#include "local-header.h"
#include "text.h"
#include "semantics.h"
#include "prettyprint.h"

#define REDUCTION_PREFIX "!"

#define PROP_SUFFIX ".preds"
#define CUMU_SUFFIX ".creds"

#define PROP_DECO "proper reductions"
#define CUMU_DECO "cumulated reductions"

GENERIC_LOCAL_FUNCTION(printed_reductions, pstatement_reductions)
static string reduction_decoration = NULL;

string reduction_operator_tag_name(tag t)
{
    switch(t)
    {
    case is_reduction_operator_none: return "none"; 
    case is_reduction_operator_sum:  return "sum";  
    case is_reduction_operator_prod: return "prod"; 
    case is_reduction_operator_min:  return "min";  
    case is_reduction_operator_max:  return "max";  
    case is_reduction_operator_and:  return "and";  
    case is_reduction_operator_or:   return "or";   
    default: pips_internal_error("unexpected reduction operator tag!");
    }

    return NULL;
}

/* allocates and returns the name of the operator
 */
string reduction_operator_name(reduction_operator o)
{
    return strdup(reduction_operator_tag_name(reduction_operator_tag(o)));
}

/* allocates and returns a list of strings for reduction r
 */
static list /* of string */ words_reduction(reduction r)
{
    return CONS(STRING, reduction_operator_name(reduction_op(r)),
	   CONS(STRING, strdup("["),
             gen_nconc( words_reference(reduction_reference(r)),
	   CONS(STRING, strdup("],"), NIL))));
}

static list /* of string */ words_reductions(string note, reductions rs)
{
    list /* of string */ ls = NIL;
    MAP(REDUCTION, r, 
	ls = gen_nconc(words_reduction(r), ls),
	reductions_list(rs));
    
    return ls? CONS(STRING, strdup(note), ls): NIL;
}

static string note_for_statement(statement s)
{
    instruction i = statement_instruction(s);
    switch (instruction_tag(i))
    {
    case is_instruction_sequence: return "seq  ";
    case is_instruction_loop: return "loop ";
    case is_instruction_test: return "test ";
    case is_instruction_call: return "call ";
    case is_instruction_unstructured: return "unst ";
    default: pips_internal_error("unexpected instruction tag");
    }
    return "";
}

static text text_reductions(entity module, int margin, statement s)
{
    text t;

    debug_on("REDUCTIONS_DEBUG_LEVEL");
    pips_debug(1, "considering statement 0x%x\n", (unsigned int) s);
    
    t = bound_printed_reductions_p(s)? /* unreachable statements? */
	words_predicate_to_commentary
	    (words_reductions(note_for_statement(s),
			      load_printed_reductions(s)), 
	     REDUCTION_PREFIX):
		make_text(NIL);

    debug_off();
    return t;
}

static text text_code_reductions(statement s)
{
    text t;
    debug_on("PRETTYPRINT_DEBUG_LEVEL");
    init_prettyprint(text_reductions);
    t = text_module(get_current_module_entity(), s);
    close_prettyprint();
    debug_off();
    return t;
}

static void 
print_reductions(
    string module_name, 
    string resource_name, 
    string decoration_name,
    string file_suffix)
{
    text t;

    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_printed_reductions((pstatement_reductions) 
	 db_get_memory_resource(resource_name, module_name, TRUE));
    set_current_module_statement((statement) 
	 db_get_memory_resource(DBR_CODE, module_name, TRUE));
    reduction_decoration = decoration_name;

    t = text_code_reductions(get_current_module_statement());
    (void) make_text_resource(module_name, DBR_PRINTED_FILE, file_suffix, t);
    /* some bug some where not investigated yet
     * free_text(t); results in a coredup much latter on.
     */

    reset_current_module_entity();
    reset_current_module_statement();
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
