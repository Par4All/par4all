/*
 * $Id$
 *
 * $Log: optimize.c,v $
 * Revision 1.3  1998/09/11 12:18:39  coelho
 * new version thru a reference (suggested by PJ).
 *
 * Revision 1.2  1998/09/11 09:42:49  coelho
 * write equalities...
 *
 * Revision 1.1  1998/04/29 09:07:42  coelho
 * Initial revision
 *
 *
 * expression optimizations by Julien.
 */

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"
#include "properties.h"

#include "resources.h"
#include "pipsdbm.h"


#define DEBUG_NAME "TRANSFORMATION_OPTIMIZE_EXPRESSIONS_DEBUG_LEVEL"

/* the list of right hand side expressions.
 */
static list /* of expression */ rhs;

/* rhs expressions of assignments.
 */
static bool call_filter(call c)
{
    if (ENTITY_ASSIGN_P(call_function(c)))
    {
	expression e = EXPRESSION(CAR(CDR(call_arguments(c))));
	rhs = CONS(EXPRESSION, e, rhs);
    }
    return FALSE;
}

/* other expressions may be found in loops and so?
 *//*
static bool expr_filter(expression e)
{
    rhs = CONS(EXPRESSION, e, rhs);
    return FALSE;
}*/

static list /* of expression */ 
get_list_of_rhs(statement s)
{
    list result;

    rhs = NIL;
    gen_multi_recurse(s,
		      expression_domain, gen_false, gen_null,
		      call_domain, call_filter, gen_null,
		      NULL);
    
    result = gen_nreverse(rhs);
    rhs = NIL;
    return result;
}

/* export a list of expression of the current module.
 */
static void write_list_of_rhs(FILE * out, list /* of expression */ le)
{
    reference astuce = make_reference(get_current_module_entity(), le);
    write_reference(out, astuce);
    reference_indices(astuce) = NIL;
    free_reference(astuce);
}

#define FILE_NAME "toeole"

/* export expressions to eole thru the newgen format.
 * both entities and rhs expressions are exported. 
 */
static void write_to_eole(string module, statement s)
{
    FILE * toeole = safe_fopen(FILE_NAME, "w");
    list /* of expression */ lc = get_list_of_rhs(s);

    pips_debug(3, "writing to eole for module %s\n", module);

    write_tabulated_entity(toeole);
    write_list_of_rhs(toeole, lc);

    safe_fclose(toeole, FILE_NAME);
    gen_free_list(lc);
}

/* pipsmake interface.
 */
bool optimize_expressions(string module_name)
{
    statement s;

    debug_on(DEBUG_NAME);

    /* get needed stuff.
     */
    set_current_module_entity(local_name_to_top_level_entity(module_name));
    set_current_module_statement((statement)
        db_get_memory_resource(DBR_CODE, module_name, TRUE));

    s = get_current_module_statement();

    /* do something here.
     */
    /*pips_user_warning("not implemented yet!\n");*/
    write_to_eole(module_name, s);

    /* return result.
     */
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, s);

    reset_current_module_entity();
    reset_current_module_statement();

    debug_off();

    return TRUE; /* okay! */
}
