/*
 * $Id$
 *
 * $Log: optimize.c,v $
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

static list /* of call */ assigns;

static bool call_filter(call c)
{
    if (ENTITY_ASSIGN_P(call_function(c)))
    {
	assigns = CONS(CALL, c, assigns);
    }
    return FALSE;
}

static list get_list_of_assigns(statement s)
{
    list result;

    assigns = NIL;
    gen_multi_recurse(s,
		      expression_domain, gen_false, gen_null,
		      call_domain, call_filter, gen_null,
		      NULL);
    
    result = gen_nreverse(assigns);
    assigns = NIL;
    return result;
}

static void write_list_of_assigns(FILE * out, list lc)
{
    int length = gen_length(lc);
    fprintf(out, "%d\n", length);
    MAP(CALL, c, 
    {
	expression e = EXPRESSION(CAR(CDR(call_arguments(c))));
	write_expression(out, e);
    },
	lc);
}

#define FILE_NAME "toeole"

static void write_to_eole(string module, statement s)
{
    FILE * toeole = safe_fopen(FILE_NAME, "w");
    list lc = get_list_of_assigns(s);

    pips_debug(3, "writing to eole for module %s\n", module);

    write_tabulated_entity(toeole);
    write_list_of_assigns(toeole, lc);

    safe_fclose(toeole, FILE_NAME);
    gen_free_list(lc);
}

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

    return TRUE;
}
