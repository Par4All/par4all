/* 
 * $Id$
 *
 * Cloning of a subroutine.
 * debug: CLONING_DEBUG_LEVEL
 *
 * $Log: clone.c,v $
 * Revision 1.1  1997/10/31 10:40:46  coelho
 * Initial revision
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genC.h"

#include "misc.h"
#include "ri.h"
#include "ri-util.h"
#include "resources.h"
#include "pipsdbm.h"
#include "properties.h"

#define ARG_TO_CLONE "TRANSFORMATION_CLONE_ON_ARGUMENT"

/* clone module calls on argument arg in caller.
 * formal parameter of module number argn must be an integer scalar.
 */
static void
clone_on_argument(
    entity module      /* the module being cloned */,
    string caller_name /* the caller of interest */,
    int argn           /* the argument number to be cloned */)
{
    entity caller;
    statement stat;

    pips_debug(2, "cloning %s in %s on %d\n", 
	       entity_local_name(module), caller_name, argn);

    caller = local_name_to_top_level_entity(caller_name);
    stat = (statement) db_get_memory_resource(DBR_CODE, caller_name, TRUE);

    pips_user_error("sorry, not implemented yet\n");    
}

/* clone module name, on the argument specified by property
 * int TRANSFORMATION_CLONE_ON_ARGUMENT.
 *
 * clone 	> CALLERS.callees
 *       	> CALLERS.code
 *       < MODULE.code
 *       < CALLERS.callees
 *       < CALLERS.preconditions
 *       < CALLERS.code
 */
bool 
clone(string name)
{
    entity module;
    statement stat;
    callees callers; /* warf, warf */
    int argn;

    debug_on("CLONING_DEBUG_LEVEL");
    pips_debug(1, "considering module %s\n", name);

    module = local_name_to_top_level_entity(name);
    pips_assert("is a function", type_functional_p(entity_type(module)));
    set_current_module_entity(module);
    
    stat = (statement) db_get_memory_resource(DBR_CODE, name, TRUE);
    set_current_module_statement(stat);

    callers = (callees) db_get_memory_resource(DBR_CALLERS, name, TRUE);

    argn = get_int_property(ARG_TO_CLONE);

    if (argn==0)
    {
	do /* perform a user request to get the argument, 0 to stop */
	{
	    string args = user_request("argument of %s to clone", name);
	    argn = atoi(args); 
	    free(args);
	}
	while (argn<0);
    }

    /* check the argument type: must be an integer scalar */
    {
	entity arg = find_ith_parameter(module, argn);
	type t;
	variable v;

	if (entity_undefined_p(arg))
	    pips_user_error("%s: no #%d formal\n", name, argn);
	
	t = entity_type(arg);
	pips_assert("arg is a variable", type_variable_p(t));
	v = type_variable(t);

	if (basic_tag(variable_basic(v))!=is_basic_int ||
	    variable_dimensions(v))
	    pips_user_error("%s: %d formal not a scalar int\n", name, argn);
    }

    MAP(STRING, caller_name, 
	clone_on_argument(module, caller_name, argn),
	callees_callees(callers));    

    reset_current_module_entity();
    reset_current_module_statement();

    pips_debug(1, "done with module %s\n", name);
    debug_off();
    return TRUE;
}
