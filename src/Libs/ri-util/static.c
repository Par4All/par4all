/* static variables and related access functions concerning the current module
 *
 * Be'atrice Apvrille, august 27, 1993
 */



/* used to store the summary transformer ? 
   to retrieve intraprocedural effects ? */

#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"

#include "ri.h"
#include "ri-util.h"
#include "misc.h"


/*********************************************************** CURRENT ENTITY */

static entity current_module_entity = entity_undefined;

void 
set_current_module_entity(e)
entity e;
{
    pips_assert("entity is a module", entity_module_p(e));

    /* FI: I should perform some kind of memorization for all static variables
       including the value maps (especially them) */

    pips_assert("current module is undefined", 
		entity_undefined_p(current_module_entity)); 

    current_module_entity = e;
}

entity 
get_current_module_entity()
{
    return current_module_entity;
}

void 
reset_current_module_entity()
{
    /* pips_assert("current entity defined", 
       !entity_undefined_p(current_module_entity)); */ /* too risky ;-) */
    current_module_entity = entity_undefined;
}

string
get_current_module_name()
{
    return module_local_name(current_module_entity);
}


/******************************************************* CURRENT STATEMENT */

/* used to retrieve the intraprocedural effects of the current module */

static statement current_module_statement = statement_undefined;

void 
set_current_module_statement(s)
statement s;
{
    pips_assert("The current module statement is undefined", 
		current_module_statement == statement_undefined);
    pips_assert("The new module statement is not undefined", 
		s != statement_undefined);
    current_module_statement = s;
}


statement 
get_current_module_statement()
{
    pips_assert("The current module statement is defined", 
		current_module_statement != statement_undefined);
    return current_module_statement;
}


void 
reset_current_module_statement()
{
    current_module_statement = statement_undefined;
}
