/* static variables and related access functions concerning the current module
 *
 * Be'atrice Apvrille, august 27, 1993
 */



/* used to store the summary transformer ? 
   to retrieve intraprocedural effects ? */

#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "ri.h"
#include "ri-util.h"
#include "misc.h"


/*********************************************************** CURRENT ENTITY */

static entity current_module_entity = entity_undefined;

void set_current_module_entity(e)
entity e;
{
    pips_assert("entity is a module", entity_module_p(e));

    /* FI: I should perform some kind of memorization for all static variables
       including the value maps (especially them) */

    pips_assert("current module is undefined", 
		entity_undefined_p(current_module_entity)); 

    current_module_entity = e;
}

entity get_current_module_entity()
{
    return current_module_entity;
}

void reset_current_module_entity()
{
    /* pips_assert("current entity defined", 
       !entity_undefined_p(current_module_entity)); */ /* too risky ;-) */
    current_module_entity = entity_undefined;
}


/******************************************************* CURRENT STATEMENT */

/* used to retrieve the intraprocedural effects of the current module */

static statement current_module_statement = statement_undefined;

void set_current_module_statement(s)
statement s;
{
    pips_assert("set_current_module_statement", 
		current_module_statement == statement_undefined);
    current_module_statement = s;
}


statement get_current_module_statement()
{
    return current_module_statement;
}


void reset_current_module_statement()
{
    current_module_statement = statement_undefined;
}









