/*
 * $Id$
 * 
 * $Log: declarations.c,v $
 * Revision 1.2  1997/10/21 15:35:50  coelho
 * includes added. plus fixes.
 *
 * Revision 1.1  1997/10/21 15:29:40  coelho
 * Initial revision
 *
 *
 * clean the declarations of a module.
 * to be called from pipsmake.
 * 
 * its not really a transformation, because declarations
 * are associated to the entity, not to the code.
 * the code is put so as to reinforce the prettyprint...
 *
 * clean_declarations > ## MODULE.code
 *     < PROGRAM.entities
 *     < MODULE.code
 */

#include <stdio.h>
#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "resources.h"
#include "pipsdbm.h"

bool
clean_declarations(string name)
{
    entity module;
    statement stat;
    module = local_name_to_top_level_entity(name);
    stat = (statement) db_get_memory_resource(DBR_CODE, name, TRUE);
    insure_declaration_coherency_of_module(module, stat);
    db_put_or_update_memory_resource(DBR_CODE, name, (char*) stat, TRUE);
    return TRUE;
}
