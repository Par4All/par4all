/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
/*
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
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "resources.h"
#include "pipsdbm.h"

/** 
 * recursievely call statement_remove_unused_declarations on all module statement
 * 
 * @param module_name name of the processed module
 */
void
clean_declarations(char * module_name)
{
    /* prelude*/
    set_current_module_entity(module_name_to_entity( module_name ));
    set_current_module_statement((statement) db_get_memory_resource(DBR_CODE, module_name, TRUE) );

    /* body*/
    entity_clean_declarations(get_current_module_entity(),get_current_module_statement());
    gen_recurse(get_current_module_statement(),statement_domain,gen_true,statement_clean_declarations);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), get_current_module_statement());

    /*postlude */
    reset_current_module_entity();
    reset_current_module_statement();
    return true;
}
