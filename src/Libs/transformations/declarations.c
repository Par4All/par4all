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
 * $Id$
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
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "resources.h"
#include "pipsdbm.h"

bool
clean_declarations(string name)
{
    entity module;
    statement stat;
    module = module_name_to_entity(name);
    set_current_module_entity(module);
    stat = (statement) db_get_memory_resource(DBR_CODE, name, TRUE);
    insure_declaration_coherency_of_module(module, stat);
    db_put_or_update_memory_resource(DBR_CODE, name, (char*) stat, TRUE);
    reset_current_module_entity();
    return TRUE;
}
