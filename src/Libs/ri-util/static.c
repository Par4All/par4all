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
/* static variables and related access functions concerning the current module
 *
 * Be'atrice Apvrille, august 27, 1993
 *
 * $Id$
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
    pips_assert("current entity defined", 
		!entity_undefined_p(current_module_entity));
    current_module_entity = entity_undefined;
}

/* To be called by an error management routine only */
void 
error_reset_current_module_entity()
{
    current_module_entity = entity_undefined;
}

string
get_current_module_name()
{
  return module_local_name(current_module_entity);
  /* return entity_user_name(current_module_entity); */
}


/******************************************************* CURRENT STATEMENT */

/* used to retrieve the intraprocedural effects of the current module */

static statement current_module_statement = statement_undefined;
static statement stacked_current_module_statement = statement_undefined;

void set_current_module_statement(statement s)
{
    pips_assert("The current module statement is undefined", 
		current_module_statement == statement_undefined);
    pips_assert("The new module statement is not undefined", 
		s != statement_undefined);
    current_module_statement = s;
}

void push_current_module_statement(statement s)
{
    pips_assert("The stacked_current module statement is undefined", 
		stacked_current_module_statement == statement_undefined);
    pips_assert("The new module statement is not undefined", 
		s != statement_undefined);
    stacked_current_module_statement = current_module_statement;
    current_module_statement = s;
}

void pop_current_module_statement(void)
{
    pips_assert("The current module statement is undefined", 
		current_module_statement != statement_undefined);
    current_module_statement = stacked_current_module_statement;
    stacked_current_module_statement = statement_undefined;
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
    pips_assert("current module statement defined", 
       !statement_undefined_p(current_module_statement));
    current_module_statement = statement_undefined;
}

/* To be called by an error management routine only */

void 
error_reset_current_module_statement()
{
    current_module_statement = statement_undefined;
    stacked_current_module_statement = statement_undefined;
}
