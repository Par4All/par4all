/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
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
#include "misc.h"

#include "ri.h"
#include "ri-util.h"


/*********************************************************** CURRENT ENTITY */

static entity current_module_entity = entity_undefined;
static list current_module_declarations=list_undefined;

/** @defgroup current_module Methods related to the current module

    Many parts of PIPS guesses that a current module is defined.

    These methods are used to set or get the module statement, entity, name...

    @{
*/

/** Set the current module entity

    It returns also the given entity to ease macro writing
*/
entity
set_current_module_entity(entity e)
{
    pips_assert("entity is a module", entity_module_p(e));

    /* FI: I should perform some kind of memorization for all static variables
       including the value maps (especially them) */

    pips_assert("current module is undefined",
		entity_undefined_p(current_module_entity));

    current_module_entity = e;
    reset_current_module_declarations();
    return e;
}


/** Get the entity of the current module
 */
entity
get_current_module_entity()
{
    return current_module_entity;
}



/** Reset the current module entity

    It asserts the module entity was previously set.
 */
void
reset_current_module_entity()
{
    pips_assert("current entity defined",
		!entity_undefined_p(current_module_entity));
    current_module_entity = entity_undefined;
    reset_current_module_declarations();
}

/** @} */

/* To be called by an error management routine only */
void
error_reset_current_module_entity()
{
    current_module_entity = entity_undefined;
    reset_current_module_declarations();
}


/** @addtogroup current_module
    @{
*/

/** Get the name of the current module */
const char*
get_current_module_name()
{
  return module_local_name(current_module_entity);
  /* return entity_user_name(current_module_entity); */
}


void set_current_module_declarations(list l)
{
  current_module_declarations = l;
}

void reset_current_module_declarations()
{
  current_module_declarations = list_undefined;
}

list get_current_module_declarations()
{
  return current_module_declarations;
}

/** @} */


/******************************************************* CURRENT STATEMENT */

/* used to retrieve the intraprocedural effects of the current module */

static statement current_module_statement = statement_undefined;
static statement stacked_current_module_statement = statement_undefined;

/** @addtogroup current_module
    @{
*/

/** Set the current module statement

    It returns also the given statement to ease macro writing
*/
statement
set_current_module_statement(statement s)
{
  pips_assert("The current module statement is undefined",
	      current_module_statement == statement_undefined);
  pips_assert("The new module statement is not undefined",
	      s != statement_undefined);
  current_module_statement = s;
  reset_current_module_declarations();
  return s;
}


/** Set the statement of the current module and push the statement of the
    previous one on a stack
 */
void push_current_module_statement(statement s)
{
    pips_assert("The stacked_current module statement is undefined", 
		stacked_current_module_statement == statement_undefined);
    pips_assert("The new module statement is not undefined", 
		s != statement_undefined);
    stacked_current_module_statement = current_module_statement;
    current_module_statement = s;
}


/** Pop the current module statement stack and use it as the current
    module statement
 */
void pop_current_module_statement(void)
{
    pips_assert("The current module statement is undefined",
		current_module_statement != statement_undefined);
    current_module_statement = stacked_current_module_statement;
    stacked_current_module_statement = statement_undefined;
}


/** Get the current module statement

    It returns also the given statement to ease macro writing
*/
statement
get_current_module_statement()
{
  pips_assert("The current module statement is defined",
	      current_module_statement != statement_undefined);
  return current_module_statement;
}


/** Reset the current module statement

    It asserts the module statement was previously set.
 */
void
reset_current_module_statement()
{
  pips_assert("current module statement defined",
	      !statement_undefined_p(current_module_statement));
  current_module_statement = statement_undefined;
  reset_current_module_declarations();
}

/** @} */


/* To be called by an error management routine only */
void
error_reset_current_module_statement()
{
    current_module_statement = statement_undefined;
    stacked_current_module_statement = statement_undefined;
    reset_current_module_declarations();
}
