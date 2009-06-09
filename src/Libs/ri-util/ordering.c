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
 * Defines a static mapping from orderings to statements.
 * must be initialize_ordering_to_statement, and afterwards
 * reset_ordering_to_statement.
 */

#include <stdio.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"

/* a hash table to map orderings (integer) to statements (pointers)
 * assumed to be valid for the current module returned by
 * get_current_module_entity(). This is assumed to hold when
 * pipsmake issues a request to lower level libraries.
 *
 * db_get_current_module() returns the module used in the request
 * to pipsmake and is usually different.
 *
 * It would be possible to expose a lower level interface to manage
 * several ordering_to_statement hash tables
 */
static hash_table OrderingToStatement = hash_table_undefined;

bool 
ordering_to_statement_initialized_p()
{
    return OrderingToStatement != hash_table_undefined;
}

void 
print_ordering_to_statement(void)
{
    HASH_MAP(ko, vs, {
	_int o = (_int) ko;
	statement s = (statement) vs;

	fprintf(stderr,"%td (%td,%td)->%s\n",
		o, ORDERING_NUMBER(o), ORDERING_STATEMENT(o),
		statement_identification(s));
    }, OrderingToStatement);
}

static statement 
apply_ordering_to_statement(hash_table ots, _int o)
{
    statement s;
    pips_assert("defined hash table...",
		ots != NULL && ots != hash_table_undefined);

    if(o == STATEMENT_ORDERING_UNDEFINED)
	pips_internal_error("Illegal ordering %td\n", o);

    s = (statement) hash_get(ots, (char *) o);

    if(s == statement_undefined) 
	pips_internal_error("no statement for order %td=(%td,%td)\n",
			    o, ORDERING_NUMBER(o), ORDERING_STATEMENT(o));

    return s;
}


/* Get the statement associated to a given ordering.

   It is useful for retrieve the statements associated with the arcs in
   the dependence graphs for example. */
statement
ordering_to_statement(int o)
{
    statement s;
    s = apply_ordering_to_statement(OrderingToStatement, o);
    return s;
}


/* Add the statement for its ordering, if any, in the hash-map. */
static bool
add_ordering_of_the_statement(statement s,
			      void * a_context) {
  hash_table ots = (hash_table) a_context;
  if (statement_ordering(s) != STATEMENT_ORDERING_UNDEFINED)
    hash_put(ots, (char *) statement_ordering(s), (char *) s);

  // Go on walking down the RI:
  return TRUE;
}


static void
rinitialize_ordering_to_statement(hash_table ots, statement s) {
  /* Simplify this with a gen_recurse to avoid dealing with all the new
     cases by hand (for-loops...).

     Apply a prefix hash-map add to be compatible with previous
     implementation and avoid different hash-map iteration later. */
  gen_context_recurse(s, ots, statement_domain,
		      add_ordering_of_the_statement,
		      gen_identity);
}


/* To be used instead of initialize_ordering_to_statement() to make
   sure that the hash table ots is in sync with the current module. */
hash_table set_ordering_to_statement(statement s)
{
    hash_table ots =  hash_table_make(hash_int, 0);
    rinitialize_ordering_to_statement(ots, s);
    OrderingToStatement = ots;
    return ots;
}

/* To be phased out.
 * FI recommands not to use this
 */
static void 
initialize_ordering_to_statement(statement s)
{
    /* FI: I do not like that automatic cleaning any more... */
    if (OrderingToStatement != hash_table_undefined) {
	reset_ordering_to_statement();
    }

    OrderingToStatement = set_ordering_to_statement(s);
}

void 
reset_ordering_to_statement(void)
{
    pips_assert("hash table is defined", 
		OrderingToStatement!=hash_table_undefined);

    hash_table_free(OrderingToStatement),
    OrderingToStatement = hash_table_undefined;
}
