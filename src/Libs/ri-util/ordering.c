/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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

/** @file ordering.c

    Defines a static mapping from orderings to statements.
    must be initialize_ordering_to_statement, and afterwards
    reset_ordering_to_statement.

    For information on ordering, see control/reorder.c
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


/* Test if the ordering to statement is initialized */
bool
ordering_to_statement_initialized_p()
{
    return OrderingToStatement != hash_table_undefined;
}


/* Dump the ordering with the corresponding statement address */
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


/* Get the statement from an ordering in a given ordering to statement
   table */
static statement
apply_ordering_to_statement(hash_table ots, _int o)
{
    statement s;
    pips_assert("defined hash table...",
		ots != NULL && ots != hash_table_undefined);

    if(o == STATEMENT_ORDERING_UNDEFINED)
	pips_internal_error("Illegal ordering %td", o);

    s = (statement) hash_get(ots, (char *) o);

    if(s == statement_undefined)
	pips_internal_error("no statement for order %td=(%td,%td)",
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
static bool add_ordering_of_the_statement(statement stat, hash_table ots)
{
  pips_assert("ordering is defined",
	      statement_ordering(stat) != STATEMENT_ORDERING_UNDEFINED);
  hash_put(ots, (void *) statement_ordering(stat), (void *) stat);
  return true;
}


/* Add the statement for its ordering, if any, in the hash-map. */
bool add_ordering_of_the_statement_to_current_mapping( statement stat )
{
  pips_assert("ordering is defined",
        statement_ordering(stat) != STATEMENT_ORDERING_UNDEFINED);
  hash_put(OrderingToStatement, (void *) statement_ordering(stat), (void *) stat);
  return true;
}

/* Overwrite the statement for its ordering, if any, in the hash-map.
 * The difference with add_ordering_of_the_statement_to_current_mapping() is
 * that this version won't trigger a warning if the mapping already exist
 */
bool overwrite_ordering_of_the_statement_to_current_mapping( statement stat )
{
  pips_assert("ordering is defined",
        statement_ordering(stat) != STATEMENT_ORDERING_UNDEFINED);
  hash_overwrite(OrderingToStatement, (void *) statement_ordering(stat), (void *) stat);
  return true;
}


/* Initialize the ordering to statement mapping by iterating from a given
   statement

   @param ots is ordering to statement hash-table to fill

   @param s is the statement to start with. Typically the module
   statement.
*/
static void rinitialize_ordering_to_statement(hash_table ots, statement s)
{
  /* Simplify this with a gen_recurse to avoid dealing with all the new
     cases by hand (for-loops...).

     Apply a prefix hash-map add to be compatible with previous
     implementation and avoid different hash-map iteration later. */
  gen_context_recurse(s, ots, statement_domain,
		      add_ordering_of_the_statement, gen_identity);
}


/* To be used instead of initialize_ordering_to_statement() to make
   sure that the hash table ots is in sync with the current module. */
hash_table set_ordering_to_statement(statement s)
{
    pips_assert("hash table \"OrderingToStatement\" is undefined",
		OrderingToStatement==hash_table_undefined);
    hash_table ots =  hash_table_make(hash_int, 0);
    rinitialize_ordering_to_statement(ots, s);
    OrderingToStatement = ots;
    return ots;
}


/* Reset the mapping from ordering to statement. */
void
reset_ordering_to_statement(void)
{
    pips_assert("hash table is defined",
		OrderingToStatement!=hash_table_undefined);

    hash_table_free(OrderingToStatement),
    OrderingToStatement = hash_table_undefined;
}
