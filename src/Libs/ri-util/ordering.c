/*
 * $Id$
 *
 * Defines a static mapping from orderings to statements.
 * must be initialize_ordering_to_statement, and afterwards
 * reset_ordering_to_statement.
 *
 * $Log: ordering.c,v $
 * Revision 1.10  1997/11/22 15:19:30  coelho
 * cleaner. RCS headers. better asser message.
 *
 */

#include <stdio.h>

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
    HASH_MAP(ko, vs,{
	int o = (int) ko;
	statement s = (statement) vs;

	fprintf(stderr,"%d (%d,%d)->%s\n",
		o, ORDERING_NUMBER(o), ORDERING_STATEMENT(o),
		statement_identification(s));
    },OrderingToStatement);
}

static statement 
apply_ordering_to_statement(hash_table ots, int o)
{
    statement s;
    pips_assert("defined hash table...",
		ots != NULL && ots != hash_table_undefined);

    if(o == STATEMENT_ORDERING_UNDEFINED)
	pips_internal_error("Illegal ordering %d\n", o);

    s = (statement) hash_get(ots, (char *) o);

    if(s == statement_undefined) 
	pips_internal_error("no statement for order %d=(%d,%d)\n",
			    o, ORDERING_NUMBER(o), ORDERING_STATEMENT(o));

    return s;
}

statement 
ordering_to_statement(int o)
{
    statement s;
    s = apply_ordering_to_statement(OrderingToStatement, o);
    return s;
}

static void 
rinitialize_ordering_to_statement(hash_table ots, statement s)
{
    instruction i = statement_instruction(s);

    if (statement_ordering(s) != STATEMENT_ORDERING_UNDEFINED)
	hash_put(ots,  
		 (char *) statement_ordering(s), (char *) s);

    switch (instruction_tag(i)) {

      case is_instruction_block:
	MAPL(ps, {
	    rinitialize_ordering_to_statement(ots, STATEMENT(CAR(ps)));
	}, instruction_block(i));
	break;

      case is_instruction_loop:
	rinitialize_ordering_to_statement(ots, loop_body(instruction_loop(i)));
	break;

      case is_instruction_test:
 	rinitialize_ordering_to_statement(ots, test_true(instruction_test(i)));
	rinitialize_ordering_to_statement(ots, test_false(instruction_test(i)));
	break;

      case is_instruction_call:
      case is_instruction_goto:
	break;

      case is_instruction_unstructured: {
	  cons *blocs = NIL ;

	  CONTROL_MAP(c, {
	      rinitialize_ordering_to_statement(ots, control_statement(c));
	  }, unstructured_control(instruction_unstructured(i)), blocs);
	  gen_free_list( blocs );

	  break;
      }
	    
      default:
	pips_error("rinitialize_ordering_to_statement", "bad tag\n");
    }
}

static hash_table 
set_ordering_to_statement(statement s)
{
    hash_table ots =  hash_table_make(hash_int, 101);
    rinitialize_ordering_to_statement(ots, s);
    return ots;
}

void 
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

    hash_table_clear(OrderingToStatement);
    OrderingToStatement = hash_table_undefined;
}
