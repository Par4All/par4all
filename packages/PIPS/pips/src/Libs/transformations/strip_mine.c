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
/*
 * STRIP_MINING()
 *
 *  Bruno Baron - Corinne Ancourt - Francois Irigoin
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"

#include "boolean.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "conversion.h"
#include "properties.h"
#include "transformations.h"

extern entity selected_label;
extern char *current_module_name;

/* loop_strip_mine():
 *
 * Sharing is (theoretically) avoided when the AST is built
 *
 * Assumption:
 *  - a loop body can be a loop (no need for a block with one statement, a loop)
 */
statement loop_strip_mine(statement loop_statement, int chunk_size, int chunk_number)
{
    loop l = instruction_loop(statement_instruction(loop_statement));
    loop new_l = loop_undefined;
    statement new_s = statement_undefined;
    expression size = expression_undefined;
    expression sizem1 = expression_undefined;
    expression lb = range_lower(loop_range(l));
    expression ub = range_upper(loop_range(l));
    entity index = loop_index(l);
    statement b = loop_body(l);
    entity new_index = entity_undefined;
    
    debug(9, "loop_strip_mine", "begin: chunk_size = %d,chunk_number = %d\n",
	  chunk_size, chunk_number);
    pips_assert("loop_strip_mine", ( chunk_size > 1 && chunk_number == -1 )
		||
		(chunk_size == -1 && chunk_number > 1) );

    if(loop_increment_value(l) != 1)
	user_error("loop_strip_mine",
		   "Loop increment has to be one for strip-mining!\n");

    if(get_debug_level()>=9) {
      print_text(stderr,text_statement(entity_undefined,0,loop_statement,NIL));
	pips_assert("loop_strip_mine", statement_consistent_p(loop_statement));
    }

    /* compute the expression for the chunk size */
    if(chunk_size==-1) {
	expression e_number = int_to_expression(chunk_number);

	size = make_op_exp(MINUS_OPERATOR_NAME, copy_expression(ub), copy_expression(lb));
	size = make_op_exp(PLUS_OPERATOR_NAME, size, copy_expression(e_number));
	size = make_op_exp(DIVIDE_OPERATOR_NAME, size, e_number);
	sizem1 = make_op_exp(MINUS_OPERATOR_NAME, copy_expression(size), int_to_expression(1));
    }
    else {
	size = int_to_expression(chunk_size);
	sizem1 = int_to_expression(chunk_size-1);
    }
    ifdebug(9) {
      pips_debug(8, "size = ");
      print_expression(size);
      pips_debug(8, "sizem1 = ");
      print_expression(sizem1);
    }
    
    /* make sure that the outer loop does not use a continue that will
    end up in the inner loop body */
    loop_label(l)=entity_empty_label();

    /* derive a new loop index (FI: only *one* name :-( */
    new_index = make_new_index_entity(index, "_1");
    AddEntityToDeclarations(new_index, get_current_module_entity());

    /* build the inner loop preserving the initial index set */
    expression fst_min_param=MakeBinaryCall(entity_intrinsic("+"),entity_to_expression(new_index),sizem1);
    expression snd_min_parameter=copy_expression(ub);
    expression min_exp = 
        MakeBinaryCall(entity_intrinsic(MIN_OPERATOR_NAME),fst_min_param,snd_min_parameter);
    new_l = make_loop(index,
		      make_range(entity_to_expression(new_index),
                  min_exp,
				 int_to_expression(1)),
		      b,
		      entity_empty_label(),
		      execution_sequential_p(loop_execution(l)) ?
		      make_execution(is_execution_sequential, UU):
		      make_execution(is_execution_parallel, UU),
		      NIL);

    new_s = instruction_to_statement(make_instruction(is_instruction_loop, new_l));

    ifdebug(8) {
      pips_debug(8, "new inner loop:");
      print_statement(new_s);
      /* print_text(stderr,text_statement(entity_undefined,0,new_s,NIL)); */
      pips_assert("loop_strip_mine", statement_consistent_p(new_s));
    }

    /* update the outer loop */
    loop_index(l) = new_index;
    range_increment(loop_range(l)) = copy_expression(size);
    loop_body(l) = new_s;

    ifdebug(8) {
      print_statement(loop_statement);
      pips_assert("loop_statement consistent_p", statement_consistent_p(loop_statement));
    }

    pips_debug(8, "end\n");
    return(loop_statement);
}



/* Hmmm... I am not sure it is a good idea to put the user_request()s
   here. Did the author want to be able to apply
   loop_chunk_size_and_strip_mine() several times on different loops
   with different strip mining parameters? */
static statement loop_chunk_size_and_strip_mine(list lls,__attribute__((unused)) bool (*unused)(statement))
{

    string resp;
    int kind, factor;
    statement stmt, new_s;
    int chunk_size = -1;
    int chunk_number = -1;

    stmt = STATEMENT(CAR(gen_last(lls)));

    /* Get the strip_mining kind from the user */
    kind=get_int_property("STRIP_MINE_KIND");
    if( kind!=0 && kind!=1 ) {
        resp = user_request("Type of strip-mining:\n - in fixed-size chunks "
                "(enter 0)\n - in a fixed number of chunks (enter 1)");
        if ( empty_string_p(resp) ) {
            goto cancel;
        }
        else {
            if(sscanf(resp, "%d", &kind)!=1 || (kind!= 0 && kind !=1)) {
                pips_user_error("strip_mining kind should be either 0 or 1!\n");
            }
        }
    }

    factor=get_int_property("STRIP_MINE_FACTOR");
    if(factor<=1) {

        /* Get the strip_mining factor from the user */
        resp = user_request("What's the stripe %s?\n"
                "(choose integer greater or egal to 2): ", 
                kind ? "number" : "size");
        if ( empty_string_p(resp) ) {
            goto cancel;
        }
        else {
            if(sscanf(resp, "%d", &factor)!=1 || factor <= 1) {
                user_error("strip_mine", 
                        "stripe size or number should be greater than 2\n");
            }
        }
    }

    if(kind==0) {
        chunk_size = factor;
        chunk_number = -1;
    }
    else {
        chunk_size = -1;
        chunk_number = factor;
    }

    pips_debug(1,"strip mine in %d chunks of size %d \n",
            chunk_number, chunk_size);
    new_s = loop_strip_mine(stmt,chunk_size,chunk_number);
    return(new_s);

cancel:
    user_log("Strip mining has been cancelled.\n");

    /* Return the statement unchanged: */
    return stmt;
}

/* Top-level function
 */

bool strip_mine(char *mod_name)
{
    entity module = module_name_to_entity(mod_name);
    statement mod_stmt;
    const char *lp_label=NULL;
    bool return_status;

    debug_on("STRIP_MINE_DEBUG_LEVEL");

    /* Get the loop label form the user */
    lp_label=get_string_property_or_ask("LOOP_LABEL","Which loop do you want to strip_mine?\n" "(give its label): ");
    if (lp_label[0] == '\0') {
        user_log("Strip mining has been cancelled.\n");
        return_status = false;
    }
    else {
        selected_label = find_label_entity(mod_name, lp_label);
        if (entity_undefined_p(selected_label)) {
            user_error("strip_mine", "loop label `%s' does not exist\n", lp_label);
        }

        set_current_module_entity(module);

        /* DBR_CODE will be changed: argument "pure" should take false but
           this would be useless since there is only *one* version of code;
           a new version will be put back in the data base after
           strip_mineing */
        mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);
        set_current_module_statement(mod_stmt);

        look_for_nested_loop_statements(mod_stmt, loop_chunk_size_and_strip_mine, selected_loop_p);

        /* Reorder the module, because new statements have been generated. */
        module_reorder(mod_stmt);

        reset_current_module_entity();
        reset_current_module_statement();

        DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
        return_status = true;
    }
    
    debug(2,"strip_mine","done for %s\n", mod_name);
    debug_off();

    return return_status;
}
