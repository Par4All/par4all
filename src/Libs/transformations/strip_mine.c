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
#include "database.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"

#include "boolean.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "conversion.h"
/* #include "generation.h" */
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
    char * module_name = db_get_current_module_name();
    entity module = local_name_to_top_level_entity(module_name);
    
    debug(9, "loop_strip_mine", "begin: chunk_size = %d,chunk_number = %d\n",
	  chunk_size, chunk_number);
    pips_assert("loop_strip_mine", ( chunk_size > 1 && chunk_number == -1 ) 
		||
		(chunk_size == -1 && chunk_number > 1) );

    if(loop_increment_value(l) != 1)
	user_error("loop_strip_mine", 
		   "Loop increment has to be one for strip-mining!\n");

    if(get_debug_level()>=9) {
	print_text(stderr,text_statement(entity_undefined,0,loop_statement));
	pips_assert("loop_strip_mine", statement_consistent_p(loop_statement));
    }

    /* compute the expression for the chunk size */
    if(chunk_size==-1) {
	expression e_number = int_to_expression(chunk_number);

	size = MakeBinaryCall(entity_intrinsic("-"), 
			      expression_dup(ub), expression_dup(lb));
	size = MakeBinaryCall(entity_intrinsic("+"), size, e_number);
	size = MakeBinaryCall(entity_intrinsic("/"), 
			      size, expression_dup(e_number));
	sizem1 = MakeBinaryCall(entity_intrinsic("-"), 
				expression_dup(size), int_to_expression(1));
    }
    else {
	size = int_to_expression(chunk_size);
	sizem1 = int_to_expression(chunk_size-1);
    }
    ifdebug(9) {
	(void) fprintf(stderr, "size = ");
	print_expression(size);
	(void) fprintf(stderr, "sizem1 = ");
	print_expression(sizem1);
    }
    
    /* make sure that the outer loop do not use a continue */
    /* that will fall in the inner loop body */
    loop_label(l)=entity_empty_label();

    /* derive a new loop index (FI: only *one* name :-( */
    new_index=make_scalar_integer_entity(
					 strdup(
						concatenate(
							    entity_local_name(index),
							    "_1", 
							    NULL)), module_name);
    add_variable_declaration_to_module(module, new_index);

    /* build the inner loop */
    new_l = make_loop(index, 
		      make_range(entity_to_expression(new_index),
				 MakeBinaryCall(entity_intrinsic("MIN"),
						MakeBinaryCall(entity_intrinsic("+"),entity_to_expression(new_index),sizem1),
						expression_dup(ub)),
				 int_to_expression(1)),
		      b,
		      entity_empty_label(),
		      execution_sequential_p(loop_execution(l)) ?
		      make_execution(is_execution_sequential, UU):
		      make_execution(is_execution_parallel, UU),
		      NIL);

    new_s = make_statement(entity_empty_label(),
			   STATEMENT_NUMBER_UNDEFINED,
			   STATEMENT_ORDERING_UNDEFINED,
			   string_undefined,
			   make_instruction(is_instruction_loop, new_l),NIL,NULL);
    ifdebug(9) {
	(void) fprintf(stderr, "new inner loop");
	print_text(stderr,text_statement(entity_undefined,0,new_s));
	pips_assert("loop_strip_mine", statement_consistent_p(new_s));
    }
    
    /* update the outer loop */
    loop_index(l) = new_index;
    range_increment(loop_range(l)) = expression_dup(size);
    loop_body(l) = new_s;

    if(get_debug_level()>=9) {
	print_text(stderr,text_statement(entity_undefined,0,loop_statement));
	pips_assert("loop_strip_mine", statement_consistent_p(loop_statement));
    }

    debug(9, "loop_strip_mine", "end\n");
    return(loop_statement);
}



/* Hmmm... I am not sure it is a good idea to put the user_request()s
   here. Did the author want to be able to apply
   loop_chunk_size_and_strip_mine() several times on different loops
   with different strip mining parameters? */
statement loop_chunk_size_and_strip_mine(cons *lls)
{

    string resp;
    int kind, factor;
    statement stmt, new_s;
    bool cancel_status = FALSE;
    int chunk_size = -1;
    int chunk_number = -1;


    for (; CDR(lls) != NIL; lls = CDR(lls));
    stmt = STATEMENT(CAR(lls));

    /* Get the strip_mining kind from the user */
    resp = user_request("Type of strip-mining:\n - in fixed-size chunks "
			"(enter 0)\n - in a fixed number of chunks (enter 1)");
    if (resp[0] == '\0') {
	cancel_status = TRUE;
    }
    else {
	/* CA(15/1/93):if(sscanf(resp, "%d", &kind)!=1 || kind !=1) { 
	   replaced by */
	if(sscanf(resp, "%d", &kind)!=1 || (kind!= 0 && kind !=1)) {
	    pips_user_error("strip_mining kind should be either 0 or 1!\n");
	}

	/* Get the strip_mining factor from the user */
	resp = user_request("What's the stripe %s?\n"
			    "(choose integer greater or egal to 2): ", 
			    kind ? "number" : "size");
	if (resp[0] == '\0') {
	    cancel_status = TRUE;
	}
	else {
	    if(sscanf(resp, "%d", &factor)!=1 || factor <= 1) {
		user_error("strip_mine", 
			   "stripe size or number should be greater than 2\n");
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
	}
    }
    if (cancel_status) {
	user_log("Strip mining has been cancelled.\n");
	/* Return the statement unchanged: */
	new_s = stmt;
    }
    else
	new_s = loop_strip_mine(stmt,chunk_size,chunk_number);
   return(new_s);
}

/* Top-level function
 */

bool strip_mine(char *mod_name)
{
    statement mod_stmt;
    char lp_label[6];
    string resp;
    bool return_status;

    debug_on("STRIP_MINE_DEBUG_LEVEL");

    /* Get the loop label form the user */
    resp = user_request("Which loop do you want to strip_mine?\n"
			"(give its label): ");
    if (resp[0] == '\0') {
	user_log("Strip mining has been cancelled.\n");
	return_status = FALSE;
    }
    else {
	sscanf(resp, "%s", lp_label);
	selected_label = find_label_entity(mod_name, lp_label);
	if (selected_label==entity_undefined) {
	    user_error("strip_mine", "loop label `%s' does not exist\n", lp_label);
	}

	/* DBR_CODE will be changed: argument "pure" should take FALSE but
	   this would be useless since there is only *one* version of code;
	   a new version will be put back in the data base after
	   strip_mineing */
	mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);

	look_for_nested_loop_statements(mod_stmt,loop_chunk_size_and_strip_mine,
					selected_loop_p);

	/* Reorder the module, because new statements have been generated. */
	module_reorder(mod_stmt);

	DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stmt);
	return_status = TRUE;
    }
    
    debug(2,"strip_mine","done for %s\n", mod_name);
    debug_off();

    return return_status;
}
