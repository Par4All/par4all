/*

  $Id$

  $Log: loop_unroll.c,v $
  Revision 1.19  1998/10/20 15:01:07  ancourt
  add whileloop entry

  Revision 1.18  1998/09/17 12:16:46  coelho
  type fixed.


 * LOOP_UNROLL()
 *
 * Bruno Baron, Francois Irigoin
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "text-util.h"

#include "boolean.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"

#include "arithmetique.h"

#include "transformations.h"

#define NORMALIZED_UPPER_BOUND_NAME "LU_NUB"
#define INTERMEDIATE_BOUND_NAME "LU_IB"
#define INDEX_NAME "LU_IND"

/* voir make_factor_expression() */
expression make_ref_expr(entity ent, cons *args)
{
    return( make_expression(make_syntax(is_syntax_reference, 
					make_reference(ent, args) ),
			    normalized_undefined ));
}

/* db_get_current_module_name() unusable because module not set,
 * and setting it causes previous current module to be closed! 
 */
/* static string current_module_name = NULL; */

void loop_unroll(statement loop_statement, int rate)
{
    debug(2, "loop_unroll", "unroll %d times\n", rate);
    pips_assert("loop_unroll", 
		instruction_loop_p(statement_instruction(loop_statement)));
    /* "bad argument type\n"); */
    pips_assert("loop_unroll", rate > 0);
    /* "loop unrolling rate not strictly positive: %d\n", rate); */

{
    loop il = instruction_loop(statement_instruction(loop_statement));
    range lr = loop_range(il);
    entity ind = loop_index(il);
    expression lb = range_lower(lr),
               ub = range_upper(lr),
               inc = range_increment(lr);
    entity nub, ib, lu_ind;
    expression rhs_expr, expr;
    string module_name = db_get_current_module_name();
    entity mod_ent = local_name_to_top_level_entity(module_name);
    entity label_entity;
    statement body, stmt;
    instruction block, inst;
    range rg;
    int lbval, ubval, incval;
    bool numeric_range_p = FALSE;

    pips_assert("loop_unroll", mod_ent != entity_undefined);
    /* "module entity undefined\n"); */
    if(get_debug_level()==7) {
	/* Start debug in Newgen */
	gen_debug |= GEN_DBG_CHECK;
    }
    /* Validity of transformation should be checked */
    /* ie.: - pas d'effets de bords dans les expressions duplique'es */

    /* get rid of labels in loop body */
    (void) clear_labels (loop_body (il));

    /* Instruction block is created and will contain everything */
    block = make_instruction_block(NIL);

    /* Entity LU_NUB is created and initializing statement is created
     * LU_NUB = ((UB - LB) + INC)/INC 
     */
    /* 
    nub = make_scalar_integer_entity(NORMALIZED_UPPER_BOUND_NAME, 
				     module_name);
    add_variable_declaration_to_module( mod_ent, nub);
    */
    nub = make_new_scalar_variable_with_prefix(NORMALIZED_UPPER_BOUND_NAME, 
					       mod_ent,
					       MakeBasic(is_basic_int));

    if (expression_integer_value(lb, &lbval) 
	&& expression_integer_value(ub, &ubval) 
	&& expression_integer_value(inc, &incval)) {
	numeric_range_p = TRUE;
	pips_assert("loop_unroll", incval != 0);
	rhs_expr = int_expr(FORTRAN_DIV(ubval-lbval+incval, incval));
    }
    else {
	expr= MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			 MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME), 
					copy_expression(ub), 
					copy_expression(lb) ),
			 copy_expression(inc) );
	rhs_expr = MakeBinaryCall(entity_intrinsic(DIVIDE_OPERATOR_NAME),
				  expr,
				  copy_expression(inc) );
    }
    expr = make_ref_expr(nub, NIL);
    stmt = make_assign_statement(expr,
				 rhs_expr );
    if(get_debug_level()==9) {
	expression_consistent_p(expr);
    }
    /* The first statement gets label of the initial loop */
    statement_label(stmt) = statement_label(loop_statement);
    statement_label(loop_statement) = entity_empty_label();
    instruction_block(block) = CONS(STATEMENT, stmt, NIL);

    /* Entity LU_IB is created and initializing statement is created 
     * LU_IB = MOD(LU_NUB, rate)
     */
    /*
    ib = make_scalar_integer_entity(INTERMEDIATE_BOUND_NAME,
				    module_name);
    add_variable_declaration_to_module( mod_ent, ib);
    */
    ib = make_new_scalar_variable_with_prefix(INTERMEDIATE_BOUND_NAME, 
					      mod_ent,
					      MakeBasic(is_basic_int));

    if (numeric_range_p) {
	rhs_expr = int_expr(FORTRAN_MOD(FORTRAN_DIV(ubval-lbval+incval, 
						    incval), rate));
    }
    else {
	rhs_expr = MakeBinaryCall(entity_intrinsic("MOD"),
				  make_ref_expr(nub, NIL),
				  int_expr(rate));
    }

    expr = make_expression(make_syntax(is_syntax_reference, 
				       make_reference(ib, NIL) ),
			   normalized_undefined);
    stmt = make_assign_statement(expr, rhs_expr);
    instruction_block(block)= gen_nconc(instruction_block(block),
					CONS(STATEMENT, stmt, NIL ));

    /* Loop for some of the first iterations created:
     * DO LU_IND = 0, LU_IB-1, 1
     *    BODY(I\(LU_IND*INC + LB))
     * ENDDO
     */
    /* Entity LU_IND is created */
    /*
    lu_ind = make_scalar_integer_entity(INDEX_NAME,module_name);
    add_variable_declaration_to_module( mod_ent, lu_ind);
    */
    lu_ind = make_new_scalar_variable_with_prefix(INDEX_NAME, 
						  mod_ent,
						  MakeBasic(is_basic_int));

    /* Loop range is created */
    rg = make_range(MakeIntegerConstantExpression("0"),
		    MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				   make_ref_expr(ib, NIL),
				   int_expr(1) ),
		    MakeIntegerConstantExpression("1") );
    if(get_debug_level()>=9) {
	pips_assert("loop_unroll", range_consistent_p(rg));
    }

    /* Create body of the loop, with updated index */
    body = copy_statement(loop_body(il));
    ifdebug(9) {
	pips_assert("loop_unroll", statement_consistent_p(body));
	/* "gen_copy_tree returns bad statement\n"); */
    }
    expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			  MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME), 
					 make_ref_expr(lu_ind, NIL),
					 copy_expression(inc)),
			  copy_expression(lb));
    ifdebug(9) {
	pips_assert("loop_unroll", expression_consistent_p(expr));
	    /* "gen_copy_tree returns bad expression(s)\n"); */
    }
    StatementReplaceReference(body,
			      make_reference(ind,NIL),
			      expr);
    free_expression(expr);

    label_entity = make_new_label(module_name);
    stmt = make_continue_statement(label_entity);
    body = make_block_statement(CONS(STATEMENT, body,
				     CONS(STATEMENT, stmt, NIL)));
    if(get_debug_level()>=9) {
	pips_assert("loop_unroll", statement_consistent_p(body));
    }

    /* Create loop and insert it in block */
    inst = make_instruction(is_instruction_loop,
			    make_loop(lu_ind, 
				      rg, 
				      body, 
				      label_entity,
				      make_execution(is_execution_sequential, 
						     UU),
				      NIL));

    if(get_debug_level()>=9) {
	pips_assert("loop_unroll", instruction_consistent_p(inst));
    }

    instruction_block(block)= gen_nconc(instruction_block(block),
					CONS(STATEMENT,
					     make_stmt_of_instr(inst),
					     NIL ));

    /* Unrolled loop created:
     * DO LU_IND = LU_IB, LU_NUB-1, rate
     *    BODY(I\(LU_IND*INC + LB))
     *    BODY(I\((LU_IND+1)*INC + LB))
     *    ...
     *    BODY(I\((LU_IND+(rate-1))*INC + LB))
     * ENDDO
     */
    /* Loop range is created */
    rg = make_range(make_ref_expr(ib, NIL),
		    MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				   make_ref_expr(nub, NIL),
				   int_expr(1) ),
		    int_expr(rate) );

    /* Create body of the loop, with updated index */
    body = make_empty_block_statement();
    label_entity = make_new_label(module_name);
    instruction_block(statement_instruction(body)) =  
	CONS(STATEMENT, make_continue_statement(label_entity), NIL);
    while(--rate>=0) {
	/* Last transformated old loop added first */
	expression tmp_expr;
	statement transformed_stmt;
	list body_block = instruction_block(statement_instruction(body));

	transformed_stmt = copy_statement(loop_body(il));
	ifdebug(9)
	    statement_consistent_p(transformed_stmt);
	tmp_expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
				  make_ref_expr(lu_ind, NIL),
				  int_expr(rate) );
	expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			      MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME), 
					     tmp_expr,
					     copy_expression(inc) ),
			      copy_expression(lb) );
	ifdebug(9) {
	    pips_assert("loop_unroll", expression_consistent_p(expr));
	}
	StatementReplaceReference(transformed_stmt,
				  make_reference(ind,NIL),
				  expr);
	ifdebug(9) {
	    pips_assert("loop_unroll", statement_consistent_p(transformed_stmt));
	}
	free_expression(expr);
	
	ifdebug(9) {
	    pips_assert("loop_unroll", statement_consistent_p(transformed_stmt));
	}

	
	/* Add the transformated old loop body (transformed_stmt) at
	 * the begining of the loop */
	instruction_block(statement_instruction(body)) = CONS(STATEMENT,
							      transformed_stmt,
							      body_block);
    }

    /* Create loop and insert it in block */
    /* ?? Should execution be the same as initial loop? */
    inst = make_instruction(is_instruction_loop,
			    make_loop(lu_ind, 
				      rg, 
				      body, 
				      label_entity,
				      make_execution(is_execution_sequential, 
						     UU),
				      NIL));

    if(get_debug_level()>=9) {
	pips_assert("loop_unroll", instruction_consistent_p(inst));
    }


    instruction_block(block)= gen_nconc(instruction_block(block),
					CONS(STATEMENT,
					     make_stmt_of_instr(inst),
					     NIL ));

    /* Generate a statement to reinitialize old index 
     * IND = LB + MAX(NUB,0)*INC
     */
    expr = MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
			  MakeBinaryCall(entity_intrinsic(MAX0_OPERATOR_NAME), 
					 make_ref_expr(nub, NIL),
					 int_expr(0) ),
			  copy_expression(inc) );
    rhs_expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			      copy_expression(lb),
			      expr);
    expr = make_ref_expr(ind, NIL);
    stmt = make_assign_statement(expr,
				 rhs_expr );
    ifdebug(9) {
	print_text(stderr,text_statement(entity_undefined,0,stmt));
	pips_assert("loop_unroll", statement_consistent_p(stmt));
    }
    instruction_block(block)= gen_nconc(instruction_block(block),
					CONS(STATEMENT, stmt, NIL ));


    /* Free old instruction and replace with block */
    /* FI: according to me, gen_copy_tree() does not create a copy sharing nothing
     * with its actual parameter; if the free is executed, Pvecteur normalized_linear
     * is destroyed (18 January 1993) */
    /* gen_free(statement_instruction(loop_statement)); */
    statement_instruction(loop_statement) = block;
    /* Do not forget to move forbidden information associated with
       block: */
    fix_sequence_statement_attributes(loop_statement);
    
    ifdebug(9) {
	print_text(stderr,text_statement(entity_undefined,0,loop_statement));
	pips_assert("loop_unroll", statement_consistent_p(loop_statement));
    }
    /* ?? Bad condition */
    if(get_debug_level()==7) {
	/* Stop debug in Newgen */
	gen_debug &= ~GEN_DBG_CHECK;
    }
}
    debug(3, "loop_unroll", "done\n");
}

/* get rid of the loop by body duplication; 
 * the loop body is duplicated as many times
 * as there were iterations
 *
 * FI: could be improved to handle symbolic lower bounds (18 January 1993)
 */
void full_loop_unroll(statement loop_statement)
{
    loop il = instruction_loop(statement_instruction(loop_statement));
    range lr = loop_range(il);
    entity ind = loop_index(il);
    expression lb = range_lower(lr),
               ub = range_upper(lr),
               inc = range_increment(lr);
    expression rhs_expr, expr;
    string module_name = db_get_current_module_name();
    entity mod_ent = local_name_to_top_level_entity(module_name);
    statement stmt;
    instruction block;
    int lbval, ubval, incval;
    int iter;

    debug(2, "full_loop_unroll", "begin\n");
    pips_assert("full_loop_unroll", mod_ent != entity_undefined);
    /* "module entity undefined\n"); */
    if(get_debug_level()==7) {
	/* Start debug in Newgen */
	gen_debug |= GEN_DBG_CHECK;
    }
    /* Validity of transformation should be checked */
    /* ie.: - pas d'effets de bords dans les expressions duplique'es */

    if (expression_integer_value(lb, &lbval) 
	&& expression_integer_value(ub, &ubval) 
	&& expression_integer_value(inc, &incval)) {
	pips_assert("full_loop_unroll", incval != 0);
    }
    else {
	user_error("full_loop_unroll", 
		   "loop range must be numerically known\n");
    }

    /* Instruction block is created and will contain everything */
    block = make_instruction_block(NIL);

    /* get rid of labels in loop body (don't worry, useful labels have
       been transformed into arcs by controlizer, you just loose loop
       labels) */
    (void) clear_labels (loop_body (il));

    for(iter = lbval; iter <= ubval; iter += incval) {
	statement transformed_stmt;

	transformed_stmt = copy_statement(loop_body(il));
	ifdebug(9)
	    statement_consistent_p(transformed_stmt);
	expr = int_to_expression(iter);
	ifdebug(9) {
	    pips_assert("full_loop_unroll", expression_consistent_p(expr));
	}
	StatementReplaceReference(transformed_stmt,
				  make_reference(ind,NIL),
				  expr);
	ifdebug(9) {
	    pips_assert("full_loop_unroll", statement_consistent_p(transformed_stmt));
	}
	free_expression(expr);

	ifdebug(9) {
	    pips_assert("full_loop_unroll", statement_consistent_p(transformed_stmt));
	}
	
	/* Add the transformated old loop body (transformed_stmt) at
	 * the end of the loop */
	 instruction_block(block) = 
	     gen_nconc(instruction_block(block),
		       CONS(STATEMENT, transformed_stmt, NIL));
    }

    /* Generate a statement to reinitialize old index 
     */
    rhs_expr = int_to_expression(iter);
    expr = make_ref_expr(ind, NIL);
    stmt = make_assign_statement(expr, rhs_expr);
    ifdebug(9) {
	print_text(stderr,text_statement(entity_undefined,0,stmt));
	pips_assert("full_loop_unroll", statement_consistent_p(stmt));
    }
    instruction_block(block)= gen_nconc(instruction_block(block),
					CONS(STATEMENT, stmt, NIL ));


    /* Free old instruction and replace with block */
    /* FI: according to me, gen_copy_tree() does not create a copy sharing nothing
     * with its actual parameter; if the free is executed, Pvecteur normalized_linear
     * is destroyed (18 January 1993) */
    free_instruction(statement_instruction(loop_statement));
    statement_instruction(loop_statement) = block;    
    /* Do not forget to move forbidden information associated with
       block: */
    fix_sequence_statement_attributes(loop_statement);
    
    ifdebug(9) {
	print_text(stderr,text_statement(entity_undefined,0,loop_statement));
	pips_assert("full_loop_unroll", statement_consistent_p(loop_statement));
    }
    /* ?? Bad condition */
    if(get_debug_level()==7) {
	/* Stop debug in Newgen */
	gen_debug &= ~GEN_DBG_CHECK;
    }

    debug(3, "full_loop_unroll", "done\n");
}

/* 
 * recursiv_loop_unroll(stmt, lb_ent, rate)
 * if stmt is a loop labeled lb_ent, unrolls it rate times and returns FALSE;
 * else go through the control graph until FALSE is returned; returns TRUE 
 * when not found.
 *
 * BB, 6.12.91
 */
bool recursiv_loop_unroll(statement stmt, entity lb_ent, int rate)
{
    instruction inst = statement_instruction(stmt);
    bool not_done = TRUE;

    debug(8, "recursiv_loop_unroll", "begin with tag %d\n", 
	  instruction_tag(inst));

    switch(instruction_tag(inst)) {
      case is_instruction_block :
	MAPL( sts, {
	    statement s = STATEMENT(CAR(sts));

	    if(!recursiv_loop_unroll(s, lb_ent, rate)) {
		not_done = FALSE;
		break;
	    }
	}, instruction_block(inst));
	break;
      case is_instruction_test : {
	  test t = instruction_test(inst);

	  not_done =  recursiv_loop_unroll(test_true(t), lb_ent, rate) ?
		 recursiv_loop_unroll(test_false(t), lb_ent, rate) : FALSE;

	  break;
      }
      case is_instruction_loop : {
	  entity do_lab_ent = loop_label(instruction_loop(inst));
	  /* is it the right label? */
	  if (do_lab_ent == entity_undefined) {
	      pips_error("recursive_loop_unroll", "DO label undefined\n");
	      not_done = TRUE;
	  }
	  else if (gen_eq(lb_ent, do_lab_ent)) {
	      loop_unroll(stmt, rate);
	      not_done = FALSE;
	  }
	  else {
	      statement b = loop_body(instruction_loop(inst));
	      debug(8, "recursiv_loop_unroll", 
		    "loop labeled %s not unrolled; goind downwards\n", 
		    local_name(entity_name(do_lab_ent)) );
	      not_done = recursiv_loop_unroll(b, lb_ent, rate);
	  }
	  break;
      }
      case is_instruction_call :
	  not_done = TRUE;
	  break;
      case is_instruction_goto :
	  pips_error("recursiv_loop_unroll", 
		       "Unexpected goto\n");
	break;
      case is_instruction_whileloop:
	  pips_error("recursiv_loop_unroll", 
		       "Unexpected while loop - not yet implemented\n");
	break;
      case is_instruction_unstructured :
	  /* ?? What should I do? go down the unstructured!
	   * there is a special macro for that! 
	   */
	  pips_error("recursiv_loop_unroll", "Sorry: unstructured not implemented\n");
	  break;
	default : 
	    pips_error("recursiv_loop_unroll", 
		       "Bad instruction tag\n");
    }

    debug(8, "recursiv_loop_unroll", "end with result: %s\n", 
	  bool_to_string(not_done));

    return not_done;
}


/* Top-level functions
 */

bool
unroll(char *mod_name)
{
    statement mod_stmt;
    instruction mod_inst;
    cons *blocs = NIL;
    char lp_label[6];
    entity lb_ent;
    int rate;
    string resp;
    bool return_status;

    debug_on("UNROLL_DEBUG_LEVEL");

    /* Get the loop label form the user */
    resp = user_request("Which loop do you want to unroll?\n(give its label): ");
    if (resp[0] == '\0')
	/* User asked to cancel: */
	return_status = FALSE;
    else {    
	sscanf(resp, "%s", lp_label);
	lb_ent = find_label_entity(mod_name, lp_label);
	if (lb_ent == entity_undefined)
	    user_error("unroll", "loop label `%s' does not exist\n", lp_label);

	/* Get the unrolling factor from the user */
	resp = user_request("How many times do you want to unroll?\n(choose integer greater or egal to 2): ");
	if (resp[0] == '\0')
	    /* User asked to cancel: */
	    return_status = FALSE;
	else {
	    if(sscanf(resp, "%d", &rate)!=1 || rate <= 1) {
		user_error("unroll", "unroll factor should be greater than 2\n");
	    }

	    debug(1,"unroll","Unroll %d times loop %s in module %s\n",
		  rate, lp_label, mod_name);

	    /* Sets the current module to "mod_name". */
	    /* current_module(local_name_to_top_level_entity(mod_name)); */

	    /* DBR_CODE will be changed: argument "pure" should take FALSE 
	       but this would be useless
	       since there is only *one* version of code; a new version 
	       will be put back in the
	       data base after unrolling */
	    mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);
	    mod_inst = statement_instruction(mod_stmt);

	    switch (instruction_tag (mod_inst)) {

	    case is_instruction_block:
		MAP(STATEMENT, stmt, {recursiv_loop_unroll (stmt, lb_ent, rate);}, 
		    instruction_block (mod_inst));
		break;

	    case is_instruction_unstructured:

		/* go through unstructured and apply recursiv_loop_unroll */
		CONTROL_MAP(ctl, {
		    statement st = control_statement(ctl);

		    debug(5, "unroll", "will replace in statement %d\n",
			  statement_number(st));
		    recursiv_loop_unroll(st, lb_ent, rate);	
		}, unstructured_control(instruction_unstructured(mod_inst)), blocs);
	
		gen_free_list(blocs);
		break;

	    default:
		user_warning ("unroll", "Non-acceptable instruction tag %d\n",
			      instruction_tag (mod_inst));
	    }

	    /* Reorder the module, because new statements have been generated. */
	    module_reorder(mod_stmt);

	    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stmt);
	    return_status = TRUE;
	}
    }
    
    debug(2,"unroll","done for %s\n", mod_name);
    debug_off();

    if (return_status == FALSE)
	user_log("Loop unrolling has been cancelled.\n");
    
    return return_status;
}

static entity searched_loop_label = entity_undefined;

bool find_loop_and_fully_unroll(statement s)
{
    instruction inst = statement_instruction (s);
    bool go_on = TRUE;

    if(instruction_loop_p(inst)) {
	  entity do_lab_ent = loop_label(instruction_loop(inst));
	  /* is it the right label? */
	  if (do_lab_ent == entity_undefined) {
	      pips_error("recursive_loop_unroll", "DO label undefined\n");
	      go_on = FALSE;
	  }
	  else if (gen_eq(searched_loop_label, do_lab_ent)) {
	      full_loop_unroll(s);
	      go_on = FALSE;
	  }
	  else {
	      go_on = TRUE;
	  }
      }

    return go_on;
}


bool
full_unroll(char * mod_name)
{
    statement mod_stmt;
    char lp_label[6];
    string resp;
    entity lb_ent = entity_undefined;
    bool return_status;

    debug_on("FULL_UNROLL_DEBUG_LEVEL");

    /* Get the loop label form the user */
    resp = user_request("Which loop do you want to unroll fully?\n"
			"(give its label): ");
    if (resp[0] == '\0') {
	user_log("Full loop unrolling has been cancelled.\n");
	return_status = FALSE;
    }
    else {    
	sscanf(resp, "%s", lp_label);
	lb_ent = find_label_entity(mod_name, lp_label);
	if (lb_ent == entity_undefined) {
	    user_error("unroll", "loop label `%s' does not exist\n", lp_label);
	}

	debug(1,"full_unroll","Fully unroll loop %s in module %s\n",
	      lp_label, mod_name);

	searched_loop_label = lb_ent;

	mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);

	gen_recurse (mod_stmt, statement_domain, 
		     find_loop_and_fully_unroll, gen_null);

	/* Reorder the module, because new statements have been generated. */
	module_reorder(mod_stmt);

	searched_loop_label = entity_undefined;

	DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stmt);
	return_status = TRUE;
    }

    debug(2,"unroll","done for %s\n", mod_name);
    debug_off();

    return return_status;
}
