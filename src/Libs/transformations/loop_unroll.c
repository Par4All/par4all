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
 
  LOOP_UNROLL()

  Bruno Baron, Francois Irigoin
*/
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "text.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"

#include "boolean.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"

#include "arithmetique.h"
#include "properties.h"

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

/* Find the label associated with the last statement executed within s. */
entity find_final_statement_label(statement s)
{
  entity fsl = entity_undefined;
  instruction i = statement_instruction(s);
  list l = list_undefined;

  switch(instruction_tag(i)) {

  case is_instruction_block:
    if(!ENDP(l=gen_last(instruction_block(i)))) {
      fsl = statement_label(STATEMENT(CAR(l)));
    }
    else {
      /* If empty blocks are allowed, we've recursed one statement too
         far. */
      pips_internal_error("Useless empty sequence. Unexpected in controlized code.");
    }
    break;

  case is_instruction_test:
    /* There cannot be a final statement. The test itself is the final
       instruction. */
    fsl = statement_label(s);
    break;

  case is_instruction_loop:
    fsl = find_final_statement_label(loop_body(instruction_loop(i)));
    break;

  case is_instruction_goto:
    pips_internal_error("Controlized code should not contain GO TO statements");
    break;

  case is_instruction_call:
    fsl = statement_label(s);
    break;

  case is_instruction_unstructured:
    /* Must be the label of the exit node. */
    fsl = find_final_statement_label
      (control_statement(unstructured_exit(instruction_unstructured(i))));
    break;

  default:
    pips_internal_error("Unknown instruction tag: %d",
	       instruction_tag(i));
  }

  /* fsl should be either a meaningful label or the empty label */
  if(entity_undefined_p(fsl)) {
    pips_internal_error("Undefined final label");
  }

  return fsl;
}

static basic basic_int_to_signed_basic(basic b)
{
	pips_assert("input is integer", basic_int_p(b));
	if(basic_int(b) > 10 && basic_int(b) < 20)
		basic_int(b) -= 10;
	return b;
}

/* db_get_current_module_name() unusable because module not set,
 * and setting it causes previous current module to be closed!
 */
/* static string current_module_name = NULL; */


/* This function unrolls DO loop by a constant factor "rate" if their
 * increment is statically known equal to 1. The initial loop is
 * replaced by one unrolled loop followed by an epilogue loop:
 *
 * DO I= LOW, UP
 *
 * becomes
 *
 * DO I = LOW, LOW + RATE * (UP-LOW+1)/RATE -1, RATE
 *
 * DO I = LOW+ RATE * (UP-LOW+1)/RATE, UP
 *
 * where (UP+LOW-1)/RATE is the number of iteration of the unrolled
 * loop.
 *
 * The initial index is reused to preserve as much readability as
 * possible. The loop bounds are not simplified. This is left to other
 * PIPS passes. Partial_eval can (hopefully) simplify the bounds and
 * suppress_dead_code can eliminate the epilogue loop.
 *
 * The upper and lower loop bound expressions are assumed side-effects
 * free.
 */
static void do_loop_unroll_with_epilogue(statement loop_statement,
				  int rate,
				  void (*statement_post_processor)(statement))
{
  loop il = instruction_loop(statement_instruction(loop_statement));
  range lr = loop_range(il);
  entity ind = loop_index(il);
  //basic indb = variable_basic(type_variable(ultimate_type(entity_type(ind))));
  expression low = range_lower(lr),
    up = range_upper(lr);
  //inc = range_increment(lr);
  //entity nub, ib, lu_ind;
  //expression rhs_expr, expr;
  //entity label_entity;
  statement body = statement_undefined;
  statement loop_stmt1 = statement_undefined; // stmt for loop 1,
					      // unrolled loop
  statement loop_stmt2 = statement_undefined; // stmt for loop 2, epilogue
  instruction block;
  range rg1 = range_undefined;
  //range rg2 = range_undefined;
  execution e = loop_execution(il);
  //intptr_t lbval, ubval, incval;
  //bool numeric_range_p = false;

  /* Validity of transformation should be checked */
  /* ie.: - no side effects in replicated expressions */

  /* get rid of labels in loop body */
  (void) clear_labels (loop_body (il));

  /* Instruction block is created and will contain the two new loops */
  block = make_instruction_block(NIL);

  /* compute the new intermediate loop bounds, nlow and nup=nlow-1 */
  /* span = up-low+1 */
  expression span = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			 MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
					copy_expression(up),
					copy_expression(low) ),
				   int_to_expression(1));
  /* count = span/rate */
  expression count = MakeBinaryCall(entity_intrinsic(DIVIDE_OPERATOR_NAME),
				    span,
				    int_to_expression(rate));
  /* nlow = low + rate*count */
  expression nlow =
    MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
		   copy_expression(low),
		   MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
				  int_to_expression(rate),
				  count));
  /* nup = nlow -1 */
  expression nup = MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				  copy_expression(nlow),
				  int_to_expression(1));

  /* The first statement gets the label of the initial loop */
  // This should be postponed and added if necessary to loop_stmt1
  //statement_label(stmt) = statement_label(loop_statement);
  //statement_label(loop_statement) = entity_empty_label();
  //instruction_block(block) = CONS(STATEMENT, stmt, NIL);


  //instruction_block(block)= gen_nconc(instruction_block(block),
  //CONS(STATEMENT,
  //instruction_to_statement(inst),
  //NIL ));

  /* The unrolled loop is generated first:
   *
   * DO I = low, nup, rate
   *    BODY(I\(I + 0))
   *    BODY(I\((I + 1))
   *    ...
   *    BODY(I\(I+(rate-1)))
   * ENDDO
   */
  /* First, the loop range is generated */
  rg1 = make_range(copy_expression(low),
		   nup,
		   int_to_expression(rate));

  /* Create body of the loop, with updated index */
  body = make_empty_block_statement();
  //label_entity = make_new_label(get_current_module_name());
  //instruction_block(statement_instruction(body)) =
  //  CONS(STATEMENT, make_continue_statement(label_entity), NIL);
  int crate = rate; // FI: I hate modifications of a formal parameter
		    // because the stack is kind of corrupted when you debug
  while(--crate>=0) {
    /* Last transformated old loop added first */
    //expression tmp_expr;
    statement transformed_stmt;
    list body_block = instruction_block(statement_instruction(body));

    clone_context cc = make_clone_context(
					  get_current_module_entity(),
					  get_current_module_entity(),
					  NIL,
					  get_current_module_statement() );
    transformed_stmt = clone_statement(loop_body(il), cc);
    free_clone_context(cc);

    ifdebug(9)
      pips_assert("the cloned body is consistent",
		  statement_consistent_p(transformed_stmt));

    // If useful, modify the references to the loop index "ind"
    if(crate>0) {
      expression expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			    make_ref_expr(ind, NIL),
			    int_to_expression(crate));
      ifdebug(9) {
	pips_assert("The expression for the initial index is consistent",
		    expression_consistent_p(expr));
      }
      /* FI: beware of dependent types? 31/10/2014 */
      replace_entity_by_expression(transformed_stmt,ind,expr);

      ifdebug(9) {
	pips_assert("The new loop body is consistent after index substitution",
		    statement_consistent_p(transformed_stmt));
      }
      free_expression(expr);

      ifdebug(9) {
	pips_assert("Still true when expr is freed",
		    statement_consistent_p(transformed_stmt));
      }
    }
    if(statement_post_processor)
      statement_post_processor(transformed_stmt);


    /* Add the transformated old loop body (transformed_stmt) at
     * the begining of the loop
     *
     * For C code, be careful with local declarations and initializations
     */
    if( get_bool_property("LOOP_UNROLL_MERGE")
	&& instruction_block_p(statement_instruction(transformed_stmt)) ) {
      if(ENDP(statement_declarations(body)))
	statement_declarations(body)=statement_declarations(transformed_stmt);
      else {
	list declaration_initializations = NIL;
	FOREACH(ENTITY,e,statement_declarations(transformed_stmt))
	  {
	    value v = entity_initial(e);
	    if( value_expression_p(v) )
	      {
		statement s =
		  make_assign_statement(entity_to_expression(e),
					copy_expression(value_expression(v)));
		declaration_initializations =
		  CONS(STATEMENT,s,declaration_initializations);
	      }
	  }
	declaration_initializations=gen_nreverse(declaration_initializations);
	instruction_block(statement_instruction(transformed_stmt))=
	  gen_nconc(declaration_initializations,
		    instruction_block(statement_instruction(transformed_stmt)));
      }
      instruction_block(statement_instruction(body)) =
	gen_nconc( instruction_block(statement_instruction(body)),
		   instruction_block(statement_instruction(transformed_stmt)));
    }
    else {
      instruction_block(statement_instruction(body)) =
	CONS(STATEMENT, transformed_stmt, body_block);
    }
  }

  /* Create loop and insert it in the top block */
  /* ?? Should execution be the same as initial loop? */
  instruction loop_inst1 = make_instruction(is_instruction_loop,
					    make_loop(ind,
						      rg1,
						      body,
						      entity_empty_label(),
						      copy_execution(e),
						      NIL));

  ifdebug(9) {
    pips_assert("the unrolled loop inst is consistent",
		instruction_consistent_p(loop_inst1));
  }


  loop_stmt1 = instruction_to_statement(loop_inst1);
  statement_label(loop_stmt1)=statement_label(loop_statement);
  statement_label(loop_statement)=entity_empty_label();
  instruction_block(block)= gen_nconc(instruction_block(block),
				      CONS(STATEMENT, loop_stmt1, NIL));

  /* The epilogue loop is generated last:
   *
   * DO I = nlow, up
   *    BODY(I)
   * ENDDO
   */
  loop_stmt2 = make_new_loop_statement(ind,
				       nlow,
				       copy_expression(up),
				       int_to_expression(1),
				       loop_body(il),
				       copy_execution(e));
  instruction_block(block)= gen_nconc(instruction_block(block),
				      CONS(STATEMENT, loop_stmt2, NIL));

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
    print_statement(loop_statement);
    pips_assert("the unrolled code is consistent",
		statement_consistent_p(loop_statement));
  }

  ifdebug(7) {
    /* Stop debug in Newgen */
    gen_debug &= ~GEN_DBG_CHECK;
  }

  pips_debug(3, "done\n");
}

/* This function unrolls any DO loop by a constant factor "rate". If
   the iteration count n cannot statically determined to be a multiple
   of "rate", a first loop executes modulo(n,rate) iterations and then
   a second loop contains the unrolled body. To cope with all posible
   combinations of bounds and increments, the iteration count "n" is
   computed according to Fortran standard and the new iterations are
   numbered from 0 to n-1. The initial loop index is replaced by the
   proper expressions. The generated code is general. It must be
   simplified with passes partial_eval and/or suppress dead_code.

   Khadija Imadoueddine experienced a slowdown with this unrolling
   scheme. She claims that it is due to the prologue loop which leads
   to unaligned accesses in the unrolled loop when the arrays accessed
   are aligned and accessed from the first element to the last.
 */
static void do_loop_unroll_with_prologue(statement loop_statement,
				  int rate,
				  void (*statement_post_processor)(statement))
{
  loop il = instruction_loop(statement_instruction(loop_statement));
  range lr = loop_range(il);
  entity ind = loop_index(il);
  basic indb = variable_basic(type_variable(ultimate_type(entity_type(ind))));
  expression lb = range_lower(lr),
    ub = range_upper(lr),
    inc = range_increment(lr);
  entity nub, ib, lu_ind;
  expression rhs_expr, expr;
  entity label_entity;
  statement body, stmt;
  instruction block, inst;
  range rg;
  intptr_t lbval, ubval, incval;
  bool numeric_range_p = false;

  ifdebug(7) {
    /* Start debug in Newgen */
    gen_debug |= GEN_DBG_CHECK;
  }

  /* Validity of transformation should be checked */
  /* ie.: - no side effects in replicated expressions */

  /* get rid of labels in loop body */
  (void) clear_labels (loop_body (il));

  /* Instruction block is created and will contain everything */
  block = make_instruction_block(NIL);


  /* Entity LU_NUB is created as well as its initializing
   * statement. It contains the number of iterations according to
   * Fortran standard:
   *
   * LU_NUB = ((UB - LB) + INC)/INC
   *
   * This should be OK with C for loops equivalent to a Fortran DO
   * loop.
   *
   * Warning: basic must be signed becuase LU_NUB may be negative
   *
   * FI: Loop Unrolled New Upper Bound? In this code generation scheme
   * the new lower bound is 0.
   */
  nub = make_new_scalar_variable_with_prefix(NORMALIZED_UPPER_BOUND_NAME,
					     get_current_module_entity(),
					     basic_int_to_signed_basic(indb)
					     /* MakeBasic(is_basic_int)*/);
  AddEntityToCurrentModule(nub);

  if (expression_integer_value(lb, &lbval)
      && expression_integer_value(ub, &ubval)
      && expression_integer_value(inc, &incval)) {
    numeric_range_p = true;
    pips_assert("The loop increment is not zero", incval != 0);
    rhs_expr = int_to_expression(FORTRAN_DIV(ubval-lbval+incval, incval));
  }
  else {
    expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
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
  ifdebug(9) {
    pips_assert("The expression for the number of iterations is consistent",
		expression_consistent_p(expr));
  }

  /* The first statement gets the label of the initial loop */
  statement_label(stmt) = statement_label(loop_statement);
  statement_label(loop_statement) = entity_empty_label();
  instruction_block(block) = CONS(STATEMENT, stmt, NIL);


  /* Entity LU_IB is created with its initializing statement
   *
   * LU_IB = MOD(LU_NUB, rate)
   *
   * and it is appended to the block
   *
   * FI: Loop Unrolled Intermediate Bound?
   */
  ib = make_new_scalar_variable_with_prefix(INTERMEDIATE_BOUND_NAME,
					    get_current_module_entity(),
					    copy_basic(indb)
					    /* MakeBasic(is_basic_int)*/);
  AddEntityToCurrentModule(ib);

  if (numeric_range_p) {
    rhs_expr = int_to_expression(FORTRAN_MOD(FORTRAN_DIV(ubval-lbval+incval,
							 incval), rate));
  }
  else {
    rhs_expr = MakeBinaryCall(entity_intrinsic("MOD"),
			      make_ref_expr(nub, NIL),
			      int_to_expression(rate));
  }

  expr = make_expression(make_syntax(is_syntax_reference,
				     make_reference(ib, NIL) ),
			 normalized_undefined);
  stmt = make_assign_statement(expr, rhs_expr);
  instruction_block(block)= gen_nconc(instruction_block(block),
				      CONS(STATEMENT, stmt, NIL ));


  /* An index LU_IND and a loop for the first iterations are created
   * to perform the prologue in case the number of iterations is not a
   * multiple of the unrolling rate:
   *
   * DO LU_IND = 0, LU_IB-1, 1
   *    BODY(I\(LU_IND*INC + LB))
   * ENDDO
   *
   * FI: Loop Unrolled INDex
   */
  lu_ind = make_new_scalar_variable_with_prefix(INDEX_NAME,
						get_current_module_entity(),
						copy_basic(indb));
  AddEntityToCurrentModule(lu_ind);

  /* Loop range is created */
  rg = make_range(int_to_expression(0),
		  MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				 make_ref_expr(ib, NIL),
				 int_to_expression(1) ),
		  int_to_expression(1) );
  ifdebug(9) {
    pips_assert("new range is consistent", range_consistent_p(rg));
  }

  /* Create body of the loop, with updated index */
  clone_context cc = make_clone_context(
					get_current_module_entity(),
					get_current_module_entity(),
					NIL,
					get_current_module_statement() );
  body = clone_statement(loop_body(il), cc);
  free_clone_context(cc);

  ifdebug(9) {
    pips_assert("cloned body is consistent", statement_consistent_p(body));
    /* "gen_copy_tree returns bad statement\n"); */
  }
  /* Substitute the initial index by its value in the body */
  expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
				       make_ref_expr(lu_ind, NIL),
				       copy_expression(inc)),
			copy_expression(lb));
  ifdebug(9) {
    pips_assert("expr is consistent", expression_consistent_p(expr));
    /* "gen_copy_tree returns bad expression(s)\n"); */
  }
  replace_entity_by_expression(body,ind,expr);
  free_expression(expr);

  label_entity = make_new_label(get_current_module_entity());
  stmt = make_continue_statement(label_entity);
  body = make_block_statement(
			      make_statement_list(body,stmt)
			      );
  ifdebug(9) {
    pips_assert("the cloned and substituted body is consistent",
		statement_consistent_p(body));
  }

  /* Create loop and insert it in top block. FI: parallelism could be
     preserved */
  inst = make_instruction(is_instruction_loop,
			  make_loop(lu_ind,
				    rg,
				    body,
				    label_entity,
				    make_execution(is_execution_sequential,
						   UU),
				    NIL));

  ifdebug(9) {
    pips_assert("prologue loop inst is consistent",
		instruction_consistent_p(inst));
  }

  instruction_block(block)= gen_nconc(instruction_block(block),
				      CONS(STATEMENT,
					   instruction_to_statement(inst),
					   NIL ));

  /* The unrolled loop is generated:
   *
   * DO LU_IND = LU_IB, LU_NUB-1, rate
   *    BODY(I\(LU_IND*INC + LB))
   *    BODY(I\((LU_IND+1)*INC + LB))
   *    ...
   *    BODY(I\((LU_IND+(rate-1))*INC + LB))
   * ENDDO
   */
  /* First, the loop range is generated */
  rg = make_range(make_ref_expr(ib, NIL),
		  MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				 make_ref_expr(nub, NIL),
				 int_to_expression(1) ),
		  int_to_expression(rate) );

  /* Create body of the loop, with updated index */
  body = make_empty_block_statement();
  label_entity = make_new_label(get_current_module_entity());
  instruction_block(statement_instruction(body)) =
    CONS(STATEMENT, make_continue_statement(label_entity), NIL);
  while(--rate>=0) {
    /* Last transformated old loop added first */
    expression tmp_expr;
    statement transformed_stmt;
    list body_block = instruction_block(statement_instruction(body));

    clone_context cc = make_clone_context(
					  get_current_module_entity(),
					  get_current_module_entity(),
					  NIL,
					  get_current_module_statement() );
    transformed_stmt = clone_statement(loop_body(il), cc);
    free_clone_context(cc);

    ifdebug(9)
      pips_assert("the cloned body is consistent",
		  statement_consistent_p(transformed_stmt));
    tmp_expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			      make_ref_expr(lu_ind, NIL),
			      int_to_expression(rate) );
    expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			  MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
					 tmp_expr,
					 copy_expression(inc) ),
			  copy_expression(lb) );
    ifdebug(9) {
      pips_assert("The expression for the initial index is consistent",
		  expression_consistent_p(expr));
    }
    replace_entity_by_expression(transformed_stmt,ind,expr);

    ifdebug(9) {
      pips_assert("The new loop body is consistent after index substitution",
		  statement_consistent_p(transformed_stmt));
    }
    free_expression(expr);

    ifdebug(9) {
      pips_assert("Still true when expr is freed",
		  statement_consistent_p(transformed_stmt));
    }
    if(statement_post_processor)
      statement_post_processor(transformed_stmt);


    /* Add the transformated old loop body (transformed_stmt) at
     * the begining of the loop
     *
     * For C code, be careful with local declarations and initializations
     */
    if( get_bool_property("LOOP_UNROLL_MERGE")
	&& instruction_block_p(statement_instruction(transformed_stmt)) ) {
      if(ENDP(statement_declarations(body)))
	statement_declarations(body)=statement_declarations(transformed_stmt);
      else {
	list declaration_initializations = NIL;
	FOREACH(ENTITY,e,statement_declarations(transformed_stmt))
	  {
	    value v = entity_initial(e);
	    if( value_expression_p(v) )
	      {
		statement s =
		  make_assign_statement(entity_to_expression(e),
					copy_expression(value_expression(v)));
		declaration_initializations =
		  CONS(STATEMENT,s,declaration_initializations);
	      }
	  }
	declaration_initializations=gen_nreverse(declaration_initializations);
	instruction_block(statement_instruction(transformed_stmt))=
	  gen_nconc(declaration_initializations,
		    instruction_block(statement_instruction(transformed_stmt)));
      }
      instruction_block(statement_instruction(body)) =
	gen_nconc( instruction_block(statement_instruction(body)),
		   instruction_block(statement_instruction(transformed_stmt)));
    }
    else {
      instruction_block(statement_instruction(body)) =
	CONS(STATEMENT, transformed_stmt, body_block);
    }
  }

  /* Create loop and insert it in the top block */
  /* ?? Should execution be the same as initial loop? */
  inst = make_instruction(is_instruction_loop,
			  make_loop(lu_ind,
				    rg,
				    body,
				    label_entity,
				    make_execution(is_execution_sequential,
						   UU),
				    NIL));

  ifdebug(9) {
    pips_assert("the unrolled loop inst is consistent",
		instruction_consistent_p(inst));
  }


  instruction_block(block)= gen_nconc(instruction_block(block),
				      CONS(STATEMENT,
					   instruction_to_statement(inst),
					   NIL ));

  /* Generate a statement to reinitialize old index
   * IND = LB + MAX(NUB,0)*INC
   */
  expr = MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
			MakeBinaryCall(entity_intrinsic(MAX0_OPERATOR_NAME),
				       make_ref_expr(nub, NIL),
				       int_to_expression(0) ),
			copy_expression(inc) );
  rhs_expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			    copy_expression(lb),
			    expr);
  expr = make_ref_expr(ind, NIL);
  stmt = make_assign_statement(expr,
			       rhs_expr );
  ifdebug(9) {
    print_text(stderr,text_statement(entity_undefined,0,stmt,NIL));
    pips_assert("The old index assignment stmt is consistent",
		statement_consistent_p(stmt));
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
    print_statement(loop_statement);
    pips_assert("the unrolled code is consistent",
		statement_consistent_p(loop_statement));
  }

  ifdebug(7) {
    /* Stop debug in Newgen */
    gen_debug &= ~GEN_DBG_CHECK;
  }

  pips_debug(3, "done\n");
}

void do_loop_unroll(statement loop_statement,
		    int rate,
		    void (*statement_post_processor)(statement))
{
  pips_debug(2, "unroll %d times\n", rate);
  pips_assert("the statement is a loop",
	      instruction_loop_p(statement_instruction(loop_statement)));
  pips_assert("the unrolling factor is strictly positive", rate > 0);

  if(rate>1) {
    loop il = instruction_loop(statement_instruction(loop_statement));
    range lr = loop_range(il);
    expression inc = range_increment(lr);
    intptr_t incval;

    // FI: a property should be checked because epilogue cannot be
    // used if backward compatibility must be maintained
    if(expression_integer_value(inc, &incval) && incval==1
       && !get_bool_property("LOOP_UNROLL_WITH_PROLOGUE"))
      do_loop_unroll_with_epilogue(loop_statement,
				   rate,
				   statement_post_processor);
    else
      do_loop_unroll_with_prologue(loop_statement,
				   rate,
				   statement_post_processor);
  }
  pips_debug(3, "done\n");
}

/* fallbacks on do_loop_unroll
 * without statement post processing
 */
void
loop_unroll(statement loop_statement, int rate)
{
    do_loop_unroll(loop_statement, rate, NULL);
}

bool loop_fully_unrollable_p(loop l)
{
  bool unroll_p = false;
  range lr = loop_range(l);
  expression lb = range_lower(lr);
  expression ub = range_upper(lr);
  expression inc = range_increment(lr);
  intptr_t lbval, ubval, incval;

  pips_debug(2, "begin\n");

  unroll_p = expression_integer_value(lb, &lbval)
    && expression_integer_value(ub, &ubval)
    && expression_integer_value(inc, &incval);

  if(unroll_p) {
    const char* s = get_string_property("FULL_LOOP_UNROLL_EXCEPTIONS");
    if(*s!=0) {
      list callees = statement_to_called_user_entities(loop_body(l));
      list exceptions = string_to_user_modules(s);
      list p = arguments_intersection(callees, exceptions);
      unroll_p = ENDP(p);
      gen_free_list(p);
    }
  }

  return unroll_p;
}

/* get rid of the loop by body duplication;
 *
 * the loop body is duplicated as many times as there were iterations
 *
 * FI: could be improved to handle symbolic lower bounds (18 January 1993)
 */
void full_loop_unroll(statement loop_statement)
{
    loop il = instruction_loop(statement_instruction(loop_statement));
    range lr = loop_range(il);
    entity ind = loop_index(il);
    entity flbl = entity_undefined; /* final loop body label */
    expression lb = range_lower(lr);
    expression ub = range_upper(lr);
    expression inc = range_increment(lr);
    expression rhs_expr, expr;
    statement stmt;
    instruction block;
    intptr_t lbval, ubval, incval;
    int iter;

    pips_debug(2, "begin\n");

    ifdebug(7) {
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
	pips_user_error("loop range for loop %s must be numerically known\n",
			label_local_name(loop_label(il)));
    }

    /* Instruction block is created and will contain everything */
    block = make_instruction_block(NIL);

    /* get rid of labels in loop body: don't worry, useful labels have
       been transformed into arcs by controlizer, you just loose loop
       labels. However, the label of the last statement in the loop body
       might be used by an outer loop and, in doubt, should be preserved. */

    flbl = find_final_statement_label(loop_body(il));

    (void) clear_labels (loop_body (il));

    for(iter = lbval; iter <= ubval; iter += incval) {
        statement transformed_stmt;

        clone_context cc = make_clone_context(
                get_current_module_entity(),
                get_current_module_entity(),
                NIL,
                get_current_module_statement() );
        transformed_stmt = clone_statement(loop_body(il), cc);
        free_clone_context(cc);

        ifdebug(9)
            statement_consistent_p(transformed_stmt);
        expr = int_to_expression(iter);
        ifdebug(9) {
            pips_assert("full_loop_unroll", expression_consistent_p(expr));
        }
        replace_entity_by_expression(transformed_stmt,ind,expr);
        ifdebug(9) {
            pips_assert("full_loop_unroll", statement_consistent_p(transformed_stmt));
        }
        free_expression(expr);

        ifdebug(9) {
            pips_assert("full_loop_unroll", statement_consistent_p(transformed_stmt));
        }

        /* Add the transformated old loop body (transformed_stmt) at
         * the end of the loop */
        if( statement_block_p(transformed_stmt) && get_bool_property("LOOP_UNROLL_MERGE"))
        {
            if(ENDP(statement_declarations(loop_statement)))
                statement_declarations(loop_statement)=statement_declarations(transformed_stmt);
            else {
                list declaration_initializations = NIL;
                FOREACH(ENTITY,e,statement_declarations(transformed_stmt))
                {
                    value v = entity_initial(e);
                    if( value_expression_p(v) )
                    {
                        statement s = make_assign_statement(
                                entity_to_expression(e),
                                copy_expression(value_expression(v)));
                        declaration_initializations=CONS(STATEMENT,s,declaration_initializations);
                    }
                }
                declaration_initializations=gen_nreverse(declaration_initializations);
                instruction_block(statement_instruction(transformed_stmt))=
                    gen_nconc(declaration_initializations,instruction_block(statement_instruction(transformed_stmt)));
            }
            instruction_block(block) =
                gen_nconc( instruction_block(block),
                        instruction_block(statement_instruction(transformed_stmt)));
        }
        else {
            instruction_block(block) = 
                gen_nconc(instruction_block(block),
                        CONS(STATEMENT, transformed_stmt, NIL));
        }
    }

    /* Generate a CONTINUE to carry the final loop body label in case an
       outer loop uses it */
    if(!entity_empty_label_p(flbl)) {
      stmt = make_continue_statement(flbl);
      ifdebug(9) {
	print_text(stderr,text_statement(entity_undefined,0,stmt,NIL));
	pips_assert("full_loop_unroll", statement_consistent_p(stmt));
      }
      instruction_block(block)= gen_nconc(instruction_block(block),
					  CONS(STATEMENT, stmt, NIL ));
    }

    /* Generate a statement to reinitialize old index */
    /* SG: only needed if index is not private */
    if(entity_in_list_p(ind,loop_locals(il))) {
        rhs_expr = int_to_expression(iter);
        expr = make_ref_expr(ind, NIL);
        stmt = make_assign_statement(expr, rhs_expr);
        ifdebug(9) {
            print_text(stderr,text_statement(entity_undefined,0,stmt,NIL));
            pips_assert("full_loop_unroll", statement_consistent_p(stmt));
        }
        instruction_block(block)= gen_nconc(instruction_block(block),
                CONS(STATEMENT, stmt, NIL ));
    }


    /* Free old instruction and replace with block */
    /* FI: according to me, gen_copy_tree() does not create a copy sharing nothing
     * with its actual parameter; if the free is executed, Pvecteur normalized_linear
     * is destroyed (18 January 1993) */
    free_instruction(statement_instruction(loop_statement));
    statement_instruction(loop_statement) = block;
    /* Do not forget to move forbidden information associated with
       block: */
    fix_sequence_statement_attributes(loop_statement);
    clean_up_sequences(loop_statement);

    ifdebug(9) {
      /* FI: how about a simpler print_statement()? */
      print_text(stderr,text_statement(entity_undefined,0,loop_statement,NIL));
	pips_assert("full_loop_unroll", statement_consistent_p(loop_statement));
    }
    /* ?? Bad condition */
    if(get_debug_level()==7) {
	/* Stop debug in Newgen */
	gen_debug &= ~GEN_DBG_CHECK;
    }

    debug(3, "full_loop_unroll", "done\n");
}

/* Top-level functions
 */

bool
unroll(char *mod_name)
{
    statement mod_stmt;
    const char *lp_label = NULL;
    entity lb_ent;
    int rate;
    bool return_status =true;

    debug_on("UNROLL_DEBUG_LEVEL");

    /* Get the loop label form the user */
    lp_label=get_string_property_or_ask("LOOP_LABEL","Which loop do you want to unroll?\n(give its label):");
    if( empty_string_p(lp_label) )
        return_status = false;
    else {
        lb_ent = find_label_entity(mod_name, lp_label);
        if (entity_undefined_p(lb_ent))
            user_error("unroll", "loop label `%s' does not exist\n", lp_label);

        /* Get the unrolling factor from the user */
        rate = get_int_property("UNROLL_RATE");
        if( rate <= 1 ) {
            string resp = user_request("How many times do you want to unroll?\n(choose integer greater or egal to 2): ");

            if (empty_string_p(resp))
                /* User asked to cancel: */
                return_status = false;
            else {
                if(sscanf(resp, "%d", &rate)!=1 || rate <= 1)
                    user_error("unroll", "unroll factor should be greater than 2\n");
            }
        }
        if( return_status ) {

            debug(1,"unroll","Unroll %d times loop %s in module %s\n",
                    rate, lp_label, mod_name);

            /* Sets the current module to "mod_name". */
            /* current_module(module_name_to_entity(mod_name)); */

            /* DBR_CODE will be changed: argument "pure" should take false 
               but this would be useless
               since there is only *one* version of code; a new version 
               will be put back in the
               data base after unrolling */

            mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);

            /* prelude */
            set_current_module_entity(module_name_to_entity( mod_name ));
            set_current_module_statement( mod_stmt);

            /* do the job */
            statement loop_statement = find_loop_from_label(mod_stmt,lb_ent);
            if( ! statement_undefined_p(loop_statement) )
                loop_unroll(loop_statement,rate);
            else
                pips_user_error("label '%s' is not linked to a loop\n", lp_label);

            /* Reorder the module, because new statements have been generated. */
            module_reorder(mod_stmt);

            DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
            /*postlude*/
            reset_current_module_entity();
            reset_current_module_statement();
            return_status = true;
        }
    }

    debug(2,"unroll","done for %s\n", mod_name);
    debug_off();

    if ( ! return_status )
        user_log("Loop unrolling has been cancelled.\n");

    return return_status;
}

static
bool apply_full_loop_unroll(struct _newgen_struct_statement_ * s)
{
  instruction inst = statement_instruction (s);
  bool go_on = true;

  if(instruction_loop_p(inst)) {
    full_loop_unroll(s);
  }

  return go_on;
}


bool
full_unroll(char * mod_name)
{
    statement mod_stmt = statement_undefined;
    const char *lp_label = string_undefined;
    entity lb_ent = entity_undefined;
    bool return_status = true;

    debug_on("FULL_UNROLL_DEBUG_LEVEL");

    /* user interaction */
    lp_label = get_string_property_or_ask("LOOP_LABEL","loop to fully unroll ?");
    if( empty_string_p(lp_label) )
        return_status = false;
    else {
        lb_ent = find_label_entity(mod_name,lp_label);
        if( entity_undefined_p(lb_ent) )
            pips_user_warning("loop label `%s' does not exist, unrolling all loops\n", lp_label);
        mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);

        /* prelude */
        set_current_module_entity(module_name_to_entity( mod_name ));
        set_current_module_statement( mod_stmt);

        /* do the job */
        if(entity_undefined_p(lb_ent)) {
            gen_recurse (mod_stmt, statement_domain, 
                    (bool(*)(void*))apply_full_loop_unroll, gen_null);
        }
        else {
            statement loop_statement = find_loop_from_label(mod_stmt,lb_ent);
            if( statement_undefined_p(loop_statement) )
                pips_user_error("label '%s' is not put on a loop\n",lp_label);
            else
                full_loop_unroll(loop_statement);
        }

        /* Reorder the module, because new statements have been generated. */
        module_reorder(mod_stmt);

        DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
        /*postlude*/
        reset_current_module_entity();
        reset_current_module_statement();
    }
    if( !return_status)
        user_log("transformation has been cancelled\n");
    debug(2,"unroll","done for %s\n", mod_name);
    debug_off();
        /* Reorder the module, because new statements have been generated. */
        module_reorder(mod_stmt);

    return return_status;
}



static int number_of_unrolled_loops = 0;
static int number_of_requested_unrollings = 0;

/* C must be a capital C... This should be improved with
   regular expressions, with a property to set the pragma or comment...

   This algorithm is more efficient if we have far more loops than
   comments... :-)

   Furthermore, people want to unroll only loops that are designated or
   all loops but those that are designated.

   Loops can be tagged by the loop index, the loop label or a pragma.

   A lot of improvement ahead before we can think of matching TSF
   flexibility... */
#define FULL_UNROLL_PRAGMA "Cxxx"

static bool find_unroll_pragma_and_fully_unroll(statement s)
{
    instruction inst = statement_instruction (s);
    bool go_on = true;

    if(!empty_comments_p(statement_comments(s))
       && strstr(statement_comments(s), FULL_UNROLL_PRAGMA)!=NULL) {
      number_of_requested_unrollings++;
      if(instruction_loop_p(inst)) {
	/* full_loop_unroll() does not work all the time! */
	full_loop_unroll(s);
	number_of_unrolled_loops++;
	go_on = true;
      }
      else {
	/* The full unroll pragma must comment a DO instruction */
	;
	go_on = false;
      }
    }

    return go_on;
}

bool full_unroll_pragma(char * mod_name)
{
  set_current_module_entity(module_name_to_entity(mod_name));
  set_current_module_statement( (statement) db_get_memory_resource(DBR_CODE, mod_name, true) );

  statement mod_stmt = statement_undefined;
  bool return_status = false;

  debug_on("FULL_UNROLL_DEBUG_LEVEL");

  debug(1,"full_unroll_pragma","Fully unroll loops with pragma in module %s\n",
	mod_name);

  /* Keep track of effects on code */
  number_of_unrolled_loops = 0;
  number_of_requested_unrollings = 0;

  /* Perform the loop unrollings */
  mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);

  // gen_recurse (mod_stmt, statement_domain, 
  // find_unroll_pragma_and_fully_unroll, gen_null);

  /* Perform the transformation bottom up to reduce the scanning and the
     number of unrollings */
  gen_recurse (mod_stmt, statement_domain, 
	       gen_true, find_unroll_pragma_and_fully_unroll);

  /* Reorder the module, because new statements have been generated. */
  module_reorder(mod_stmt);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);

  /* Provide statistics about changes performed */
  if(number_of_unrolled_loops == number_of_requested_unrollings) {
    user_log("%d loop%s unrolled as requested\n", number_of_unrolled_loops,
	     number_of_unrolled_loops>1?"s":"");
    return_status = true;
  }
  else {
    int failures = number_of_requested_unrollings - number_of_unrolled_loops;
    user_log("%d loop%s unrolled as requested\n", number_of_unrolled_loops,
	     number_of_unrolled_loops>1?"s":"");
    user_log("%d loop%s could not be unrolled as requested\n", failures,
	     failures>1?"s":"");
    return_status = false;
  }

  debug(1,"full_unroll_pragma","done for %s\n", mod_name);
  debug_off();

  reset_current_module_entity();
  reset_current_module_statement();

  return return_status;
}
