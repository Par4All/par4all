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
 
  LOOP_UNROLL()

  Bruno Baron, Francois Irigoin
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
#include "properties.h"
#include "preprocessor.h"

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
      pips_error("find_final_statement_label",
		 "Useless empty sequence. Unexpected in controlized code.\n");
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
    pips_error("find_final_statement_label",
	       "Controlized code should not contain GO TO statements\n");
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
    pips_error("find_final_statement_label",
	       "Unknown instruction tag: %d\n",
	       instruction_tag(i));
  }

  /* fsl should be either a meaningful label or the empty label */
  if(entity_undefined_p(fsl)) {
    pips_error("find_final_statement_label",
	       "Undefined final label\n");
  }

  return fsl;
}

/* db_get_current_module_name() unusable because module not set,
 * and setting it causes previous current module to be closed! 
 */
/* static string current_module_name = NULL; */

void do_loop_unroll(statement loop_statement, int rate, void (*statement_post_processor)(statement))
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
        basic indb = variable_basic(type_variable(entity_type(ind)));
        expression lb = range_lower(lr),
                   ub = range_upper(lr),
                   inc = range_increment(lr);
        entity nub, ib, lu_ind;
        expression rhs_expr, expr;
        entity label_entity;
        statement body, stmt;
        instruction block, inst;
        range rg;
        int lbval, ubval, incval;
        bool numeric_range_p = FALSE;
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
        nub = make_new_scalar_variable_with_prefix(NORMALIZED_UPPER_BOUND_NAME, 
                get_current_module_entity(),
                copy_basic(indb)
                /* MakeBasic(is_basic_int)*/);
        AddEntityToCurrentModule(nub);

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
        ib = make_new_scalar_variable_with_prefix(INTERMEDIATE_BOUND_NAME, 
                get_current_module_entity(),
                copy_basic(indb)
                /* MakeBasic(is_basic_int)*/);
        AddEntityToCurrentModule(ib);

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
            lu_ind = make_new_scalar_variable_with_prefix(INDEX_NAME, 
                    get_current_module_entity(),
                    copy_basic(indb)
                    );
        AddEntityToCurrentModule(lu_ind);

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
        replace_entity_by_expression(body,ind,expr);
        free_expression(expr);

        label_entity = make_new_label(get_current_module_name());
        stmt = make_continue_statement(label_entity);
        body = make_block_statement(
                make_statement_list(body,stmt)
                );
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
        label_entity = make_new_label(get_current_module_name());
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
            replace_entity_by_expression(transformed_stmt,ind,expr);

            ifdebug(9) {
                pips_assert("loop_unroll", statement_consistent_p(transformed_stmt));
            }
            free_expression(expr);

            ifdebug(9) {
                pips_assert("loop_unroll", statement_consistent_p(transformed_stmt));
            }
            if( statement_post_processor)
                statement_post_processor(transformed_stmt);


            /* Add the transformated old loop body (transformed_stmt) at
             * the begining of the loop */
            if( get_bool_property("LOOP_UNROLL_MERGE") && instruction_block_p(statement_instruction(transformed_stmt)) ) {
                if(ENDP(statement_declarations(body)))
                    statement_declarations(body)=statement_declarations(transformed_stmt);
                else {
                    list declaration_initializations = NIL;
                    FOREACH(ENTITY,e,statement_declarations(transformed_stmt))
                    {
                        value v = entity_initial(e);
                        if( value_expression_p(v) )
                        {
                            statement s = make_assign_statement(
                                    make_expression_from_entity(e),
                                    copy_expression(value_expression(v)));
                            declaration_initializations=CONS(STATEMENT,s,declaration_initializations);
                        }
                    }
                    declaration_initializations=gen_nreverse(declaration_initializations);
                    instruction_block(statement_instruction(transformed_stmt))=
                        gen_nconc(declaration_initializations,instruction_block(statement_instruction(transformed_stmt)));
                }
                instruction_block(statement_instruction(body)) =
                    gen_nconc( instruction_block(statement_instruction(body)),
                            instruction_block(statement_instruction(transformed_stmt)));
            }
            else {
                instruction_block(statement_instruction(body)) = CONS(STATEMENT,
                        transformed_stmt,
                        body_block);
            }
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
  bool unroll_p = FALSE;
  range lr = loop_range(l);
  expression lb = range_lower(lr);
  expression ub = range_upper(lr);
  expression inc = range_increment(lr);
  int lbval, ubval, incval;

  pips_debug(2, "begin\n");

  unroll_p = expression_integer_value(lb, &lbval)
    && expression_integer_value(ub, &ubval)
    && expression_integer_value(inc, &incval);

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
    int lbval, ubval, incval;
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

        transformed_stmt = copy_statement(loop_body(il));
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
                                make_expression_from_entity(e),
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
	print_text(stderr,text_statement(entity_undefined,0,stmt));
	pips_assert("full_loop_unroll", statement_consistent_p(stmt));
      }
      instruction_block(block)= gen_nconc(instruction_block(block),
					  CONS(STATEMENT, stmt, NIL ));
    }

    /* Generate a statement to reinitialize old index */
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

/* Top-level functions
 */

bool
unroll(char *mod_name)
{
    statement mod_stmt;
    instruction mod_inst;
    char *lp_label = NULL;
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

            /* DBR_CODE will be changed: argument "pure" should take FALSE 
               but this would be useless
               since there is only *one* version of code; a new version 
               will be put back in the
               data base after unrolling */

            mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);
            mod_inst = statement_instruction(mod_stmt);

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
bool apply_full_loop_unroll(statement s)
{
  instruction inst = statement_instruction (s);
  bool go_on = TRUE;

  if(instruction_loop_p(inst)) {
    full_loop_unroll(s);
  }

  return go_on;
}


bool
full_unroll(char * mod_name)
{
    statement mod_stmt;
    char *lp_label = string_undefined;
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
        mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);

        /* prelude */
        set_current_module_entity(module_name_to_entity( mod_name ));
        set_current_module_statement( mod_stmt);

        /* do the job */
        if(entity_undefined_p(lb_ent)) {
            gen_recurse (mod_stmt, statement_domain, 
                    apply_full_loop_unroll, gen_null);
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

bool find_unroll_pragma_and_fully_unroll(statement s)
{
    instruction inst = statement_instruction (s);
    bool go_on = TRUE;

    if(!empty_comments_p(statement_comments(s))
       && strstr(statement_comments(s), FULL_UNROLL_PRAGMA)!=NULL) {
      number_of_requested_unrollings++;
      if(instruction_loop_p(inst)) {
	/* full_loop_unroll() does not work all the time! */
	full_loop_unroll(s);
	number_of_unrolled_loops++;
	go_on = TRUE;
      }
      else {
	/* The full unroll pragma must comment a DO instruction */
	;
	go_on = FALSE;
      }
    }

    return go_on;
}

bool full_unroll_pragma(char * mod_name)
{
  statement mod_stmt = statement_undefined;
  bool return_status = FALSE;

  debug_on("FULL_UNROLL_DEBUG_LEVEL");

  debug(1,"full_unroll_pragma","Fully unroll loops with pragma in module %s\n",
	mod_name);

  /* Keep track of effects on code */
  number_of_unrolled_loops = 0;
  number_of_requested_unrollings = 0;

  /* Perform the loop unrollings */
  mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);

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
    return_status = TRUE;
  }
  else {
    int failures = number_of_requested_unrollings - number_of_unrolled_loops;
    user_log("%d loop%s unrolled as requested\n", number_of_unrolled_loops,
	     number_of_unrolled_loops>1?"s":"");
    user_log("%d loop%s could not be unrolled as requested\n", failures,
	     failures>1?"s":"");
    return_status = FALSE;
  }

  debug(1,"full_unroll_pragma","done for %s\n", mod_name);
  debug_off();

  return return_status;
}
