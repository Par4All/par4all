
#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "semantics.h"
#include "effects-generic.h"
#include "transformations.h"

#include "control.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "sac.h"

#include "properties.h"

#include <limits.h>

#include "text-util.h"

#define NORMALIZED_UPPER_BOUND_NAME "LU_NUB"
#define INTERMEDIATE_BOUND_NAME "LU_IB"
#define INDEX_NAME "LU_IND"

static void loCallReplaceReference(call c, reference ref, expression next);

static void loExpressionReplaceReference(expression e, reference ref, expression next);

/* 
 * statement loStatementReplaceReference(statement s)
 *
 * This function can be used only if legality has been asserted
 * ref and next are only read, not written.
 */
static void loStatementReplaceReference(statement s, reference ref, expression next)
{
    instruction inst = statement_instruction(s);

    debug(5, "loStatementReplaceReference", "begin\n");

    switch(instruction_tag(inst)) {
      case is_instruction_block :
	/* legal if no statement redefines ref */
	MAPL( sts, {
	    loStatementReplaceReference(STATEMENT(CAR(sts)), ref, next);
	}, instruction_block(inst));
	break;
      case is_instruction_test : {
	  /* legal if no statement redefines ref */
	  test t = instruction_test(inst);
	  loExpressionReplaceReference(test_condition(t), ref, next);
	  loStatementReplaceReference(test_true(t), ref, next);
	  loStatementReplaceReference(test_false(t), ref, next);
	  break;
      }
      case is_instruction_loop : {
	  /* legal if:
	     - index is not ref
	     - body does not redefine ref 
	     */
	  loop l = instruction_loop(inst);
	  RangeReplaceReference(loop_range(l), ref, next);
	  loStatementReplaceReference(loop_body(l), ref, next);
	  break;
      }   
    case is_instruction_whileloop : {
	whileloop l = instruction_whileloop(inst);
	loExpressionReplaceReference(whileloop_condition(l), ref, next);
	loStatementReplaceReference(whileloop_body(l), ref, next);
	break;
    }
      case is_instruction_call :
	loCallReplaceReference(instruction_call(inst), ref, next);
	break;
      case is_instruction_goto :
	pips_error("loStatementReplaceReference", "case is_instruction_goto");
	break;
    case is_instruction_unstructured : {
      list nodes = NIL;
      unstructured u = instruction_unstructured(inst);
      control entry_node = unstructured_control(u);

      FORWARD_CONTROL_MAP(c, {
	statement st = control_statement(c);

	loStatementReplaceReference(st, ref, next);
      }, entry_node, nodes);
      gen_free_list(nodes);
      /* pips_error("loStatementReplaceReference", 
	 "case is_instruction_unstructured"); */
	break;
    }
	default : 
	pips_error("loStatementReplaceReference", 
		   "Bad instruction tag");
    }
}


/*    loExpressionReplaceReference(e, ref, next)
 * e is the expression in which we replace the reference ref by 
 * the expression next.
 */
static void loExpressionReplaceReference(expression e, reference ref, expression next)
{
    syntax s = expression_syntax(e);

    switch(syntax_tag(s)) {
      case is_syntax_reference : {
	  reference r = syntax_reference(s);
	  /* replace if equal to ref */
	  if ( reference_indices(r) == NIL ) {
	      if ( simple_ref_eq_p(syntax_reference(s), ref)) {
		  syntax new_syn;
		  /* s replaced by expression_syntax(next) */
		  new_syn = copy_syntax(expression_syntax(next));
		  if(get_debug_level()>=5) {
		      fprintf(stderr, 
			      "Field syntax of replacing expression: ");
		      print_syntax(new_syn);
		      fprintf(stderr, "\n");
		  }
		  free_syntax(s);
		  expression_syntax(e) = new_syn;
		  /* ?? What should happen to expression_normalized(e)? */
	      }
	  }
	  else {
	      MAPL(lexpr, {
		  expression indice = EXPRESSION(CAR(lexpr));
		  loExpressionReplaceReference(indice, ref, next);
	      }, reference_indices(r));
	  }
      }
	break;
      case is_syntax_range :
	loExpressionReplaceReference(range_lower(syntax_range(s)), ref, next);
	loExpressionReplaceReference(range_upper(syntax_range(s)), ref, next);
	loExpressionReplaceReference(range_increment(syntax_range(s)), ref, next);
	/*
	pips_error("loExpressionReplaceReference", 
		   "tag syntax_range not implemented\n");
	*/
	break;
      case is_syntax_call :
	loCallReplaceReference(syntax_call(s), ref, next);
	break;
      default : 
	pips_error("loExpressionReplaceReference", "unknown tag: %d\n", 
		   (int) syntax_tag(expression_syntax(e)));
    }
}


/* void loCallReplaceReference()
 */
static void loCallReplaceReference(call c, reference ref, expression next)
{
    value vin;
    entity f;

    f = call_function(c);
    vin = entity_initial(f);
	
    switch (value_tag(vin)) {
      case is_value_constant:
	/* nothing to replace */
	break;
      case is_value_symbolic:
	/* 
	pips_error("loCallReplaceReference", 
		   "case is_value_symbolic: replacement not implemented\n");
		   */
	/* FI: I'd rather assume, nothing to replace for symbolic constants */
	break;
      case is_value_intrinsic:
      case is_value_code:
      case is_value_unknown:
	/* We assume that it is legal to replace arguments (because it should
	   have been verified with the effects that the index is not WRITTEN).
	   */
	MAPL(a, {
	    loExpressionReplaceReference(EXPRESSION(CAR(a)), ref, next);
	}, call_arguments(c));
	break;
	/*      case is_value_code:
	pips_error("loCallReplaceReference", 
		   "case is_value_code: interprocedural replacement for"
		   " call to \"%s\" is impossible\n", module_local_name(f));
	break;*/
      default:
	pips_error("loCallReplaceReference", "unknown tag: %d\n", 
		   (int) value_tag(vin));

	abort();
    }
}

static void addSimdCommentToStat(statement s, int num)
{
   char* comment = malloc(256);

   sprintf(comment, "c %s%d\n", SIMD_COMMENT, num);

   insert_comments_to_statement(s, comment);
}

static void addSimdCommentToStats(statement s)
{
   if(instruction_sequence_p(statement_instruction(s)))
   {
      sequence seq = instruction_sequence(statement_instruction(s));
      int num = 0;

      MAP(STATEMENT, curStat,
      {
	addSimdCommentToStat(curStat, num);
	num++;
      }, sequence_statements(seq));
   }
   else
   {
      addSimdCommentToStat(s, 0);
   }
}

void simd_loop_unroll(statement loop_statement, int rate)
{
    debug(2, "simd_loop_unroll", "unroll %d times\n", rate);
    pips_assert("simd_loop_unroll", 
		instruction_loop_p(statement_instruction(loop_statement)));
    /* "bad argument type\n"); */
    pips_assert("simd_loop_unroll", rate > 0);
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
    string module_name = db_get_current_module_name();
    entity mod_ent = module_name_to_entity(module_name);
    entity label_entity;
    statement body, stmt;
    instruction block, inst;
    range rg;
    int lbval, ubval, incval;
    bool numeric_range_p = FALSE;

    pips_assert("simd_loop_unroll", mod_ent != entity_undefined);
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
					       copy_basic(indb)
					       /* MakeBasic(is_basic_int)*/);

    //add_variable_declaration_to_module( mod_ent, nub);

    if (expression_integer_value(lb, &lbval) 
	&& expression_integer_value(ub, &ubval) 
	&& expression_integer_value(inc, &incval)) {
	numeric_range_p = TRUE;
	pips_assert("simd_loop_unroll", incval != 0);
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
					      copy_basic(indb)
					      /* MakeBasic(is_basic_int)*/);

    //add_variable_declaration_to_module( mod_ent, ib);

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
						  copy_basic(indb)
						  /* MakeBasic(is_basic_int)*/);

    //add_variable_declaration_to_module( mod_ent, lu_ind);

    /* Loop range is created */
    rg = make_range(MakeIntegerConstantExpression("0"),
		    MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				   make_ref_expr(ib, NIL),
				   int_expr(1) ),
		    MakeIntegerConstantExpression("1") );
    if(get_debug_level()>=9) {
	pips_assert("simd_loop_unroll", range_consistent_p(rg));
    }

    /* Create body of the loop, with updated index */
    body = copy_statement(loop_body(il));
    ifdebug(9) {
	pips_assert("simd_loop_unroll", statement_consistent_p(body));
	/* "gen_copy_tree returns bad statement\n"); */
    }
    expr = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
			  MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME), 
					 make_ref_expr(lu_ind, NIL),
					 copy_expression(inc)),
			  copy_expression(lb));
    ifdebug(9) {
	pips_assert("simd_loop_unroll", expression_consistent_p(expr));
	    /* "gen_copy_tree returns bad expression(s)\n"); */
    }
    loStatementReplaceReference(body,
			      make_reference(ind,NIL),
			      expr);
    free_expression(expr);

    label_entity = make_new_label(module_name);
    stmt = make_continue_statement(label_entity);
    body = make_block_statement(CONS(STATEMENT, body,
				     CONS(STATEMENT, stmt, NIL)));
    if(get_debug_level()>=9) {
	pips_assert("simd_loop_unroll", statement_consistent_p(body));
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
	pips_assert("simd_loop_unroll", instruction_consistent_p(inst));
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
	    pips_assert("simd_loop_unroll", expression_consistent_p(expr));
	}
	loStatementReplaceReference(transformed_stmt,
				  make_reference(ind,NIL),
				  expr);

	addSimdCommentToStats(transformed_stmt);

	ifdebug(9) {
	    pips_assert("simd_loop_unroll", statement_consistent_p(transformed_stmt));
	}
	free_expression(expr);
	
	ifdebug(9) {
	    pips_assert("simd_loop_unroll", statement_consistent_p(transformed_stmt));
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
	pips_assert("simd_loop_unroll", instruction_consistent_p(inst));
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
	pips_assert("simd_loop_unroll", statement_consistent_p(stmt));
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
	pips_assert("simd_loop_unroll", statement_consistent_p(loop_statement));
    }
    /* ?? Bad condition */
    if(get_debug_level()==7) {
	/* Stop debug in Newgen */
	gen_debug &= ~GEN_DBG_CHECK;
    }
}
    debug(3, "simd_loop_unroll", "done\n");
}
