/* replace.c
 *
 * Bruno BARON 29.11.91
 *
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "control.h"

#include "pipsdbm.h"
#include "resources.h"
/* #include "loop_normalize.h" */

#include "transformations.h"

bool simple_ref_eq_p(reference r1, reference r2)
{
    entity ent1 = reference_variable(r1);
    entity ent2 = reference_variable(r2);
    cons *ind1 = reference_indices(r1);
    cons *ind2 = reference_indices(r2);

    return( same_entity_p(ent1, ent2)
	   && ind1 == NIL
	   && ind2 == NIL );
}

/* 
 * statement StatementReplaceReference(statement s)
 *
 * This function can be used only if legality has been asserted
 * ref and next are only read, not written.
 */
void StatementReplaceReference(statement s, reference ref, expression next)
{
    instruction inst = statement_instruction(s);

    debug(5, "StatementReplaceReference", "begin\n");

    switch(instruction_tag(inst)) {
      case is_instruction_block :
	/* legal if no statement redefines ref */
	MAPL( sts, {
	    StatementReplaceReference(STATEMENT(CAR(sts)), ref, next);
	}, instruction_block(inst));
	break;
      case is_instruction_test : {
	  /* legal if no statement redefines ref */
	  test t = instruction_test(inst);
	  ExpressionReplaceReference(test_condition(t), ref, next);
	  StatementReplaceReference(test_true(t), ref, next);
	  StatementReplaceReference(test_false(t), ref, next);
	  break;
      }
      case is_instruction_loop : {
	  /* legal if:
	     - index is not ref
	     - body does not redefine ref 
	     */
	  loop l = instruction_loop(inst);
	  RangeReplaceReference(loop_range(l), ref, next);
	  StatementReplaceReference(loop_body(l), ref, next);
	  break;
      }
      case is_instruction_call :
	CallReplaceReference(instruction_call(inst), ref, next);
	break;
      case is_instruction_goto :
	pips_error("StatementReplaceReference", "case is_instruction_goto");
	break;
      case is_instruction_unstructured :
	pips_error("StatementReplaceReference", 
		   "case is_instruction_unstructured");
	break;
	default : 
	pips_error("StatementReplaceReference", 
		   "Bad instruction tag");
    }
}


/*    ExpressionReplaceReference(e, ref, next)
 * e is the expression in which we replace the reference ref by 
 * the expression next.
 */
void ExpressionReplaceReference(expression e, reference ref, expression next)
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
		  new_syn = gen_copy_tree(expression_syntax(next));
		  if(get_debug_level()>=5) {
		      fprintf(stderr, 
			      "Field syntax of replacing expression: ");
		      print_syntax(new_syn);
		      fprintf(stderr, "\n");
		  }
		  gen_free(s);
		  expression_syntax(e) = new_syn;
		  /* ?? What should happen to expression_normalized(e)? */
	      }
	  }
	  else {
	      MAPL(lexpr, {
		  expression indice = EXPRESSION(CAR(lexpr));
		  ExpressionReplaceReference(indice, ref, next);
	      }, reference_indices(r));
	  }
      }
	break;
      case is_syntax_range :
	pips_error("ExpressionReplaceReference", 
		   "tag syntax_range not implemented\n");
	break;
      case is_syntax_call :
	CallReplaceReference(syntax_call(s), ref, next);
	break;
      default : 
	pips_error("ExpressionReplaceReference", "unknown tag: %d\n", 
		   (int) syntax_tag(expression_syntax(e)));
    }
}


/* RangeReplaceReference() */
void RangeReplaceReference(range r, reference ref, expression next)
{
    expression rl = range_lower(r), ru = range_upper(r);
    expression ri = range_increment(r);

    ExpressionReplaceReference(rl, ref, next);
    ExpressionReplaceReference(ru, ref, next);
    ExpressionReplaceReference(ri, ref, next);
}


/* void CallReplaceReference()
 */
void CallReplaceReference(call c, reference ref, expression next)
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
	pips_error("CallReplaceReference", 
		   "case is_value_symbolic: replacement not implemented\n");
		   */
	/* FI: I'd rather assume, nothing to replace for symbolic constants */
	break;
      case is_value_intrinsic:
      case is_value_unknown:
	/* We assume that it is legal to replace arguments (because it should
	   have been verified with the effects that the index is not WRITTEN).
	   */
	MAPL(a, {
	    ExpressionReplaceReference(EXPRESSION(CAR(a)), ref, next);
	}, call_arguments(c));
	break;
      case is_value_code:
	pips_error("CallReplaceReference", 
		   "case is_value_code: interprocedural replacement impossible\n");
	break;
      default:
	pips_error("CallReplaceReference", "unknown tag: %d\n", 
		   (int) value_tag(vin));

	abort();
    }
}


/*
 * ReplaceReference is the top level function. It takes the module name 
 * as argument. The code of the module is required and modified.
 *
 * Note that the module body is reordered. It is necessary in order to
 * reuse the generated code.
 */
void ReplaceReference(char *mod_name, reference ref, expression next_expr)
{
    statement mod_stat;
    instruction mod_inst;
    cons *blocs = NIL;

    debug_on("REPLACE_REFERENCE_DEBUG_LEVEL");
    debug(1,"ReplaceReference","ReplaceReference for %s\n", mod_name);

    /* Sets the current module to "mod_name". */
    set_current_module_entity(local_name_to_top_level_entity(mod_name));

    /* FI: who calls ReplaceReference? Since there is a put DBR_CODE,
     * the get must be TRUE instead of FALSE
     */
    mod_stat = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);

    mod_inst = statement_instruction(mod_stat);
    pips_assert("ReplaceReference", instruction_unstructured_p(mod_inst));
    /* "unstructured expected\n"); */

    CONTROL_MAP(ctl, {
	statement st = control_statement(ctl);

	debug(5, "ReplaceReference", "will replace in statement %d\n",
	      statement_number(st));
	StatementReplaceReference(st, ref, next_expr);	
    }, unstructured_control(instruction_unstructured(mod_inst)), blocs);

    gen_free_list(blocs);

    /* Reorder the module, because new statements have been generated. */
    module_reorder(mod_stat);

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(mod_name), mod_stat);

    debug(2,"ReplaceReference","Done for %s\n", mod_name);
    debug_off();
}
