/** Some passes to transform for-loops into do-loops or while-loops that
    may be easier to analyze by PIPS.
*/

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "control.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"


/* Test if the for-loop initialization is static.

   @return TRUE if yes and a new plb expression of this initialization
   value and the related index variable pli.
*/
static bool init_expression_to_index_and_initial_bound(expression init,
						       entity * pli,
						       expression * plb)
{
  bool success = FALSE;
  syntax init_s = expression_syntax(init);

  pips_debug(5, "Begin\n");

  if(syntax_call_p(init_s)) {
    call c = syntax_call(init_s);
    entity op = call_function(c);

    if(ENTITY_ASSIGN_P(op)) {
      expression lhs = EXPRESSION(CAR(call_arguments(c)));
      syntax lhs_s = expression_syntax(lhs);
      if(syntax_reference_p(lhs_s)) {
	reference r = syntax_reference(lhs_s);
	entity li = reference_variable(r);
	type lit = entity_type(li);

	if(ENDP(reference_indices(r)) /* scalar reference */
	   && type_variable_p(lit)
	   && basic_int_p(variable_basic(type_variable(lit)))) {
	  *pli = li;
	  /* To avoid sharing and intricate free operations */
	  *plb = copy_expression(EXPRESSION(CAR(CDR(call_arguments(c)))));
	  success = TRUE;
	  pips_debug(5, "Static initialization found.\n");
	}
      }
    }
  }

  ifdebug(5) {
    if(success) {
      pips_debug(5, "End with loop index %s and init expression:\n", entity_local_name(*pli));
      print_expression(*plb);
    }
    else {
      pips_debug(5, "End: DO conversion failed with initialization expression\n");
    }
  }

  return success;
}


/* Test if the final value @param li of a for-loop is static.

   @return:
   - TRUE if the final value of a for-loop is static

   - pub is the new expression of the final value

   - p_is_upper_p is set to TRUE if the final value is upper-bound (index
     is increasing)

   - p_is_lower_p is set to TRUE if the final value is loweer-bound (index
     is decreasing)

   Depending on cond, the so-called upper bound may end up being a lower
   bound with a decreasing step loop.
 */
static bool condition_expression_to_final_bound(expression cond,
						entity li,
						bool* p_is_upper_p,
						bool* p_is_lower_p,
						expression * pub)
{
  bool success = FALSE;
  syntax cond_s = expression_syntax(cond);
  bool strict_p = FALSE;

  ifdebug(5) {
    pips_debug(5, "Begin with expression\n");
    print_expression(cond);
  }

  if(syntax_call_p(cond_s)) {
    call c = syntax_call(cond_s);
    entity op = call_function(c);

    /* Four operators are accepted */
    if (ENTITY_LESS_THAN_P(op) || ENTITY_LESS_OR_EQUAL_P(op)
	|| ENTITY_GREATER_THAN_P(op) || ENTITY_GREATER_OR_EQUAL_P(op)) {
      expression e1 = EXPRESSION(CAR(call_arguments(c)));
      expression e2 = EXPRESSION(CAR(CDR(call_arguments(c))));
      syntax e1_s = expression_syntax(e1);
      syntax e2_s = expression_syntax(e2);

      strict_p = ENTITY_LESS_THAN_P(op) || ENTITY_GREATER_THAN_P(op);

      if(syntax_reference_p(e1_s) && reference_variable(syntax_reference(e1_s))==li) {
	*p_is_upper_p = ENTITY_LESS_THAN_P(op) || ENTITY_LESS_OR_EQUAL_P(op);
	*pub = convert_bound_expression(e2, TRUE, ENTITY_LESS_OR_EQUAL_P(op));
	success = TRUE;
      }
      else if(syntax_reference_p(e2_s) && reference_variable(syntax_reference(e2_s))==li) {
	*p_is_lower_p = ENTITY_GREATER_THAN_P(op) || ENTITY_GREATER_OR_EQUAL_P(op);
	*pub = convert_bound_expression(e2, FALSE, ENTITY_GREATER_OR_EQUAL_P(op));
	success = TRUE;
      }
    }
  }

  ifdebug(5) {
    if(success) {
	pips_debug(5, "Static final value found!\n"
		   "\tEnd with expression\n");
      print_expression(cond);
      pips_debug(5, "Loop counter is increasing: %s\n", bool_to_string(*p_is_upper_p));
      pips_debug(5, "Loop counter is decreasing: %s\n", bool_to_string(*p_is_lower_p));
      pips_debug(5, "Loop condition is strict (< or > instead of <= or >=) : %s\n", bool_to_string(strict_p));
    }
    else
      pips_debug(5, "End: no final bound available\n");
  }

  return success;
}


/**
   Test if a for-loop incrementation expression is do-loop compatible and,
   if yes, compute the increment expression.

   Most of this could be trivial by using the transformers but this
   function is to be used from the controlizer where the semantics
   informations are far from being available yet.

   @param incr for-loop incrementation expression
   @param li is the index variable

   @return TRUE if the incrementation is do-loop compatible \`a la Fortran
   - @param is_increasing_p is set to TRUE is the loop index is increasing
   - @param is_decreasing_p is set to TRUE is the loop index is decreasing
   - @param pincrement is genereted with the detected increment value expression
 */
static bool incrementation_expression_to_increment(expression incr,
						   entity li,
						   bool * is_increasing_p,
						   bool * is_decreasing_p,
						   expression * pincrement)
{
  bool success = FALSE;
  syntax incr_s = expression_syntax(incr);

  if(syntax_call_p(incr_s)) {
    call incr_c = syntax_call(incr_s);
    entity op = call_function(incr_c);
    if (! ENDP(call_arguments(incr_c))) {
      expression e = EXPRESSION(CAR(call_arguments(incr_c)));

      /* Look for i++ or ++i: */
      if ((ENTITY_POST_INCREMENT_P(op) || ENTITY_PRE_INCREMENT_P(op))
	  && is_expression_reference_to_entity_p(e,li)) {
	* is_increasing_p = TRUE;
	* pincrement = int_to_expression(1);
	success = TRUE;
	pips_debug(5, "Increment operator found!\n");
      }
      /* Look for i-- or --i: */
      else if ((ENTITY_POST_DECREMENT_P(op) || ENTITY_PRE_DECREMENT_P(op))
	       && is_expression_reference_to_entity_p(e, li)) {
	* is_decreasing_p = TRUE;
	* pincrement = int_to_expression(-1);
	success = TRUE;
	pips_debug(5, "Decrement operator found!\n");
      }
      else if (! ENDP(CDR(call_arguments(incr_c)))) {
	e =  EXPRESSION(CAR(CDR(call_arguments(incr_c))));
	/* Look for "i += integer": */
	if (ENTITY_PLUS_UPDATE_P(op)
	    && extended_integer_constant_expression_p(e)) {
	  int v = expression_to_int(e);
	  if (v != 0) {
	    * pincrement = e;
	    success = TRUE;
	    if (v > 0 ) {
	      * is_increasing_p = TRUE;
	      pips_debug(5, "Found += with positive increment!\n");
	    }
	    else {
	      * is_decreasing_p = TRUE;
	      pips_debug(5, "Found += with negative increment!\n");
	    }
	  }
	}
	/* Look for "i -= integer": */
	else if(ENTITY_MINUS_UPDATE_P(op)
		&& extended_integer_constant_expression_p(e)) {
	  int v = expression_to_int(e);
	  if (v != 0) {
	    * pincrement = int_to_expression(-v);
	    success = TRUE;
	    if (v < 0 ) {
	      * is_increasing_p = TRUE;
	      pips_debug(5, "Found -= with negative increment!\n");
	    }
	    else {
	      * is_decreasing_p = TRUE;
	      pips_debug(5, "Found -= with positive increment!\n");
	    }
	  }
	}
	else {
	  /* Look for "i = i + v" or "i = v + i": */
	  expression inc_v =
	    expression_verbose_incrementation_p_and_return_increment(incr);
	  if (inc_v != expression_undefined
	      && extended_integer_constant_expression_p(inc_v)) {
	    int v = expression_to_int(e);
	    if (v != 0) {
	      * pincrement = inc_v;
	      success = TRUE;
	      if (v > 0 ) {
		* is_increasing_p = TRUE;
		pips_debug(5, "Found \"i = i + v\" or \"i = v + i\" with positive increment!\n");
	      }
	      else {
		* is_decreasing_p = TRUE;
		pips_debug(5, "Found \"i = i + v\" or \"i = v + i\" with negative increment!\n");
		    
	      }
	    }
	  }
	}
      }
    }
  }
  return success;
}


/* Try to convert a C-like for-loop into a Fortran-like do-loop.

   @return the do-loop if the transformation worked or loop_undefined if
   it failed.
 */
loop for_to_do_loop_conversion(expression init,
			       expression cond,
			       expression incr,
			       statement body)
{
  loop l = loop_undefined;
  entity li = entity_undefined; /* loop index */
  range lr = range_undefined; /* loop bound and increment expressions */
  expression lb = expression_undefined;
  expression ub = expression_undefined;
  expression increment = expression_undefined;

  /* Pattern checked:

   * The init expression is an assignment to an integer scalar (not fully necessary)
     It failed on the first submitted for loop which had two assignments as
     initialization step. One of the them was the loop index assignment.
     More work is needed to retrieve all DO loops programmed as for loops.

   * The condition is a relational operation with the index

   * The incrementation expression does not really matter as long as it
   * updates the loop index.

   */

  if (init_expression_to_index_and_initial_bound(init, &li, &lb)) {
      bool is_lower_p = FALSE;
      bool is_upper_p = FALSE;

    if (condition_expression_to_final_bound(cond, li, &is_upper_p,
					    &is_lower_p, &ub)) {
      bool is_increasing_p = FALSE;
      bool is_decreasing_p = FALSE;

      if (incrementation_expression_to_increment(incr, li, &is_increasing_p,
						 &is_decreasing_p,
						 &increment)) {
	if(is_lower_p && is_increasing_p)
	  pips_user_warning("Loop with lower bound and increasing index %s\n",
			    entity_local_name(li));
	if(is_upper_p && is_decreasing_p)
	  pips_user_warning("Loop with upper bound and decreasing index %s\n",
			    entity_local_name(li));
	lr = make_range(lb, ub, increment);
	l = make_loop(li, lr, body, entity_empty_label(),
		      make_execution(is_execution_sequential, UU), NIL);
      }
      else
	/* Remove the lower bound expression that won't be used: */
	free_expression(ub);
    }
    else
      /* Remove the lower bound expression that won't be used: */
      free_expression(lb);
  }

  return l;
}

sequence for_to_while_loop_conversion(expression init,
				      expression cond,
				      expression incr,
				      statement body,
				      string comments) {
  syntax s_init = expression_syntax(init);
  syntax s_cond = expression_syntax(cond);
  syntax s_incr = expression_syntax(incr);
  sequence wlseq = sequence_undefined;

  pips_debug(5, "Begin\n");

  if(!syntax_call_p(s_init)
     || !(syntax_call_p(s_cond)||syntax_reference_p(s_cond))
     || !syntax_call_p(s_incr)) {
    pips_internal_error("expression arguments must be calls "
			"(in C, the condition may be a reference)\n");
  }
  else {
    statement init_st = make_statement(entity_empty_label(),
				       STATEMENT_NUMBER_UNDEFINED,
				       STATEMENT_ORDERING_UNDEFINED,
				       empty_comments,
				       make_instruction(is_instruction_call,
							syntax_call(s_init)),
				       NIL,NULL);
    statement incr_st =  make_statement(entity_empty_label(),
					STATEMENT_NUMBER_UNDEFINED,
					STATEMENT_ORDERING_UNDEFINED,
					empty_comments,
					make_instruction(is_instruction_call,
							 syntax_call(s_incr)),
					NIL,NULL);
    sequence body_seq = make_sequence(CONS(STATEMENT, body,
					CONS(STATEMENT, incr_st, NIL)));
    statement n_body = make_statement(entity_empty_label(),
				      STATEMENT_NUMBER_UNDEFINED,
				      STATEMENT_ORDERING_UNDEFINED,
				      empty_comments,
				      make_instruction(is_instruction_sequence,
						       body_seq),
				     NIL,NULL);
    whileloop wl_i = make_whileloop(cond, n_body, entity_empty_label(),
				    make_evaluation_before() );
    statement wl_st =  make_statement(entity_empty_label(),
				     STATEMENT_NUMBER_UNDEFINED,
				     STATEMENT_ORDERING_UNDEFINED,
				     comments,
				     make_instruction(is_instruction_whileloop,
						      wl_i),
				     NIL,NULL);

    ifdebug(5) {
      pips_debug(5, "Initialization statement:\n");
      print_statement(init_st);
      pips_debug(5, "Incrementation statement:\n");
      print_statement(incr_st);
      pips_debug(5, "New body statement with incrementation:\n");
      print_statement(n_body);
      pips_debug(5, "New whileloop statement:\n");
      print_statement(wl_st);
    }

    wlseq = make_sequence(CONS(STATEMENT, init_st,
			       CONS(STATEMENT, wl_st, NIL)));

    /* Clean-up: only cond is reused */

    /* They cannot be freed because a debugging statement at the end of
       controlize does print the initial statement */

    syntax_call(s_init) = call_undefined;
    syntax_call(s_incr) = call_undefined;
    free_expression(init);
    free_expression(incr);
  }

  ifdebug(5) {
    statement d_st =  make_statement(entity_empty_label(),
				     STATEMENT_NUMBER_UNDEFINED,
				     STATEMENT_ORDERING_UNDEFINED,
				     string_undefined,
				     make_instruction(is_instruction_sequence,
						      wlseq),
				     NIL,NULL);

    pips_debug(5, "End with statement:\n");
    print_statement(d_st);
    statement_instruction(d_st) = instruction_undefined;
    free_statement(d_st);
  }

  return wlseq;
}




/* Try to to transform the C-like for-loops into Fortran-like do-loops.

   Assume we are in a gen_recurse since we use gen_get_recurse_ancestor(wl).
 */
void
try_to_transform_a_for_loop_into_a_do_loop(forloop f) {
  loop new_l = for_to_do_loop_conversion(forloop_initialization(f),
					 forloop_condition(f),
					 forloop_increment(f),
					 forloop_body(f));
  if (!loop_undefined_p(new_l)) {
    pips_debug(3, "do-loop has been generated.\n");

    /* Get the englobing statement of the "for" assuming we are called
       from a gen_recurse()-like function: */
    instruction i = INSTRUCTION(gen_get_recurse_ancestor(f));
    /* Modify the enclosing instruction to be a do-loop instead. It works
       even if we are in a gen_recurse because we are in the bottom-up
       phase of the recursion. */
    instruction_tag(i) = is_instruction_loop;
    instruction_loop(i) = new_l;
    /* Detach informations of the while-loop before freeing it
    whileloop_body(wl) = statement_undefined;
    whileloop_condition(wl) = expression_undefined;: */
    free_forloop(f);
  }
}


/* For-loop to do-loop transformation phase.

   Now use pure local analysis on the code but we could imagine further
   use of semantics analysis some day...
 */
bool
for_loop_to_do_loop(char * module_name) {
  statement module_statement;

  /* Get the true ressource, not a copy, since we modify it in place. */
  module_statement =
    (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

  set_current_module_statement(module_statement);
  entity mod = module_name_to_entity(module_name);
  set_current_module_entity(mod);

  debug_on("FOR_LOOP_TO_DO_LOOP_DEBUG_LEVEL");
  pips_assert("Statement should be OK before...", statement_consistent_p(module_statement));

  /* We need to access to the instruction containing the current
     for-loops, so ask NewGen gen_recurse to keep this informations for
     us: */
  gen_start_recurse_ancestor_tracking();
  /* Iterate on all the for-loops: */
  gen_recurse(module_statement,
	      forloop_domain,
              /* Since for-loop statements can be nested, only restructure in
		 a bottom-up way, : */
	      gen_true,
	      try_to_transform_a_for_loop_into_a_do_loop);
  gen_stop_recurse_ancestor_tracking();

  pips_assert("Statement should be OK after...", statement_consistent_p(module_statement));

  pips_debug(2, "done");

  debug_off();

  /* Reorder the module, because some statements have been replaced. */
  module_reorder(module_statement);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_statement);

  reset_current_module_statement();
  reset_current_module_entity();

  /* Should have worked: */
  return TRUE;
}
