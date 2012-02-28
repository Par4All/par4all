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




/* Test if the final value @param li of a for-loop is static.

   @return:
   - true if the final value of a for-loop is static

   - pub is the new expression of the final value

   - *is_upper_p is set to true if the final value is upper-bound (index
     is probably increasing) and to false if the final value is
     lower-bound (index is probably decreasing)

   Depending on cond, the so-called upper bound may end up being a lower
   bound with a decreasing step loop.
 */
static bool condition_expression_to_final_bound(expression cond,
						entity li,
						bool * is_upper_p,
						expression * pub)
{
  bool success = false;
  syntax cond_s = expression_syntax(cond);
  bool strict_p = false;

  ifdebug(5) {
    pips_debug(5, "Begin with expression\n");
    print_expression(cond);
  }

  if(syntax_call_p(cond_s)) {
    call c = syntax_call(cond_s);
    entity op = call_function(c);

    /* Five operators are accepted */
    if (ENTITY_LESS_THAN_P(op) || ENTITY_LESS_OR_EQUAL_P(op)
	|| ENTITY_GREATER_THAN_P(op) || ENTITY_GREATER_OR_EQUAL_P(op)
	// FI: wrong result for !=; no way to derive a correct bound
	// without information about the increment
	/*|| ENTITY_NON_EQUAL_P(op)*/ ) {
      expression e1 = EXPRESSION(CAR(call_arguments(c)));
      expression e2 = EXPRESSION(CAR(CDR(call_arguments(c))));
      syntax e1_s = expression_syntax(e1);
      syntax e2_s = expression_syntax(e2);

      strict_p = ENTITY_LESS_THAN_P(op) || ENTITY_GREATER_THAN_P(op) || ENTITY_NON_EQUAL_P(op);

      if (syntax_reference_p(e1_s)
	  && reference_variable(syntax_reference(e1_s)) == li ) {
	*is_upper_p = ENTITY_LESS_THAN_P(op)
	  || ENTITY_LESS_OR_EQUAL_P(op);
	*pub = convert_bound_expression(e2, *is_upper_p, !strict_p);
	success = true;
      }
      else if (syntax_reference_p(e2_s)
	       && reference_variable(syntax_reference(e2_s)) == li) {
	*is_upper_p = ENTITY_GREATER_THAN_P(op)
	  || ENTITY_GREATER_OR_EQUAL_P(op);
	*pub = convert_bound_expression(e1, *is_upper_p, !strict_p);
	success = true;
      }
    }
  }

  ifdebug(5) {
    if(success) {
      pips_debug(5, "Static final value found!\n"
		 "\tEnd with expression:\n");
      print_expression(*pub);
      pips_debug(5, "Loop counter is probably %s\n",
		 *is_upper_p ? "increasing" : "decreasing");
      pips_debug(5, "Loop condition is strict (< or > instead of <= or >=) : %s\n", bool_to_string(strict_p));
    }
    else
      pips_debug(5, "End: no final bound available\n");
  }

  return success;
}


/*
   Test if a for-loop incrementation expression is do-loop compatible
   (i.e. the number of iterations can be computed before entering the
   loop and hence the increment sign is known and the increment is
   unchanged by the loop iterations) and, if yes, compute the
   increment expression. We *have* to fail on symbolic increments with
   unknown sign because we won't be able to restore a correct code.

   Most of this could be trivial by using the transformers but this
   function is to be used from the controlizer where the semantics
   informations are far from being available yet.

   @param incr for-loop incrementation expression
   @param li is the index variable

   @return true if the incrementation is do-loop compatible \`a la Fortran
   - @param is_increasing_p is set to true is the loop index is increasing
   - @param pincrement is generated with the detected increment value expression
 */
static bool incrementation_expression_to_increment(expression incr,
						   entity li,
						   bool * is_increasing_p,
						   expression * pincrement)
{
    bool success = false;
    syntax incr_s = expression_syntax(incr);

    if (syntax_call_p(incr_s)) {
        call incr_c = syntax_call(incr_s);
        entity op = call_function(incr_c);
        if (! ENDP(call_arguments(incr_c))) {
            expression e = EXPRESSION(CAR(call_arguments(incr_c)));

            /* The expression should concern the loop index: */
            if (is_expression_reference_to_entity_p(e,li)) {
                /* Look for i++ or ++i: */
                if ((ENTITY_POST_INCREMENT_P(op) || ENTITY_PRE_INCREMENT_P(op))) {
                    * is_increasing_p = true;
                    * pincrement = int_to_expression(1);
                    success = true;
                    pips_debug(5, "Increment operator found!\n");
                }
                /* Look for i-- or --i: */
                else if ((ENTITY_POST_DECREMENT_P(op) || ENTITY_PRE_DECREMENT_P(op))) {
                    * is_increasing_p = false;
                    * pincrement = int_to_expression(-1);
                    success = true;
                    pips_debug(5, "Decrement operator found!\n");
                }
                else if (! ENDP(CDR(call_arguments(incr_c)))) {
                    /* Look for stuff like "i += integer". Get the rhs: */
                    /* This fails with floating point indices as found in
                       industrial code and with symbolic increments as in
                       "for(t=0.0; t<t_max; t += delta_t). Floating point
                       indices should be taken into account. The iteration
                       direction of the loop should be derived from the bound
                       expression, using the comparison operator. */
                    expression inc_v =  EXPRESSION(CAR(CDR(call_arguments(incr_c))));
                    bool entity_plus_update_p = ENTITY_PLUS_UPDATE_P(op);
                    bool entity_minus_update_p = ENTITY_MINUS_UPDATE_P(op);
                    if ( (entity_plus_update_p||entity_minus_update_p) ) {
                        if(expression_constant_p(inc_v)) {
			    int v = expression_to_int(inc_v);
			    // bool eval_p = expression_integer_value(inc_v);
			    //pips_assert("The expression can be evaluated",
			    //		eval_p);
                            if (v != 0) {
                                int sign = entity_plus_update_p ? 1 : -1 ;
                                * pincrement = int_to_expression(sign * v);
                                success = true;
                                if (v > 0 ) {
                                    * is_increasing_p = entity_plus_update_p;
                                    pips_debug(5, "Found += with positive increment!\n");
                                }
                                else {
                                    * is_increasing_p = entity_minus_update_p;
                                    pips_debug(5, "Found += with negative increment!\n");
                                }
                            }
                        }
                        /* SG: we checked the no-write-effect-on-increment earlier, we can go on safely,
                         * but we will not know if the increment is positive or not, assume yes ?
                         */
                        else {
			  // We must know the sign of inc_v and be
			  // sure that its value cannot be changed by the loop body
			  bool pos_p = positive_expression_p(inc_v);
			  bool neg_p = negative_expression_p(inc_v);
			  if(pos_p || neg_p) {
			    if(entity_minus_update_p) {
			      // FI: I assume we are not dealing with pointers
			      entity uminus = entity_intrinsic(UNARY_MINUS_OPERATOR_NAME);
			      * pincrement = MakeUnaryCall(uminus, copy_expression(inc_v));
			    }
			    else
			      * pincrement = copy_expression(inc_v);
			    *is_increasing_p = (entity_plus_update_p && pos_p)
			      || (entity_minus_update_p && neg_p);
			    success = true;
			  }
			  else
                            // Here we have to fail to be safe ! See validation/C_syntax/decreasing_loop01
                            success = false;
                        }
                    }
                    else {
                        /* Look for "i = i + v" (only for v positive here...)
                           or "i = v + i": */
                        expression inc_v = expression_verbose_reduction_p_and_return_increment(incr, add_expression_p);
                        if (inc_v != expression_undefined ) {
                            if(extended_integer_constant_expression_p(inc_v)) {
                                int v = expression_to_int(inc_v);
                                if (v != 0) {
                                    * pincrement = inc_v;
                                    success = true;
                                    if (v > 0 ) {
                                        * is_increasing_p = true;
                                        pips_debug(5, "Found \"i = i + v\" or \"i = v + i\" with positive increment!\n");
                                    }
                                    else {
                                        * is_increasing_p = false;
                                        pips_debug(5, "Found \"i = i + v\" or \"i = v + i\" with negative increment!\n");
                                    }
                                }
                            }
                            /* SG: we checked the no-write-effect-on-increment earlier, we can go on safely,
                             * but we will not know if the increment is positive or not, assume yes ?
                             */
                            else {
                                * pincrement = copy_expression(inc_v);
				if(positive_expression_p(inc_v)) {
				  * is_increasing_p = true;
				  success = true;
				}
				else if(negative_expression_p(inc_v)) {
				  * is_increasing_p = false;
				  success = true;
				}
				  else
                                // Here we have to fail to be safe ! See validation/C_syntax/decreasing_loop01
				    success = false;
                            }
                        }
                        /* SG: i am duplicating code, next generation of phd will clean it */
                        else {
                            inc_v = expression_verbose_reduction_p_and_return_increment(incr,sub_expression_p);
                            if (inc_v != expression_undefined ) {
                                inc_v=MakeUnaryCall(entity_intrinsic(UNARY_MINUS_OPERATOR_NAME),inc_v);
                                if(extended_integer_constant_expression_p(inc_v)) {
                                    int v = expression_to_int(inc_v);
                                    if (v != 0) {
                                        * pincrement = inc_v;
                                        success = true;
                                        if (v <= 0 ) {
                                            * is_increasing_p = true;
                                            pips_debug(5, "Found \"i = i - v\" or \"i = v - i\" with positive increment!\n");
                                        }
                                        else {
                                            * is_increasing_p = false;
                                            pips_debug(5, "Found \"i = i - v\" or \"i = v - i\" with negative increment!\n");
                                        }
                                    }
                                }
                                /* SG: we checked the no-write-effect-on-increment earlier, we can go on safely,
                                 * but we will not know if the increment is positive or not, assume yes ?
                                 */
                                else {
				  // FI: I'm lost here
                                    * pincrement = inc_v;
				    if(positive_expression_p(inc_v)) {
				      * is_increasing_p = false;
				      success = true;
				    }
				    else if(negative_expression_p(inc_v)) {
				      * is_increasing_p = true;
				      success = true;
				    }
				    else
				      // Here we have to fail to be safe!
				      success = false;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return success;
}

/** 
 * parameter of effect guesser
 */
typedef struct {
    entity target; ///< entity to find effect on
    bool written;  ///< wheter the entity seems written
} guesser_param;

/** 
 * try hard to guess wether the call @param c writes @param p
 * Note that it is just aguess and may be wwrong
 * this function only exist because effects are not available in the controlizer
 * moreover, no aliasing information is available, so the guess **may** be wrong
 */
static bool guess_write_effect_on_entity_walker(call c, guesser_param *p)
{
    entity op = call_function(c);
    list args = call_arguments(c);
    if( ENTITY_ASSIGN_P(op) ||
            ENTITY_PLUS_UPDATE_P(op)||ENTITY_MINUS_UPDATE_P(op)||
            ENTITY_BITWISE_AND_UPDATE_P(op)||ENTITY_BITWISE_OR_UPDATE_P(op)||ENTITY_BITWISE_XOR_UPDATE_P(op)||
            ENTITY_DIVIDE_UPDATE_P(op)||ENTITY_MULTIPLY_UPDATE_P(op)||
            ENTITY_MODULO_UPDATE_P(op)||
            ENTITY_LEFT_SHIFT_UPDATE_P(op)||ENTITY_RIGHT_SHIFT_UPDATE_P(op)||
            ENTITY_ADDRESS_OF_P(op)||
            ENTITY_PRE_INCREMENT_P(op)||ENTITY_POST_INCREMENT_P(op)||
            ENTITY_PRE_DECREMENT_P(op)||ENTITY_POST_DECREMENT_P(op)
      )

    {
        expression lhs = EXPRESSION(CAR(args));
        if(expression_reference_p(lhs) &&
                same_entity_p(p->target,reference_variable(expression_reference(lhs))))
        {
            p->written=true;
            gen_recurse_stop(0);
        }

    }
    return true;
}

/** 
 * classical statement walker, gen_recurse does not dive into statement declarations
 */
static bool guess_write_effect_on_entity_stmt_walker(statement st, guesser_param *p)
{
    FOREACH(ENTITY,e,statement_declarations(st))
        if( !value_undefined_p(entity_initial(e) ) ) {
            gen_context_recurse(entity_initial(e),p,call_domain, guess_write_effect_on_entity_walker,gen_null);
        }
    return true;
}

/** 
 * call guess_write_effect_on_entity_walker on each call of @param exp
 * in order to guess write effect on @param loop_index
 * 
 * @return true if we are sure the entity was written, false otherwise, let's be optimistic :)
 */
static bool guess_write_effect_on_entity(void* exp, entity loop_index)
{
    guesser_param p = { loop_index, false };
    gen_context_multi_recurse(exp,&p,
            call_domain,guess_write_effect_on_entity_walker,gen_null,
            statement_domain, guess_write_effect_on_entity_stmt_walker,gen_null,
            0);
    return p.written;


}

/** 
 * guess the increment of a loop
 * the condition is: the increment must be a reference that is constant in the loop body
 * 
 * @param e candidate expression
 * @param loop_index entity guessed as index
 * @param body loop body
 * 
 * @return selected increment expression
 */
static
expression guess_loop_increment_walker(expression e, entity loop_index, statement body)
{
    if(expression_call_p(e))
    {
        call c = expression_call(e);
        list args = call_arguments(c);
        entity op = call_function(c);
        if( ENTITY_PRE_INCREMENT_P(op)||ENTITY_POST_INCREMENT_P(op)||
                ENTITY_PRE_DECREMENT_P(op)||ENTITY_POST_DECREMENT_P(op)||
                ENTITY_PLUS_UPDATE_P(op)||ENTITY_MINUS_UPDATE_P(op)||
                ENTITY_ASSIGN_P(op) /* this one needs further processing */ )
        {
            expression lhs = EXPRESSION(CAR(args));
            if(expression_reference_p(lhs) &&
                    same_entity_p(loop_index,reference_variable(expression_reference(lhs))))
            {
                if(!ENTITY_ASSIGN_P(op)) {
                    return e;
                }
                else {
                    expression rhs = EXPRESSION(CAR(CDR(args)));
                    if(expression_call_p(rhs))
                    {
                        call rhs_c = expression_call(rhs);
                        entity rhs_op = call_function(rhs_c);
                        list rhs_args = call_arguments(rhs_c);
                        if( ENTITY_PLUS_P(rhs_op) || ENTITY_PLUS_C_P(rhs_op) || 
                                    ENTITY_MINUS_P(rhs_op) || ENTITY_MINUS_C_P(rhs_op) )
                        {
                            expression rhs_rhs = EXPRESSION(CAR(rhs_args));
                            expression rhs_lhs = EXPRESSION(CAR(CDR(rhs_args)));
                            if( (expression_reference_p(rhs_rhs)&& same_entity_p(loop_index,reference_variable(expression_reference(rhs_rhs)))
                                    && (!( expression_reference_p(rhs_lhs) && guess_write_effect_on_entity(body,reference_variable(expression_reference(rhs_lhs)))))) ||
                                (expression_reference_p(rhs_lhs)&& same_entity_p(loop_index,reference_variable(expression_reference(rhs_lhs)))
                                    && (!( expression_reference_p(rhs_rhs) && guess_write_effect_on_entity(body,reference_variable(expression_reference(rhs_rhs))))))
                              )
                            {
                                return e;
                            }
                        }
                    }
                }
            }
        }
    }
    return expression_undefined;
}


/**
 * iterate over @param incr, a comma separated expression and checks for an (in|de)crement
 * of @param loop_index
 * if @param loop_index is written twice in @param incr , the result is expression_undefined
 *
 * @return expression_undefined if no (in|de) crement was found, the *crementing expression otherwise
 */
static
expression
guess_loop_increment(expression incr, entity loop_index, statement body)
{
    if(expression_call_p(incr))
    {
        call c =expression_call(incr);
        list args = call_arguments(c);
        entity op = call_function(c);
        if(ENTITY_COMMA_P(op))
        {
	  // FI: Serge, why don't you loop over args? O(N^2)...
            expression rhs = EXPRESSION(CAR(args));
            expression lhs = EXPRESSION(CAR(CDR(args)));

            expression lhs_guessed = guess_loop_increment(lhs,loop_index,body);
            if( !expression_undefined_p(lhs_guessed) && !guess_write_effect_on_entity(rhs,loop_index))
                return lhs_guessed;

            expression rhs_guessed = guess_loop_increment(rhs,loop_index,body);
            if( !expression_undefined_p(rhs_guessed) && !guess_write_effect_on_entity(lhs,loop_index))
                return rhs_guessed;
        }
        return guess_loop_increment_walker(incr,loop_index,body);
    }
    return expression_undefined;
}

/**
 * iterate over the comma-separeted expression @param init
 * and look for an initialization of @param loop_index
 *
 * @return expression_undefined if none found, the initialization expression otherwise
 */
static
expression
guess_loop_lower_bound(expression init, entity loop_index)
{
    if(expression_call_p(init))
    {
        call c =expression_call(init);
        list args = call_arguments(c);
        entity op = call_function(c);
        if(ENTITY_COMMA_P(op))
        {
            expression rhs = EXPRESSION(CAR(args));
            expression lhs = EXPRESSION(CAR(CDR(args)));

            expression guessed = guess_loop_lower_bound(lhs,loop_index);
            if(expression_undefined_p(guessed))
                return guess_loop_lower_bound(rhs,loop_index);
            return guessed;
        }
        else if(ENTITY_ASSIGN_P(op))
        {
            expression lhs = EXPRESSION(CAR(call_arguments(c)));
            if(expression_reference_p(lhs) &&
                    same_entity_p(loop_index,reference_variable(expression_reference(lhs))))
                return init;
        }
    }
    return expression_undefined;
}

/**
 * given an expression @param seed that can be found in @param comma_list
 * iterate over @param comma_list and remove @param seed from the list, updating pointers properly
 *
 */
static
void
remove_expression_from_comma_list(expression comma_list,expression seed)
{
    if(expression_call_p(comma_list))
    {
        call c =expression_call(comma_list);
        list args = call_arguments(c);
        entity op = call_function(c);
        if(ENTITY_COMMA_P(op))
        {
            expression rhs = EXPRESSION(CAR(args));
            expression lhs = EXPRESSION(CAR(CDR(args)));
            if( lhs == seed ) {
                expression_syntax(comma_list)=expression_syntax(rhs);
                expression_normalized(comma_list)=normalized_undefined;
            }
            else if(rhs==seed ) {
                expression_syntax(comma_list)=expression_syntax(lhs);
                expression_normalized(comma_list)=normalized_undefined;
            }
            else {
                remove_expression_from_comma_list(lhs,seed);
                remove_expression_from_comma_list(rhs,seed);
            }
        }
    }
}

/* Try to convert a C-like for-loop into a Fortran-like do-loop.

   Assume to match what is done in the prettyprinter C_loop_range().

   @return a sequence containing the do-loop if the transformation worked or sequence_undefined if
   it failed.
*/
sequence for_to_do_loop_conversion(forloop theloop, statement parent)
{

    sequence output = sequence_undefined;
    expression init = forloop_initialization(theloop);
    expression cond = forloop_condition(theloop);
    expression incr = forloop_increment(theloop);
    statement body  = forloop_body(theloop);

    set cond_entities = get_referenced_entities(cond);
    /* This does not filter scalar integers... */
    set incr_entities = get_referenced_entities(incr);
    set cond_inter_incr_entities = set_make(set_pointer);
    cond_inter_incr_entities = set_intersection(cond_inter_incr_entities,incr_entities,cond_entities);

    SET_FOREACH(entity,loop_index,cond_inter_incr_entities)
    {
      /* Consider only scalar integer variables as loop indices */
      type lit = ultimate_type(entity_type(loop_index));
      if(scalar_integer_type_p(lit) /* && type_depth(lit)==1*/ ) {
        if(!guess_write_effect_on_entity(body,loop_index))
        {
            bool is_upper_p,is_increasing_p;
            expression upper_bound;
            if(condition_expression_to_final_bound(cond,loop_index,&is_upper_p, &upper_bound))
            {
                set upper_bound_entities = get_referenced_entities(upper_bound);
                bool upper_bound_entity_written=false;
                SET_FOREACH(entity,e,upper_bound_entities)
                {
                    if(guess_write_effect_on_entity(body,e) || guess_write_effect_on_entity(incr,e))
                    {
                        upper_bound_entity_written=true;
                        break;
                    }
                }
                set_free(upper_bound_entities);
                if(!upper_bound_entity_written){
                    /* We got a candidate loop index and final bound,
		       let's check the increment */
                    expression increment_expression =
		      guess_loop_increment(incr,loop_index,body);
                    expression increment;
                    if( !expression_undefined_p(increment_expression) &&
			incrementation_expression_to_increment(increment_expression,
							       loop_index,
							       &is_increasing_p,
							       &increment))
                    {
                        /* We have found a do-loop compatible for-loop: */
                        output=make_sequence(NIL);
                        if(increment_expression!=incr){
                            remove_expression_from_comma_list(incr,increment_expression);
                            insert_statement(body,instruction_to_statement(make_instruction_expression(incr)),false);
			    //statement_consistent_p(body);
                        }

                        /* guess lower bound */
                        expression lower_bound = guess_loop_lower_bound(init,loop_index);
                        if(expression_undefined_p(lower_bound))
                            lower_bound=entity_to_expression(loop_index);
                        else {
                            if( lower_bound!= init) {
                                remove_expression_from_comma_list(init,lower_bound);
                                sequence_statements(output)=gen_append(sequence_statements(output),
                                        CONS(STATEMENT,instruction_to_statement(make_instruction_call(expression_call(init))),NIL));
                            }
                            lower_bound=EXPRESSION(CAR(CDR(call_arguments(expression_call(lower_bound)))));
                        }

                        if (!is_upper_p && is_increasing_p)
                            pips_user_warning("Loop with lower bound and increasing index %s\n", entity_local_name(loop_index));
                        if (is_upper_p && !is_increasing_p)
                            pips_user_warning("Loop with upper bound and decreasing index %s\n", entity_local_name(loop_index));

                        range lr;
			if(is_upper_p)
			  lr = make_range(lower_bound, upper_bound, increment);
			else {
			  // FI: Unfortunately, the problem must be
			  // postponed to the prettyprinter
			  lr = make_range(lower_bound, upper_bound, increment);
			}
                        loop l = make_loop(loop_index, lr, body, statement_label(parent),
					   make_execution_sequential(),NIL);

                        /* try hard to reproduce statement content */
                        statement sl = make_statement(
                                statement_label(parent),
                                statement_number(parent),
                                STATEMENT_ORDERING_UNDEFINED,
                                statement_comments(parent),
                                make_instruction_loop(l),
                                statement_declarations(parent),
                                statement_decls_text(parent),
                                statement_extensions(parent));

                        statement_label(parent)=entity_empty_label();
                        statement_ordering(parent)=STATEMENT_ORDERING_UNDEFINED;
                        statement_number(parent)=STATEMENT_NUMBER_UNDEFINED;
                        statement_comments(parent)=empty_comments;
                        statement_declarations(parent)=NIL;
                        statement_decls_text(parent)=NULL;
                        statement_extensions(parent)=empty_extensions();

                        sequence_statements(output)=gen_append(sequence_statements(output),CONS(STATEMENT,sl,NIL));

                    }
                }
            }
        }
      }
    }
    set_free(cond_entities);
    set_free(incr_entities );
    set_free( cond_inter_incr_entities);
    return output;
}


/* Try to to transform the C-like for-loops into Fortran-like do-loops.

   Assume we are in a gen_recurse since we use gen_get_recurse_ancestor(f).
 */
void
try_to_transform_a_for_loop_into_a_do_loop(forloop f) {

    /* Get the englobing statement of the "for" assuming we are called
       from a gen_recurse()-like function: */
    statement parent=
        (statement)gen_get_recurse_ancestor(
                (instruction)gen_get_recurse_ancestor(f));

    sequence new_l = for_to_do_loop_conversion(f,parent);

    if (!sequence_undefined_p(new_l)) {
        pips_debug(3, "do-loop has been generated.\n");
        statement_instruction(parent)=make_instruction_sequence(new_l);
        forloop_body(f)=statement_undefined;
        forloop_condition(f)=expression_undefined;
        forloop_increment(f)=expression_undefined;
        forloop_initialization(f)=expression_undefined;
        free_forloop(f);
    }
}


/* For-loop to do-loop transformation phase.

   This transformation transform for example a
\begin{lstlisting}
for (i = lb; i < ub; i += stride)
  body;
\end{lstlisting}
   into a
\begin{lstlisting}[language=fortran]
do i = lb, ub - 1, stride
  body
end do
\end{lstlisting}

   Now use pure local analysis on the code but we could imagine further
   use of semantics analysis some day...
*/
bool
for_loop_to_do_loop(char * module_name) {
  statement module_statement;

  /* Get the true ressource, not a copy, since we modify it in place. */
  module_statement =
    (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  set_current_module_statement(module_statement);
  entity mod = module_name_to_entity(module_name);
  set_current_module_entity(mod);

  debug_on("FOR_LOOP_TO_DO_LOOP_DEBUG_LEVEL");
  pips_assert("Statement should be OK before...", statement_consistent_p(module_statement));

  /* We need to access to the instruction containing the current
     for-loops, so ask NewGen gen_recurse to keep this informations for
     us: */
  /* Iterate on all the for-loops: */
  gen_recurse(module_statement,
              // Since for-loop statements can be nested, only restructure in
	      // a bottom-up way, : 
	      forloop_domain, gen_true, try_to_transform_a_for_loop_into_a_do_loop);

  pips_assert("Statement should be OK after...", statement_consistent_p(module_statement));

  pips_debug(2, "done\n");

  debug_off();

  /* Reorder the module, because some statements have been replaced. */
  module_reorder(module_statement);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_statement);

  reset_current_module_statement();
  reset_current_module_entity();

  /* Should have worked: */
  return true;
}


/** Build a sequence with a while-loop from for-loop parameters.

    The API is a little weird to comply with the controlizer
    implementation...
*/
sequence for_to_while_loop_conversion(expression init,
				      expression cond,
				      expression incr,
				      statement body,
				      extensions exts) {
  pips_debug(5, "Begin\n");

  statement init_st = make_expression_statement(init);
  statement incr_st = make_expression_statement(incr);
  sequence wlseq = sequence_undefined;

  /* Build the loop body of the while with { body; incr_st; } : */
  statement n_body = make_statement_from_statement_varargs_list(body,
								incr_st,
								NULL);
  /* Build the while(cond) { n_body } statement: */
  statement wl_st =  make_whileloop_statement(cond,
					      n_body,
					      STATEMENT_NUMBER_UNDEFINED,
					      true);
  if(!empty_extensions_p(exts)) {
      free_extensions(statement_extensions(wl_st));
      statement_extensions(wl_st)=copy_extensions(exts);
  }

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
  return wlseq;
}


/* Try to to transform the C-like for-loops into Fortran-like do-loops.

   Assume we are in a gen_recurse since we use gen_get_recurse_ancestor(f).

   The forloop is freed and replaced by a while loop in the statement
   owning the for-loop
 */
void
transform_a_for_loop_into_a_while_loop(forloop f) {
  pips_debug(5, "Begin\n");

  /* Get the instruction owning the forloop: */
  instruction i = (instruction) gen_get_recurse_ancestor(f);
  /* Get the statement owning instruction owning the forloop: */
  statement st = (statement) gen_get_recurse_ancestor(i);

  /* Get a sequence with a while-loop instead: */
  sequence wls = for_to_while_loop_conversion(forloop_initialization(f),
					      forloop_condition(f),
					      forloop_increment(f),
					      forloop_body(f),
					      statement_extensions(st));

  /* These fields have been re-used, so protect them from memory
     recycling: */
  forloop_initialization(f) = expression_undefined;
  forloop_condition(f) = expression_undefined;
  forloop_increment(f) = expression_undefined;
  forloop_body(f) = statement_undefined;

  /* We need to replace the for-loop instruction by the sequence
     instruction. The cleaner way should be to delete the first one and
     make the other one, but since we are in a gen_recurse() and we
     iterate on the first one, it is dangerous. Well, I've tried and it
     works, but valgrind complains a bit. :-)

     So change the type of the instruction on the fly instead: */
  instruction_tag(i) = is_instruction_sequence;
  instruction_sequence(i) = wls;
  /* And discard the old for: */
  free_forloop(f);

  /* Since we have replaced a statement that may have comments and labels
     by a sequence, do not forget to forward them where they can be: */
  /* FI: one issue: st is an ancestor of the current object and some
     of its pointers are going to be modified although they are being
     processed by gen_recurse()... */
  fix_sequence_statement_attributes(st);

  /* Removed useless instructions that may remain: */
  clean_up_sequences(st);

  ifdebug(5) {
    print_statement(st);
    pips_debug(5, "Exiting with statement\n");
  }
}

/* Same as above, but with no calls to ancestors */
void
transform_a_for_loop_statement_into_a_while_loop(statement st) {
  if(forloop_statement_p(st)) {
    pips_debug(5, "Begin\n");

    /* Get the instruction owning the forloop: */
    //instruction i = (instruction) gen_get_recurse_ancestor(f);
    instruction i = statement_instruction(st);
    /* Get the statement owning instruction owning the forloop: */
    //statement st = (statement) gen_get_recurse_ancestor(i);
    forloop f = instruction_forloop(i);

    /* Get a sequence with a while-loop instead: */
    sequence wls = for_to_while_loop_conversion(forloop_initialization(f),
						forloop_condition(f),
						forloop_increment(f),
						forloop_body(f),
						statement_extensions(st));

    /* These fields have been re-used, so protect them from memory
       recycling: */
    forloop_initialization(f) = expression_undefined;
    forloop_condition(f) = expression_undefined;
    forloop_increment(f) = expression_undefined;
    forloop_body(f) = statement_undefined;

    /* We need to replace the for-loop instruction by the sequence
       instruction. The cleaner way should be to delete the first one and
       make the other one, but since we are in a gen_recurse() and we
       iterate on the first one, it is dangerous. Well, I've tried and it
       works, but valgrind complains a bit. :-)

       So change the type of the instruction on the fly instead: */
    instruction_tag(i) = is_instruction_sequence;
    instruction_sequence(i) = wls;
    /* And discard the old for: */
    free_forloop(f);

    /* Since we have replaced a statement that may have comments and labels
       by a sequence, do not forget to forward them where they can be: */
    /* FI: one issue: st is an ancestor of the current object and some
       of its pointers are going to be modified although they are being
       processed by gen_recurse()... */
    fix_sequence_statement_attributes(st);

    /* Removed useless instructions that may remain: */
    clean_up_sequences(st);

    ifdebug(5) {
      print_statement(st);
      pips_debug(5, "Exiting with statement\n");
    }
  }
}


/* For-loop to while-loop transformation phase.

   This transformation transforms a
\begin{lstlisting}
for (init; cond; update)
  body;
\end{lstlisting}
    into a
\begin{lstlisting}
{
  init;
  while(cond) {
    body;
    update;
  }
}
\end{lstlisting}
*/
bool
for_loop_to_while_loop(char * module_name) {
  statement module_statement;

  /* Get the true ressource, not a copy, since we modify it in place. */
  module_statement =
    (statement) db_get_memory_resource(DBR_CODE, module_name, true);

  set_current_module_statement(module_statement);
  entity mod = module_name_to_entity(module_name);
  set_current_module_entity(mod);

  debug_on("FOR_LOOP_TO_WHILE_LOOP_DEBUG_LEVEL");
  pips_assert("Statement should be OK before...", statement_consistent_p(module_statement));

  /* We need to access to the instruction containing the current
     for-loops, so ask NewGen gen_recurse to keep this informations for
     us: */
  /* Iterate on all the for-loops: */
  gen_recurse(module_statement,
              // Since for-loop statements can be nested, only restructure in
	      // a bottom-up way, :
	      //forloop_domain, gen_true, transform_a_for_loop_into_a_while_loop);
	      statement_domain, gen_true, transform_a_for_loop_statement_into_a_while_loop);

  pips_assert("Statement should be OK after...", statement_consistent_p(module_statement));

  pips_debug(2, "done\n");

  debug_off();

  /* Reorder the module, because some statements have been replaced. */
  module_reorder(module_statement);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_statement);

  reset_current_module_statement();
  reset_current_module_entity();

  /* Should have worked: */
  return true;
}
