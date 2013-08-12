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
/* An atomizer that uses the one made by Fabien Coelho for HPFC,
 * and is in fact just a hacked version of the one made by Ronan
 * Keryell...
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "arithmetique.h"
#include "reductions.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "properties.h"

#include "sac.h"

static statement orginal_statement = NULL;


static 
void patch_constant_size(syntax s, bool* patch_all)
{
    switch(syntax_tag(s))
    {
        case is_syntax_call:
            {
                call c = syntax_call(s);
                if(entity_constant_p(call_function(c)) )
                {
                    basic b = variable_basic(type_variable(functional_result(type_functional(entity_type(call_function(c))))));
                       if( basic_int_p(b) && (basic_int(b) == DEFAULT_INTEGER_TYPE_SIZE) )
                            basic_int(b)= 1;
                }
            } break;
        case is_syntax_reference:
            *patch_all = false;
            break;
        default:
            break;
    };
}
static expression current_lhs = expression_undefined;

/* returns the assignment statement if moved, or NULL if not.
 */
statement simd_atomize_this_expression(entity (*create)(entity, basic),
        expression e)
{
    basic bofe = basic_undefined;

    /* it does not make sense to atomize a range...
    */
    if (syntax_range_p(expression_syntax(e))) return NULL;

    /* SG: in case of expression similar to (a+2), if a is a short (or a char ...),
     * the user may expect the result is a short too 
     * the C syntax expect 2 is an int
     * set the property to false if you want to override this behavior
     */
    if( get_bool_property("SIMD_OVERRIDE_CONSTANT_TYPE_INFERENCE") )
    {
        expression etemp = copy_expression(e);
        bool patch_all=true;
        /* force all integer to char, then compute the basic
         * validate only if we can guess the size from something not a constant integer
         */
        gen_context_recurse(etemp,&patch_all,syntax_domain,gen_true,patch_constant_size);
        if( !patch_all )
            bofe=basic_of_expression(etemp);
        free_expression(etemp);
    }

    if(basic_undefined_p(bofe) )
        bofe = basic_of_expression(e);

    if(!basic_undefined_p(bofe)) {
        if (!basic_overloaded_p(bofe))
        {
            entity newvar = (*create)(get_current_module_entity(), bofe);
	    AddEntityToCurrentModule(newvar);
            expression rhs = make_expression(expression_syntax(e), normalized_undefined);
            normalize_all_expressions_of(rhs);
            statement assign = make_assign_statement(entity_to_expression(newvar),rhs);
            expression_syntax(e) = make_syntax_reference( make_reference(newvar, NIL));

            return assign;
        }

        free_basic(bofe);
    }
    return NULL;
}


/* This function computes the maximum width of all the variables used in a call
 */
static void get_type_max_width(call ca, int* maxWidth)
{
    FOREACH(EXPRESSION, arg,call_arguments(ca))
    {
        syntax s = expression_syntax(arg);

        switch(syntax_tag(s))
        {
            case is_syntax_call:
                {
                    call c = syntax_call(s);

                    entity op = call_function(c);
                    if ( ENTITY_FIELD_P(op) || ENTITY_POINT_TO_P(op))
                    {
                        basic bas = basic_of_expression(binary_call_rhs(c));
                        *maxWidth=MAX(*maxWidth , basic_type_size(bas));
                        free_basic(bas);
                    }
                    else if (!call_constant_p(c))
                        get_type_max_width(c, maxWidth);
                } break;

            case is_syntax_reference:
                {
                    basic bas = basic_of_reference(syntax_reference(s));
                    *maxWidth=MAX(*maxWidth , basic_type_size(bas));
                    free_basic(bas);
                } break;
            case is_syntax_subscript:
                {
                    basic bas = basic_of_expression(subscript_array(syntax_subscript(s)));
                    *maxWidth=MAX(*maxWidth , basic_type_size(bas));
                    free_basic(bas);
                } break;
            case is_syntax_cast:
                {
                    cast ca = syntax_cast(s);
                    type t = cast_type(ca);
                    *maxWidth=MAX(*maxWidth ,type_memory_size(t));
                } break;
			case is_syntax_sizeofexpression:
				*maxWidth=MAX(*maxWidth ,DEFAULT_INTEGER_TYPE_SIZE);
				break;

            default:pips_internal_error("syntax_tag %u not supported yet",syntax_tag(s));

        }
    }

}

/* This function aims at changing the basic size of the left expression of
 * the newly created assign statement
 */
static void change_basic_if_needed(statement stat)
{
	if( statement_call_p(stat) && ENTITY_ASSIGN_P(call_function(statement_call(stat))))
	{
		int maxWidth = -1;
		expression lExp = EXPRESSION(CAR(call_arguments(statement_call(stat))));
		expression rExp = EXPRESSION(CAR(CDR(call_arguments(statement_call(stat)))));

		// Check that the right expression is a call statement
		if(expression_call_p(rExp))
		{

			// Check that the statement can be potentially integrated in a 
			// SIMD statement 
            // SG: atomizer is no longer limited to sac
			// if(match_statement(stat) != NIL)
			{
				get_type_max_width(syntax_call(expression_syntax(rExp)), &maxWidth);
			}

			// If the maxWidth of the right expression is smaller than the width 
			// of the current left expression, then replace the left expression width 
			// by maxWidth
			if(maxWidth > 0)
			{
				basic lExpBasic = expression_basic(lExp);
                maxWidth=MIN(maxWidth,basic_type_size(lExpBasic));
			}
		}
	}
}

/* This function insert stat before orginal_statement in the code
 */
static void simd_insert_statement(statement cs, statement stat)
{
	change_basic_if_needed(stat);
	// If cs is already a sequence, we just need to insert stat in cs
	if(instruction_sequence_p(statement_instruction(cs)))
	{
		instruction_block(statement_instruction(cs)) = gen_insert_before(stat,
				orginal_statement,
				instruction_block(statement_instruction(cs)));
	}
	// If cs is not a sequence, we have to create one sequence composed of
	// cs then orginal_statement
	else
	{
		statement_label(stat) = statement_label(cs);

		orginal_statement = make_statement(entity_empty_label(), 
				statement_number(cs),
				statement_ordering(cs),
				statement_comments(cs),
				statement_instruction(cs),
				statement_declarations(cs),
				NULL,
				statement_extensions(cs), make_synchronization_none());

		statement_instruction(cs) =
			make_instruction_block(CONS(STATEMENT, stat,
						CONS(STATEMENT,
							orginal_statement,
							NIL)));

		statement_label(cs) = entity_empty_label();
		statement_number(cs) = STATEMENT_NUMBER_UNDEFINED;
		statement_ordering(cs) = STATEMENT_ORDERING_UNDEFINED;
		statement_comments(cs) = empty_comments;
		statement_extensions(cs) = empty_extensions ();
	}
}

static
entity sac_make_new_variable(entity module, basic b)
{
    entity e = make_new_scalar_variable(module, copy_basic(b));
    AddLocalEntityToDeclarations(e,module,get_current_module_statement());
    return e;
}

static
void simd_do_atomize(expression ce, statement cs)
{
	syntax s = expression_syntax(ce);
	statement stat =statement_undefined;

	// Atomize expression only if this is a call expression
	if(syntax_call_p(s))
	{
		call cc = syntax_call(s);

		// Atomize expression only if the call is not a constant
		if(FUNC_TO_ATOMIZE_P(cc))
		{
			/* If the current call is not an assign call,
			 * let's atomize the current argument
             * sg: also try to be smart and make reduction appear if any
             */
			
			if( (stat = simd_atomize_this_expression(sac_make_new_variable, ce)) )
				simd_insert_statement(cs, stat);
		}
	}
}
static bool reference_filter(expression exp, __attribute__((unused)) statement cs)
{
    if( expression_reference_p(exp) ) {
        if( get_bool_property("SIMD_ATOMIZER_ATOMIZE_REFERENCE") ) return true;
        else {
            reference r = expression_reference(exp);
            FOREACH(EXPRESSION, ind, reference_indices(r)) {
                NORMALIZE_EXPRESSION(ind);
                if(expression_linear_p(ind))
                    gen_recurse_stop(ind);
            }
        }
    }
    if(expression_call_p(exp))
    {
        call c = expression_call(exp);
        entity op = call_function(c);
        return !ENTITY_POINT_TO_P(op) && !ENTITY_FIELD_P(op);
    }
    return true;
}

/* This function is called for each call statement and atomize it
*/
static void atomize_call_statement(statement cs)
{
    call c = instruction_call(statement_instruction(cs));

    // Initialize orginal_statement if this is the first argument
    orginal_statement=cs;

    // For each call argument, the argument is atomized if needed
	if( ENTITY_ASSIGN_P(call_function(c)) )
	{
        current_lhs = binary_call_lhs(c);
		expression rhs = binary_call_rhs(c);
		if( expression_call_p(rhs) )
		{
			FOREACH(EXPRESSION, arg,call_arguments(expression_call(rhs)))
				gen_context_recurse(arg,cs,expression_domain,reference_filter,simd_do_atomize);
		}
        else if (expression_reference_p(rhs)) {
				gen_context_recurse(rhs,cs,expression_domain,reference_filter,simd_do_atomize);
        }
        if(get_bool_property("SIMD_ATOMIZER_ATOMIZE_LHS"))
        {
            expression lhs = EXPRESSION(CAR(call_arguments(c)));
            gen_context_recurse(lhs,cs,expression_domain,reference_filter,simd_do_atomize);
        }
	}
}

/* This function is called for all statements in the code
*/
static void atomize_statements(statement cs)
{
    // Only a call statement can be atomized
    if (instruction_call_p(statement_instruction(cs)))
    {
        atomize_call_statement(cs);
    }
}

bool simd_atomizer(char * mod_name)
{
    /* get the resources */
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, true);

    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));

    debug_on("SIMD_ATOMIZER_DEBUG_LEVEL");


    /* Now do the job */
    gen_recurse(mod_stmt, statement_domain, gen_true, atomize_statements);

    /* Reorder the module, because new statements have been added */  
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);

    /* update/release resources */
    reset_current_module_statement();
    reset_current_module_entity();

    debug_off();

    return true;
}

static bool do_expression_reduction(statement s, reduction r, expression e) {
    reference redref = reduction_reference(r);
    entity redop = reduction_operator_entity(reduction_op(r));

    if(expression_call_p(e)) {
      call ce = expression_call(e);
      if(same_entity_p(call_function(ce), redop)) {
        expression lhs = binary_call_lhs(ce),
                   rhs = binary_call_rhs(ce);

        do_expression_reduction(s, r, lhs);
        do_expression_reduction(s, r, rhs);

        if(expression_reference_p(lhs) &&
            reference_equal_p(expression_reference(lhs), redref)) {
          update_expression_syntax(e, copy_syntax(expression_syntax(rhs)));
        }
        else if(expression_reference_p(rhs) &&
                 reference_equal_p(expression_reference(rhs), redref)) {
            update_expression_syntax(e, copy_syntax(expression_syntax(lhs)));
        }
        else
        {
            statement snew = make_assign_statement(
                reference_to_expression(copy_reference(redref)),
                MakeBinaryCall(redop,
                    reference_to_expression(copy_reference(redref)),
                    copy_expression(rhs)));
            store_cumulated_reductions(snew,copy_reductions(load_cumulated_reductions(s)));
            update_expression_syntax(e, copy_syntax(expression_syntax(lhs)));
            insert_statement(s, snew, false);
            store_cumulated_reductions(STATEMENT(CAR(statement_block(s))),copy_reductions(load_cumulated_reductions(s)));
        }
        return true;
      }
    }
    return false;
}


static bool do_reduction_atomization(statement s) {
    list reductions = reductions_list(load_cumulated_reductions(s));
    if(!ENDP(reductions)) {
        FOREACH(REDUCTION,r,reductions) {
            /* the reduction must be of the pattern red = red op exp1 op exp2 */
            reference redref = reduction_reference(r);
            entity redop = reduction_operator_entity(reduction_op(r));
            if(statement_call_p(s)) {
                call c = statement_call(s);
                entity assign = call_function(c);
                if(ENTITY_ASSIGN_P(assign)) {
                    expression lhs = binary_call_lhs(c);
                    if(expression_reference_p(lhs)) {
                        reference  rlhs = expression_reference(lhs);
                        if(reference_equal_p(rlhs,redref)) {
                            expression rhs = binary_call_rhs(c);
                            if(do_expression_reduction(s, r, rhs))
                                update_expression_syntax(rhs, expression_syntax(
                                            MakeBinaryCall(redop,
                                                reference_to_expression(copy_reference(redref)),
                                                copy_expression(rhs))));
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool reduction_atomization(const char * module_name) {
    /* prelude */
    set_current_module_entity(module_name_to_entity(module_name));
    set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, module_name,true));
    set_cumulated_reductions((pstatement_reductions) db_get_memory_resource(DBR_CUMULATED_REDUCTIONS, module_name, true));

    /* do the job */
    gen_recurse(get_current_module_statement(),statement_domain,do_reduction_atomization,gen_null);
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, get_current_module_name(), get_current_module_statement());

    /* postlude */
    reset_cumulated_reductions();
    reset_current_module_statement();
    reset_current_module_entity();
    return true;
}
