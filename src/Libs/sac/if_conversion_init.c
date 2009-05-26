
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <malloc.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "complexity_ri.h"
#include "text.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "ri-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "transformer.h"
#include "semantics.h"
#include "control.h"
#include "transformations.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "properties.h"
#include "atomizer.h"

#include "expressions.h"
#include "complexity.h"

#include "sac-local.h"

#define IF_CONV_THRESH 40

static statement oriStat = NULL;

static void simd_insert_statement(statement cs, statement stat)
{
    if(instruction_sequence_p(statement_instruction(cs)))
    {
        instruction_block(statement_instruction(cs)) = gen_insert_before(stat,
                oriStat,
                instruction_block(statement_instruction(cs)));
    }
    else
    {
        statement_label(stat) = statement_label(cs);

        oriStat = make_statement(entity_empty_label(), 
				 statement_number(cs),
				 statement_ordering(cs),
				 statement_comments(cs),
				 statement_instruction(cs),
				 NIL,
				 NULL,
				 statement_extensions(cs));

        statement_instruction(cs) =
            make_instruction_block(CONS(STATEMENT, stat,
                        CONS(STATEMENT,
                            oriStat,
                            NIL)));

        statement_label(cs) = entity_empty_label();
        statement_number(cs) = STATEMENT_NUMBER_UNDEFINED;
        statement_ordering(cs) = STATEMENT_ORDERING_UNDEFINED;
        statement_comments(cs) = empty_comments;
	statement_extensions(cs) = empty_extensions ();
    }
}

statement simd_atomize_this_expression(entity (*create)(entity, basic),
        expression e);

#if 0
static void simd_atomize_call(call c, statement cs)
{
    MAP(EXPRESSION, ce,
    {
        syntax s = expression_syntax(ce);

        if(syntax_call_p(s))
        {
            call cc = syntax_call(s);

            if(ENTITY_FUNCTION_P(call_function(cc)))
            {
                simd_atomize_call(cc, cs);

                statement stat = simd_atomize_this_expression(hpfc_new_variable, ce);

                simd_insert_statement(cs, stat);
            }
        }
    }, call_arguments(c));
}
#endif

static bool check_if_conv_stat(statement s)
{
    bool success = TRUE;
    instruction inst = statement_instruction(s);

    switch(instruction_tag(inst)) 
    {
        case is_instruction_block :
            MAPL( sts,
            {
                success = check_if_conv_stat(STATEMENT(CAR(sts)));
                if(!success)
                    break;
            }, instruction_block(inst));
    break;

        case is_instruction_test :{
                                      test t = instruction_test(inst);
                                      success = check_if_conv_stat(test_true(t));
                                      success &= check_if_conv_stat(test_false(t));
                                      break;
                                  }
        case is_instruction_call :

                                  if(!ENTITY_ASSIGN_P(call_function(instruction_call(inst))))
                                  {
                                      return FALSE;
                                  }

                                  success = TRUE;
                                  break;
        case is_instruction_loop :
        case is_instruction_whileloop :
        case is_instruction_goto :
        case is_instruction_unstructured :
        default : 
                                  return success = FALSE;
                                  break;
    }

    return success;
}

static void atomize_condition(statement cs)
{
    pips_assert("statement is a test", instruction_test_p(statement_instruction(cs)));

    test t = instruction_test(statement_instruction(cs));

    oriStat = cs;

    expression cond = test_condition(t);
    syntax s = expression_syntax(cond);

    if(syntax_call_p(s))
    {
        call cc = syntax_call(s);

        if(ENTITY_FUNCTION_P(call_function(cc)))
        {
            /*
               simd_atomize_call(cc, cs);

               statement stat = simd_atomize_this_expression(hpfc_new_variable, cond);

               simd_insert_statement(cs, stat);
               */
            entity new_ent = make_new_scalar_variable_with_prefix(strdup("L"),
                    get_current_module_entity(),
                    make_basic(is_basic_logical, (void *)1));

            test_condition(t) = entity_to_expression(new_ent);

            statement stat = make_assign_statement(entity_to_expression(new_ent),
                    cond);

            simd_insert_statement(cs, stat);
        }
    }
}

static bool check_if_conv(test t)
{
    bool success = TRUE;

    if(test_true(t) != statement_undefined)
    {
        success = check_if_conv_stat(test_true(t));
    }

    if(test_false(t) != statement_undefined)
    {
        success &= check_if_conv_stat(test_false(t));      
    }

    return success;
}

/*
   This fonction fill in the two lists lStat and lCond
   */
static void analyse_true_false_stat(statement stat, list *lStat, list *pStat,
        list *lCond, list *pCond, expression curCond)
{
    instruction inst = statement_instruction(stat);

    switch(instruction_tag(inst)) 
    {
        case is_instruction_block :

            // For each statement of the sequence, call analyse_true_false_stat()
            MAP(STATEMENT, cStat,
            {
                analyse_true_false_stat(cStat, lStat, pStat, 
                        lCond, pCond, curCond);
            }, instruction_block(inst));
            break;

        case is_instruction_test : 
            {
                test t = instruction_test(statement_instruction(stat));

                list lArg = CONS(EXPRESSION, copy_expression(curCond), 
                        CONS(EXPRESSION, copy_expression(test_condition(t)), NIL));

                expression cond = call_to_expression(
                        make_call(entity_intrinsic(AND_OPERATOR_NAME), lArg));

                // If test_true() is not empty, then add the true statement to lStat
                // and the condition to lCond
                if(test_true(t) != statement_undefined &&
                        !empty_statement_or_labelless_continue_p(test_true(t)))
                {
                    CDR(*pStat) = CONS(STATEMENT, copy_statement(test_true(t)), NIL);
                    *pStat = CDR(*pStat);
                    CDR(*pCond) = CONS(EXPRESSION, cond, NIL);
                    *pCond = CDR(*pCond);
                }

                // If test_false() is not empty, then add the true statement to lStat
                // and the condition to lCond
                if(test_false(t) != statement_undefined &&
                        !empty_statement_or_labelless_continue_p(test_false(t)))
                {

                    expression not = call_to_expression(
                            make_call(entity_intrinsic(NOT_OPERATOR_NAME), 
                                CONS(EXPRESSION, copy_expression(test_condition(t)), NIL)));

                    lArg = CONS(EXPRESSION, copy_expression(curCond), 
                            CONS(EXPRESSION, not, NIL));

                    cond = call_to_expression(
                            make_call(entity_intrinsic(AND_OPERATOR_NAME), lArg));

                    CDR(*pStat) = CONS(STATEMENT, copy_statement(test_false(t)), NIL);
                    *pStat = CDR(*pStat);
                    CDR(*pCond) = CONS(EXPRESSION, cond, NIL);
                    *pCond = CDR(*pCond);
                }
                break;
            }
            // Simply add the call statement to the lStat and the current 
            //condition to lCond
        case is_instruction_call :
            {
                if(*lStat == NIL)
                {
                    *lStat = *pStat = CONS(STATEMENT, copy_statement(stat), NIL);
                    *lCond = *pCond = CONS(EXPRESSION, copy_expression(curCond), NIL);
                }
                else
                {
                    CDR(*pStat) = CONS(STATEMENT, copy_statement(stat), NIL);
                    *pStat = CDR(*pStat);
                    CDR(*pCond) = CONS(EXPRESSION, copy_expression(curCond), NIL);
                    *pCond = CDR(*pCond);
                }
                break;
            }
        default : 
            print_statement(stat);
            pips_assert("stat err2", FALSE);
            break;
    }
}

/*
   This fonction replaces the old statement instruction to the new one
   using the lists lStat and lCond
   */
static void transform_if_stat(statement stat, list lStat, list lCond)
{
    instruction inst = statement_instruction(stat);

    expression cond;
    expression lastCond;

    lastCond = EXPRESSION(CAR(lCond));

    list lIfStat = NIL;
    list pIfStat = NIL;

    list lOutStat = NIL;
    list pOutStat = NIL;

    // If stat is a sequence, it means that the first statements were created
    // by the atomize_condition function and the last statement is the
    // original if statement
    if(statement_block_p(stat))
    {
        MAP(STATEMENT, cStat,
        {
            // The last statement has been reached
            if(statement_test_p(cStat))
                break;

            statement newStat = copy_statement(cStat);

            // Just add the statement
            if(lOutStat == NIL)
            {
                statement_label(newStat) = statement_label(stat);
                lOutStat = pOutStat = CONS(STATEMENT, newStat, NIL);
            }
            else
            {
                CDR(pOutStat) = CONS(STATEMENT, newStat, NIL);
                pOutStat = CDR(pOutStat);
            }
        }, statement_block(stat));
    }

    // Go through the lStat list
    MAP(STATEMENT, cStat,
    {
        cond = EXPRESSION(CAR(lCond));

        // If the condition is the same as the last one encountered,
        // then add the statement to lIfStat list
        if(same_expression_p(cond, lastCond))
        {
            if(lIfStat == NIL)
            {
                lIfStat = CONS(STATEMENT, copy_statement(cStat), NIL);
                pIfStat = lIfStat;
            }
            else
            {
                CDR(pIfStat) = CONS(STATEMENT, copy_statement(cStat), NIL);
                pIfStat = CDR(pIfStat);
            }
        }
        // Else it means that an if statement needs to be created
        else
        {
            statement newTrue = statement_undefined;

            // Create the if statement from lIfStat
            if(gen_length(lIfStat) == 1)
            {
                newTrue = STATEMENT(CAR(lIfStat));
            }
            else
            {
                newTrue = make_block_statement(lIfStat);
            }

            test t = make_test(copy_expression(lastCond), 
                    newTrue, make_empty_statement());

            statement newIf = make_statement(entity_empty_label(), 
					     STATEMENT_NUMBER_UNDEFINED,
					     STATEMENT_ORDERING_UNDEFINED,
					     IF_TO_CONVERT,
					     make_instruction(is_instruction_test, t),
					     NIL,
					     NULL,
					     empty_extensions ());

            // Add the created statement to the list of output statements
            if(lOutStat == NIL)
            {
                statement_label(newIf) = statement_label(stat);
                lOutStat = pOutStat = CONS(STATEMENT, copy_statement(newIf), NIL);
            }
            else
            {
                CDR(pOutStat) = CONS(STATEMENT, copy_statement(newIf), NIL);
                pOutStat = CDR(pOutStat);
            }

            lIfStat = CONS(STATEMENT, copy_statement(cStat), NIL);
            pIfStat = lIfStat;
        }

        lastCond = cond;
        lCond = CDR(lCond);
    }, lStat);

    // If lIfStat is not empty, it means that a last
    // if-statement needs to be created
    if(lIfStat != NIL)
    {
        statement newTrue = statement_undefined;

        // Create the if statement from lIfStat
        if(gen_length(lIfStat) == 1)
        {
            newTrue = STATEMENT(CAR(lIfStat));
        }
        else
        {
            newTrue = make_block_statement(lIfStat);
        }

        test t = make_test(copy_expression(lastCond), 
                newTrue, make_empty_statement());

        statement newIf = make_statement(entity_empty_label(), 
					 STATEMENT_NUMBER_UNDEFINED,
					 STATEMENT_ORDERING_UNDEFINED,
					 IF_TO_CONVERT,
					 make_instruction(is_instruction_test, t),
					 NIL,
					 NULL,
					 empty_extensions ());

        // Add the created statement to the list of output statements
        if(lOutStat == NIL)
        {
            statement_label(newIf) = statement_label(stat);
            lOutStat = pOutStat = CONS(STATEMENT, copy_statement(newIf), NIL);
        }
        else
        {
            CDR(pOutStat) = CONS(STATEMENT, copy_statement(newIf), NIL);
            pOutStat = CDR(pOutStat);
        }

        lIfStat = NIL;
    }

    // Free the old instruction
    free_instruction(inst);

    instruction newInst = make_instruction_block(lOutStat);

    // Insert the new instruction
    statement_instruction(stat) = newInst;

    statement_label(stat) = entity_empty_label();
    statement_number(stat) = STATEMENT_NUMBER_UNDEFINED;
    statement_ordering(stat) = STATEMENT_ORDERING_UNDEFINED;
    statement_comments(stat) = empty_comments;
}

/*
   This function is called for each test statement in the code
   */
static void if_conv_init_statement(statement stat)
{
    // Only interested in the test statements
    if(!instruction_test_p(statement_instruction(stat)))
        return;

    complexity stat_comp = load_statement_complexity(stat);

    if(stat_comp == (complexity) HASH_UNDEFINED_VALUE)
        return;

    Ppolynome poly = complexity_polynome(stat_comp);

    if(!polynome_constant_p(poly))
        return;

    int cost = polynome_TCST(poly);

    pips_debug(3,"cost %d\n", cost);

    test t = instruction_test(statement_instruction(stat));

    pips_assert("statement is a test", instruction_test_p(statement_instruction(stat)));

    bool success;

    // lStat an lCond are two lists that associate a statement 
    // to a condition
    list lStat = NIL;
    list pStat = NIL;
    list lCond = NIL;
    list pCond = NIL;

    // Get the number of calls in the if statement
    success = check_if_conv(t);

    // If the number of calls is smaller than IF_CONV_THRESH
    if(success && cost < IF_CONV_THRESH)
    {
        // Atomize the condition
        atomize_condition(stat);

        expression not = call_to_expression(
                make_call(entity_intrinsic(NOT_OPERATOR_NAME), 
                    CONS(EXPRESSION, copy_expression(test_condition(t)), NIL)));

        // Process the true statement
        analyse_true_false_stat(test_true(t), &lStat, &pStat, &lCond, &pCond, test_condition(t));

        // Process the false statement
        analyse_true_false_stat(test_false(t), &lStat, &pStat, &lCond, &pCond, not);

        // Do the transformation from the two lists lStat and lCond
        transform_if_stat(stat, lStat, lCond);
    }
}

/*
   This phase changes:

   if(a > 0)
   {
   i = 3;
   if(b == 3)
   {
   x = 5;
   }
   }
   else
   {
   if(b < 3)
   {
   x = 6;
   }
   j = 3;
   }

into:

L0 = (a > 0);
//IF_TO_CONVERT
if(L0)
{
i = 3;
}
L1 = (b == 3);
//IF_TO_CONVERT
if(L0 && L1)
{
x = 5;
}
L2 = (b < 3);
//IF_TO_CONVERT
if(!L0 && L2)
{
x = 6;
}
//IF_TO_CONVERT
if(!L0)
{
j = 3;
}

This transformation is done if:
- there is only call- or sequence-
statements in the imbricated if-statements
- the number of calls in the imbricated if-statements
is smaller than IF_CONV_THRESH
*/
boolean if_conversion_init(char * mod_name)
{
    // get the resources
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, TRUE);

    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));

    set_complexity_map( (statement_mapping)
            db_get_memory_resource(DBR_COMPLEXITIES, mod_name, TRUE));

    debug_on("IF_CONVERSION_INIT_DEBUG_LEVEL");
    // Now do the job

    gen_recurse(mod_stmt, statement_domain,
            gen_true, if_conv_init_statement);

    // Reorder the module, because new statements have been added 
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name, 
            compute_callees(mod_stmt));

    // update/release resources
    reset_complexity_map();
    reset_current_module_statement();
    reset_current_module_entity();

    debug_off();

    return TRUE;
}
