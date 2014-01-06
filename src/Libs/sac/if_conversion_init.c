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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "complexity_ri.h"
#include "text.h"

#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "transformations.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "properties.h"

#include "complexity.h"
#include "sac.h"


/** 
 * atomize the condition of a test and returns generated statements
 * 
 * @param cs test statement
 * 
 * @return generated statement
 */
static statement atomize_condition(statement cs)
{
        test t = statement_test(cs);
        expression cond = test_condition(t);
        if(!expression_reference_p(cond))
            return simd_atomize_this_expression(make_new_scalar_variable, cond);
        return statement_undefined;
}


/** 
 * accept if conversion if the call is an assignment call
 */
static bool check_if_conv_call(call c, bool * success)
{
    call tmp = copy_call(c);
    split_update_call(tmp);
    entity op = call_function(tmp);
    *success &=ENTITY_ASSIGN_P(op)||ENTITY_CONTINUE_P(op)||call_constant_p(c);
    free_call(tmp);
    return *success;
}

static bool check_if_conv_test(test t,bool *success);
static bool check_if_conv_walker(instruction i, bool * success)
{
    switch(instruction_tag(i))
    {
        case is_instruction_call:
            return check_if_conv_call(instruction_call(i),success);
        case is_instruction_test:
            return check_if_conv_test(instruction_test(i),success);
        case is_instruction_block:
            return true;
        default:
            return false;
    };
}

/** 
 * checks if a test statement is suitable for if conversion, that is
 * it only contains sequence of assignments
 * 
 */
static bool check_if_conv_test(test t,bool *success)
{
    statement branches[2] = { test_true(t), test_false(t) };

    for(size_t i=0;i<2;i++)
        gen_context_recurse(branches[i],success,
                instruction_domain,check_if_conv_walker, gen_null);
    return false;
}

/** 
 * checks if if conversion can be performed on test statement
 * 
 * @param stat test statement
 * 
 * @return true if conversion possible
 */
static bool check_if_conv(statement stat)
{
    test t = instruction_test(statement_instruction(stat));

    pips_assert("statement is a test", statement_test_p(stat));

    bool success = true;
    check_if_conv_test(t,&success);
    return success;
}


/** 
 * adds @param cond to the condtion of @param t
 */
static void update_test_condition(test t, expression cond)
{
    test_condition(t)=MakeBinaryCall(entity_intrinsic(AND_OPERATOR_NAME),copy_expression(cond),test_condition(t));
}

/** 
 * create a test statement with appropriate extension and a test under @a cond with a single true branch @a branch
 */
static statement make_if_converted_test_statement(expression cond, statement branch)
{
    statement s =  instruction_to_statement(
                                make_instruction_test(
                                    make_test(cond,branch,make_empty_statement())));
    extensions_extension(statement_extensions(s))=CONS(EXTENSION,make_extension_pragma(make_pragma_string(strdup(IF_TO_CONVERT))),NIL);
    return s;
}

/** 
 * merge content of a test with the test itself
 * 
 * @param branch content of the test
 * @param cond condition of the test
 * 
 * @return merged statements
 */
static statement
do_transform_if_statements(statement branch, expression cond)
{
    // only add the condition if content is a test
    if(statement_test_p(branch))
    {
        update_test_condition( statement_test(branch) ,cond);
        return copy_statement(branch);
    }
    // the big part : merge tests into a single block
    // the idea is too stack statements until a test is met.
    // stacked statements are put under condition, then the test is added and so on
    else if(statement_block_p(branch))
    {
        list block=NIL;
        list curr_block = NIL;
        FOREACH(STATEMENT,st,statement_block(branch))
        {
            if(statement_test_p(st))
            {
                update_test_condition( statement_test(st) ,cond);
                if(!ENDP(curr_block))
                    block=CONS(STATEMENT,make_if_converted_test_statement(copy_expression(cond),make_block_statement(gen_nreverse(curr_block))),block);
                block=CONS(STATEMENT,copy_statement(st),block);
                curr_block=NIL;
            }
            else
            {
                curr_block=CONS(STATEMENT,copy_statement(st),curr_block);
            }
        }
        if(!ENDP(curr_block))
            block=CONS(STATEMENT,make_if_converted_test_statement(copy_expression(cond),make_block_statement(gen_nreverse(curr_block))),block);
        return make_block_statement(gen_nreverse(block));
    }
    // not much to do there, ony allpy test condition
    else
        return make_if_converted_test_statement(copy_expression(cond),copy_statement(branch));
}


/*
   This function is called for each test statement in the code
   */
static
void if_conv_init_statement(statement stat)
{
    // Only interested in the test statements
    if(statement_test_p(stat))
    {

        gen_chunk * ancestor = gen_get_recurse_current_ancestor();
        if (ancestor && INSTANCE_OF(control, ancestor)) {
            // Hmmm, we are inside a control node
            control c = (control) ancestor;
            if (gen_length(control_successors(c)) == 2) {
                // A control node with 2 successors is an unstructured test:
                pips_user_warning("not converting a non structured test yet...\n");
                return;
            }
        }

        complexity stat_comp = load_statement_complexity(stat);
        if(stat_comp != (complexity) HASH_UNDEFINED_VALUE)
        {

            Ppolynome poly = complexity_polynome(stat_comp);
            if(polynome_constant_p(poly))
            {
                clean_up_sequences(stat);
                pips_debug(2,"analyzing statement:\n");
                ifdebug(2) {
                    print_statement(stat);
                }

                // Get the number of calls in the if statement
                int cost = polynome_TCST(poly);
                pips_debug(3,"cost %d\n", cost);

                // ensure the statement is valid
                bool success = check_if_conv(stat);

                // If the number of calls is smaller than IF_CONV_THRESH
                if(success && cost < get_int_property("IF_CONVERSION_INIT_THRESHOLD"))
                {
                    test t = statement_test(stat);

                    // Atomize the condition
                    statement atom = atomize_condition(stat);

                    expression not_test_condition=
                        MakeUnaryCall(
                                entity_intrinsic(NOT_OPERATOR_NAME),
                                copy_expression(test_condition(t))
                                );
                    list block = 
                        make_statement_list(
                                do_transform_if_statements(test_true(t),test_condition(t)),
                                do_transform_if_statements(test_false(t),not_test_condition)
                                );
                    if(!statement_undefined_p(atom))
                        block=CONS(STATEMENT,atom,block);
                    update_statement_instruction(stat,make_instruction_block(block));
                    ifdebug(3) {
                        pips_debug(3,"new test:\n");
                        print_statement(stat);
                    }
                }
            }
            else
                pips_user_warning("not converting a test, complexity too ... complex \n");

        }
        else
            pips_user_warning("not converting a test, complexity not available\n");
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
#pragma IF_TO_CONVERT
if(L0)
{
i = 3;
}
L1 = (b == 3);
#pragma IF_TO_CONVERT
if(L0 && L1)
{
x = 5;
}
L2 = (b < 3);
#pragma IF_TO_CONVERT
if(!L0 && L2)
{
x = 6;
}
#pragma IF_TO_CONVERT
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
bool if_conversion_init(char * mod_name)
{
    // get the resources
    set_current_module_entity(module_name_to_entity(mod_name));
    statement root = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);
    statement fake = make_empty_block_statement();

    set_current_module_statement(fake);// to prevent a complex bug with gen_recurse and AddEntityToCurrentModule
    set_complexity_map( (statement_mapping) db_get_memory_resource(DBR_COMPLEXITIES, mod_name, true));

    debug_on("IF_CONVERSION_INIT_DEBUG_LEVEL");

    ifdebug(1) {
      pips_debug(1, "Code before if_conversion_init:\n");
      print_statement(root);
    }

    // Now do the job
    gen_recurse(root, statement_domain, gen_true, if_conv_init_statement);

    ifdebug(1) {
      pips_debug(1, "Code after if_conv_init_statement:\n");
      print_statement(root);
    }

    // and share decl
    FOREACH(STATEMENT,s,statement_block(fake))
        insert_statement_no_matter_what(root,copy_statement(s),true);
    free_statement(fake);

    ifdebug(1) {
      pips_debug(1, "Code after copying from fake statement:\n");
      print_statement(root);
    }

    // Reorder the module, because new statements have been added
    module_reorder(root);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, root);

    // update/release resources
    reset_complexity_map();
    reset_current_module_statement();
    reset_current_module_entity();

    debug_off();

    return true;
}
