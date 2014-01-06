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
#include "transformations.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "properties.h"

#include "callgraph.h"

#include "sac.h"


/** 
 * creates a phi-instruction using the entity given in property
 * the generated instruction as the form
 * @a lRef = PHI(@a cond,@a ref1, @a ref2)
 * all parameters are copied
 */
static instruction
make_phi_assign_instruction(reference lRef, expression cond,
        expression ref1, expression ref2)
{
    entity phiEntity = module_name_to_runtime_entity(get_string_property("IF_CONVERSION_PHI"));
    expression phiExp = MakeTernaryCall(
            phiEntity,
            copy_expression(cond),
            copy_expression(ref1),
            copy_expression(ref2)
            );

    return make_assign_instruction(reference_to_expression(copy_reference(lRef)), phiExp);
}

/*
   The if_conversion phase only supports assign statement whose right
   expression does not have any write effect.

   So, this function returns true if the statement stat is supported.
   */
static bool simd_supported_stat_p(statement stat)
{
    if(instruction_call_p(statement_instruction(stat)))
    {
        call c = instruction_call(statement_instruction(stat));
        split_update_call(c); // this will take car of splitting a+=2 into a = a+2
        entity op = call_function(c);
        // Only the assign statements with no side effects are supported
        if(ENTITY_ASSIGN_P(op) )
        {
            expression rExp = EXPRESSION(CAR(CDR(call_arguments(c))));
            list effects = expression_to_proper_effects(rExp);
            bool has_write_effect_p = effects_write_at_least_once_p(effects);
            free(effects);
            return !has_write_effect_p;
        }
    }
    return false;
}

/** 
 * converts statement @a stat into a phi-statement if possible
 * 
 * @param cond condition of the potential new phi - statement
 * @param stat statement to check
 * 
 * @return true if processing was ok (sg:unclean, event to me :))
 */
static bool
process_true_call_stat(expression cond, statement stat)
{

    // Only the assign statements with no side effects are supported
    if(simd_supported_stat_p(stat))
    {
        call c = copy_call(instruction_call(statement_instruction(stat)));

        // lRef is the left reference of the assign call
        expression lhs = binary_call_lhs(c);
        reference lhs_ref = expression_reference(lhs);
        lhs=copy_expression(lhs); /* to prevent side effect from further inplace modification */
        entity e = reference_variable(lhs_ref);

        basic newBas = basic_of_reference(lhs_ref);

        if(!basic_undefined_p(newBas))
        {
            ifdebug(1) {
                pips_debug(1,"converting statement\n");
                print_statement(stat);
            }
            // Create a new entity if rhs is not a reference or a constant itself
            expression rhs = binary_call_rhs(c);
            expression ref = expression_undefined;
            statement to_add = statement_undefined;
            if(expression_reference_p(rhs)||expression_constant_p(rhs))
                ref = rhs;
            else {
                entity newVar = make_new_scalar_variable_with_prefix(entity_local_name(e),
                        get_current_module_entity(),
                        newBas);
                AddEntityToCurrentModule(newVar);

                syntax_reference(expression_syntax(binary_call_lhs(c))) = make_reference(newVar, NIL);
                ref=reference_to_expression(make_reference(newVar,NIL));
                to_add = call_to_statement(c);
            }
            // Make an assign statement to insert before the phi-statement
            instruction assign = make_phi_assign_instruction(lhs_ref, cond, ref, lhs);
            update_statement_instruction(stat,assign);
            if(!statement_undefined_p(to_add))
                insert_statement(stat,to_add,true);
            ifdebug(1) {
                pips_debug(1,"into statement\n");
                print_statement(stat);
            }
            free_expression(lhs);
            return true;
        }
        free_expression(lhs);

    }
    else if(declaration_statement_p(stat))
        return true;// leave statement untouched
    return false;
}

/*
   This function changes the true statement stat in two list.

   For example, if the true statement stat is:

   A(I) = I + 1
   J = J + 1

   then outStat will be:

   A0 = I + 1
   J0 = J + 1

   and postlude will be:

   A(I) = PHI(COND, A0, A(I))
   J = J0
   */
static void process_true_stat(statement parent, expression cond, statement stat)
{

    // It must have been verified in the if_conversion_init phase
    pips_assert("stat is a call or a sequence statement", 
            (instruction_call_p(statement_instruction(stat)) ||
             instruction_sequence_p(statement_instruction(stat)) ||
             statement_loop_p(stat)));

    // If stat is a call statement, ...
    if(instruction_call_p(statement_instruction(stat)))
    {
        if(process_true_call_stat(cond, stat))
        {
            statement_instruction(parent)=instruction_undefined;
            update_statement_instruction(parent,statement_instruction(stat));
            statement_instruction(stat)=instruction_undefined;
            free_statement(stat);
        }

    }
    // recurse for for loops
    else if( statement_loop_p(stat))
        process_true_stat(stat,cond,loop_body(statement_loop(stat)));

    // If stat is a sequence statement, ...
    else if(statement_block_p(stat))
    {
        // first split initalizations
        statement_split_initializations(stat);

        // then do the processing
        bool something_bad_p=false;
        if(statement_block_p(stat))
        {
            FOREACH(STATEMENT,st,statement_block(stat))
            {
                something_bad_p|=!process_true_call_stat(cond, st);
            }
        }
        else
            something_bad_p|=!process_true_call_stat(cond, stat);
        if(!something_bad_p)
        {
            statement_instruction(parent)=instruction_undefined;
            update_statement_instruction(parent,statement_instruction(stat));
            statement_instruction(stat)=instruction_undefined;
            free_statement(stat);
        }
    }
}

/*
   This function is called for each code statement.
   */
static void if_conv_statement(statement cs)
{
    // If the statement comment contains the string IF_TO_CONVERT,
    // then it means that this statement must be converted ...
    extension ex;
    if ( (ex = get_extension_from_statement_with_pragma(cs, IF_TO_CONVERT)) )
    {
        // remove the pragma
        gen_remove(&extensions_extension(statement_extensions(cs)),ex);

        // Process the "true statements" (test_false(t) is empty because if_conversion
        // phase is done after if_conversion_init phase).
        test t = instruction_test(statement_instruction(cs));
        process_true_stat(cs,test_condition(t), test_true(t));
    }
}

/*
   This phase do the actual if conversion.
   It changes:

   c IF_TO_CONVERT
   if(L1) then
   A(I) = I + 1
   J = J + 1
   endif

into:

A0 = I + 1
J0 = J + 1
A(I) = PHI(L1, A0, A(I))
J = PHI(L1, J0, J)

This phase MUST be used after if_conversion_init phase

*/
bool if_conversion(char * mod_name)
{
    // get the resources
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, true);

    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));

    debug_on("IF_CONVERSION_DEBUG_LEVEL");
    // Now do the job

    gen_recurse(mod_stmt, statement_domain, gen_true, if_conv_statement);

    // Reorder the module, because new statements have been added 
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name, 
            compute_callees(mod_stmt));

    // update/release resources
    reset_current_module_statement();
    reset_current_module_entity();

    debug_off();

    return true;
}

static statement do_loop_nest_unswitching_purge(statement adam, list conditions) {
    if(ENDP(conditions)) {
	    clone_context cc = make_clone_context(get_current_module_entity(),get_current_module_entity(),NIL,get_current_module_statement());
	    statement eve = clone_statement(adam,cc);
	    free_clone_context(cc);
	    return eve;
    }
    statement eve =  instruction_to_statement(
		    make_instruction_test(
			    make_test(
				    copy_expression(EXPRESSION(CAR(conditions))),
				    do_loop_nest_unswitching_purge(adam,CDR(conditions)),
				    do_loop_nest_unswitching_purge(adam,CDR(conditions))
				    )
			    )
            );
    return eve;
}

static void do_loop_nest_unswitching(statement st,list *conditions) {
    if(statement_loop_p(st)) {
        loop l =statement_loop(st);
        range r = loop_range(l);
        expression u = range_upper(r);
        if(expression_minmax_p(u)) {//only handle the case of two args right now ... */
            call c = expression_call(u);
            if(gen_length(call_arguments(c)) > 2 ) pips_internal_error("do not handle more than 2 args");
            expression hs[]= {
                binary_call_lhs(c),
                binary_call_rhs(c)
                    };
            for(int i=0;i<sizeof(hs)/sizeof(hs[0]);i++) {
                expression hss = hs[i];
                hs[i]=copy_expression(hs[i]);
                NORMALIZE_EXPRESSION(hs[i]);
                if(normalized_complex_p(expression_normalized(hs[i]))) {
                    /* will help for partial eval later */
                    entity etmp = make_new_scalar_variable(get_current_module_entity(),basic_of_expression(hs[i]));
                    AddEntityToCurrentModule(etmp);
                    hs[i]=make_assign_expression(entity_to_expression(etmp),hs[i]);
                    update_expression_syntax(hss,expression_syntax(entity_to_expression(etmp)));
                }
            }

            *conditions=
                CONS(EXPRESSION,
                        binary_intrinsic_expression(GREATER_THAN_OPERATOR_NAME,hs[0],hs[1]),
                        *conditions);
        }

        statement sparent = (statement)gen_get_ancestor(statement_domain,st);
        /* some conditions left */
        if(!ENDP(*conditions) && (!statement_loop_p(sparent) || !sparent)) {
            *st = *(do_loop_nest_unswitching_purge(st,*conditions));
            //add_pragma_str_to_statement(st,get_string_property("OUTLINE_PRAGMA"),true);
            gen_full_free_list(*conditions);
            *conditions=NIL;
        }
        /* parent is a loop : check for a conflict */
        else {
            list toremove=NIL;
            list tconditions=gen_copy_seq(*conditions);
            FOREACH(EXPRESSION,cond,tconditions) {
                set s = get_referenced_entities(cond);
                if(set_belong_p(s,loop_index(statement_loop(sparent)))) {
                    toremove=CONS(EXPRESSION,cond,toremove);
                    gen_remove_once(conditions,cond);
                }
                set_free(s);
            }
            gen_free_list(tconditions);
            if(!ENDP(toremove)) {
                *st = *(do_loop_nest_unswitching_purge(st,toremove));
                //add_pragma_str_to_statement(st,get_string_property("OUTLINE_PRAGMA"),true);
            }
            gen_full_free_list(toremove);
        }
    }
    else pips_assert("everything is ok",ENDP(*conditions));
}

bool loop_nest_unswitching(const char *module_name) {
    set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, module_name,true));
    set_current_module_entity(module_name_to_entity(module_name));

    list l=NIL;
    gen_context_recurse(get_current_module_statement(),&l,
            statement_domain,gen_true,do_loop_nest_unswitching);
    pips_assert("everything went well\n",ENDP(l));


    // Reorder the module, because new statements have been added 
    module_reorder(get_current_module_statement());
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

    reset_current_module_statement();
    reset_current_module_entity();
    return true;
}
