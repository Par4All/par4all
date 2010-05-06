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

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "sac-local.h"

#include "sac.h"

#include "control.h"

#include "ricedg.h"
#include "effects-simple.h"

static graph dependence_graph;


#define SIZEOFSTRING(s) (sizeof((s))-1)


static bool simd_save_stat_p(statement stat)
{
    if(statement_call_p(stat))
    {

        string funcName = entity_local_name(call_function(statement_call(stat)));
        return    same_stringn_p(funcName, SIMD_SAVE_NAME, SIZEOFSTRING(SIMD_SAVE_NAME))
               || same_stringn_p(funcName, SIMD_GEN_SAVE_NAME, SIZEOFSTRING(SIMD_GEN_SAVE_NAME))
               || same_stringn_p(funcName, SIMD_CONS_SAVE_NAME, SIZEOFSTRING(SIMD_CONS_SAVE_NAME))
               || same_stringn_p(funcName, SIMD_MASKED_SAVE_NAME, SIZEOFSTRING(SIMD_MASKED_SAVE_NAME));
    }
    else
    {
        return FALSE;
    }
}
static bool simd_load_stat_p(statement stat)
{
    if(statement_call_p(stat))
    {

        string funcName = entity_local_name(call_function(statement_call(stat)));
        return    same_stringn_p(funcName, SIMD_LOAD_NAME, SIZEOFSTRING(SIMD_LOAD_NAME))
               || same_stringn_p(funcName, SIMD_GEN_LOAD_NAME, SIZEOFSTRING(SIMD_GEN_LOAD_NAME))
               || same_stringn_p(funcName, SIMD_CONS_LOAD_NAME,SIZEOFSTRING(SIMD_CONS_LOAD_NAME))
               || same_stringn_p(funcName, SIMD_MASKED_LOAD_NAME, SIZEOFSTRING(SIMD_MASKED_LOAD_NAME));
    }
    else
    {
        return FALSE;
    }
}
/* This function returns true if the statement is a simd loadsave
 * statement
 */
bool simd_loadsave_stat_p(statement stat)
{
    return simd_load_stat_p(stat) || simd_save_stat_p(stat);
}

/* This function returns true if the statement is a simd
 * statement
 */
static bool simd_stat_p(statement stat)
{
    return statement_call_p(stat)
        && same_stringn_p( entity_local_name(call_function(statement_call(stat))) , SIMD_NAME, SIZEOFSTRING(SIMD_NAME));
}

/* This function checks if two list of expression are equals (modulo the & operator)
 */
static bool list_eq_expression(list args1, list args2, bool allow_addressing)
{
    if(gen_length(args1) == gen_length(args2))
    {
        list pArgs2 = args2;
        FOREACH(EXPRESSION, exp1, args1)
        {
            expression exp2 = EXPRESSION(CAR(pArgs2));
            /* hack to handle the & operator in C */
            if(allow_addressing)
            {
                if( expression_call_p(exp1) )
                {
                    call c = expression_call(exp1);
                    if( entity_an_operator_p( call_function(c) , ADDRESS_OF ) )
                        exp1 = EXPRESSION(CAR( call_arguments(c) ) );
                }
                if( expression_call_p(exp2) )
                {
                    call c = expression_call(exp2);
                    if( entity_an_operator_p( call_function(c) , ADDRESS_OF ) )
                        exp2 = EXPRESSION(CAR( call_arguments(c) ) );
                }
            }
            if(!same_expression_p(exp1, exp2))
                return FALSE;
            pArgs2 = CDR(pArgs2);
        }
        return TRUE;
    }
    else
    {
        return FALSE;
    }
}


/* This function returns true if the arguments of the simd statement theStat do
 * not depend on the loop iteration
 */
static bool constant_argument_list_p(list args, statement theStat, list forstats, list l_reg)
{
    /* consider each vertex */
    FOREACH(VERTEX, a_vertex, graph_vertices(dependence_graph) )
    {
        statement stat1 = vertex_to_statement(a_vertex);

        if (stat1 != theStat)
            continue;

        /* check this vertex successor */
        FOREACH(SUCCESSOR, suc, vertex_successors(a_vertex) )
        {
            statement stat2 = vertex_to_statement(successor_vertex(suc));

            /* skip if the successor is not in the loop body */
            if (statement_in_statements_p(stat2, forstats)  && (stat1 != stat2))
            {

                FOREACH(CONFLICT, c,dg_arc_label_conflicts(successor_arc_label(suc)) )
                {
                    //conflict c = CONFLICT(CAR(l3));
                    // If stat2 is not a simd statement, then return FALSE
                    if(!simd_stat_p(stat2)&&!declaration_statement_p(stat2))
                    {
                        pips_debug(1,"conflicting with:\n");
                        ifdebug(1) { print_statement(stat2); }
                        return false;
                    }
                    // If stat2 is a loadsave statement and that there is a conflict
                    // between the arguments of
                    else if(   simd_loadsave_stat_p(stat2)
                            && !list_eq_expression(args, CDR(call_arguments(statement_call(stat2))),true)
                           )
                    {
                        pips_debug(1,"conflicting with:\n");
                        ifdebug(1) { print_statement(stat2); }
                        return false;
                    }
                    else
                    {
                        pips_debug(1,"conflict avoided with outer statement\n");
                        ifdebug(1) { print_statement(stat2); }
                    }
                }
            }
            else
            {
                if( (stat1 != stat2))
                {
                    pips_debug(1,"conflict avoided with outer statement\n");
                    ifdebug(1) { print_statement(stat2); }
                }
            }
        }
    }

    return  true;
}
static bool statements_uses_loop_index(statement s0,statement s1)
{
    statement fake = copy_statement(s1);
    if(statement_loop_p(fake)) {
        free_statement(loop_body(statement_loop(fake)));
        loop_body(instruction_loop(statement_instruction(fake)))=statement_undefined;
    }
    else if(statement_whileloop_p(fake))
    {
        free_statement(whileloop_body(statement_whileloop(fake)));
        whileloop_body(instruction_whileloop(statement_instruction(fake)))=statement_undefined;
    }
    else if(statement_forloop_p(fake))
    {
        free_statement(forloop_body(statement_forloop(fake)));
        forloop_body(instruction_forloop(statement_instruction(fake)))=statement_undefined;
    }


    set S0 = get_referenced_entities(s0);
    set S1 = get_referenced_entities(fake);
    free_statement(fake);
    bool uses = set_intersection_p(S0,S1);
    set_free(S0);
    set_free(S1);
    return uses;


}

/* This function searches for simd load or save statements that can be
 * put out of the loop body. It stores these statements in
 * constArgs hash table.
 */
static bool searchForConstArgs(statement head,statement body, hash_table constArgs, list l_reg)
{
    if(statement_block_p(body))
    {
        FOREACH(STATEMENT, curStat, statement_block(body) )
        {
            pips_debug(1,"examining statement:\n");
            ifdebug(1) { print_statement(curStat); }

            // If if it is a simd load or save statement, ...
            if(simd_loadsave_stat_p(curStat))
            {
                /* first argument is always a vector */
                list args = CDR(call_arguments(statement_call(curStat)));

                /* If the arguments of the statement do not depend on the iteration,
                 * then store the statement in constArgs
                 */
                if(!statements_uses_loop_index(curStat,head) && constant_argument_list_p(args, curStat, statement_block(body), l_reg))
                {
                    pips_debug(1,"no conflict ^^\n");
                    hash_put(constArgs, curStat, args);
                }
            }
            else {
                pips_debug(1,"not a save / load statement !\n");
            }
        }
        return true;
    }
    else
    {
        return false;
    }
}

static list simdstatement_to_entities(statement s)
{
    list out = NIL;
    FOREACH(EXPRESSION,exp,call_arguments(statement_call(s)))
    {
        if(expression_call_p(exp))
        {
            call c = expression_call(exp);
            if(ENTITY_ADDRESS_OF_P(call_function(c)))
            {
                exp=EXPRESSION(CAR(call_arguments(c)));
                if(expression_reference_p(exp))
                    out=CONS(ENTITY,expression_to_entity(exp),out);
                else if(expression_field_p(exp))
                    out=CONS(ENTITY,expression_to_entity(binary_call_lhs(expression_call(exp))),out);
            }

        }
        else if(expression_reference_p(exp))
            out=CONS(ENTITY,expression_to_entity(exp),out);
    }
    return out;
}

/* This function moves the statements in constArgs out of the loop body
 */
static void moveConstArgsStatements(statement s, statement body, hash_table constArgs)
{
    //hash_table argsToVect = hash_table_make(hash_pointer, 0);
    //hash_table argsToFunc = hash_table_make(hash_pointer, 0);
    list bodySeq = NIL;
    list headerSeq = NIL;
    list footerSeq = NIL;
    instruction old = statement_instruction(s);
    statement_instruction(s)=instruction_undefined;
    update_statement_instruction(s,make_instruction_block(NIL));

    /* Go through the statements in the loop body to fill argsToVect
     * and argsToFunc and to replace simdVector, if necessary
     */
    /* gather moved vector here */
    set moved_vectors = set_make(set_pointer);
    /* reverse now so that everything is built in correct order */
    sequence_statements(instruction_sequence(statement_instruction(body)))=gen_nreverse(statement_block(body));

    /* work on a copy beacuase we remove statement during iteration */
    list body_statement_copies = gen_copy_seq(statement_block(body));
    
    FOREACH(STATEMENT, curStat, body_statement_copies)
    {
        list args = (list) hash_get(constArgs, curStat);

        /* add it to argsToVect / argsToFunc */
        if(args != HASH_UNDEFINED_VALUE)
        {
            list moved_entities=simdstatement_to_entities(curStat);
            set_append_list(moved_vectors,moved_entities);
            gen_free_list(moved_entities);
            if(simd_load_stat_p(curStat))
                headerSeq = CONS(STATEMENT, curStat,headerSeq);
            else if(simd_save_stat_p(curStat) )
                footerSeq = CONS(STATEMENT, curStat,footerSeq);
            else
                pips_user_error("const statement not a load or save !?!");
        }
        /* should we moved the declaration too ? thnaks to the reverse transversal, we already now 
         * moved entities from moved_vectors */
        else if(declaration_statement_p(curStat))
        {
            /* copy beacause cannot modify in place using RemoveLocal... */
            list declared_entities = gen_copy_seq(statement_declarations(curStat));
            bool something_did_not_move = false;
            FOREACH(ENTITY,e,declared_entities)
            {
                if(set_belong_p(moved_vectors,e))
                {
                    RemoveLocalEntityFromDeclarations(e,get_current_module_entity(),body);
                    AddLocalEntityToDeclarations(e,get_current_module_entity(),s);
                }
                else
                {
                    something_did_not_move=true;
                }
            }
            gen_free_list(declared_entities);
            if(something_did_not_move)
                bodySeq = CONS(STATEMENT, curStat, bodySeq);
        }
        /* keep it in for's body */
        else
        {
            bodySeq = CONS(STATEMENT, curStat, bodySeq);
        }
    }
    gen_free_list(body_statement_copies);
    set_free(moved_vectors);
    /* put everything together */
    statement_instruction(body)=instruction_undefined;
    update_statement_instruction(body,make_instruction_block(bodySeq));
    insert_statement(s,make_block_statement(gen_nconc(headerSeq,CONS(STATEMENT,instruction_to_statement(old),footerSeq))),false);
}

/* This function is called for each statement and performs the
 * simd_loop_const_elim on loop
 */
static void simd_loop_const_elim_rwt(statement s)
{
    instruction i = statement_instruction(s);

    /* We are only interested in loops */
    statement body = statement_undefined;
    switch(instruction_tag(i))
    {
        case is_instruction_loop:
            body = loop_body(instruction_loop(i));
            break;

        case is_instruction_whileloop:
            body = whileloop_body(instruction_whileloop(i));
            break;

        case is_instruction_forloop:
            body = forloop_body(instruction_forloop(i));
            break;

        default:
            return;
    }

    /* Load the read write effects of the loop in l_reg */
    list l_reg = load_cumulated_rw_effects_list(s);

    /* Search for simd load or save statements that can be
     * put out of the loop body. It stores these statements in
     * constArgs hash table
     * Move the statements in constArgs out of the loop body
     */
    hash_table constArgs = hash_table_make(hash_pointer, 0);
    if(searchForConstArgs(s,body, constArgs, l_reg))
        moveConstArgsStatements(s, body, constArgs);
    hash_table_free(constArgs);
}


/* This phase looks for load or save statements that can be
 * put out of the loop body and move these statements, if possible.
 */
bool simd_loop_const_elim(char * module_name)
{
    /* Get the code of the module. */
    entity module = module_name_to_entity(module_name);
    statement module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, TRUE);
	set_ordering_to_statement(module_stat);
    set_current_module_entity( module);
    set_current_module_statement( module_stat);
    set_cumulated_rw_effects(
            (statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE)
    );

    dependence_graph = (graph) db_get_memory_resource(DBR_DG, module_name, TRUE);
    push_generated_variable_commenter(sac_commenter);

    debug_on("SIMD_LOOP_CONST_ELIM_DEBUG_LEVEL");

    /* Go through all the statements */
    gen_recurse(module_stat, statement_domain,
            gen_true, simd_loop_const_elim_rwt);
    clean_up_sequences(module_stat);

    pips_assert("Statement is consistent after SIMD_LOOP_CONST_ELIM",
            statement_consistent_p(module_stat));
    pop_generated_variable_commenter();

    module_reorder(module_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);

    debug_off();

    reset_current_module_entity();
	reset_ordering_to_statement();
    reset_current_module_statement();
    reset_cumulated_rw_effects();

    return TRUE;
}
