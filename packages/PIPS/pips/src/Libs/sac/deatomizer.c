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

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"

#include "effects-generic.h"

#include "sac.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "ricedg.h"
#include "control.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

static void daCheckCallReplace(call c, reference ref);

static int gInIndex = 0;
static bool gRepOnlyInIndex = 0;
static bool gReplaceAllowed = false;

static bool gRepDone = false;

static graph dep_graph;

static void checkReplaceReference(expression e, reference ref)
{
    syntax s = expression_syntax(e);

    switch(syntax_tag(s)) {
        case is_syntax_reference : {
                                       reference r = syntax_reference(s);
                                       /* replace if equal to ref */
                                       if ( reference_indices(r) == NIL ) {
                                           if ( reference_equal_p(syntax_reference(s), ref) && 
                                                   ((gRepOnlyInIndex && (gInIndex != 0)) || !gRepOnlyInIndex)) {
                                               gReplaceAllowed = true;
                                           }
                                       }
                                       else {
                                           gInIndex++;

                                           FOREACH(EXPRESSION,indice,reference_indices(r))
                                           {
                                               checkReplaceReference(indice, ref);
                                           }

                                       gInIndex--;
                                       }
                                   }
                                   break;
        case is_syntax_range :
                                   checkReplaceReference(range_lower(syntax_range(s)), ref);
                                   checkReplaceReference(range_upper(syntax_range(s)), ref);
                                   checkReplaceReference(range_increment(syntax_range(s)), ref);
                                   break;
        case is_syntax_call :
                                   daCheckCallReplace(syntax_call(s), ref);
                                   break;
        default : 
                                   pips_internal_error("unknown tag: %d", 
                                           (int) syntax_tag(expression_syntax(e)));
    }
}

static void daCheckCallReplace(call c, reference ref)
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
            /* I'd rather assume, nothing to replace for symbolic constants */
            break;
        case is_value_intrinsic:
        case is_value_unknown:
        case is_value_code:
            /* We assume that it is legal to replace arguments (because it should
               have been verified with the effects that the index is not WRITTEN).
               */
            {
                FOREACH(EXPRESSION,e,call_arguments(c))
                {
                    checkReplaceReference(e, ref);
                }
            }
    break;
        default:
    pips_internal_error("unknown tag: %d", 
            (int) value_tag(vin));

    abort();
    }
}

static void daCallReplaceReference(call c, reference ref, expression next);

static void daExpressionReplaceReference(list e, reference ref, expression next)
{
    syntax s = expression_syntax(EXPRESSION(CAR(e)));

    switch(syntax_tag(s)) {
        case is_syntax_reference : {
                                       reference r = syntax_reference(s);
                                       /* replace if equal to ref */
                                       if ( reference_indices(r) == NIL ) {
                                           if ( reference_equal_p(syntax_reference(s), ref)) {
                                               expression exp = EXPRESSION(CAR(e));
                                               expression_syntax(exp) = copy_syntax(expression_syntax(next));
                                               gRepDone = true;

                                               if(expression_normalized(EXPRESSION(CAR(e))) != normalized_undefined)
                                               {
                                                   free_normalized(expression_normalized(EXPRESSION(CAR(e))));
                                                   expression_normalized(EXPRESSION(CAR(e))) = normalized_undefined;
                                               }

                                               NORMALIZE_EXPRESSION(EXPRESSION(CAR(e)));
                                           }
                                       }
                                       else {
                                           MAPL(lexpr,
                                           {
                                               daExpressionReplaceReference(lexpr, ref, next);
                                           }, reference_indices(r));
                                       }
                                   }
                                   break;
        case is_syntax_range :
                                   pips_internal_error("tag syntax_range not implemented");
                                   break;
        case is_syntax_call :
                                   daCallReplaceReference(syntax_call(s), ref, next);
                                   break;
        default : 
                                   pips_internal_error("unknown tag: %d", 
                                           (int) syntax_tag(expression_syntax(EXPRESSION(CAR(e)))));
    }
}


static void daCallReplaceReference(call c, reference ref, expression next)
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
            /* I'd rather assume, nothing to replace for symbolic constants */
            break;
        case is_value_intrinsic:
        case is_value_unknown:
        case is_value_code:
            /* We assume that it is legal to replace arguments (because it should
               have been verified with the effects that the index is not WRITTEN).
               */
            MAPL(a,
            {
                daExpressionReplaceReference(a, ref, next);
            }, call_arguments(c));
    break;
        default:
    pips_internal_error("unknown tag: %d", 
            (int) value_tag(vin));

    abort();
    }
}

/*
   Add stat to the statement list start
   */
static list addStatementToSequence(statement stat, list seq, list * start)
{
    if(seq == NIL)
    {
        seq = CONS(STATEMENT, stat, seq);
        *start = seq;
    }
    else
    {
        seq = CDR(seq) = CONS(STATEMENT, stat, NIL);
    }

    return seq;
}

/*
   This function returns true if expr has a write effect on the reference ref.
   */
static bool expr_has_write_eff_ref_p(reference ref, expression expr)
{
    bool actionWrite = false;

    list ef = expression_to_proper_effects(expr);

    FOREACH(EFFECT, f, ef)
    {
        entity effEnt = effect_entity(f);

        if(action_write_p(effect_action(f)) && 
                same_entity_p(reference_variable(ref), effEnt))
        {
            actionWrite = true;
        }

    }

    gen_free_list(ef);

    return actionWrite;
}

/*
   This function returns true if stat has a write effect on the reference ref.
   */
static bool stat_has_write_eff_ref_p(reference ref, statement stat)
{
    bool actionWrite = false;

    FOREACH(EFFECT, f, load_proper_rw_effects_list(stat))
    {
        entity effEnt = effect_entity(f);

        if(action_write_p(effect_action(f)) && same_entity_p(reference_variable(ref), effEnt))
        {
            actionWrite = true;
        }

    }

    return actionWrite;
}

static bool add_const_expr_p(statement stat)
{
    pips_assert("stat is an assign call statement", 
            instruction_call_p(statement_instruction(stat)) &&
            ENTITY_ASSIGN_P(call_function(instruction_call(statement_instruction(stat)))));

    expression exp = EXPRESSION(CAR(CDR(call_arguments(instruction_call(statement_instruction(stat))))));

    syntax syn = expression_syntax(exp);

    switch(syntax_tag(syn))
    {
        // exp is a call expression
        case is_syntax_call:
            {
                call ca = syntax_call(syn);

                if(call_constant_p(ca))
                {
                    constant cn = value_constant(entity_initial(call_function(ca)));
                    if(!constant_int_p(cn))
                        return false;
                }
                else if(ENTITY_PLUS_P(call_function(ca)) || ENTITY_MINUS_P(call_function(ca)))
                {
                    cons * arg = call_arguments(ca);

                    // Strange error
                    if ((arg == NIL) || (CDR(arg) == NIL))
                        return false;

                    expression e1 = EXPRESSION(CAR(arg));
                    expression e2 = EXPRESSION(CAR(CDR(arg)));

                    if(!expression_constant_p(e1) && !expression_constant_p(e2))
                        return false;
                }
                else
                    return false;

            }

        case is_syntax_reference:
            break;

        default:
            return false;
    }

    return true;
}

/*
   This function returns true if there is a read-write conflict between
   si and sj, between references refi and refj
   */
static bool stats_has_rw_conf_p(statement si, statement sj, 
        reference refi, reference refj)
{
    if((gen_length(reference_indices(refi)) != 0) ||
            (gen_length(reference_indices(refj)) != 0))
    {
        return false;
    }

    FOREACH(VERTEX, v1,graph_vertices(dep_graph))
    {
        FOREACH(SUCCESSOR, suc,vertex_successors(v1))
        {
            FOREACH(CONFLICT, conf,dg_arc_label_conflicts(successor_arc_label(suc)))
            {
                statement s1 = vertex_to_statement(v1);
                statement s2 = vertex_to_statement(successor_vertex(suc));

                if((s1 == si) &&
                        (s2 == sj) &&
                        effect_read_p(conflict_source(conf)) &&
                        effect_write_p(conflict_sink(conf)) &&
                        !reference_equal_p(refi, refj))
                {
                    return true;
                }

            }
        }
    }

    return false;
}

/*
   This function really does the job.
   */
static list da_process_list(list seq, bool repOnlyInIndex, bool (*stat_to_process)(statement ))
{
    cons * i;

    // At the end of thi function, newSeqStart contains 
    // the new statements sequence
    list newSeq = NIL;
    list newSeqStart = NIL;

    gRepOnlyInIndex = repOnlyInIndex;

    pips_debug(2, "begin da_process_list\n");

    // outStats associates a statement with a replaced reference
    hash_table outStats = hash_table_make(hash_pointer, 0);

    // notSup associates a statement which is not supported
    // with a reference
    hash_table notSup = hash_table_make(hash_pointer, 0);

    // Go through the sequence of statements
    for( i = seq;
            i != NIL;
            i = CDR(i) )
    {
        cons * j;
        cons * last;
        statement si = STATEMENT(CAR(i));

        // Go through the hash table notSup
        HASH_MAP(ent, lStat,
        {
            // If the current statement si is in the hash table then it is not supported.
            // So, let's insert before this statement the out statement that corresponds
            // to the reference.
            if(lStat == i)
            {

                statement statToInsert = (statement)hash_get(outStats, ent);
                newSeq = addStatementToSequence(statToInsert, newSeq, &newSeqStart);
                hash_del(outStats, ent);
            }
        }, notSup);

        ifdebug(2)
        {
            fprintf(stderr,"si\n");
            print_statement(si);
        }

        // If si is not an assign statement, then just add it to the new sequence
        if(!instruction_call_p(statement_instruction(si)) ||
                !ENTITY_ASSIGN_P(call_function(instruction_call(statement_instruction(si)))))
        {
            newSeq = addStatementToSequence(si, newSeq, &newSeqStart);
            continue;
        }

        // ei is the expression of the left value of si
        expression ei = EXPRESSION(CAR(call_arguments(
                        instruction_call(statement_instruction(si)))));

        pips_assert("ei is a reference", 
                syntax_tag(expression_syntax(ei)) == is_syntax_reference);

        // ei is the expression of the right value of si
        expression rVal = EXPRESSION(CAR(CDR(call_arguments(
                            instruction_call(statement_instruction(si))))));

        // ei is the reference of the left value of si
        reference refi = syntax_reference(expression_syntax(ei));

        // If refi is modified by the right part of si, give up
        if(expr_has_write_eff_ref_p(refi, rVal) ||
                !(*stat_to_process)(si))
        {
            newSeq = addStatementToSequence(si, newSeq, &newSeqStart);
            continue;
        }

        // If the left value of si is not a scalar, then just add the statement
        // to the new sequence without doing anything
        if(gen_length(reference_indices(refi)) != 0)
        {
            newSeq = addStatementToSequence(si, newSeq, &newSeqStart);
            continue;
        }

        // gReplaceAllowed is a global variable and is initialized to FALSE.
        // After the end of the next loop, if gReplaceAllowed is true, then
        // means that refi was found in at least one reference index of a sj
        // statement.
        gReplaceAllowed = false;

        // true at the end of the newt loop, if one sj statement is not supported
        bool statNotSupported = false;

        // Go through the remaining statements in the sequence
        last = NIL;
        for( j = CDR(i);
                j != NIL;
                j = CDR(j) )
        {
            statement sj = STATEMENT(CAR(j));

            // If true, sj is not supported. Let's stop the analysis.
            if(!instruction_call_p(statement_instruction(sj)) ||
                    !ENTITY_ASSIGN_P(call_function(instruction_call(statement_instruction(sj)))))
            {
                last = j;
                statNotSupported = true;
                break;
            }

            // At this point, sj has a write effect on refi and is an assign statement.
            // ej is the expression of the sj left value.
            expression ej = EXPRESSION(CAR(call_arguments(
                            instruction_call(statement_instruction(sj)))));

            expression rValj = EXPRESSION(CAR(CDR(call_arguments(
                                instruction_call(statement_instruction(sj))))));

            pips_assert("ej is a reference", 
                    syntax_tag(expression_syntax(ej)) == is_syntax_reference);

            // refj is the reference of the sj left value.
            reference refj = syntax_reference(expression_syntax(ej));

            ifdebug(2)
            {
                printf("sj\n");
                print_statement(sj);
            }

            if(stats_has_rw_conf_p(si, sj, refi, refj))
            {
                ifdebug(2)
                {
                    printf("read write conflict\n");
                    print_statement(si);
                    print_statement(sj);
                }

                last = j;
                statNotSupported = true;
                break;
            }

            // sj doesn't change refi value, then sj is supported.
            // So, let's call daCheckCallReplace() to see if refi is 
            // used in a reference index in this statement.
            if(!stat_has_write_eff_ref_p(refi, sj))
            {
                ifdebug(2)
                {
                    printf("left write effect\n");
                    print_statement(sj);
                }
                gInIndex = 0;
                daCheckCallReplace(instruction_call(statement_instruction(sj)), refi);

                continue;
            }

            // If refi and refj are equal and that refj is not modified by 
            // sj left part, ...
            if(reference_equal_p(refi, refj) &&
                    !expr_has_write_eff_ref_p(refi, rValj))
            {
                last = j;
                break;
            }

            last = j;
            statNotSupported = true;
            break;
        }

        // If refi has not been found in a reference index in at least one sj
        // statement, ...
        if(!gReplaceAllowed)
        {
            // Add si to the new sequence
            newSeq = addStatementToSequence(si, newSeq, &newSeqStart);

            // Delete the out statement corresponding to refi because
            // si is writing refi
            HASH_MAP(ent, stat,
            {
                if(same_entity_p(ent, reference_variable(refi)))
                {
                    hash_del(outStats, ent);
                }
            }, outStats);
            continue;
        }

        // If gReplaceAllowed is true, then do the replacement until last
        for( j = CDR(i);
                (j != last) && (j != NIL);
                j = CDR(j) )
        {

            statement sj = STATEMENT(CAR(j));

            // This function replaces refi by rVal in  sj call
            daCallReplaceReference(instruction_call(statement_instruction(sj)), refi, rVal);
        }

        // curStat holds the current out statement corresponding to the reference refi
        statement curStat = (statement)hash_get(outStats, reference_variable(refi));

        if(curStat != HASH_UNDEFINED_VALUE)
            free_statement(curStat);

        // Put the new out statement corresponding to the reference refi
        hash_put(outStats, reference_variable(refi), si);

        // If the statement "last" is supported, ...
        if(!statNotSupported && last != NIL)
        {
            ifdebug(2)
            {
                printf("replace last\n");
                print_statement(STATEMENT(CAR(last)));
            }

            statement sLast = STATEMENT(CAR(last));

            expression eLast = EXPRESSION(CAR(call_arguments(
                            instruction_call(statement_instruction(sLast)))));

            pips_assert("eLast is a reference", 
                    syntax_tag(expression_syntax(eLast)) == is_syntax_reference);

            reference refLast = expression_reference(eLast);

            list lastRVal = CDR(call_arguments(
                        instruction_call(statement_instruction(sLast))));

            daExpressionReplaceReference(lastRVal, refi, rVal);

            if(same_entity_p(reference_variable(refLast),
                        reference_variable(refi)))
            {
                hash_del(outStats, reference_variable(refi));
            }
        }
        else
        {
            // In notSup hash table, associate last to refi
            hash_put(notSup, reference_variable(refi), last);
        }

    }

    // At the end of the new sequence, add the out statement for each reference
    HASH_MAP(ent, stat,
    {
        newSeq = addStatementToSequence(stat, newSeq, &newSeqStart);
    }, outStats);

    hash_table_free(outStats);
    hash_table_free(notSup);

    return newSeqStart;
}

/*
   This function calls da_process_list() to get the new statements sequence.
   Then, it replaces the old statements sequence by the new one.
   */
static void da_simple_statements_pass(statement s)
{
    list seq;
    list newseq;

    if (!instruction_sequence_p(statement_instruction(s)))
        /* not much we can do with a single statement, or with
         * "complex" statements (ie, with tests/loops/...)
         */
        return;

    seq = sequence_statements(instruction_sequence(statement_instruction(s)));
    newseq = da_process_list(seq, true, add_const_expr_p);

    sequence_statements(instruction_sequence(statement_instruction(s))) = newseq;
    gen_free_list(seq);
}

/*
   This function will call da_simple_statements_pass() to do
   the job until da_simple_statements_pass() is called without
   modification.
   */
static void da_simple_statements(statement s)
{
    // It is a global variable
    gRepDone = true;

    int debCount = 0;

    while(gRepDone && (debCount < 2))
    {
        gRepDone = false;

        debCount++;

        da_simple_statements_pass(s);

        if(debCount == 2)
        {
            pips_user_warning("too many iterations\n");
        }
    }
}

static bool da_simple_sequence_filter(statement s)
{
    instruction i;

    /* Do not recurse through simple calls, for better performance */ 
    i = statement_instruction(s);
    if (instruction_call_p(i))
        return false;
    else
        return true;
}

/*
   This phase aims to deatomize an assign statement in a sequence 
   if this statement is used in a reference index in at least one 
   following statement of the sequence.
   For instance:

   K=K+1
   A(I) = A(K) + 1
   A(J) = A(J) + 1

   will be transformed into:

   A(I) = A(K+1) + 1
   A(J) = A(J) + 1
   K=K+1
   */
bool deatomizer(char * mod_name)
{
    /* get the resources */
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, true);

    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));
	set_ordering_to_statement(mod_stmt);

    dep_graph = (graph) db_get_memory_resource(DBR_DG, mod_name, true);

    set_proper_rw_effects((statement_effects) 
            db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, true));

    debug_on("DEATOMIZER");

    /* Now do the job */

    // To prevent some warnings
    hash_dont_warn_on_redefinition();

    gen_recurse(mod_stmt, statement_domain,
            da_simple_sequence_filter, da_simple_statements);

    // Restore the warning
    hash_warn_on_redefinition();

    pips_assert("Statement is consistent after DEATOMIZER", 
            statement_consistent_p(mod_stmt));

    /* Reorder the module, because new statements have been added */  
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);

    /* update/release resources */
	reset_ordering_to_statement();
    reset_current_module_statement();
    reset_current_module_entity();
    reset_proper_rw_effects();

    debug_off();

    return true;
}
