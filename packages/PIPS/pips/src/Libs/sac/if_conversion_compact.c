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

#include "effects-generic.h"
#include "effects-simple.h"
#include "properties.h"

#include "sac.h"
#include "ricedg.h"


/** 
 * checks wether a statement is a phi function call of the form
 * a=phi(cond, vt,vf)
 * where phi is given by a property
 * 
 * @param s statement to check
 * 
 * @return result of the check
 */
static bool
statement_phi_function_p(statement s)
{
    if(assignment_statement_p(s))
    {
        expression rhs = binary_call_rhs(statement_call(s));
        if(expression_call_p(rhs))
        {
            const char* phi_name = get_string_property("IF_CONVERSION_PHI");
            entity phi = entity_intrinsic(phi_name);
            return same_entity_p(phi,call_function(expression_call(rhs)));
        }
    }
    return false;
}


/** 
 * try to compact two phi-statements into a single one
 * 
 * @param s0 first statement to compact
 * @param s1 second statement to compact
 * 
 * @return true if compaction succeeded, false otherwise
 */
static bool compact_phi_functions(statement s0,statement s1)
{
    /* all datas are handeled as pairs in an array of 2 elements
     * we use i and !i to denote current index and its opposite
     */
    statement s [2] = { s0,s1 };

    expression lhs[2];
    for(size_t i=0;i<2;i++)
        lhs[i]=binary_call_lhs(statement_call(s[i]));

    /* let's compare assignment part */
    if(same_expression_p(lhs[0],lhs[1]) && expression_reference_p(lhs[0]))
    {

        /* ok it may be good, now
           let's compare both phi functions */
        call phis[2];
        reference ref[2];
        expression cond[2], true_val[2], false_val[2];
        for(size_t i=0;i<2;i++)
        {
            ref[i]=expression_reference(lhs[i]);
            phis[i]=expression_call(binary_call_rhs(statement_call(s[i])));
            cond[i]=EXPRESSION(CAR(call_arguments(phis[i])));
            true_val[i]=EXPRESSION(CAR(CDR(call_arguments(phis[i]))));
            false_val[i]=EXPRESSION(CAR(CDR(CDR(call_arguments(phis[i])))));
        }

        /* first constraint : ref[i] == false_val[i] */
        if( expression_reference_p(false_val[0]) && expression_reference_p(false_val[1]))
        {
            if(reference_equal_p(ref[0],expression_reference(false_val[0])) &&
                    reference_equal_p(ref[1],expression_reference(false_val[1])))
            {
                /* second constraint : cond[0] == !cond[1] */
                bool ok = false;
                size_t i;
                for(i =0;i<2;i++)
                {
                    if(expression_call(cond[!i]) && ENTITY_NOT_P(call_function(expression_call(cond[!i]))))
                    {
                        if((ok=expression_equal_p(EXPRESSION(CAR(call_arguments(expression_call(cond[!i])))),
                                        cond[i])))
                            break;
                    }
                }
                if(ok)
                {
                    /* yes, we have in s1 a = cond ? b:a; and in s2 a = !cond ? c: a;
                     * it will become a = cond ? b :c ;*/

                    /* 1:replace false_val[i] by true_val[!i]*/
                    free_expression(false_val[i]);
                    *REFCAR(CDR(CDR(call_arguments(phis[i]))))=(gen_chunkp)true_val[!i];
                    /* 2:unlink true_val[!i]*/
                    *REFCAR(CDR(call_arguments(phis[!i])))=gen_chunk_undefined;
                    /* 3: replace s[!i] by a continue */
                    update_statement_instruction(s[!i],make_continue_instruction());
                    return true;
                }
            }
        }
    }
    return false;
}

/** 
 * checks if there is a write-read conflict between @a source and @a sink
 * according to dg successors @a source_successors
 */
static bool
statement_conflicts_p(statement sink,list source_successors)
{
    FOREACH(SUCCESSOR,s,source_successors)
    {
        statement ss = vertex_to_statement(successor_vertex(s));
        if(ss==sink)
        {
            FOREACH(CONFLICT,c,dg_arc_label_conflicts(successor_arc_label(s)))
            {
                /* if there is a write-read conflict, we cannot do much */
                if ( (effect_write_p(conflict_source(c)) && effect_read_p(conflict_sink(c))) ||
                        (effect_read_p(conflict_source(c)) && effect_write_p(conflict_sink(c))) )
                    return true;

            }
        }
    }
    return false;
}

/*
   This function does the job for each sequence.
   */
static void if_conversion_compact_stats(statement stat,graph dg)
{
    // Only look at the sequence statements
    if(statement_block_p(stat))
    {
        hash_table successors = statements_to_successors(statement_block(stat),dg);
        
        for(list iter = statement_block(stat);!ENDP(iter);POP(iter))
        {
            statement st = STATEMENT(CAR(iter));
            if(statement_phi_function_p(st))
            {
                ifdebug(1) {
                    pips_debug(1,"checking statement:\n");
                    print_statement(st);
                }
                /* iterate over the trailing statements to find a legal statement to compact
                 * we stop when a conflict is found or when we achived our goal
                 */
                list succ = hash_get(successors,st);
                FOREACH(STATEMENT,next,CDR(iter))
                {

                    if(statement_phi_function_p(next))
                    {
                        if(compact_phi_functions(st,next))
                        {
                            ifdebug(1) {
                                pips_debug(1,"compacted into statement:\n");
                                print_statement(st);
                            }
                            break;
                        }
                    }
                    /* look for a write - read conflict between st and next */
                    else if(statement_conflicts_p(next, succ)) break;
                }
            }
        }
        hash_table_free(successors);
    }
}

/*
   This phase is applied after if_conversion phase and will changes:

   .
   .
   .
   I = PHI(L, I1, I)
   .
   .
   .
   I = PHI(.NOT.L, I2, I)
   .
   .
   .

into:

.
.
.
I = PHI(L, I1, I2)
.
.
.
*/
bool if_conversion_compact(char * mod_name)
{
    // get the resources
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, true);

    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));
	set_ordering_to_statement(mod_stmt);

    graph dg = (graph) db_get_memory_resource(DBR_DG, mod_name, true);

    set_proper_rw_effects((statement_effects) 
            db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, true));

    debug_on("IF_CONVERSION_COMPACT_DEBUG_LEVEL");
    // Now do the job

    gen_context_recurse(mod_stmt, dg, statement_domain, gen_true, if_conversion_compact_stats);

    // Reorder the module, because new statements have been added 
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);

    // update/release resources
	reset_ordering_to_statement();
    reset_current_module_statement();
    reset_current_module_entity();
    reset_proper_rw_effects();

    debug_off();

    return true;
}
