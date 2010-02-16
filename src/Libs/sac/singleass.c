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

#include "ricedg.h"
#include "control.h"
#include "callgraph.h"
#include "preprocessor.h"
#include "sac.h"

//Creates a new entity to replace the given one
static entity make_replacement_entity(entity e)
{
    entity new_ent = make_new_scalar_variable_with_prefix(entity_user_name(e),
            get_current_module_entity(),
            entity_basic(e));
    AddLocalEntityToDeclarations(new_ent,get_current_module_entity(),
				 get_current_module_statement());
    return new_ent;
}




static void single_assign_statement(graph dg) {
    hash_table nbPred = hash_table_make(hash_pointer, 0);

    //First, compute the number of incoming DU arcs for each reference
    FOREACH(VERTEX, a_vertex, graph_vertices(dg)) {
        FOREACH(SUCCESSOR, suc, vertex_successors(a_vertex)) {
            FOREACH(CONFLICT, c, dg_arc_label_conflicts(successor_arc_label(suc))) {
                reference r = effect_any_reference(conflict_sink(c));
                int nbRef;
                /* Consider only potential DU arcs (may or must does not matter)
                   and do not consider arrays */
                if ((gen_length(reference_indices(effect_any_reference(conflict_source(c)))) != 0)
                        || (gen_length(reference_indices(effect_any_reference(conflict_sink(c)))) != 0)
                        || !effect_write_p(conflict_source(c))
                        || !effect_read_p(conflict_sink(c)))
                    continue;

                nbRef = (_int) hash_get(nbPred, r);
                if (nbRef == (_int) HASH_UNDEFINED_VALUE)
                    nbRef = 0;
                nbRef++;
                hash_put(nbPred, r, (void*)(_int)nbRef);
            }
        }
    }
    //Then, for each reference which does never stem from more than one Def,
    //change the variable name
    FOREACH(VERTEX, a_vertex, graph_vertices(dg)) {
        hash_table toBeDone = hash_table_make(hash_pointer, 0);
        hash_table hashSuc = hash_table_make(hash_pointer, 0);
        bool var_created = FALSE;
        entity se = entity_undefined;

        FOREACH(SUCCESSOR, suc, vertex_successors(a_vertex)) {
            FOREACH(CONFLICT, c, dg_arc_label_conflicts(successor_arc_label(suc))) {
                list l;
                list lSuc;

                //do something only if we are sure to write
                if ((gen_length(reference_indices(effect_any_reference(conflict_source(c)))) != 0) ||
                        (gen_length(reference_indices(effect_any_reference(conflict_sink(c)))) != 0) ||
                        !effect_write_p(conflict_source(c)) ||
                        !effect_must_p(conflict_source(c)) ||
                        !effect_read_p(conflict_sink(c)))
                    continue;

                //if the module has an OUT effect on the variable, do not replace
                if (0)
                    continue;

                l = hash_get(toBeDone, effect_any_reference(conflict_source(c)));
                lSuc = hash_get(hashSuc, effect_any_reference(conflict_source(c)));

                //If the sink reference has more than one incoming arc, do not change
                //the variable name.
                //In this caeffect_entity(conflict_source(c))se, previous conflicts related to this reference are removed
                //from the work list, and the list is set to NIL in the work list: this way
                //it can be seen in later conflicts also.
                if ((_int)hash_get(nbPred, effect_any_reference(conflict_sink(c))) > 1)
                {
                    if (l != HASH_UNDEFINED_VALUE)
                        gen_free_list(l);
                    l = NIL;

                    if (lSuc != HASH_UNDEFINED_VALUE)
                        gen_free_list(lSuc);
                    lSuc = NIL;
                }
                else if (l != NIL)
                {
                    if(simd_supported_stat_p(vertex_to_statement(a_vertex)) &&
                            simd_supported_stat_p(vertex_to_statement(successor_vertex(suc))))
                    {
                        if (l == HASH_UNDEFINED_VALUE)
                            l = NIL;
                        l = CONS(CONFLICT, c, l);

                        if (lSuc == HASH_UNDEFINED_VALUE)
                            lSuc = NIL;
                        lSuc = CONS(SUCCESSOR, suc, lSuc);
                    }
                    else
                    {
                        if (l != HASH_UNDEFINED_VALUE)
                            gen_free_list(l);
                        l = NIL;

                        if (lSuc != HASH_UNDEFINED_VALUE)
                            gen_free_list(lSuc);
                        lSuc = NIL;
                    }
                }

                hash_put(toBeDone, effect_any_reference(conflict_source(c)), l);
                hash_put(hashSuc, effect_any_reference(conflict_source(c)), lSuc);
            }
        }

        HASH_MAP(r, l, {
                list lSuc = hash_get(hashSuc, r);
                list lCurSuc = lSuc;
                FOREACH(CONFLICT, c, (list)l) {
                entity ne;
                reference rSource;
                reference rSink;
                entity eSource;
                entity eSink;

                // Get the entity corresponding to the source and to the sink
                eSource = effect_entity(conflict_source(c));
                eSink = effect_entity(conflict_sink(c));

                rSource = effect_any_reference(conflict_source(c));
                rSink = effect_any_reference(conflict_sink(c));

                // Get the successor
                successor suc = SUCCESSOR(CAR(lCurSuc));

                statement stat2 = vertex_to_statement(successor_vertex(suc));

                // If the source variable hasn't be replaced yet for the source
                if(var_created == FALSE)
                {
                    // Create a new variable
                    ne = make_replacement_entity(eSource);

                    // Replace the source by the created variable
                    reference_variable(rSource) = ne;

                    pips_debug(1, "ref created %s\n",
                            entity_local_name(effect_entity(conflict_source(c))));

                    // Save the entity corresponding to the created variable
                    se = ne;
                    var_created = TRUE;
                }

                bool actionWrite = FALSE;
                FOREACH(EFFECT, f, load_proper_rw_effects_list(stat2)) {
                    entity effEnt = effect_entity(f) ;

                    if(action_write_p(effect_action(f)) && same_entity_p(eSink, effEnt))
                    {
                        actionWrite = TRUE;
                    }
                }

                expression exp2 = EXPRESSION(CAR(call_arguments(instruction_call(statement_instruction(stat2)))));

                if(!actionWrite)
                {
                    replace_reference(exp2,rSink,se);
                }

                exp2 = EXPRESSION(CAR(CDR(call_arguments(instruction_call(statement_instruction(stat2))))));
                replace_reference(exp2,rSink,se);

                lCurSuc = CDR(lCurSuc);
                }

                var_created = FALSE;

                gen_free_list(l);
                gen_free_list(lSuc);
        },
            toBeDone);

        hash_table_free(toBeDone);
        hash_table_free(hashSuc);
    }
    hash_table_free(nbPred);
}


/* Put a module in a single assignment form.
 */
bool single_assignment(char * mod_name)
{
    /* Get the resources */
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, TRUE);
    graph dg = (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);

    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));
    /* Construct the ordering-to-statement mapping so that we can use the
       dependence graph: */
    set_ordering_to_statement(mod_stmt);

    set_proper_rw_effects((statement_effects)
            db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, TRUE));

    debug_on("SINGLE_ASSIGNMENT_DEBUG_LEVEL");

    /* Now do the job */
    single_assign_statement(dg);

    // Restore the warning
    hash_warn_on_redefinition();

    clean_up_sequences(mod_stmt);

    pips_assert("Statement is consistent after SINGLE_ASSIGNMENT",
            statement_consistent_p(mod_stmt));

    /* Reorder the module, because new statements have been added */
    module_reorder(mod_stmt);

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name,
            compute_callees(mod_stmt));

    /* update/release resources */
    reset_current_module_statement();
    reset_current_module_entity();
    reset_proper_rw_effects();
	reset_ordering_to_statement();

    debug_off();

    return TRUE;
}
