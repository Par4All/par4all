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
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "pipsdbm.h"

#include "effects-generic.h"

#include "control.h"
#include "callgraph.h"
#include "effects-simple.h"
#include "alias-classes.h"
#include "sac.h"
#include "ricedg.h"
#include "atomizer.h"

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

static void single_assign_replace(statement in, reference old, entity new,bool replace_assign)
{
    list actionRead=NIL,actionWrite=NIL;
    FOREACH(EFFECT, f, load_proper_rw_effects_list(in)) {
        reference effRef = effect_any_reference(f) ;

        if(effect_write_p(f) && references_must_conflict_p(old, effRef))
        {
            actionWrite = CONS(EFFECT,f,actionWrite);
        }
        if(effect_read_p(f) && references_must_conflict_p(old, effRef))
        {
            actionRead = CONS(EFFECT,f,actionRead);
        }
    }
    if(!ENDP(actionWrite)||!ENDP(actionRead))
    {
        if((!ENDP(actionRead)&&ENDP(actionWrite)) || (!ENDP(actionWrite)&&ENDP(actionRead)&&replace_assign))
        {
            pips_debug(1,"replacing entity %s by %s in statement:\n",entity_user_name(reference_variable(old)),entity_user_name(new));
            ifdebug(1) { print_statement(in); }
            replace_reference(in,old,new);
            FOREACH(EFFECT,f,actionRead) replace_entity(f,reference_variable(old),new);
        }
        else if(!ENDP(actionWrite)&&ENDP(actionRead)){
            pips_debug(1,"not replacing entity %s by %s in statement:\n",entity_user_name(reference_variable(old)),entity_user_name(new));
            ifdebug(1) { print_statement(in); }
        }
        /* we are in trouble there: we should only change the read part */
        else
        {
            /* if it is an assign expression ,checks lhs part for only write effect and rhs for only read effects */
            if(assignment_statement_p(in))
            {
                call c = statement_call(in);
                expression lhs = binary_call_lhs(c);
                expression rhs = binary_call_rhs(c);
                list reffects = proper_effects_of_expression(rhs);
                if(!effects_write_variable_p(reffects,reference_variable(old)) && effects_read_variable_p(reffects,reference_variable(old)))
                {
                    pips_debug(1,"replacing entity %s by %s in lhs of assign statement:\n",entity_user_name(reference_variable(old)),entity_user_name(new));
                    ifdebug(1) { print_statement(in); }
                    replace_reference(lhs,old,new);
                    FOREACH(EFFECT,f,actionRead) replace_entity(f,reference_variable(old),new);
                }
                else
                    pips_internal_error("cannot handle this case\n");
                gen_full_free_list(reffects);
            }
            else
                pips_internal_error("cannot handle this case\n");

        }
        gen_free_list(actionWrite);
        gen_free_list(actionRead);
    }
}

static void single_assign_statement(graph dg) {

    set all_entities = set_make(set_pointer);
    set_assign_list(all_entities,entity_declarations(get_current_module_entity()));

    //First, compute the number of incoming DU arcs for each entity
    FOREACH(VERTEX, a_vertex, graph_vertices(dg)) {
        FOREACH(SUCCESSOR, suc, vertex_successors(a_vertex)) {
            FOREACH(CONFLICT, c, dg_arc_label_conflicts(successor_arc_label(suc))) {
                /* Consider only potential DU arcs (may or must does not matter)
                   and do not consider arrays */
                if (reference_scalar_p(effect_any_reference(conflict_source(c))) &&
                        reference_scalar_p(effect_any_reference(conflict_sink(c))) &&
                        effect_write_p(conflict_source(c)) &&
                        effect_read_p(conflict_sink(c)) &&
                        vertex_ordering(a_vertex) >= vertex_ordering(successor_vertex(suc)))
                {
                    entity e = reference_variable(effect_any_reference(conflict_sink(c)));
                    pips_debug(1,"removing %s from ssa entities, conflicts betwwen statements:\n",entity_user_name(e));
                    ifdebug(1) { print_statement(vertex_to_statement(a_vertex)); print_statement(vertex_to_statement(successor_vertex(suc))); }

                    set_del_element(all_entities,all_entities,e);
                }
            }
        }
    }

    /* Then, for each entity suitable for a substitution,change the variable name
     */
    FOREACH(VERTEX, a_vertex, graph_vertices(dg)) {
        statement currStat= vertex_to_statement(a_vertex);
        hash_table toBeDone = hash_table_make(hash_pointer, 0);
        hash_table hashSuc = hash_table_make(hash_pointer, 0);
        bool var_created = FALSE;
        entity se = entity_undefined;

        FOREACH(SUCCESSOR, suc, vertex_successors(a_vertex)) {
            FOREACH(CONFLICT, c, dg_arc_label_conflicts(successor_arc_label(suc))) {
                list l;
                list lSuc;

                //do something only if we are sure to write
                if (set_belong_p(all_entities, reference_variable(effect_any_reference(conflict_sink(c)))))
                {
                    if (reference_scalar_p(effect_any_reference(conflict_source(c))) &&
                            reference_scalar_p(effect_any_reference(conflict_sink(c))) &&
                            effect_write_p(conflict_source(c)) &&
                            effect_read_p(conflict_sink(c)))
                    {

                        l = hash_get_default_empty_list(toBeDone, effect_any_reference(conflict_source(c)));
                        lSuc = hash_get_default_empty_list(hashSuc, effect_any_reference(conflict_source(c)));

                        l = CONS(CONFLICT, c, l);
                        lSuc = CONS(SUCCESSOR, suc, lSuc);

                        hash_put_or_update(toBeDone, effect_any_reference(conflict_source(c)), l);
                        hash_put_or_update(hashSuc, effect_any_reference(conflict_source(c)), lSuc);
                    }
                }
            }
        }


        void *hash_iter=NULL;
        reference r;
        list l;
        while( (hash_iter = hash_table_scan(toBeDone,hash_iter,(void**)&r,(void**)&l)) )
        {
            list lSuc = hash_get(hashSuc, r);
            list lCurSuc = lSuc;
            FOREACH(CONFLICT, c, l) {
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
                if(!var_created)
                {
                    // Create a new variable
                    ne = make_replacement_entity(eSource);
                    pips_debug(1, "ref created %s\n",
                            entity_local_name(effect_entity(conflict_source(c))));

                    single_assign_replace(currStat,rSource,ne,true);

                    // Save the entity corresponding to the created variable
                    se = ne;
                    var_created = true;
                }
                single_assign_replace(stat2,rSink,se,false);


                lCurSuc = CDR(lCurSuc);
            }

            var_created = false;

            gen_free_list(l);
            gen_free_list(lSuc);
        }

        hash_table_free(toBeDone);
        hash_table_free(hashSuc);
    }
    set_free(all_entities);
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
