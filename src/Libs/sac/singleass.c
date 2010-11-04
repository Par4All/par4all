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
#include "transformations.h"
#include "callgraph.h"
#include "effects-simple.h"
#include "sac.h"
#include "ricedg.h"

/* collect the first vertex that writes an entity, for each local sclar entity
 * and store it into the returned table entity -> first writing vertex
 * Note: I a musing ordering to now what is the first statement, it may be unreliable ..
 */
static hash_table graph_to_definition(graph dg) {
    hash_table ht = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
    FOREACH(VERTEX,v,graph_vertices(dg)) {
        statement st = vertex_to_statement(v);
        list effects = load_proper_rw_effects_list(st);
        list weffects = effects_write_effects(effects);
        FOREACH(EFFECT,eff,weffects) {
            entity e = effect_any_entity(eff);
            if(entity_scalar_p(e) && local_entity_of_module_p(e,get_current_module_entity())) {
                vertex def = (vertex) hash_get(ht,e);
                if(def == HASH_UNDEFINED_VALUE )
                    hash_put(ht,e,v);
                else if(statement_ordering(vertex_to_statement(v)) < statement_ordering(vertex_to_statement(def)))
                    hash_update(ht,e,v);
            }
        }
        gen_free_list(weffects);
    }
    return ht;
}

/* apply scalar renaming of @p e into @p replacment and follow def-use @p chain
 */
static void do_scalar_renaming_in_def_use(entity e, list chain) {
    entity new= make_new_scalar_variable_with_prefix(entity_user_name(e),get_current_module_entity(),copy_basic(entity_basic(e)));
    AddEntityToCurrentModule(new);
    FOREACH(VERTEX,v,chain) {
        replace_entity(vertex_to_statement(v),e,new);
    }
}
static list vertex_to_full_chain(entity e, vertex v) {
    list chain = CONS(VERTEX,v,NIL);
    set visited = set_make(set_pointer);
    while(!ENDP(vertex_successors(v))) {
        /* among all successors, choose the first non backward */
        successor suc = successor_undefined;
        FOREACH(SUCCESSOR,s,vertex_successors(v)) {
            list effects = load_proper_rw_effects_list(
                    vertex_to_statement(successor_vertex(s))
                    );
            if(!set_belong_p(visited,successor_vertex(s)) &&
                    (effects_write_variable_p(effects,e) ||effects_read_variable_p(effects,e) ) ) 
            {
                if(successor_undefined_p(suc))
                    suc=s;
                else if(statement_ordering(vertex_to_statement(successor_vertex(s))) <
                        statement_ordering(vertex_to_statement(successor_vertex(suc))))
                    suc=s;
            }
        }
        if(successor_undefined_p(suc))
            break;
        else {
            v=successor_vertex(suc);
            chain=CONS(VERTEX,v,chain);
        }
        set_add_element(visited,visited,v);
    }
    set_free(visited);

    return gen_nreverse(chain);
}

static list vertex_to_def_uses(entity e, vertex v) {
    list chains = NIL;
    list chain=vertex_to_full_chain(e,v);
    set visited = set_make(set_pointer);
    /* chain now holds the longest def-use chain, 
     * we will split on each write without cycle */
    set_clear(visited);
    list curr = NIL;
    FOREACH(VERTEX,v,chain) {
        bool backward_dep_p = false;
        set_add_element(visited,visited,v);
        /* get source effects */
        list veffects = load_proper_rw_effects_list(
                vertex_to_statement(v)
                );
        bool vread_p=effects_read_variable_p(veffects,e),
             vwrite_p = effects_write_variable_p(veffects,e);
        /* look for a backward dep, we could use conflicts, indeed */
        FOREACH(SUCCESSOR,s,vertex_successors(v)) {
            list effects = load_proper_rw_effects_list(
                    vertex_to_statement(successor_vertex(s))
                    );
            bool read_p=effects_read_variable_p(effects,e),
                 write_p = effects_write_variable_p(effects,e);
            if(set_belong_p(visited,successor_vertex(s)) && 
                    ( (read_p&&vwrite_p)||(write_p&vread_p) ) 
              )/* got a backward dep */
            {
                backward_dep_p=true;
                break;
            }
        }
        /* split the chain if possible */
        if(vwrite_p && !vread_p && !backward_dep_p) {
            if(!ENDP(curr)) {
                chains=CONS(LIST,gen_nreverse(curr),chains);
                curr=NIL;
            }
        }
        curr=CONS(VERTEX,v,curr);
    }
    if(!ENDP(curr))
        chains=CONS(LIST,gen_nreverse(curr),chains);
    /* drop first chain, it has the side effect of being compliant with replace_entity that cannot change entity declaration */
    for(list iter=chains;!ENDP(iter);POP(iter)) {
        if(ENDP(CDR(iter))) { /* the last */
            list* car = (list*)REFCAR(iter);
            gen_free_list(*car);
            *car=NIL;
        }
    }
    return chains;
}


static void do_scalar_renaming_in_chain(entity e, vertex v) {
    list chains = vertex_to_def_uses(e,v);
    FOREACH(LIST,chain,chains) {
        do_scalar_renaming_in_def_use(e,chain);
        gen_free_list(chain);
    }
    gen_free_list(chains);
}


/* do scalar renaming for graph @p dg
 *
 * the algorithm is simple:
 * first collect all entity -> first write mapping
 * then follow the chain and rename each time you find a write
 * backward dependencies stop the process.
 */
static void do_scalar_renaming_statement(graph dg) {

    hash_table def = graph_to_definition(dg);
    /* we follow the def - use chain for each entity */
    FOREACH(ENTITY,e,entity_declarations(get_current_module_entity())) {
        /* only local entity */
        if(local_entity_of_module_p(e,get_current_module_entity()) &&
                entity_scalar_p(e)
                ) {
            vertex v = (vertex)hash_get(def,e);
            if(v!= HASH_UNDEFINED_VALUE)
                do_scalar_renaming_in_chain(e,v);
        }
    }
    hash_table_free(def);
}


/* rename scalars to remove some false dependencies
 */
bool scalar_renaming(char * mod_name)
{
    /* Get the resources */
    statement mod_stmt = (statement)
        db_get_memory_resource(DBR_CODE, mod_name, true);
    graph dg = (graph) db_get_memory_resource(DBR_DG, mod_name, true);

    set_current_module_statement(mod_stmt);
    set_current_module_entity(module_name_to_entity(mod_name));
    set_ordering_to_statement(mod_stmt); /* needed for vertex_to_statement */

    set_proper_rw_effects((statement_effects)
                    db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, true));

    debug_on("SCALAR_RENAMING_DEBUG_LEVEL");

    /* Now do the job */
    do_scalar_renaming_statement(dg);

    pips_assert("Statement is consistent after SCALAR_RENAMING",
            statement_consistent_p(mod_stmt));

    /* Reorder the module, because new statements have been added */
    module_reorder(mod_stmt);

    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);

    /* update/release resources */
    reset_current_module_statement();
    reset_current_module_entity();
    reset_proper_rw_effects();
	reset_ordering_to_statement();

    debug_off();

    return true;
}
