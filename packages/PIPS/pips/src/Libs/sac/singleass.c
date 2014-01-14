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
#include "text-util.h"
#include "pipsdbm.h"

#include "effects-generic.h"

#include "control.h"
#include "transformations.h"
#include "effects-simple.h"
#include "sac.h"
#include "ricedg.h"

/* helper thats checks if entity @p e is involved in a conflict of successor @p s
 * */
static bool successor_conflicts_on_entity_p(successor s, entity e) {
    FOREACH(CONFLICT,c,dg_arc_label_conflicts(successor_arc_label(s))) {
        if(same_entity_p(e,effect_any_entity(conflict_source(c))) ||
                same_entity_p(e,effect_any_entity(conflict_sink(c)))
          )
            return true;
    }
    return false;
}
/* helper that checks if there is a cycle from @p curr back to @p v following chains of @p e */
static bool vertex_in_cycle_aux_p(vertex v, entity e, vertex curr,set visited) {
    if(v==curr) return true;
    if(set_belong_p(visited,curr)) return false;
    set_add_element(visited,visited,curr);
    FOREACH(SUCCESSOR,s,vertex_successors(v)){
        if(successor_conflicts_on_entity_p(s,e) &&
                vertex_in_cycle_aux_p(v,e,successor_vertex(s),visited))
            return true;
    }
    return false;
}

/* check if there is a cycle from @pv to @p v following chains from @p e */
static bool vertex_in_cycle_p(vertex v, entity e ) {
    set visited = set_make(set_pointer);
    FOREACH(SUCCESSOR,s,vertex_successors(v)){
        if(successor_conflicts_on_entity_p(s,e) &&
                vertex_in_cycle_aux_p(v,e,successor_vertex(s),visited)) {
                set_free(visited);
                return true;
            }
    }
    set_free(visited);
    return false;
}

/* @return the list of vertices for which a write on @p e leads to a new declaration
 * that is writes without reductions or cycles
 */
static set graph_to_live_writes(graph g, entity e) {
    set live_writes = set_make(set_pointer);
    FOREACH(VERTEX,v,graph_vertices(g)){
        list effects = load_proper_rw_effects_list(vertex_to_statement(v));
        bool read_p = effects_read_variable_p(effects,e),
             write_p = effects_write_variable_p(effects,e);
        if( write_p && !read_p && !vertex_in_cycle_p(v,e) )
            set_add_element(live_writes,live_writes,v);
    }
    return live_writes;
}

static void do_scalar_renaming_in_successors(vertex v, entity e, entity new, set live_writes, hash_table visited, hash_table renamings)
{
    FOREACH(SUCCESSOR,s,vertex_successors(v)) {
        if(successor_conflicts_on_entity_p(s,e)) {
            vertex v2 = successor_vertex(s);
            bool read_p = false;
            /* check if sink is read */
            FOREACH(CONFLICT,c,dg_arc_label_conflicts(successor_arc_label(s))) {
                if(effect_read_p(conflict_sink(c))) {
                    read_p= true;
                    break;
                }
            }
            /* this successor belongs to the def-use chain */
            if(read_p && !set_belong_p(live_writes,v2)) {
                /* was it already renamed ? */
                entity renamed = (entity)hash_get(visited,v2);
                /* no -> proceeed */
                if(renamed == HASH_UNDEFINED_VALUE) {
                    hash_put(visited,v2,new);
                    replace_entity(vertex_to_statement(v2),e,new);
                    do_scalar_renaming_in_successors(v2,e,new,live_writes,visited,renamings);
                }
                /* yes, but no conflict */
                else if (same_entity_p(renamed,new) ) {
                    continue;
                }
                /* yes and a conbflict -> fix it by adding an assign */
                else { 
                    set renaming = (set) hash_get(renamings,v);
                    if(renaming == HASH_UNDEFINED_VALUE || !set_belong_p(renaming,new)) {
                        insert_statement(vertex_to_statement(v),
                                make_assign_statement(entity_to_expression(renamed),entity_to_expression(new)),
                                false);
                        if(renaming == HASH_UNDEFINED_VALUE)
                            renaming=set_make(set_pointer);
                        set_add_element(renaming,renaming,new);
                        hash_put_or_update(renamings,v,renaming);
                    }
                    continue;
                }
            }
        }
    }
}

static void do_scalar_renaming_in_vertex(vertex v, entity e, set live_writes, hash_table visited, hash_table renamings) {

    /* create new assigned value */
    entity new = make_new_scalar_variable_with_prefix(entity_user_name(e),get_current_module_entity(),copy_basic(entity_basic(e)));
    entity_initial(new)=copy_value(entity_initial(e));
    AddEntityToCurrentModule(new);
    replace_entity(vertex_to_statement(v),e,new);
    /* propagate it to each reading successor */
    do_scalar_renaming_in_successors(v,e,new,live_writes,visited,renamings);
}


static void do_scalar_renaming_in_graph(graph g, entity e) {
    set live_writes = graph_to_live_writes(g, e);
    hash_table visited = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
    hash_table renamings = hash_table_make(hash_pointer,HASH_DEFAULT_SIZE);
    list l = set_to_sorted_list(live_writes,compare_vertex);
    FOREACH(VERTEX,v,l) {
        statement s = vertex_to_statement(v);
        if(!declaration_statement_p(s) ||
                    gen_chunk_undefined_p(gen_find_eq(e,statement_declarations(s)))) 
            do_scalar_renaming_in_vertex(v,e,live_writes,visited,renamings);
    }
    hash_table_free(visited);
    HASH_FOREACH(entity,k,set,v,renamings) set_free(v);
    hash_table_free(renamings);
    gen_free_list(l);
    set_free(live_writes);
}

/* do scalar renaming for graph @p dg of module @p module and statements @p module_statement
 */
static void do_scalar_renaming(entity module, statement module_statement, graph dg) {

    FOREACH(ENTITY,e,entity_declarations(module)) {
        /* only local non static scalar entities
         * non static as a quick fix ...*/
        if(local_entity_of_module_p(e,module) && entity_scalar_p(e) && !entity_static_variable_p(e))
            do_scalar_renaming_in_graph(dg,e);
    }
    module_clean_declarations(module,module_statement);
    unnormalize_expression(module_statement); // overkill ? It's not a game of kick-the-orphe
}

/* recursievly computes the set of all chains involving e starting from v */
static set vertex_to_chains(vertex v, entity e, set visited) {
    set all = set_make(set_pointer);
    if(!set_belong_p(visited,v)) {
        set_add_element(visited,visited,v);
        list effects = load_proper_rw_effects_list(vertex_to_statement(v));
        bool read_p = effects_read_variable_p(effects,e),
             write_p = effects_write_variable_p(effects,e);
        if(read_p||write_p) {
            list this = CONS(VERTEX,v,NIL);
            set_add_element(all,all,this);
            FOREACH(SUCCESSOR,s,vertex_successors(v)) {
                if(successor_conflicts_on_entity_p(s,e)) {
                    vertex v2 = successor_vertex(s);
                    if(statement_ordering(vertex_to_statement(v2)) >
                            statement_ordering(vertex_to_statement(v)))
                    {
                        set tmp2=vertex_to_chains(v2,e,visited);
                        SET_FOREACH(list,ltmp,tmp2)
                            set_add_element(all,all,CONS(VERTEX,v,ltmp));
                        set_free(tmp2);
                    }
                }
            }
        }
    }
    return all;
}

/* we know l is included in l2, let's remove redundant arcs */
static void do_prune_arcs(list l, list l2) {
    bool same_chain_p = false;
    vertex prev = vertex_undefined;
    for(;!ENDP(l)&&!ENDP(l2);POP(l2)) {
        vertex v = VERTEX(CAR(l)),
               v2=VERTEX(CAR(l2));
        if(v==v2) {
            same_chain_p=true;
            prev=v;
            POP(l);
        }
        else if(same_chain_p) {
            /* arc between prev and v are not needed */
            set remove = set_make(set_pointer);
            FOREACH(SUCCESSOR,s,vertex_successors(prev))
                if(v==successor_vertex(s))
                    set_add_element(remove,remove,s);
            SET_FOREACH(successor,s,remove)
                gen_remove_once(&vertex_successors(prev),s);
            set_free(remove);
        }
    }
}

/* remove all conflicts that involve entity e
 * and that can be regenerated from another conflict chain
 */
static void do_simplify_dg(graph g, entity e) {
    set chains = set_make(set_pointer);
    /* SG: I am not sure it is ok to prune the exploration space like this */
    set visited = set_make(set_pointer);
    FOREACH(VERTEX,v,graph_vertices(g)) {
        set tmp = vertex_to_chains(v,e,visited);
        set_union(chains,chains,tmp);
        set_free(tmp);
    }
    set_free(visited);
    /* arcs now holds all possible arcs of g that impact e
     * let's remove the chains that are not needed, that is those that have an englobing chain */
    SET_FOREACH(list,l,chains) {
        SET_FOREACH(list,l2,chains) {
            /* if s is included in s2, arcs in s not is s2 are not needed */
            if(l!=l2) {
                set s = set_make(set_pointer);
                set_assign_list(s,l);
                set s2 = set_make(set_pointer);
                set_assign_list(s2,l2);
                /* prune some arcs */
                if(set_inclusion_p(s,s2))
                    do_prune_arcs(l,l2);
                set_free(s);
                set_free(s2);
            }
        }
    }
#if 0 //that's some ugly debug !
    FILE * fd = fopen("/tmp/a.dot","w");
    prettyprint_dot_dependence_graph(fd,get_current_module_statement(),g);
    fclose(fd);
#endif

}
/* removes redundant arcs from dg.
 * An arc from v to v' is redundant if there exist a chain in dg that 
 * goes from v to v'
 */
static void simplify_dg(entity module, graph dg) {
    FOREACH(ENTITY,e,entity_declarations(get_current_module_entity())) {
        /* only local entity */
        if(local_entity_of_module_p(e,module) && entity_scalar_p(e) ) {
            do_simplify_dg(dg,e);
        }
    }
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

    /* prune graph */
    simplify_dg(get_current_module_entity(),dg);

    /* Now do the job */
    do_scalar_renaming(get_current_module_entity(),get_current_module_statement(),dg);

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
