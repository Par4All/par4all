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

/** 
* @file delay.c
* 
* @author Serge Guelton <serge.guelton@enst-bretagne.fr>
* @version 
* @date 2010-12-15
* 
* Implementation of inter procedural load / store delaying
*
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
#include "preprocessor.h"

#include "effects-generic.h"
#include "effects-convex.h"
#include "properties.h"

#include "callgraph.h"
#include "transformations.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"
#include "control.h"

#include "ricedg.h"
#include "effects-simple.h"
#include "accel-util.h"

static bool delay_communications_interprocedurally_p ;

/* helper to transform preferences in references */
static void do_remove_preference(cell c){
    if(cell_preference_p(c)) {
        reference r = copy_reference(
                preference_reference(cell_preference(c))
                );
        free_preference(cell_preference(c));
        cell_tag(c)=is_cell_reference;
        cell_reference(c)=r;
    }
}

/* entry point to transform preferences in references */
void remove_preferences(void * obj) {
    gen_recurse(obj,cell_domain,do_remove_preference,gen_null);
}

static graph dependence_graph=graph_undefined;

static bool simd_load_call_p(call c) {
  const char* funcName = entity_local_name(call_function(c));
  const char* simd= get_string_property("ACCEL_LOAD");
  return    same_stringn_p(funcName, simd, strlen(simd));
}
static bool simd_work_call_p(call c) {
  const char* funcName = entity_local_name(call_function(c));
  const char* simd= get_string_property("ACCEL_WORK");
  return    same_stringn_p(funcName, simd, strlen(simd));
}
static bool simd_store_call_p(call c) {
  const char* funcName = entity_local_name(call_function(c));
  const char* simd= get_string_property("ACCEL_STORE");
  return    same_stringn_p(funcName, simd, strlen(simd));
}

bool simd_load_stat_p(statement stat)
{
  return statement_call_p(stat) && simd_load_call_p(statement_call(stat));
}
bool simd_work_stat_p(statement stat)
{
  return statement_call_p(stat) && simd_work_call_p(statement_call(stat));
}
bool simd_store_stat_p(statement stat)
{
  return statement_call_p(stat) && simd_store_call_p(statement_call(stat));
}

/* This function returns true if the statement is a simd loadsave
 * statement
 */
bool simd_dma_stat_p(statement stat)
{
    return simd_load_stat_p(stat) || simd_store_stat_p(stat);
}

/* This function returns true if the statement is a simd
 * statement
 */
bool simd_stat_p(statement stat)
{
    return simd_dma_stat_p(stat) || simd_work_stat_p(stat);
}



static bool dma_conflict_p(conflict c)
{
    effect source = conflict_source(c),
           sink = conflict_sink(c);
    descriptor dsource = effect_descriptor(source),
               dsink = effect_descriptor(sink);
    if(descriptor_convex_p(dsource) && descriptor_convex_p(dsink))
    {
        Psysteme psource = descriptor_convex(dsource),
                 psink = descriptor_convex(dsink);
        return !(sc_inclusion_p(psink,psource) || sc_inclusion_p(psource,psink));
    }
    /* be conservative */
    return true;
}

/* checks if there exist a conflict between @p s0 and @p s1
 * according to the dependency graph
 */
static bool statements_conflict_p(statement s0, statement s1) {
    /* in intra procedural, a store always conflicts with a return */
    if(!delay_communications_interprocedurally_p &&
            ( simd_store_stat_p(s0) || simd_store_stat_p(s1) ) &&
            ( return_statement_p(s0) || return_statement_p(s1) ) )
        return true;

    /* special hook for loop statements: dependency on the index are not well generated */
    if(statement_loop_p(s1)) {
        entity index = loop_index(statement_loop(s1));
        set re = get_referenced_entities(s0);
        bool conflict = set_belong_p(re,index);
        set_free(re);
        return conflict;
    }

    FOREACH(VERTEX,v,graph_vertices(dependence_graph)) {
        statement s = vertex_to_statement(v);
        if(statement_ordering(s) == statement_ordering(s0) || statement_ordering(s) == statement_ordering(s1)) {
            intptr_t expected = statement_ordering(s) == statement_ordering(s0) ? statement_ordering(s1) : statement_ordering(s0);
            FOREACH(SUCCESSOR,su,vertex_successors(v)) {
                if(statement_ordering(vertex_to_statement(successor_vertex(su))) == expected ) {
                    return true;
                }
            }
        }
    }
  return false;
}
/* same as statements_conflict_p but W-* conflicts are ignored if load_p,
 * R-* conflicts are ignored if not load_p */
static bool statements_conflict_relaxed_p(statement s0, statement s1, bool load_p) {
    /* in intra procedural, a store always conflicts with a return */
    if(!delay_communications_interprocedurally_p &&
            ( simd_store_stat_p(s0) || simd_store_stat_p(s1) ) &&
            ( return_statement_p(s0) || return_statement_p(s1) ) )
        return true;

    FOREACH(VERTEX,v,graph_vertices(dependence_graph)) {
        statement s = vertex_to_statement(v);
        if(statement_ordering(s) == statement_ordering(s0) ) {
            intptr_t expected = statement_ordering(s1) ;
            FOREACH(SUCCESSOR,su,vertex_successors(v)) {
                if(statement_ordering(vertex_to_statement(successor_vertex(su))) == expected ) {
                    FOREACH(CONFLICT,c,dg_arc_label_conflicts(successor_arc_label(su))) {
                        if( (load_p && !effect_write_p(conflict_source(c))) ||
                                (!load_p && !effect_read_p(conflict_source(c))) )
                            return true;
                    }
                }
            }
        }
    }
  return false;
}
static bool dma_statements_conflict_p(statement s0, statement s1) {
    FOREACH(VERTEX,v,graph_vertices(dependence_graph)) {
        statement s = vertex_to_statement(v);
        if(statement_ordering(s) == statement_ordering(s0) || statement_ordering(s) == statement_ordering(s1)) {
            intptr_t expected = statement_ordering(s) == statement_ordering(s0) ? statement_ordering(s1) : statement_ordering(s0);
            FOREACH(SUCCESSOR,su,vertex_successors(v)) {
                if(statement_ordering(vertex_to_statement(successor_vertex(su))) == expected ) {
                    FOREACH(CONFLICT,c,dg_arc_label_conflicts(successor_arc_label(su)))
                        if(dma_conflict_p(c)) return true;
                }
            }
        }
    }
  return false;
}

/* helper for do_recurse_statements_conflict_p */
typedef struct {
    statement s;
    bool conflict;
} conflict_t;

/* helper for recurse_statements_conflict_p */
static void do_recurse_statements_conflict_p(statement s, conflict_t *c) {
    if(statement_ordering(s) == statement_ordering(c->s))
        c->conflict=dma_statements_conflict_p(s,c->s);
    else if((c->conflict=statements_conflict_relaxed_p(c->s,s,simd_load_stat_p(c->s))))
            gen_recurse_stop(0);
}

/* checks if there is a conflict between @p s and any statement in @p in */
static bool recurse_statements_conflict_p(statement s, statement in) {
    conflict_t c = { s, false };
    gen_context_recurse(in,&c,statement_domain,gen_true,do_recurse_statements_conflict_p);
    return c.conflict;
}

typedef struct {
  bool result;
  list stats;
  bool backward;
  bool need_flatten;
  entity caller;
  statement caller_statement;
} context;

static context context_dup(context *c) {
  context cp = { c->result, gen_copy_seq(c->stats), c->backward, c->need_flatten , c->caller };
  return cp;
}

static void delay_communications_statement(statement, context *, list *block);

static void delay_communications_sequence(sequence s, context *c, list *block) {
  list stats = gen_copy_seq(sequence_statements(s));
  if(c->backward) stats=gen_nreverse(stats); //reverse the walking when performing backward movements
  FOREACH(STATEMENT,st,stats) {
    delay_communications_statement(st,c,&sequence_statements(s));
  }
  gen_free_list(stats);
}
static void create_block_if_needed(statement *s, list **block) {
    if(!statement_block_p(*s) && !*block) {
        pips_assert("declaration must be in a block\n",!declaration_statement_p(*s));
        statement scopy = instruction_to_statement(statement_instruction(*s));
        statement_instruction(*s)=instruction_undefined;
        list tmp = CONS(STATEMENT,scopy,NIL);
        update_statement_instruction(*s,make_instruction_block(tmp));
        *block=&sequence_statements(statement_sequence(*s));
        *s=scopy;

    }
}

static void insert_statement_in_block(statement s, statement inserted, bool before, list *block) {
    if(statement_block_p(s))
        insert_statement(s,inserted,before);
    else {
        create_block_if_needed(&s,&block);
        if(before)
            *block=gen_insert_before(inserted,s,*block);
        else
            gen_insert_after(inserted,s,*block);
    }
}

static void manage_conflicts(statement s, context *c,bool before, list *block) {
  /* check conflicts with current stats */
  list tstats = gen_copy_seq(c->stats);
  FOREACH(STATEMENT,st,tstats) {
    if(statements_conflict_p(s,st)) {
      insert_statement_in_block(s,st,before,block);
      gen_remove_once(&c->stats,st);
    }
  }
  gen_free_list(tstats);
}

static void delay_communications_call(statement s, context *c, list *block) {
  manage_conflicts(s,c,!c->backward,block);
  /* memorize additional dma */
  if( (c->backward && simd_load_stat_p(s)) ||
          (!c->backward && simd_store_stat_p(s)) ) {
    c->stats=CONS(STATEMENT,copy_statement(s),c->stats);
    intptr_t o = statement_ordering(s);
    update_statement_instruction(s,make_continue_instruction());
    statement_ordering(s)=o;
    gen_free_list(statement_declarations(s)); statement_declarations(s)=NIL;
    if(!empty_comments_p(statement_comments(s))) free(statement_comments(s)); statement_comments(s)=empty_comments;
  }
}

static void delay_communications_test(statement s, context *c, list *block) {
  manage_conflicts(s,c,!c->backward, block);
  /* explore both branches independently */
  test t = statement_test(s);
  context c0 = context_dup(c), c1=context_dup(c);
  delay_communications_statement(test_true(t),&c0,NULL);
  delay_communications_statement(test_false(t),&c1,NULL);
  /* manage state consistency after s :
   * currently very pessimistic: if a dma is made on a branch, it must be
   * done at the end of the branch.
   * A more elaborated version could be to check that same pointers are stored
   * and that they can be merged later ...
   */
  list tstats = gen_copy_seq(c0.stats);
  FOREACH(STATEMENT,st0,tstats) {
    intptr_t o = statement_ordering(st0);
    bool new = true;
    FOREACH(STATEMENT,st,c->stats) {
      if(statement_ordering(st) == o ) {
        new=false;
        break;
      }
    }
    if(new) {
      insert_statement_in_block(test_true(t),copy_statement(st0),c->backward,NULL);
      gen_remove_once(&c0.stats,st0);
      gen_remove_once(&c->stats,st0);
    }
  }
  gen_free_list(tstats);
  /* the same for the false branch */
  tstats = gen_copy_seq(c1.stats);
  FOREACH(STATEMENT,st1,tstats) {
    intptr_t o = statement_ordering(st1);
    bool new = true;
    FOREACH(STATEMENT,st,c->stats) {
      if(statement_ordering(st) == o ) {
        new=false;
        break;
      }
    }
    if(new) {
      insert_statement_in_block(test_false(t),copy_statement(st1),c->backward,NULL);
      gen_remove_once(&c1.stats,st1);
      gen_remove_once(&c->stats,st1);
    }
  }
  gen_free_list(tstats);
  /* now we have the opposite case: a dma was made on a branch
   * but not on the other one,
   * in that case we must do it at the end of the other branch
   */
  tstats = gen_copy_seq(c->stats);
  FOREACH(STATEMENT,st,tstats) {
    intptr_t o = statement_ordering(st);
    /* true branch */
    bool tfound =false,ffound=false;
    FOREACH(STATEMENT,st0,c0.stats)
      if((tfound= (o == statement_ordering(st0)))) break;
    /* false branch */
    FOREACH(STATEMENT,st1,c1.stats)
      if((ffound= (o == statement_ordering(st1)))) break;
    /* insert it if it's missing somewhere */
    if(tfound && !ffound ) {
      insert_statement_in_block(test_true(t),copy_statement(st),c->backward,block);
      gen_remove_once(&c->stats,st);
    }
    if(!tfound && ffound ) {
      insert_statement_in_block(test_false(t),copy_statement(st),c->backward,block);
      gen_remove_once(&c->stats,st);
    }
  }
  gen_free_list(tstats);
}

static void delay_communications_anyloop(statement s, context *c, list *block) {
  statement body=statement_undefined;
  if(statement_loop_p(s)) body =loop_body(statement_loop(s));
  else if(statement_forloop_p(s)) body =forloop_body(statement_forloop(s));
  else if(statement_whileloop_p(s)) body =whileloop_body(statement_whileloop(s));
  pips_assert("all loops have body\n",!statement_undefined_p(s));

  /* first step is to check conflict with the head */
  manage_conflicts(s,c,!c->backward,block);
  /* then we check if there is a conflict inside the loop.
   * In that case better insert before the loop than inside */
  list tstats = gen_copy_seq(c->stats);
  FOREACH(STATEMENT,st,tstats) {
    if(recurse_statements_conflict_p(st,body) ) { // conflict with other iterations
      insert_statement_in_block(s,st,!c->backward,block);
      gen_remove_once(&c->stats,st);
    }
  }
  gen_free_list(tstats);

  /* then we propagate the dma inside the body */
  context cb = context_dup(c);
  delay_communications_statement(body,c,NULL);


  /* then we check for conflicts with indices or over iterations */
  tstats = gen_copy_seq(c->stats);
  FOREACH(STATEMENT,st,tstats) {
    if(statements_conflict_p(st,s) ||// conflict with the iteration
            recurse_statements_conflict_p(st,body) ) { // conflict with other iterations
      insert_statement_in_block(body,st,c->backward,NULL);
      gen_remove_once(&c->stats,st);
    }
  }
  gen_free_list(tstats);
  /* if statements have been moved outside the loop
   * then we (may) need to flatten
   */
  FOREACH(STATEMENT,st,c->stats) {
      if(gen_chunk_undefined_p(gen_find_eq(st,cb.stats))) {
          c->need_flatten|=true;
          break;
      }
  }
  gen_free_list(cb.stats);

}

static void delay_communications_statement(statement s, context *c, list *block) {
  instruction i = statement_instruction(s);
  switch(instruction_tag(i)) {
    case is_instruction_expression:/* we could do better */
    case is_instruction_call:
      delay_communications_call(s,c,block);break;
    case is_instruction_sequence:
      delay_communications_sequence(instruction_sequence(i),c,NULL);break;
    case is_instruction_test:
      delay_communications_test(s,c,NULL);break;
    case is_instruction_loop:
    case is_instruction_whileloop:
    case is_instruction_forloop:
      delay_communications_anyloop(s,c,NULL);break;
    default:
      pips_user_warning("not implemented yet, full memory barrier is assumed\n");
      {
          list tstats= gen_copy_seq(c->stats);
          FOREACH(STATEMENT,st,tstats) {
              insert_statement_in_block(s,st,!c->backward,block);
              gen_remove_once(&c->stats,st);
          }
          gen_free_list(tstats);
      }

  };
  pips_assert("everything is ok",statement_consistent_p(s));
}

static statement translate_arguments(call ca, statement s) {
  statement new = copy_statement(s);
  list fp = module_formal_parameters(get_current_module_entity());
  pips_assert("as many parameters as formal arguments\n",gen_length(call_arguments(ca)) == gen_length(fp));
  for(list aiter = call_arguments(ca), piter=fp;
      !ENDP(aiter);
      POP(aiter),POP(piter)) {
    expression arg = EXPRESSION(CAR(aiter));
    entity fe = ENTITY(CAR(piter));
    replace_entity_by_expression(new,fe,arg);// this perform the formal / effective parameter substitution
  }
  gen_free_list(fp);
  return new;
}

/* if some local variables are going to be accessed inter procedurally,
 * promote them to global variables
 */
static void promote_local_entities(statement s, entity caller, statement caller_statement) {
  set re = get_referenced_entities(s);
  list le = NIL;
  SET_FOREACH(entity,e,re) {
    if(local_entity_of_module_p(e,get_current_module_entity()) &&
        !formal_parameter_p(e) ) {
      le=CONS(ENTITY,e,le);
    }
  }
  set_free(re);
  gen_sort_list(le,(gen_cmp_func_t)compare_entities);//ensure determinism
  FOREACH(ENTITY,e,le) {
    entity new = entity_undefined;
    if(entity_scalar_p(e)) {
      new= make_new_scalar_variable_with_prefix(
          entity_user_name(e),
          module_entity_to_compilation_unit_entity(get_current_module_entity()),
          copy_basic(entity_basic(e))
          );
      
    }
    else {
      pips_assert("not scalars -> array\n",entity_array_p(e));
      new= make_new_array_variable_with_prefix(
          entity_user_name(e),
          module_entity_to_compilation_unit_entity(get_current_module_entity()),
          copy_basic(entity_basic(e)),
          gen_full_copy_list(variable_dimensions(type_variable(entity_type(e))))
          );
    }
    if(!value_unknown_p(entity_initial(e)))
      statement_split_initializations(get_current_module_statement());
    RemoveLocalEntityFromDeclarations(e,get_current_module_entity(),get_current_module_statement());
    AddEntityToModuleCompilationUnit(new,get_current_module_entity());
    AddEntityToModuleCompilationUnit(new,module_entity_to_compilation_unit_entity(get_current_module_entity()));
    replace_entity(get_current_module_statement(),e,new);
  }
}

static void do_delay_communications_interprocedurally(call ca, context *c) {
  if(same_entity_p(call_function(ca),get_current_module_entity())) {
    statement parent = (statement) gen_get_ancestor(statement_domain,ca);
    FOREACH(STATEMENT,s,c->stats) {
      promote_local_entities(s,c->caller,c->caller_statement);
      statement new = translate_arguments(ca,s);
      /* there should be checks there ! */
      insert_statement(parent,new,c->backward);
    }
  }
}


/* transform each caller into a load / call /store sequence */
static void delay_communications_intraprocedurally(statement module_stat, context *c) {
    FOREACH(STATEMENT,s,c->stats)
        insert_statement(module_stat,s,c->backward);
    gen_free_list(c->stats);c->stats=NIL;
}
static void delay_communications_interprocedurally(context *c) {
    list callers = callees_callees((callees)db_get_memory_resource(DBR_CALLERS,module_local_name(get_current_module_entity()), true));
    if(ENDP(callers)) {
        pips_user_warning("no caller for function `%s', falling back to intra-procedural delaying\n",get_current_module_name());
        delay_communications_intraprocedurally(get_current_module_statement(),c);
    }
    else {
        list callers_statement = callers_to_statements(callers);

        for(list citer=callers,siter=callers_statement;!ENDP(citer);POP(citer),POP(siter)) {
            c->caller=module_name_to_entity( STRING(CAR(citer)) );
            c->caller_statement = STATEMENT(CAR(siter));
            gen_context_recurse(c->caller_statement,c,call_domain,gen_true,do_delay_communications_interprocedurally);
            clean_up_sequences(c->caller_statement);
        }

        for(list citer=callers,siter=callers_statement;!ENDP(citer);POP(citer),POP(siter)) {
            string caller_name = STRING(CAR(citer));
            statement caller_statement = STATEMENT(CAR(siter));
            module_reorder(caller_statement);
            DB_PUT_MEMORY_RESOURCE(DBR_CODE, caller_name,caller_statement);
            DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, caller_name,compute_callees(caller_statement));
        }
    }
}


static bool __delay_communications_patch_properties;
static void delay_communications_init() {
    pips_assert("reset called",graph_undefined_p(dependence_graph));
    __delay_communications_patch_properties = delay_communications_interprocedurally_p &&
        ENDP(callees_callees((callees)db_get_memory_resource(DBR_CALLERS,module_local_name(get_current_module_entity()), true))); 
    if(__delay_communications_patch_properties)
        delay_communications_interprocedurally_p=false;
}
static void delay_communications_reset() {
    pips_assert("init called",!graph_undefined_p(dependence_graph));
    if(__delay_communications_patch_properties)
        delay_communications_interprocedurally_p=true;
    dependence_graph=graph_undefined;

}

/* This phase looks for load or save statements that can be
 * put out of the loop body and move these statements, if possible.
 */
bool delay_load_communications(char * module_name)
{
    /* Get the code of the module. */
    entity module = module_name_to_entity(module_name);
    statement module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, true);
    set_ordering_to_statement(module_stat);
    set_current_module_entity( module);
    set_current_module_statement( module_stat);
    set_proper_rw_effects(
            (statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true)
    );
    remove_preferences(get_proper_rw_effects());
    set_cumulated_rw_effects(
            (statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true)
    );


    debug_on("DELAY_COMMUNICATIONS_DEBUG_LEVEL");
    delay_communications_init();
    dependence_graph = (graph) db_get_memory_resource(DBR_DG, module_name, true);

    /* Go through all the statements */
    context c = { true, NIL, true, false };

    /* then a backward translation */
    delay_communications_statement(module_stat,&c,NULL);

    /* propagate inter procedurally , except if we have no caller*/
    if(delay_communications_interprocedurally_p)
        delay_communications_interprocedurally(&c);
    else
        delay_communications_intraprocedurally(module_stat,&c);

    if(c.need_flatten)
        statement_flatten_declarations(module,module_stat);

    /* clean badly generated sequences */
    clean_up_sequences(module_stat);

    delay_communications_reset();


    pips_assert("Statement is consistent\n" , statement_consistent_p(module_stat));

    module_reorder(module_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(module_stat));

    debug_off();

    reset_current_module_entity();
    reset_ordering_to_statement();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_proper_rw_effects();

    return c.result;
}
bool delay_load_communications_inter(char * module_name) {
  delay_communications_interprocedurally_p = true;
  return delay_load_communications(module_name);
}
bool delay_load_communications_intra(char * module_name) {
  delay_communications_interprocedurally_p = false;
  return delay_load_communications(module_name);
}

bool delay_store_communications(char * module_name)
{
    /* Get the code of the module. */
    entity module = module_name_to_entity(module_name);
    statement module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, true);
    set_ordering_to_statement(module_stat);
    set_current_module_entity( module);
    set_current_module_statement( module_stat);
    set_proper_rw_effects(
            (statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true)
    );
    remove_preferences(get_proper_rw_effects());
    set_cumulated_rw_effects(
            (statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true)
    );
    delay_communications_init();

    dependence_graph = (graph) db_get_memory_resource(DBR_DG, module_name, true);

    debug_on("DELAY_COMMUNICATIONS_DEBUG_LEVEL");



    /* Go through all the statements */
    context c = { true, NIL, false, false };

    /* a first forward translation */
    delay_communications_statement(module_stat,&c,NULL);

    /* propagate inter procedurally , except if we have no caller*/
    if(delay_communications_interprocedurally_p)
        delay_communications_interprocedurally(&c);
    else
        delay_communications_intraprocedurally(module_stat,&c);

    if(c.need_flatten)
        statement_flatten_declarations(module,module_stat);

    /* clean badly generated sequences */
    clean_up_sequences(module_stat);

    delay_communications_reset();

    pips_assert("Statement is consistent\n" , statement_consistent_p(module_stat));

    module_reorder(module_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(module_stat));

    debug_off();

    reset_current_module_entity();
    reset_ordering_to_statement();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_proper_rw_effects();

    return c.result;
}
bool delay_store_communications_inter(char * module_name) {
  delay_communications_interprocedurally_p = true;
  return delay_store_communications(module_name);
}
bool delay_store_communications_intra(char * module_name) {
  delay_communications_interprocedurally_p = false;
  return delay_store_communications(module_name);
}

static bool dmas_invert_p(statement s0, statement s1) {
    pips_assert("called on dmas",simd_dma_stat_p(s0) && simd_dma_stat_p(s1));
    list p_reg = load_cumulated_rw_effects_list(s0);
    list s_reg = load_cumulated_rw_effects_list(s1);
    list pr=NIL,pw=NIL,sr=NIL,sw=NIL;
    FOREACH(REGION,r,p_reg) {
        if(array_reference_p(region_any_reference(r))) {
            if(region_write_p(r)) {
                pw=CONS(REGION,r,pw);
            }
            else {
                pr=CONS(REGION,r,pr);
            }
        }
    }
    FOREACH(REGION,r,s_reg) {
        if(array_reference_p(region_any_reference(r))) {
            if(region_write_p(r)) {
                sw=CONS(REGION,r,sw);
            }
            else {
                sr=CONS(REGION,r,sr);
            }
        }
    }
    /* this occurs when structures are involved */
    if( ENDP(sw) ||
            ENDP(sr) ||
            ENDP(pw) ||
            ENDP(pr)) {
        pips_user_warning("regions for dma not properly computed\n");
        return false;
    }
    else {
        /* dma annihilates */
        bool annihilate = gen_length(pw) == gen_length(sr) &&
            gen_length(pr) == gen_length(sw) ;
        if( annihilate) {
            for(list i=pw,j=sr;!ENDP(i)&&!ENDP(j);POP(i),POP(j)) {
                region ri = REGION(CAR(i)),
                       rj = REGION(CAR(j));
                if(!( annihilate = (reference_equal_p(region_any_reference(ri),region_any_reference(rj)) &&
                                sc_equal_p(region_system(ri),region_system(rj)) )
                    ) ) break;

            }
            if(annihilate) {
                for(list i=sw,j=pr;!ENDP(i)&&!ENDP(j);POP(i),POP(j)) {
                    region ri = REGION(CAR(i)),
                           rj = REGION(CAR(j));
                    if(!( annihilate = (reference_equal_p(region_any_reference(ri),region_any_reference(rj)) &&
                                    sc_equal_p(region_system(ri),region_system(rj)) )
                        ) ) break;

                }
            }
        }
        return annihilate;
    }
}

static void do_remove_redundant_communications_in_sequence(sequence s, bool *need_flatten){
    list ts = gen_copy_seq(sequence_statements(s));
    statement prev = statement_undefined;
    FOREACH(STATEMENT,st,ts) {
        if(simd_dma_stat_p(st)) {
            if(!statement_undefined_p(prev)) { // do they annihilate each other ?
                if(dmas_invert_p(st,prev)) {
                /* remove the second statements from the original sequence
                 * if we have load ; store , the store is useless
                 * and the load will be removed by a used_def_elim phase
                 * if we have store ; load, it 's the same
                 */
                //    gen_remove_once(&sequence_statements(s),prev);
                    gen_remove_once(&sequence_statements(s),st);
                }
                else
                    prev=st;
                prev=statement_undefined;
            }
            else
                prev=st;
        }
        else
            prev=statement_undefined;
    }
    gen_free_list(ts);
}

static void select_independent_dmas(list * stats, statement parent) {
    list rtmp = NIL;
    FOREACH(STATEMENT,st,*stats)
        if(simd_dma_stat_p(st)) {
            bool conflict = false;
            FOREACH(STATEMENT,st2,rtmp) {
                if((conflict=statements_conflict_p(st,st2)))
                    break;
            }
            conflict|=statements_conflict_p(st,parent);
            if(conflict) break;
            rtmp=CONS(STATEMENT,st,rtmp);
        }
        else
            break;
    gen_free_list(*stats);
    *stats=rtmp;
}

static bool do_remove_redundant_communications_in_anyloop(statement parent, statement body) {
    bool need_flatten=false;
    if(statement_block_p(body)) {
        list b = statement_block(body);
        list iordered=b;
        /* skip declarations */
        while(!ENDP(iordered) &&
                declaration_statement_p(STATEMENT(CAR(iordered))) ) POP(iordered);
        bool skipped_declarations= (iordered != b);
        list reversed = gen_nreverse(gen_copy_seq(iordered));
        /* look for heading dma statements */
        iordered=gen_copy_seq(iordered); // to be consistent with `reversed' allocation
        select_independent_dmas(&iordered, parent);
        /* look for trailing dma statements */
        select_independent_dmas(&reversed, parent);

        bool did_something=false;
        for(list oter=iordered;!ENDP(oter);POP(oter)) {
            statement os = STATEMENT(CAR(oter));
            FOREACH(STATEMENT,rs,reversed) {
                if(simd_dma_stat_p(os) &&
                        simd_dma_stat_p(rs) &&
                        dmas_invert_p(os,rs)) {
                    gen_remove_once(&b,rs);
                    gen_remove_once(&b,os);
                    /* insert them around the loop */
                    insert_statement(parent,os,true);
                    insert_statement(parent,rs,false);
                    /* flatten_code if needed */
                    set re = get_referenced_entities(os);
                    set de = set_make(set_pointer);
                    set_assign_list(de,statement_declarations(body));
                    if(set_intersection_p(de,re)) need_flatten = true;
                    set_free(de);set_free(re);
                    /* and remove them from the alternate list */
                    gen_remove_once(&b,rs);
                    gen_remove_once(&reversed,os);
                    did_something=true;
                    break;
                }
            }
        }
        gen_free_list(iordered);
        gen_free_list(reversed);
        if(did_something && skipped_declarations)
            need_flatten=true;
        sequence_statements(statement_sequence(body)) = b;
    }
    return need_flatten;
}

static void do_remove_redundant_communications_in_loop( loop l, bool *need_flatten) {
    *need_flatten|=do_remove_redundant_communications_in_anyloop((statement)gen_get_ancestor(statement_domain,l),loop_body(l));
}

static void do_remove_redundant_communications_in_whileloop( whileloop l, bool *need_flatten) {
    *need_flatten|=do_remove_redundant_communications_in_anyloop((statement)gen_get_ancestor(statement_domain,l),whileloop_body(l));
}

static void do_remove_redundant_communications_in_forloop( forloop l, bool *need_flatten) {
    *need_flatten|=do_remove_redundant_communications_in_anyloop((statement)gen_get_ancestor(statement_domain,l),forloop_body(l));
}

static bool remove_redundant_communications(statement s) {
    bool need_flatten=false;
    gen_context_multi_recurse(s,&need_flatten,
            sequence_domain,gen_true,do_remove_redundant_communications_in_sequence,
            loop_domain,gen_true,do_remove_redundant_communications_in_loop,
            whileloop_domain,gen_true,do_remove_redundant_communications_in_whileloop,
            forloop_domain,gen_true,do_remove_redundant_communications_in_forloop,
            NULL);
    return need_flatten;
}
static bool delay_communications(const char * module_name)
{
    /* Get the code of the module. */
    entity module = module_name_to_entity(module_name);
    statement module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, true);
    set_current_module_entity( module);
    set_current_module_statement( module_stat);
    set_ordering_to_statement(module_stat);

    set_proper_rw_effects(
            (statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true)
    );
    remove_preferences(get_proper_rw_effects());
    set_cumulated_rw_effects(
            (statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, true)
    );
    dependence_graph = (graph) db_get_memory_resource(DBR_DG, module_name, true);

    debug_on("DELAY_COMMUNICATIONS_DEBUG_LEVEL");
    bool need_flatten=
        remove_redundant_communications(module_stat);

    /* clean badly generated sequences */
    clean_up_sequences(module_stat);

    if(need_flatten)
        statement_flatten_declarations(module,module_stat);


    pips_assert("Statement is consistent\n" , statement_consistent_p(module_stat));

    module_reorder(module_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
    DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(module_stat));

    debug_off();
    dependence_graph=graph_undefined;

    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_proper_rw_effects();
    reset_ordering_to_statement();

    return true;
}
bool delay_communications_inter(const char *module_name) {
  delay_communications_interprocedurally_p = true;
  return delay_communications(module_name);
}
bool delay_communications_intra(const char *module_name) {
  delay_communications_interprocedurally_p = false;
  return delay_communications(module_name);
}
