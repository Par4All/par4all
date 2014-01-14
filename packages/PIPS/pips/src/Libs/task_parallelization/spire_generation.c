#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif


#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "boolean.h"
#include <stdbool.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"
#include "accel-util.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "conversion.h"
#include "properties.h"
#include "semantics.h"
#include "transformations.h"

#include "effects-convex.h"
#include "genC.h"

#include "complexity_ri.h"
#include "dg.h"

/* Instantiation of the dependence graph: */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"
#include "ricedg.h"
#include "chains.h"
#include "task_parallelization.h"

list com_declarations_to_add;

/* 
 *return SPIRE for a cluster stage
 */
static statement cluster_stage_spire(persistant_statement_to_cluster stmt_to_cluster, graph tg, list cluster_stage, int p) {
  int i = -1;
  list list_cl = NIL;
  statement stmt_spawn = statement_undefined, stmt_finish = statement_undefined; 
  int stage_mod = gen_length(cluster_stage);
  // int minus; FI: minus is not really used...
  int Ps = p - stage_mod;// + 1;  // plus one if we use the current cluster
  bool costly_p = false;
  FOREACH(LIST, list_stmts, cluster_stage){
    FOREACH(statement, st, list_stmts){
      if(!get_bool_property("COSTLY_TASKS_ONLY") | costly_task(st)){ 
	costly_p = true;
	i = apply_persistant_statement_to_cluster(stmt_to_cluster, statement_ordering(st));
	if(Ps > 0){
	  cluster_stage_spire_generation(stmt_to_cluster, tg, st, Ps);
	}
     }
    }
    int physical_cluster = i;
    instruction ins_spawn = make_instruction_sequence(make_sequence(list_stmts));
    if(costly_p){
      // FI: minus is not used
      // minus = (NBCLUSTERS==p)?0:1;
      physical_cluster = NBCLUSTERS - p + i;// + 1 ;
      FOREACH(statement, st, list_stmts){
	update_persistant_statement_to_cluster(stmt_to_cluster, statement_ordering(st), physical_cluster);
      }
      entity k_ent = make_constant_entity(itoa(physical_cluster), is_basic_int, 4);
      stmt_spawn = make_statement(
				  entity_empty_label(),
				  STATEMENT_NUMBER_UNDEFINED,
				  STATEMENT_ORDERING_UNDEFINED,
				  empty_comments,
				  ins_spawn,
				  NIL, NULL, empty_extensions(), 
				  make_synchronization_spawn(k_ent));
    }
    else
      stmt_spawn = make_statement(
				  entity_empty_label(),
				  STATEMENT_NUMBER_UNDEFINED,
				  STATEMENT_ORDERING_UNDEFINED,
				  empty_comments,
				  ins_spawn,
				  NIL, NULL, empty_extensions(), 
				  make_synchronization_none());
    gen_consistent_p((gen_chunk*)stmt_spawn);
    gen_consistent_p((gen_chunk*) stmt_spawn);
    list_cl = CONS(STATEMENT, stmt_spawn, list_cl);
  }
  instruction ins_finish = make_instruction_sequence(make_sequence(gen_nreverse(list_cl)));
  synchronization sync = make_synchronization_none();
  (costly_p)? sync = make_synchronization_barrier() : make_synchronization_none();
  stmt_finish = make_statement(
				 entity_empty_label(),
				 STATEMENT_NUMBER_UNDEFINED,
				 STATEMENT_ORDERING_UNDEFINED,
				 empty_comments,
				 ins_finish,
				 NIL, NULL, empty_extensions(), 
				 sync);
  gen_consistent_p((gen_chunk*)stmt_finish);
  return stmt_finish; 
}

/*the main function*/
void cluster_stage_spire_generation(persistant_statement_to_cluster stmt_to_cluster, graph tg, statement stmt, int P)
{
  gen_consistent_p((gen_chunk*)stmt);
  if(!get_bool_property("COSTLY_TASKS_ONLY") | costly_task(stmt)){ 
    statement st_finish = statement_undefined;
    instruction inst = statement_instruction(stmt);
    switch(instruction_tag(inst)){
    case is_instruction_block:{
      list cluster_stages = topological_sort(stmt);
      list list_cl = NIL;
      FOREACH(LIST, cluster_stage, cluster_stages) {
	st_finish = cluster_stage_spire(stmt_to_cluster, tg, cluster_stage, P);
	gen_consistent_p((gen_chunk*)st_finish);
	list_cl = CONS(STATEMENT, st_finish, list_cl);
      }
      instruction ins_seq = make_instruction_sequence(make_sequence(gen_nreverse(list_cl)));
      statement_instruction(stmt) = ins_seq;
      FOREACH(ENTITY, decl, gen_nreverse(statement_declarations(stmt))) 
	add_declaration_statement_at_beginning(stmt, decl);
      gen_free_list(cluster_stages);
      break;
    }
    case is_instruction_test:{
      test t = instruction_test(inst);
      cluster_stage_spire_generation(stmt_to_cluster, tg, test_true(t), P);
      cluster_stage_spire_generation(stmt_to_cluster, tg, test_false(t), P);
      break;
    }
    case is_instruction_loop :{
      loop l = statement_loop(stmt);
      statement body = loop_body(l);
      cluster_stage_spire_generation(stmt_to_cluster, tg, body, P);
      break;
    }
    default:
      break;
    }
  }
  gen_consistent_p((gen_chunk*)stmt);
  return;
}


bool spire_shared_unstructured_to_structured (char * module_name)
{ 
  statement module_stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  statement module_stat_i = copy_statement(module_stat);
  set_ordering_to_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement(module_stat);

  kdg = (graph) db_get_memory_resource (DBR_SDG, module_name, true );
  persistant_statement_to_cluster  stmt_to_cluster_i = (persistant_statement_to_cluster)db_get_memory_resource(DBR_SCHEDULE, module_name, true);
  gen_consistent_p((gen_chunk*)stmt_to_cluster_i);
  stmt_to_cluster = copy_persistant_statement_to_cluster(stmt_to_cluster_i);
  NBCLUSTERS = get_int_property("BDSC_NB_CLUSTERS");
  MEMORY_SIZE = get_int_property("BDSC_MEMORY_SIZE");
  cluster_stage_spire_generation(stmt_to_cluster, kdg, module_stat, NBCLUSTERS);
  if(!statement_undefined_p(return_st))
    insert_statement(module_stat, return_st, false);
  module_reorder(module_stat);
  gen_consistent_p((gen_chunk*)module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_SHARED_SPIRE_CODE, strdup(module_name), module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat_i);

  reset_ordering_to_statement();
  gen_consistent_p((gen_chunk*)stmt_to_cluster);
  reset_current_module_statement();
  reset_current_module_entity();
  return true;
}

bool spire_distributed_unstructured_to_structured (char * module_name)
{ 
  statement module_stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  statement module_stat_i = copy_statement(module_stat);
  set_ordering_to_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement(module_stat);
  set_precondition_map((statement_mapping)db_get_memory_resource(DBR_PRECONDITIONS, module_name, true));
  set_transformer_map((statement_mapping)
		      db_get_memory_resource(DBR_TRANSFORMERS, module_name, true));
  /* The proper effect to detect the I/O operations: */
  set_proper_rw_effects((statement_effects)db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true));
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
  module_to_value_mappings(get_current_module_entity());
  set_rw_effects((statement_effects) 
		 db_get_memory_resource(DBR_REGIONS, module_name, true));
  set_in_effects((statement_effects) 
		 db_get_memory_resource(DBR_IN_REGIONS, module_name, true));
  set_out_effects((statement_effects) 
		  db_get_memory_resource(DBR_OUT_REGIONS, module_name, true));
  set_methods_for_convex_effects();
  init_convex_rw_prettyprint(module_name);
 
  kdg = (graph) db_get_memory_resource (DBR_SDG, module_name, true );
  persistant_statement_to_cluster 
  stmt_to_cluster_i = (persistant_statement_to_cluster)db_get_memory_resource(DBR_SCHEDULE, module_name, true);
  stmt_to_cluster =  copy_persistant_statement_to_cluster(stmt_to_cluster_i);

  NBCLUSTERS = get_int_property("BDSC_NB_CLUSTERS");
  MEMORY_SIZE = get_int_property("BDSC_MEMORY_SIZE");
  list entities = gen_filter_tabulated(gen_true, entity_domain);
  char *rtl_prefix = "_rtl";
  FOREACH(entity, e, entities) {
    if (strncmp(entity_local_name(e), rtl_prefix, strlen(rtl_prefix)) == 0){
      gen_clear_tabulated_element((gen_chunkp)e);
    }
  }
  com_declarations_to_add = NIL;
  cluster_stage_spire_generation(stmt_to_cluster, kdg, module_stat, NBCLUSTERS);
  communications_construction(kdg, module_stat, stmt_to_cluster, -1);
  FOREACH(entity, e, com_declarations_to_add) 
    module_stat = add_declaration_statement(module_stat, e);
  if(!statement_undefined_p(return_st))
    insert_statement(module_stat, return_st, false);
  /* Reorder the module, because new statements have been generated. */
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_DISTRIBUTED_SPIRE_CODE, strdup(module_name), module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_stat_i);
  reset_ordering_to_statement();
  reset_proper_rw_effects();
  reset_cumulated_rw_effects();
  reset_rw_effects();
  reset_in_effects();
  reset_out_effects();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_precondition_map();
  reset_transformer_map();
  generic_effects_reset_all_methods();
  free_value_mappings();
  return true;
}



