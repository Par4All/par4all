#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

// To point out problems
#define DOUNIA

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

/* 
 *return SPIRE for a cluster stage
 */
static statement cluster_stage_spire(persistant_statement_to_schedule stmt_to_schedule, graph tg, list cluster_stage, int p) {
  int i = -1;
  list list_cl = NIL;
  statement stmt_spawn = statement_undefined, stmt_finish = statement_undefined; 
  int stage_mod = gen_length(cluster_stage);
  int physical_cluster;
  // int minus; FI: minus is not really used...
  int Ps = p - stage_mod;// + 1;  // plus one if we use the current cluster
  bool costly_p = false;
  FOREACH(LIST, list_stmts, cluster_stage){
    FOREACH(statement, st, list_stmts){
      if(!get_bool_property("COSTLY_TASKS_ONLY") | costly_task(st)){ 
	costly_p = true;
	i = apply_persistant_statement_to_schedule(stmt_to_schedule, st);
	if(Ps > 0){
	  cluster_stage_spire_generation(stmt_to_schedule, tg, st/*new_s*/, Ps);
	}
     }
    }
    instruction ins_spawn = make_instruction_sequence(make_sequence(list_stmts));
    if(costly_p){
      // FI: minus is not used
      // minus = (NBCLUSTERS==p)?0:1;
      physical_cluster = NBCLUSTERS - p + i;// + 1 ;
      FOREACH(statement, st, list_stmts){
	update_persistant_statement_to_schedule(stmt_to_schedule, st, physical_cluster);
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
  return stmt_finish; 
}

/*the main function*/
void cluster_stage_spire_generation(persistant_statement_to_schedule stmt_to_schedule, graph tg, statement stmt, int P)
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
	st_finish = cluster_stage_spire(stmt_to_schedule, tg, cluster_stage, P);
	gen_consistent_p((gen_chunk*)st_finish);
	list_cl = CONS(STATEMENT, st_finish, list_cl);
      }
      instruction ins_seq = make_instruction_sequence(make_sequence(gen_nreverse(list_cl)));
      statement_instruction(stmt) = ins_seq;
      FOREACH(ENTITY, decl, statement_declarations(stmt)) 
	add_declaration_statement_at_beginning(stmt, decl);
      gen_free_list(cluster_stages);
      break;
    }
    case is_instruction_test:{
      test t = instruction_test(inst);
      cluster_stage_spire_generation(stmt_to_schedule, tg, test_true(t), P);
      cluster_stage_spire_generation(stmt_to_schedule, tg, test_false(t), P);
      break;
    }
    case is_instruction_loop :{
      loop l = statement_loop(stmt);
      statement body = loop_body(l);
      cluster_stage_spire_generation(stmt_to_schedule, tg, body, P);
      break;
    }
    default:
      break;
    }
  }
  return;
}


bool spire_unstructured_to_structured (char * module_name)
{ 
  // entity module = local_name_to_top_level_entity(module_name);
  statement module_stat = (statement) db_get_memory_resource(DBR_CODE, module_name, true);
  set_ordering_to_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement(module_stat);
 
  kdg = (graph) db_get_memory_resource (DBR_DG, module_name, true );
  persistant_statement_to_schedule stmt_to_schedule = (persistant_statement_to_schedule)db_get_memory_resource(DBR_SCHEDULE, module_name, true);

  NBCLUSTERS = get_int_property("BDSC_NB_CLUSTERS");
  MEMORY_SIZE = get_int_property("BDSC_MEMORY_SIZE");
  INSTRUMENTED_FILE = strdup(get_string_property("BDSC_INSTRUMENTED_FILE"));
  cluster_stage_spire_generation(stmt_to_schedule, kdg, module_stat, NBCLUSTERS);
  /* FI: next statement was commented out... It has to be executed by
     any pass that generates new statement to maintain the consistency
     of PIPS internal representation */
#ifndef DOUNIA
  module_reorder(module_stat);
#endif
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_SCHEDULE, module_name, stmt_to_schedule);
  /* FI: next statement was commented out... It has to be executed by
     any pass that uses ordering_to_statement to maintain the global
     consistency of PIPS */
#ifndef DOUNIA
  reset_ordering_to_statement();
#endif
  reset_current_module_statement();
  reset_current_module_entity(); 
  return true;
}
