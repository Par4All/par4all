#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif


#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "boolean.h"
#include <stdbool.h>
#include <limits.h>

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
#include "complexity.h"
#include "dg.h"

/* Instantiation of the dependence graph: */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"
#include "ricedg.h"
#include "chains.h"
#include "task_parallelization.h"

/* Global variables */
 
graph kdg, ddg;
statement return_st = statement_undefined;
#define MAX_ITER 1

persistant_statement_to_cluster stmt_to_cluster;// = persistant_statement_to_cluster_undefined;


static gen_array_t schedule_failsafe(){
  gen_array_t annotations_s = gen_array_make(0);
  FOREACH(VERTEX, v,  graph_vertices(kdg)){
    statement s = vertex_to_statement(v);
    annotation *item = (annotation *)malloc(sizeof(annotation)); 
    annotation *vs = gen_array_item(annotations,(int)statement_ordering(s));
    item->scheduled = vs->scheduled;
    item->cluster = vs->cluster;
    item->edge_cost = gen_array_make(gen_length(vertex_successors(v)));
    FOREACH(SUCCESSOR, su, (vertex_successors(v))){
      vertex s = successor_vertex(su);
      statement child = vertex_to_statement(s);
      gen_array_addto(item->edge_cost, statement_ordering(child), gen_array_item(vs->edge_cost,statement_ordering(child)));
    }
    gen_array_addto(annotations_s, (int)statement_ordering(s), item); 
  }
  return annotations_s;
}


static void cancel_schedule(gen_array_t annotations_s, list stmts)
{
  FOREACH(STATEMENT, s,  stmts){
    annotation *vs = gen_array_item(annotations_s,(int)statement_ordering(s));
    annotation *item = gen_array_item(annotations,(int)statement_ordering(s));
    item->scheduled = vs->scheduled;
    item->cluster = vs->cluster;
    if(bound_persistant_statement_to_cluster_p(stmt_to_cluster, statement_ordering(s)))
      update_persistant_statement_to_cluster(stmt_to_cluster, statement_ordering(s), item->cluster);
    item->edge_cost = vs->edge_cost;
    gen_array_addto(annotations, (int)statement_ordering(s), item);
  }
  return;
}
static void cancel_schedule_stmt(gen_array_t annotations_s, statement st)
{
  instruction inst = statement_instruction(st);
  switch(instruction_tag(inst)){
  case is_instruction_block:
    cancel_schedule(annotations_s,sequence_statements(statement_sequence(st)));
    break;
  case is_instruction_loop:
    cancel_schedule_stmt(annotations_s, loop_body(statement_loop(st)));
    break;
  default:
    break;
  }
  return; 
}
static double critical_path_length(int nbclusters){
  double max = -1;
  int cl;
  for(cl = 0; cl < nbclusters; cl++)
    {
      cluster *cl_s = gen_array_item(clusters, cl);
      double time = cl_s->time;
      if(time > max) 
	max = time;
    }
  return max;
}

/*reset to zero for each new sequence to handle*/
static void initialization_clusters(bool first_p)
{
  int i;
  cluster *cl_s;
  for(i=0;i<NBCLUSTERS;i++){
    if(first_p)
      cl_s = (cluster *)malloc(sizeof(cluster));
    else
      cl_s = gen_array_item(clusters, i);
    cl_s->time = 0;
    cl_s->data = NIL;
    gen_array_addto(clusters, i, cl_s);
  }
  return; 
}






/*
 *rebuild cluster_stages by forming a list of list of statements
 */
static list rebuild_topological_sort(list stages){
  list K = NIL, cluster_stages = NIL;
  int i;
  FOREACH(LIST, stage, stages) {
    K=NIL;
    for(i = 0; i < NBCLUSTERS; i++){
      list list_stmts = NIL;
      FOREACH(STATEMENT, st, stage) {
	if(!declaration_statement_p(st) && !return_statement_p(st)) {
	  if(bound_persistant_statement_to_cluster_p(stmt_to_cluster,statement_ordering(st))){
	    if(apply_persistant_statement_to_cluster(stmt_to_cluster, statement_ordering(st)) == i)
	      list_stmts = CONS(STATEMENT, st, list_stmts);
	  }
	}
	/* we suppose that we have only one return statement at the
	  end of the module :( */
	if(return_statement_p(st))
	  return_st = st;
      }
      if(list_stmts != NIL)
	K = CONS(LIST, list_stmts, K);
    }
    if(K == NIL){
      list list_stmts = NIL;
      FOREACH(STATEMENT, st, stage) {
	if(!declaration_statement_p(st) && !return_statement_p(st)) {
	  if(bound_persistant_statement_to_cluster_p(stmt_to_cluster,statement_ordering(st))){
	    if(apply_persistant_statement_to_cluster(stmt_to_cluster, statement_ordering(st)) == -1) 
	      list_stmts = CONS(STATEMENT, st, list_stmts);
	  }
	  else
	    list_stmts = CONS(STATEMENT, st, list_stmts);
	}
	if(return_statement_p(st))
	  return_st = st;
      }
      if(list_stmts != NIL){
	K =  CONS(LIST, list_stmts, K);
      }
    }
    if(K != NIL)
      cluster_stages = CONS(LIST, gen_nreverse(K), cluster_stages);
  }
  return cluster_stages;
}

list topological_sort(statement stmt)
{
  list K = NIL; uint i;
  list cluster_stages = NIL;
  list vertices = graph_vertices(kdg);
  gen_array_t I = gen_array_make(0);
  for(i = 0;i<gen_length(vertices);i++){
      gen_array_addto(I, i, 0);
  }
  FOREACH(statement, st, sequence_statements(statement_sequence(stmt))){ 
    FOREACH(VERTEX, v, vertices) {
      statement parent = vertex_to_statement(v);
      if(statement_equal_p(st,parent)){
	FOREACH(SUCCESSOR, su, (vertex_successors(v))) {
	  vertex w = successor_vertex(su);
	  statement child = vertex_to_statement(w);
	  gen_array_addto(I, (int)statement_ordering(child), gen_array_item(I, (int)statement_ordering(child)) + 1);
	}
      }
    }
  }
  FOREACH(statement, st, sequence_statements(statement_sequence(stmt))){
    if(gen_array_item(I, statement_ordering(st)) == 0)
      K = CONS(STATEMENT, st, K);
  }
  list L =  gen_copy_seq(K);
  cluster_stages = CONS(LIST, L, cluster_stages);
  bool insert_p = false;
  while(K != NIL){
    statement st = STATEMENT(CAR(gen_last(K)));
    gen_remove_once(&K, st);
    FOREACH(VERTEX, pre, vertices) { 
      statement parent = vertex_to_statement(pre);
      if(statement_equal_p(parent, st)){
	  FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	    vertex v = successor_vertex(su);
	    statement child = vertex_to_statement(v);
	    gen_array_addto(I, (int)statement_ordering(child), gen_array_item(I, (int)statement_ordering(child)) - 1);
	    if(gen_array_item(I, statement_ordering(child)) == 0 && gen_occurences(child, K) == 0) {
	      K = CONS(STATEMENT, child, K);
	      insert_p = true;
	    }
	  }
	  break;
      }
    }
    list M = NIL;
    bool ins_p = true,found_p=false;
    FOREACH(STATEMENT, stmt, K){
      found_p = false;
      FOREACH(STATEMENT, stmtl, L){
	if(statement_equal_p(stmt,stmtl))
	  found_p = true;
      }
      if(!found_p)
	M = CONS(STATEMENT,stmt,M);
      else{
	ins_p = false;
	found_p = false;
      }
    }
    if( ins_p && gen_length(M) > 0 && insert_p){
      cluster_stages = CONS(LIST, M, cluster_stages);
      insert_p = false;
      L =  gen_copy_seq(K);
    }
  }
  gen_array_free(I);
  return rebuild_topological_sort(cluster_stages);
}


static double transfer_cost(statement s, int nbclusters){
  if(nbclusters == 0 || !get_bool_property("BDSC_DISTRIBUTED_MEMORY"))
    return 0;
  list l_in = regions_dup(load_statement_in_regions(s));
  list l_out = regions_dup(load_statement_out_regions(s));
  double size = size_of_regions(l_in) + size_of_regions(l_out);
  return size;
}


static void hierarchical_schedule_step(statement stmt, int P, int M, bool dsc_p)
{
  list cluster_stages  = topological_sort(stmt);
  int nbclustersL, nbclusters, nbclusters_s;
  double task_time, task_time_s;
  FOREACH(LIST, cluster_stage, cluster_stages) {
    int stage_mod = gen_length(cluster_stage);
    int Ps = P - stage_mod;// + 1;//use current processor
    FOREACH(LIST, L, cluster_stage){
      nbclustersL = 0;
      FOREACH(statement, st, L){
	if(Ps <= 0){
	  pips_user_warning("NBCLUSTERS is not sufficient to handle nested parts in statement_ordering = %d\n", statement_ordering(st));
	}
	else{
	if(!get_bool_property("COSTLY_TASKS_ONLY") | costly_task(st)){
	  annotation *item = gen_array_item(annotations,(int)statement_ordering(st));
	  nbclusters = item->nbclusters;
	  task_time = item->task_time;
	  gen_array_t annotations_s = schedule_failsafe();
	  nbclusters_s = hierarchical_schedule(st, (item->cluster == -1)?0:item->cluster, Ps, M, dsc_p);
	  item = gen_array_item(annotations,(int)statement_ordering(st));
	  task_time_s = item->task_time;
	  if(nbclusters_s < nbclusters || task_time + transfer_cost(st, nbclusters) < task_time_s + transfer_cost(st, nbclusters_s)){
	    item->nbclusters = nbclusters;
	    cancel_schedule_stmt(annotations_s,st);
	  }
	  nbclustersL = nbclustersL > nbclusters? nbclustersL : nbclusters; 
	  gen_array_free(annotations_s);
	}
	}
      }
      Ps = Ps - nbclustersL;
    }
  }
  return;
}
int hierarchical_schedule(statement stmt, int k, int P, int M, bool dsc_p)
{
  int nbclusters = 0, nbcl; double cpl;
  if(!get_bool_property("COSTLY_TASKS_ONLY") | costly_task(stmt)){ //granularity management
    instruction inst = statement_instruction(stmt);
    annotation *item = gen_array_item(annotations,(int)statement_ordering(stmt));
    switch(instruction_tag(inst))
      {
      case is_instruction_block:{
	sequence seq = statement_sequence(stmt);
	if(gen_length(sequence_statements(seq))>0){
	  gen_array_t annotations_i = schedule_failsafe(), annotations_s;
	  if(!dsc_p)
	    nbcl = BDSC(seq, P, M, statement_ordering(stmt));
	  else{
	    nbcl = DSC(seq, statement_ordering(stmt));
	    NBCLUSTERS = NBCLUSTERS > nbcl ? NBCLUSTERS:nbcl;
	  }
	  double cpl1 = critical_path_length(nbcl);
	  int iter = 0;
	  
	  do{
	    hierarchical_schedule_step(stmt, P, M, dsc_p);
	    nbclusters = nbcl;
	    cpl = cpl1;
	    annotations_s = schedule_failsafe();
	    cancel_schedule(annotations_i, sequence_statements(seq));
	    if(!dsc_p){
	      nbcl = BDSC(seq, P, M, statement_ordering(stmt));
	    }
	    else{
	      nbcl = DSC(seq, statement_ordering(stmt));
	      NBCLUSTERS = NBCLUSTERS > nbcl ? NBCLUSTERS:nbcl;
	    }
	    cpl1 = critical_path_length(nbcl); 
	    if(nbcl == 0){ 
	      fprintf (stderr, "Unable to schedule with NBCLUSTERS  equal to %d\n",P);
	      exit (EXIT_FAILURE);
	    }
	    if(nbcl == -1){ 
	      fprintf (stderr, "Unable to schedule with MEMORY SIZE  equal to %d\n",M);
	      exit (EXIT_FAILURE);
	    }
	    iter ++;
	  } while(cpl1 < cpl && nbcl <= nbclusters && iter <= MAX_ITER);
	  cancel_schedule(annotations_s, sequence_statements(seq));
	  item->task_time = cpl;
	  item->data = NIL;
	  gen_array_free(annotations_i);
	  gen_array_free(annotations_s);
	}
	break;
      }
      case is_instruction_test : {
	test t = instruction_test(inst);
	int nbt = hierarchical_schedule(test_true(t), k, P, M, dsc_p);
	int nbf = hierarchical_schedule(test_false(t), k, P, M, dsc_p);
	nbclusters = nbt > nbf? nbt: nbf; 
	break;
      }
      case is_instruction_loop :{
	loop l = statement_loop(stmt);
	statement body = loop_body(l);
	nbclusters = hierarchical_schedule(body, k, P, M, dsc_p);
	break;
      }
      case is_instruction_forloop :{
	forloop l = statement_forloop(stmt);
	statement body = forloop_body(l);
	nbclusters = hierarchical_schedule(body, k, P, M, dsc_p);
	break;
      }
      default:
	break;
      }
    item->nbclusters = nbclusters;
    allocate_task_to_cluster(stmt, k, 0);
  }
  return nbclusters;
}

/*The main function for performing the hierarchical scheduling
 *(scheduled SDG) using BDSC and generating the graph (unstructured)
 *  BDSC-based top-down hierarchical scheduling
*/
bool hbdsc_parallelization(char * module_name)
{ 
  statement	module_stat;
  string tg_name = NULL;
  FILE *ftg;
  if (!get_bool_property("COMPLEXITY_EARLY_EVALUATION")) 
    set_bool_property("COMPLEXITY_EARLY_EVALUATION", true);
  module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, true);
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
  set_in_effects((statement_effects) db_get_memory_resource(DBR_IN_REGIONS, module_name, true));
  set_out_effects((statement_effects) db_get_memory_resource(DBR_OUT_REGIONS, module_name, true));
  set_methods_for_convex_effects();
  init_convex_rw_prettyprint(module_name);
  kdg = (graph) db_get_memory_resource (DBR_SDG, module_name, true );
  /*Complexities (task processing time)*/
  set_complexity_map( (statement_mapping) db_get_memory_resource(DBR_COMPLEXITIES, module_name, true));
  /*Properties to set the parameters of BDSC*/
  NBCLUSTERS = get_int_property("BDSC_NB_CLUSTERS");
  MEMORY_SIZE = get_int_property("BDSC_MEMORY_SIZE");
  INSTRUMENTED_FILE = strdup(get_string_property("BDSC_INSTRUMENTED_FILE"));

  /*cost model generation */
  annotations = gen_array_make(0);
  clusters = gen_array_make(0);
  initialization_clusters(true);
  if(sizeof(INSTRUMENTED_FILE) == 8)
    initialization(kdg, annotations);
  else
    parse_instrumented_file(INSTRUMENTED_FILE, kdg, annotations);
  stmt_to_cluster = make_persistant_statement_to_cluster();
  hierarchical_schedule(module_stat, 0, NBCLUSTERS, MEMORY_SIZE, false);
  tg_name = strdup(concatenate(db_get_current_workspace_directory(),
			       "/",module_name,"/",module_name, "_scheduled_sdg.dot", NULL));
  ftg = safe_fopen(tg_name, "w");
  fprintf( ftg, "digraph {\n compound=true;ratio=fill; node[fontsize=24,fontname=\"Courier\",labelloc=\"t\"];nodesep=.05;\n" );
  print_SDGs(module_stat, kdg, ftg, annotations);
  fprintf( ftg, "\n}\n" );
  safe_fclose(ftg, tg_name);
  free(tg_name);
  
  DB_PUT_MEMORY_RESOURCE(DBR_SDG, module_name, (char*) kdg);
  gen_consistent_p((gen_chunk*)stmt_to_cluster);
  DB_PUT_MEMORY_RESOURCE(DBR_SCHEDULE, module_name, (char*) stmt_to_cluster);
  reset_proper_rw_effects();
  reset_cumulated_rw_effects();
  reset_ordering_to_statement();
  reset_rw_effects();
  reset_in_effects();
  reset_out_effects();
  reset_precondition_map();
  reset_complexity_map();
  reset_transformer_map();
  reset_current_module_statement();
  reset_current_module_entity();
  gen_array_full_free(annotations);
  gen_array_full_free(clusters);
  generic_effects_reset_all_methods();
  free_value_mappings();
  return true;
}



/*The main function for performing the hierarchical scheduling
 *(scheduled SDG) using DSC and generating SPIRE
*/
bool dsc_code_parallelization(char * module_name)
{ 
  statement	module_stat;
  if (!get_bool_property("COMPLEXITY_EARLY_EVALUATION")) 
    set_bool_property("COMPLEXITY_EARLY_EVALUATION", true);
  module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, true);
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
  set_in_effects((statement_effects) db_get_memory_resource(DBR_IN_REGIONS, module_name, true));
  set_out_effects((statement_effects) db_get_memory_resource(DBR_OUT_REGIONS, module_name, true));
  set_methods_for_convex_effects();
  init_convex_rw_prettyprint(module_name);
 
  kdg = (graph) db_get_memory_resource (DBR_DG, module_name, true );
  
  /*Complexities (task processing time)*/
  set_complexity_map( (statement_mapping) db_get_memory_resource(DBR_COMPLEXITIES, module_name, true));

  /*Properties to set the parameters of BDSC*/
  NBCLUSTERS = 1;//INT_MAX;
  MEMORY_SIZE = -1;
  INSTRUMENTED_FILE = strdup(get_string_property("BDSC_INSTRUMENTED_FILE"));
  /*cost model generation */
  annotations = gen_array_make(0);
  clusters = gen_array_make(0);
  if(sizeof(INSTRUMENTED_FILE) == 8)
    initialization(kdg, annotations);
  else
    parse_instrumented_file(INSTRUMENTED_FILE, kdg, annotations);

  stmt_to_cluster = make_persistant_statement_to_cluster();
  /*DSC-based top-down hierarchical scheduling*/
  hierarchical_schedule(module_stat, 0, NBCLUSTERS, MEMORY_SIZE, true);
  /* Reorder the module, because new statements have been generated. */
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
  reset_ordering_to_statement();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_proper_rw_effects();
  reset_cumulated_rw_effects();
  reset_rw_effects();
  reset_in_effects();
  reset_out_effects();
  reset_precondition_map();
  reset_complexity_map();
  reset_transformer_map();
  gen_array_free(annotations);
  gen_array_free(clusters);
  generic_effects_reset_all_methods();
  free_value_mappings();
  return true;
}

