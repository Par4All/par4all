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
 

typedef struct {
  double min_tlevel;
  statement min_tau;
}min_start_time;


static bool ready_node(statement st){
  bool ready_p = true;
  list vertices = graph_vertices(kdg);
  FOREACH(VERTEX, pre, vertices) {
    statement parent = vertex_to_statement(pre);
    FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
      vertex s = successor_vertex(su);
      statement child = vertex_to_statement(s);
      if(statement_equal_p(child, st))	{
	annotation *anp = gen_array_item(annotations, (int)statement_ordering(parent));
	bool sc_p = anp->scheduled;
	if(!sc_p){
	  ready_p = false;
	  break;
	}
      }
    }
  }
  return ready_p;
}


static void update_priority_values(statement ready_st)
{
  //update the priorities of ready_st successors
  FOREACH(SUCCESSOR, su, (vertex_successors(statement_to_vertex(ready_st, kdg)))) {
    vertex s = successor_vertex(su);
    statement child = vertex_to_statement(s);
    annotation *anc = gen_array_item(annotations, (int)statement_ordering(child));
    anc->tlevel = -1;
  }
  FOREACH(SUCCESSOR, su, (vertex_successors(statement_to_vertex(ready_st, kdg)))) {
    vertex s = successor_vertex(su);
    statement child = vertex_to_statement(s);
    annotation *anc = gen_array_item(annotations, (int)statement_ordering(child));
    t_level(s, kdg, annotations);
    anc->prio = anc->tlevel + anc->blevel;
  }
  return; 
}

void allocate_task_to_cluster(statement ready_st, int cl_p, int order){
  annotation *an = gen_array_item(annotations, (int)statement_ordering(ready_st));
  list vertices = graph_vertices(kdg);
  an->scheduled = true;
  an->order_sched = order;
  an->cluster = cl_p;
  if(!bound_persistant_statement_to_cluster_p(stmt_to_cluster, statement_ordering(ready_st)))
    extend_persistant_statement_to_cluster(stmt_to_cluster, statement_ordering(ready_st), cl_p);
  else
    update_persistant_statement_to_cluster(stmt_to_cluster, statement_ordering(ready_st), cl_p);
  //delete ready_st from successors(tasks(cl_p))
  FOREACH(VERTEX, pre, vertices) {
    statement parent = vertex_to_statement(pre);
    annotation *anp = gen_array_item(annotations, (int)statement_ordering(parent));
    if(anp->cluster == cl_p)
      { 
	FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	  vertex v = successor_vertex(su);
	  statement child = vertex_to_statement(v);
	  if(statement_equal_p(child, ready_st)){
	    double *ec = (double *)malloc(sizeof(double));
	    *ec = 0;
	    gen_array_addto(anp->edge_cost,(int)statement_ordering(ready_st), ec);
	    gen_array_addto(annotations, (int)statement_ordering(parent), anp); 
	  }
	}
      }
  }

  cluster *cl = gen_array_item(clusters, cl_p);
  double time = cl->time;
  double time1 = (time > an->tlevel) ? (time + an->task_time) : (an->tlevel + an->task_time);
  cl->time = time1;
  if(gen_length(vertex_successors(statement_to_vertex(ready_st,kdg)))> 0){
    list l_data = RegionsMustUnion(regions_dup(an->data), regions_dup(cl->data), r_w_combinable_p);
    cl->data = l_data;
  }
  gen_array_addto(clusters, cl_p, cl); 
  update_priority_values(ready_st);
  return;
}

static void move_task_to_cluster(statement ready_st, int cl_p){
  annotation *an = gen_array_item(annotations, (int)statement_ordering(ready_st));
  //int cl_i = an->cluster;
  cluster *cl = gen_array_item(clusters, an->cluster);
  double time = cl->time;
  cl->time = time -  an->task_time;
  gen_array_addto(clusters, an->cluster, cl); 
  an->cluster = cl_p;
  extend_persistant_statement_to_cluster(stmt_to_cluster, statement_ordering(ready_st), cl_p);
  cl = gen_array_item(clusters, cl_p);
  time = cl->time;
  cl->time = time > an->tlevel + an->task_time? time : an->tlevel + an->task_time;
  if(gen_length(vertex_successors(statement_to_vertex(ready_st,kdg)))> 0){
    list l_data = RegionsMustUnion(regions_dup(an->data), regions_dup(cl->data), r_w_combinable_p);
    cl->data = l_data;
  }
  gen_array_addto(clusters, cl_p, cl); 
  update_priority_values(ready_st);
  return;
}

static bool MCW(statement ready_st, int cl, int M)
{
  if(gen_length(vertex_successors(statement_to_vertex(ready_st,kdg)))==0)
    return true;
  if(cl != -1){
    annotation *an = gen_array_item(annotations, (int)statement_ordering(ready_st));
    cluster *cl_s = gen_array_item(clusters, cl);
    list l_data = RegionsMustUnion(regions_dup(an->data), regions_dup(cl_s->data), w_w_combinable_p);
    return (size_of_regions(l_data) < M);
  }
  return false;
}
static min_start_time tlevel_decrease(statement ready_st, int M){
  list vertices = graph_vertices(kdg);
  annotation *an = gen_array_item(annotations, (int)statement_ordering(ready_st));
  min_start_time min_pred; 
  double  time;
  statement min_predecessor = statement_undefined;
  double  min_tlevel = an->tlevel; 
  FOREACH(VERTEX, pre, vertices) {
    statement parent = vertex_to_statement(pre);
    annotation *anp = gen_array_item(annotations, (int)statement_ordering(parent));
    if(anp->scheduled && (MEMORY_SIZE == -1 || MCW(ready_st,anp->cluster,M)))
      {
	FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	  vertex s = successor_vertex(su);
	  statement child = vertex_to_statement(s);
	  if(statement_equal_p(child, ready_st)) //pre is predecessor of ready_st
	    {
	      cluster *cl_s =  gen_array_item(clusters, anp->cluster);
	      double new_tlevel = cl_s->time;
	      FOREACH(VERTEX, pre1, vertices) {// all predecessors minus this parent
		statement parent1 = vertex_to_statement(pre1);
		annotation *anp1 = gen_array_item(annotations, (int)statement_ordering(parent1));
		if(anp1->scheduled){
		  FOREACH(SUCCESSOR, su1, (vertex_successors(pre1))) {
		    vertex s1 = successor_vertex(su1);
		    statement child1 = vertex_to_statement(s1);
		    if(statement_equal_p(child1, ready_st))//pre1 is predecessor of ready_st
		      {
			if (!statement_equal_p(parent, parent1)){
			  double edge_c = *(double *)(gen_array_item(anp1->edge_cost,statement_ordering(ready_st)));
			  time = anp1->tlevel + anp1->task_time + edge_c /*edge_cost(parent1,ready_st);*/  ;
			  new_tlevel = new_tlevel > time ? new_tlevel : time;
			}
		      }
		    }
		}
	      }
	      if(new_tlevel <= min_tlevel){
		min_tlevel = new_tlevel;
		min_predecessor = parent;
	      }
	    }
	}
      }
  }
  min_pred.min_tlevel = min_tlevel;
  min_pred.min_tau = min_predecessor;
  return min_pred;
}
//only when targetting distributed memory systems 
static bool zeroing_multiple_edges(statement ready_st, int order, int M)
{
  list vertices = graph_vertices(kdg);
  annotation *an = gen_array_item(annotations, (int)statement_ordering(ready_st));
  double time;
  int i, j, min_cluster = -1;
  bool zeroing_p = false;
  typedef struct {
    statement predecessor;
    double time;
  }zeroing;
  zeroing *zeroing_can; 
  gen_array_t sorted_predecessors = gen_array_make(0);
  double min_tlevel = an->tlevel; 
  FOREACH(VERTEX, pre, vertices) {
    statement parent = vertex_to_statement(pre);
    annotation *anp = gen_array_item(annotations, (int)statement_ordering(parent));
    if(anp->cluster != -1)
      {
	FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	  vertex s = successor_vertex(su);
	  statement child = vertex_to_statement(s);
	  if(statement_equal_p(child, ready_st)) //pre is predecessor of ready_st
	    {
	      double edge_c = *(double *)(gen_array_item(anp->edge_cost,statement_ordering(ready_st)));
	      cluster  *cl = gen_array_item(clusters, anp->cluster);
	      double new_tlevel = cl->time;
	      time = anp->tlevel + anp->task_time + edge_c; 
	      new_tlevel = new_tlevel > time ? new_tlevel : time;
	      if(new_tlevel <= min_tlevel + edge_c){
		int array_size = gen_array_nitems(sorted_predecessors);
		zeroing *zer_new = (zeroing *)malloc(sizeof(zeroing));
		zer_new->predecessor = parent;
		zer_new->time = new_tlevel;
		i=0;
		if(array_size > 0){
		  zeroing_can = gen_array_item(sorted_predecessors, i);
		  while(i< array_size)
		    {
		      zeroing_can = gen_array_item(sorted_predecessors, i);
		      if(new_tlevel <= zeroing_can->time)
			break;
		      i++;
		    }
		  if(new_tlevel <= zeroing_can->time) //shift
		    {
		      zeroing *zer = gen_array_item(sorted_predecessors, i+1);
		      gen_array_addto(sorted_predecessors, i+1, zeroing_can);
		      gen_array_addto(sorted_predecessors, i, zer_new);
		      for(j = i+1; j< array_size+1; j++)
			{
			  zeroing_can = gen_array_item(sorted_predecessors, j+1);
			  gen_array_addto(sorted_predecessors, j+1, zer);
			  zer = zeroing_can;
			}
		    }
		}
		if(i == array_size)
		  gen_array_addto(sorted_predecessors, i, zer_new);
	      }
	    }
	}
      }
  }
  i = 0;
  while(i<(int)gen_array_nitems(sorted_predecessors)){
    zeroing *zer= gen_array_item(sorted_predecessors, i);
    statement min_predecessor = copy_statement(zer->predecessor); 
    annotation *anp = gen_array_item(annotations, (int)statement_ordering(min_predecessor));
    min_cluster = anp->cluster;
    if(MEMORY_SIZE == -1 || MCW(ready_st,min_cluster,M))
      {
	zeroing_p = true;
	allocate_task_to_cluster(ready_st, min_cluster,order);
	break;
      }
    i++;
  }
  i= i+1;
  double sum = 0;
  while(i<(int)gen_array_nitems(sorted_predecessors))
    {
      zeroing *zer= gen_array_item(sorted_predecessors, i);
      statement parent = zer->predecessor; 
      annotation *anp = gen_array_item(annotations, (int)statement_ordering(parent));
      double edge_c = *(double *)(gen_array_item(anp->edge_cost,statement_ordering(ready_st)));
      sum = sum + anp->task_time;
      if(sum <= edge_c && (int)(gen_length(vertex_successors(statement_to_vertex(parent,kdg))) == 1)
	 && (MEMORY_SIZE == -1 || MCW(parent,min_cluster,M)))
	{
	  move_task_to_cluster(parent,min_cluster);
	}
      else
	break;
      i++;
    }
  gen_array_free( sorted_predecessors);
  return zeroing_p;
}
/*apply the second priority : load balancing : length_cluster(c_i) <
 *tlevel(ready_st) and forall node n_y in c_i
 *scheduled(successors(n_y)) = true or successors(n_y) \subset successors(ready_st) 
 */
static int end_idle_clusters(statement ready_st, int nbclusters){
  list vertices = graph_vertices(kdg);
  annotation *an = gen_array_item(annotations, (int)statement_ordering(ready_st));
  double min_time = an->tlevel, time_cluster;
  bool found_p;
  int min_cluster = -1;
  vertex ready_vertex = statement_to_vertex(ready_st, kdg);
  for(int cl = 0; cl < nbclusters; cl++){
    cluster *cl_s =  gen_array_item(clusters, cl); ;
    time_cluster = cl_s->time;
    if(time_cluster <= min_time)
      {
	found_p = true;
	FOREACH(VERTEX, pre, vertices) {
	  statement parent = vertex_to_statement(pre);
	  annotation *anp = gen_array_item(annotations, (int)statement_ordering(parent));
	  if(anp->cluster == cl){	       
	    FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	      vertex v = successor_vertex(su);
	      statement child = vertex_to_statement(v);
	      annotation *anc = gen_array_item(annotations, (int)statement_ordering(child));
	      if(!anc->scheduled)
		{
		  FOREACH(SUCCESSOR, su_r, vertex_successors(ready_vertex)) {
		    if(!statement_equal_p(vertex_to_statement(successor_vertex(su_r)), child)) 
		      {
			found_p = false;
			break;
		      }
		  }
		}
	    }
	  }
	}
	if(found_p)
	  {
	    min_time = min_time > time_cluster ? time_cluster : min_time;
	    min_cluster = (min_time > time_cluster || min_cluster == -1) ? cl : min_cluster;
	  }
      }
  }
  return min_cluster;
}

/*apply the third priority if bounded number of processors is exceeded
 *start-time(ready_st) = min(length_cluster(c_i))
 *tlevel(ready_st) can be increased.
 */
static int min_start_time_cluster(int nbclusters){
  double min = -1;
  int min_cluster = -1, cl;
  for(cl = 0; cl < nbclusters; cl++)
    {
      cluster *cl_s = gen_array_item(clusters, cl);
      double time = cl_s->time;
      if(time <= min || min == -1)
       {
	 min_cluster = cl;
	 min = time;
       }
    }
  return min_cluster;
}

/*used to compute the parallel task time of a task*/
static double max_start_time_cluster(int nbclusters){
  double max = -1;
  int cl;
  for(cl = 0; cl < nbclusters; cl++)
    {
      cluster *cl_s = gen_array_item(clusters, cl);
      double time = cl_s->time;
      if(time >= max || max == -1)
       {
	 max = time;
       }
    }
  return max;
}
static bool DSRW(statement ready_st, statement unready_st, int order, int M){
  int cl_p; bool DSRW_p = false;
  min_start_time r_min_pred_s = tlevel_decrease(ready_st, M);
  statement r_min_pred = r_min_pred_s.min_tau;
  if(r_min_pred != statement_undefined)
    {
      min_start_time u_min_pred_s = tlevel_decrease(unready_st,M);
      double ptlevel_before = u_min_pred_s.min_tlevel; 
      annotation *anp = gen_array_item(annotations, (int)statement_ordering(r_min_pred));
      cl_p = anp->cluster;
      u_min_pred_s = tlevel_decrease(unready_st,M);
      double ptlevel_after = u_min_pred_s.min_tlevel; 
      if(ptlevel_after > ptlevel_before){
	DSRW_p = true;
      }
      else
	allocate_task_to_cluster(ready_st,cl_p, order);
    }
  if ((r_min_pred == statement_undefined) || DSRW_p)    
    return true;
  else 
    return false;
}

static statement select_task_with_highest_priority(list tasks, statement ready){
  double max;
  statement highest_task = statement_undefined;
  if(!statement_undefined_p(ready)){
    annotation *item = gen_array_item(annotations, statement_ordering(ready));
    max = item->prio;
  }
  else
    max = -1;
  FOREACH(statement, st, tasks) {
    annotation *item = gen_array_item(annotations, statement_ordering(st));
    if(item->prio > max)//just for unexamined nodes
      {
	max = item->prio;
	highest_task = st;
      }
  }
  return highest_task;
}

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

static void update_parallel_task(int ordering, int nbclusters)
{
  annotation *item = gen_array_item(annotations,ordering);
  item->data = NIL;
  item->task_time = max_start_time_cluster(nbclusters);
  return;
}

static int find_cluster(statement ready_task, int nbclusters, int P, int M, int order, list stmts, gen_array_t annotations_s )
{
  //second priority: load balancing
  int cl_p = end_idle_clusters(ready_task, nbclusters);
  if(cl_p == -1 ||  (MEMORY_SIZE != -1 && !MCW(ready_task,cl_p,M)))
    { //third priority: PCW
      if(nbclusters < P && (MEMORY_SIZE == -1 || MCW(ready_task,nbclusters,M)))
	cl_p = nbclusters ++;
      else
	{
	  cl_p = min_start_time_cluster(nbclusters);
	  if(MEMORY_SIZE != -1 && !MCW(ready_task,cl_p,M))
	    {
	      cancel_schedule(annotations_s, stmts);
	      fprintf (stderr, "OUUPS, Not Enough Memory but let's try the hierarchical enclosed tasks\n");
	      return -1;
	    }
	}
    }
  allocate_task_to_cluster(ready_task,cl_p, order);
  return nbclusters;
}


int BDSC(sequence seq, int P, int M, int ordering) {
  if(P <= 0)
    return 0;
  int nbclusters = 0;
  int order = 0, cl_p = -1;
  list unscheduled_tasks = NIL;
  list ready_tasks = NIL, unready_tasks = NIL;
  list stmts = sequence_statements(seq);
  statement ready_task = statement_undefined, unready_task = statement_undefined;
  gen_array_t annotations_s = schedule_failsafe();
  bool other_rules_p = false; 
  initialization_clusters(false);
  top_level(kdg, annotations);
  bottom_level(kdg, annotations);
  priorities(annotations);
  FOREACH(statement, st, stmts){
    if(!declaration_statement_p(st)) {
      unscheduled_tasks = CONS(STATEMENT, st, unscheduled_tasks);
      if(ready_node(st))
	ready_tasks = CONS(STATEMENT, st, ready_tasks);
      else
	unready_tasks = CONS(STATEMENT, st, unready_tasks);
    }
  }
  while(gen_length(unscheduled_tasks) > 0 ){
    ready_task = select_task_with_highest_priority(ready_tasks, statement_undefined);
    unready_task = select_task_with_highest_priority(unready_tasks, ready_task);
    other_rules_p = false; 
    if(statement_undefined_p(unready_task)){
      cl_p = -1;
      statement min_pred = ready_task;
      bool zeroing_p = true;
      /*if(get_bool_property("BDSC_DISTRIBUTED_MEMORY")) 
	zeroing_p = zeroing_multiple_edges(ready_task, order,M);
	//function Not validated
	else*/
	{
	  min_start_time min_pred_s =  tlevel_decrease(ready_task,M);
	  
	  min_pred =  min_pred_s.min_tau; 
	  if(min_pred != statement_undefined)
	    {
	      annotation *anp = gen_array_item(annotations, (int)statement_ordering(min_pred));
	      cl_p = anp->cluster;
	      allocate_task_to_cluster(ready_task,cl_p, order);
	    }
	}
      if(min_pred == statement_undefined || !zeroing_p)
	other_rules_p = true;
    }
    else {
      other_rules_p = DSRW(ready_task, unready_task,order,M);
    }
    if(other_rules_p)
      nbclusters = find_cluster(ready_task, nbclusters, P, M, order, stmts, annotations_s);
    if(nbclusters == -1)//not enough memorry
      return -1;
    gen_remove_once(&unscheduled_tasks, ready_task);
    FOREACH(statement, st, unready_tasks){
      if(ready_node(st))
	gen_remove_once(&unready_tasks, st);
    }
    ready_tasks = NIL; 
    FOREACH(statement, st, unscheduled_tasks) {
      if(ready_node(st))
	ready_tasks = CONS(STATEMENT, st, ready_tasks);
    }
    order ++;
  }
  gen_array_free(annotations_s);
  update_parallel_task(ordering, nbclusters);
  return nbclusters;
}

int DSC(sequence seq, int ordering) {
  int M = -1;
  int nbclusters = 0;
  int order = 0, cl_p = -1;
  list unscheduled_tasks = NIL;
  list ready_tasks = NIL, unready_tasks = NIL;
  list stmts = sequence_statements(seq);
  statement ready_task = statement_undefined, unready_task = statement_undefined;
  gen_array_t annotations_s = schedule_failsafe();
  bool other_rules_p = false; 
  top_level(kdg, annotations);
  bottom_level(kdg, annotations);
  priorities(annotations);
  FOREACH(statement, st, stmts){
    unscheduled_tasks = CONS(STATEMENT, st, unscheduled_tasks);
    if(ready_node(st))
      ready_tasks = CONS(STATEMENT, st, ready_tasks);
    else
      unready_tasks = CONS(STATEMENT, st, unready_tasks);
  }
  while(gen_length(unscheduled_tasks) > 0 ){
    ready_task = select_task_with_highest_priority(ready_tasks, statement_undefined);
    unready_task = select_task_with_highest_priority(unready_tasks, ready_task);
    other_rules_p = false; 
    if(statement_undefined_p(unready_task)){
      cl_p = -1;
      statement min_pred = ready_task;
      bool zeroing_p = true;
      if(get_bool_property("BDSC_DISTRIBUTED_MEMORY")) 
	zeroing_p = zeroing_multiple_edges(ready_task, order,M);
      else
	{
	  min_start_time min_pred_s =  tlevel_decrease(ready_task,M);
	  min_pred =  min_pred_s.min_tau; 
	  if(min_pred != statement_undefined)
	    {
	      annotation *anp = gen_array_item(annotations, (int)statement_ordering(min_pred));
	      cl_p = anp->cluster;
		allocate_task_to_cluster(ready_task,cl_p, order);
	    }
	}
      if(min_pred == statement_undefined || !zeroing_p)
	other_rules_p = true;
    }
    else {
      other_rules_p = DSRW(ready_task, unready_task,order,M);
    }
    if(other_rules_p){
      cluster *cl_s = (cluster *)malloc(sizeof(cluster));
      cl_s->time = 0;
      cl_s->data = NIL;
      int i = nbclusters++;
      gen_array_addto(clusters, i, cl_s);
      allocate_task_to_cluster(ready_task, i, order);
    }
    gen_remove_once(&unscheduled_tasks, ready_task);
    FOREACH(statement, st, unready_tasks){
      if(ready_node(st))
	gen_remove_once(&unready_tasks, st);
    }
    ready_tasks = NIL; 
    FOREACH(statement, st, unscheduled_tasks) {
      if(ready_node(st))
	ready_tasks = CONS(STATEMENT, st, ready_tasks);
    }
    order ++;
  }
  gen_array_free(annotations_s);
  update_parallel_task(ordering, nbclusters);
  return nbclusters;
}


