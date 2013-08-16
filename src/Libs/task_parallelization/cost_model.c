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
#include "c_syntax.h"
#include "conversion.h"
#include "properties.h"
#include "transformations.h"

#include "effects-convex.h"
#include "genC.h"

#include "complexity_ri.h"
#include "complexity.h"
#include <time.h> 
#include "dg.h"

/* Instantiation of the dependence graph: */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"
#include "ricedg.h"
#include "chains.h"
#include "task_parallelization.h"


bool costly_task(statement st){
  bool costly_p = false;
  instruction inst = statement_instruction(st);
  if (statement_contains_user_call_p(st))
    if(user_call_p(statement_call(st)))
       return true;
  else{
    switch(instruction_tag(inst)){
    case is_instruction_block:{
      MAPL( stmt_ptr,
	    {
	      statement local_stmt = STATEMENT(CAR(stmt_ptr ));
	      costly_p = costly_p || costly_task(local_stmt);
   	    },
	    instruction_block(inst));
      return costly_p;
      break;
    }
    case is_instruction_test :{
      test t = instruction_test(inst);
      return costly_task(test_true(t)) ||
	costly_task(test_false(t));
      break;
    }
    case is_instruction_loop :{
      return true;
    }
    case is_instruction_forloop :{
      return true;
    }
    case is_instruction_whileloop :{
      return true;
    }
    default:
      return false;
    }
  }
}

double polynomial_to_numerical (Ppolynome poly_amount)
{
   double size = 0.f;
   /* if polynomial is not a constant monomial, we use    
    * an heuristic to map it into a numerical constant 
    * take the higher degree monomial
    * and decide upon its coefficient
    */
   if(!POLYNOME_UNDEFINED_P(poly_amount))
     {
       int max_degree = polynome_max_degree(poly_amount);
       for(Ppolynome p = poly_amount; !POLYNOME_NUL_P(p); p = polynome_succ(p)) {
	 int curr_degree =  (int)vect_sum(monome_term(polynome_monome(p)));
	 if(curr_degree == max_degree) {
	   size = monome_coeff(polynome_monome(p));
	   break;
	 }
       }
     }
   return size;
}

double size_of_regions(list l_data)
{
  Ppolynome transfer_time = POLYNOME_NUL;
  FOREACH(REGION,reg,l_data){
    Ppolynome reg_footprint= region_enumerate(reg); 
    reg_footprint = polynome_mult(reg_footprint, expression_to_polynome(int_to_expression(SizeOfElements(variable_basic(type_variable(entity_type(region_entity(reg))))))));
    polynome_add(&transfer_time, reg_footprint);
  }   
  return polynomial_to_numerical(transfer_time); 
}

static list used_data(statement(st)){
  list l_data = NIL;
  list l_read = regions_dup(regions_read_regions(load_statement_local_regions(st)));  
  list l_write = regions_dup(regions_write_regions(load_statement_local_regions(st)));
  l_data =  RegionsMustUnion(regions_dup(l_read), regions_dup(l_write), r_w_combinable_p);
  return l_data;
}

static double task_time(statement s)
{
  complexity stat_comp = load_statement_complexity(s);
  Ppolynome poly = complexity_polynome(stat_comp);
  return polynomial_to_numerical(poly);
}

/*
  Per-byte transfer time(tb)is the time to transmit one byte along a
  data communication channel; the duration of this transmission
  is defined by the communication channel bandwidth
*/

double edge_cost(statement s1, statement s2)
{
  Ppolynome transfer_time = POLYNOME_NUL;
  list l_write = regions_dup(regions_write_regions(load_statement_local_regions(s1)));
  list l_read = regions_dup(regions_read_regions(load_statement_local_regions(s2)));  
  list l_communications = RegionsIntersection(regions_dup(l_write), regions_dup(l_read), w_r_combinable_p);
  FOREACH(REGION,reg,l_communications){
    Ppolynome reg_footprint = region_enumerate(reg); 
    //reg_footprint =  polynome_mult(reg_footprint, expression_to_polynome(int_to_expression(SizeOfElements(variable_basic(type_variable(entity_type(region_entity(reg))))))));
    //reg_footprint =
    //polynome_mult(reg_footprint,expression_to_polynome(int_to_expression(2.5)));//\beta= 2.5 on Cmmcluster
    polynome_add(&transfer_time, reg_footprint);
  }
  //polynome_add(&transfer_time, latency);// the latency \alpha = 15000 on Cmmcluster
  //polynome_fprint(stderr,transfer_time,entity_local_name,default_is_inferior_var);
  return polynomial_to_numerical(transfer_time);
}


/* First parameter is the top level (earliest start time) for each node
*/
double t_level(vertex v, graph dg, gen_array_t annotations)
{
  double max = 0, level;
  list vertices = (graph_vertices(dg));
  annotation *an,*anp;
  statement sv = vertex_to_statement(v);
  FOREACH(VERTEX, pre, vertices) {
    statement parent = vertex_to_statement(pre);
      FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	vertex s = successor_vertex(su);
	statement child = vertex_to_statement(s);
	if(statement_equal_p(child, sv) && !statement_equal_p(child,parent) && gen_array_item(annotations, (int)statement_ordering(parent))){	    
	  double tl_p;
	  anp = gen_array_item(annotations, (int)statement_ordering(parent));
	  double edge_c = *(double *)/*(intptr_t)*/(gen_array_item(anp->edge_cost, statement_ordering(sv)));
	  if(anp->tlevel != -1)
	    tl_p = anp->tlevel;
	  else
	    tl_p =  t_level(pre, dg, annotations);
	  level = tl_p + anp->task_time + edge_c ;
	  if(level > max)
	    max = level;
	}
	else {
	  if(statement_equal_p(child, sv) && !statement_equal_p(child,parent) && gen_array_item(annotations, (int)statement_ordering(parent))==NULL )
	    {
	      anp = gen_array_item(annotations, (int)statement_ordering(parent));
	      double edge_c = *(double *)(gen_array_item(anp->edge_cost,statement_ordering(sv)));
	      level = t_level(pre, dg, annotations) + anp->task_time + edge_c;
	      if(level > max)
		max = level;
	    }
	}
      }
  }
  if((int)statement_ordering(sv) < (int) gen_array_size(annotations) && gen_array_item(annotations, (int)statement_ordering(sv))!=NULL)
    an = gen_array_item(annotations, statement_ordering(sv));
  else
    an = (annotation *)malloc(sizeof(annotation));
  an->tlevel = max;
  gen_array_addto(annotations, (int)statement_ordering(sv), an);
  return max; 
}
void top_level(graph dg, gen_array_t annotations)
{
  list vertices = graph_vertices(dg);
  FOREACH(VERTEX, v, vertices){ 
    t_level(v, dg, annotations);
  }
  return; 
}

/* Second parameter is the bottom level (latest start time) for each
   node
*/
void bottom_level( graph dg, gen_array_t annotations)
{
  double max, level;
  list vertices = (graph_vertices(dg));
  annotation *an, *anp;
  FOREACH(VERTEX, v, vertices) {
    max = 0;
    statement sv = vertex_to_statement(v);
    FOREACH(VERTEX, pre, (graph_vertices(dg))) {
      statement parent = vertex_to_statement(pre);
      anp = gen_array_item(annotations, (int)statement_ordering(parent));
      FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	vertex s = successor_vertex(su);
	statement child = vertex_to_statement(s);
	if(statement_equal_p(child, sv) && gen_array_item(annotations, (int)statement_ordering(parent)) )
	  {
	    double edge_c = *(double *)(gen_array_item(anp->edge_cost,statement_ordering(sv)));
	    level = anp->blevel + edge_c;
	    if(level > max)
	      max = level;
	  }
      }
    }
    an = gen_array_item(annotations, statement_ordering(sv));
    an->blevel = an->task_time + max;
    gen_array_addto(annotations, (int)statement_ordering(sv), an);
  }
  return;
}

void priorities(gen_array_t annotations)
{
  size_t i;
  for(i = 0; i< gen_array_nitems(annotations); i++){
    if(gen_array_item(annotations, i) != NULL){
      annotation *item = gen_array_item(annotations, i);
      item->prio = item->tlevel + item->blevel;
      gen_array_addto(annotations, i, item);
    }
  }
  return;
}


void initialization(graph dg, gen_array_t annotations)
{
  list vertices = graph_vertices(dg);
  int sigma =  get_int_property("BDSC_SENSITIVITY");
  float clock = 1, overhead = 0;
  srand(time(NULL));
  FOREACH(VERTEX, v, vertices){
    statement stmt = vertex_to_statement(v);
    annotation *item = (annotation *)malloc(sizeof(annotation));
    float x = (float)((float)rand()/RAND_MAX * (float)sigma/100);
    item->task_time = clock * task_time(stmt) * (1 + x ) + overhead;
    item->edge_cost = gen_array_make(gen_length(vertex_successors(v)));//0
    FOREACH(SUCCESSOR, su, (vertex_successors(v))){//statement_to_vertex(stmt, dg)))) {
      vertex s = successor_vertex(su);
      statement child = vertex_to_statement(s);
      double *ec = (double *)malloc(sizeof(double));
      *ec = clock * edge_cost(stmt, child) * (1+((float)rand()/RAND_MAX * sigma/100)) + overhead;
      gen_array_addto(item->edge_cost, statement_ordering(child), ec);
    }
    item->scheduled = false;
    item->order_sched = -1;
    item->cluster = -1;
    item->nbclusters = 0;
    item->data =  used_data(stmt);
    gen_array_addto(annotations, (int)statement_ordering(stmt), item); 
  }
  return;
}

void parse_instrumented_file(char *file_name, graph dg, gen_array_t annotations)
{
  FILE *finstrumented;
  double cost;
  int ordering, ordering2;
  finstrumented = fopen(file_name,"r");
  list vertices = graph_vertices(dg);
  annotation *item;
  FOREACH(VERTEX, v, vertices){
    statement stmt = vertex_to_statement(v);
    item = (annotation *)malloc(sizeof(annotation));
    item->edge_cost = gen_array_make(0);
    FOREACH(SUCCESSOR, su, (vertex_successors(statement_to_vertex(stmt, dg)))) {
      vertex s = successor_vertex(su);
      statement child = vertex_to_statement(s);
      double *ec = (double *)malloc(sizeof(double));
      *ec = 0;
      gen_array_addto(item->edge_cost, statement_ordering(child), ec);
    }
    gen_array_addto(annotations, (int)statement_ordering(stmt), item);
  }
  while (!feof(finstrumented) && (fscanf(finstrumented,"%d->%d = %lf \n", &ordering,&ordering2, &cost)))
    {
      if(ordering2 == -1){
	item = gen_array_item(annotations, ordering);
	item->task_time = (double)cost;
	if(item->edge_cost == NULL)
	  item->edge_cost = gen_array_make(0);
	gen_array_addto(annotations, ordering, item); 
      }
      else { 
	item = gen_array_item(annotations, ordering);
	if(item->edge_cost == NULL)
	  item->edge_cost = gen_array_make(0);
	double *ec = (double *)malloc(sizeof(double));
	*ec = cost;
	gen_array_addto(item->edge_cost, ordering2, ec);
	gen_array_addto(annotations, ordering, item);
      }
     
    } 	 
  fclose(finstrumented);  
  return;
}
 
