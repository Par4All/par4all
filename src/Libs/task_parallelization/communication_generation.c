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

#define SUCCESSORS true
#define PREDECESSORS false

bool com_instruction_p(instruction i)
{
  return native_instruction_p(i, SEND_FUNCTION_NAME)
    || native_instruction_p(i, RECV_FUNCTION_NAME);
}

static int count_data(expression exp, list args)
{
  int count = 0;
  FOREACH(EXPRESSION, e, args){
    if(reference_variable(syntax_reference(expression_syntax(e))) == reference_variable(syntax_reference(expression_syntax(exp))))
       count ++;
  }
  return count;
}

static list transfer_regions(statement parent, statement child)
{
  list l_write = regions_dup(load_statement_out_regions(parent));
  list l_read = regions_dup(load_statement_in_regions(child));  
  return RegionsIntersection(regions_dup(l_write), regions_dup(l_read), w_r_combinable_p);
}

static list list_communications(list l_communications, list args_com)
{
  expression size;
  FOREACH(REGION,reg,l_communications){
    if (!region_empty_p(reg) && region_entity(reg) != entity_undefined){
      reference rr = region_any_reference(reg);
      expression exp_phi = region_reference_to_expression(rr);
      expression exp = make_entity_expression(region_entity(reg), NIL);
      if(count_data(exp, args_com) == 0){
	/*basic b = basic_of_expression(exp);
	if(basic_pointer_p(b)) 
	search for the size of the malloc instruction*/
	Ppolynome reg_footprint = region_enumerate(reg);
	if(POLYNOME_UNDEFINED_P(reg_footprint))
	  size = int_to_expression(-1);//this will print UNDEFINED_COST
	else
	  size = polynome_to_expression(reg_footprint);
	if(expression_constant_p(size)) 
	  if(expression_to_int(size) == 1)
	    args_com = CONS(EXPRESSION, exp_phi, args_com);
	  else
	    args_com = CONS(EXPRESSION, exp, args_com);
	else
	  args_com = CONS(EXPRESSION, exp, args_com);
	args_com = CONS(EXPRESSION, size, args_com);
      }
    }
  }
  return args_com;
}

static statement com_call(bool neighbor, list args_com, int k)
{
  if(gen_length(args_com)>0){
    entity new_ent = make_constant_entity(itoa(k), is_basic_int, 4);
    expression exp = make_entity_expression(new_ent, NIL);
    args_com = CONS(EXPRESSION, exp, args_com);
    string com = (neighbor) ? SEND_FUNCTION_NAME : RECV_FUNCTION_NAME;
    return make_call_statement(com,
			       args_com,
			       entity_undefined,
			       empty_comments);
  }
  return statement_undefined;
}

static list hierarchical_com( statement s, list kdg_args_com, bool neighbor, int kp)
{
  list h_sequence = NIL, com_regions = NIL, h_args_com = NIL;
  FOREACH(LIST, l, kdg_args_com){
    FOREACH(REGION, reg, l){
      com_regions = CONS(REGION, reg, com_regions);
    }
  }
  list h_regions_com = (neighbor)?regions_dup(load_statement_out_regions(s)):regions_dup(load_statement_in_regions(s));
  //list h_regions_com = (kdg_args_com = NIL) ? all_regions:RegionsEntitiesInfDifference(all_regions, com_regions,r_w_combinable_p);
  if(gen_length(h_regions_com)>0){
    statement new_s = make_statement(
				     statement_label(s),
				     STATEMENT_NUMBER_UNDEFINED,
				     STATEMENT_ORDERING_UNDEFINED,
				     statement_comments(s),
				     statement_instruction(s),
				     NIL, NULL, statement_extensions(s), statement_synchronization(s));
    h_args_com = list_communications(h_regions_com, h_args_com);
    statement com = com_call(neighbor, h_args_com, kp);
    if(!neighbor)
      h_sequence = CONS(STATEMENT,com,h_sequence);
    h_sequence = CONS(STATEMENT,new_s,h_sequence);
    if(neighbor)
      h_sequence = CONS(STATEMENT,com,h_sequence);
    if(gen_length(h_sequence) > 1){ 
      instruction ins_seq = make_instruction_sequence(make_sequence(gen_nreverse(h_sequence)));
      statement_instruction(s) = ins_seq;
      //free_extensions(statement_extensions(s));
      statement_extensions(s) = empty_extensions();
      statement_comments(s) = empty_comments;
      statement_synchronization(s) = make_synchronization_none();
    }
  }
  return gen_full_copy_list(h_args_com);
}


static list gen_send_communications(statement s, vertex tau, persistant_statement_to_cluster st_to_cluster, graph tg, int kp)
{
  int i;
  list args_send, list_st = NIL, kdg_args_com = NIL, h_args_com = NIL;
  statement new_s = make_statement(
				   statement_label(s),
				   STATEMENT_NUMBER_UNDEFINED,
				   STATEMENT_ORDERING_UNDEFINED,
				   statement_comments(s),
				   statement_instruction(s),
				   NIL, NULL, statement_extensions(s), statement_synchronization(s));
  list_st = CONS(STATEMENT, new_s, NIL);
  for(i = 0; i < NBCLUSTERS; i++){
    args_send = NIL;
    FOREACH(SUCCESSOR, su, vertex_successors(tau)) {
      vertex taus = successor_vertex(su);
      statement ss = vertex_to_statement(taus);
      if(bound_persistant_statement_to_cluster_p(st_to_cluster, statement_ordering(ss))) {
	if(apply_persistant_statement_to_cluster(st_to_cluster, statement_ordering(s))
	   != apply_persistant_statement_to_cluster(st_to_cluster, statement_ordering(ss))
	   &&
	   apply_persistant_statement_to_cluster(st_to_cluster, statement_ordering(ss)) == i) {
	  list com_regions = transfer_regions (vertex_to_statement(statement_to_vertex(s,tg)),vertex_to_statement(statement_to_vertex(ss,tg))); // transfer_regions (s,ss); 
	  kdg_args_com = CONS(LIST, com_regions, kdg_args_com);
	  args_send = list_communications(com_regions, args_send);
	}
      }
    }
    if(gen_length(args_send)>0){
      list_st = CONS(STATEMENT, com_call(SUCCESSORS, args_send, i),list_st);
    }
  }
  if(gen_length(list_st) > 1){ 
    instruction ins_seq = make_instruction_sequence(make_sequence(gen_nreverse(list_st)));
    //free_instruction(statement_instruction(parent)); 
    statement_instruction(s) = ins_seq;
    //free_extensions(statement_extensions(s));
    statement_extensions(s) = empty_extensions();
    statement_synchronization(s) = make_synchronization_none();
    statement_comments(s) = empty_comments;
  }
  if(apply_persistant_statement_to_cluster(st_to_cluster, statement_ordering(s)) != kp && (kp != -1))
    h_args_com = hierarchical_com(s, kdg_args_com, SUCCESSORS, kp);
  return h_args_com;
}

static list predecessors(statement st, graph tg)
{
  list vertices = graph_vertices(tg); 
  list preds = NIL;
  FOREACH(VERTEX, v, vertices) {
    statement parent = vertex_to_statement(v);
    FOREACH(SUCCESSOR, su, (vertex_successors(v))) {
      vertex s = successor_vertex(su);
      statement child = vertex_to_statement(s);
      if(statement_equal_p(child, st))
	preds = CONS(STATEMENT, parent, preds);
    }
  }
  return preds;
}

static list gen_recv_communications(statement sv, persistant_statement_to_cluster st_to_cluster, graph tg, int kp)
{
  int i;
  list args_recv, list_st = NIL,  kdg_args_com = NIL, h_args_com = NIL;
  statement new_s = make_statement(
				   statement_label(sv),
				   STATEMENT_NUMBER_UNDEFINED,
				   STATEMENT_ORDERING_UNDEFINED,
				   statement_comments(sv),
				   statement_instruction(sv),
				   NIL, NULL, statement_extensions(sv), statement_synchronization(sv));
  list_st = CONS(STATEMENT, new_s, NIL);
  for(i = 0;i < NBCLUSTERS; i++){
    args_recv = NIL;
    list preds = predecessors(sv, tg);
    FOREACH(STATEMENT, parent, preds){
      if(bound_persistant_statement_to_cluster_p(st_to_cluster, statement_ordering(parent))) {
	if(apply_persistant_statement_to_cluster(st_to_cluster, statement_ordering(parent)) != apply_persistant_statement_to_cluster(st_to_cluster, statement_ordering(sv)) && apply_persistant_statement_to_cluster(st_to_cluster, statement_ordering(parent)) == i){ 
	  list com_regions = transfer_regions (vertex_to_statement(statement_to_vertex(parent,tg)),vertex_to_statement(statement_to_vertex(sv,tg))); 
	  kdg_args_com = CONS(LIST, com_regions, kdg_args_com);
	  args_recv = list_communications(com_regions, args_recv);
	}
      }
    }
    if(gen_length(args_recv) > 0){
      list_st = CONS(STATEMENT, com_call(PREDECESSORS, args_recv, i), list_st);
    }
  }
  if(gen_length(list_st) > 1){
    instruction ins_seq = make_instruction_sequence(make_sequence((list_st)));//make_statement_list(new_s, st_send)));
    statement_instruction(sv) = ins_seq;
    //free_extensions(statement_extensions(sv));
    statement_extensions(sv) = empty_extensions();
    statement_synchronization(sv) = make_synchronization_none();
    statement_comments(sv)=empty_comments;
  }
  if(apply_persistant_statement_to_cluster(st_to_cluster, statement_ordering(sv)) != kp && (kp != -1))
    h_args_com = hierarchical_com(sv,kdg_args_com,PREDECESSORS,kp);
  return  h_args_com;
}


void communications_construction(graph tg, statement stmt, persistant_statement_to_cluster st_to_cluster, int kp)
{
  if(bound_persistant_statement_to_cluster_p(st_to_cluster, statement_ordering(stmt))) {
    gen_consistent_p((gen_chunk*)stmt);
    instruction inst = statement_instruction(stmt);
    switch(instruction_tag(inst)){
    case is_instruction_block:{
      list vertices = graph_vertices(tg), coms_send = NIL, coms_recv = NIL, coms_st = NIL; 
      list barrier = NIL;
      MAPL(stmt_ptr,
	   {
	     statement ss = STATEMENT(CAR(stmt_ptr ));
	     if(statement_block_p(ss)){
	       instruction sinst = statement_instruction(ss);
	       MAPL(sb_ptr,
		    {
		      statement sb = STATEMENT(CAR(sb_ptr ));
		      if(statement_block_p(sb)){
			instruction sbinst = statement_instruction(sb);
			MAPL(ss_ptr,
			     {
			       statement s = STATEMENT(CAR(ss_ptr ));
			       barrier = CONS(STATEMENT,s,barrier);
			     },
			     instruction_block(sbinst));
		      }
		    else
		      barrier = CONS(STATEMENT,sb,barrier);
		    },
		    instruction_block(sinst));
	     }
	     else
	       barrier = CONS(STATEMENT,ss,barrier);
	   },
	 instruction_block(inst));
      FOREACH(STATEMENT, s, barrier)
	{
	  bool found_p = false;
	  FOREACH(VERTEX, pre, vertices) {
	    statement this = vertex_to_statement(pre);
	    if(statement_equal_p(this,s) && bound_persistant_statement_to_cluster_p(st_to_cluster, statement_ordering(s))) {
	      found_p = true;
	      break;
	    }
	  }
	  if(found_p){
	    int ki = apply_persistant_statement_to_cluster(st_to_cluster, statement_ordering(s));
	  list args_send =  gen_recv_communications(s, st_to_cluster, tg, kp);
	  list args_recv =  gen_send_communications(s, statement_to_vertex(s,tg), st_to_cluster, tg, kp);
	  if(gen_length(args_recv) > 0 && (kp != ki) )
	    coms_recv = CONS(STATEMENT, com_call(PREDECESSORS, args_recv, ki), coms_recv);
	  if(gen_length(args_send) > 0 && (kp != ki) )
	    coms_send = CONS(STATEMENT, com_call(SUCCESSORS, args_send, ki), coms_send);
	  communications_construction(tg, s, st_to_cluster,  ki);
	  }
	  else
	    communications_construction(tg, s, st_to_cluster, kp);
	}
    if((gen_length(coms_send) > 0 || gen_length(coms_recv) > 0) && (kp != -1)){
      statement new_s = make_statement(
				       statement_label(stmt),
				       STATEMENT_NUMBER_UNDEFINED,
				       STATEMENT_ORDERING_UNDEFINED,
				       statement_comments(stmt),
				       statement_instruction(stmt),
				       NIL, NULL, statement_extensions(stmt), statement_synchronization(stmt));
      if(gen_length(coms_recv) > 0){
	FOREACH(STATEMENT, st, coms_recv){
	  coms_st = CONS(STATEMENT, st, coms_st);
	}
      }
      coms_st = CONS(STATEMENT, new_s, coms_st);
      if(gen_length(coms_send) > 0){
	FOREACH(STATEMENT, st, coms_send){
	  coms_st = CONS(STATEMENT, st, coms_st);
	}
      }
      instruction seq = make_instruction_sequence(make_sequence((coms_st)));
      statement_instruction(stmt) = seq;
      statement_extensions(stmt) = empty_extensions();
      statement_synchronization(stmt) = make_synchronization_none();
      statement_comments(stmt) = empty_comments;
    }
    break;
    }
    case is_instruction_test:{
      test t = instruction_test(inst);
      communications_construction(tg, test_true(t), st_to_cluster, kp);
      communications_construction(tg, test_false(t), st_to_cluster, kp);
    break;
    }
  case is_instruction_loop :{
    loop l = statement_loop(stmt);
    statement body = loop_body(l);
    communications_construction(tg, body, st_to_cluster, kp);
    break;
  }
    default:
    break;
    }
  }
  return;
}

