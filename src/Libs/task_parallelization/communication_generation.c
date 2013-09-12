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

#include "c_syntax.h"
#include "syntax.h"
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
#include "regions_to_loops.h"
#include "task_parallelization.h"

#define SUCCESSORS true
#define PREDECESSORS false

statement make_com_loopbody(entity v, bool neighbor, list vl, int k) {
  entity new_ent = make_constant_entity(itoa(k), is_basic_int, 4);
  expression exp = make_entity_expression(new_ent, NIL);
  expression e = reference_to_expression(make_reference(v, gen_full_copy_list(vl)));
  list args_com = CONS(EXPRESSION, e, CONS(EXPRESSION, exp, NIL));
  string com = (neighbor) ? SEND_FUNCTION_NAME : RECV_FUNCTION_NAME;
  statement s = make_call_statement(com,
				    gen_nreverse(args_com),
				    entity_undefined,
				    empty_comments);
  pips_assert("com body is not properly generated", statement_consistent_p(s));
  return s;
}

static statement Psysteme_to_loop_nest(entity v,list vl, Pbase b, Psysteme p, bool neighbor, list l_var, int k) {
  Psysteme condition, enumeration;
  statement body = make_com_loopbody(v, neighbor, l_var, k);;
  algorithm_row_echelon_generic(p, b, &condition, &enumeration, true);
  statement s = systeme_to_loop_nest(enumeration, vl, body, entity_intrinsic(DIVIDE_OPERATOR_NAME));
  pips_assert("s is not properly generated (systeme_to_loop_nest)", statement_consistent_p(s));
  return s;
}

/* Returns the entity corresponding to the global name */
static entity global_name_to_entity( const char* package, const char* name ) {
  return gen_find_tabulated(concatenate(package, MODULE_SEP_STRING, name, NULL), entity_domain);
}


statement region_to_com_nest (region r, bool isRead, int k) {
  reference ref = effect_any_reference(r);
  entity v = reference_variable(ref);
  type t = entity_type(v);
  statement s = statement_undefined; 
  if (type_variable_p(t)) {
    Psysteme p = region_system(r);
    Pbase base = BASE_NULLE;
    // Build the base
    FOREACH(expression, e, reference_indices(ref)) {
      entity phi = reference_variable(syntax_reference(expression_syntax(e)));
      base = base_add_variable(base, (Variable)phi);
    }
    s = Psysteme_to_loop_nest(v, base_to_list(base), base, p, isRead, reference_indices(ref), k);
  }
  else {
    pips_internal_error("unexpected type \n");
  }
  pips_assert("s is properly generated", statement_consistent_p(s));
  return s;
  
}


/* This function is in charge of replacing the PHI entity of the region by generated indices.
   PHI values has no correspondance in the code. Therefore we have to create actual indices and
   replace them in the region in order for the rest to be build using the right entities.
*/
static void replace_indices_region_com(region r, list* dadd, int indNum, entity module) {
  Psysteme ps = region_system(r);
  reference ref = effect_any_reference(r);
  list ref_indices = reference_indices(ref);
  list l_var = base_to_list(sc_base(ps));
  list l_var_new = NIL;
  list li = NIL;
  // Default name given to indices
  char* s = "_rtl";
  char s2[128];
  int indIntern = 0;
  list l_var_temp = gen_nreverse(gen_copy_seq(l_var));
  bool modified = false;
  // The objective here is to explore the indices and the variable list we got from the base in order to compare and
  // treat only the relevant cases
  FOREACH(entity, e, l_var_temp) {
    if (!ENDP(ref_indices)) {
      FOREACH(expression, exp, ref_indices) {
	entity phi = reference_variable(syntax_reference(expression_syntax(exp)));
	if (!strcmp(entity_name(phi), entity_name(e))) {
	  // If the names match, we generate a new name for the variable
	  sprintf(s2, "%s:%s_%d_%d", module_local_name(module),s, indNum, indIntern);
	  indIntern++;
	  // We make a copy of the entity with a new name
	  entity ec = make_entity_copy_with_new_name(e, s2, false);
	  // However the new variable still has a rom type of storage, therefore we create a new ram object
	  entity dynamic_area = global_name_to_entity(module_local_name(module), DYNAMIC_AREA_LOCAL_NAME);
	  ram r =  make_ram(module, dynamic_area, CurrentOffsetOfArea(dynamic_area, e), NIL);
	  entity_storage(ec) = make_storage_ram(r);
	  s2[0] = '\0';
	  // We build the list we are going to use to rename the variables of our system
	  l_var_new = CONS(ENTITY, ec, l_var_new);
	  // We build the list which will replace the list of indices of the region's reference
	  li = CONS(EXPRESSION, entity_to_expression(ec), li);
	  // We build the list which will be used to build the declaration statement 
	  *dadd = CONS(ENTITY, ec, *dadd);
	  modified = true;
	}
      }
      if (!modified) {
	gen_remove_once(&l_var, e);
      }
    }
    modified = false;
  }
  pips_assert("different length \n", gen_length(l_var) == gen_length(l_var_new));
  // Renaming the variables of the system and replacing the indice list of the region's reference
  ps = sc_list_variables_rename(ps, l_var, l_var_new);
  reference_indices(ref) = gen_nreverse(gen_full_copy_list(li));
  pips_assert("region is not consistent", region_consistent_p(r));
}


static statement com_call(bool neighbor, list args_com, int k)
{
  list declarations = NIL;
  list sl = NIL;
  int indNum = 0;
  statement s_com = make_continue_statement(entity_empty_label());
  if(gen_length(args_com)>0){
    FOREACH(effect, reg, args_com){
      entity e = region_entity(reg);
      if(!io_entity_p(e) && !stdin_entity_p(e)){
	list phi = NIL;
	replace_indices_region_com(reg, &phi, indNum, get_current_module_entity());
	statement s = region_to_com_nest(reg, neighbor, k);
	sl = CONS(STATEMENT, s, sl);
	indNum++;
	declarations = gen_nconc(declarations, phi);
      }
    }
    com_declarations_to_add = gen_nconc(com_declarations_to_add, declarations);
    if(gen_length(sl) > 0)
      s_com =  make_block_statement(sl);
  }
  return s_com;
}

static list transfer_regions(statement parent, statement child)
{
  list l_write = regions_dup(load_statement_out_regions(parent));
  list l_read = regions_dup(load_statement_in_regions(child));  
  return RegionsIntersection(regions_dup(l_write), regions_dup(l_read), w_r_combinable_p);
}


static list hierarchical_com(statement s, bool neighbor, int kp)
{
  list h_sequence = NIL;
  list h_regions_com = (neighbor)?regions_dup(load_statement_out_regions(s)):regions_dup(load_statement_in_regions(s));
  if(gen_length(h_regions_com)>0){
    statement new_s = make_statement(
				     statement_label(s),
				     STATEMENT_NUMBER_UNDEFINED,
				     STATEMENT_ORDERING_UNDEFINED,
				     statement_comments(s),
				     statement_instruction(s),
				     NIL, NULL, statement_extensions(s), statement_synchronization(s));
    statement com = com_call(neighbor, h_regions_com, kp);
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
  return h_regions_com;
}


static list gen_send_communications(statement s, vertex tau, persistant_statement_to_cluster st_to_cluster, graph tg, int kp)
{
  int i;
  list args_send, list_st = NIL, h_args_com = NIL;
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
	  list com_regions = transfer_regions (vertex_to_statement(statement_to_vertex(s,tg)),vertex_to_statement(statement_to_vertex(ss,tg))); 
	  if(gen_length(com_regions)>0)
	    list_st = CONS(STATEMENT, com_call(SUCCESSORS, com_regions, i),list_st);
	  break;
	}
      }
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
    h_args_com = hierarchical_com(s, SUCCESSORS, kp);
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
  list args_recv, list_st = NIL,  h_args_com = NIL;
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
	  list_st = CONS(STATEMENT, com_call(PREDECESSORS, com_regions, i), list_st);
	  break;
	}
      }
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
    h_args_com = hierarchical_com(sv, PREDECESSORS, kp);
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

