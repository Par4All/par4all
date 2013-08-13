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
#include "semantics.h"
#include "task_parallelization.h"

static transformer p_transformer;
static graph sdg;



static
void get_private_entities_walker(loop l, set s)
{
  set_append_list(s,loop_locals(l));
}

static
set get_private_entities(void *s)
{
    set tmp = set_make(set_pointer);
    gen_context_recurse(s,tmp,loop_domain,gen_true,get_private_entities_walker);
    return tmp;
}


struct cpv {
    entity e;
    bool rm;
};


static
void check_private_variables_call_walker(call c,struct cpv * p)
{
    set s = get_referenced_entities(c);
    if(set_belong_p(s,p->e)){
        p->rm=true;
        gen_recurse_stop(0);
    }
    set_free(s);
}
static
bool check_private_variables_loop_walker(loop l, struct cpv * p)
{
    return !has_entity_with_same_name(p->e,loop_locals(l));
}


static list private_variables(statement stat)
{
    set s = get_private_entities(stat);
    list l =NIL;
    SET_FOREACH(entity,e,s) {
        struct cpv p = { .e=e, .rm=false };
        gen_context_multi_recurse(stat,&p,
                call_domain,gen_true,check_private_variables_call_walker,
                loop_domain,check_private_variables_loop_walker,gen_null,
                0);
        if(!p.rm)
            l=CONS(ENTITY,e,l);
    }
    set_free(s);

    return l;
}


bool statement_equal_p(statement s1, statement s2)
{
    return (statement_number(s1) == statement_number(s2) && statement_ordering(s1) == statement_ordering(s2) );
}

 
/* *************SDG: Sequence Dependence Graph ***************************/ 

vertex statement_to_vertex(statement s, graph g)
{
    MAP(VERTEX, v, {
        statement sv = vertex_to_statement(v);
	if (statement_equal_p(s, sv))
	  return v;
    }, graph_vertices(g));
    return vertex_undefined;
}

static list enclosed_statements_ast(statement stmt, list children_s)
{
  instruction inst = statement_instruction(stmt);
  switch(instruction_tag(inst))
    {
    case is_instruction_block :
      {
	MAPL( stmt_ptr,
	      {
		statement local_stmt = STATEMENT(CAR( stmt_ptr ));
		children_s = CONS( STATEMENT, local_stmt, children_s);
		children_s = enclosed_statements_ast(local_stmt,children_s);
	      },
	      instruction_block( inst ) );
	break;
      }
    case is_instruction_test :
      {
	test t = instruction_test(inst);
	children_s = CONS( STATEMENT, test_true(t), children_s);
	children_s = CONS( STATEMENT, test_false(t), children_s);
	children_s = enclosed_statements_ast(test_true(t),children_s);
	children_s = enclosed_statements_ast(test_false(t),children_s);
	break;
      }
    case is_instruction_loop :
      {
	loop l = statement_loop(stmt);
	statement body = loop_body(l);
	children_s = CONS( STATEMENT, body, children_s);
	children_s = enclosed_statements_ast(body,children_s);
	break;
      }
    case is_instruction_forloop :
      {
	forloop l = statement_forloop(stmt);
	statement body = forloop_body(l);
	children_s = CONS( STATEMENT, body, children_s);
	children_s = enclosed_statements_ast(body,children_s);
	break;
      }
    case is_instruction_whileloop :
      {
	whileloop l = statement_whileloop(stmt);
	statement body = whileloop_body(l);
	children_s = CONS( STATEMENT, body, children_s);
	children_s = enclosed_statements_ast(body,children_s);
	break;
      }
    case is_instruction_call:
      //children_s = CONS( STATEMENT, stmt, children_s);
      break;
    default:
      break;
    }
  return children_s;
}

static bool same_level_p(statement s1, statement s2, bool found_p)
{
    
  if(statement_equal_p(s1, s2))
    return true;
  else {
    instruction inst = statement_instruction(s2);
    switch(instruction_tag(inst)){
    case is_instruction_block :{
      MAPL( stmt_ptr,
	    {
	      statement local_stmt = STATEMENT(CAR( stmt_ptr ));
	      found_p = found_p || same_level_p(s1, local_stmt, found_p);
	    },
	    instruction_block(inst));
      break;
    }
    case is_instruction_test :{
      test t = instruction_test(inst);
      found_p = found_p || same_level_p(s1, test_true(t), found_p);
      return found_p || same_level_p(s1, test_false(t), found_p);
      break;
    }
    case is_instruction_loop :{
      loop l = statement_loop(s2);
      statement body = loop_body(l);
      return found_p || same_level_p(s1, body, found_p);
      break;
    }
    case is_instruction_forloop :{
      forloop l = statement_forloop(s2);
      statement body = forloop_body(l);
      return found_p || same_level_p(s1, body, found_p);
      break;
    }
    case is_instruction_whileloop :{
      whileloop l = statement_whileloop(s2);
      statement body = whileloop_body(l);
      return found_p || same_level_p(s1, body, found_p);
      break;
    }
    default:
      break;
    }
  }
  return found_p;
}
static statement in_same_sequence(statement child, sequence seq)
{
  statement enclosing_st = statement_undefined;
  bool found_p;
  list stmts = sequence_statements(seq);
  FOREACH(STATEMENT, st, stmts){
    found_p = same_level_p(child, st, false);
    if(found_p){
      enclosing_st = st;
      return st;
    }
  }
  return enclosing_st;
}

/* for precision in dependences in arrays, we use array regions in this function*/

static bool test_dependence_using_regions(statement s1, statement s2)
{
  bool dependence_b = false;
  list l_write_1 = regions_dup(regions_write_regions(load_statement_local_regions(s1)));
  list private_ents1 = NIL, private_ents2 = NIL;
  if(statement_loop_p(s1))
    private_ents1 =  loop_locals(statement_loop(s1));
  private_ents1= gen_nconc(private_ents1,private_variables(s1));
  if(statement_loop_p(s2))
     private_ents2 = loop_locals(statement_loop(s2));
  private_ents2= gen_nconc(private_ents2, private_variables(s2));
  FOREACH(ENTITY,e,private_ents1){
    FOREACH(REGION,reg,l_write_1){
      if (same_entity_p(region_entity(reg),e))
	gen_remove(&l_write_1, reg);
    }
  }
  list l_write_2 = regions_dup(regions_write_regions(load_statement_local_regions(s2)));
  FOREACH(ENTITY,e,private_ents2){
    FOREACH(REGION,reg,l_write_2){
      if (same_entity_p(region_entity(reg),e))
	gen_remove(&l_write_2, reg);
    }
  }
  l_write_2 = convex_regions_transformer_compose(l_write_2, p_transformer);
  list l_read_2 = regions_dup(regions_read_regions(load_statement_local_regions(s2)));  
  FOREACH(ENTITY,e,private_ents2){
    FOREACH(REGION,reg,l_read_2) {
      if (same_entity_p(region_entity(reg),e))
	gen_remove(&l_read_2, reg);
    }
  }
  l_read_2 = convex_regions_transformer_compose(l_read_2, p_transformer);
  list l_read_1 = regions_dup(regions_read_regions(load_statement_local_regions(s1)));  
  FOREACH(ENTITY,e,private_ents1){
    FOREACH(REGION,reg,l_read_1){
      if (same_entity_p(region_entity(reg),e))
	gen_remove(&l_read_1, reg);
    }
  }
  list dependence_rw = RegionsIntersection(regions_dup(l_read_1), regions_dup(l_write_2), r_w_combinable_p);
  list dependence_wr = RegionsIntersection(regions_dup(l_write_1), regions_dup(l_read_2), w_r_combinable_p);
  list dependence_ww = RegionsIntersection(regions_dup(l_write_1), regions_dup(l_write_2), w_w_combinable_p);
  if(size_of_regions(dependence_rw)==0)
    {
      if(size_of_regions(dependence_wr)==0)
	{
	  if(size_of_regions(dependence_ww)>0)
	    dependence_b=true;
	}
      else
	dependence_b=true;
    }
  else
    dependence_b = true;
  return dependence_b;
}

static bool sequence_dg(statement stmt)
{
  if(statement_sequence_p(stmt)){
    sequence seq = statement_sequence(stmt);
    list stmts = sequence_statements(seq);
    path pbegin;
    path pend;
    FOREACH(STATEMENT, st_g, stmts) {
      {
	list children_s = CONS(STATEMENT, st_g, NIL);
	list ls = NIL;
	children_s = enclosed_statements_ast(st_g,children_s);
	vertex pre_g = statement_to_vertex(st_g, sdg);
	FOREACH(STATEMENT, st, children_s){
	  vertex pre = statement_to_vertex(st, sdg);
	  FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	    vertex s = successor_vertex(su);
	    statement child = vertex_to_statement(s);
	    statement enclosing_stmt = statement_undefined;
	    enclosing_stmt = in_same_sequence(child, seq);
	    if(statement_undefined_p(enclosing_stmt))
	      enclosing_stmt = in_same_sequence(st, seq);

	    p_transformer =  transformer_identity();
	    path_initialize(stmt, st,child, &pbegin, &pend);
	    //p_transformer = compute_path_transformer(stmt, pbegin, pend);
	    /*update the sdg for st(parent) using enclosing_stmt instead of
	     *the enclosed one child*/
	    if(!statement_equal_p(st_g, enclosing_stmt) && test_dependence_using_regions(st_g,enclosing_stmt)){
	      if(statement_ordering(st_g)<statement_ordering(enclosing_stmt)){
		vertex new_v = statement_to_vertex(enclosing_stmt, sdg);
		successor_vertex(su) = new_v;
		ls = CONS(SUCCESSOR, su, ls);
	      }
	    }
	  }
	}
	vertex_successors(pre_g) = ls; 
      }
    }
  }
  return true;
}

static bool statement_in_sequence_p(statement s, statement stmt, bool found_p)
{
  instruction inst = statement_instruction(stmt);
  switch(instruction_tag(inst))
    {
    case is_instruction_block :
      {
	MAPL( stmt_ptr,
	      {
		statement local_stmt = STATEMENT(CAR( stmt_ptr ));
		if(statement_equal_p(s, local_stmt))
		  return true;
		else
		  found_p = found_p ||  statement_in_sequence_p(s, local_stmt, found_p);
	      },
	      instruction_block( inst ) );
	break;
      }
    case is_instruction_test :
      {
	test t = instruction_test(inst);
	found_p = found_p ||  statement_in_sequence_p(s, test_true(t), found_p);
	return found_p ||  statement_in_sequence_p(s, test_false(t), found_p);
	break;
      }
    case is_instruction_loop :
      {
	loop l = statement_loop(stmt);
	statement body = loop_body(l);
	return found_p ||  statement_in_sequence_p(s, body, found_p);
	break;
      }
    case is_instruction_forloop :
      {
	forloop l = statement_forloop(stmt);
	statement body = forloop_body(l);
	return found_p ||  statement_in_sequence_p(s, body, found_p);
	break;
      }
    case is_instruction_whileloop :
      {
	whileloop l = statement_whileloop(stmt);
	statement body = whileloop_body(l);
	return found_p ||  statement_in_sequence_p(s, body, found_p);
	break;
      }
    default:
      break;
    }
  return found_p;
}


/* Second step to form a clustered DG (SDG), delete dependences between  statement s1 and another statement S2
   if s1 is not in a sequence  and redondont dependences*/  
static graph clean_sdg(statement module_stmt, graph tg)
{
  list vertices = graph_vertices(tg);
  statement s;
  FOREACH(VERTEX, pre, vertices)
    {
      s = vertex_to_statement(pre);
      bool found_p =  statement_in_sequence_p(s, module_stmt, false);
      if(!found_p)
	{
	  gen_free_list(vertex_successors(pre));
	  vertex_successors(pre) = NIL;
	}
      else{
	int count;
	list ls = NIL, lv = NIL;
	FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	  vertex ssu = successor_vertex(su);
	  statement child = vertex_to_statement(ssu);
	  count = gen_occurences(child, ls);
	  if(count == 0 && statement_ordering(s)<statement_ordering(child))
	    {
	      ls = CONS(STATEMENT, child,ls);
	      lv = CONS(SUCCESSOR,su,lv);
	    }
	}
	vertex_successors(pre) = lv; 
      }
    }
  return tg;
}


graph partitioning_sdg(statement module_stmt)
{
  gen_recurse(module_stmt, statement_domain, sequence_dg, gen_null);
  return clean_sdg(module_stmt, sdg);
}






/** \def dot_print_label_string( fd, str )
  print the string str in file descriptor fd, removing all \n
 */
#define dot_print_label_string( ftg, str )				\
  fprintf(ftg," [fontsize=24,fontcolor=black, label=\"");	   \
  while ( *str ) {			   \
    char c = *str++;				    \
    if ( c == '"' ) { /* some char must be escaped */	\
      (void) putc( '\\', ftg);				\
    }							\
    (void) putc( c, ftg);				\
  }\
  fprintf(ftg," \"]; \n"); 


static string prettyprint_dot_label(statement s, string label1 ) {
  string label2="";
  // saving comments
  string i_comments = statement_comments(s);
  // remove them
  statement_comments(s) = string_undefined;
  // Get the text without comments
  text txt = text_statement(entity_undefined, 0, s, NIL);
  // Restoring comments
  statement_comments(s) = i_comments;
  label2 = strdup (concatenate (label1, text_to_string(txt),"\\n", NULL));
  free_text(txt);
  return label2;
 }

/* 
 *return a dot graph for SDG, print only nodes that have at least one
 *successor
 */
static void print_task(FILE *ftg, gen_array_t annotations, statement stmt) {
  string ver = "";
  ver = prettyprint_dot_label(stmt,ver);
  fprintf( ftg,"%d",(int)statement_ordering(stmt) );
  if(strlen(ver)>1000)
    ver = "too large statement label with ordering \\n";
  if(gen_array_nitems(annotations)>0){
    annotation *an=gen_array_item(annotations, (int)statement_ordering(stmt));
    if(an->cluster != -1)
      {
	ver = strdup (concatenate (ver, "cluster = ", NULL));ver = strdup (concatenate (ver, itoa(an->cluster),"\\n", NULL));
	ver = strdup (concatenate (ver, "order = ", NULL)); ver = strdup (concatenate (ver, itoa(an->order_sched),"\\n", NULL));
	ver = strdup (concatenate (ver, "time = ", NULL)); ver = strdup (concatenate (ver, itoa(an->task_time),"\\n", NULL));
      }
  }
  dot_print_label_string( ftg, ver);	
  return;
}

static int count = 0;
void print_SDGs(statement stmt, graph tg, FILE *ftg, gen_array_t annotations) {
  instruction inst = statement_instruction(stmt);
  switch(instruction_tag(inst))
    {
    case is_instruction_block :
      {
      sequence seq = statement_sequence(stmt);
      list stmts = sequence_statements(seq);
      fprintf( ftg, "subgraph cluster%d { color = blue; \n ",count );
      FOREACH(STATEMENT, s, stmts) {
	list vertices = graph_vertices(tg);
	annotation *anp = NULL;
	FOREACH(VERTEX, pre, vertices) {
	  statement parent = vertex_to_statement(pre);
	  if(statement_equal_p(parent, s)){
	    if(gen_array_nitems(annotations)>0)
	      anp = gen_array_item(annotations, (int)statement_ordering(parent));
	    print_task(ftg, annotations, parent);
	    FOREACH(SUCCESSOR, su, (vertex_successors(pre))) {
	      vertex s = successor_vertex(su);
	      statement child = vertex_to_statement(s);
	      print_task(ftg, annotations, child);
	      if(gen_array_nitems(annotations)>0){
		annotation *an=gen_array_item(annotations, (int)statement_ordering(child));
		if(an->cluster!=-1)
		  {//FFT
		    double edge_c = (intptr_t)(gen_array_item(anp->edge_cost,statement_ordering(child)));
		    fprintf( ftg,"%d -> %d [style=filled,color=blue,fontsize=16,label=\"%ld\",color=black];\n", (int)statement_ordering(parent),(int)statement_ordering(child),(long)(edge_c));
		  }
	      }
	      else
		fprintf( ftg,"%d -> %d [style=filled,color=blue,fontsize=16,color=black];\n", (int)statement_ordering(parent),(int)statement_ordering(child));
	    }
	  }
	}
      }
      fprintf( ftg, "} \n " );
      count ++;
      stmts = sequence_statements(seq);
      FOREACH(STATEMENT, s, stmts) 
	print_SDGs(s,tg, ftg, annotations);
      break;
      }
    case is_instruction_test:
      {
	test t = instruction_test(inst);
	print_SDGs(test_true(t),tg, ftg, annotations); 
	print_SDGs(test_false(t),tg, ftg, annotations); 
	break;
      }
    case is_instruction_loop :
      {
	loop l = statement_loop(stmt);
	print_SDGs(loop_body(l),tg, ftg, annotations);
	break;
      }
    case is_instruction_forloop :
      {
	forloop l = statement_forloop(stmt);
	print_SDGs(forloop_body(l),tg, ftg, annotations);
	break;
      }
    case is_instruction_whileloop :
      {
	whileloop l = statement_whileloop(stmt);
	print_SDGs(whileloop_body(l),tg, ftg, annotations);
	break;
      }
    default:
      break;
    }
  return; 
}
bool sequence_dependence_graph(char * module_name)
{ 
  entity	module;
  statement	module_stat;
  string tg_name = NULL;
  FILE *ftg;
  module = local_name_to_top_level_entity(module_name);
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
 
  sdg = (graph) db_get_memory_resource (DBR_DG, module_name, true );
  /*ddg contains the original dependences before clustering and  scheduling, it
    is used to construct the SDG*/
  ddg = copy_graph(sdg);
  sdg = partitioning_sdg(module_stat);
  tg_name = strdup(concatenate(db_get_current_workspace_directory(),
			       "/",module_name,"/",module_name, "_sdg.dot", NULL));
  ftg = safe_fopen(tg_name, "w");
  fprintf( ftg, "digraph {\n compound=true;ratio=fill; node[fontsize=24,fontname=\"Courier\",labelloc=\"t\"];nodesep=.05;\n" );
  print_SDGs(module_stat, sdg, ftg, gen_array_make(0));
  fprintf( ftg, "\n}\n" );
  safe_fclose(ftg, tg_name);
  free(tg_name);
  
  reset_ordering_to_statement();
  DB_PUT_MEMORY_RESOURCE(DBR_DG, module_name, (char*) sdg);
  
  reset_proper_rw_effects();
  reset_cumulated_rw_effects();
  reset_rw_effects();
  reset_in_effects();
  reset_out_effects();
  reset_precondition_map();
  reset_transformer_map();
  reset_current_module_statement();
  reset_current_module_entity();
  generic_effects_reset_all_methods();
  free_value_mappings();
  return true;
}

