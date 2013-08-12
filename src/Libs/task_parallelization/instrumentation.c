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
#include "complexity.h"
#include "dg.h"

/* Instantiation of the dependence graph: */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"
#include "ricedg.h"
#include "chains.h"
#include "task_parallelization.h"




//profiling using instrumentation
static void task_time_polynome(statement s)
{
  statement st_poly ;
  if(statement_equal_p(s, get_current_module_statement()))
    {
      string dir = strdup(concatenate("char *file_name = \"", "./instrumented_",get_current_module_name(),".in\";", NULL));
      string mode = "\"w\"";
      string new_name = strdup(concatenate("FILE *finstrumented;\n   ", dir, "\n   finstrumented = fopen(file_name,", mode, ");\n", NULL)) ;
      entity new_ent = make_constant_entity(new_name, is_basic_string, 1000);
      st_poly = make_expression_statement(make_entity_expression(new_ent, NIL));
    }
  else{
    complexity stat_comp = load_statement_complexity(s);
    string r;
    if(stat_comp != (complexity) HASH_UNDEFINED_VALUE && !complexity_zero_p(stat_comp)) {
      cons *pc = CHAIN_SWORD(NIL, complexity_sprint(stat_comp, false, true));
      r = words_to_string(pc);
    }
    else 
      r = i2a(1);
    string new_name3 = "finstrumented";
    entity new_ent3 = make_constant_entity(new_name3, is_basic_string, 100);
    expression exp3 = make_entity_expression(new_ent3, NIL);
    string new_name = strdup(concatenate( "\"", itoa(statement_ordering(s)),"->-1 = %lf \\n", "\"", NULL));  
    entity new_ent = make_constant_entity(new_name, is_basic_string, 1000);
    expression exp = make_entity_expression(new_ent, NIL);
    entity new_ent2 = make_constant_entity(r, is_basic_string, 100);
    expression exp2 = make_entity_expression(new_ent2, NIL);
    list args = CONS(EXPRESSION, exp3, CONS(EXPRESSION,exp,CONS(EXPRESSION,exp2,NIL)));
    st_poly = make_call_statement(FPRINTF_FUNCTION_NAME,
					    args,
					    entity_undefined,
					    empty_comments);
  }
  statement new_s = copy_statement(s);
  instruction ins_seq = make_instruction_sequence(make_sequence(make_statement_list(st_poly, new_s)));
  statement_instruction(s) = ins_seq;
  free_extensions(statement_extensions(s));
  statement_extensions(s)=empty_extensions();
  if(!string_undefined_p(statement_comments(s))) free(statement_comments(s));
  statement_comments(s)=empty_comments;
  return; 
}    

static void edge_cost_polynome(statement s1, statement s2)
{ 
  Ppolynome transfer_time = POLYNOME_NUL;
  string r;
  list l_write_1 = regions_dup(regions_write_regions(load_statement_local_regions(s1)));
  list l_read_2 = regions_dup(regions_read_regions(load_statement_local_regions(s2)));
  list l_communications = RegionsIntersection(regions_dup(l_write_1), regions_dup(l_read_2), w_r_combinable_p);
  FOREACH(REGION,reg,l_communications){
    Ppolynome reg_footprint = region_enumerate(reg); 
    reg_footprint = polynome_mult(reg_footprint, expression_to_polynome(int_to_expression(SizeOfElements(variable_basic(type_variable(entity_type(region_entity(reg))))))));
    polynome_add(&transfer_time, reg_footprint);
 
  }
  if(!POLYNOME_UNDEFINED_P(transfer_time)){
    cons *pc = words_syntax(expression_syntax(polynome_to_expression(transfer_time)),NIL);
    r = words_to_string(pc);
  }
  else
    r = "0";
  string new_name3 = "finstrumented";
  entity new_ent3 = make_constant_entity(new_name3, is_basic_string, 100);
  expression exp3 = make_entity_expression(new_ent3, NIL);
  string s2_ordering = strdup(concatenate( "\"", itoa(statement_ordering(s1)),NULL));
  string new_name = strdup(concatenate( s2_ordering, "->", itoa(statement_ordering(s2))," = %lf \\n", "\"", NULL));  
  entity new_ent = make_constant_entity(new_name, is_basic_string, 1000);
  expression exp = make_entity_expression(new_ent, NIL);
  entity new_ent2 = make_constant_entity(r, is_basic_string, 100);
  expression exp2 = make_entity_expression(new_ent2, NIL);
  list args = CONS(EXPRESSION, exp3, CONS(EXPRESSION,exp,CONS(EXPRESSION,exp2,NIL)));
  statement st_poly = make_call_statement(FPRINTF_FUNCTION_NAME,
				args,
				entity_undefined,
				empty_comments);
  statement new_s = copy_statement(s2);
  instruction ins_seq = make_instruction_sequence(make_sequence(make_statement_list(st_poly, new_s)));
  statement_instruction(s2) = ins_seq;
  free_extensions(statement_extensions(s2));
  statement_extensions(s2)=empty_extensions();
  if(!string_undefined_p(statement_comments(s2))) free(statement_comments(s2));
  statement_comments(s2)=empty_comments;
  return;
}

bool bdsc_code_instrumentation(char * module_name)
{ 
  entity	module;
  statement	module_stat;
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
  //set_in_effects((statement_effects) db_get_memory_resource(DBR_IN_REGIONS, module_name, true));
  //set_out_effects((statement_effects) db_get_memory_resource(DBR_OUT_REGIONS, module_name, true));
  set_methods_for_convex_effects();
  init_convex_rw_prettyprint(module_name);
 
  kdg = (graph) db_get_memory_resource (DBR_DG, module_name, true );
  
  /*Complexities (task processing time)*/
  set_complexity_map( (statement_mapping) db_get_memory_resource(DBR_COMPLEXITIES, module_name, true));

 
  /*first step is to cumulate dependences hierarchically (between
    sequences) on granularities : loop, test and simple instruction*/
  list vertices = graph_vertices(kdg);
  FOREACH(VERTEX, v, vertices){
    statement stmt = vertex_to_statement(v);
    task_time_polynome(stmt);
    FOREACH(SUCCESSOR, su, (vertex_successors(statement_to_vertex(stmt, kdg)))) {
      vertex s = successor_vertex(su);
      statement child = vertex_to_statement(s);
      edge_cost_polynome(stmt,child);
    }
  }
  /* Reorder the module, because new statements have been generated. */
  module_reorder(module_stat);
  
  reset_proper_rw_effects();
  reset_cumulated_rw_effects();
  reset_rw_effects();
  reset_precondition_map();
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();
  reset_complexity_map();
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
  return true;
}

