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

static bool omp_parallel = false;


static void gen_omp_taskwait(statement stmt)
{
  statement st = make_continue_statement(entity_empty_label());
  string data = strdup(concatenate( "omp taskwait ",NULL));
  add_pragma_str_to_statement (st, data, true);
  statement_synchronization(stmt) =  make_synchronization_none();
  list list_stmts = CONS(STATEMENT, st, CONS(STATEMENT, copy_statement(stmt), NIL));
  statement_instruction(stmt) = make_instruction_sequence(make_sequence(gen_nreverse(list_stmts)));  
  return;
}
static void gen_omp_parallel(statement stmt){
  string data = strdup(concatenate( "omp parallel default(shared) ",NULL));
  if(statement_sequence_p(stmt)){
    list stmts = sequence_statements(statement_sequence(stmt)), body = NIL, decls = NIL;
    statement st_body = statement_undefined;
    FOREACH(STATEMENT, st, stmts){
      if(declaration_statement_p(st))
	decls =  CONS(STATEMENT, st, decls);
      else
	if(!return_statement_p(st))
	  body = CONS(STATEMENT, st, body);
	else
	  return_st = st;
    }
    if(gen_length(body)>0){
      st_body = make_block_statement(gen_nreverse(body));
      add_pragma_str_to_statement (st_body, data, true);
      decls =  CONS(STATEMENT, st_body, decls);
    }
    if(gen_length(decls)>0){
      statement_instruction(stmt) = make_instruction_sequence(make_sequence(gen_nreverse(decls)));
    }
  }      
  else
    add_pragma_str_to_statement (stmt, data, true);
  return;
}

static bool gen_synchronization(statement stmt, bool nested_p, int length)
{
 synchronization sync  = statement_synchronization(stmt);
  switch(synchronization_tag(sync)){
  case is_synchronization_spawn:
    if(length>1){
      add_pragma_str_to_statement(stmt, "omp task", true);
      if(!omp_parallel)
	omp_parallel = true;
    }
    break;
  case is_synchronization_barrier:
    if(!nested_p){
      add_pragma_str_to_statement(stmt, "omp single", true);
      nested_p = true;
    }
    else
      {
	if(gen_length(sequence_statements(statement_sequence(stmt)))>1)
	  gen_omp_taskwait(stmt);
      }
    if(!omp_parallel)
      omp_parallel = true;
    break;
  default:
    break;
  }
  if(com_instruction_p(statement_instruction(stmt)))
      statement_instruction(stmt) = make_continue_instruction();
  return nested_p;
}

static bool gen_openmp(statement stmt, bool nested_p){
  instruction inst = statement_instruction(stmt);
  switch(instruction_tag(inst))
    {
    case is_instruction_block:
      {
	MAPL(stmt_ptr,
	      {
		statement st = STATEMENT(CAR( stmt_ptr));
		bool nested_p_local = gen_synchronization(st,nested_p, gen_length(sequence_statements(statement_sequence(stmt))));
		gen_openmp(st, nested_p_local);
	      },
	      instruction_block(inst));
	break;
      }
    case is_instruction_test :
      {
	test t = instruction_test(inst);
	bool nested_p_t = gen_synchronization(test_true(t),nested_p,1);
	bool nested_p_f = gen_synchronization(test_false(t),nested_p,1);
	gen_openmp(test_true(t), nested_p_t);
	gen_openmp(test_false(t), nested_p_f);
	break;
      }
    case is_instruction_loop :
      {
	loop l = statement_loop(stmt);
	statement body = loop_body(l);
	nested_p = gen_synchronization(body,nested_p,1);
	gen_openmp(body, nested_p);
	break;
      }
    default:
      break;
    }
  return true;
}


            
/* OpenMP generation pass */
bool openmp_task_generation(char * module_name)
{ 
  statement module_stat_i = (statement)db_get_memory_resource(DBR_SHARED_SPIRE_CODE, module_name, true);
  statement module_stat = copy_statement(module_stat_i);
  set_ordering_to_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement(module_stat);
  gen_openmp(module_stat, false);
  if(omp_parallel)
    gen_omp_parallel(module_stat);
  if(!statement_undefined_p(return_st)) 
    insert_statement(module_stat, return_st, false);
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_PARALLELIZED_CODE, module_name, module_stat);
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();
  return true;
}

