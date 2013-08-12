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

static int nb_omp_parallel = 0;

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
      if(nb_omp_parallel == 0){
	gen_omp_parallel(stmt);
	nb_omp_parallel = 1;
      }
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
    if(nb_omp_parallel == 0){
      gen_omp_parallel(stmt);
      nb_omp_parallel = 1;
    }
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
  entity	module;
  statement	module_stat;
  module = local_name_to_top_level_entity(module_name);
  module_stat = (statement)db_get_memory_resource(DBR_CODE, module_name, false);
  set_ordering_to_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement(module_stat);
  if (get_bool_property("SPIRE_GENERATION")) 
    set_bool_property("SPIRE_GENERATION", false);
  gen_openmp(module_stat, false);
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_PARALLELIZED_CODE, module_name, module_stat);
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();
  return true;
}

