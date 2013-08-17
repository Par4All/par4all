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
#include "c_syntax.h"
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

static statement init_stmt = statement_undefined;
static entity rank, mpi_status, mpi_request;

static void gen_mpi_send_recv(statement stmt)
{
  list args = NIL, list_stmts = NIL;
  list lexpr = call_arguments(instruction_call(statement_instruction(stmt)));
  expression dest = EXPRESSION(CAR(lexpr)); expression size; 
  statement st;
  int pair = 0; 
  entity name;
  if(native_instruction_p(statement_instruction(stmt), SEND_FUNCTION_NAME)){
    name = make_constant_entity("MPI_Isend",is_basic_string, 100);
    expression exp_req = make_entity_expression(mpi_request, NIL);
    args = CONS(EXPRESSION, make_address_of_expression(exp_req), args);
  }
  else{
    name = make_constant_entity("MPI_Recv",is_basic_string, 100);
    expression exp_st = make_entity_expression(mpi_status, NIL);
    args = CONS(EXPRESSION, make_address_of_expression(exp_st), args);
  }
  args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("MPI_COMM_WORLD", is_basic_string, 100), NIL), args);
  args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("MPI_ANY_TAG", is_basic_string, 100), NIL), args);
  args = CONS(EXPRESSION, dest, args);
  list args_init = args;
  FOREACH(EXPRESSION, expr, CDR(lexpr)){
    pair++;
    if(pair == 2) {
      pair = 0;
      args = args_init;
      if (basic_int_p(variable_basic(type_variable(entity_type(expression_to_entity(expr))))))
	args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("MPI_INT", is_basic_string, 100), NIL), args);
      else
	args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("MPI_FLOAT", is_basic_string, 100), NIL), args);
      args = CONS(EXPRESSION, size, args);
      args = CONS(EXPRESSION, expr, args);
      st = make_statement(
			  statement_label(stmt),
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED,
			  empty_comments,
			  make_instruction(is_instruction_call, make_call(name, (args))),
			  NIL, NULL, empty_extensions(), make_synchronization_none());
      list_stmts = CONS(STATEMENT, st, list_stmts);
    }
    else
      size = expr;
  }
  statement_instruction(stmt) = make_instruction_sequence(make_sequence(gen_nreverse(list_stmts)));  
  return;
}

static void gen_if_rank(statement stmt, synchronization sync){
  expression test_condition = MakeBinaryCall (entity_intrinsic(EQUAL_OPERATOR_NAME),
					      entity_to_expression (rank),
					      entity_to_expression (synchronization_spawn(sync)));
  statement st = make_statement(
		      statement_label(stmt),
		      STATEMENT_NUMBER_UNDEFINED,
		      STATEMENT_ORDERING_UNDEFINED,
		      empty_comments,
		      copy_instruction(statement_instruction(stmt)),
		      NIL, NULL, empty_extensions(), make_synchronization_none());
  test new_test = make_test( test_condition,
			st,
			make_continue_statement(entity_undefined));
  instruction inst = make_instruction(is_instruction_test, new_test);
  free_instruction(statement_instruction(stmt));
  statement_instruction(stmt) = inst;
  return;
}


static bool gen_mpi(statement stmt){
  synchronization sync  = statement_synchronization(stmt);
  entity name;
  instruction inst;
  statement st, new_s; list list_stmts = NIL, args = NIL;
  switch(synchronization_tag(sync)){
  case is_synchronization_spawn:
    gen_if_rank(stmt, sync);
    break;
  case is_synchronization_barrier:
    name = make_constant_entity("MPI_Barrier",is_basic_string, 100);
    args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("MPI_COMM_WORLD", is_basic_string, 100), NIL), NIL); 
    st = make_statement(
			statement_label(stmt),
			STATEMENT_NUMBER_UNDEFINED,
			STATEMENT_ORDERING_UNDEFINED,
			empty_comments,
			make_instruction(is_instruction_call, make_call(name, gen_nreverse(args))),
			NIL, NULL, empty_extensions(), make_synchronization_none());
    new_s = make_statement(
			   statement_label(stmt),
			   STATEMENT_NUMBER_UNDEFINED,
			   STATEMENT_ORDERING_UNDEFINED,
			   statement_comments(stmt),
			   statement_instruction(stmt),
			   NIL, NULL, statement_extensions(stmt), make_synchronization_none());
    list_stmts = CONS(STATEMENT, new_s, CONS(STATEMENT, st, NIL));
    inst = make_instruction_sequence(make_sequence(list_stmts));
    statement_instruction(stmt) = inst;
    statement_comments(stmt) = empty_comments;
    break;
  default:
    break;
  }
  if(com_instruction_p(statement_instruction(stmt))){
    gen_mpi_send_recv(stmt);
  }
  return true;
}


/* Generate
 * int rank0;
 * MPI_Status status0; 
 * MPI_Request *request0;
 * ierr = MPI_Init( &argc, &argv );
 * ierr = MPI_Comm_rank( MPI_COMM_WORLD, &rank );
 */
static statement mpi_initialize(statement stmt, entity module){
  list args = NIL, list_stmts = NIL;
  statement st = statement_undefined;
  rank = make_new_scalar_variable_with_prefix("rank", module, MakeBasic(is_basic_int));
  entity stat =   FindOrCreateEntity("mpi_status_a", TYPEDEF_PREFIX "MPI_Status");
  if(storage_undefined_p(entity_storage(stat)))
    {
      entity_storage(stat) = make_storage_rom();
      put_new_typedef("MPI_Status");
    }
  type stat_t =MakeTypeVariable(make_basic_typedef(stat), NIL);
  mpi_status = make_new_scalar_variable_with_prefix("status", module,MakeBasic(is_basic_int) );
  entity_type(mpi_status) = stat_t;
  
  entity req =   FindOrCreateEntity("mpi_request_a", TYPEDEF_PREFIX "MPI_Request");
  if(storage_undefined_p(entity_storage(req)))
    {
      entity_storage(req) = make_storage_rom();
      put_new_typedef("MPI_Request");
    }
  type req_t =MakeTypeVariable(make_basic_typedef(req), NIL);
  mpi_request = make_new_scalar_variable_with_prefix("request", module,MakeBasic(is_basic_int) );
  entity_type(mpi_request) = req_t;
  
  entity name = make_constant_entity("MPI_Init",is_basic_string, 100);
  args = CONS(EXPRESSION, make_address_of_expression(make_entity_expression(make_constant_entity("argc", is_basic_string, 100), NIL)),args);
  args = CONS(EXPRESSION, make_address_of_expression(make_entity_expression(make_constant_entity("argv", is_basic_string, 100), NIL)), args);
  st = make_statement(
		      statement_label(stmt),
		      STATEMENT_NUMBER_UNDEFINED,
		      STATEMENT_ORDERING_UNDEFINED,
		      empty_comments,
		      make_instruction(is_instruction_call, make_call(name, gen_nreverse(args))),
		      NIL, NULL, empty_extensions(), make_synchronization_none());
  list_stmts = CONS(STATEMENT, st, list_stmts);
  name = make_constant_entity("MPI_Comm_rank",is_basic_string, 100);
  args =NIL;
  args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("MPI_COMM_WORLD", is_basic_string, 100), NIL), args); 
  args = CONS(EXPRESSION, make_address_of_expression(entity_to_expression(copy_entity(rank))), args);
  st = make_statement(
		      statement_label(stmt),
		      STATEMENT_NUMBER_UNDEFINED,
		      STATEMENT_ORDERING_UNDEFINED,
		      empty_comments,
		      make_instruction(is_instruction_call, make_call(name, gen_nreverse(args))),
		      NIL, NULL, empty_extensions(), make_synchronization_none());
  list_stmts = CONS(STATEMENT, st, list_stmts);
  st = make_statement(
		      statement_label(stmt),
		      STATEMENT_NUMBER_UNDEFINED,
		      STATEMENT_ORDERING_UNDEFINED,
		      empty_comments,
		      make_instruction_sequence(make_sequence(gen_nreverse(list_stmts))),
		      NIL, NULL, empty_extensions(), make_synchronization_none());
  statement new_s = make_statement(
				   statement_label(stmt),
				   statement_number(stmt),
				   statement_ordering(stmt),
				   statement_comments(stmt),
				   statement_instruction(stmt),
				   NIL, NULL, statement_extensions(stmt), statement_synchronization(stmt));
  list_stmts = CONS(STATEMENT, new_s, CONS(STATEMENT, st, NIL));
  statement_instruction(stmt) = make_instruction_sequence(make_sequence(gen_nreverse(list_stmts)));
  //free_extensions(statement_extensions(stmt));
  statement_extensions(stmt) = statement_extensions(stmt);// empty_extensions();
  statement_comments(stmt) = empty_comments;
  statement_synchronization(stmt) = make_synchronization_none();
  AddLocalEntityToDeclarations(rank, module, stmt);
  AddLocalEntityToDeclarations(mpi_status, module, stmt);
  AddLocalEntityToDeclarations(mpi_request, module, stmt);
  return st;
}
            
static void mpi_finalize(statement stmt){
  entity name = make_constant_entity("MPI_Finalize",is_basic_string, 100);
  list args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("", is_basic_string, 100), NIL), NIL); 
  statement st = make_statement(
		      statement_label(stmt),
		      STATEMENT_NUMBER_UNDEFINED,
		      STATEMENT_ORDERING_UNDEFINED,
		      empty_comments,
		      make_instruction(is_instruction_call, make_call(name, args)),
		      NIL, NULL, empty_extensions(), make_synchronization_none());
  list list_stmts = CONS(STATEMENT, st, CONS(STATEMENT, copy_statement(stmt), NIL));
  statement_instruction(stmt) = make_instruction_sequence(make_sequence(gen_nreverse(list_stmts)));  
  return;
}

bool mpi_task_generation(char * module_name)
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
  init_stmt = mpi_initialize(module_stat, module);
  gen_recurse(module_stat, statement_domain, gen_mpi, gen_null);
  mpi_finalize(module_stat);
  gen_consistent_p((gen_chunk*)module_stat);
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_PARALLELIZED_CODE, strdup(module_name), (char*)module_stat);
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();
  return true;
}
