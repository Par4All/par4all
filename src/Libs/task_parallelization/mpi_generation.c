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
#include "syntax.h"
#include "bootstrap.h"

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

static bool com_statement_p(statement s)
{
  if(statement_loop_p(s)){
    statement body = loop_body(statement_loop(s));
    return com_statement_p(body);
  }
  else{
    instruction i = statement_instruction(s);  
    return native_instruction_p(i, SEND_FUNCTION_NAME)
      || native_instruction_p(i, RECV_FUNCTION_NAME);
  }
}

static void gen_mpi_send_recv(statement stmt)
{
  list args = NIL, list_stmts = NIL;
  list lexpr = call_arguments(instruction_call(statement_instruction(stmt)));
  expression dest = EXPRESSION(CAR(lexpr)); 
  expression size = int_to_expression(1); 
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
  expression expr =  EXPRESSION(CAR(CDR(lexpr))); 
  basic bas = variable_basic(type_variable(entity_type(expression_to_entity(expr))));
  switch(basic_tag(bas)){
  case is_basic_int:
    args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("MPI_INT", is_basic_string, 100), NIL), args);
    break;
  case is_basic_float:
    args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("MPI_FLOAT", is_basic_string, 100), NIL), args);
    break;
  default:
    pips_user_warning("type not handled yet in MPI\n");
    break;
  }
  args = CONS(EXPRESSION, size, args);
  args = CONS(EXPRESSION, make_address_of_expression(expr), args);
  instruction com_call = make_instruction(is_instruction_call, make_call(name, (args)));
  statement st = make_statement(
			  entity_empty_label(),
			  STATEMENT_NUMBER_UNDEFINED,
			  STATEMENT_ORDERING_UNDEFINED,
			  empty_comments,
			  com_call,
			  NIL, NULL, empty_extensions(), make_synchronization_none());
  list_stmts = CONS(STATEMENT, st, list_stmts);
  statement_instruction(stmt) = make_instruction_sequence(make_sequence(gen_nreverse(list_stmts)));  
  statement_extensions(stmt) = statement_extensions(stmt);
  statement_comments(stmt) = empty_comments;
  statement_synchronization(stmt) = make_synchronization_none();
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
			     make_empty_statement());
  instruction inst = make_instruction(is_instruction_test, new_test);
  statement_instruction(stmt) = inst;
  statement_extensions(stmt) = statement_extensions(stmt);
  statement_comments(stmt) = empty_comments;
  statement_synchronization(stmt) = make_synchronization_none();
  return;
}

/* nesting_level is used to generate only a flat MPI (nesting_level = 1)
 * hierarchical mpi is not implemented yet (nesting_level = 2)*/
static int gen_mpi(statement stmt, int nesting_level){
  synchronization sync  = statement_synchronization(stmt);
  instruction inst;
  statement new_s; list list_stmts = NIL, args = NIL;
  switch(synchronization_tag(sync)){
  case is_synchronization_spawn:
    nesting_level = (nesting_level == 0)? 1 : ((nesting_level == 1) ? 2 : nesting_level);
    gen_if_rank(stmt, sync);
    break;
  case is_synchronization_barrier:
    args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("MPI_COMM_WORLD", is_basic_string, 100), NIL), NIL); 
    statement st = make_call_statement(MPI_BARRIER, args, entity_undefined, string_undefined);
    new_s = make_statement(
			   statement_label(stmt),
			   STATEMENT_NUMBER_UNDEFINED,
			   STATEMENT_ORDERING_UNDEFINED,
			   statement_comments(stmt),
			   copy_instruction(statement_instruction(stmt)),
			   NIL, NULL, statement_extensions(stmt), make_synchronization_none());
    list_stmts = CONS(STATEMENT, new_s, CONS(STATEMENT, st, NIL));
    inst = make_instruction_sequence(make_sequence(list_stmts));
    statement_instruction(stmt) = inst;
    statement_comments(stmt) = empty_comments;
    break;
  default:
    break;
  }
  return nesting_level;
}

static void gen_flat_mpi(statement stmt, int nesting_level)
{
  instruction inst = statement_instruction(stmt);
  switch(instruction_tag(inst))
    {
    case is_instruction_block:
      {
	int nesting_level_sequence = nesting_level;
	MAPL(stmt_ptr,
	     {
	       statement st = STATEMENT(CAR( stmt_ptr));
	       if(nesting_level != 2){
		 nesting_level_sequence  =  gen_mpi(st, nesting_level) ;
	       }
	       gen_flat_mpi(st, nesting_level_sequence);
	     },
	     instruction_block(inst));
	break;
      }
    case is_instruction_test :
      {
	test t = instruction_test(inst);
	int nesting_level_t = gen_mpi(test_true(t), nesting_level);
	int nesting_level_f = gen_mpi(test_false(t), nesting_level);
	gen_flat_mpi(test_true(t), nesting_level_t);
	gen_flat_mpi(test_false(t), nesting_level_f);
	break;
      }
    case is_instruction_loop :
      {
	loop l = statement_loop(stmt);
	statement body = loop_body(l);
	if(!com_statement_p(body))
	  nesting_level =(nesting_level == 1)? 2:nesting_level;
	gen_flat_mpi(body, nesting_level);
	break;
      }
    case is_instruction_call:
      if(com_statement_p(stmt)){
	if(nesting_level == 2)
	  update_statement_instruction(stmt, make_continue_instruction());
	else
	  gen_mpi_send_recv(stmt);
      }
      break;
    default:
      break;
    }
  return;
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
  statement st_body = statement_undefined, st_decls = statement_undefined;
  rank = make_new_scalar_variable_with_prefix("rank", module, MakeBasic(is_basic_int));
  entity stat =   FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, MPI_STATUS);
  if(storage_undefined_p(entity_storage(stat)))
    {
      entity_storage(stat) = make_storage_rom();
      put_new_typedef(MPI_STATUS);
    }
  type stat_t =MakeTypeVariable(make_basic_typedef(stat), NIL);
  if(type_undefined_p(entity_type(stat))) {
    entity_type(stat) = ImplicitType(stat);
  }
  mpi_status = make_new_scalar_variable_with_prefix("status", module,MakeBasic(is_basic_int) );
  entity_type(mpi_status) = stat_t;
  
  entity req =   FindOrCreateEntity(TOP_LEVEL_MODULE_NAME, MPI_REQUEST);
  if(storage_undefined_p(entity_storage(req)))
    {
      entity_storage(req) = make_storage_rom();
      put_new_typedef(MPI_REQUEST);
    }
  type req_t =MakeTypeVariable(make_basic_typedef(req), NIL);
  if(type_undefined_p(entity_type(req))) {
    entity_type(req) = ImplicitType(req);
  }
  mpi_request = make_new_scalar_variable_with_prefix("request", module,MakeBasic(is_basic_int) );
  entity_type(mpi_request) = req_t;
  gen_consistent_p((gen_chunk*)mpi_request);

  if(statement_sequence_p(stmt)){
    list stmts = sequence_statements(statement_sequence(stmt)), body = NIL, decls = NIL;
    FOREACH(STATEMENT, s, stmts){
      if(declaration_statement_p(s)){
	decls =  CONS(STATEMENT, s, decls);
	print_statement(s);
      }
      else
	if(!return_statement_p(s))
	  body = CONS(STATEMENT, s, body);
	else
	  return_st = s;
    }
    if(gen_length(body)>0)
      st_body = make_block_statement(gen_nreverse(body));
    if(gen_length(decls)>0){
      st_decls = make_block_statement(gen_nreverse(decls));
    }
  }
  else
    st_body = stmt;
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
				   STATEMENT_NUMBER_UNDEFINED,
				   STATEMENT_ORDERING_UNDEFINED,
				   statement_comments(stmt),
				   copy_instruction(statement_instruction(st_body)),
				   NIL, NULL, statement_extensions(stmt), statement_synchronization(stmt));
  list_stmts = NIL;
  if(!statement_undefined_p(st_decls))
    list_stmts = CONS(STATEMENT, st_decls, list_stmts);
  list_stmts = CONS(STATEMENT, new_s, CONS(STATEMENT, st, list_stmts));
  
  statement_instruction(stmt) = make_instruction_sequence(make_sequence(gen_nreverse(list_stmts)));
  statement_extensions(stmt) = statement_extensions(stmt);
  statement_comments(stmt) = empty_comments;
  statement_synchronization(stmt) = make_synchronization_none();
  AddLocalEntityToDeclarations(rank, module, stmt);
  AddLocalEntityToDeclarations(mpi_status, module, stmt);
  AddLocalEntityToDeclarations(mpi_request, module, stmt);
  return st;
}
            
static void mpi_finalize(statement stmt){
  list args = CONS(EXPRESSION, make_entity_expression(make_constant_entity("", is_basic_string, 100), NIL), NIL); 
  statement st = make_call_statement(MPI_FINALIZE, args, entity_undefined, string_undefined);
  list list_stmts = CONS(STATEMENT, st, CONS(STATEMENT, copy_statement(stmt), NIL));
  if(!statement_undefined_p(return_st)) 
    list_stmts = CONS(STATEMENT, return_st, list_stmts);
  statement_instruction(stmt) = make_instruction_sequence(make_sequence(gen_nreverse(list_stmts)));  
  statement_extensions(stmt) = statement_extensions(stmt);
  statement_comments(stmt) = empty_comments;
  statement_synchronization(stmt) = make_synchronization_none();
  return;
}

bool mpi_task_generation(char * module_name)
{ 
  entity    module = local_name_to_top_level_entity(module_name);
  statement module_stat_i = (statement)db_get_memory_resource(DBR_DISTRIBUTED_SPIRE_CODE, module_name, true);
  statement module_stat = copy_statement(module_stat_i);
  set_ordering_to_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement(module_stat);
  init_stmt = mpi_initialize(module_stat, module);
  gen_flat_mpi(module_stat, 0);
  mpi_finalize(module_stat);
  module_reorder(module_stat);
  gen_consistent_p((gen_chunk*)module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_PARALLELIZED_CODE, module_name, module_stat);
  reset_current_module_statement();
  reset_current_module_entity();
  reset_ordering_to_statement();
  return true;
}
