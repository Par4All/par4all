/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"
#include "effects-convex.h"

static statement build_call_STEP_MastertoAllScalar(entity module, entity scalar)
{
  /* subroutine STEP_OnetoAllScalar(scalar, algorithm, type)
   */
  list arglist = CONS(EXPRESSION, entity_to_expression(scalar),
		      CONS (EXPRESSION, step_symbolic(STEP_NONBLOCKING_NAME, module), NIL));
  return call_STEP_subroutine(RT_STEP_MasterToAllScalar, arglist,entity_type(scalar));
}

static statement build_call_STEP_MastertoAllRegion(entity module, entity array)
{
  /*subroutine STEP_OnetoAllRegion_I(array, algorithm)   
   */
  expression expr_array = entity_to_expression(array);
  expression expr_algorithm = step_symbolic(STEP_NONBLOCKING_NAME, module);
  list arglist = CONS(EXPRESSION, expr_array,
		      CONS(EXPRESSION, expr_algorithm, NIL));
  return call_STEP_subroutine(RT_STEP_MasterToAllRegion, arglist, type_undefined);
}

static statement build_call_STEP_AlltoMasterRegion(entity module, entity array)
{
  /*subroutine STEP_AlltoOneRegion_I(array, algorithm)   
   */
  expression expr_array = entity_to_expression(array);
  expression expr_algorithm = step_symbolic(STEP_NONBLOCKING_NAME, module);
  list arglist = CONS(EXPRESSION, expr_array,
		      CONS(EXPRESSION, expr_algorithm, NIL));
  return call_STEP_subroutine(RT_STEP_AllToMasterRegion, arglist, type_undefined);
}

static entity master_SR_array(entity mpi_module, region reg)
{
  entity array = region_entity(reg);
  return step_local_arrayRegions(STEP_SR_NAME(array), mpi_module, array, expression_undefined);
}

static entity master_RR_array(entity mpi_module, region reg)
{
  entity array = region_entity(reg);
  return step_local_arrayRegions(STEP_RR_NAME(array), mpi_module, array, expression_undefined);
}

static statement build_mpi_before_master(entity directive_module,entity mpi_module)
{
  step_analyses master_analyse=load_global_step_analyses(directive_module);

  statement begin = call_STEP_subroutine(RT_STEP_Begin, CONS(EXPRESSION,step_symbolic(STEP_MASTER_NAME,mpi_module),NIL), type_undefined);

  //Calcul des regions SEND
  list send_regions=NIL;
  FOREACH(REGION,reg,step_analyses_send(master_analyse))
    {
      if(!region_scalar_p(reg))
	send_regions=CONS(REGION,reg,send_regions);
    }

  expression expr_rank = step_local_rank(mpi_module);
  if (!fortran_module_p(get_current_module_entity()))
    {
      expr_rank = make_call_expression(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), CONS(EXPRESSION, expr_rank, NIL));
    }
  statement rank_stmt = call_STEP_subroutine(RT_STEP_Get_rank, CONS(EXPRESSION,expr_rank,NIL), type_undefined);
  list body = CONS(STATEMENT,step_build_arraybounds(mpi_module, send_regions, master_SR_array, true),
		   CONS(STATEMENT,rank_stmt,NIL));

  if (get_bool_property("STEP_OPTIMIZED_COMMUNICATIONS"))
    {
      //Calcul des regions RECV
      list recv_regions=NIL;
      list flush_list = NIL;
      
      FOREACH(REGION,reg,step_analyses_recv(master_analyse))
	{
	  if(!(region_scalar_p(reg)||
	       io_effect_entity_p(region_entity(reg))))
	    {
	      entity array = region_entity(reg);
	      recv_regions=CONS(REGION,reg,recv_regions);
	      flush_list = CONS(STATEMENT,build_call_STEP_AlltoMasterRegion(mpi_module, array), flush_list);
	    }
	}

      body=CONS(STATEMENT,step_build_arraybounds(mpi_module, recv_regions, master_RR_array, false),
		CONS(STATEMENT,build_call_STEP_WaitAll(flush_list),body));
    }


  return make_block_statement(CONS(STATEMENT, begin, body));
 }

static statement build_mpi_master(entity mpi_module, statement work)
{
  // if (STEP_RANK.EQ.0) ...
  statement if_stmt = instruction_to_statement(make_instruction_test(make_test(MakeBinaryCall(gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME, ".EQ."), entity_domain),step_local_rank(mpi_module), int_to_expression(0)), work, make_block_statement(NIL))));
  return make_block_statement(CONS(STATEMENT,if_stmt,NIL));
}

static statement build_mpi_after_master(entity directive_module,entity mpi_module)
{
  step_analyses master_analyse = load_global_step_analyses(directive_module);

  // communications OUT
  pips_debug(1, "mpi_module = %p\n", mpi_module);
  list flush_list = NIL;
      
  FOREACH(REGION, r, step_analyses_send(master_analyse))
    {
      statement stmt;
      entity array = region_entity(r);
      pips_debug(2, "region %s\n", entity_name(array));
      if(!region_scalar_p(r))
	stmt = build_call_STEP_MastertoAllRegion(mpi_module, array);
      else
	stmt = build_call_STEP_MastertoAllScalar(mpi_module, array);
      
      flush_list = CONS(STATEMENT, stmt, flush_list);
    }

  statement end = call_STEP_subroutine(RT_STEP_End, CONS(EXPRESSION,step_symbolic(STEP_MASTER_NAME,mpi_module),NIL), type_undefined);
  statement flush = build_call_STEP_WaitAll(flush_list);
  pips_debug(1, "End\n");

  return make_block_statement(CONS(STATEMENT, flush, 
				   CONS(STATEMENT, end, NIL)));
}

statement step_compile_master(int step_transformation, entity new_module, statement work)
{
  entity directive_module=get_current_module_entity();
  directive d=load_global_directives(directive_module);
  pips_assert("is master directive",type_directive_omp_master_p(directive_type(d)));
  
  if (step_transformation == STEP_TRANSFORMATION_OMP)
    {
      add_pragma_entity_to_statement(work, directive_module);
      return work;
    }
  else
    {
      statement before_work = build_mpi_before_master(directive_module, new_module);
      statement after_work = build_mpi_after_master(directive_module, new_module);
      statement body = make_block_statement(CONS(STATEMENT, before_work,
						 CONS(STATEMENT, build_mpi_master(new_module, work),
						      CONS(STATEMENT, after_work,NIL))));
      
      if (step_transformation == STEP_TRANSFORMATION_HYBRID &&
	  !empty_statement_or_continue_p(body))
	body = step_guard_hybride(body);

      return body;
    }
}
