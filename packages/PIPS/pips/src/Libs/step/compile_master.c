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

static statement build_call_STEP_MastertoAllScalar(entity module, expression expr_requests, entity scalar)
{
 /* subroutine STEP_OnetoAllScalar(scalar,
    &     step_max_nb_request, step_requests, step_nb_request,
    &     algorithm)
  */
  list arglist = CONS(EXPRESSION, entity_to_expression(scalar),
		      CONS(EXPRESSION, step_parameter_max_nb_request(module,expression_undefined),
			   CONS(EXPRESSION,expr_requests,
				CONS(EXPRESSION,step_local_nb_request(module),
				     CONS (EXPRESSION, step_symbolic(STEP_NONBLOCKING_NAME, module), NIL)))));
  return call_STEP_subroutine(RT_STEP_MasterToAllScalar, arglist,entity_type(scalar));
}

static statement build_call_STEP_MastertoAllRegion(entity module, expression expr_requests, entity array)
{
  /*subroutine STEP_OnetoAllRegion_I(array,
     &     dim,region,size,
     &     max_nb_request,requests,nb_request
     &     algorithm)   
  */
  entity array_region = step_local_SR(module,array, expression_undefined);
  expression expr_dim = copy_expression(dimension_upper(DIMENSION(gen_nth(1, variable_dimensions(type_variable(entity_type(array_region)))))));
  expression expr_region = entity_to_expression(array_region);
  expression expr_origine = reference_to_expression(make_reference(array_region,
								   CONS(EXPRESSION, step_symbolic(STEP_INDEX_SLICE_LOW_NAME, module),
									CONS(EXPRESSION, int_to_expression(1), NIL))));
  expression expr_size = step_function(RT_STEP_SizeRegion, CONS(EXPRESSION, copy_expression(expr_dim),
								CONS(EXPRESSION, expr_origine, NIL)));
  expression expr_array = entity_to_expression(array);
  expression expr_max_nb_request = step_parameter_max_nb_request(module,expression_undefined);
  expression expr_nb_request = step_local_nb_request(module);
  expression expr_algorithm = step_symbolic(STEP_NONBLOCKING_NAME, module);
 
  list arglist = CONS(EXPRESSION, expr_array,
		      CONS(EXPRESSION, expr_dim,
			   CONS(EXPRESSION, expr_region,
				CONS(EXPRESSION, expr_size,
				     CONS(EXPRESSION, expr_max_nb_request,
					  CONS(EXPRESSION, expr_requests,
					       CONS(EXPRESSION, expr_nb_request,
						    CONS(EXPRESSION, expr_algorithm, NIL))))))));
 return call_STEP_subroutine(RT_STEP_MasterToAllRegion, arglist, entity_type(array));
}


static entity master_SR_array(entity mpi_module, region reg)
{
  return step_local_SR(mpi_module,region_entity(reg),expression_undefined);
}

static statement build_mpi_before_master(entity directive_module,entity mpi_module)
{
  step_analyses master_analyse=load_global_step_analyses(directive_module);
  
  //Calcul des regions SEND
  list send_regions=NIL;
  FOREACH(REGION,reg,step_analyses_send(master_analyse))
    {
      if(!region_scalar_p(reg))
	send_regions=CONS(REGION,reg,send_regions);
    }
  statement rank_stmt = call_STEP_subroutine(RT_STEP_Get_rank, CONS(EXPRESSION,step_local_rank(mpi_module),NIL), type_undefined);
  return make_block_statement(CONS(STATEMENT,step_build_arrayRegion(mpi_module, send_regions, master_SR_array, entity_undefined),
				   CONS(STATEMENT,rank_stmt,NIL)));
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
  list seqlist_one2all = NIL;
  int nb_communication_max = gen_length(step_analyses_send(master_analyse));
  
  if (nb_communication_max != 0)
    {
      expression expr_requests = step_local_requests_array(mpi_module, int_to_expression(nb_communication_max));
      
      FOREACH(REGION, r, step_analyses_send(master_analyse))
	{
	  statement stmt;
	  entity array = region_entity(r);
	  pips_debug(2, "region %s\n", entity_name(array));
	  
	  if(!region_scalar_p(r))
	    stmt = build_call_STEP_MastertoAllRegion(mpi_module, copy_expression(expr_requests), array);
	  else
	    stmt = build_call_STEP_MastertoAllScalar(mpi_module, copy_expression(expr_requests), array);
	  
	  seqlist_one2all = CONS(STATEMENT, stmt, seqlist_one2all);
	}
      free_expression(expr_requests);
    }
  pips_debug(1, "End\n");

  return step_handle_comm_requests(mpi_module, seqlist_one2all, nb_communication_max);
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
      
      if (step_transformation == STEP_TRANSFORMATION_HYBRID)
	return step_guard_hybride(body);
      else
	return body;
    }
}
