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

list arguments=NIL;
list references=NIL;
int num_critical=0;


static statement build_call_STEP_Critical_set_CurrentUptodateScalar(entity scalar)
{
   
  list arglist = CONS(EXPRESSION, entity_to_expression(scalar), NIL);
  return call_STEP_subroutine(RT_STEP_Critical_set_CurrentUptodateScalar, arglist,entity_type(scalar));
}

static statement build_call_STEP_Critical_set_CurrentUptodateRegion(entity array)
{
   
  expression expr_array = entity_to_expression(array);
  list arglist = CONS(EXPRESSION, expr_array,NIL);
  return call_STEP_subroutine(RT_STEP_Critical_set_CurrentUptodateRegion, arglist, type_undefined);
}

static statement build_call_STEP_Critical_get_CurrentUptodateScalar(entity scalar)
{
   
  list arglist = CONS(EXPRESSION, entity_to_expression(scalar), NIL);
  return call_STEP_subroutine(RT_STEP_Critical_get_CurrentUptodateScalar, arglist,entity_type(scalar));
}

static statement build_call_STEP_Critical_get_CurrentUptodateRegion(entity array)
{
   
  expression expr_array = entity_to_expression(array);
  list arglist = CONS(EXPRESSION, expr_array,NIL);
  return call_STEP_subroutine(RT_STEP_Critical_get_CurrentUptodateRegion, arglist, type_undefined);
}

static statement build_call_STEP_Critical_FinalUptodateScalar(entity scalar)
{
   
  list arglist = CONS(EXPRESSION, entity_to_expression(scalar), NIL);
  return call_STEP_subroutine(RT_STEP_Critical_FinalUptodateScalar, arglist,entity_type(scalar));
}

static statement build_call_STEP_Critical_FinalUptodateRegion(entity array)
{
   
  expression expr_array = entity_to_expression(array);
  list arglist = CONS(EXPRESSION, expr_array,NIL);
  return call_STEP_subroutine(RT_STEP_Critical_FinalUptodateRegion, arglist, type_undefined);
}

static entity critical_SR_array(entity mpi_module, region reg)
{
  entity array = region_entity(reg);
  return step_local_arrayRegions(STEP_SR_NAME(array), mpi_module, array, expression_undefined);
}

static statement build_mpi_before_critical(entity directive_module,entity mpi_module)
{

  
  statement request = call_STEP_subroutine(RT_STEP_Request, CONS(EXPRESSION,make_integer_constant_expression(num_critical),NIL), type_undefined);
  //section critique nommée est non pas encore prise en compte--> on considèrent les constructions de critical comme une seule section critique
  //num_critical++; 
  
  step_analyses critical_analyse=load_global_step_analyses(directive_module);

  statement begin = call_STEP_subroutine(RT_STEP_Begin, CONS(EXPRESSION,step_symbolic(STEP_CRITICAL_NAME,mpi_module),NIL), type_undefined);

// communications IN
  pips_debug(1, "mpi_module = %p\n", mpi_module);
  statement flush;
  FOREACH(REGION, r, step_analyses_send(critical_analyse))
    {
      //statement flush;//stmt;
      entity array = region_entity(r);
      pips_debug(2, "region %s\n", entity_name(array));
      
      if(!region_scalar_p(r))
	flush = build_call_STEP_Critical_get_CurrentUptodateRegion(array);
      else
	flush = build_call_STEP_Critical_get_CurrentUptodateScalar(array);
      
    }

  //Calcul des regions SEND
  list send_regions=NIL;
  FOREACH(REGION,reg,step_analyses_send(critical_analyse))
    {
      if(!region_scalar_p(reg))
	send_regions=CONS(REGION,reg,send_regions);
    }
  statement send_regions_stmt = step_build_arraybounds(mpi_module, send_regions,critical_SR_array, true);
  //statement comment;
  put_a_comment_on_a_statement(send_regions_stmt, strdup("\nC     Compute send regions...\n"));
  
  statement before_critical= make_block_statement(CONS(STATEMENT,begin,
				CONS(STATEMENT,send_regions_stmt,  
				CONS(STATEMENT, request,
				   CONS(STATEMENT,flush,NIL)))));
  before_critical = step_guard_hybride(before_critical); 
  return before_critical;
 }

static statement build_mpi_critical(statement work)
{
  statement s_begin = make_continue_statement(entity_empty_label());
  statement_comments(s_begin) = strdup("\n!$omp critical\n");
  put_a_comment_on_a_statement(work, strdup("\nC     Where work is done...\n"));
  statement s_end = make_continue_statement(entity_empty_label());
  statement_comments(s_end) = strdup("\n!$omp end critical\n!$omp barrier\n\n");
  return  make_block_statement(CONS(STATEMENT,s_begin,CONS(STATEMENT, work,CONS(STATEMENT,s_end, NIL))));
  //return work;
}

static statement build_mpi_after_critical(entity directive_module,entity mpi_module)
{
  list arglist=NIL;
  
  statement next_process = call_STEP_subroutine(RT_STEP_Get_Nextprocess, arglist, type_undefined);
  
  statement release = call_STEP_subroutine(RT_STEP_Release,arglist, type_undefined);
  
  step_analyses critical_analyse = load_global_step_analyses(directive_module);
  // communications OUT
  pips_debug(1, "mpi_module = %p\n", mpi_module);
  list flush_list = NIL;
  
  FOREACH(REGION, r, step_analyses_send(critical_analyse))
    {
      statement stmt;
      entity array = region_entity(r);
      pips_debug(2, "region %s\n", entity_name(array));
      
      if(!region_scalar_p(r))
	stmt = build_call_STEP_Critical_set_CurrentUptodateRegion(array);
      else
	stmt = build_call_STEP_Critical_set_CurrentUptodateScalar(array);
      
      flush_list = CONS(STATEMENT, stmt, flush_list);
    }

  statement end = call_STEP_subroutine(RT_STEP_End, CONS(EXPRESSION,step_symbolic(STEP_CRITICAL_NAME,mpi_module),NIL), type_undefined);

  statement flush = critical_build_call_STEP_WaitAll(flush_list);
  
  pips_debug(1, "End\n");
  
 statement after_critical= make_block_statement(CONS(STATEMENT,next_process,CONS(STATEMENT, flush,CONS(STATEMENT,release, 
				   CONS(STATEMENT, end, NIL)))));
  after_critical = step_guard_hybride(after_critical); 
  return after_critical;
}


statement step_compile_critical(int step_transformation, entity new_module, statement work)
{
	//gen_recurse(work, statement_domain, compile_statement_filter, gen_null);
	    
  entity directive_module=get_current_module_entity();
  directive d=load_global_directives(directive_module);
  pips_assert("is critical directive",type_directive_omp_critical_p(directive_type(d)));
  
  if (step_transformation == STEP_TRANSFORMATION_OMP)
    {
      add_pragma_entity_to_statement(work, directive_module);
      return work;
    }
  else
    {
      statement before_work = build_mpi_before_critical(directive_module, new_module);
      statement after_work = build_mpi_after_critical(directive_module, new_module);
      statement body = make_block_statement(CONS(STATEMENT, before_work,
						 CONS(STATEMENT, build_mpi_critical(work),
						      CONS(STATEMENT, after_work,NIL))));
      
      return body;
    }
}

//A la fin, la dernière reception mise à jour globale:

statement step_compile_critical_update(int step_transformation, entity mpi_module)
{
  statement work;
  statement stmt;
  entity directive_module=get_current_module_entity();
  directive d=load_global_directives(directive_module);
  pips_assert("is critical directive",type_directive_omp_critical_p(directive_type(d)));
  
  statement barrier=call_STEP_subroutine(RT_STEP_Barrier,NIL,type_undefined);

  step_analyses critical_analyse = load_global_step_analyses(directive_module);
  pips_debug(1, "mpi_module = %p\n", mpi_module);
  FOREACH(REGION, r, step_analyses_send(critical_analyse))
    {
      //statement stmt;
      entity array = region_entity(r);
      pips_debug(2, "region %s\n", entity_name(array));
      
      if(!region_scalar_p(r))
	stmt = build_call_STEP_Critical_FinalUptodateRegion(array);
      else
	stmt = build_call_STEP_Critical_FinalUptodateScalar(array);
      
      //flush_list = CONS(STATEMENT, stmt, flush_list);
    }
  //statement flush = critical_build_call_STEP_WaitAll(flush_list);
  	
  pips_debug(1, "End\n");
  work =  make_block_statement(CONS/*(STATEMENT,work,CONS*/(STATEMENT,barrier,CONS(STATEMENT, /*flush*/stmt,NIL)))/*)*/;
  work = step_guard_hybride(work);
  return work;
}

