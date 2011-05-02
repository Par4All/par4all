/* Copyright 2007, 2008 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif

#include "defines-local.h"
#include "effects-generic.h"
#include "effects-convex.h"

/*############################################################################################*/

static list step_build_ComputeLoopSlices(entity mpi_module, list l_loop_data)
{
  list block = NIL;

  // une longueur >1 pourra etre envisage pour la clause "omp collapse"
  pips_assert("length(l_loop_data)=1", gen_length(l_loop_data) == 1);

  /* 
     Creation de l'instruction 'CALL STEP_ComputeLoopSlices(I_L, I_U, loop_step, STEP_Size, MAX_NB_LOOPSLICES, STEP_I_LOOPSLICES)'
     creant un tableau step_i_loopslices contenant les tranches d'indices a traiter par chaque noeud
  */
  FOREACH(LOOP_DATA, data, l_loop_data)
    {
      expression expr_lower = entity_to_expression(loop_data_lower(data));
      expression expr_upper = entity_to_expression(loop_data_upper(data));
      expression expr_incr = int_to_expression(loop_data_step(data));
      expression expr_nb_workchunk = step_local_size(mpi_module);
      statement statmt = call_STEP_subroutine(RT_STEP_ComputeLoopSlices,
					      CONS(EXPRESSION, expr_lower,
						   CONS(EXPRESSION, expr_upper,
							CONS(EXPRESSION, expr_incr,
							     CONS(EXPRESSION, expr_nb_workchunk, NIL)))),
					      type_undefined);
      block = CONS(STATEMENT, statmt, block);
    }
  return block;
}

static entity loop_SR_array(entity mpi_module, region reg)
{
  entity array = region_entity(reg);
  return step_local_arrayRegions(STEP_SR_NAME(array), mpi_module, array, step_symbolic(STEP_MAX_NB_LOOPSLICES_NAME, mpi_module));
}

static entity loop_RR_array(entity mpi_module, region reg)
{
  entity array = region_entity(reg);
  return step_local_arrayRegions(STEP_RR_NAME(array), mpi_module, array, step_symbolic(STEP_MAX_NB_LOOPSLICES_NAME, mpi_module));
}

static list build_new_loop_bounds(entity new_module, entity index)
{
  /*
    Generation de :
    CALL STEP_GETLOOPBOUNDS(STEP_Rank, I_SLICE_LOW, I_SLICE_UP)
  */
  expression expr_rank = step_local_rank(new_module);
  expression expr_index_low = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_LOW(index)));
  expression expr_index_up = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_UP(index)));
  if (!fortran_module_p(get_current_module_entity()))
    {
      expr_rank = make_call_expression(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), CONS(EXPRESSION, expr_rank, NIL));
      expr_index_low = make_call_expression(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), CONS(EXPRESSION, expr_index_low, NIL));
      expr_index_up = make_call_expression(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), CONS(EXPRESSION, expr_index_up, NIL));
    }
  statement rank_statmt = call_STEP_subroutine(RT_STEP_Get_rank, CONS(EXPRESSION,expr_rank,NIL), type_undefined);
  expression expr_id_workchunk = step_local_rank(new_module);
  list args = CONS(EXPRESSION, expr_id_workchunk,
		   CONS(EXPRESSION, expr_index_low,
			CONS(EXPRESSION, expr_index_up, NIL)));
  statement statmt = call_STEP_subroutine(RT_STEP_GetLoopBounds, args, type_undefined);
  list body = CONS(STATEMENT, rank_statmt,
		   CONS(STATEMENT, statmt, NIL));
  
  put_a_comment_on_a_statement(rank_statmt, strdup("\nC     Where work is done...\n"));
  return body;
}


static list build_SR_array(entity directive_module, entity mpi_module, loop_data data, step_analyses loop_analyse)
{ 
  entity loop_index=loop_data_index(data);
  entity loop_lower=loop_data_lower(data);
  entity loop_upper=loop_data_upper(data);
  entity low_index = step_local_loop_index(directive_module, STEP_INDEX_LOW_NAME(loop_index));
  entity up_index = step_local_loop_index(directive_module, STEP_INDEX_UP_NAME(loop_index));
  list set_SR = NIL;
  list interlaced_array = step_analyses_interlaced(loop_analyse);

  /*
    Remplace dans le systeme de contraintes les variables l et u de
    l'ancienne fonction outlinee par les variables i_l et i_u de la
    nouvelle fonction MPI 
  */
  list send_regions=NIL;
  FOREACH(REGION, reg, step_analyses_send(loop_analyse))
    {
      Psysteme sys = region_system(reg);
      sys = sc_variable_rename(sys, (Variable)loop_lower, (Variable)low_index);
      sys = sc_variable_rename(sys, (Variable)loop_upper, (Variable)up_index);

      if(!(region_scalar_p(reg) ||
	   step_private_p(region_entity(reg)) ||
	   io_effect_entity_p(region_entity(reg))
	   ))
	send_regions=CONS(REGION, reg, send_regions);
    }

  statement SR_array = step_build_arrayRegion(mpi_module, send_regions, loop_SR_array, loop_index);
  if (!continue_statement_p(SR_array))
    insert_comments_to_statement(SR_array,
				 concatenate("\nC     Put array boundaries into SEND region arrays",
					     "\nC     First dimension: lower and upper bounds of each slice",
					     "\nC     Second dimension: for each dimension of the original array",
					     "\nC     Third dimension: store the boundaries of the local chunk.\n",NULL));
  
  FOREACH(REGION, reg, send_regions)
    {
      entity array = region_entity(reg);
      entity regions = loop_SR_array(mpi_module, reg);
      expression expr_nb_workchunk = step_local_size(mpi_module);
      bool is_interlaced=gen_in_list_p(region_entity(reg), interlaced_array);
      bool is_reduction=array_reduction_p(directive_module, array);
      set_SR=CONS(STATEMENT, build_call_STEP_set_send_region(array, expr_nb_workchunk, regions, is_interlaced, is_reduction),set_SR);
    }
  
  // suppression des variables crees lors du renommage
  list *declaration_directive_module = &code_declarations(EntityCode(directive_module));
  FOREACH(REGION, reg, step_analyses_send(loop_analyse))
    {
      gen_remove(declaration_directive_module, low_index);
      gen_remove(declaration_directive_module, up_index);
    }

  return CONS(STATEMENT, SR_array, gen_nreverse(set_SR));
}

static list build_RR_array(entity directive_module, entity mpi_module, loop_data data, step_analyses loop_analyse)
{ 
  entity loop_index = loop_data_index(data);
  entity loop_lower = loop_data_lower(data);
  entity loop_upper = loop_data_upper(data);
  entity low_index = step_local_loop_index(directive_module, STEP_INDEX_LOW_NAME(loop_index));
  entity up_index = step_local_loop_index(directive_module, STEP_INDEX_UP_NAME(loop_index));
  list set_RR = NIL;

  /*
    Remplace dans le systeme de contraintes les variables l et u de
    l'ancienne fonction outlinee par les variables i_l et i_u de la
    nouvelle fonction MPI 
  */
  list recv_regions = NIL;
  FOREACH(REGION, reg, step_analyses_recv(loop_analyse))
    {
      Psysteme sys = region_system(reg);
      sys = sc_variable_rename(sys, (Variable)loop_lower, (Variable)low_index);
      sys = sc_variable_rename(sys, (Variable)loop_upper, (Variable)up_index);

      if(!(region_scalar_p(reg) ||
	   step_private_p(region_entity(reg)) ||
	   io_effect_entity_p(region_entity(reg))
	   ))
	recv_regions=CONS(REGION, reg, recv_regions);
    }

  statement RR_array = step_build_arrayRegion(mpi_module, recv_regions, loop_RR_array, loop_index);
  if (!continue_statement_p(RR_array))
    insert_comments_to_statement(RR_array,
				 concatenate("\nC     Put array boundaries into RECV region arrays",
					     "\nC     First dimension: lower and upper bounds of each slice",
					     "\nC     Second dimension: for each dimension of the original array",
					     "\nC     Third dimension: store the boundaries of the local chunk.\n",NULL));
  FOREACH(REGION, reg, recv_regions)
    {
      entity array = region_entity(reg);
      entity regions = loop_RR_array(mpi_module, reg);
      expression expr_nb_workchunk = step_local_size(mpi_module);
      set_RR=CONS(STATEMENT, build_call_STEP_set_recv_region(array, expr_nb_workchunk, regions), set_RR);
    }

  // suppression des variables crees lors du renomage
  list *declaration_directive_module = &code_declarations(EntityCode(directive_module));
  FOREACH(REGION, reg, step_analyses_recv(loop_analyse))
    {
      gen_remove(declaration_directive_module, low_index);
      gen_remove(declaration_directive_module, up_index);
    }

  return CONS(STATEMENT, RR_array, gen_nreverse(set_RR));
}


static statement build_mpi_before_loop(entity directive_module, entity mpi_module, list loop_data_l, bool parallel_do)
{
  pips_debug(1, "mpi_module = %p\n", mpi_module);
  pips_assert("loop_data",gen_length(loop_data_l) == 1);
  loop_data data = LOOP_DATA(CAR(loop_data_l));
  step_analyses loop_analyse = load_global_step_analyses(directive_module);

  statement begin = call_STEP_subroutine(RT_STEP_Begin, CONS(EXPRESSION,step_symbolic(parallel_do?STEP_PARALLELDO_NAME:STEP_DO_NAME,mpi_module),NIL), type_undefined);
  expression expr_comm_size=step_local_size(mpi_module);
  if (!fortran_module_p(get_current_module_entity()))
    {
      expr_comm_size = make_call_expression(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), CONS(EXPRESSION, expr_comm_size, NIL));
    }
  
  statement size_stmt = call_STEP_subroutine(RT_STEP_Get_Commsize, CONS(EXPRESSION,expr_comm_size,NIL), type_undefined);

  step_private_before(directive_module);

  list init_arrayregion = NIL;
  if (parallel_do)
    init_arrayregion = CONS(STATEMENT, step_share_before(directive_module, mpi_module), init_arrayregion);
  list InitReduction = step_reduction_before(directive_module,mpi_module);  
  list ComputeLoopSlices = CONS(STATEMENT, size_stmt, step_build_ComputeLoopSlices(mpi_module, loop_data_l));
  list SR_array = build_SR_array(directive_module, mpi_module, data, loop_analyse);
  list body = CONS(STATEMENT, begin,
		   gen_nconc(init_arrayregion,
			     gen_nconc(InitReduction, ComputeLoopSlices)));

  if (get_bool_property("STEP_OPTIMIZED_COMMUNICATIONS"))
    {
      list RR_array = build_RR_array(directive_module, mpi_module, data, loop_analyse);
      list flush_list = NIL;

      /* FSC: a commenter */
      /* AM: l'utilisation de la variable 'nb_communication_max' a pour origine la déclaration 
	 dans le code genere du tableau contenant l'ensemble des requettes de communication MPI.
	 Dans le cas des communications optimisees, ces handlers seront stockes au niveau de la runtime.
	 Cette variable ne sera donc plus utile. cf 'build_mpi_after_loop' et 'step_handle_comm_requests'.
       */
      int nb_communication_max = gen_length(step_analyses_recv(loop_analyse));
      
      if (nb_communication_max != 0)
	{ 
	  FOREACH(REGION, reg, step_analyses_recv(loop_analyse))
	    {
	      if(!(region_scalar_p(reg) ||
		   step_private_p(region_entity(reg)) ||
		   io_effect_entity_p(region_entity(reg))
		   ))
		{
		  /* FSC: pourquoi ne fait-on pas un test ici pour
		     savoir si ce module donne lieu à des
		     communications optimisées? */
		  /* AM: l'ensemble des communications generees en debut de construction OpenMP
		     (suite à l'existance de regions RECV) font l'objet de communication optimisees.
		     Les communications non optimisees ayant lieu elles en fin de construction openMP
		     (cf build_mpi_after_loop)
		   */

		  bool is_optimized = true;
		  bool is_interlaced = false;
		  entity array = region_entity(reg);
		  pips_debug(2, "region %s\n",entity_name(array));
		  flush_list = CONS(STATEMENT,build_call_STEP_AllToAll(mpi_module, array, is_optimized, is_interlaced), flush_list);
		}
	    }
	}

      statement flush = build_call_STEP_WaitAll(flush_list);
      body = gen_nconc(body,
		       gen_nconc(RR_array, CONS(STATEMENT, flush, NIL)));
    }

  statement guarded =  make_block_statement(gen_nconc(body, SR_array));
  if (!empty_statement_or_continue_p(guarded))
    guarded = step_guard_hybride(guarded);
  
  pips_debug(1, "End\n");
  return guarded;
}

static statement build_mpi_after_loop(entity directive_module, entity mpi_module, bool parallel_do)
{
  step_analyses loop_analyse = load_global_step_analyses(directive_module);

  // communications OUT
  pips_debug(1,"mpi_module = %p\n", mpi_module);
  list flush_list = NIL;
  int nb_communication_max = gen_length(step_analyses_send(loop_analyse));
  list interlaced_l = step_analyses_interlaced(loop_analyse);
  map_entity_bool optimizable = step_analyses_optimizable(loop_analyse);

  if (nb_communication_max != 0)
    {
      FOREACH(REGION, r, step_analyses_send(loop_analyse))
	{
	  if(!(region_scalar_p(r) || step_private_p(region_entity(r))))
	    {
	      entity array = region_entity(r);
	      bool interlaced_p = gen_in_list_p(array, interlaced_l);
	      bool optimize_com_p = (get_bool_property("STEP_OPTIMIZED_COMMUNICATIONS") &&
				     step_com_optimize_p(optimizable, array, directive_module));
	      bool ar_reduction=array_reduction_p(directive_module, array);
	      
	      pips_debug(2,"region %s\n",entity_name(array));
	      
	      if(ar_reduction==false)
		{		
		  flush_list = CONS(STATEMENT,build_call_STEP_AllToAll(mpi_module, array, optimize_com_p, interlaced_p) ,flush_list);
		}
	    }
	}
    }
  statement end = call_STEP_subroutine(RT_STEP_End, CONS(EXPRESSION,step_symbolic(parallel_do?STEP_PARALLELDO_NAME:STEP_DO_NAME,mpi_module),NIL), type_undefined);
  statement flush = build_call_STEP_WaitAll(flush_list);
  list commReduction = step_reduction_after(directive_module);
  step_private_after();

  return make_block_statement(CONS(STATEMENT, flush, 
				   gen_nconc(commReduction,
					     CONS(STATEMENT, end, NIL))));
}

static statement build_mpi_loop(entity new_module, list loop_data_l, statement work, int critical_p)
{
  pips_assert("work is sequence", instruction_sequence_p(statement_instruction(work)));
  list block = sequence_statements(instruction_sequence(statement_instruction(work)));
  statement stat = STATEMENT(CAR(block));

  if (continue_statement_p(stat)) // cas d'un continue inserer en C pour porter les declarations locales
    stat=STATEMENT(CAR(CDR(block)));

  pips_assert("is loop",instruction_loop_p(statement_instruction(stat)));
  pips_assert("length(loop_data_l)=1", gen_length(loop_data_l) == 1);
  entity loop_index = loop_data_index(LOOP_DATA(CAR(loop_data_l)));
  
  step_local_loop_index(new_module, entity_user_name(loop_index));

  // substitution des bornes de boucles
  loop l = instruction_loop(statement_instruction(stat));
  range r = loop_range(l);
  range_lower(r) = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_LOW(loop_index)));
  range_upper(r) = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_UP(loop_index)));
  
  if(critical_p==1)
    {
      list arglist = CONS(EXPRESSION, range_upper(r), CONS(EXPRESSION, range_lower(r), NIL));
      statement check_n_threads = call_STEP_subroutine(RT_STEP_Critical_Check_n_threads, arglist, type_undefined);
      return make_block_statement(gen_nconc(build_new_loop_bounds(new_module, loop_index), // Association rank -> work_chunk
					    CONS(STATEMENT, check_n_threads, CONS(STATEMENT, work, NIL))));
    }
  else
    return make_block_statement(gen_nconc(build_new_loop_bounds(new_module, loop_index), // Association rank -> work_chunk
					  CONS(STATEMENT, work, NIL)));

 
}


statement step_compile_do(int step_transformation, entity new_module, statement work, statement work_critical)
{
  entity directive_module = get_current_module_entity();
  directive d = load_global_directives(directive_module);

  if (step_transformation == STEP_TRANSFORMATION_OMP )
    {
      add_pragma_entity_to_statement(work, directive_module);
      return work;
    }
  else
    {
      bool parallel_do;
      list loop_data_l;
      switch(type_directive_tag(directive_type(d)))
	{
	case is_type_directive_omp_parallel_do:
	  loop_data_l = type_directive_omp_parallel_do(directive_type(d));
	  parallel_do = true;
	  break;
	case is_type_directive_omp_do:
	  loop_data_l = type_directive_omp_do(directive_type(d));
	  parallel_do = false;
	  break;
	default:
	  pips_assert("is loop directive", false);
	}

      if (step_transformation == STEP_TRANSFORMATION_HYBRID)
	add_pragma_entity_to_statement(work, directive_module);

      statement before_work = build_mpi_before_loop(directive_module, new_module, loop_data_l, parallel_do);
      statement after_work = build_mpi_after_loop(directive_module, new_module, parallel_do);
      if (step_transformation == STEP_TRANSFORMATION_HYBRID &&
	  !empty_statement_or_continue_p(after_work))
	after_work = step_guard_hybride(after_work);
      
      if(work_critical)  	
      		return make_block_statement(CONS(STATEMENT, before_work,
						 CONS(STATEMENT, build_mpi_loop(new_module, loop_data_l, work, 1),
					CONS(STATEMENT, work_critical,
					    CONS(STATEMENT, after_work,
						 NIL)))));
      else
		return make_block_statement(CONS(STATEMENT, before_work,
						 CONS(STATEMENT, build_mpi_loop(new_module, loop_data_l, work, 0),
					    CONS(STATEMENT, after_work,
						 NIL))));

    }
}
