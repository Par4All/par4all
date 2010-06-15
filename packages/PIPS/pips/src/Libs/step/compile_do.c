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

// for :  string text_to_string(text t)...
#include "icfg.h"
#include "graph.h"
#include "ricedg.h"


/*############################################################################################*/

static list step_build_InitInterlaced(entity mpi_module, list l_interlaced)
{
  list block=NIL;
  
  FOREACH(REGION,r,l_interlaced)
    {
      entity e=region_entity(r);    // e correspond au tableau (ancienne entite (dans la fonction outlinee))
      pips_debug(2,"entity region : %s\n",entity_name(e));
      if (!(step_reduction_p(e) || step_private_p(e)))
	{
	  if (entity_scalar_p(e))
	    pips_user_warning("STEP : possible data race with : %s\n\n",entity_name(e));
	  else
	    block = CONS(STATEMENT, build_call_STEP_InitInterlaced(mpi_module, e), block);
	}
    }
  return block;
}

static list step_build_ComputeLoopSlices(entity mpi_module,list l_loop_data)
{
  list block=NIL;

  // une longueur >1 pourra etre envisage pour la clause "omp collapse"
  pips_assert("length(l_loop_data)=1", gen_length(l_loop_data)==1);

  /* 
     Creation de l'instruction 'CALL STEP_ComputeLoopSlices(I_L, I_U, loop_step, STEP_Size, MAX_NB_LOOPSLICES, STEP_I_LOOPSLICES)'
     creant un tableau step_i_loopslices contenant les tranches d'indices a traiter par chaque noeud
  */
  FOREACH(LOOP_DATA,data,l_loop_data)
    {
      entity i = loop_data_index(data);
      statement statmt = call_STEP_subroutine(RT_STEP_ComputeLoopSlices,
					      CONS(EXPRESSION, entity_to_expression(loop_data_lower(data)),
						   CONS(EXPRESSION, entity_to_expression(loop_data_upper(data)),
							CONS(EXPRESSION, int_to_expression(loop_data_step(data)),
							     CONS(EXPRESSION, step_local_size(mpi_module),
								  CONS(EXPRESSION, step_symbolic(STEP_MAX_NB_LOOPSLICES_NAME, mpi_module),
								       CONS(EXPRESSION, entity_to_expression(step_local_loopSlices(mpi_module,i)),
									    NIL)))))), type_undefined);
      statement_comments(statmt)=strdup("\n");
      block = CONS(STATEMENT, statmt, block);
    }
  return block;
}

static entity loop_SR_array(entity mpi_module, region reg)
{
  return step_local_SR(mpi_module, region_entity(reg), step_symbolic(STEP_MAX_NB_LOOPSLICES_NAME, mpi_module));
}

static statement build_new_loop_bounds(entity new_module, entity index)
{
  expression rank_expr_p1 = binary_intrinsic_expression(PLUS_OPERATOR_NAME,step_local_rank(new_module),make_expression_1());  

  // creation  et declaration des variables locales STEP_I_LOW, STEP_I_UP
  entity loopSlice = step_local_loopSlices(new_module,index);
  entity bound_low = step_local_loop_index(new_module, STEP_BOUNDS_LOW(index));
  entity bound_up = step_local_loop_index(new_module, STEP_BOUNDS_UP(index));


  // add statement : STEP_I_LOW = STEP_I_LOOPSLICES(I_SLICE_LOW, STEP_Rank + 1)
  // add statement : STEP_I_UP = STEP_I_LOOPSLICES(I_SLICE_UP, STEP_Rank + 1)
  expression expr_l = reference_to_expression(make_reference(loopSlice,
							     CONS(EXPRESSION, step_symbolic(STEP_INDEX_SLICE_LOW_NAME, new_module),
								  CONS(EXPRESSION,rank_expr_p1,NIL))));
  expression expr_u = reference_to_expression(make_reference(loopSlice,
							     CONS(EXPRESSION,step_symbolic(STEP_INDEX_SLICE_UP_NAME, new_module),
								  CONS(EXPRESSION,rank_expr_p1,NIL))));

  statement body =  make_block_statement(CONS(STATEMENT,make_assign_statement(entity_to_expression(bound_low), expr_l),
					      CONS(STATEMENT,make_assign_statement(entity_to_expression(bound_up), expr_u),NIL)));
  put_a_comment_on_a_statement(body,strdup("\nC     Where work is done...\n"));
  return body;
}


static statement build_SR_array(entity directive_module, entity mpi_module, loop_data data, step_analyses loop_analyse)
{ 
  entity loop_index=loop_data_index(data);
  entity loop_lower=loop_data_lower(data);
  entity loop_upper=loop_data_upper(data);
  entity low_index = step_local_loop_index(directive_module, STEP_INDEX_LOW_NAME(loop_index));
  entity up_index = step_local_loop_index(directive_module, STEP_INDEX_UP_NAME(loop_index));

  /*
    Remplace dans le systeme de contraintes les variables l et u de
    l'ancienne fonction outlinee par les variables i_l et i_u de la
    nouvelle fonction MPI 
  */
  list send_regions=NIL;
  FOREACH(REGION, reg, step_analyses_send(loop_analyse))
    {
      Psysteme sys = region_system(reg);
      sys = sc_variable_rename(sys,(Variable)loop_lower,(Variable)low_index);
      sys = sc_variable_rename(sys,(Variable)loop_upper,(Variable)up_index);

      if(!(region_scalar_p(reg) || step_private_p(region_entity(reg))))
	send_regions=CONS(REGION,reg,send_regions);
    }

  statement SR_array = step_build_arrayRegion(mpi_module, send_regions, loop_SR_array, loop_index);

  // suppression des variables crees lors du renomage
  list *declaration_directive_module=&code_declarations(EntityCode(directive_module));
  FOREACH(REGION, reg, step_analyses_send(loop_analyse))
    {
      gen_remove(declaration_directive_module,low_index);
      gen_remove(declaration_directive_module,up_index);
    }

  return SR_array;
}

static statement build_mpi_before_loop(entity directive_module, entity mpi_module, list loop_data_l)
{
  pips_debug(1, "mpi_module = %p\n", mpi_module);
  pips_assert("loop_data",gen_length(loop_data_l)==1);
  loop_data data=LOOP_DATA(CAR(loop_data_l));
  step_analyses loop_analyse= load_global_step_analyses(directive_module);

  statement size_stmt = call_STEP_subroutine(RT_STEP_Get_size, CONS(EXPRESSION,step_local_size(mpi_module),NIL), type_undefined);
  statement rank_stmt = call_STEP_subroutine(RT_STEP_Get_rank, CONS(EXPRESSION,step_local_rank(mpi_module),NIL), type_undefined);

  step_private_before(directive_module);
  list InitReduction=step_reduction_before(directive_module,mpi_module);  
  list InitInterlaced=step_build_InitInterlaced(mpi_module,step_analyses_interlaced( loop_analyse));
  list ComputeLoopSlices=step_build_ComputeLoopSlices(mpi_module,loop_data_l);
  statement SR_array = build_SR_array(directive_module, mpi_module, data, loop_analyse);

  pips_debug(1, "End\n");
  return make_block_statement(CONS(STATEMENT,size_stmt,
				   CONS(STATEMENT,rank_stmt,
					gen_nconc(InitInterlaced,
						  gen_nconc(InitReduction,
							    gen_nconc(ComputeLoopSlices,
								      CONS(STATEMENT,SR_array,NIL)))))));
}

static statement build_mpi_after_loop(entity directive_module, entity mpi_module)
{
  step_analyses loop_analyse = load_global_step_analyses(directive_module);

  // communications OUT
  pips_debug(1,"mpi_module = %p\n", mpi_module);
  list seqlist_all2all = NIL;
  int nb_communication_max = gen_length(step_analyses_send(loop_analyse));

  if (nb_communication_max != 0)
    {
      FOREACH(REGION, r, step_analyses_send(loop_analyse))
	{
	  if(!(region_scalar_p(r) || step_private_p(region_entity(r))))
	    {
	      expression nb_region = step_local_size(mpi_module);
	      boolean interlaced_p = FALSE;
	      entity array = region_entity(r);
	      list interlaced_l = gen_copy_seq(step_analyses_interlaced(loop_analyse));
	      list interlaced_ll = interlaced_l;
	      pips_debug(2,"region %s\n",entity_name(array));
	      while(!interlaced_p && !ENDP(interlaced_l))
		{
		  interlaced_p = array==region_entity(REGION(CAR(interlaced_l)));
		  POP(interlaced_l);
		}
	      seqlist_all2all = CONS(STATEMENT, build_call_STEP_AlltoAllRegion(mpi_module, nb_communication_max, interlaced_p, array,nb_region),
				     seqlist_all2all);
	      gen_free_list(interlaced_ll);
	    }
	}
    }

  statement commRequest = step_handle_comm_requests(mpi_module, seqlist_all2all, nb_communication_max);
  list commReduction = step_reduction_after(directive_module);
  step_private_after();

  return make_block_statement(CONS(STATEMENT,commRequest,commReduction));
}


// pour eviter un bug de gcc ...
static void fixe_gcc_bug(entity new_module)
{
  string new_name=entity_local_name(new_module);

  entity area = FindOrCreateEntity(new_name, DYNAMIC_AREA_LOCAL_NAME);
  entity index=find_ith_formal_parameter(new_module,1);
  int offset = add_variable_to_area(area,index);
  storage strg=entity_storage(index);
  entity_storage(index)=make_storage(is_storage_ram,make_ram(new_module,area,offset,NIL));
  
  string index_name=strdup(concatenate(entity_local_name(index),"_DUMMY",NULL));
  entity index_ = FindOrCreateEntity(new_name,index_name);
  entity_type(index_)=MakeTypeVariable(make_basic_int(DefaultLengthOfBasic(is_basic_int)), NIL);
  entity_storage(index_)=strg;
  code_declarations(value_code(entity_initial(new_module)))=CONS(ENTITY,index_,code_declarations(value_code(entity_initial(new_module))));
  free(index_name);
}


static statement build_mpi_loop(entity new_module, list loop_data_l, statement work)
{
  pips_assert("work is sequence",instruction_sequence_p(statement_instruction(work)));
  list block=sequence_statements(instruction_sequence(statement_instruction(work)));
  statement stat=STATEMENT(CAR(block));
  pips_assert("is loop",instruction_loop_p(statement_instruction(stat)));

  pips_assert("length(loop_data_l)=1", gen_length(loop_data_l)==1);
  entity loop_index = loop_data_index(LOOP_DATA(CAR(loop_data_l)));
  
  // substitution des bornes de boucles
  loop l=instruction_loop(statement_instruction(stat));
  range r= loop_range(l);
  range_lower(r) = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_LOW(loop_index)));
  range_upper(r) = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_UP(loop_index)));


  return make_block_statement(CONS(STATEMENT, build_new_loop_bounds(new_module, loop_index), // Association rank -> work_chunk
					     CONS(STATEMENT,work,NIL)));
}


statement step_compile_do(int step_transformation, entity new_module, statement work)
{
  entity directive_module=get_current_module_entity();
  directive d=load_global_directives(directive_module);

  fixe_gcc_bug(new_module);

  if (step_transformation == STEP_TRANSFORMATION_OMP )
    {
      add_pragma_entity_to_statement(work, directive_module);
      return work;
    }
  else
    {
      list loop_data_l;
      switch(type_directive_tag(directive_type(d)))
	{
	case is_type_directive_omp_parallel_do:
	  loop_data_l = type_directive_omp_parallel_do(directive_type(d));
	  break;
	case is_type_directive_omp_do:
	  loop_data_l = type_directive_omp_do(directive_type(d));
	  break;
	default:
	  pips_assert("is loop directive",false);
	}

      if (step_transformation == STEP_TRANSFORMATION_HYBRID)
	add_pragma_entity_to_statement(work, directive_module);

      statement before_work = build_mpi_before_loop(directive_module, new_module, loop_data_l);
      statement after_work = step_guard_hybride(build_mpi_after_loop(directive_module, new_module));
      return make_block_statement(CONS(STATEMENT, before_work,
				       CONS(STATEMENT, build_mpi_loop(new_module, loop_data_l, work),
					    CONS(STATEMENT, after_work,
						 NIL))));
    }
}
