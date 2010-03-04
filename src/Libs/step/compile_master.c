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

static statement build_call_STEP_MastertoAllScalar(entity scalar)
{
  /* subroutine STEP_OnetoAllScalar_I(scalar,
    &     algorithm)
  */
  expression expr_scalar = entity_to_expression(scalar);
  expression expr_algorithm = int_to_expression(0);
  list arglist = CONS(EXPRESSION,expr_scalar,
		      CONS(EXPRESSION,expr_algorithm,NIL));
  return call_STEP_subroutine(strdup(concatenate(RT_STEP_MasterToAllScalar,step_type_suffix(scalar),NULL)),arglist);
}

static statement build_call_STEP_MastertoAllRegion(entity array)
{
  /*subroutine STEP_OnetoAllRegion_I(dim,
     &     region,size,array,
     &     algorithm)   
  */
  entity array_region = load_send_region_entities(array);
  expression expr_dim = copy_expression(dimension_upper(DIMENSION(gen_nth(1,variable_dimensions(type_variable(entity_type(array_region)))))));
  expression expr_region = entity_to_expression(array_region);
  expression expr_origine = make_expression(make_syntax_reference(make_reference(array_region,
										 CONS(EXPRESSION,entity_to_expression(step_i_slice_low),
										      CONS(EXPRESSION,int_to_expression(1),NIL)))),
					    normalized_undefined);
  expression expr_size = make_call_expression(step_sizeRegion,CONS(EXPRESSION,copy_expression(expr_dim),
								   CONS(EXPRESSION,expr_origine,NIL)));
  expression expr_array = entity_to_expression(array);
  expression expr_max_nb_request = entity_to_expression(step_max_nb_request);
  expression expr_requests = entity_to_expression(step_requests);
  expression expr_nb_request = entity_to_expression(step_nb_request);
  expression expr_algorithm = int_to_expression(0);
 
  list arglist = CONS(EXPRESSION,expr_array,
		      CONS(EXPRESSION,expr_dim,
			   CONS(EXPRESSION,expr_region,
				CONS(EXPRESSION,expr_size,
				     CONS(EXPRESSION,expr_array,
					  CONS(EXPRESSION,expr_max_nb_request,
					       CONS(EXPRESSION,expr_requests,
						    CONS(EXPRESSION,expr_nb_request,
							 CONS(EXPRESSION,expr_algorithm,NIL)))))))));
  return call_STEP_subroutine(strdup(concatenate(RT_STEP_MasterToAllRegion,step_type_suffix(array),NULL)),arglist);
}

static void step_create_mpi_before_master(step_region_analyse master_analyse, entity mpi_module)
{
  statement assigne_region = make_block_statement(NIL);
  step_add_parameter(mpi_module);

  //Calcul des regions SEND
  MAP(REGION,r,{
      if(!region_scalar_p(r))
	{
	  entity send=region_entity(r);
	  store_or_update_send_region_entities(region_entity(r),
					       step_create_region_array(mpi_module,strdup(STEP_SR_NAME(send)),send,FALSE));
	  insert_statement(assigne_region,build_assigne_region0(0,r,load_send_region_entities(region_entity(r))),FALSE);
	}
    },step_region_analyse_send(master_analyse));
  step_seqlist = CONS(STATEMENT,assigne_region, step_seqlist);
}

static void step_call_outlined_master(entity directive_module)
{
  list arglist, exprlist;
  statement call_stmt;

  pips_debug(1, "original_module = %p\n", directive_module);

  exprlist=NIL;
  arglist=NIL;

  MAP(ENTITY,e,{
      entity ne = load_new_entities(e);
      pips_debug(2,"call : entity %s -> %s\n", entity_name(e), entity_name(ne));

      exprlist = CONS(EXPRESSION,entity_to_expression(ne),exprlist);
    },
    outline_data_formal(load_outline(directive_module)));
  exprlist=gen_nreverse(exprlist);
      
  call_stmt = make_stmt_of_instr(make_instruction_call(make_call(directive_module, gen_append(arglist,exprlist))));

  // if (STEP_RANK.EQ.0)
  // call ....
  step_seqlist = CONS(STATEMENT, make_stmt_of_instr(make_instruction_test(make_test(MakeBinaryCall(gen_find_tabulated(make_entity_fullname(TOP_LEVEL_MODULE_NAME, ".EQ."), entity_domain),entity_to_expr(MakeConstant(STEP_RANK_NAME,is_basic_string)), int_expr(0)),
										    call_stmt,
										    make_block_statement(NIL)))),
		      step_seqlist);
}

static void step_create_mpi_after_master(step_region_analyse step_analyse, entity mpi_module)
{
  // communications OUT
  pips_debug(1,"mpi_module = %p\n", mpi_module);
  list seqlist_one2all = NIL;
  entity requests_array = step_declare_requests_array(mpi_module,step_region_analyse_send(step_analyse));// effet de bord : initialisation de l'entite "step_nb_max_request" (constante symbolique)

  MAP(REGION,r,{
      statement stmt;
      entity array = region_entity(r);
      pips_debug(2,"region %s\n",entity_name(array));
      
      if(!region_scalar_p(r))
	stmt = build_call_STEP_MastertoAllRegion(array);
      else
	stmt = build_call_STEP_MastertoAllScalar(array);

      seqlist_one2all=CONS(STATEMENT,stmt,seqlist_one2all);
    },step_region_analyse_send(step_analyse));

  if (!ENDP(seqlist_one2all))
    code_declarations(EntityCode(mpi_module)) =gen_append(code_declarations(EntityCode(mpi_module)),
							  CONS(ENTITY,step_max_nb_request,CONS(ENTITY,step_requests,NIL)));

    step_seqlist = gen_append(step_handle_comm_requests(requests_array,seqlist_one2all),step_seqlist);

  pips_debug(1, "End\n");
}


entity step_create_mpi_master(entity directive_module)
{
  string new_name = step_find_new_module_name(directive_module,STEP_MPI_SUFFIX);
  entity mpi_module = make_empty_subroutine(new_name);
  statement statmt;
  step_region_analyse master_analyse = load_step_analyse_map(directive_module);
  /* 
     ajout des variables formelles pour la nouvelle fonction MPI
     (identiques a celles de la fonction outlinee
  */
  init_new_entities();
  init_old_entities();
  init_send_region_entities();

  step_add_formal_copy(mpi_module,outline_data_formal(load_outline(directive_module)));
  step_create_mpi_before_master(master_analyse, mpi_module);

  step_call_outlined_master(directive_module);

  statmt = make_continue_statement(entity_undefined);
  statement_comments(statmt)=strdup("\nC     Communicating data to other nodes\n");
  step_seqlist = CONS(STATEMENT, statmt, step_seqlist);

  step_create_mpi_after_master(master_analyse, mpi_module);
  step_seqlist = CONS(STATEMENT, make_return_statement(mpi_module), step_seqlist);

  close_new_entities();
  close_old_entities();
  close_send_region_entities();

  return mpi_module;
}
