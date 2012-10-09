/* Copyright 2007-2012 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"

#include "accel-util.h" // for outliner_patch_parameters, outliner_file, ...
#include "semantics.h" // for expression_and_precondition_to_integer_interval
#include "pipsmake.h" // for compilation_unit_of_module

static string step_head_hook(entity __attribute__ ((unused)) e)
{
  return strdup(concatenate
		("      implicit none\n",
		 "      include \"STEP.h\"\n", NULL));
}

static void loopbounds_substitution(entity new_module, loop loop_stmt)
{
  entity index = loop_index(loop_stmt);
  range r = loop_range(loop_stmt);

  range_lower(r) = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_LOW(index)));
  range_upper(r) = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_UP(index)));
}

static statement compile_loopslices(entity new_module, loop loop_stmt)
{
  statement stmt;
  statement commsize_stmt;
  statement loopslice_stmt;
  pips_debug(1, "begin new_module = %p, loop_stmt = %p\n", new_module, loop_stmt);

  commsize_stmt = generate_call_get_commsize(new_module);
  loopslice_stmt = generate_call_compute_loopslices(new_module, loop_stmt);

  stmt = make_block_statement(make_statement_list(commsize_stmt, loopslice_stmt));
  pips_debug(1, "end\n");
  return stmt;
}

static statement compile_loopbounds(entity new_module, loop loop_stmt, statement work_stmt)
{
  statement stmt;
  statement rank_stmt; 
  statement get_loopbounds_stmt;
  pips_debug(1, "begin\n");


  rank_stmt = generate_call_get_rank(new_module);
  get_loopbounds_stmt = generate_call_get_rank_loopbounds(new_module, loop_stmt);

  stmt = make_block_statement(make_statement_list(rank_stmt, get_loopbounds_stmt, work_stmt));

  pips_debug(1, "end\n");
  return stmt;
}

static void compile_reduction(set reductions_l[STEP_UNDEF_REDUCE], statement before_stmt, statement after_stmt)
{
  int op;
  string op_name[STEP_UNDEF_REDUCE]={STEP_PROD_REDUCE_NAME, STEP_MAX_REDUCE_NAME, STEP_MIN_REDUCE_NAME, STEP_SUM_REDUCE_NAME};

  pips_debug(1, "begin\n");

  pips_assert("block before", !statement_undefined_p(before_stmt) && statement_block_p(before_stmt));
  pips_assert("block after", !statement_undefined_p(after_stmt) && statement_block_p(after_stmt));

  for (op = 0; op < STEP_UNDEF_REDUCE; op++)
    if (!set_empty_p(reductions_l[op]))
      {
	FOREACH(ENTITY, variable, set_to_sorted_list(reductions_l[op], (gen_cmp_func_t)compare_entities))
	  {
	    expression variable_expr, variable_type_expr;
	    statement reduction_init_stmt;
	    /*
	      Generation of
	      STEP_INITREDUCTION(&sum, STEP_SUM, STEP_INTEGER4);
	     */

	    check_entity_step_type(variable);

	    variable_expr = entity_to_expression(variable);
	    variable_type_expr = entity_to_expression(step_type(variable));
	    reduction_init_stmt = call_STEP_subroutine2(RT_STEP_initreduction, get_expression_addr(variable_expr), entity_to_expression(MakeConstant(op_name[op], is_basic_string)), variable_type_expr, NULL);

	    insert_statement(before_stmt,
			     reduction_init_stmt, false);

	    /*
	      Generation of
	      STEP_REDUCTION(&sum);
	    */

	    /*
	    list arglist = CONS(EXPRESSION, variable_expr,
			   CONS(EXPRESSION, entity_to_expression(MakeConstant(op_name[op], is_basic_string)),
				NIL));
	    insert_statement(before_stmt,
			     call_STEP_subroutine(RT_STEP_initreduction, arglist, variable), false);
	    */
			     
	    insert_statement(after_stmt,
			     call_STEP_subroutine2(RT_STEP_reduction, get_expression_addr(variable_expr), NULL), false);
	  }
      }

  pips_debug(1, "end\n");
}

/*
  IMPORTANT: add constraints I_L<=i<=I_U.

  (see add_index_range_conditions in semantics/loop.c)
*/
static void loop_basic_workchunk_to_workchunk(statement stmt, list *recv_l, list *send_l)
{
  pips_debug(1, "begin\n");
  list dir_block = statement_block(stmt);
  loop l = statement_loop(STATEMENT(CAR(dir_block)));
  entity index = loop_index(l);
  range r = loop_range(l);

  /* find the real upper and lower bounds */
  int incr_lb = 0;
  int incr_ub = 0;
  expression e_incr = range_increment(r);
  transformer loop_prec = make_transformer(NIL, make_predicate((Psysteme)sc_rn((Pbase)vect_new((Variable)index, VALUE_ONE))));
  // is the loop increment numerically known? Is its sign known?
  expression_and_precondition_to_integer_interval(e_incr, loop_prec, &incr_lb, &incr_ub);

  int incr = 0;
  if (incr_lb == incr_ub) {
    if (incr_lb == 0) {
      pips_user_error("Illegal null increment\n");
    }
    else
      incr = incr_lb;
  }
  else if (incr_lb >= 1) {
    incr = 1;
  }
  else if (incr_ub <= -1) {
    incr = -1;
  }
  entity e_lb = step_local_loop_index(get_current_module_entity(), strdup(STEP_BOUNDS_LOW(index)));
  entity e_ub = step_local_loop_index(get_current_module_entity(), strdup(STEP_BOUNDS_UP(index)));

  // to check later in update_referenced_entities
  set_add_element(step_created_entity, step_created_entity, e_lb);
  set_add_element(step_created_entity, step_created_entity, e_ub);

  if (incr < 0)
    {
      entity tmp = e_lb;
      e_lb = e_ub;
      e_ub = tmp;
    }
  /*
    Add constraints
    Projection
  */
  Psysteme sc_bounds = sc_new();
  // contrainte lb - index <= 0
  insert_ineq_begin_sc(sc_bounds, contrainte_make(vect_make(VECTEUR_NUL, (Variable)e_lb, VALUE_ONE, (Variable)index, VALUE_MONE, TCST, VALUE_ZERO)));
  // contrainte index - ub <= 0
  insert_ineq_begin_sc(sc_bounds, contrainte_make(vect_make(VECTEUR_NUL, (Variable)index, VALUE_ONE, (Variable)e_ub, VALUE_MONE, TCST, VALUE_ZERO)));
  sc_creer_base(sc_bounds);

  *send_l = all_regions_sc_append(*send_l, sc_bounds, true);
  *recv_l = all_regions_sc_append(*recv_l, sc_bounds, true);
  project_regions_along_variables(*send_l, CONS(entity, index, NIL));
  project_regions_along_variables(*recv_l, CONS(entity, index, NIL));

  FOREACH(REGION, reg, *send_l)
    {
      rectangularization_region(reg);
    }
  FOREACH(REGION, reg, *recv_l)
    {
      rectangularization_region(reg);
    }

  ifdebug(1)
    {
      pips_debug(1, "SEND after projection and rectangularization : \n");
      print_rw_regions(*send_l);
      pips_debug(1, "RECV after projection and rectangularization : \n");
      print_rw_regions(*recv_l);
    }

  pips_debug(1, "end\n");
}


static statement compile_master(entity new_module, statement work_stmt)
{
  statement stmt;
  pips_debug(1, "begin\n");

  entity rank = get_entity_step_rank(new_module);
  expression expr_rank = entity_to_expression(rank);


  /*
    Generation of
    STEP_GET_RANK(&STEP_COMM_RANK);
  */
  statement rank_stmt = call_STEP_subroutine2(RT_STEP_get_rank, get_expression_addr(expr_rank), NULL);

  /*
    Generation of
    if (STEP_COMM_RANK==0) {}
  */
  statement if_stmt = instruction_to_statement(make_instruction_test(make_test(MakeBinaryCall(entity_intrinsic(EQUAL_OPERATOR_NAME),
											      expr_rank,
											      int_to_expression(0)),
									       work_stmt, make_block_statement(NIL))));

  stmt = make_block_statement(make_statement_list(rank_stmt, if_stmt));

  pips_debug(1, "end\n");
  return stmt;
}

static statement compile_barrier(entity __attribute__ ((unused)) new_module, statement work_stmt)
{
  pips_debug(1, "begin\n");

  statement call_stmt = call_STEP_subroutine2(RT_STEP_barrier, NULL);
  insert_statement(work_stmt, call_stmt, false);

  pips_debug(1, "end\n");
  return work_stmt;
}

static bool step_get_directive_reductions(step_directive drt, set *reductions_l)
{
  bool reduction_p = false;
  int op;

  pips_debug(1, "begin\n");

  for(op = 0; op < STEP_UNDEF_REDUCE; op++)
    reductions_l[op] = set_make(set_pointer);

  FOREACH(STEP_CLAUSE, c, step_directive_clauses(drt))
    {
      switch (step_clause_tag(c))
	{
	case is_step_clause_reduction:
	  MAP_ENTITY_INT_MAP(variable, op, {
	      set_add_element(reductions_l[op], reductions_l[op], variable);
	      reduction_p = true;
	    }, step_clause_reduction(c));
	  break;
	case is_step_clause_transformation:
	case is_step_clause_nowait:
	case is_step_clause_private:
	case is_step_clause_shared:
	case is_step_clause_threadprivate:
	case is_step_clause_firstprivate:
	case is_step_clause_copyin:
	case is_step_clause_schedule:
	  break;
	default: pips_assert("clause not compiled", 0);
	}
    }

  pips_debug(1, "end\n");
  return reduction_p;
}

/*
  This test removes entities declared inside parallel regions and not outside.
  It is useful when analysis are not precise enough.

  A consequence is that ANYMODULE ANYWHERE regions will not be transformed into statement. */
static list duplicate_regions_referenced_entity(list referenced_entities, list regions_l)
{
  list l = NIL;

  FOREACH(REGION, reg, regions_l)
    {
      if (gen_in_list_p(region_entity(reg), referenced_entities))
	l = CONS(REGION, copy_effect(reg), l);
      else
	pips_debug(2,"drop entity %s not in referenced_entities\n", entity_name(region_entity(reg)));
    }
  return gen_nreverse(l);
}

static list keep_recv_or_send_referenced_entity(list recv_l, list send_l, list referenced_entities)
{
   list l = NIL;
   set recv_s = set_make(set_pointer);
   set send_s = set_make(set_pointer);

   FOREACH(REGION, r, recv_l)
     {
       recv_s = set_add_element(recv_s, recv_s, region_entity(r));
     }
   FOREACH(REGION, r, send_l)
     {
       send_s = set_add_element(send_s, send_s, region_entity(r));
     }

   FOREACH(ENTITY, e, referenced_entities)
     {
       if (set_belong_p(recv_s, e) || set_belong_p(send_s, e))
	 l = CONS(ENTITY, e, l);
       else
	 pips_debug(2, "entity %s in referenced entities but not in recv nor send regions\n", entity_name(e));
     }

   set_free(recv_s);
   set_free(send_s);

  return gen_nreverse(l);
}

static void compile_body(statement directive_stmt, entity new_module, step_directive drt, int transformation, list referenced_entities,
			 statement new_body, statement mpi_begin_stmt, statement work_stmt, statement mpi_end_stmt)
{
  bool loop_p = false;
  set reductions_l[STEP_UNDEF_REDUCE];
  bool reduction_p = false;

  assert(!entity_undefined_p(get_current_module_entity()));

  statement_effects send_regions = (statement_effects)db_get_memory_resource(DBR_STEP_SEND_REGIONS, get_current_module_name(), true);
  list send_l =  duplicate_regions_referenced_entity(referenced_entities, effects_effects(apply_statement_effects(send_regions, directive_stmt)));
  statement_effects recv_regions = (statement_effects)db_get_memory_resource(DBR_STEP_RECV_REGIONS, get_current_module_name(), true);
  list recv_l =  duplicate_regions_referenced_entity(referenced_entities, effects_effects(apply_statement_effects(recv_regions, directive_stmt)));


  pips_assert("step_directive_type(drt)",
	      step_directive_type(drt) == STEP_PARALLEL ||
	      step_directive_type(drt) == STEP_PARALLEL_DO ||
	      step_directive_type(drt) == STEP_DO ||
	      step_directive_type(drt) == STEP_MASTER ||
	      step_directive_type(drt) == STEP_BARRIER ||
	      step_directive_type(drt) == STEP_THREADPRIVATE);

  step_directive_type_print(drt);

  loop loop_stmt = loop_undefined;

  if (step_directive_type(drt) == STEP_PARALLEL)
    {
      generate_call_init_regionArray(referenced_entities, mpi_begin_stmt, mpi_end_stmt);
    }
  if (step_directive_type(drt) == STEP_PARALLEL_DO)
    {
      list l = keep_recv_or_send_referenced_entity(recv_l, send_l, referenced_entities);
      generate_call_init_regionArray(l, mpi_begin_stmt, mpi_end_stmt);
      gen_free_list(l);
    }

  if (step_directive_type(drt) == STEP_PARALLEL_DO || step_directive_type(drt) == STEP_DO)
    {
      statement loopslices_stmt = statement_undefined;
      list work_block = statement_block(work_stmt);

      loop_stmt = statement_loop(STATEMENT(CAR(statement_block(STATEMENT(CAR(work_block))))));
      loop_p = true;

      pips_debug(2, "loop_stmt = %p\n", loop_stmt);

      loopslices_stmt = compile_loopslices(new_module, loop_stmt);
      insert_statement(mpi_begin_stmt, loopslices_stmt, false);

      loopbounds_substitution(new_module, loop_stmt);
      work_stmt = compile_loopbounds(new_module, loop_stmt, work_stmt);
      loop_basic_workchunk_to_workchunk(directive_stmt, &recv_l, &send_l);
    }


  if (step_directive_type(drt) ==  STEP_MASTER)
    {
      work_stmt = compile_master(new_module, work_stmt);
    }

  if (step_directive_type(drt) ==  STEP_BARRIER)
    {
      work_stmt = compile_barrier(new_module, work_stmt);
    }

  reduction_p = step_get_directive_reductions(drt, reductions_l);
  if (reduction_p)
    compile_reduction(reductions_l, mpi_begin_stmt, mpi_end_stmt);

  if (recv_l != NIL)
    {
      pips_debug(2, "compile RECV regions\n");
      list send_as_comm_l = list_undefined;
      compile_regions(new_module, recv_l, loop_p, loop_stmt, send_as_comm_l, reductions_l, mpi_begin_stmt, mpi_end_stmt);
    }

  if (send_l != NIL)
    {
      pips_debug(2, "compile SEND regions\n");
      list send_as_comm_l = effects_effects(apply_statement_effects(send_regions, directive_stmt));
      compile_regions(new_module, send_l, loop_p, loop_stmt, send_as_comm_l, reductions_l, mpi_begin_stmt, mpi_end_stmt);
    }

  generate_call_construct_begin_construct_end(new_module, drt, mpi_begin_stmt, mpi_end_stmt);

  ifdebug(2)
    {
      pips_debug(2, "mpi_begin_stmt\n");
      print_statement(mpi_begin_stmt);
      pips_debug(2, "mpi_end_stmt\n");
      print_statement(mpi_end_stmt);
    }

  /*
    Body building
  */
  statement begin_work_stmt = make_plain_continue_statement();
  statement end_work_stmt = make_plain_continue_statement();
  put_a_comment_on_a_statement(begin_work_stmt, strdup("\nBEGIN WORK"));
  put_a_comment_on_a_statement(end_work_stmt, strdup("END WORK\n"));
  insert_statement(work_stmt, begin_work_stmt, true);
  insert_statement(work_stmt, end_work_stmt, false);
  insert_statement(new_body, work_stmt, false);

  if (transformation == STEP_TRANSFORMATION_HYBRID &&
      step_directive_type(drt) != STEP_MASTER)
    {
      compile_omp(directive_stmt, drt);
      add_omp_guard(&mpi_begin_stmt);
      add_omp_guard(&mpi_end_stmt);
    }


  if(!empty_statement_p(mpi_begin_stmt))
    insert_statement(new_body, mpi_begin_stmt, true);
  if(!empty_statement_p(mpi_end_stmt))
    insert_statement(new_body, mpi_end_stmt, false);

  if (transformation == STEP_TRANSFORMATION_HYBRID &&
      step_directive_type(drt) == STEP_MASTER)
    {
      add_omp_guard(&new_body);
    }
}

// Some entities may be added by region_to_statement. Ensure they will be passed as parameters
static void update_referenced_entities(statement new_body, list *referenced_entities)
{
  int local_dbg_lvl = 2;
  set new_entities = set_make(set_pointer);
  list l_body = make_statement_list(new_body);

  set_assign_list(new_entities, outliner_statements_referenced_entities(l_body));

  if(!set_undefined_p(step_created_symbolic))
    {
      SET_FOREACH(entity, step_symbolic_e, step_created_symbolic)
	{
	  pips_debug(local_dbg_lvl, "drop entities step_symbolic : %s\n", entity_name(step_symbolic_e));
	  new_entities = set_del_element(new_entities, new_entities, step_symbolic_e);
	}
    }

  FOREACH(ENTITY, e, *referenced_entities)
    {
      pips_debug(local_dbg_lvl, "drop referenced entity : %s\n", entity_name(e));
      new_entities = set_del_element(new_entities, new_entities, e);
    }

  if(step_created_entity != set_undefined)
    {
      SET_FOREACH(entity, created_e, step_created_entity)
	{
	  pips_debug(local_dbg_lvl, "drop entities step_created : %s\n", entity_name(created_e));
	  new_entities = set_del_element(new_entities, new_entities, created_e);
	}
    }

  FOREACH(ENTITY, e, statements_to_declarations(l_body))
    {
      pips_debug(local_dbg_lvl, "drop declared entity : %s\n", entity_name(e));
      new_entities = set_del_element(new_entities, new_entities, e);
    }

  if(c_module_p(get_current_module_entity()))
    {
      FOREACH(ENTITY, e, entity_declarations(module_name_to_entity(compilation_unit_of_module(get_current_module_name()))))
	{
	  if(top_level_entity_p(e) && set_belong_p(new_entities, e))
	    {
	      pips_debug(local_dbg_lvl, "drop global entity : %s\n", entity_name(e));
	      new_entities = set_del_element(new_entities, new_entities, e);
	    }
	}
    }

  list new_e = set_to_sorted_list(new_entities, (gen_cmp_func_t)compare_entities);
  set_free(new_entities);

  pips_debug(local_dbg_lvl, "New entities : "); print_entities(new_e); fprintf(stderr, "\n\n");

  *referenced_entities = gen_nconc(*referenced_entities, new_e);
}

statement compile_mpi(statement directive_stmt, string new_module_name, step_directive drt, int transformation)
{
  pips_debug(1, "begin current module %p : %s\n", get_current_module_entity(), get_current_module_name());

  list statements_to_outline = make_statement_list(directive_stmt);
  entity new_module = gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, new_module_name, NULL), entity_domain);
  pips_assert("new_module", !entity_undefined_p(new_module));

  // Init hash table
  hash_table entity_to_effective_parameter = outliner_init(new_module, statements_to_outline);

  // Find referenced entities in statements_to_outline
  statement new_body = make_empty_block_statement();
  list referenced_entities = outliner_scan(new_module, statements_to_outline, new_body);

  // Generate the body of the new module
  statement work_stmt = make_block_statement(gen_copy_seq(statements_to_outline));
  statement mpi_begin_stmt = make_empty_block_statement();
  statement mpi_end_stmt = make_empty_block_statement();
  compile_body(directive_stmt, new_module, drt, transformation, referenced_entities, new_body, mpi_begin_stmt, work_stmt, mpi_end_stmt);

  // Some entities may be added by region_to_statement. Ensure they will be passed as parameters
  update_referenced_entities(new_body, &referenced_entities);

  // Update hash table and build parameters
  list effective_parameters = NIL, formal_parameters = NIL;
  outliner_parameters(new_module, new_body, referenced_entities, entity_to_effective_parameter, &effective_parameters, &formal_parameters);

  // Patch parameters for side effects
  if(c_module_p(get_current_module_entity()))
    outliner_patch_parameters(statements_to_outline, referenced_entities, effective_parameters, formal_parameters, new_body, mpi_begin_stmt, mpi_end_stmt);

  step_RT_set_local_declarations(new_module, new_body);

  // Source file generation
  set_prettyprinter_head_hook(step_head_hook);
  outliner_file(new_module, formal_parameters, &new_body);
  reset_prettyprinter_head_hook();

  step_RT_clean_local();

  statement call_stmt = outliner_call(new_module, statements_to_outline, effective_parameters);

  pips_debug(1, "end\n");
  return call_stmt;
}

