/* Copyright 2007, 2008, 2009 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h" // for STEP_SENTINELLE

#include "effects-generic.h" // needed by effects-convex.h for descriptor
#include "effects-convex.h" // for region

#include "semantics.h" // for expression_and_precondition_to_integer_interval
#include "accel-util.h" // for outliner_patch_parameters, outliner_file, ...


#define STEP_GENERATED_SUFFIX_F ".step_generated.f"
#define STEP_GENERATED_SUFFIX_C ".step_generated.c"

/*
  Voir si n'existe pas dans Libs/effect
*/
static bool combinable_regions_p(region r1, region r2)
{
  bool same_var, same_act;

  if (region_undefined_p(r1) || region_undefined_p(r2))
    return(true);

  same_var = (region_entity(r1) == region_entity(r2));
  same_act = action_equal_p(region_action(r1), region_action(r2));

  return(same_var && same_act);
}

bool step_install(const char* __attribute__ ((unused)) program_name)
{
  debug_on("STEP_INSTALL_DEBUG_LEVEL");

  /* Generation des fichiers sources dans workspace.database/Src/ */
  string dest_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
  gen_array_t modules = db_get_module_list_initial_order();
  int n = gen_array_nitems(modules), i;
  hash_table user_files = hash_table_make(hash_string, 2*n);

  for (i=0; i<n; i++)
    {
      string module_name = gen_array_item(modules, i);
      string user_file = db_get_memory_resource(DBR_USER_FILE, module_name, true);
      string new_file = hash_get(user_files, user_file);
      if (new_file == HASH_UNDEFINED_VALUE)
	{
	  string base_name = pips_basename(user_file, NULL);
	  new_file = strdup(concatenate(dest_dir, "/", base_name, NULL));
	  hash_put(user_files, user_file, new_file);
	}

      string step_file = db_get_memory_resource(DBR_STEP_FILE, module_name, true);



      pips_debug(1, "Module: \"%s\"\n\tuser_file: \"%s\"\n\tinstall file : \"%s\"\n", module_name, user_file, new_file);
      FILE *out = safe_fopen(new_file, "a");
      FILE *in = safe_fopen(step_file, "r");

      safe_cat(out, in);

      safe_fclose(out, new_file);
      safe_fclose(in, step_file);
    }

  hash_table_free(user_files);
  free(dest_dir);

  /* Instalation des fichiers generes */
  dest_dir = strdup(get_string_property("STEP_INSTALL_PATH"));
  string src_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
  if (empty_string_p(dest_dir))
    {
      free(dest_dir);
      dest_dir = db_get_directory_name_for_module(WORKSPACE_SRC_SPACE);
    }

  safe_system(concatenate("step_install ", dest_dir, " ", src_dir,  NULL));

  free(src_dir);
  free(dest_dir);

  debug_off();
  return true;
}


/*
  Suppression du pragma string "STEP" ajoute par step_directive
*/
static statement compile_seq(statement stmt)
{
  list ext_l = NIL;

  FOREACH(EXTENSION, ext, extensions_extension(statement_extensions(stmt)))
    {
      pragma p = extension_pragma(ext);
      if(pragma_string_p(p) && strncmp(pragma_string(p), STEP_SENTINELLE, strlen(STEP_SENTINELLE))==0)
	{
	  pips_debug(2,"drop pragma : %s\n", pragma_string(p));
	  free_extension(ext);
	}
      else
	ext_l = CONS(EXTENSION, ext, ext_l);
    }
  gen_free_list(extensions_extension(statement_extensions(stmt)));
  extensions_extension(statement_extensions(stmt)) = gen_nreverse(ext_l);

  return stmt;
}

static statement compile_omp(statement stmt, step_directive d)
{
  string begin_txt, end_txt;
  bool is_fortran = fortran_module_p(get_current_module_entity());
  bool is_block_construct = step_directive_to_strings(d, is_fortran, &begin_txt, &end_txt);

  if(!string_undefined_p(begin_txt))
    {
      if(!is_block_construct)
	{
	  if(ENDP(statement_block(stmt)))
	    insert_statement(stmt, make_plain_continue_statement(), false);
	  stmt = STATEMENT(CAR(statement_block(stmt)));
	}
      add_pragma_str_to_statement(stmt, begin_txt, false);
    }

  if(!string_undefined_p(end_txt))
    {
      insert_statement(stmt, make_plain_continue_statement(), false);
      stmt = last_statement(stmt);
      add_pragma_str_to_statement(stmt, end_txt, false);
    }

  return stmt;
}

static void add_guard(statement *block)
{
  pips_assert("block", block && !statement_undefined_p(*block) && statement_block_p(*block));

  if (!empty_statement_or_continue_p(*block))
    {
      statement barrier_stmt = make_empty_block_statement();
      step_directive barrier_guard = make_step_directive(STEP_BARRIER, statement_undefined, NIL);
      step_directive master_guard = make_step_directive(STEP_MASTER, statement_undefined, NIL);
      compile_omp(barrier_stmt, barrier_guard);
      compile_omp(*block, master_guard);
      free_step_directive(barrier_guard);
      free_step_directive(master_guard);

      *block = make_block_statement(CONS(STATEMENT, *block, CONS(STATEMENT, barrier_stmt, NIL)));
    }
}

static void compile_share(entity new_module, list shared, statement before, statement after)
{
  list init_block = NIL;
  list flush_block = NIL;
  bool is_optimized = false;
  bool is_interlaced = false;
  FOREACH(ENTITY, e, shared)
    {
      if (type_variable_p(entity_type(e)) &&
	  !entity_scalar_p(e))
	{
	  init_block = CONS(STATEMENT, build_call_STEP_init_arrayregions(e), init_block);
	  flush_block = CONS(STATEMENT, build_call_STEP_AllToAll(new_module, e, is_optimized, is_interlaced), flush_block);
	}
    }
  insert_statement(before, make_block_statement(gen_nreverse(init_block)), false);
  insert_statement(after, build_call_STEP_WaitAll(gen_nreverse(flush_block)), false);
}

static void compile_reduction(set reductions_l[STEP_OP_UNDEFINED], statement before_stmt, statement after_stmt)
{
  int op;
  list arglist;
  string op_name[STEP_OP_UNDEFINED]={STEP_PROD_NAME, STEP_MAX_NAME, STEP_MIN_NAME, STEP_SUM_NAME};

  pips_assert("block before", !statement_undefined_p(before_stmt) && statement_block_p(before_stmt));
  pips_assert("block after", !statement_undefined_p(after_stmt) && statement_block_p(after_stmt));

  for(op=0; op<STEP_OP_UNDEFINED; op++)
    if(!set_empty_p(reductions_l[op]))
      {
	FOREACH(ENTITY, variable, set_to_sorted_list(reductions_l[op], (gen_cmp_func_t)compare_entities))
	  {
	    arglist = CONS(EXPRESSION, entity_to_expression(variable),
			   CONS(EXPRESSION, entity_to_expression(MakeConstant(op_name[op], is_basic_string)),
				NIL));
	    insert_statement(before_stmt,
			     call_STEP_subroutine(RT_STEP_initreduction, arglist, entity_type(variable)), false);

	    arglist = CONS(EXPRESSION, entity_to_expression(variable),
			   NIL);
	    insert_statement(after_stmt,
			     call_STEP_subroutine(RT_STEP_reduction, arglist, type_undefined), false);
	  }
      }
}

/*
  Ajout des contraintes I_L<=i<=I_U (voir add_index_range_conditions dans semantics/loop.c)
*/
static void loop_basic_workchunk_to_workchunk(statement stmt, list *recv_l, list *send_l)
{
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
  if(incr_lb==incr_ub) {
    if(incr_lb==0) {
      pips_user_error("Illegal null increment\n");
    }
    else
      incr = incr_lb;
  }
  else if(incr_lb>=1) {
    incr = 1;
  }
  else if(incr_ub<=-1) {
    incr = -1;
  }
  entity e_lb = step_local_loop_index(get_current_module_entity(), strdup(STEP_BOUNDS_LOW(index)));
  entity e_ub = step_local_loop_index(get_current_module_entity(), strdup(STEP_BOUNDS_UP(index)));
  if(incr<0)
    {
      entity tmp = e_lb;
      e_lb = e_ub;
      e_ub = tmp;
    }
  /*
    Ajout des contraintes et projection
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

  ifdebug(1)
    {
      pips_debug(1, "SEND after projection: \n");
      print_rw_regions(*send_l);
      pips_debug(1, "RECV after projection: \n");
      print_rw_regions(*recv_l);
    }
}


static bool region_interlaced_p(list send_regions, region r)
{
  bool interlaced = false;
  if(region_write_p(r))
    {
      bool first = true;

      FOREACH(REGION, reg, send_regions)
	{
	  if(combinable_regions_p(reg, r))
	    {
	      assert(first);
	      interlaced = step_interlaced_p(reg);
	      first = false;
	    }
	}
    }
  return interlaced;
}

static bool comm_partial_p(list send_regions, region r)
{
  bool partial = true;
  if(region_write_p(r))
    {
      bool first = true;

      FOREACH(REGION, reg, send_regions)
	{
	  if(combinable_regions_p(reg, r))
	    {
	      assert(first);
	      partial = step_partial_p(reg);
	      first = false;
	    }
	}
    }
  return partial;
}

static bool reduction_p(set reductions_l[STEP_OP_UNDEFINED], entity e)
{
  bool is_reduction = false;
  int op;
  for(op=0; !is_reduction && op<STEP_OP_UNDEFINED; op++)
    is_reduction = set_belong_p(reductions_l[op], e);
  return is_reduction;
}


/*
  si equality=true :
  recherche la premiere contrainte d'egalite portant sur la variable PHI du systeme de contraintes sys et transforme cette contrainte en 2 expressions, l'une traduisant le contrainte d'inferiorite (ajouter a expr_l), l'autre traduisant la contrainte de superiorite (ajoutée a expr_u)
  retourne true si une contrainte d'egalite a ete trouve, false sinon.

  si equality=false :
  traduit l'ensemble des contraintes d'inegalite portant sur la variable PHI.
  Les expressions traduisant une contrainte d'inferiorie (de superiorite) sont ajoutes à la liste expr_l (expr_u)
*/
/*
  Voir make_contrainte_expression
*/

static bool contraintes_to_expression(bool equality, entity phi, Psysteme sys, list *expr_l, list *expr_u)
{
  Pcontrainte c;
  bool found_equality=false;

  for(c = equality?sc_egalites(sys):sc_inegalites(sys); !found_equality && !CONTRAINTE_UNDEFINED_P(c); c = c->succ)
    {
      int coef_phi=VALUE_TO_INT(vect_coeff((Variable)phi,c->vecteur));
      if(coef_phi != 0)
	{
	  expression expr;
	  Pvecteur coord,v = vect_del_var(c->vecteur, (Variable)phi);
	  bool up_bound = (coef_phi > 0);
	  bool low_bound = !up_bound;

	  //construction des expressions d'affectation
	  if(VECTEUR_NUL_P(v))
	    expr = int_to_expression(0);
	  else
	    {
	      if (coef_phi > 0) //contrainte de type : coef_phi*phi  <= "vecteur"
		{
		  for (coord = v; coord!=NULL; coord=coord->succ)
		    val_of(coord) = -val_of(coord);
		  coef_phi = -coef_phi;
		}

	      expr = make_vecteur_expression(v);
	      if (coef_phi != -1)
		expr = make_op_exp("/", expr, int_to_expression(-coef_phi));
	    }

	  if (equality || low_bound)
	    *expr_l = CONS(EXPRESSION,copy_expression(expr), *expr_l);
	  if (equality || up_bound)
	    *expr_u = CONS(EXPRESSION,copy_expression(expr), *expr_u);

	  free_expression(expr);
	  found_equality = equality;
	}
    }
  return found_equality;
}

/*
  genere un statement de la forme :
  array_region(bound_name,dim,index) = expr_bound
*/
static statement bound_to_statement(entity mpi_module, list expr_bound, entity array_region, string bound_name, int dim, list index)
{
  pips_assert("expression", !ENDP(expr_bound));
  statement s;
  entity op = entity_undefined;
  list dims = CONS(EXPRESSION, step_symbolic(bound_name, mpi_module),
		   CONS(EXPRESSION, int_to_expression(dim), gen_full_copy_list(index)));

  bool is_fortran = fortran_module_p(get_current_module_entity());
  if (!is_fortran)
    {
      list l = NIL;
      FOREACH(EXPRESSION, e, dims)
	{
	  l = CONS(EXPRESSION, MakeBinaryCall(CreateIntrinsic(MINUS_C_OPERATOR_NAME),
					      e,
					      int_to_expression(1)),
		   l);
	}
      dims = l;
    }

  expression expr = reference_to_expression(make_reference(array_region, dims));

  if ( gen_length(expr_bound) != 1 )
    {
      if(strncmp(bound_name, STEP_INDEX_SLICE_LOW_NAME, strlen(bound_name)) == 0)
	op = entity_intrinsic(MAX_OPERATOR_NAME);
      else if (strncmp(bound_name, STEP_INDEX_SLICE_UP_NAME, strlen(bound_name)) == 0)
	op = entity_intrinsic(MIN_OPERATOR_NAME);
      else
	pips_internal_error("unexpected bound name %s", bound_name);
      s = make_assign_statement(expr, call_to_expression(make_call(op, expr_bound)));
    }
  else
    s = make_assign_statement(expr, EXPRESSION(CAR(expr_bound)));

  return s;
}

static void regions_to_statement(entity index, entity mpi_module, list regions, entity workchunk_id, list send_pur, set reductions_l[STEP_OP_UNDEFINED],
				 statement *compute_regions, statement *set_regions, statement *flush_regions)
{
  list index_slice = NIL;

  *compute_regions = make_empty_block_statement();
  *set_regions = make_empty_block_statement();
  *flush_regions = make_empty_block_statement();

  if(!entity_undefined_p(workchunk_id))
    index_slice = CONS(EXPRESSION, entity_to_expression(workchunk_id), index_slice);

  FOREACH(REGION, reg, regions)
    {
      /*
	Ajout de la region en commentaire
      */
      string str_eff = text_to_string(text_rw_array_regions(CONS(REGION, reg, NIL)));
      statement comment_stmt = make_plain_continue_statement();
      put_a_comment_on_a_statement(comment_stmt, str_eff);
      insert_statement(*compute_regions, comment_stmt, false);

      /*
	Compute regions
      */
      entity array = region_entity(reg);
      Psysteme sys = region_system(reg);
      list bounds_array = variable_dimensions(type_variable(entity_type(region_entity(reg))));
      entity array_region = step_local_arrayRegions(region_read_p(reg)?STEP_RR_NAME(array):STEP_SR_NAME(array),
						    mpi_module, array,
						    entity_undefined_p(workchunk_id)?expression_undefined:step_symbolic(STEP_MAX_NB_LOOPSLICES_NAME, mpi_module));

      // on parcourt dans l'ordre des indices (PHI1, PHI2, ...) chaque PHIi correspond a une dimension dim
      int dim = 0;
      FOREACH(EXPRESSION, expr, reference_indices(effect_any_reference(reg)))
	{
	  list expr_l = NIL;
	  list expr_u = NIL;
	  entity phi = reference_variable(syntax_reference(expression_syntax(expr)));
	  dimension bounds_d = DIMENSION(gen_nth(dim, bounds_array));   // gen_nth numerote les element a partir de 0 et ...
	  dim++; // ... les tableaux de region numerote les dimensions a partir de 1

	  // on determine les listes d'expression expr_l (et expr_u) correspondant aux
	  // contraites low (et up) portant sur la variable PHI courante
	  // ex: L <= PHI1 + 1  et PHI1 -1 <= U
	  // expr_l contient l'expression (L-1) et expr_u contient l'expression (U+1)

	  //	  pips_assert("empty list", ENDP(expr_l) && ENDP(expr_u));
	  // recherche et transformation des contraintes d'equalites portant sur phi
	  contraintes_to_expression(true, phi, sys, &expr_l, &expr_u);
	  // recherche et transformation des contraintes d'inequalites portant sur phi
	  contraintes_to_expression(false, phi, sys, &expr_l, &expr_u);

	  // ajout contraintes liees aux bornes d'indexation du tableau pour la dimension courante
	  if(ENDP(expr_l))
	    expr_l = CONS(EXPRESSION, copy_expression(dimension_lower(bounds_d)), expr_l);
	  if(ENDP(expr_u))
	    expr_u = CONS(EXPRESSION, copy_expression(dimension_upper(bounds_d)), expr_u);

	  /*
	    generation des statements : array_region(bound_name, dim, index_slice) = expr_bound
	  */
	  statement b1 = bound_to_statement(mpi_module, expr_l, array_region, STEP_INDEX_SLICE_LOW_NAME, dim, index_slice);
	  statement b2 = bound_to_statement(mpi_module, expr_u, array_region, STEP_INDEX_SLICE_UP_NAME, dim, index_slice);

	  insert_statement(*compute_regions, b1, false);
	  insert_statement(*compute_regions, b2, false);
	}

      /*
	Set regions
      */
      expression expr_nb_workchunk = !entity_undefined_p(workchunk_id)?step_local_size(mpi_module):int_to_expression(1);
      bool is_interlaced = region_interlaced_p(send_pur, reg);
      bool is_reduction = region_write_p(reg) && reduction_p(reductions_l, array);
      statement set_stmt = region_read_p(reg)?
	build_call_STEP_set_recv_region(array, expr_nb_workchunk, array_region):
	build_call_STEP_set_send_region(array, expr_nb_workchunk, array_region, is_interlaced, is_reduction);
      insert_statement(*set_regions, set_stmt, false);

      /*
	Flush_regions
      */
      if(!is_reduction)
	{
	  bool is_optimized = comm_partial_p(send_pur, reg);
	  statement comm_stmt = build_call_STEP_AllToAll(mpi_module, array, is_optimized, is_interlaced);
	  insert_statement(*flush_regions, comm_stmt, false);
	}
    }

  /*
    ajout de la boucle parcourant les workchunks
  */
  if(!entity_undefined_p(workchunk_id) &&
     !entity_undefined_p(index) &&
     !ENDP(statement_block(*compute_regions)))
    {
      expression expr_id_workchunk = make_op_exp(PLUS_OPERATOR_NAME, int_to_expression(-1), entity_to_expression(workchunk_id));
      expression expr_index_low = entity_to_expression(step_local_loop_index(mpi_module, STEP_BOUNDS_LOW(index)));
      expression expr_index_up = entity_to_expression(step_local_loop_index(mpi_module, STEP_BOUNDS_UP(index)));
      if (!fortran_module_p(get_current_module_entity()))
	{
	  expr_index_low = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), expr_index_low);
	  expr_index_up = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), expr_index_up);
	}
      list args = CONS(EXPRESSION, expr_id_workchunk,
		       CONS(EXPRESSION, expr_index_low,
			    CONS(EXPRESSION, expr_index_up, NIL)));
      statement get_bounds_stmt = call_STEP_subroutine(RT_STEP_get_loopbounds, args, type_undefined);
      insert_statement(*compute_regions, get_bounds_stmt, true);

      range rng = make_range(int_to_expression(1), step_local_size(mpi_module), int_to_expression(1));

      statement loop_stmt = instruction_to_statement(make_instruction_loop(make_loop(workchunk_id, rng, *compute_regions, entity_empty_label(), make_execution_sequential(), NIL)));

      *compute_regions = make_empty_block_statement();
      insert_statement(*compute_regions, loop_stmt, true);
    }
  /*
    ajout du Flush
  */
  if(!ENDP(statement_block(*flush_regions)))
    {
      statement flush_stmt = call_STEP_subroutine(RT_STEP_flush, NIL, type_undefined);
      insert_statement(*flush_regions, flush_stmt, false);
    }
}

static statement compile_loop(entity new_module, statement work_stmt, statement *schedule, entity *index_)
{
  list work_block = statement_block(work_stmt);
  list dir_block = statement_block(STATEMENT(CAR(work_block)));

  loop l = statement_loop(STATEMENT(CAR(dir_block)));
  entity index = loop_index(l);
  range r = loop_range(l);
  *index_ = index;

  // LoopSlice
  expression expr_comm_size = step_local_size(new_module);
  if (!fortran_module_p(get_current_module_entity()))
    expr_comm_size = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), expr_comm_size);
  statement size_stmt = call_STEP_subroutine(RT_STEP_get_commsize, CONS(EXPRESSION, expr_comm_size, NIL), type_undefined);
  statement loop_slice_stmt = call_STEP_subroutine(RT_STEP_compute_loopslices,
						   CONS(EXPRESSION, range_lower(r),
							CONS(EXPRESSION, range_upper(r),
							     CONS(EXPRESSION, range_increment(r),
								  CONS(EXPRESSION, step_local_size(new_module), NIL)))),
						   type_undefined);
  *schedule = make_block_statement(make_statement_list(size_stmt, loop_slice_stmt));

  // Substitution des bornes de boucles
  range_lower(r) = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_LOW(index)));
  range_upper(r) = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_UP(index)));

  /*
    Generation de :
    CALL STEP_GET_RANK(STEP_COMM_RANK)
    CALL STEP_GETLOOPBOUNDS(STEP_Rank, I_SLICE_LOW, I_SLICE_UP)
  */
  expression expr_rank = step_local_rank(new_module);
  expression expr_index_low = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_LOW(index)));
  expression expr_index_up = entity_to_expression(step_local_loop_index(new_module, STEP_BOUNDS_UP(index)));
  if (!fortran_module_p(get_current_module_entity()))
    {
      expr_rank = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), expr_rank);
      expr_index_low = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), expr_index_low);
      expr_index_up = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), expr_index_up);
    }
  statement rank_stmt = call_STEP_subroutine(RT_STEP_get_rank, CONS(EXPRESSION, expr_rank, NIL), type_undefined);
  expression expr_id_workchunk = step_local_rank(new_module);
  list args = CONS(EXPRESSION, expr_id_workchunk,
		   CONS(EXPRESSION, expr_index_low,
			CONS(EXPRESSION, expr_index_up, NIL)));
  statement get_bounds_stmt = call_STEP_subroutine(RT_STEP_get_loopbounds, args, type_undefined);

  return make_block_statement(make_statement_list(rank_stmt, get_bounds_stmt, work_stmt));
}

static statement compile_master(entity new_module, statement work_stmt)
{
  expression expr_rank = step_local_rank(new_module);
  if (!fortran_module_p(get_current_module_entity()))
    expr_rank = MakeUnaryCall(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), expr_rank);
  statement rank_stmt = call_STEP_subroutine(RT_STEP_get_rank, CONS(EXPRESSION, expr_rank, NIL), type_undefined);
  statement if_stmt = instruction_to_statement(make_instruction_test(make_test(MakeBinaryCall(entity_intrinsic(EQUAL_OPERATOR_NAME),
											      step_local_rank(new_module),
											      int_to_expression(0)),
									       work_stmt, make_block_statement(NIL))));
  return make_block_statement(make_statement_list(rank_stmt, if_stmt));
}

static statement compile_barrier(entity __attribute__ ((unused)) new_module, statement work_stmt)
{
  statement call_stmt = call_STEP_subroutine(RT_STEP_barrier, NIL, type_undefined);
  insert_statement(work_stmt, call_stmt, false);
  return work_stmt;
}

static string step_head_hook(entity __attribute__ ((unused)) e)
{
  return strdup(concatenate
		("      implicit none\n",
		 "      include \"STEP.h\"\n", NULL));
}



static statement compile_mpi(statement stmt, string new_module_name, step_directive d)
{
  statement_effects send_regions = (statement_effects)db_get_memory_resource(DBR_STEP_SEND_REGIONS, get_current_module_name(), true);
  list send_lpur = effects_effects(apply_statement_effects(send_regions, stmt));
  list send_l = gen_full_copy_list(send_lpur);
  statement_effects recv_regions = (statement_effects)db_get_memory_resource(DBR_STEP_RECV_REGIONS, get_current_module_name(), true);
  list recv_l = gen_full_copy_list(effects_effects(apply_statement_effects(recv_regions, stmt)));

  statement comment_stmt;
  statement mpi_begin_stmt = make_empty_block_statement();
  statement mpi_end_stmt = make_empty_block_statement();

  entity module = get_current_module_entity();
  pips_debug(1,"current module %p : %s\n", module, get_current_module_name());

  entity new_module = gen_find_tabulated(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, new_module_name, NULL), entity_domain);
  pips_assert("new_module", !entity_undefined_p(new_module));

  list statements_to_outline = make_statement_list(stmt);
  statement work_stmt = make_block_statement(gen_copy_seq(statements_to_outline));
  statement new_body = make_empty_block_statement();

  hash_table entity_to_effective_parameter = outliner_init(new_module, statements_to_outline);
  list referenced_entities = outliner_scan(new_module, statements_to_outline, new_body);

  list effective_parameters = NIL;
  list formal_parameters = NIL;
  outliner_parameters(new_module, new_body, referenced_entities, entity_to_effective_parameter, &effective_parameters, &formal_parameters);

  /*
    MPI generation
  */
  bool nowait = false;
  bool reduction = false;
  int transformation = -1;
  int op;
  set reductions_l[STEP_OP_UNDEFINED];
  for(op=0; op<STEP_OP_UNDEFINED; op++)
    reductions_l[op] = set_make(set_pointer);

  FOREACH(STEP_CLAUSE, c, step_directive_clauses(d))
    {
      switch (step_clause_tag(c))
	{
	case is_step_clause_transformation:
	  transformation = step_clause_transformation(c);
	  break;
	case is_step_clause_reduction:
	  MAP_ENTITY_INT_MAP(variable, op, {
	      set_add_element(reductions_l[op], reductions_l[op], variable);
	      reduction = true;
	    }, step_clause_reduction(c));
	  break;
	case is_step_clause_nowait:
	  nowait = true;
	  break;
	case is_step_clause_private:
	case is_step_clause_shared:
	  break;
	default: assert(0);
	}
    }

  nowait=nowait; // << nowait is not used ! fix this

  assert(!entity_undefined_p(get_current_module_entity()));
  string directive_txt;
  entity index = entity_undefined;
  statement schedule = statement_undefined;
  switch(step_directive_type(d))
    {
    case STEP_PARALLEL:
      directive_txt = strdup(STEP_PARALLEL_NAME);
      compile_share(new_module, referenced_entities, mpi_begin_stmt, mpi_end_stmt);
      break;
    case STEP_DO:
      directive_txt = strdup(STEP_DO_NAME);
      work_stmt = compile_loop(new_module, work_stmt, &schedule, &index);
      loop_basic_workchunk_to_workchunk(stmt, &recv_l, &send_l);
      break;
    case STEP_PARALLEL_DO:
      directive_txt = strdup(STEP_PARALLEL_DO_NAME);
      compile_share(new_module, referenced_entities, mpi_begin_stmt, mpi_end_stmt);
      work_stmt = compile_loop(new_module, work_stmt, &schedule, &index);
      loop_basic_workchunk_to_workchunk(stmt, &recv_l, &send_l);
      break;
    case STEP_MASTER:
      directive_txt = strdup(STEP_MASTER_NAME);
      work_stmt = compile_master(new_module, work_stmt);
      break;
    case STEP_SINGLE:
      directive_txt = strdup(STEP_SINGLE_NAME);
      break;
    case STEP_BARRIER:
      directive_txt = string_undefined;
      work_stmt = compile_barrier(new_module, work_stmt);
      break;
    default: assert(0);
    }

  if(reduction)
    compile_reduction(reductions_l, mpi_begin_stmt, mpi_end_stmt);

  entity workchunk_id = entity_undefined;
  if(!statement_undefined_p(schedule))
    {
      workchunk_id = step_local_slice_index(new_module);
      insert_statement(mpi_begin_stmt, schedule, false);
    }

  statement compute_regions_recv, set_regions_recv, flush_regions_recv;
  statement compute_regions_send, set_regions_send, flush_regions_send;
  regions_to_statement(index, new_module, recv_l, workchunk_id, list_undefined, reductions_l,
		       &compute_regions_recv, &set_regions_recv, &flush_regions_recv);
  regions_to_statement(index, new_module, send_l, workchunk_id, send_lpur, reductions_l,
		       &compute_regions_send, &set_regions_send, &flush_regions_send);

  /* RECV */
  if(!ENDP(statement_block(compute_regions_recv)))
    {
      comment_stmt = make_plain_continue_statement();
      put_a_comment_on_a_statement(comment_stmt, strdup("\nRECV REGIONS"));
      insert_statement(mpi_begin_stmt, comment_stmt, false);
      insert_statement(mpi_begin_stmt, compute_regions_recv, false);
      insert_statement(mpi_begin_stmt, set_regions_recv, false);
      insert_statement(mpi_begin_stmt, flush_regions_recv, false);
    }
  else
    {
      free_statement(compute_regions_recv);
      free_statement(set_regions_recv);
      free_statement(flush_regions_recv);
    }

  /* SEND */
  if(!ENDP(statement_block(compute_regions_send)))
    {
      comment_stmt = make_plain_continue_statement();
      put_a_comment_on_a_statement(comment_stmt, strdup("\nSEND REGIONS"));
      insert_statement(mpi_begin_stmt, comment_stmt, false);
      insert_statement(mpi_begin_stmt, compute_regions_send, false);
      insert_statement(mpi_begin_stmt, set_regions_send, false);
      if(step_directive_type(d) != STEP_PARALLEL_DO &&
	 step_directive_type(d) != STEP_PARALLEL)
	insert_statement(mpi_end_stmt, flush_regions_send, false); // en cas de construction pararallel, le flush est fait par "compile_share"
    }
  else
    {
      free_statement(compute_regions_send);
      free_statement(set_regions_send);
      free_statement(flush_regions_send);
    }
  if(!string_undefined_p(directive_txt))
    {
      insert_statement(mpi_begin_stmt,
		       call_STEP_subroutine(RT_STEP_construct_begin, CONS(EXPRESSION, step_symbolic(directive_txt, new_module), NIL), type_undefined),
		       true);
      insert_statement(mpi_end_stmt,
		       call_STEP_subroutine(RT_STEP_construct_end, CONS(EXPRESSION, step_symbolic(directive_txt, new_module), NIL), type_undefined),
		       false);
      free(directive_txt);
    }

  if(c_module_p(get_current_module_entity()))
    outliner_patch_parameters(statements_to_outline, referenced_entities, effective_parameters, formal_parameters, new_body, mpi_begin_stmt, mpi_end_stmt);

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
      step_directive_type(d) != STEP_MASTER)
    {
      compile_omp(stmt, d);
      add_guard(&mpi_begin_stmt);
      add_guard(&mpi_end_stmt);
    }


  if(!empty_statement_p(mpi_begin_stmt))
    insert_statement(new_body, mpi_begin_stmt, true);
  if(!empty_statement_p(mpi_end_stmt))
    insert_statement(new_body, mpi_end_stmt, false);

  if (transformation == STEP_TRANSFORMATION_HYBRID &&
      step_directive_type(d) == STEP_MASTER)
    {
      add_guard(&new_body);
    }

  step_RT_set_local_declarations(new_module, new_body);
  /*
    Source file generation
  */
  set_prettyprinter_head_hook(step_head_hook);
  outliner_file(new_module, formal_parameters, &new_body);
  reset_prettyprinter_head_hook();

  step_RT_clean_local();

  statement new_stmt = outliner_call(new_module, statements_to_outline, effective_parameters);

  return new_stmt;
}

static bool compile_filter(statement stmt, list *last_module_name)
{
  if(!step_directives_bound_p(stmt))
    return true;

  bool new_module;
  int transformation = -1;
  bool is_fortran = fortran_module_p(get_current_module_entity());
  string previous_name, directive_txt, transformation_txt;
  step_directive d = step_directives_get(stmt);

  previous_name = STRING(CAR(*last_module_name));

  switch(step_directive_type(d))
    {
    case STEP_PARALLEL:
      directive_txt = strdup("PAR");
      break;
    case STEP_DO:
      directive_txt = strdup(is_fortran?"DO":"FOR");
      break;
    case STEP_PARALLEL_DO:
      directive_txt = strdup(is_fortran?"PARDO":"PARFOR");
      break;
    case STEP_MASTER:
      directive_txt = strdup("MASTER");
      break;
    case STEP_SINGLE:
      directive_txt = strdup("SINGLE");
      break;
    case STEP_BARRIER:
      directive_txt = strdup("BARRIER");
      break;
    default: assert(0);
    }

  FOREACH(STEP_CLAUSE, c, step_directive_clauses(d))
    {
      if(step_clause_transformation_p(c))
	transformation = step_clause_transformation(c);
    }
   switch(transformation)
    {
    case STEP_TRANSFORMATION_MPI:
      transformation_txt = strdup("MPI");
      new_module = true;
      break;
    case STEP_TRANSFORMATION_HYBRID:
      transformation_txt = strdup("HYBRID");
      new_module = true;
      break;
    case STEP_TRANSFORMATION_SEQ:
    case STEP_TRANSFORMATION_OMP:
      transformation_txt = strdup("");
      new_module = false;
      break;
    default:
      assert(0);
    }

  if(new_module)
    {
      string prefix;
      assert(asprintf(&prefix,"%s_%s_%s", previous_name, directive_txt, transformation_txt)>=0);
      string new_module_name = build_new_top_level_module_name(prefix, true);
      entity new_module = make_empty_subroutine(new_module_name, copy_language(module_language(get_current_module_entity())));

      free(prefix);

      *last_module_name = CONS(STRING, new_module_name, *last_module_name);
      pips_debug(1, "new_module %p : %s\n", new_module, new_module_name);
    }

  free(directive_txt);
  free(transformation_txt);

  return true;
}

static void compile_rewrite(statement stmt, list *last_module_name)
{
  if(!step_directives_bound_p(stmt))
    return;

  string new_module_name = STRING(CAR(*last_module_name));
  pips_debug(1, "stack_name current name=%s\n", new_module_name);

  step_directive d = step_directives_get(stmt);
  ifdebug(3)
    step_directive_print(d);

  compile_seq(stmt); // remove pragma "STEP"

  int transformation = -1;
  FOREACH(STEP_CLAUSE, c, step_directive_clauses(d))
    {
      if(step_clause_transformation_p(c))
	transformation = step_clause_transformation(c);
    }
  pips_debug(2,"TRANSFORMATION : %d\n", transformation);

  switch(transformation)
    {
    case STEP_TRANSFORMATION_SEQ:
      break;
    case STEP_TRANSFORMATION_OMP:
      compile_omp(stmt, d);
      break;
    case STEP_TRANSFORMATION_MPI:
    case STEP_TRANSFORMATION_HYBRID:
      compile_mpi(stmt, new_module_name, d);
      POP(*last_module_name);
      break;
    default:
      assert(0);
    }
}

bool step_compile(const char* module_name)
{
  debug_on("STEP_COMPILE_DEBUG_LEVEL");
  pips_debug(1, "considering module %s\n", module_name);

  entity module = local_name_to_top_level_entity(module_name);
  bool is_fortran = fortran_module_p(module);

  string finit_name;
  string init_name = db_build_file_resource_name(DBR_STEP_FILE, module_name, is_fortran?  STEP_GENERATED_SUFFIX_F :  STEP_GENERATED_SUFFIX_C);
  assert(asprintf(&finit_name,"%s/%s" , db_get_current_workspace_directory(), init_name)>=0);

  if(step_analysed_module_p(module_name))
    {
      statement stmt = (statement)db_get_memory_resource(DBR_CODE, module_name, false);
      set_current_module_entity(module);
      set_current_module_statement(stmt);
      step_directives_init();
      load_step_comm();

      statement_effects rw_effects = (statement_effects)db_get_memory_resource(DBR_REGIONS, module_name, false);
      statement_effects cummulated_rw_effects = (statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, false);
      set_rw_effects(rw_effects);
      set_cumulated_rw_effects(cummulated_rw_effects);

      /* Code transformation */
      list last_module_name = CONS(STRING, (string)module_name, NIL);
      gen_context_recurse(stmt, &last_module_name, statement_domain, compile_filter, compile_rewrite);

      if (entity_main_module_p(module))
	{
	  statement init_stmt = call_STEP_subroutine(fortran_module_p(get_current_module_entity())?
						     RT_STEP_init_fortran_order:
						     RT_STEP_init_c_order, NIL, type_undefined);
	  statement finalize_stmt = call_STEP_subroutine(RT_STEP_finalize, NIL, type_undefined);

	  insert_statement(stmt, init_stmt, true);
	  insert_statement(find_last_statement(stmt), finalize_stmt, true);
	}

      reset_cumulated_rw_effects();
      reset_rw_effects();
      free_statement_effects(cummulated_rw_effects);
      free_statement_effects(rw_effects);

      reset_step_comm();
      step_directives_reset();
      reset_current_module_statement();
      reset_current_module_entity();

      /* File generation */
      text code_txt = text_named_module(module, module, stmt);
      bool saved_b1 = get_bool_property("PRETTYPRINT_ALL_DECLARATIONS");
      bool saved_b2 = get_bool_property("PRETTYPRINT_STATEMENT_NUMBER");
      set_bool_property("PRETTYPRINT_ALL_DECLARATIONS", true);
      set_bool_property("PRETTYPRINT_STATEMENT_NUMBER", false);

      FILE *f = safe_fopen(finit_name, "w");
      print_text(f, code_txt);
      safe_fclose(f, finit_name);

      set_bool_property("PRETTYPRINT_ALL_DECLARATIONS", saved_b1);
      set_bool_property("PRETTYPRINT_STATEMENT_NUMBER", saved_b2);
      free_text(code_txt);
      free_statement(stmt);
    }
  else
    {
      /*
	generated source: no analyse and no compilation necessary. Keep the source as it is.
      */
      string fsource_file;
      string source_file = db_get_memory_resource(is_fortran?DBR_INITIAL_FILE:DBR_C_SOURCE_FILE, module_name, true);
      assert(asprintf(&fsource_file,"%s/%s" , db_get_current_workspace_directory(), source_file)>=0);
      safe_copy(fsource_file, finit_name);
    }

  DB_PUT_FILE_RESOURCE(DBR_STEP_FILE, module_name, finit_name);

  pips_debug(1, "End\n");
  debug_off();
  return true;
}
