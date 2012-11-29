/* Copyright 2007-2012 Alain Muller, Frederique Silber-Chaussumier

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include "defines-local.h"

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

static bool region_interlaced_p(list send_regions, region r)
{
  bool interlaced = false;
  pips_debug(1, "begin send_regions = %p, r = %p\n", send_regions, r);

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
  pips_debug(1, "end\n");
  return interlaced;
}

static bool reduction_p(set reductions_l[STEP_UNDEF_REDUCE], entity e)
{
  bool is_reduction = false;
  int op;

  pips_debug(1, "begin\n");

  for(op=0; !is_reduction && op<STEP_UNDEF_REDUCE; op++)
    is_reduction = set_belong_p(reductions_l[op], e);

  pips_debug(1, "end\n");
  return is_reduction;
}


static bool comm_partial_p(list send_regions, region r)
{
  bool partial_p = true;
  pips_debug(1, "begin\n");
  if(region_write_p(r))
    {
      bool first = true;

      FOREACH(REGION, reg, send_regions)
	{
	  if(combinable_regions_p(reg, r))
	    {
	      assert(first);
	      partial_p = step_partial_p(reg);
	      first = false;
	    }
	}
    }
  pips_debug(1, "end partial_p = %d\n", partial_p);
  return partial_p;
}

/*
  genere un statement de la forme :
  array_region(bound_name,dim,index) = expr_bound
*/
static statement bound_to_statement(entity mpi_module, list expr_bound, entity array_region, string bound_name, int dim, list index)
{
  pips_debug(1, "begin array_region = %p\n", array_region);

  pips_assert("expression", !ENDP(expr_bound));
  statement s;
  entity op = entity_undefined;
  list dims = CONS(EXPRESSION, step_symbolic_expression(bound_name, mpi_module),
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

  pips_debug(1, "end\n");
  return s;
}

static list phi_free_contraints_to_expressions(bool equality, region reg)
{
  Psysteme sys = region_system(reg);
  Pcontrainte c;
  list expression_l = NIL;

  entity comparator;
  if(c_module_p(get_current_module_entity()))
    comparator = entity_intrinsic(equality?C_EQUAL_OPERATOR_NAME:C_LESS_OR_EQUAL_OPERATOR_NAME);
  else
    comparator = entity_intrinsic(equality?EQUAL_OPERATOR_NAME:LESS_OR_EQUAL_OPERATOR_NAME);

  for(c = equality?sc_egalites(sys):sc_inegalites(sys); !CONTRAINTE_UNDEFINED_P(c); c = c->succ)
    {
      int coef_phi = 0;
      FOREACH(EXPRESSION, expr, reference_indices(effect_any_reference(reg)))
	{
	  entity phi = reference_variable(syntax_reference(expression_syntax(expr)));
	  coef_phi = VALUE_TO_INT(vect_coeff((Variable)phi,c->vecteur));
	  if(coef_phi != 0)
	    break;
	}
      if(coef_phi == 0)
	{
	  expression expr = MakeBinaryCall(comparator,
					   make_vecteur_expression(c->vecteur),
					   int_to_expression(0));
	  expression_l = CONS(EXPRESSION, expr, expression_l);
	}
    }
  return expression_l;
}

/*
  si equality=true :

  recherche la premiere contrainte d'egalite portant sur la variable
  PHI du systeme de contraintes sys et transforme cette contrainte en
  2 expressions, l'une traduisant le contrainte d'inferiorite (ajouter
  a expr_l), l'autre traduisant la contrainte de superiorite (ajoutée
  a expr_u)

  retourne true si une contrainte d'egalite a ete trouve, false sinon.

  si equality=false :
  traduit l'ensemble des contraintes d'inegalite portant sur la variable PHI.

  Les expressions traduisant une contrainte d'inferiorie (de
  superiorite) sont ajoutes à la liste expr_l (expr_u)

*/
/*
  Voir make_contrainte_expression
*/

static bool contraintes_to_expression(bool equality, entity phi, Psysteme sys, list *expr_l, list *expr_u)
{
  Pcontrainte c;
  bool found_equality=false;

  pips_debug(1, "begin\n");
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
  pips_debug(1, "end\n");
  return found_equality;
}

/*
  Generation of

  // compute_region_statement
  STEP_RR_a[IDX-1][1-1][STEP_INDEX_SLICE_LOW-1] = MAX(STEP_i_LOW, 0);
  STEP_RR_a[IDX-1][1-1][STEP_INDEX_SLICE_UP-1] = MIN(STEP_i_UP, 99999);

  or

  if( phi_free_contrainte_1 && ... && phi_free_contrainte_n ) ) {
  // compute_region_statement
     STEP_RR_a[IDX-1][1-1][STEP_INDEX_SLICE_LOW-1] = MAX(STEP_i_LOW, 0);
     STEP_RR_a[IDX-1][1-1][STEP_INDEX_SLICE_UP-1] = MIN(STEP_i_UP, 99999);
  }
  else {
  // empty_region_statement
     STEP_RR_a[IDX-1][1-1][STEP_INDEX_SLICE_LOW-1] = 99999;
     STEP_RR_a[IDX-1][1-1][STEP_INDEX_SLICE_UP-1] = 0;
  }

*/


static void compute_region(entity mpi_module, region reg, entity array_region, bool loop_p, statement *compute_regions_stmt)
{
  pips_debug(1, "begin array_region = %p\n", array_region);
  Psysteme sys = region_system(reg);
  list bounds_array = variable_dimensions(type_variable(entity_type(region_entity(reg))));

  if(ENDP(bounds_array))
    {
      pips_debug(0,"Current array : %s\n", entity_name(region_entity(reg)));
      pips_assert("defined array bounds", 0);
    }

  list index_slice = NIL;
  if (loop_p)
    {
      entity workchunk_id = step_local_slice_index(mpi_module);
      index_slice = CONS(EXPRESSION, entity_to_expression(workchunk_id), index_slice);
    }

  statement empty_region_stmt = statement_undefined;
  list phi_free_contraints = NIL;
  phi_free_contraints = gen_nconc(phi_free_contraints_to_expressions(true, reg), phi_free_contraints);
  phi_free_contraints = gen_nconc(phi_free_contraints_to_expressions(false, reg), phi_free_contraints);
  if (!ENDP(phi_free_contraints))
    empty_region_stmt =  make_empty_block_statement();

  statement compute_region_stmt = make_empty_block_statement();
  int dim = 0;

  // on parcourt dans l'ordre des indices (PHI1, PHI2, ...) chaque PHIi correspond a une dimension dim

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

      ifdebug(2)
	{
	  pips_debug(2, "b1\n");
	  print_statement(b1);
	  pips_debug(2, "b2\n");
	  print_statement(b2);
	}
      insert_statement(compute_region_stmt, b1, false);
      insert_statement(compute_region_stmt, b2, false);

      if (!ENDP(phi_free_contraints))
	{
	  expr_l = CONS(EXPRESSION, copy_expression(dimension_upper(bounds_d)), NIL);
	  expr_u = CONS(EXPRESSION, copy_expression(dimension_lower(bounds_d)), NIL);

	  b1 = bound_to_statement(mpi_module, expr_l, array_region, STEP_INDEX_SLICE_LOW_NAME, dim, index_slice);
	  b2 = bound_to_statement(mpi_module, expr_u, array_region, STEP_INDEX_SLICE_UP_NAME, dim, index_slice);

	  insert_statement(empty_region_stmt, b1, false);
	  insert_statement(empty_region_stmt, b2, false);
	}

    }

  if (!ENDP(phi_free_contraints))
    {
      entity and_op = entity_intrinsic(c_module_p(get_current_module_entity())?C_AND_OPERATOR_NAME:AND_OPERATOR_NAME);
      expression cond_expr = expression_list_to_binary_operator_call(phi_free_contraints, and_op);

      insert_statement(*compute_regions_stmt,
		       instruction_to_statement(make_instruction(is_instruction_test,
								 make_test(cond_expr,
									   compute_region_stmt,
									   empty_region_stmt))),
		       false);
      string comment_str = strdup(" Inverted bounds correspond to empty regions\n Used when work concerns specific data ex: print A[5]\n In this case, only concerned process sends non empty regions\n");
      put_a_comment_on_a_statement(empty_region_stmt, comment_str);
    }
  else
    insert_statement(*compute_regions_stmt, compute_region_stmt, false);

  pips_debug(1, "end\n");
}

static bool region_reduction_p(set reductions_l[STEP_UNDEF_REDUCE], region reg)
{
  entity array = region_entity(reg);

  return region_write_p(reg) && reduction_p(reductions_l, array);
}

static void generate_call_set_regionarray(entity mpi_module, region reg, entity array_region, bool loop_p, bool is_reduction, bool is_interlaced, statement *set_regions_stmt)
{
  pips_debug(1, "begin\n");

  entity array = region_entity(reg);

  expression expr_nb_workchunk = loop_p?entity_to_expression(get_entity_step_commsize(mpi_module)):int_to_expression(1);
  statement set_stmt = region_read_p(reg)?
    build_call_STEP_set_recvregions(array, expr_nb_workchunk, array_region):
    build_call_STEP_set_sendregions(array, expr_nb_workchunk, array_region, is_interlaced, is_reduction);
  insert_statement(*set_regions_stmt, set_stmt, false);

  pips_debug(1, "end\n");
}

/* pourquoi flush et non pas alltoall ? */
static void generate_call_stepalltoall(entity mpi_module, region reg, bool is_reduction, bool is_interlaced, bool is_partial, statement *stepalltoall_stmt)
{
  entity array = region_entity(reg);

  if(!is_reduction)
    {
      statement comm_stmt = build_call_STEP_AllToAll(mpi_module, array, is_partial, is_interlaced);
      insert_statement(*stepalltoall_stmt, comm_stmt, false);
    }
}

static void region_to_statement(entity mpi_module, region reg, bool loop_p, bool is_reduction, bool is_interlaced, bool is_partial, statement *compute_regions_stmt, statement *set_regionarray_stmt, statement *stepalltoall_stmt)
{
  pips_debug(1,"begin mpi_module = %s, region = %p\n", entity_name(mpi_module), reg);

  entity array = region_entity(reg);
  pips_debug(2, "array = %s\n", entity_name(array));

  /*
    Elimination of redundancies
  */
  region_system(reg) = sc_safe_elim_redund(region_system(reg));

  /*
    Add region description in comments
  */
  string str_eff = text_to_string(text_rw_array_regions(CONS(REGION, reg, NIL)));
  statement comment_stmt = make_plain_continue_statement();
  put_a_comment_on_a_statement(comment_stmt, str_eff);

  insert_statement(*compute_regions_stmt, comment_stmt, false);


  /*
    Create STEP_RR and STEP_SR array entities
   */
  expression expr_nb_region = loop_p?step_symbolic_expression(STEP_MAX_NB_LOOPSLICES_NAME, mpi_module):expression_undefined;

  /* bug bug bug si on inverse les instr expr_nb_region et region_array_name alors bug. POURQUOI??*/

  string region_array_name = region_read_p(reg)?STEP_RR_NAME(array):STEP_SR_NAME(array);
  pips_debug(2, "region_array_name = %s\n", region_array_name);

  entity region_array = step_local_regionArray(mpi_module, array, region_array_name, expr_nb_region);
  pips_debug(2, "region_array = %p\n", region_array);

  /*
    Compute regions
  */
  compute_region(mpi_module, reg, region_array, loop_p, compute_regions_stmt);

  /*
    Set regions
  */

  generate_call_set_regionarray(mpi_module, reg, region_array, loop_p, is_reduction, is_interlaced, set_regionarray_stmt);

  /*
    stepalltoall_stmt

    Pour les régions SEND ET (?) RECV
  */

  generate_call_stepalltoall(mpi_module, reg, is_reduction, is_interlaced, is_partial, stepalltoall_stmt);

  pips_debug(1, "end\n");
}

static void transform_regions_to_statement(entity mpi_module, list regions_l, bool loop_p, list send_as_comm_l, set reductions_l[STEP_UNDEF_REDUCE],
				 statement *compute_regions_stmt, statement *set_regionarray_stmt, statement *stepalltoall_stmt)
{
  pips_debug(1, "begin regions_l = %p, pure_send_l = %p\n", regions_l, send_as_comm_l);

  *compute_regions_stmt = make_empty_block_statement();
  *set_regionarray_stmt = make_empty_block_statement();
  *stepalltoall_stmt = make_empty_block_statement();

  FOREACH(REGION, reg, regions_l)
    {
      bool is_reduction = region_reduction_p(reductions_l, reg);
      bool is_interlaced = region_interlaced_p(send_as_comm_l, reg);
      bool is_partial = comm_partial_p(send_as_comm_l, reg);

      region_to_statement(mpi_module, reg, loop_p, is_reduction, is_interlaced, is_partial, compute_regions_stmt, set_regionarray_stmt, stepalltoall_stmt);
    }

  pips_debug(1, "end\n");
}

static void add_workchunk_loop(entity mpi_module, bool loop_p, loop loop_stmt, statement *compute_regions_stmt)
{
  pips_debug(1, "begin\n");
  /*
    ajout de la boucle parcourant les workchunks
    suppression du test !entity_undefined_p(index) : redondant avec loop_p?
  */
  if( loop_p &&
      !ENDP(statement_block(*compute_regions_stmt)))
    {
      generate_call_get_workchunk_loopbounds(mpi_module, loop_stmt, compute_regions_stmt);

      generate_loop_workchunk(mpi_module, compute_regions_stmt);
    }
  pips_debug(1, "end\n");
}

void compile_regions(entity new_module, list regions_l, bool loop_p, loop loop_stmt, list send_as_comm_l, set reductions_l[STEP_UNDEF_REDUCE], statement mpi_begin_stmt, statement mpi_end_stmt)
{
  statement compute_regions_stmt, set_regions_stmt, flush_regions_stmt;

  pips_debug(1, "begin regions_l = %p\n", regions_l);

  transform_regions_to_statement(new_module, regions_l, loop_p, send_as_comm_l, reductions_l,
				 &compute_regions_stmt, &set_regions_stmt, &flush_regions_stmt);

  add_workchunk_loop(new_module, loop_p, loop_stmt, &compute_regions_stmt);
  generate_call_flush(&flush_regions_stmt);

  if(!ENDP(statement_block(compute_regions_stmt)))
    {
      string comment;
      statement comment_stmt = make_plain_continue_statement();
      insert_statement(mpi_begin_stmt, comment_stmt, false);
      insert_statement(mpi_begin_stmt, compute_regions_stmt, false);
      insert_statement(mpi_begin_stmt, set_regions_stmt, false);

      if (list_undefined_p(send_as_comm_l))
	{
	  comment = strdup("\nRECV REGIONS");
	  insert_statement(mpi_begin_stmt, flush_regions_stmt, false);
	}
      else
	{
	  comment = strdup("\nSEND REGIONS");
	  insert_statement(mpi_end_stmt, flush_regions_stmt, false);
	}
      put_a_comment_on_a_statement(comment_stmt, comment);
    }
  else
    {
      free_statement(compute_regions_stmt);
      free_statement(set_regions_stmt);
      free_statement(flush_regions_stmt);
    }

}

