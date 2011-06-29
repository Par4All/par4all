/* Copyright 2009 Alain Muller

This file is part of STEP.

The program is distributed under the terms of the GNU General Public
License.
*/

#include "defines-local.h"
#include "effects-generic.h"
#include "effects-convex.h"


/*
  genere un statement de la forme : 
  array_region(bound_name,dim,index) = expr_bound
*/
static statement bound_to_statement(entity mpi_module, list expr_bound, entity array_region, string bound_name, int dim, list index)
{
  pips_assert("expression", !ENDP(expr_bound));
  entity op=entity_undefined;  
  statement s;
  list dims=CONS(EXPRESSION,step_symbolic(bound_name, mpi_module),
		 CONS(EXPRESSION,int_to_expression(dim),gen_full_copy_list(index)));

  bool is_fortran=fortran_module_p(get_current_module_entity());
  if (!is_fortran)
    {
      list l=NIL;
      FOREACH(EXPRESSION,e,dims)
	{
	  l=CONS(EXPRESSION, MakeBinaryCall(CreateIntrinsic(MINUS_C_OPERATOR_NAME),
					  e,
					  int_to_expression(1)),l);
	}
      dims=l;
    }

  expression expr = reference_to_expression(make_reference(array_region, dims));
  
  if ( gen_length(expr_bound) != 1 )
    {
      if(strncmp(bound_name,STEP_INDEX_SLICE_LOW_NAME,strlen(bound_name))==0)
	op=entity_intrinsic(MAX_OPERATOR_NAME);
      else if (strncmp(bound_name,STEP_INDEX_SLICE_UP_NAME,strlen(bound_name))==0)
	op=entity_intrinsic(MIN_OPERATOR_NAME);
      else
	pips_internal_error("unexpected bound name %s",bound_name);
      s = make_assign_statement(expr, call_to_expression(make_call(op,expr_bound)));
    }
  else
    s = make_assign_statement(expr, EXPRESSION(CAR(expr_bound)));

  return s;
}


/*
  si equality=TRUE :
  recherche la premiere contrainte d'egalite portant sur la variable PHI du systeme de contraintes sys et transforme cette contrainte en 2 expressions, l'une traduisant le contrainte d'inferiorite (ajouter a expr_l), l'autre traduisant la contrainte de superiorite (ajoutée a expr_u)
  retourne true si une contrainte d'egalite a ete trouve, false sinon.

  si equality=FALSE :
  traduit l'ensemble des contraintes d'inegalite portant sur la variable PHI.
  Les expressions traduisant une contrainte d'inferiorie (de superiorite) sont ajoutes à la liste expr_l (expr_u)
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
	  bool low_bound=false;
	  bool up_bound=false;

	  //construction des expressions d'affectation
	  if(VECTEUR_NUL_P(v))
	    expr = int_to_expression(0);	    
	  else
	    {
	      if (coef_phi > 0) //contrainte de type : coef_phi*phi  <= "vecteur"
		{
		  up_bound = true;
		  for (coord = v; coord!=NULL; coord=coord->succ) 
		    val_of(coord) = -val_of(coord);
		  coef_phi = -coef_phi;
		}
	      low_bound = !up_bound;
	      
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
  mets a jour expr_l (et expr_u) liste des expressions des contraites low (et up) portant sur la variable PHI du systeme de contraintes sys
*/
static void systeme_to_expression(entity phi, Psysteme sys, list *expr_l, list *expr_u)
{
  pips_assert("empty list", ENDP(*expr_l) && ENDP(*expr_u));
  
  // recherche et transformation des contraintes d'equalites portant sur phi
  bool equality=contraintes_to_expression(true, phi, sys, expr_l, expr_u);
  
  // recherche et transformation des contraintes d'inequalites portant sur phi
  if (!equality) contraintes_to_expression(false, phi, sys, expr_l, expr_u);
}


/*
  Ajoute les contraines lies aux dimensions du tableau d'origine selon la dimension dim
  Genere les statements traduisant les contraines inférieur et suppérieur :
     array_region(LOW,dim,index,) = expr_l
     array_region(UP,dim,index,) = expr_u
*/
static statement build_regionBounds(entity mpi_module, dimension bounds_d, entity array_region, int index_dim, list index_slice, list expr_l, list expr_u)
{
  if(ENDP(expr_l))
    expr_l=CONS(EXPRESSION,copy_expression(dimension_lower(bounds_d)),expr_l);
  if(ENDP(expr_u)) 
    expr_u=CONS(EXPRESSION,copy_expression(dimension_upper(bounds_d)),expr_u);
  
  return make_block_statement(CONS(STATEMENT,bound_to_statement(mpi_module, expr_l,array_region,STEP_INDEX_SLICE_LOW_NAME,index_dim,index_slice),
				   CONS(STATEMENT,bound_to_statement(mpi_module, expr_u,array_region,STEP_INDEX_SLICE_UP_NAME,index_dim,index_slice),NIL)));
}

statement step_build_arraybounds(entity mpi_module, list regions, entity (*array_region_)(entity, region), bool send)
{
  list l_block = NIL;
  list set_R = NIL;

  // generation des statements specifiant l'espace des indices du tableau d'origine
  FOREACH(REGION, reg, regions)
    {
      entity array = region_entity(reg);
      if (!io_effect_entity_p(array))
	{
	  entity array_region=(*array_region_)(mpi_module,reg);
	  list bounds_array = variable_dimensions(type_variable(entity_type(region_entity(reg))));
	  int dim=1;
	  statement s;
	  
	  FOREACH(DIMENSION, bounds_dim, bounds_array)
	    {
	      l_block=CONS(STATEMENT, build_regionBounds(mpi_module, bounds_dim,array_region,dim++,NIL,NIL,NIL), l_block);
	    }
	  if (send)
	    s=build_call_STEP_set_send_region(array, int_to_expression(1), array_region, false, false);
	  else
	    s=build_call_STEP_set_recv_region(array, int_to_expression(1), array_region);

	  set_R=CONS(STATEMENT, s, set_R);
	}
    }

  return make_block_statement(gen_nreverse(gen_nconc(set_R,l_block)));
}

/*
  Transformation d'un système de contrainte (une region de tableau) en statements
  Exemple :
  C  <reg(PHI1,PHI2)-write-EXACT-{1<=PHI1, PHI1<=N, I_LOW<=PHI2, 1<=PHI2, PHI2<=I_UP, PHI2<=N}>
     array_region(IDX_SLICE_LOW,1,IDX) = 1
     array_region(IDX_SLICE_UP,1,IDX) = N
     array_region(IDX_SLICE_LOW,2,IDX) = MAX(I_LOW, 1)
     array_region(IDX_SLICE_UP,2,IDX) = MIN(I_UP, N)
*/
static statement step_build_compute_region(entity mpi_module,region reg,entity (*array_region_)(entity, region))
{
  list expr_l,expr_u;
  int dim=0;
  Psysteme sys = region_system(reg);
  entity array_region=(*array_region_)(mpi_module,reg);
  list bounds_array = variable_dimensions(type_variable(entity_type(region_entity(reg))));
  list index_slice=CONS(EXPRESSION,entity_to_expression(step_local_slice_index(mpi_module)), NIL);

  //ajout de la region en commentaire au body
  reset_action_interpretation();
  string commentaire = text_to_string(text_region(reg));
  statement statmt = make_continue_statement(entity_empty_label());
  put_a_comment_on_a_statement(statmt, commentaire);
  list l_body=CONS(STATEMENT, statmt, NIL);
  pips_debug(2,"region : %s\n",commentaire);

  /* 
     on parcour dans l'ordre des indices (PHI1, PHI2, ...)
     chaque PHIi correspond a une dimension d
  */
  FOREACH(EXPRESSION, expr, reference_indices(effect_any_reference(reg)))
    {
      entity phi = reference_variable(syntax_reference(expression_syntax(expr)));
      dimension bounds_d = DIMENSION(gen_nth(dim,bounds_array));   // gen_nth numerote les element a partir de 0 et ...
      dim++; // ... les tableaux de region numerote les dimensions a partir de 1

      /*
	on determine les listes d'expression expr_l (et expr_u) correspondant aux
	contraites low (et up) portant sur la variable PHI courrante
	ex: L <= PHI1 + 1  et PHI1 -1 <= U
	expr_l contient l'expression (L-1) et expr_u contient l'expression (U+1)
      */
      expr_l = expr_u = NIL;
      systeme_to_expression(phi, sys, &expr_l, &expr_u);

      // ajout contraintes liees aux bornes d'indexation du tableau pour la dimension courante et generation des statements
      l_body=CONS(STATEMENT, build_regionBounds(mpi_module, bounds_d, array_region, dim, index_slice, expr_l, expr_u), l_body);
    }
  return make_block_statement(gen_nreverse(l_body));
}

statement step_build_arrayRegion(entity mpi_module, list regions, entity (*array_region_)(entity, region), entity loop_index)
{
  pips_assert("loop index",!entity_undefined_p(loop_index));

  /*
    Generation des statements traduisant le systeme de contraintes (region reg) pour different work_chunk
  */
  list l_body_region = NIL;
  statement body_region;
  FOREACH(REGION, reg, regions)
    {
      l_body_region = CONS(STATEMENT, step_build_compute_region(mpi_module, reg, array_region_), l_body_region);
    }
  
  if (!ENDP(l_body_region))
    {
      /*
	Generation de :
	CALL STEP_GETLOOPBOUNDS(IDX-1, J_LOW, J_UP)
      */
      entity slice_index = step_local_slice_index(mpi_module);
      expression expr_id_workchunk = make_op_exp(PLUS_OPERATOR_NAME, int_to_expression(-1), entity_to_expression(slice_index));
      expression expr_index_low = entity_to_expression(step_local_loop_index(mpi_module, STEP_INDEX_LOW_NAME(loop_index)));
      expression expr_index_up = entity_to_expression(step_local_loop_index(mpi_module, STEP_INDEX_UP_NAME(loop_index)));
      if (!fortran_module_p(get_current_module_entity()))
	{
	  expr_index_low = make_call_expression(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), CONS(EXPRESSION, expr_index_low, NIL));
	  expr_index_up = make_call_expression(entity_intrinsic(ADDRESS_OF_OPERATOR_NAME), CONS(EXPRESSION, expr_index_up, NIL));
	}
      list args = CONS(EXPRESSION, expr_id_workchunk,
		       CONS(EXPRESSION, expr_index_low,
			    CONS(EXPRESSION, expr_index_up, NIL)));
      statement statmt = call_STEP_subroutine(RT_STEP_GetLoopBounds, args, type_undefined);
      
      body_region = make_block_statement(CONS(STATEMENT, statmt,
					      gen_nreverse(l_body_region))); 
      /*
	Generation de la boucle parcourant les differents work_chunk
      */
      range rng = make_range(int_to_expression(1), step_local_size( mpi_module), int_to_expression(1));
      instruction loop_instr = make_instruction_loop(make_loop(slice_index, rng, body_region, entity_empty_label(), make_execution_sequential(), NIL));

      return make_block_statement(CONS(STATEMENT, make_statement(entity_empty_label(),
								 STATEMENT_NUMBER_UNDEFINED,
								 STATEMENT_ORDERING_UNDEFINED,
								 strdup("\n"),
								 loop_instr, NIL, string_undefined,
								 empty_extensions()), NIL));
    }
  else
    return make_continue_statement(entity_undefined);
}
