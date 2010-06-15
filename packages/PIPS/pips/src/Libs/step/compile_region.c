/* Copyright 2009 Alain Muller

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


/*
  genere un statement de la forme : 
  array_region(bound_name,dim,index) = expr_bound
*/
static statement bound_to_statement(entity mpi_module, list expr_bound, entity array_region, string bound_name, int dim, list index)
{
  pips_assert("expression", !ENDP(expr_bound));
  entity op=entity_undefined;  
  statement s;
  expression expr= reference_to_expression(make_reference(array_region,
							  CONS(EXPRESSION,step_symbolic(bound_name, mpi_module),
							       CONS(EXPRESSION,int_to_expression(dim),gen_full_copy_list(index))))); 
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
  retourne TRUE si une contrainte d'egalite a ete trouve, FALSE sinon.

  si equality=FALSE :
  traduit l'ensemble des contraintes d'inegalite portant sur la variable PHI.
  Les expressions traduisant une contrainte d'inferiorie (de superiorite) sont ajoutes à la liste expr_l (expr_u)
*/
static boolean contraintes_to_expression(bool equality, entity phi, Psysteme sys, list *expr_l, list *expr_u)
{
  Pcontrainte c;
  boolean found_equality=FALSE;

  for(c = equality?sc_egalites(sys):sc_inegalites(sys); !found_equality && !CONTRAINTE_UNDEFINED_P(c); c = c->succ)
    {
      int coef_phi=VALUE_TO_INT(vect_coeff((Variable)phi,c->vecteur));
      if(coef_phi != 0)
	{
	  expression expr;
	  Pvecteur coord,v = vect_del_var(c->vecteur, (Variable)phi);
	  boolean low_bound=FALSE;
	  boolean up_bound=FALSE;

	  //construction des expressions d'affectation
	  if(VECTEUR_NUL_P(v))
	    expr = int_to_expression(0);	    
	  else
	    {
	      if (coef_phi > 0) //contrainte de type : coef_phi*phi  <= "vecteur"
		{
		  up_bound = TRUE;
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
  boolean equality=contraintes_to_expression(true, phi, sys, expr_l, expr_u);
  
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


static statement step_build_assigne_region0(entity mpi_module, list regions, entity (*array_region_)(entity, region), entity loop_index)
{
  list l_block=NIL;
  list index0 = entity_undefined_p(loop_index)?NIL:CONS(EXPRESSION,int_to_expression(0),NIL);

  // generation des statements pour la region d'indice 0 specifiant l'espace des indices du tableau d'origine
  FOREACH(REGION, reg, regions)
    {
      entity array_region=(*array_region_)(mpi_module,reg);
      list bounds_array = variable_dimensions(type_variable(entity_type(region_entity(reg))));
      int dim=1;
      FOREACH(DIMENSION, bounds_dim, bounds_array)
	{
	  l_block=CONS(STATEMENT, build_regionBounds(mpi_module, bounds_dim,array_region,dim++,index0,NIL,NIL), l_block);
	}
    }
  gen_full_free_list(index0);

  return make_block_statement(gen_nreverse(l_block));
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
  list l_body=CONS(STATEMENT, make_call_statement(CONTINUE_FUNCTION_NAME, NIL, entity_empty_label(),commentaire), NIL);
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
  statement assigne_dim = step_build_assigne_region0(mpi_module, regions, array_region_, loop_index);

  if(!entity_undefined_p(loop_index) && !empty_statement_p(assigne_dim))
    {
      /*
	Generation de :
	I_LOW = STEP_I_LOOPSLICES(I_SLICE_LOW, IDX) 
	I_UP = STEP_I_LOOPSLICES(I_SLICE_UP, IDX)
      */
      entity loopSliceArray = step_local_loopSlices(mpi_module,loop_index);
      entity slice_index = step_local_slice_index(mpi_module);
      expression index_low = entity_to_expression(step_local_loop_index(mpi_module, STEP_INDEX_LOW_NAME(loop_index)));
      expression index_up = entity_to_expression(step_local_loop_index(mpi_module, STEP_INDEX_UP_NAME(loop_index)));
      
      expression low_expr = reference_to_expression(make_reference(loopSliceArray,
								   CONS(EXPRESSION, step_symbolic(STEP_INDEX_SLICE_LOW_NAME, mpi_module),
									CONS(EXPRESSION, entity_to_expression(slice_index), NIL))));
      expression up_expr = reference_to_expression(make_reference(loopSliceArray,
								  CONS(EXPRESSION, step_symbolic(STEP_INDEX_SLICE_UP_NAME, mpi_module),
								       CONS(EXPRESSION, entity_to_expression(slice_index), NIL))));
      /*
	Generation des statements traduisant le systeme de contraintes (region reg) pour different work_chunk
      */
      list l_body_region = NIL;
      statement body_region;
      FOREACH(REGION, reg, regions)
	{
	  l_body_region = CONS(STATEMENT, step_build_compute_region(mpi_module, reg, array_region_), l_body_region);
	}
      body_region = make_block_statement(CONS(STATEMENT, make_assign_statement(index_low,low_expr),
					      CONS(STATEMENT, make_assign_statement(index_up,up_expr),
						   gen_nreverse(l_body_region)))); 
      /*
	Generation de la boucle parcourant les differents work_chunk
      */
      range rng = make_range(make_expression_1(), step_local_size(mpi_module), make_expression_1());
      instruction loop_instr = make_instruction_loop(make_loop(slice_index, rng, body_region, MakeLabel(""), make_execution_sequential(), NIL));

      insert_comments_to_statement(assigne_dim,
				   concatenate("\nC     Put array boundaries into region arrays (SR: Send region)",
					       "\nC     First dimension: lower and upper bounds of each slice",
					       "\nC     Second dimension: for each dimension of the original array",
					       "\nC     Third dimension: store the boundaries of the local chunk.\n",
					       "\nC     The first element stores initial boundaries,",
					       "\nC     then one element for each process\n",NULL));
      return make_block_statement(CONS(STATEMENT,assigne_dim,
				       CONS(STATEMENT, make_statement(entity_empty_label(),
								      STATEMENT_NUMBER_UNDEFINED,
								      STATEMENT_ORDERING_UNDEFINED,
								      strdup("\nC     Region computation\n"),
								      loop_instr, NIL, string_undefined,
								      empty_extensions()), NIL)));
    }
  else
    return assigne_dim;
}
