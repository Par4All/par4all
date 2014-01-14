/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
 /* Code Generation for Distributed Memory Machines
  *
  * Code generating routine for a given partitioning and a given loop nest
  *
  * File: code.c
  *
  * PUMA, ESPRIT contract 2701
  *
  * Francois Irigoin, Corinne Ancourt, Lei Zhou
  * 1991
  */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "effects.h"

#include "matrice.h"
#include "tiling.h"

#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
/* #include "generation.h" */
#include "movements.h"

#include "wp65.h"

Value offset_dim1 = VALUE_ZERO;
Value offset_dim2 = VALUE_ZERO;

static entity tile_indice_entity1;
static bool ref_in_statement1;
static Value var_minmax = VALUE_ZERO;
static Variable var_to_evaluate;

static void ref_found_p(reference ref)
{
    entity m = reference_variable(ref);
    ref_in_statement1 = ref_in_statement1 || (strcmp(entity_local_name(m),
				entity_local_name(tile_indice_entity1)) == 0);
 	
}

static bool reference_in_statement_p(statement l, entity v)
{
    tile_indice_entity1= v;
    ref_in_statement1 = false;
    gen_recurse(l,
		reference_domain,
		gen_true,
		ref_found_p);
    return(ref_in_statement1);
}


static void eval_var(reference ref)
{
    cons * ind1;
    expression expr1 = expression_undefined;
    expression expr2 = expression_undefined;
    cons *expr=NIL;
    Value coeff = VALUE_ZERO;
    bool debut=true;
    ind1 = reference_indices(ref);

    if (ind1) {
	for (debut =true; ind1 != NULL; ind1 = CDR(ind1))
	{ normalized norm1;
	  expr1 = EXPRESSION(CAR(ind1));
	  norm1 = (normalized) NORMALIZE_EXPRESSION(expr1);
	  if (normalized_linear_p(norm1)) {
	      Pvecteur pv1 = (Pvecteur) normalized_linear(norm1);
	      fprintf(stderr,"\n expression->vecteur:");
		  vect_fprint(stderr,pv1,(string(*)(Variable))entity_local_name);
	      if (value_notzero_p(coeff = vect_coeff(var_to_evaluate,pv1)))
	      {
		  vect_chg_coeff(&pv1,var_to_evaluate,VALUE_ZERO);
		  vect_add_elem(&pv1,TCST,
				value_mult(var_minmax,coeff));
		  fprintf(stderr,"\n nouvelle expression:");
		  vect_fprint(stderr,pv1,(string(*)(Variable))entity_local_name);
		  expr2 = make_vecteur_expression(pv1);
	      }
	      else expr2 = expr1;
	      print_words(stderr,words_expression(expr2, NIL));
	  }

	  if (debut) {
	      expr = CONS(EXPRESSION,expr2,NIL);
	      print_words(stderr,words_expression(expr2, NIL));
	      debut = false;
	  }
	  else expr = gen_nconc(expr,CONS(EXPRESSION,expr2,NIL));

      }
	reference_indices(ref) = expr;
    }

}


void eval_variable_in_statement(entity module,statement s,Variable v,int min)
{
    var_to_evaluate = v;
    var_minmax = min;
    ifdebug(8) {
	(void) fprintf(stderr, "Loop body :\n");
	wp65_debug_print_text(module, s);
    }
    gen_recurse(s,
		reference_domain,
		gen_true,
		eval_var);
    ifdebug(8) {
	(void) fprintf(stderr, "New loop body :\n");
	wp65_debug_print_text(module, s);
    }

}


void tile_change_of_basis(Psysteme tile_domain,Pbase initial_basis,Pbase tile_basis,
Pbase tile_init_basis,tiling tile)
{ 
    int dim = base_dimension(initial_basis);
    int l,c;
    for(l=1; l <= dim; l++) {
	Pcontrainte eq;
	Pvecteur v = VECTEUR_NUL;
	int td = base_dimension(tile_basis);
	for(c=1; c<=td; c++)
	    vect_add_elem(&v, variable_of_rank(tile_basis, c), 
			  (Value) ACCESS((matrice) tiling_tile(tile), dim, l, c));
	
	vect_add_elem(&v, variable_of_rank(tile_init_basis, l), VALUE_MONE);
	vect_add_elem(&v, TCST,
		      vect_coeff(variable_of_rank(initial_basis, l),
				 (Pvecteur) tiling_origin(tile)));
	eq = contrainte_make(v);
	sc_add_egalite(tile_domain, eq);
    }
}





/* make_scanning_over_tiles()
 *
 * generates a nest of loops to enumerate all tiles containing at
 * least one iteration in iteration_domain, and even sometimes
 * zero because a rational projection is used; empty tiles
 * are taken care of at a lower level to make sure that no iterations
 * are performed.
 *
 * The following system is built:
 *
 *   ->   ->
 * B i <= b
 *
 * ->     -->  ->
 * t1 = P t0 + o
 *
 * ->      -1  ->  -->     ---->
 * 0 <= k P  ( i - t1 ) <= (k-1)
 *
 * where (B,b) defines the iteration domain, t1 is the tile origin in the
 * initial basis, t0 is the tile origin in the tile basis, P is the
 * partitioning matrix, o is the partitioning origin, i is an iteration,
 * and k is the denominator of the inverse of P.
 *
 * i and t1 are eliminated to obtain constraints on t0 only. These constraints
 * are used to derive the loop bounds.
 *
 * This piece of code could also be used to generate tiled code for
 * a shared memory machine without any change.
 *
 * Notes:
 *  - the outermost loop is assumed parallel;
 *  - the outermost loop is statically distributed over the processors
 *    if proc_id is defined as a formal parameter (compute code)
 *  - proc_id has to be derived from the outermost loop index for the
 *    memory code, when it is a local variable
 *
 * Algorithm described in PPoPP'91
 */
statement 
make_scanning_over_tiles(
    entity module,
    list body,
    entity proc_id,
    int pn,
    tiling tile,
    Pbase initial_basis,
    Pbase tile_basis_in_tile_basis,
    Pbase tile_basis_in_initial_basis,
    Psysteme iteration_domain,
    int first_parallel_level,
    int last_parallel_level)
{
    statement s = statement_undefined;
    entity ind;
    Psysteme tile_domain = sc_dup(iteration_domain);
    Psysteme ordered_tile_domain = SC_UNDEFINED;
    Psysteme sctmp;
    int id = base_dimension(initial_basis);
    int td = base_dimension(tile_basis_in_tile_basis);
    int l,t;
    int keep_indice[11]; 
    Value min,max;
    int first_indice;  
    debug(8,"make_scanning_over_tiles", "begin for module %s\n",
	  module_local_name(module));

    /* only fully-dimensional partitioning for the time being */
    pips_assert("make_scanning_over_tiles",id==td); 

    /* add change of basis equations to iteration domain:
       tile_basis_in_tile_basis to tile_basis_in_initial_basis;
       they would be of no use for a shared memory machine */

   tile_change_of_basis(tile_domain,initial_basis,
			tile_basis_in_tile_basis,
			tile_basis_in_initial_basis,tile);

    /* add the tile membership inequalities */
    tile_domain = sc_append(tile_domain, 
			    tile_membership((matrice) tiling_tile(tile),
					    tile_basis_in_initial_basis,
					    initial_basis));

    /* update the basis for system tile_domain */
    base_rm(sc_base(tile_domain));
    sc_base(tile_domain) = BASE_NULLE;
    sc_creer_base(tile_domain);

    tile_domain=sc_normalize(tile_domain);

    ifdebug(8) {
	(void) fprintf(stderr, "Tile basis -> initial basis:\n");
	sc_fprint(stderr, tile_domain,(string(*)(Variable)) entity_local_name);
    }

    /* get rid of initial indices; they would be preserved to generate
       shared memory code */
    for(l=1; l <= id; l++) {
	entity ind = (entity) variable_of_rank(initial_basis, l);
	tile_domain = sc_projection_pure(tile_domain, (Variable) ind);
    }

    /* get rid of tile_basis_in_initial_basis; we might as well (?) keep
       them and get rid of tile_basis_in_tile_basis */
    for(l=1; l <= td; l++) {
	entity ind = (entity) variable_of_rank(tile_basis_in_initial_basis, l);
	tile_domain = sc_projection_pure(tile_domain, (Variable) ind);
    }

    ifdebug(8) {
	(void) fprintf(stderr, "Constraints on tile indices:\n");
	sc_fprint(stderr, tile_domain, (string(*)(Variable))entity_local_name);
    }

    /* apply a row echelon transformation */
    ordered_tile_domain = 
	new_loop_bound(tile_domain, tile_basis_in_tile_basis);
    sc_transform_eg_in_ineg(ordered_tile_domain);

    ifdebug(8) {
	(void) fprintf(stderr, "Ordered constraints on tile indices:\n");
	sc_fprint(stderr, ordered_tile_domain, (string(*)(Variable))entity_local_name);
    }



    /* transform these constraints into a loop nest with the right body,
       starting with the innermost loop */
    s = make_block_statement(body);
    
    pips_debug(9, "body statement:");
    ifdebug(9) wp65_debug_print_text(module, s);

    for(t = td; t >= 1; t--) { 
	keep_indice[t]=true;
	sctmp = sc_dup(ordered_tile_domain);
	sc_minmax_of_variable(sctmp,
			      variable_of_rank(tile_basis_in_tile_basis, t),
			      &min,
			      &max); 
	ind = (entity) variable_of_rank(tile_basis_in_tile_basis, t);
	if ( value_eq(min,max) && !reference_in_statement_p(s,ind)) { 
	    fprintf(stderr,"indice de tuile inutile %d, %s\n",t,
		    entity_local_name(ind));
	    keep_indice[t] = false;
	}
    }
   
    for (t=1;t<=td && keep_indice[t]== false;t++);
    first_indice = (t==td+1) ? td:t;
    ifdebug(7) 
	fprintf(stderr,"first tile index %d, %s\n",first_indice,
		entity_local_name
		((entity) variable_of_rank(tile_basis_in_tile_basis, t)));

    for(t = td; t >= 1; t--)
    {
	expression lower;
	expression upper;
	range r;
	loop new_l;
	entity new_label;
	statement cs = statement_undefined;
	statement ps = statement_undefined;

	ind = (entity) variable_of_rank(tile_basis_in_tile_basis, t);

	/* optimization : Loop indices that are constant and 
	   don't appear in the program body are not generated */

	if (keep_indice[t]){ 

	    make_bound_expression((Variable) ind,
				  tile_basis_in_tile_basis,
				  ordered_tile_domain,
				  &lower, &upper);

	    /* distribute work statically on processors using the outermost
	       loop (assumed parallel!) if proc_id is properly defined;
	       this should not be the case for bank tiles */
	    if(t !=first_parallel_level || 
	       !storage_formal_p(entity_storage(proc_id))) 
		r = make_range(lower, upper, int_to_expression(1));
	    else {
		normalized n = NORMALIZE_EXPRESSION(lower);
		if(normalized_linear_p(n) 
		   && VECTEUR_NUL_P((Pvecteur) normalized_linear(n)))
		    lower = entity_to_expression(proc_id);
		else
		    lower = MakeBinaryCall
			(local_name_to_top_level_entity(PLUS_OPERATOR_NAME), 
			 lower,
			 entity_to_expression(proc_id));
		r = make_range(lower, upper, int_to_expression(pn));
	    }

	    /* I may need a definition for PROC_ID = MOD(I_0, PROCESSOR_NUMBER) */
	    if(t==first_parallel_level && 
	       !storage_formal_p(entity_storage(proc_id))) {
		ps = make_assign_statement
		    (entity_to_expression(proc_id),
		     MakeBinaryCall
		     (local_name_to_top_level_entity(MOD_INTRINSIC_NAME),
		      entity_to_expression(ind),
		      int_to_expression(pn)));
	    }
	    else ps = statement_undefined;
	 
	    /* I need new labels and new continues for my loops!
	       make_loop_label() needs (at least) a module name */
	    new_label = make_loop_label(9000, module);
	    cs = make_continue_statement(new_label);

	    if(instruction_block_p(statement_instruction(s))) 
		(void) gen_nconc(instruction_block(statement_instruction(s)),
				 CONS(STATEMENT, cs, NIL));
	    else 
		s = make_block_statement(CONS(STATEMENT, s,
					 CONS(STATEMENT, cs,
					      NIL)));
	    
	    /* Now, s is certainly a block; prefix it with proc_id def */
	    if(ps != statement_undefined)
		instruction_block(statement_instruction(s)) = 
		    CONS(STATEMENT, ps,
			 instruction_block(statement_instruction(s)));

	    new_l = make_loop(ind, r, s, 
			      new_label,
			      make_execution(is_execution_sequential,UU), 
			      NIL);
	    s = loop_to_statement(new_l);
	}
    }

    statement_comments(s) = 
	strdup(concatenate("\nC     To scan the tile set for ",
			   module_local_name(module), "\n", NULL));
    
    ifdebug(8) {
	(void) fprintf(stderr, "Loop nest over tiles:\n");
	wp65_debug_print_text(module, s);
    }

    pips_debug(8,"end\n");

    return s;
}

/* make_scanning_over_one_tile():
 *
 * generates a nest of loops to enumerate all iterations contained in
 * one tile; the loop bounds are such that empty tiles execute no
 * iteration at all;
 *
 * The following system is built:
 *
 *   ->   ->
 * B i <= b
 *
 * ->     -->  ->
 * t1 = P t0 + o
 *
 * ->  -->  ->
 * i = t1 + l
 *
 * ->      -1  ->  -->     ---->
 * 0 <= k P  ( i - t1 ) <= (k-1)
 *
 * where (B,b) defines the iteration domain, t1 is the tile origin in the
 * initial basis, t0 is the tile origin in the tile basis, P is the
 * partitioning matrix, o is the partitioning origin, i is an iteration,
 * and k is the denominator of the inverse of P. l is an iteration in
 * the local (to the current tile) basis.
 *
 * Because the loops over the tiles are built with t1 and because we need
 * to access the copy with local coordinates, i and t0 are eliminated.
 * 
 * A few changes would make this function generate loops for a shared
 * memory machine. The local_basis would be useless and the initial
 * basis should be chosen as indices for the loops so as not to have
 * to update the loop body. So l would not be introduced and i would
 * not be projected.
 *
 * Algorithm described in PPoPP'91.
 */
statement make_scanning_over_one_tile(module, body, 
				      tile, initial_basis, local_basis,
				      tile_basis_in_tile_basis,
				      tile_basis_in_initial_basis,
				      iteration_domain,
				      first_parallel_level,
				      last_parallel_level)
entity module;
statement body;
tiling tile;
Pbase initial_basis;
Pbase local_basis;
Pbase tile_basis_in_tile_basis;
Pbase tile_basis_in_initial_basis;
Psysteme iteration_domain;
int first_parallel_level,last_parallel_level;
{
    Psysteme tile_domain = sc_dup(iteration_domain);
    Psysteme ordered_tile_domain = SC_UNDEFINED;
    Psysteme origin_domain = SC_UNDEFINED;
    int id = base_dimension(initial_basis);
    int td = base_dimension(tile_basis_in_tile_basis);
    int l,t,i;
    statement s = statement_undefined;
    Pvecteur pv;

    debug(8,"make_scanning_over_one_tile", "begin for module %s\n",
	  module_local_name(module));

    /* only fully-dimensional partitioning for the time being */
    pips_assert("make_scanning_over_one_tile",id==td); 

    /* add change of basis equations to iteration domain:
       tile_basis_in_tile_basis to tile_basis_in_initial_basis */
  
 
     tile_change_of_basis(tile_domain,initial_basis,
			tile_basis_in_tile_basis,
			tile_basis_in_initial_basis,tile);

    /* add translation equations from initial basis to local basis
       using tile_basis_in_initial_basis: i == t1 + l */
 
    for(l=1; l <= id; l++) {
	Pcontrainte eq;
	Pvecteur v = VECTEUR_NUL;
	vect_add_elem(&v, variable_of_rank(initial_basis, l), VALUE_ONE);
	vect_add_elem(&v, variable_of_rank(tile_basis_in_initial_basis, l),
		      VALUE_MONE);
	vect_add_elem(&v, variable_of_rank(local_basis, l), VALUE_MONE);
	eq = contrainte_make(v);
	sc_add_egalite(tile_domain, eq);
    }

 
    /* add the tile membership inequalities */
    tile_domain = sc_append(tile_domain, 
			    tile_membership((matrice) tiling_tile(tile),
					    tile_basis_in_initial_basis,
					    initial_basis));

    /* update the basis for system tile_domain */
    base_rm(sc_base(tile_domain));
    sc_base(tile_domain) = BASE_NULLE;
    sc_creer_base(tile_domain);

    ifdebug(8) {
	(void) fprintf(stderr, "Full system before projections:\n");
	sc_fprint(stderr, tile_domain,(string(*)(Variable)) entity_local_name);
    }

    /* get rid of tile_basis_in_initial_basis; we might as well (?) keep
       them and get rid of tile_basis_in_tile_basis */
    for(l=1; l <= td; l++) {
	entity ind = (entity) variable_of_rank(tile_basis_in_initial_basis, l);
	tile_domain = sc_projection_pure(tile_domain, (Variable) ind);
    }

    ifdebug(8) {
	(void) fprintf(stderr, 
		       "Constraints on local tile indices parametrized by tile origin and initial indices:\n");
	sc_fprint(stderr, tile_domain,(string(*)(Variable)) entity_local_name);
    }

    /* get rid of initial indices */
    for(l=1; l <= id; l++) {
	entity ind = (entity) variable_of_rank(initial_basis, l);
	tile_domain = sc_projection_pure(tile_domain, (Variable) ind);
    }

    ifdebug(8) {
	(void) fprintf(stderr, 
		       "Constraints on local tile indices parametrized by tile origin:\n");
	sc_fprint(stderr, tile_domain, (string(*)(Variable))entity_local_name);
    }

    /* TEMPTATIVE */

    /* compute general information on loop bound origins
       this is done to take into account information carried by the
       outer loops, scanning the tile set */
    origin_domain = sc_dup(tile_domain);

    /* get rid of local indices */
    for(l=1; l <= id; l++) {
	entity ind = (entity) variable_of_rank(local_basis, l);
	origin_domain = sc_projection_pure(origin_domain, (Variable) ind);
    }

    ifdebug(8) {
	(void) fprintf(stderr, 
		       "Absolute constraints on tile origins:\n");
	sc_fprint(stderr, origin_domain,(string(*)(Variable)) entity_local_name);
    }

    /* inject this redundant information */
    tile_domain = sc_append(origin_domain, tile_domain);

    ifdebug(8) {
	(void) fprintf(stderr, 
		       "Constraints on local tile indices parametrized by tile origin, with redundant information:\n");
	sc_fprint(stderr, tile_domain,(string(*)(Variable)) entity_local_name);
    }

    pv =tile_basis_in_tile_basis;
    for (i=1; i<= last_parallel_level &&  !VECTEUR_NUL_P(pv); i++, pv = pv->succ);
    
    for ( ;  !VECTEUR_NUL_P(pv); pv = pv->succ) {
	sc_force_variable_to_zero(tile_domain,vecteur_var(pv));
    }


    /* END OF TEMPTATIVE SECTION */

    /* apply a row echelon transformation */
    ordered_tile_domain = new_loop_bound(tile_domain, local_basis);
    sc_transform_eg_in_ineg(ordered_tile_domain);
    
    ifdebug(8) {
	(void) fprintf(stderr, "Ordered constraints on local indices:\n");
	sc_fprint(stderr, ordered_tile_domain,(string(*)(Variable)) entity_local_name);
    }

    /* transform these constraints into a loop nest with the right body,
       starting with the innermost loop */
  
    s = body;

    /* test pour optimiser le nid de boucles genere */
    for (t =id; t >= 1; t--) {
	Value min,max; 
	Variable var = variable_of_rank(local_basis, t);
	
	Psysteme sctmp = sc_dup(ordered_tile_domain);
	sc_minmax_of_variable(sctmp,var,&min,&max); 
    }


    for(t = id; t >= 1; t--) {
	expression lower;
	expression upper;
	range r;
	loop new_l;
	entity ind;
	entity new_label;
	statement cs;

	ind = (entity) variable_of_rank(local_basis, t);
	make_bound_expression((Variable) ind,
			      local_basis,
			      ordered_tile_domain,
			      &lower, &upper);
	r = make_range(lower, upper, int_to_expression(1));
	/* I need new labels and new continues for my loops!
	   make_loop_label() needs (at least) a module name */
	new_label = make_loop_label(9000, module);
	cs = make_continue_statement(new_label);
	if(instruction_block_p(statement_instruction(s)))
	    (void) gen_nconc(instruction_block(statement_instruction(s)),
			     CONS(STATEMENT, cs, NIL));
	else 
	    s = make_block_statement(CONS(STATEMENT, s,
					  CONS(STATEMENT, cs, NIL)));
	
	new_l = make_loop(ind, r, s, 
			  new_label,
			  make_execution(is_execution_sequential,UU), NIL);
	s = loop_to_statement( new_l);
    }
    statement_comments(s) = 
	strdup("C           To scan each iteration of the current tile\n");
    ifdebug(8) {
	(void) fprintf(stderr, "Loop nest over tiles:\n");
	wp65_debug_print_text(module, s);
    }

    debug(8,"make_scanning_over_one_tile", "end\n");

    return s;
}



list make_compute_block(module, body, offsets, r_to_llv, 
			tile, initial_basis, local_basis,
			tile_basis_in_tile_basis,
			tile_basis_in_initial_basis,
			iteration_domain,first_parallel_level,last_parallel_level)
entity module;
statement body; /* IN, 
		   but modified by side effect to avoid copying statements */
Pvecteur offsets;
hash_table r_to_llv;
tiling tile;
Pbase initial_basis;
Pbase local_basis;
Pbase tile_basis_in_tile_basis;
Pbase tile_basis_in_initial_basis;
Psysteme iteration_domain;
int first_parallel_level,last_parallel_level;
{
    statement s;

    list lt = NIL;
    lt = reference_conversion_statement(module,body, &lt, r_to_llv, offsets, initial_basis,
				   tile_basis_in_tile_basis,local_basis,tile);
    body = make_block_statement(gen_nconc(lt,instruction_block(statement_instruction(body))));
    s = make_scanning_over_one_tile(module, body,
				    tile, initial_basis, local_basis,
				    tile_basis_in_tile_basis,
				    tile_basis_in_initial_basis,
				    iteration_domain,first_parallel_level,last_parallel_level);

    return CONS(STATEMENT, s, NIL);
}

/* void reference_conversion_statement(body, r_to_llv, offsets, initial_basis,
 * local_basis): 
 *
 * All references in body which appear in r_to_llv
 * are replaced by references to one of the local variables 
 * associated via the r_to_llv hash_table; the choice of one specific
 * local variable is a function of offsets, which is used to generate
 * pipelined code (not implemented).
 *
 * Statement numbers are set to STATEMENT_NUMBER_UNDEFINED.
 */
list reference_conversion_statement(module,body, lt,r_to_llv, offsets, initial_basis, 
					 tile_indices,local_basis,tile)

entity module;
statement body;
list *lt;
hash_table r_to_llv;
Pvecteur offsets;
Pbase initial_basis,tile_indices,local_basis;
tiling tile;
{
    instruction i = statement_instruction(body);

    pips_debug(8, "begin statement: \n");
    ifdebug(8) {
	wp65_debug_print_text(entity_undefined, body);
    }

    statement_number(body) = STATEMENT_NUMBER_UNDEFINED;

    switch(instruction_tag(i)) {
    case is_instruction_block:
	MAPL(cs, {
	    *lt = reference_conversion_statement(module,STATEMENT(CAR(cs)),lt,
					   r_to_llv, offsets,
					   initial_basis,tile_indices, local_basis,tile);
	}, 
	     instruction_block(i)); 
	return(*lt);
	break;
    case is_instruction_test:
	pips_internal_error("Reference conversion not implemented for tests");
	break;
    case is_instruction_loop: 

	
	*lt = reference_conversion_statement(module,
					     loop_body(instruction_loop(i)),
					     lt, r_to_llv, offsets,
					     initial_basis, 
					     tile_indices,local_basis,tile); 
	 return(*lt);
	break;
    case is_instruction_goto:
	pips_internal_error("Unexpected goto (in restructured code)");
	break;
    case is_instruction_call:
	/* the function is assumed to be unchanged */
	MAP(EXPRESSION, argument,
	{  
	    *lt = reference_conversion_computation(module,lt,argument,
						   initial_basis,
						   tile_indices,
						   local_basis, tile);
	    reference_conversion_expression(argument, 
					    r_to_llv, offsets,
					    initial_basis, local_basis);
	}, 
	     call_arguments(instruction_call(i)));
	 return(*lt);
	break;
    case is_instruction_unstructured:
	pips_internal_error("Reference conversion not implemented for unstructureds");
	break;
    default:
	break;
    }

    debug(8,"reference_conversion_statement", "return statement: \n");
    ifdebug(8) {
	wp65_debug_print_text(entity_undefined, body);
    } 
    return(*lt);
} 


list reference_conversion_computation(
    entity compute_module,
    list *lt,
    expression expr,
    Pbase initial_basis,
    Pbase tile_indices,
    Pbase tile_local_indices,
    tiling tile)
{
    syntax s = expression_syntax(expr);
   
    switch(syntax_tag(s)) {
    case is_syntax_reference: {
	reference r = syntax_reference(s);
	entity rv = reference_variable(r);
	int i;
	
	if((i=rank_of_variable(initial_basis, (Variable) rv)) > 0) {
	    
	    entity ent1 = make_new_module_variable(compute_module,200);
	    Pvecteur pv2 = vect_new((Variable) ent1, VALUE_ONE);
	    Pvecteur pvt = VECTEUR_NUL;
	    expression exp1,exp2;
	    statement stat= statement_undefined;
	    Pvecteur pv = make_loop_indice_equation(initial_basis,tile, pvt,
						    tile_indices,
						    tile_local_indices,i);
	    AddEntityToDeclarations(ent1,compute_module);   
	    reference_variable(r) = ent1 ;
	    exp1= make_vecteur_expression(pv);
	    exp2 = make_vecteur_expression(pv2);
	    stat = make_assign_statement(exp2,exp1);
	    *lt = gen_nconc(*lt,CONS(STATEMENT,stat,NIL));
	} 
	return (*lt);
	break;
    }
    case is_syntax_call:
	/* the called function is assumed to be unchanged */
	MAPL(ce, {
	    *lt = reference_conversion_computation(compute_module,lt,
						   EXPRESSION(CAR(ce)),initial_basis,
					     tile_indices, tile_local_indices,tile);
	}, 
	     call_arguments(syntax_call(s)));
	     return (*lt);
	break;
    default:
	pips_internal_error("Unexpected syntax tag %d", syntax_tag(s));
	break;
    }
    return(*lt);
}


entity 
reference_translation(reference r,Pbase initial_basis,Pbase local_basis)
{
    int i;
    entity rv = reference_variable(r);
    if((i=rank_of_variable(initial_basis, (Variable) rv)) > 0)
	return((entity) variable_of_rank(local_basis, i));
    else return(entity_undefined);
}

void reference_conversion_expression(e, r_to_llv, offsets, initial_basis,
				     local_basis)
expression e;
hash_table r_to_llv;
Pvecteur offsets;
Pbase initial_basis;
Pbase local_basis;
{
    syntax s = expression_syntax(e);
    int i;
    debug(8,"reference_conversion_expression", "begin expression:");
    ifdebug(8) {
      print_words(stderr, words_expression(e, NIL));
	(void) fputc('\n', stderr);
    }

    switch(syntax_tag(s)) {
    case is_syntax_reference: {
	reference r = syntax_reference(s);
	entity rv = reference_variable(r);
	list llv;

	if((llv = (list) hash_get(r_to_llv, (char *) r))
	   != (list) HASH_UNDEFINED_VALUE) {
	    /* no pipeline, select the first entity by default */
	    entity new_v = ENTITY(CAR(llv));
	    reference_variable(r) = new_v;
	}
	else 
	    if ((i=rank_of_variable(initial_basis, 
				    (Variable) rv)) > 0)
	
		reference_variable(r)=
		    (entity) variable_of_rank(local_basis, i);
	
	MAPL(ce, {
	    reference_conversion_expression(EXPRESSION(CAR(ce)), 
					    r_to_llv, offsets,
					    initial_basis, local_basis);
	}, 
	     reference_indices(r));
	break;
    }
    case is_syntax_range:
	pips_internal_error("Ranges are not (yet) handled");
	break;
    case is_syntax_call:
	/* the called function is assumed to be unchanged */
	MAPL(ce, {
	    reference_conversion_expression(EXPRESSION(CAR(ce)), 
					    r_to_llv, offsets,
					    initial_basis, local_basis);
	}, 
	     call_arguments(syntax_call(s)));
	break;
    default:
	pips_internal_error("Unexpected syntax tag %d", syntax_tag(s));
	break;
    }

    debug(8,"reference_conversion_expression", "end expression:\n");
    ifdebug(8) {
      print_words(stderr, words_expression(e, NIL));
	(void) fputc('\n', stderr);
    }
}

/* This function checks if two references have a uniform dependence. 
 * It assumes that some verifications have been made before. The two 
 * references r1 and r2 must reference the same array with the same 
 * dimension.
 *
 * FI: could/should be moved in ri-util/expression.c
 */

bool uniform_dependence_p(r1,r2)
reference r1,r2;
{

    bool uniform = true;
    cons * ind1, *ind2;

    debug(8,"uniform_dependence_p", "begin\n");

    ind1 = reference_indices(r1);
    ind2 = reference_indices(r2); 
    for (; uniform && ind1 != NULL && ind2!= NULL;
	 ind1 = CDR(ind1), ind2=CDR(ind2)) 
    { 
    expression expr1= EXPRESSION(CAR(ind1));
    expression expr2= EXPRESSION(CAR(ind2));
	normalized norm1 = (normalized) NORMALIZE_EXPRESSION(expr1);
	normalized norm2 = (normalized) NORMALIZE_EXPRESSION(expr2);
	if (normalized_linear_p(norm1) && normalized_linear_p(norm2)) {
	    Pvecteur pv1 = (Pvecteur) normalized_linear(norm1);
	    Pvecteur pv2 = (Pvecteur) normalized_linear(norm2);
	    Pvecteur pv3 = vect_substract(pv1,pv2);
	    if (vect_size(pv3) >1 || 
		((vect_size (pv3)==1) && (vecteur_var(pv3) != TCST)))
		uniform = false;
	    vect_rm(pv3);
	}
    }
    debug(8,"uniform_dependence_p", "end\n");
    return(uniform);
}


/* This function classifies the references in lists. All the references 
 * belonging to the same list are uniform dependent references 
*/
list classify_reference(llr,r)
list llr;
reference r;
{
    list plr,lr,bllr,lr2;
    list blr = NIL;
    bool trouve = false;
    reference r2;

    debug(8,"classify_reference", "begin\n");
    for (plr = llr,bllr = NIL; plr != NIL && !trouve; plr = CDR(plr)) {
	for (lr = LIST(CAR(plr)), blr=NIL; lr!= NIL && !trouve;
	     lr = CDR(lr)) {
	    r2 = REFERENCE(CAR(lr));
	    if (uniform_dependence_p(r2,r)) 
		trouve = true;
	}
	blr = (trouve) ? CONS(REFERENCE,r,LIST(CAR(plr))) :
	    LIST(CAR(plr));
	bllr = CONS(LIST,blr,bllr);  
    }
    if (!trouve) {
	lr2 = CONS(REFERENCE,r,NIL);
	bllr = CONS(LIST,lr2,bllr);
    }
    debug(8,"classify_reference", "end\n");
    return(bllr);

}

/* build_sc_with_several_uniform_ref():
 *
 * Build the set of constraints describing the array function for the 
 * set of uniform references to the array and update the system 
 * describing the domain with constraints on new variables 
 */
Psysteme build_sc_with_several_uniform_ref(module,lr,sc_domain,new_index_base)
entity module;
cons * lr;
Psysteme sc_domain;
Pbase *new_index_base;
{
    cons * expr;

    Psysteme sc_array_function = sc_new();
    Psysteme sc2;
    Pcontrainte pc,pc3=NULL;
    Pvecteur pv,pv1,pv2,pv3;
    Pvecteur pv_sup = VECTEUR_NUL;
    normalized norm;
    reference r;
    cons * ind; 
    Variable var=VARIABLE_UNDEFINED;
    Value cst = VALUE_ZERO;
    Pcontrainte pc1,pc2;
    expression expr1;

    debug(8,"build_sc_with_several_uniform_ref", "begin\n");
    ifdebug(8) {
	list crefs;
	(void) fprintf(stderr, "Associated reference list:");
	for(crefs=lr; !ENDP(crefs); POP(crefs)) {
	    reference r1 = REFERENCE(CAR(crefs));
	    print_words(stderr, words_reference(r1, NIL));
	    if(ENDP(CDR(crefs)))
		(void) putc('\n', stderr);
	    else
		(void) putc(',', stderr);
	}
    }


    r= REFERENCE(CAR(lr));
    ind = reference_indices(r);
    
    /* build the system of constraints describing the array function. 
       One constraint for each dimension of the array is built. 
       This constraint is put in the system of inequalities of 
       sc_array_function */

    for (expr = ind;expr!= NULL; expr = CDR(expr)) {
	expr1 = EXPRESSION(CAR(expr));
	norm = (normalized) NORMALIZE_EXPRESSION(expr1);
	if (normalized_linear_p(norm)) {
	    pv1 = (Pvecteur) normalized_linear(norm);
	    pc =  contrainte_make(vect_dup(pv1));
	    if (sc_array_function->inegalites == NULL) {
		sc_array_function->inegalites = pc;
	    }
	    else pc3->succ = pc;
	    pc3=pc;
	    sc_array_function->nb_ineq ++;
	}
	else {
	    pips_internal_error("Non-linear subscript expression");
	}
    }
    sc_creer_base(sc_array_function);

    /* update the system describing the image function with the others array 
       functions having a uniform dependence with the first one 
       for example, assume we have  A(i,j) and A(i+2,j+3)  
       then the final system will be :
       i + 2 X1 <= 0
       j - 3 X1 <= 0
       X1 is new variable
       */
    sc2 =sc_dup(sc_dup(sc_array_function));
    for (lr = CDR(lr); lr != NIL;lr = CDR(lr)) {
	bool new_variable = false;
	r= REFERENCE(CAR(lr));
	ind = reference_indices(r);
	for (expr = ind,pc = sc_array_function->inegalites,
	     pc2 = sc2->inegalites;
	     expr!= NULL; expr = CDR(expr), pc=pc->succ,pc2 = pc2->succ) {

	    norm = (normalized)  NORMALIZE_EXPRESSION(EXPRESSION(CAR(expr)));
	    pv1 = (Pvecteur) normalized_linear(norm);
	    pv3 = vect_substract(pv1,pc2->vecteur);
	    if (vect_size(pv3) == 1) {
		if (value_notzero_p(cst=vecteur_val(pv3))) {
		    if (!new_variable) {
			new_variable = true;
			var = sc_add_new_variable_name(module,
						    sc_array_function);
		    
			vect_chg_coeff(&pv_sup,var,VALUE_ONE);
		    }
		    vect_chg_coeff(&pc->vecteur,var,cst);
		}
	    }
	    else if (vect_size(pv3) >1) 
		pips_internal_error("Non uniform dependent references");
	}
    }

    /* add to the system of constraints describing the domain the 
       set of constraints on the new variables. If X1,X2,...Xn are these
       new variables the set of constraints added is:
       0 <= X1 <= 1
       0 <= X2 <= 1 - X1
       0 <= X3 <= 1 - X1 -X2
       0<= Xn <= 1 - X1 - X2 ... -Xn-1
       */
    for (pv = pv_sup; !VECTEUR_NUL_P(pv); pv = pv->succ) {
	sc_domain->base = vect_add_variable(sc_domain->base,pv->var);
	sc_domain->dimension++;
	*new_index_base = vect_add_variable(*new_index_base,pv->var);
	pv1 = vect_dup(pv);
	vect_chg_coeff(&pv1,TCST,VALUE_MONE);
	pc1= contrainte_make(pv1);
	sc_add_ineg(sc_domain,pc1);
	pv2 = vect_dup(pv);
	vect_chg_sgn(pv2);
	pc2 = contrainte_make(pv2);
	sc_add_ineg(sc_domain,pc2);
    }

    ifdebug(8) { 
	(void) fprintf(stderr," The array function :\n");
	(void) sc_fprint(stderr,sc_array_function,(string(*)(Variable))entity_local_name);
    }

    debug(8,"build_sc_with_several_uniform_ref", "end\n");


    return(sc_array_function);
}

static void initialize_offsets(list lt)
{
    list ltmp = LIST(CAR(lt));
    reference ref1 = REFERENCE(CAR(ltmp));
    list lind = reference_indices(ref1);
    expression expr1 = EXPRESSION(CAR(lind));
    normalized norm = (normalized) NORMALIZE_EXPRESSION(expr1);
    Pvecteur  pv1 = (Pvecteur) normalized_linear(norm);

    offset_dim1 =  vect_coeff(TCST,pv1);
    if (CDR(lind) != NIL) { 
	expr1 = EXPRESSION(CAR(CDR(lind)));
	norm = (normalized) NORMALIZE_EXPRESSION(expr1);
	pv1 = (Pvecteur) normalized_linear(norm);
	offset_dim2 =  vect_coeff(TCST,pv1);
    }
	
}

static void nullify_offsets()
{ offset_dim1=offset_dim2=0;}

void make_store_blocks(
entity initial_module,
entity compute_module,
entity memory_module,
entity var,         /* entity  */
entity shared_variable,      /* emulated shared variable for example ES_A */
entity local_variable,       /* local variable for example L_A_1_1*/
list lrefs,
hash_table r_to_ud,
Psysteme sc_domain,        /* domain of iteration */
Pbase index_base,          /* index basis */
Pbase bank_indices,        /* contains the index describing the bank:
			      bank_id, L (ligne of bank) and O (offset in 
			      the ligne) */
Pbase tile_indices,         /* contains the local indices  LI, LJ of  tile */
Pbase loop_body_indices,
entity Proc_id,              /* corresponds to a processeur identicator */
int pn, int bn, int ls,      /* bank number and line size (depends on the 
			      machine) */
statement * store_block,
statement * bank_store_block,
int first_parallel_level,
int last_parallel_level)
{
    list ldr;
    list lldr = NIL;
    list lrs;
    Psysteme sc_image,sc_array_function;
    int n,dim_h;
    Pbase const_base;
    Pbase new_index_base= BASE_NULLE;
    Pbase new_tile_indices = BASE_NULLE;
    reference r;
    bool bank_code;			/* is true if it is the generation 
					   of code for bank false if it is 
					   for engine */
    bool receive_code;		/* is true if the generated code 
					   must be a RECEIVE, false if it 
					   must be a SEND*/
    statement stat1,stat2,sb,bsb;
    cons * bst_sb = NIL;
    cons * bst_bsb = NIL;
    Pbase var_id;
    Pvecteur ppid = vect_new((char *) Proc_id, VALUE_ONE);
    Psysteme sc_image2= SC_UNDEFINED;
    debug(8,"make_store_blocks",
	  "begin variable=%s, shared_variable=%s, local_variable=%s\n",
	  entity_local_name(var), entity_local_name(shared_variable),
	  entity_local_name(local_variable));

    ifdebug(8) {
	list crefs;
	(void) fprintf(stderr, "Associated reference list:");
	for(crefs=lrefs; !ENDP(crefs); POP(crefs)) {
	    reference r1 = REFERENCE(CAR(crefs));
	    print_words(stderr, words_reference(r1, NIL));
	    if(ENDP(CDR(crefs)))
		(void) putc('\n', stderr);
	    else
		(void) putc(',', stderr);
	}
    }


    /* Cases where the references are scalar variables */

    for (lrs =lrefs ; !ENDP(lrs) ; POP(lrs)) {
	r = REFERENCE(CAR(lrs));
	if (reference_indices(r) ==NIL) {
	    receive_code = false;
	    *store_block=make_movement_scalar_wp65(compute_module,
						   receive_code,
						   r,Proc_id);
	    receive_code = true;
	    *bank_store_block=
		make_movement_scalar_wp65(memory_module,receive_code,
					  r,(entity) bank_indices->var);
	    return;
	}
    }

    /* In the other cases the references must be classified in lists of 
       uniform dependent references */

    for ( lrs = lrefs; lrs != NIL ; lrs = CDR(lrs) ) {
	r = REFERENCE(CAR(lrefs));
	if (  (intptr_t)hash_get(r_to_ud, r) == (intptr_t)is_action_write) 
	    lldr = classify_reference(lldr,r);
    }

    /* For each list of uniform dependent references, "store_block" and 
       "bank_store_block" are computed */

    for (ldr = lldr;ldr != NIL;ldr = CDR(ldr)) {
	
	sc_array_function=
	    build_sc_with_several_uniform_ref(initial_module,
					      LIST(CAR(ldr)),
					      sc_domain,&new_index_base);
	new_index_base = vect_add(new_index_base,
				  vect_dup(index_base));

	sc_image =  sc_image_computation(initial_module,
					 var,sc_domain,
					 sc_array_function,new_index_base,
					 &const_base,Proc_id,bank_indices,
					 tile_indices,&new_tile_indices,
					 pn,bn,ls,
					 &n,&dim_h);
	nullify_offsets();
	sc_image2 = sc_dup(sc_image);
	bank_code = true ;
	receive_code = true;
	var_id = (Pbase) vect_new(vecteur_var(ppid),
				  vecteur_val(ppid));
	stat1 = movement_computation(memory_module,false,bank_code,
				     receive_code,
				     shared_variable,sc_image,
				     const_base,bank_indices,
				     new_tile_indices,
				     var_id,loop_body_indices,n,dim_h);
	initialize_offsets(ldr);
	bank_code = false ;
	receive_code = false;
	var_id = (Pbase) vect_new(vecteur_var(bank_indices),
				  vecteur_val(bank_indices));
	stat2 = movement_computation(compute_module,false,bank_code,
				     receive_code,
				     local_variable,sc_image2,
				     const_base,bank_indices,
				     new_tile_indices,
				     var_id,loop_body_indices,n,dim_h);

	bst_sb = CONS(STATEMENT,stat2,bst_sb);
	bst_bsb =CONS(STATEMENT,stat1,bst_bsb);
    }
    sb = make_block_statement(bst_sb);
    bsb=make_block_statement(bst_bsb);
    *store_block = sb;
    *bank_store_block = bsb;

    debug(8,"make_store_blocks", "end\n");
}

void make_load_blocks(initial_module,compute_module,memory_module,var,shared_variable,local_variable,lrefs,
		      r_to_ud,sc_domain,
		      index_base,bank_indices,tile_indices,loop_body_indices, Proc_id,
		      pn,bn,ls,
		      load_block, bank_load_block,first_parallel_level,last_parallel_level
)
entity initial_module;
entity compute_module;
entity memory_module;
entity var;         /* entity  */
entity shared_variable;      /* emulated shared variable for example ES_A */
entity local_variable;       /* local variable for example L_A_1_1*/
list lrefs;                /* list of references associated to the local
			      variable */
hash_table r_to_ud;
Psysteme sc_domain;        /* domain of iteration */
Pbase index_base;          /* index basis */
Pbase bank_indices;        /* contains the index describing the bank:
			      bank_id, L (ligne of bank) and O (offset in 
			      the ligne) */
Pbase tile_indices;       /* contains the local indices  LI, LJ of  tile */
Pbase loop_body_indices;
entity Proc_id;              /* corresponds to a processeur identicator */
int pn,bn,ls;                 /* bank number and line size (depends on the 
			      machine) */
statement  * load_block;
statement * bank_load_block;
int first_parallel_level,last_parallel_level;
{
    list lur;
    list llur = NIL;
    list lrs = lrefs;
    Psysteme sc_image,sc_array_function;
    int n,dim_h;
    Pbase const_base;
    Pbase new_index_base = BASE_NULLE;
    Pbase new_tile_indices = BASE_NULLE;
    bool bank_code;			/* is true if it is the generation 
					   of code for bank false if it is 
					   for engine */
    bool receive_code;		/* is true if the generated code 
					   must be a RECEIVE, false if it 
					   must be a SEND*/
    reference r;
    statement stat1,stat2,lb,blb;
    cons * bst_lb= NIL;
    cons * bst_blb= NIL;
    Pbase var_id;
    Pvecteur ppid = vect_new((char *) Proc_id, VALUE_ONE);
    Psysteme sc_image2= SC_UNDEFINED;
    debug(8,"make_load_blocks",
	  "begin variable=%s, shared_variable=%s, local_variable=%s\n",
	  entity_local_name(var), entity_local_name(shared_variable),
	  entity_local_name(local_variable));

    ifdebug(8) {
	list crefs;
	(void) fprintf(stderr, "Associated reference list:");
	for(crefs=lrefs; !ENDP(crefs); POP(crefs)) {
	    reference r1 = REFERENCE(CAR(crefs));
	    print_words(stderr, words_reference(r1, NIL));
	    if(ENDP(CDR(crefs)))
		(void) putc('\n', stderr);
	    else
		(void) putc(',', stderr);
	}
    }


    /* Cases where the references are scalar variables */

    for (lrs =lrefs ; !ENDP(lrs) ; POP(lrs)) {
	r = REFERENCE(CAR(lrs));
	if (reference_indices(r) ==NIL) {

	    receive_code = true;
	    *load_block =make_movement_scalar_wp65(compute_module,receive_code,
						   r,Proc_id);
	    receive_code = false;
	    *bank_load_block=
		make_movement_scalar_wp65(memory_module,receive_code,
					  r,(entity) bank_indices->var);
	    return;
	}
    }
    /* In the other cases the references must be classified in lists of 
       uniform dependent references */

    for (lrs =lrefs ; !ENDP(lrs) ; POP(lrs)) {
	r = REFERENCE(CAR(lrs));
  	if ( (intptr_t) hash_get(r_to_ud, r) == (intptr_t)is_action_read) 
	    llur = classify_reference(llur,r);
    }

    /* For each list of uniform dependent references, "load_block" and 
       "bank_load_block" are computed */

    for (lur=llur; lur != NIL; lur = CDR(lur)) {
	sc_array_function=
	    build_sc_with_several_uniform_ref(initial_module,
					      LIST(CAR(lur)),
					      sc_domain,&new_index_base);
	new_index_base = vect_add(new_index_base,
				  vect_dup(index_base));

	sc_image =  sc_image_computation(initial_module,var,sc_domain,
					 sc_array_function,new_index_base,
					 &const_base,Proc_id,bank_indices,
					 tile_indices,&new_tile_indices,
					 pn,bn,ls,
					 &n,&dim_h);
	nullify_offsets();
	sc_image2= sc_dup(sc_image);
	bank_code = true ;
	receive_code = false;
	var_id = (Pbase) vect_new(vecteur_var(ppid),
				  vecteur_val(ppid));
	stat1 = movement_computation(memory_module,
				     true,
				     bank_code,
				     receive_code,
				     shared_variable,sc_image,
				     const_base,bank_indices,new_tile_indices,
				     var_id, loop_body_indices,n,dim_h);
	initialize_offsets(lur);
	bank_code = false ;
	receive_code = true;
	var_id = (Pbase) vect_new(vecteur_var(bank_indices),
				  vecteur_val(bank_indices));
	stat2 = movement_computation(compute_module,
				     true,
				     bank_code,
				     receive_code,
				     local_variable,sc_image2,
				     const_base,bank_indices,new_tile_indices,
				     var_id, loop_body_indices,n,dim_h);

	bst_lb = CONS(STATEMENT,stat2,bst_lb);	
	bst_blb = CONS(STATEMENT,stat1,bst_blb);	
    }

    lb = make_block_statement(bst_lb);
    blb = make_block_statement(bst_blb);
    *load_block =lb;
    *bank_load_block=blb;

    debug(8,"make_load_blocks", "end\n");
}

/* Psysteme tile_membership(P, origin, member):
 *
 * builds a linear constraint system to express the fact that iteration
 * "member" belongs to a P tile with origin "origin". "origin" and "member"
 * are expressed in the initial basis.
 */
Psysteme tile_membership(P, origin, member)
matrice P;
Pbase origin;
Pbase member;
{
    Psysteme m = sc_new();
    int d = base_dimension(origin);
    matrice IP = matrice_new(d,d);
    Pcontrainte c1;
    int i, j;
    Value k;

    debug(8,"tile_membership", "begin\n");

    pips_assert("tile_membership", d == base_dimension(member)); 
    pips_assert("tile_membership", value_one_p(DENOMINATOR(P)));

    ifdebug(8) {
	(void) fprintf(stderr,"Partitioning matrix P:\n");
	matrice_fprint(stderr, P, d, d);
    }

    matrice_general_inversion(P, IP, d);
    matrice_normalize(IP, d, d);
    k = DENOMINATOR(IP);

/*   pips_assert("tile_membership", k > 1); */

    ifdebug(8) {
	(void) fprintf(stderr,"Inverse of partitioning matrix, IP:\n");
	matrice_fprint(stderr, IP, d, d);
    }

    for ( i=1; i<=d; i++) {
	Pcontrainte c = contrainte_new();

	for ( j=1; j<=d; j++) {
	    vect_add_elem(&contrainte_vecteur(c), 
			  variable_of_rank(member,i),
			  value_uminus(ACCESS(IP, d, j, i)));
	    vect_add_elem(&contrainte_vecteur(c), 
			  variable_of_rank(origin,i),
			  ACCESS(IP, d, j, i));
	}
        sc_add_inegalite(m, c);
	c1 = contrainte_dup(c);
	contrainte_chg_sgn(c1);
	vect_add_elem(&contrainte_vecteur(c1), TCST, value_minus(VALUE_ONE,k));
	sc_add_inegalite(m, c1);
    }

    sc_creer_base(m);
    matrice_free(IP);

    ifdebug(8) {
	(void) fprintf(stderr,"Tile membership conditions:\n");
	sc_fprint(stderr, m, (string(*)(Variable))entity_local_name);
    }

    debug(8,"tile_membership", "end\n");

    return m;
}
