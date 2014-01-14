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
/*
 * PACKAGE MOVEMENTS
 *
 * Corinne Ancourt  - juin 1990
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "constants.h"
#include "text-util.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"
#include "sommet.h"
#include "matrix.h"
#include "sparse_sc.h"
#include "tiling.h"
#include "movements.h"

/* build the base of the image domain. It corresponds to the 
 * loop indices of the generated code. Then it is 
 * 
 *  Si COLUMN_MAJOR TRUE
 *        Bank_id, LJ,L,LI in case of engine code and
 *        Proc_id, LJ,L,O in case of bank code
 *  Si COLUMN_MAJOR FALSE
 *        Bank_id, LI,L,LJ in case of engine code and
 *        Proc_id, LI,L,O in case of bank code
 * 
 * In these examples L_I, resp. L_J, corresponds to the first, resp. second,
 *  array subscript.
 *
 */

#define sys_debug(level, msg, sc)					\
    ifdebug(level) {							\
	pips_debug(level, msg);						\
	sc_fprint(stderr, sc, (get_variable_name_t) entity_local_name);	\
    }

Pbase build_image_base(
    bool bank_code,
    Pbase proc_id,
    Pbase bank_indices,
    Pbase tile_indices)
{
    Pvecteur pb,ti;
    Pbase invt = base_reversal(tile_indices);
    Pbase dupt = base_dup(tile_indices);

    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(8,"build_image_base","begin\n");

    ti = (COLUMN_MAJOR) ? dupt:invt ;
    
    if (bank_code)
	pb = vect_new(vecteur_var(bank_indices->succ->succ),VALUE_ONE); 
    else 
	pb = vect_new(ti->var,VALUE_ONE);
    vect_add_elem(&pb,vecteur_var(bank_indices->succ),VALUE_ONE);
    
    if (!VECTEUR_NUL_P(ti->succ))
	vect_add_elem(&pb,vecteur_var(ti->succ),VALUE_ONE);
    
    if (bank_code) 
	vect_add_elem(&pb,vecteur_var(proc_id),VALUE_ONE);
    else 
	vect_add_elem(&pb,vecteur_var(bank_indices),VALUE_ONE);
    
    vect_rm(dupt);
    vect_rm(invt);
    debug(8,"build_image_base","end\n");
    debug_off();
    return ((Pbase) pb);
}


void 
print_fullname_base(Pbase sb)
{
    Pvecteur pv;
    for (pv = sb; pv !=NULL;
	 (void) fprintf(stderr,"%s_%s,",
			entity_module_name((entity)(pv->var)),
			entity_local_name((entity)(pv->var))),
	 pv=pv->succ);
    (void) fprintf(stderr,"\n");
}

/* Update  all the basis needed for data movement generation.

  -loop_body_offsets: indices such as O or LI or LJ used to describe
   the range of contiguous values accessed on line (of tile or bank).

  -loop_body_indices: list of the loop indices that are not 
   parameters of the tiling transformation and are situated in the 
   loop body of the tile 
*/

void 
update_basis(scbase,index_base,const_base,image_base,bank_indices,tile_indices,lindex,lvar_coeff_nunit,lvar_coeff_unit,loop_body_offsets,loop_body_indices, bank_code,ppid)
Pbase scbase;
Pbase *index_base;
Pbase *const_base;
Pbase *image_base;
Pbase bank_indices;
Pbase tile_indices;
Pbase *lindex;
Pbase *lvar_coeff_nunit;
Pbase *lvar_coeff_unit;
Pbase *loop_body_offsets;	
Pbase *loop_body_indices;	
bool bank_code;
Pbase ppid;
{

    Pvecteur pv,pv1;

    debug(8,"update_basis","begin\n");
    ifdebug(8) {
	(void) fprintf(stderr,"BASIS PRINTING: \n");  
	(void) fprintf(stderr,"     sc_base :");
	print_fullname_base(scbase);
	(void) fprintf(stderr,"     const_base :");
	print_fullname_base(*const_base);
	(void) fprintf(stderr,"     bank_indices ");
	print_fullname_base(bank_indices);
	(void) fprintf(stderr,"     index_base");
	print_fullname_base(*index_base);
	(void) fprintf(stderr,"     tile_indices");
	print_fullname_base(tile_indices);  
	(void) fprintf(stderr,"     loop_body_indices");
	print_fullname_base(*loop_body_indices); }

    for (pv = bank_indices; !VECTEUR_NUL_P(pv); pv = pv->succ)
	vect_chg_coeff(const_base,pv->var, VALUE_ZERO);

    vect_chg_coeff(const_base,ppid->var, VALUE_ZERO);

    if (bank_code) 
	vect_chg_coeff(const_base,vecteur_var(bank_indices), VALUE_ONE);
    else 
	vect_chg_coeff(const_base,vecteur_var(ppid), VALUE_ONE);

    for (pv = tile_indices; !VECTEUR_NUL_P(pv); pv = pv->succ)
	vect_chg_coeff(const_base,pv->var, VALUE_ZERO);

    for (pv = *loop_body_indices;  !VECTEUR_NUL_P(pv); pv = pv->succ)
	vect_erase_var(const_base,pv->var);

    *index_base = vect_dup(scbase);
    *lindex = vect_dup(scbase);
    *lvar_coeff_nunit = base_dup(scbase);

    for (pv = *const_base; !BASE_NULLE_P(pv);
	 vect_erase_var(lvar_coeff_nunit, vecteur_var(pv)),
	 vect_erase_var(index_base, vecteur_var(pv)), 
	 vect_erase_var(lindex, vecteur_var(pv)), 
	 pv= pv->succ);

    *image_base = build_image_base(bank_code,ppid,bank_indices,tile_indices);

    for (pv1 = *image_base; 
	 !BASE_NULLE_P(pv1);
	 vect_erase_var(lvar_coeff_nunit,vecteur_var(pv1)),
	 vect_erase_var(index_base,vecteur_var(pv1)),
	 vect_erase_var(lindex,vecteur_var(pv1)),
	 pv1=pv1->succ);
 
    *lvar_coeff_unit = base_dup(*lvar_coeff_nunit);
    *index_base = vect_add(vect_dup(*index_base),vect_dup(*image_base));
    *lindex = vect_add(*image_base,vect_dup(*lindex));
    *lindex = vect_add(vect_dup(*lindex),vect_dup(*const_base));
    
     if (bank_code)
	 *loop_body_offsets = vect_dup(bank_indices->succ);
    else 
	*loop_body_offsets = base_dup(tile_indices);
    ifdebug(8) {
	(void) fprintf(stderr,"New BASIS:");
	print_fullname_base(*image_base);    
	(void) fprintf(stderr,"    base lindex:");
	print_fullname_base(*lindex);    
	(void) fprintf(stderr,"   base index:");
	print_fullname_base(*index_base);    	
	(void) fprintf(stderr,"    lvar_coeff_nunit:");
	print_fullname_base(*lvar_coeff_nunit);    
	(void) fprintf(stderr,"    lvar_coeff_unit:");
	print_fullname_base(*lvar_coeff_unit);    
    }
    debug(8,"update_basis","end\n");
}


/* Sort the tile indices base, such that the indices correspond to the 
 * tile indices of the array elements accessed by the local entity.
 * Example: If A[I,K] is referenced. the tile indices base sould be 
 * L_I,L_K...
*/
void 
sort_tile_indices(tile_indices,new_tile_indices,Q,m) 
Pbase tile_indices;
Pbase *new_tile_indices;
matrice Q;
int m;
{
    register int i,j; 
    Pvecteur pv,pv2;
    Pvecteur pv3=VECTEUR_UNDEFINED;

    for (i=1;i<=m;i++) {
	for (j=1,pv=tile_indices; pv!=NULL;j++,pv=pv->succ) {
	    if (ACCESS(Q,m,i,j) &&  vect_coeff(pv->var,pv3)==0 ) {
		pv2 = vect_new(pv->var,ACCESS(Q,m,i,j)); 
		if (*new_tile_indices ==BASE_NULLE) 
		    *new_tile_indices =pv2 ;
		else pv3->succ= pv2;
		pv3 = pv2;
	    }
	}
    }
}

/*  Build the system of inequations of sc1 no-redundant with system sc2 */

Psysteme 
elim_redund_sc_with_sc(sc1,sc2,index_base,dim)
Psysteme sc1,sc2;
Pbase index_base;
int dim;
{
    Pcontrainte pc,pc2;
    Psysteme ps1 = sc_init_with_sc(sc1);
    Psysteme ps2 = sc_dup(sc2);
	
    for (pc =sc1->inegalites; pc != NULL;pc = pc->succ) {

	pc2 = contrainte_dup(pc);
	if (search_higher_rank(pc2->vecteur,index_base) > dim || 
	    !ineq_redund_with_sc_p(ps2,pc2)) {
	    pc2 = contrainte_dup(pc);
	    sc_add_ineg(ps1,pc2);
	    pc2 = contrainte_dup(pc);
	    sc_add_ineg(ps2,pc2);
	} 
    }
    ps2 = sc_dup(sc2);

    for (pc = ps1->inegalites; pc != NULL;pc = pc->succ) {
	pc2 = contrainte_dup(pc);
	if (search_higher_rank(pc2->vecteur,index_base) > dim || 
	    !ineq_redund_with_sc_p(ps2,pc2)) {
	    pc2 = contrainte_dup(pc);
	    sc_add_ineg(ps2,pc2);
	} 
	else   eq_set_vect_nul(pc);    
    }	

    sc_rm_empty_constraints(ps1, false);
    return (ps1);
}

Pbase 
variables_in_declaration_list(entity __attribute__ ((unused)) module,
			      code ce) {
    Pbase b= BASE_NULLE;
    MAPL(p,{
	entity e = ENTITY(CAR(p));
	if (BASE_UNDEFINED_P(b))
	    b = vect_new((Variable) e, VALUE_ONE);
	else vect_add_elem(&b,(Variable) e, VALUE_ONE);
    }, code_declarations(ce));

    return(b);
}


/* Calcul des nouvelles bornes des boucles et de la nouvelle fonction d'acces a
 * une reference d'un tableau permettant d'exprimer l'ensemble des elements
 * references dans une base. Cette base est pour le moment la base de Hermite 
 * associee a la fonction d'acces au tableau
 */


statement 
movement_computation(
    entity module,
    bool used_def,
    bool bank_code,  /* is true if it is the generation of code for bank
			   false if it is for engine */
    bool receive_code,      /* is true if the generated code must be a 
				  RECEIVE, false if it must be a SEND*/
    entity private_entity,       /* local entity */
    Psysteme sc_image,         /* domain of image  */
    Pbase const_base,
    Pbase bank_indices,        /* contains the index describing the bank:
			      bank_id, L (ligne of bank) and O (offset in 
			      the ligne) */
    Pbase tile_indices, /* contains the local indices  LI, LJ,.. of  tile */
    Pbase ppid,             
    Pbase loop_body_indices,   /* contains the loop indices situated in the 
				  tile*/
    int n,
    int dim_h)
{
    Psysteme	sc_proj,*list_of_systems,sc_proj2,sc_image2,sc_on_constants;
    statement  stat = statement_undefined;
    Pvecteur lvar_coeff_nunit = VECTEUR_NUL; 
    /* constains the list of variables for which integer projection 
       might be necessary */
    Pvecteur lvar_coeff_unit= VECTEUR_NUL;

    Pbase lindex = BASE_NULLE;		
    /* constains the variables remaining in the system after all the 
       projections i.e. constants, index loops. It is usefull to project (FM)
       on  these variables at the end for collecting more  informations on 
       variables and to eliminate redundant constraints */
  
    Pbase image_base=BASE_NULLE;	
    /* corresponds to the loop indices of the generated code. Then it is 
       Bank_id, LJ,L,LI in case of engine code 
       and Bank_id, LJ,L,O in case of bank code */
  

    Pbase loop_body_offsets;		
    /* contains the local indices O and L if it is the generation of bank 
       code and LI, LJ if it is for engine code */

    Pbase index_base=BASE_NULLE;
    Pbase const_base2;
    Pbase var_id;
    int dim_h2= dim_h;
    unsigned	space;
#define maxscinfosize 100
    /* int sc_info[sc_image->dimension+1][3]; // this is NOT ANSI C */
    int sc_info[maxscinfosize][4];
    int i;
    Pvecteur pv1= NULL;
    Pbase btmp = BASE_NULLE;
    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(3,"movement_computation","begin\n");

    assert(sc_image->dimension<maxscinfosize); /* added */

    /* Translate each entity in its appropriated entity full name 
       for generating module code */

    btmp =  variables_in_declaration_list(module,entity_code(module));
    sc_image = sc_variables_rename(sc_image, bank_indices, btmp,
				   (get_variable_name_t) entity_local_name);
    sc_image = sc_variables_rename(sc_image, const_base, btmp,
				   (get_variable_name_t) entity_local_name);
    sc_image = sc_variables_rename(sc_image,tile_indices, btmp,
				   (get_variable_name_t) entity_local_name);
    bank_indices= vect_rename(bank_indices, btmp,
			      (get_variable_name_t) entity_local_name);
    tile_indices= vect_rename(tile_indices, btmp,
			      (get_variable_name_t) entity_local_name);
    const_base= vect_rename(const_base, btmp,
			    (get_variable_name_t) entity_local_name);
    ppid = vect_rename(ppid, btmp,
		       (get_variable_name_t) entity_local_name);

    const_base2 =base_dup(const_base);
    sc_image2 = sc_dup(sc_image);
    sc_proj = sc_dup(sc_image2);
    n = sc_image2->dimension;

    /* allocation d'un tableau de systemes et d'une table 
       contenant des infos sur ces systemes*/

    space = (n+1) * sizeof(Ssysteme);
    list_of_systems = (Psysteme *) malloc((unsigned) space);
    /* update the different basis */

    update_basis(sc_image2->base,&index_base,&const_base2,&image_base,
		 bank_indices,tile_indices,&lindex,&lvar_coeff_nunit,
		 &lvar_coeff_unit,&loop_body_offsets,&loop_body_indices,
		 bank_code,ppid);


    dim_h2 = vect_size(image_base);

    sys_debug(2, "Domain before Projection :\n",sc_proj);

    /* Projection on each variable having unity coefficients in the system */

    for (pv1 = lvar_coeff_unit;!VECTEUR_UNDEFINED_P(pv1);pv1=pv1->succ) {
	if (var_with_unity_coeff_p(sc_proj,vecteur_var(pv1))) {
	    sc_proj = sc_projection(sc_proj,vecteur_var(pv1));
	    sc_proj= sc_normalize(sc_proj);
	    vect_chg_coeff(&lvar_coeff_nunit,vecteur_var(pv1),0);
	}
    } 

    sys_debug(2, " After FM projection :\n", sc_proj);

    sc_proj->inegalites = contrainte_sort(sc_proj->inegalites, 
					  sc_proj->base,  index_base,
					  true,false);
    ifdebug(2) {
	pips_debug(2," After FM projection  and sort:\n");
	sc_fprint(stderr, sc_proj, (get_variable_name_t) entity_local_name);
    }

    /* Projection on the others variables having to be eliminated from 
       the system. In case of Copy-in local memory code generation, the 
       FM projection algorithm is used. For Copy-back code generation
       interger projection algorithm is used.*/

    if (!used_def) 	
	sc_proj = sc_integer_projection_along_variables(
	    sc_image2,sc_proj, index_base, lvar_coeff_nunit, sc_info,dim_h2,n);
    else { 
	for (pv1 = lvar_coeff_nunit;!VECTEUR_UNDEFINED_P(pv1);pv1=pv1->succ) {
	    sc_proj = sc_projection(sc_proj,vecteur_var(pv1));
	    sc_proj= sc_normalize(sc_proj);
	}	    
	/* vect_chg_coeff(&lvar_coeff_nunit,vecteur_var(pv1),0);*/

    }
    
    ifdebug(2) {
	sys_debug(2," Before contrainte sort :\n",sc_proj);
	pips_debug(2," Base index :\n");
	vect_fprint(stderr, index_base, (get_variable_name_t) entity_local_name);
    }
    sc_proj->inegalites = contrainte_sort(sc_proj->inegalites, 
					  sc_proj->base,  index_base,
					  true,false); 

    sys_debug(5," After  contrainte sort:\n",sc_proj);

    build_sc_nredund_1pass(&sc_proj);

    sys_debug(5,"After Integer Projection :\n",sc_proj);

    /* Computation of sample constraints contraining only index variables 
     */
    sc_proj2 = sc_dup(sc_proj);

    sys_debug(4, "sc_proj2 [dup] = \n", sc_proj2);

    if (vect_size(sc_image2->base) <= 11)  /* why 11? */
    {
	for (pv1 = const_base2; pv1 != NULL; pv1 = pv1->succ)
	    vect_erase_var(&lindex, vecteur_var(pv1));
 
	sc_proj = sc_projection_on_list_of_variables
	    (sc_image2, image_base, lindex);
	sys_debug(9, "sc_proj = \n", sc_proj);

	sc_proj2 = sc_intersection(sc_proj2,sc_proj2,sc_proj);
	sys_debug(4, "sc_proj2 [inter] = \n", sc_proj2);

	sc_proj2 = sc_normalize(sc_proj2); 
	sys_debug(4, "sc_proj2 [norm] = \n", sc_proj2);

	sys_debug(9, "sc_image2 [minmax] = \n", sc_image2);
	sc_minmax_of_variables(sc_image2,sc_proj2,const_base2);
	sys_debug(4, "sc_proj2 [minmax] = \n", sc_proj2);
    }
    else				/*more restrictive system */
	sc_minmax_of_variables(sc_image2,sc_proj2,image_base);

    sys_debug(2, "Iterat. Domain Before redundancy elimin.:\n",sc_proj2);
    
    /* Elimination of redundant constraints for integer systems*/

    sc_proj2->inegalites = 
	contrainte_sort(sc_proj2->inegalites, 
			sc_proj2->base,  index_base, 
			false,false);

    sys_debug(2,"Iterat. Domain After 1rst sort:\n",sc_proj2);

    sc_integer_projection_information(sc_proj2,index_base, sc_info,dim_h2,n);
    sc_proj2=build_integer_sc_nredund(sc_proj2,index_base,sc_info,1,dim_h2,n);  
    sys_debug(2," After redundancy elimination :\n",sc_proj2);

    for (i=1;i<=n;i++) 
	list_of_systems[i] = sc_init_with_sc(sc_proj2);
	
    /* Constraints distribution. Lsc[i] will contain all constraints
       contraining the i-th index variable */

    constraint_distribution(sc_proj2,list_of_systems,index_base,sc_info);

    /* Computation of constraints contraining symbolic constants */
    sc_on_constants= sc_init_with_sc(sc_proj2);
    for (pv1 = tile_indices; pv1!=NULL; pv1=pv1->succ)
	vect_erase_var(&const_base2, vecteur_var(pv1));
    for (pv1 = bank_indices; pv1!=NULL; pv1=pv1->succ)
	vect_erase_var(&const_base2, vecteur_var(pv1));
    for (pv1 = ppid; pv1!=NULL; pv1=pv1->succ)
	vect_erase_var(&const_base2, vecteur_var(pv1));

    /* Elimination of constraints redundant in list_of_systems[i] with the 
       "symbolic constant system" */

    sc_minmax_of_variables(sc_image2,sc_on_constants,const_base2);
    for (i = 1; sc_on_constants != NULL && i <=vect_size(image_base);i++) {
	list_of_systems[i] = elim_redund_sc_with_sc(list_of_systems[i],
						    sc_on_constants,
						    index_base,
						    dim_h2);
	ifdebug(8) {
	    pips_debug(8,"Constraints on the %d-th var.:\n",i);
	    sc_fprint(stderr, list_of_systems[i], (get_variable_name_t) entity_local_name);
	}
    }

    var_id = (bank_code) ? vect_dup(ppid) : 
    vect_new(vecteur_var(bank_indices), VALUE_ONE);
    stat = bound_generation(module,bank_code,receive_code,
			    private_entity,
			    loop_body_offsets,var_id,
			    list_of_systems,index_base,n,sc_info);

    ifdebug(4) {
	/* if you get through this pp, it core dumps much later on:-) */
	pips_debug(3, "returning:\n");
	wp65_debug_print_text(entity_undefined, stat);
    }
    debug(3,"movement_computation","end\n");
    debug_off();

    return (stat);
}


/* This function computes the system of constraints characterizing the 
 * image by the array function of the iteration domain 
*/


Psysteme  sc_image_computation(module,entity_var,sc_domain,sc_array_function, index_base,const_base,proc_id,bank_indices,tile_indices,new_tile_indices,pn,bn,ls,n,dim_h)
entity module;             /* module  */
entity entity_var;         /* entity  */
Psysteme sc_domain;        /* domain of iteration */
Psysteme sc_array_function;/* system of constraints of the array function */
Pbase index_base;          /* index basis */
Pbase *const_base;
entity proc_id;
Pbase bank_indices;        /* contains the index describing the bank:
			      bank_id, L (ligne of bank) and O (offset in 
			      the ligne) */
Pbase tile_indices;
Pbase *new_tile_indices;
int pn,bn,ls;                 /* bank number and line size (depends on the 
			      machine) */
int *n,*dim_h;
{

    Psysteme sc_image = SC_UNDEFINED;
    Psysteme sc_machine = SC_UNDEFINED;
    Psysteme sc_domain2 = sc_dup(sc_domain);
    Pcontrainte pc = CONTRAINTE_UNDEFINED;
    Pbase new_index_base = BASE_NULLE;
    Pvecteur pv = VECTEUR_NUL;
    Pvecteur pvnew = VECTEUR_NUL;
    Pvecteur pv1= VECTEUR_NUL; 
    Pvecteur pvi=VECTEUR_NUL;
	   
    Pbase pbv, list_new_var=BASE_NULLE;
    matrice A,B,F0,P,HERM,HERF,R;
    matrice F = 0;
    matrice Q = NULL;
    int n1,mb,i;
    Value det_p, det_q;
    int n2 = 0;
    int ma =0;
    Variable new_var;

    debug_on("MOVEMENT_DEBUG_LEVEL");
    debug(3,"sc_image_computation","begin\n");

    ifdebug(3) {
	pips_debug(3,"ITERATION DOMAIN:\n");
	sc_fprint(stderr, sc_domain2, (get_variable_name_t) entity_local_name);
	pips_debug(3,"ARRAY FUNCTION :\n");
	sc_fprint(stderr, sc_array_function, (get_variable_name_t) entity_local_name);
    }
    if (!sc_faisabilite(sc_domain2))
	(void) fprintf(stderr,"systeme non faisable en  reels  \n");
    else {

	/* Computation of the system depending on the machine */	
	sc_machine = build_sc_machine(pn,bn,ls,sc_array_function,proc_id,
				      bank_indices,entity_var);
	sc_domain2 = sc_append(sc_domain2,sc_machine);

	/* conversion des egalites en deux inegalites */
	sc_transform_eg_in_ineg(sc_domain2);

	/*update the base of constants in the system */

	pv1 =vect_dup(sc_domain2->base);
	for (pv = index_base; 
	     !VECTEUR_NUL_P(pv); 
	     vect_chg_coeff(&pv1,vecteur_var(pv),0),pv=pv->succ);
	*const_base = pv1;
	ifdebug(8) {
	    pips_debug(8,"\n constant basis - sc_image_computation:");
	    base_fprint(stderr, *const_base,
			(get_variable_name_t) entity_local_name);
	}

	/* initialisation du nombre de constantes symboliques du systeme */

	for (pv1 = index_base, ma=0; 
	     !VECTEUR_NUL_P(pv1);
	     ma ++, pv1 = pv1->succ);
	for (pv1 = *const_base, mb=1; 
	     !VECTEUR_NUL_P(pv1);
	     mb ++, pv1 = pv1->succ);
	
	sc_image = sc_new();
	n1 = sc_domain2->nb_ineq;
	n2 = sc_array_function->nb_ineq;
 
	/* allocation et initialisation des matrices utiles */
	A = matrice_new(n1,ma);
	B = matrice_new(n1,mb);
	F = matrice_new(n2,ma);
	F0 = matrice_new(n2,mb);
	R = matrice_new(n1,ma);
	P = matrice_new(n2,n2);
	Q = matrice_new(ma,ma);
	HERM = matrice_new(n2,ma);
	HERF= matrice_new(n2,ma);

	matrice_nulle(R,n1,ma);
	matrice_nulle(HERF,n2,ma);
	matrice_identite(P,n2,0);

	*n= ma;

	/* conversion du premier systeme relatif au domaine d'iteration 
	   et du deuxieme systeme relatif a la fonction d'acces
	   aux elements du tableau  */
	loop_sc_to_matrices(sc_domain2,index_base,*const_base,A,B,n1,ma,mb);

	loop_sc_to_matrices(sc_array_function,index_base,
			    *const_base,F,F0,n2,ma,mb);

	/* mise sous forme normale de matrice_hermite */
	matrice_hermite(F,n2,ma,P,HERM,Q,&det_p,&det_q);
	ifdebug(8) {
	    sc_fprint(stderr, sc_array_function, (get_variable_name_t) entity_local_name);
	    (void) fprintf(stderr," matrice F\n");
	    matrice_fprint(stderr,F,n2,ma);
	  }

	ifdebug(8) { 
	    (void) fprintf(stderr," matrice P\n");
	    matrice_fprint(stderr,P,n2,n2);
	    (void) fprintf(stderr," matrice Q\n");
	    matrice_fprint(stderr,Q,ma,ma);
	}

	/* calcul de la dimension reelle de la fonction d'acces */
	*dim_h = dim_H(HERM,n2,ma);

	/** Computation of the new iteration domain **/
	matrice_multiply(A,Q,R,n1,ma,ma);
	ifdebug(8) { 
	    (void) fprintf(stderr," matrix of new iteration domain \n");
	    matrice_fprint(stderr,R,n1,ma);
	}
	/* conversion de la matrice en systeme */
	for (i = 1; i<= ma;i++) { 
	    new_var = sc_add_new_variable_name(module,sc_image); 
	    if (i==1) {
		new_index_base= vect_new(new_var,VALUE_ONE);
		pvi = new_index_base;
	    }
	    else { pvi->succ = vect_new(new_var,VALUE_ONE);
		   pvi=pvi->succ;
	       }
	}
	matrices_to_loop_sc(sc_image,new_index_base,
			    *const_base,R,B,n1,ma,mb);
	list_new_var = new_index_base;
	for (i = 1,pc=sc_array_function->inegalites,pbv = list_new_var; 
	     i<= ma && !CONTRAINTE_UNDEFINED_P(pc) && !VECTEUR_NUL_P(pbv);
	     i++,pc =pc->succ,pbv=pbv->succ) { 
	    if (vect_size(pc->vecteur) == 1 && pc->vecteur->var == TCST) {
		pvnew = vect_make(NULL,
				  pbv->var,VALUE_ONE,
				  TCST,value_uminus(pc->vecteur->val));
		sc_add_inegalite(sc_image,contrainte_make(pvnew));
		pvnew = vect_make(NULL,
				  pbv->var,VALUE_MONE,
				  TCST,pc->vecteur->val);
		sc_add_inegalite(sc_image,contrainte_make(pvnew));
	    }
	}
	ifdebug(3) {
	    pips_debug(3,"NEW ITERATION DOMAIN:\n");
	    sc_fprint(stderr, sc_image, (get_variable_name_t) entity_local_name);
	}

	/** Computation of the new matrix for array function **/
	matrice_multiply(P,HERM,HERF,n2,n2,ma);
	ifdebug(3) { 
	    pips_debug(3," New Matrix for Array Function \n");
	    matrice_fprint(stderr,HERF,n2,ma);
	    matrice_fprint(stderr,F0,n2,mb);
	}
	/* conversion from matrix to system */
	matrices_to_loop_sc(sc_array_function,
			    new_index_base,*const_base,
			    HERF,
			    F0,
			    n2,
			    ma,mb);
	ifdebug(3) { 
	    pips_debug(3,"New Array Function :\n");
	    sc_fprint(stderr, sc_array_function, (get_variable_name_t) entity_local_name);
	}
    }
   
    /*    sc_transform_ineg_in_eg(sc_image);*/
    *new_tile_indices = BASE_NULLE;
    sort_tile_indices(tile_indices,new_tile_indices,F,n2);


    ifdebug(3) {
	pips_debug(3,"New Iteration Domain :\n");
	sc_fprint(stderr, sc_image, (get_variable_name_t) entity_local_name);
    }
    debug(3,"sc_image_computation","end\n");
    debug_off();

    return(sc_image);
}

