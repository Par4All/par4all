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
/* PACKAGE MOVEMENTS
 *
 * Corinne Ancourt  - septembre 1991
 */

#include <stdio.h>



#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"

#include "matrice.h"
#include "tiling.h"
#include "movements.h"

/* this function builds the following system of constraints depending
 *  on the machine. It describes the implementation of array elements in
 * the memory in function of bank, ligne size, ligne bank,...
 * 
 * bn is the number of banks, ls the ligne size, ms the first array dimension
 * bank is a variable giving the bank id., ligne a variable corresponding to
 * a  ligne  in the bank, ofs a variable corresponding to an offset 
 * in a ligne of the bank. 
 *
 * if COLUMN_MAJOR is true the system is the following one
 *
 *      (VAR1-1) + (VAR2-1) *ms == bn*ls* ligne +ls*bank+ofs,
 *        1 <= bank <= bn ,
 *        1 <= proc <= pn ,
 *        0 <= ofs <= ls-1
 *
 *else it is
 *
 *      (VAR1-1) * ms + (VAR2-1) == bn*ls*ligne +ls*bank+ofs,
 *        1 <= bank <= bn ,
 *        1 <= proc <= pn ,
 *        0 <= ofs <= ls-1
*/

Psysteme build_sc_machine(
    int pn,
    int bn,
    int ls,
    Psysteme sc_array_function,
    entity proc_id,
    Pbase bank_indices,
    entity entity_var)
{

    type t = entity_type(entity_var);
    Value ms=0;
    Variable vbank,vligne,vofs;
    Psysteme sc = sc_init_with_sc(sc_array_function);
    Pcontrainte pc;
    Pvecteur pv1,pv2;
    int nb_bytes=1;

    debug(8,"build_sc_machine","begin\n");
    if (type_variable_p(t)) {
	variable var = type_variable(t);
	cons * dims = variable_dimensions(var);
	dimension dim1 = DIMENSION(CAR(dims));
	expression lower= dimension_lower(dim1);
	normalized norm1 = NORMALIZE_EXPRESSION(lower);
	expression upper= dimension_upper(dim1);
	normalized norm2 = NORMALIZE_EXPRESSION(upper);
	Value min_ms =VALUE_ZERO, max_ms=VALUE_ZERO;
	if (normalized_linear_p(norm1) && normalized_linear_p(norm2)) {
	    min_ms = vect_coeff(TCST,(Pvecteur) normalized_linear(norm1));
	    max_ms = vect_coeff(TCST,(Pvecteur) normalized_linear(norm2));
	}
	ms = value_plus(value_minus(max_ms,min_ms), VALUE_ONE);

	/* Si l'on veut utiliser le nombre d'octets il faut remplacer l'equation 
	   par deux inequations du type 
	   
	   if COLUMN_MAJOR is true the system is the following one
	   
	   (VAR1-1) + (VAR2-1) *ms <= bn*ls* (ligne-1) +ls*(bank-1)+ofs,
	   bn*ls* (ligne-1) +ls*(bank-1)+ofs <=  (VAR1) + (VAR2-1) *ms

	   else it is
	   
	   (VAR1-1) * ms + (VAR2-1) <= bn*ls*(ligne-1) +ls*(bank-1)+ofs,
	   bn*ls*(ligne-1) +ls*(bank-1)+ofs <=  (VAR1-1) * ms + (VAR2)

	   */

	/*	  nb_bytes = SizeOfElements(bas);*/
	
    }

    ifdebug(8) {  
	fprint_string_Value(stderr," MS = ",ms);
	fprintf(stderr, " \n"); 
    }

    vbank = vecteur_var(bank_indices);
    vligne = vecteur_var(bank_indices->succ);
    vofs = vecteur_var(bank_indices->succ->succ);
    sc->base = vect_add_variable(sc->base,vbank);
    sc->base = vect_add_variable(sc->base,vligne);
    sc->base = vect_add_variable(sc->base,vofs);

    sc->dimension +=3;

    /* bank_indices is assumed to belong the three variables
       bank_id, L and O (see documentation for more details) */

    /* if COLUMN_MAJOR is true then build the constraint   
       (VAR1-1) + (VAR2-1) *ms == bn*ls*L +ls*bank_id+O,
       else build the constraint
       (VAR1-1) * ms + (VAR2-1) == bn*ls*L +ls*bank_id+O,
       VAR1 and VAR2 correspond to the image array function indices */
    pv1 = vect_new(vbank,int_to_value(-ls));
    vect_add_elem(&pv1,vligne,int_to_value((-bn*ls)));
    vect_add_elem(&pv1,vofs,VALUE_MONE);
    if (COLUMN_MAJOR)
	pc = sc_array_function->inegalites;
    else pc = sc_array_function->inegalites->succ;
    /* to deal with MONO dimensional array */
    if (pc==NULL) pc= contrainte_make(vect_new(TCST,VALUE_ONE));
    pv2 = vect_dup(pc->vecteur);
    vect_add_elem(&pv2,TCST,VALUE_MONE);
    pv2 = vect_multiply(pv2,int_to_value(nb_bytes));
    pv1 = vect_add(pv1,pv2);
    if (COLUMN_MAJOR)
	pc = pc->succ;
    else pc =  sc_array_function->inegalites;    
    /* to deal with MONO dimensional array */
    if (pc==NULL) pc=  contrainte_make(vect_new(TCST,VALUE_ONE));
    pv2 = vect_dup(pc->vecteur);
    vect_add_elem(&pv2,TCST,VALUE_MONE);
    pv2 = vect_multiply(pv2,value_mult(ms,int_to_value(nb_bytes)));
    pv1 = vect_add(pv1,pv2);
    pc = contrainte_make(pv1);
    sc_add_eg(sc,pc);

    /* build the constraints 0 <= bank_id <= bn-1 */

    pv2 = vect_new(vbank, VALUE_MONE);
    pc = contrainte_make(pv2);
    sc_add_ineg(sc,pc);
    pv2 = vect_new(vbank, VALUE_ONE);
    vect_add_elem(&pv2,TCST,int_to_value(- bn+1));
    pc = contrainte_make(pv2);
    sc_add_ineg(sc,pc);

    /* build the constraints 0 <= proc_id <= pn-1 */
    sc->base = vect_add_variable(sc->base,(char *) proc_id);
    sc->dimension++;
    pv2 = vect_new((char *) proc_id, VALUE_MONE);
    pc = contrainte_make(pv2);
    sc_add_ineg(sc,pc);
    pv2 = vect_new((char *) proc_id, VALUE_ONE);
    vect_add_elem(&pv2,TCST,int_to_value(- pn+1));
    pc = contrainte_make(pv2);
    sc_add_ineg(sc,pc);


    /* build the constraints 0 <= O <= ls -1 */

    pv2 = vect_new(vofs, VALUE_MONE);
    pc = contrainte_make(pv2);
    sc_add_ineg(sc,pc);
    pv2 = vect_new(vofs, VALUE_ONE);
    vect_add_elem(&pv2,TCST,int_to_value(- ls +1));
    pc = contrainte_make(pv2);
    sc_add_ineg(sc,pc);


    /* build the constraints 0 <= L   */

    pc = contrainte_make(vect_new(vligne,VALUE_MONE));
    sc_add_ineg(sc,pc);
    ifdebug(8)  {
	(void) fprintf(stderr,"Domain Machine :\n");
	sc_fprint(stderr, sc, (get_variable_name_t) entity_local_name);
    }
    debug(8,"build_sc_machine","end\n");

    return(sc);

}





