/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

/************************************************************* pnome-reduc.c
 *
 * REDUCTIONS ON POLYNOMIALS
 *
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "polynome.h"

/* computes the possible roots of a polynomial
 * currently only works for degree 0,1,2)
 * returned value is a vector of polynomial
 * if p is independent of var, returns VECTEUR_UNDEFINED
 */
Pvecteur polynome_roots(Ppolynome p, Variable var) {
    switch(polynome_degree(p,var)) {
        /* should we verify the other components are equal to zero ? */
        case 0: 
            return VECTEUR_UNDEFINED;
        /* gather a x var + b information */
        case 1: {
                    Ppolynome a = POLYNOME_NUL, b = POLYNOME_NUL,pp;
                    for(pp=p ; pp != POLYNOME_NUL; pp = polynome_succ(pp)) {
                        Pmonome m = polynome_monome(pp);
                        Value val ;
                        if((val=vect_coeff(var, monome_term(m)))!=VALUE_ZERO) {
                            Pmonome dup = monome_dup(m);
                            if(vect_size(monome_term(dup))>1)
                                vect_del_var(monome_term(dup),var);
                            else
                                vect_chg_var(&monome_term(dup),var,TCST);
                            polynome_monome_add(&a,dup);
                            monome_rm(&dup);
                        }
                        else {
                            polynome_monome_add(&b,m);
                        }
                    }
                    /* the root is -b/a */
                    polynome_negate(&b);
                    b=polynome_div(b,a);
                    polynome_rm(&a);
                    return vect_new(b,1);
                } 
        /* gather a x^2 + b x + c informations */
        case 2:{
                   Ppolynome a = POLYNOME_NUL, b = POLYNOME_NUL, c = POLYNOME_NUL, pp;
                   for(pp=p ; pp != POLYNOME_NUL; pp = polynome_succ(pp)) {
                       Pmonome m = polynome_monome(pp);
                       Value val =vect_coeff(var, monome_term(m));
                       if(val==2) {
                           Pmonome dup = monome_dup(m);
                           vect_chg_var(&monome_term(dup),var,TCST);
                           polynome_monome_add(&a,dup);
                           monome_rm(&dup);
                       }
                       else if(val == 1 ) {
                           Pmonome dup = monome_dup(m);
                           vect_chg_var(&monome_term(dup),var,TCST);
                           polynome_monome_add(&b,dup);
                           monome_rm(&dup);
                       }
                       else {
                           polynome_monome_add(&c,m);
                       }
                   }
                   /* compute determinant */
                   Ppolynome delta=polynome_mult(a,c),tmp;
                   delta=polynome_scalar_multiply(delta,4);
                   polynome_negate(&delta);/* delta =-4 a c*/
                   tmp=polynome_mult(b,b);
                   polynome_add(&delta,tmp);
                   polynome_rm(&tmp);

                   /* take its square root if possible */
                   Ppolynome sqdelta = polynome_nth_root(delta,2);
                   if(POLYNOME_UNDEFINED_P(sqdelta))
                       polynome_error(__FUNCTION__,"cannot solve this degree 2 polynomial symbolically\n");

                   /* the roots are (-b +sqdelta) / 2a and (-b -sqdelta) / 2a */
                   Ppolynome r0,r1=polynome_dup(b);
                   r0=polynome_addition(b,sqdelta);
                   polynome_negate(&r0);
                   polynome_scalar_mult(&a,2);// warning: modifies a in place */
                   r0=polynome_div(tmp=r0,a);
                   polynome_rm(&tmp);

                   polynome_negate(&b); //warning modifies b in place
                   r1=polynome_addition(b,sqdelta);
                   r1=polynome_div(tmp=r1,a);
                   polynome_rm(&tmp);
                   Pvecteur roots=vect_new(r0,1);
                   vect_add_elem(&roots,r1,1);
                   return roots;
               }
        default:
                polynome_error(__FUNCTION__,"solving polynome not implemented yet in that case\n");
    }
    return VECTEUR_NUL;

           
}
