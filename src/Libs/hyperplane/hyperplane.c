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
/* package hyperplane
 */

#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "misc.h"
#include "text.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrice.h"
#include "matrix.h"

#include "sparse_sc.h"
#include "ri-util.h"
#include "effects-util.h"
#include "conversion.h"

#include "hyperplane.h"


/* void hyperplane(cons *lls)
 *  Implementation of the hyperplane method
 * "lls" is the list of nested loops
 */

statement hyperplane(lls)
cons * lls;
{
    Psysteme sci;			/* sc initial */
    Psysteme scn;			/* sc nouveau */
    Psysteme sc_row_echelon;
    Psysteme sc_newbase;
    Pbase base_oldindex = NULL;
    Pbase base_newindex = NULL;
    matrice A;
    matrice G;
    matrice AG;
    int n;				/* number of index */
    int m ;				/* number of constraints */
    Value *h;
    int i;
    statement s_lhyp;
    Pvecteur *pvg;
    Pbase pb;
    expression lower, upper;
    Pvecteur pv1;
    loop l;

    debug_on("HYPERPLANE_DEBUG_LEVEL");

    debug(8," hyperplane","Begin:\n");

    /* make the  system "sc" of constraints of iteration space */
    sci = loop_iteration_domaine_to_sc(lls, &base_oldindex);

    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"base_oldindex 0:\n");
      base_fprint(stderr,base_oldindex,default_variable_to_string);
    }

    /* create the  matrix A of coefficients of  index in (Psysteme)sci */
    n = base_dimension(base_oldindex);
    m = sci->nb_ineq;
    A = matrice_new(m,n);
    sys_matrice_index(sci, base_oldindex, A, n, m);

    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"base_oldindex 1:\n");
      base_fprint(stderr,base_oldindex,default_variable_to_string);
    }
    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"Matrice A:\n");
      matrice_fprint(stderr,A,m,n);
    }
    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"sci:\n");
      sc_default_dump(sci);
    }
    /* computation of the hyperplane direction */
    /*  use the  hyperplane direction  */
    h = (Value*)(malloc(n*sizeof(Value)));

    if(!interactive_hyperplane_direction(h, n)) {
	pips_user_error("A proper hyperplane direction was not provided\n");
    }

    ifdebug(8) {
	debug(8," hyperplane","Vector h :");
	for (i = 0; i<n; i++) {
	    (void) fprintf(stderr," " VALUE_FMT, *(h+i));
	}
	(void) fprintf(stderr,"\n");
    }

    G = matrice_new(n,n);
    /* computation of the  scanning base G */
    scanning_base_hyperplane(h, n, G);
    ifdebug(8) {
	(void) fprintf(stderr,"The scanning base G is:");
	matrice_fprint(stderr, G, n, n);
    }

    /* the new matrice of constraints AG = A * G */
    AG = matrice_new(m,n);
    matrice_multiply(A, G, AG, m, n, n);

    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"Matrice AG:\n");
      matrice_fprint(stderr,AG,m,n);
    }
    /* create the new system of constraintes (Psysteme scn) with
       AG and sci */
    scn = sc_dup(sci);

    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"base_oldindex 2:\n");
      base_fprint(stderr,base_oldindex,default_variable_to_string);
    }
    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"scn before matrice_index_sys :\n");
      sc_default_dump(scn);
    }
    matrice_index_sys(scn, base_oldindex, AG, n,m );

    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"scn after matrice_index_sys :\n");
      sc_default_dump(scn);
    }

    /* computation of the new iteration space in the new basis G */
    sc_row_echelon = new_loop_bound(scn, base_oldindex);

    ifdebug(8) {
      /*DNDEBUG*/
      /*      (void) fprintf(stderr,"scn after new_loop_bound:\n");
	      sc_default_dump(scn);
	      has been destroyed
      */
      (void) fprintf(stderr,"sc_row_echelon:\n");
      sc_default_dump(sc_row_echelon);
    }

    /* change of basis for index */
    change_of_base_index(base_oldindex, &base_newindex);

    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"base_oldindex:\n");
      base_fprint(stderr,base_oldindex,default_variable_to_string);
      (void) fprintf(stderr,"base_newindex:\n");
      base_fprint(stderr,base_newindex,default_variable_to_string);
    }

    sc_newbase = sc_change_baseindex(sc_dup(sc_row_echelon), base_oldindex, base_newindex);

    ifdebug(8) {
      /*DNDEBUG*/
      (void) fprintf(stderr,"sc_newbase:\n");
      sc_default_dump(sc_newbase);
    }

    /* generation of hyperplane  code */
    /*  generation of bounds */
    for (pb=base_newindex; pb!=NULL; pb=pb->succ) {
	make_bound_expression(pb->var, base_newindex, sc_newbase, &lower, &upper);
    }

    /* loop body generation */
    pvg = (Pvecteur *)malloc((unsigned)n*sizeof(Svecteur));
    scanning_base_to_vect(G,n,base_newindex,pvg);
    pv1 = sc_row_echelon->inegalites->succ->vecteur;
    (void)vect_change_base(pv1,base_oldindex,pvg);

    l = instruction_loop(statement_instruction(STATEMENT(CAR(lls))));
    lower = range_upper(loop_range(l));
    upper= expression_to_expression_newbase(lower, pvg, base_oldindex);


    s_lhyp = code_generation(lls, pvg, base_oldindex, base_newindex, sc_newbase, true);

    debug(8," hyperplane","End\n");

    debug_off();

    return(s_lhyp);
}
