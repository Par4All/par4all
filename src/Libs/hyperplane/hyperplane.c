/* package hyperplane
 *
 * $Id$
 * 
 * $Log: hyperplane.c,v $
 * Revision 1.8  1998/10/12 10:42:43  irigoin
 * Typo in RCS variable
 *
 * Revision 1.7  1998/10/09 15:49:46  irigoin
 * Reformatting + conversion from int to Value
 *
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
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
#include "prettyprint.h"
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
    Pvecteur pv1, pv2;
    loop l;

    debug_on("HYPERPLANE_DEBUG_LEVEL");

    debug(8," hyperplane","Begin:\n");

    /* make the  system "sc" of constraints of iteration space */
    sci = loop_iteration_domaine_to_sc(lls, &base_oldindex);
    
    /* create the  matrix A of coefficients of  index in (Psysteme)sci */
    n = base_dimension(base_oldindex);
    m = sci->nb_ineq;
    A = matrice_new(m,n);
    sys_matrice_index(sci, base_oldindex, A, n, m);

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

    /* create the new system of constraintes (Psysteme scn) with  
       AG and sci */
    scn = sc_dup(sci);
    matrice_index_sys(scn, base_oldindex, AG, n,m );

    /* computation of the new iteration space in the new basis G */
    sc_row_echelon = new_loop_bound(scn, base_oldindex);

    /* change of basis for index */
    change_of_base_index(base_oldindex, &base_newindex);
    sc_newbase = sc_change_baseindex(sc_dup(sc_row_echelon), base_oldindex, base_newindex);
    
    /* generation of hyperplane  code */
    /*  generation of bounds */
    for (pb=base_newindex; pb!=NULL; pb=pb->succ) {
	make_bound_expression(pb->var, base_newindex, sc_newbase, &lower, &upper);
    }
  
    /* loop body generation */
    pvg = (Pvecteur *)malloc((unsigned)n*sizeof(Svecteur));
    scanning_base_to_vect(G,n,base_newindex,pvg);
    pv1 = sc_row_echelon->inegalites->succ->vecteur;
    pv2 = vect_change_base(pv1,base_oldindex,pvg);    

    l = instruction_loop(statement_instruction(STATEMENT(CAR(lls))));
    lower = range_upper(loop_range(l));
    upper= expression_to_expression_newbase(lower, pvg, base_oldindex);


    s_lhyp = code_generation(lls, pvg, base_oldindex, base_newindex, sc_newbase);

    debug(8," hyperplane","End\n");

    debug_off();

    return(s_lhyp);
}
