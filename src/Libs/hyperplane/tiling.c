/* package hyperplane
 *
 * $Id$
 * 
 * $Log: tiling.c,v $
 * Revision 1.1  1998/10/12 10:03:33  irigoin
 * Initial revision
 *
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

/* Query the user for a partitioning matrix P
 */

bool
interactive_partitioning_matrix(matrice P, int n)
{
    int i;
    int n_read;
    string resp = string_undefined;
    string cn = string_undefined;
    bool return_status = FALSE;
    int row;
    int col;

    /* Query the user for h's coordinates */
    pips_assert("interactive_partitioning_matrix", n>=1);
    debug(8, "interactive_partitioning_matrix", "Reading P\n");

    for(row=0; row<n; row++) {
	resp = user_request("Partitioning matrix?\n"
			    "(give all its integer coordinates on one line per row): ");
	if (resp[0] == '\0') {
	    user_log("Tiling loop transformation has been cancelled.\n");
	    return_status = FALSE;
	}
	else {    
	    cn = strtok(resp, " \t");

	    return_status = TRUE;
	    for( col = 0; col<n; col++) {
		if(cn==NULL) {
		    user_log("Too few coordinates. "
			     "Tiling loop transformation has been cancelled.\n");
		    return_status = FALSE;
		    break;
		}
		n_read = sscanf(cn," " VALUE_FMT, &ACCESS(P, n, row, col));
		if(n_read!=1) {
		    user_log("Too few coordinates. "
			     "Hyperplane loop transformation has been cancelled.\n");
		    return_status = FALSE;
		    break;
		}
		cn = strtok(NULL, " \t");
	    }
	}

	if(cn!=NULL) {
	    user_log("Too many coordinates. "
		     "Tiling loop transformation has been cancelled.\n");
	    return_status = FALSE;
	}
    }

    ifdebug(8) {
	if(return_status) {
	    pips_debug(8, "Partitioning matrix:\n");
	    matrice_fprint(stderr, P, n, n);
	    (void) fprintf(stderr,"\n");
	    pips_debug(8, "End\n");
	}
	else {
	    pips_debug(8, "Ends with failure\n");
	}
    }

    return return_status;
}


/* Generate tiled code for a loop nest, PPoPP'91
 */

statement 
tiling( list lls)
{
    Psysteme sci;			/* iteration domain */
    Psysteme scn;			/* sc nouveau */
    Psysteme sc_row_echelon;
    Psysteme sc_newbase;
    Pbase base_oldindex = NULL;
    Pbase base_newindex = NULL;
    matrice B;
    matrice P;
    matrice S;
    matrice AG;
    matrice A;
    matrice G;
    int n;				/* number of indices */
    int m ;				/* number of constraints */
    Value *h;
    int i;   
    statement s_lhyp;
    Pvecteur *pvg;
    Pbase pb;  
    expression lower, upper;
    Pvecteur pv1, pv2;
    loop l;
    int row;
    int col;

    debug_on("TILING_DEBUG_LEVEL");

    debug(8,"tiling","Begin:\n");

    /* make the  constraint system for the iteration space */
    sci = loop_iteration_domaine_to_sc(lls, &base_oldindex);
    
    /* create the constraint matrix B for the loop bounds */
    n = base_dimension(base_oldindex);
    m = sci->nb_ineq;
    B = matrice_new(m,n);
    sys_matrice_index(sci, base_oldindex, B, n, m);

    /* computation of the partitioning matrix P */
    P = matrice_new(n, n);

    if(!interactive_partitioning_matrix(P, n)) {
	pips_user_error("A proper partitioning matrix was not provided\n");
    }

    ifdebug(8) {
	debug(8,"tiling","Partitioning matrix P:");
	matrice_fprint(stderr, P, n, n);
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

    debug(8," tiling","End\n");

    debug_off();

    return(s_lhyp);
}

bool
loop_tiling(string module_name)
{
    bool return_status = FALSE;

    return_status = interactive_loop_transformation(module_name, tiling);
    
    return return_status;
}
