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
/* Name     : pip.c
 * Package  : pip
 * Author   : F. Dumontet, A. Platonoff and A. Leservot
 * Date     : 30 july 1993
 * Historic :
 * - 16 nov 93, few changes (e.g. "test1" to "pip_in", etc.), AP
 * - 08 dec 93, creation of a new package named pip.
 * - 10 dec 93, new version of pip_solve : direct call to pip. AL
 * - 31 jul 95, cleaning, pip_solve_min, pip_solve_max, old_pip_solve,
 * old2_pip_solve, AP
 *
 * Documents:
 * Comments : file containing the functions that resolve a Parametric Integer
 * Programming problem with the PIP solver.
 */

/* Ansi includes 	*/
#include <setjmp.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes 	*/
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "bootstrap.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "text.h"
#include "paf-util.h"
#include "pip.h"

/* Macros and functions Used by the old_pip_solve version 	*/
#define PIP_BIN "pip4"
#define PIP_OPTION "-s"
#define PIP_IN_FILE "pip_in"
#define PIP_OUT_FILE "pip_out"

/* Variables for the direct call to PIP version */
#define INLENGTH 1024
long    int cross_product, limit;/* External variables for direct call to PIP */
int     allocation, comptage;
char    inbuff[INLENGTH];
int     inptr = 256;
int     proviso = 0;
int     verbose = 0; /* Should not be used : put here for Pip copatibility */
FILE    *dump = NULL;
char    dump_name[] = "XXXXXX";

/* Global variables 	*/
quast 	quast_act;
Pbase 	base_var_ref, base_ref, old_base, old_base_var;
int   	ind_min_max;

/*===========================================================================*/
/* quast old_pip_solve(Psysteme ps_dep, ps_context, int nb_unknowns,
 *		   int min_or_max): Pip resolution.
 *
 * Parameters: ps_dep:     system of constraints on the unknowns
 *			   (and parameters)
 *	       ps_context: system of constraints on the parameters
 *	       nb_unknowns: number of unknowns in "ps_dep"
 *	       min_or_max:  tag for min or max resolution
 *
 * Result: a quast giving the value of the unknowns
 *
 * Note: The basis of "ps_dep" must contain all the unknowns and the
 * parameters, with first the unknowns in order to allow us to catch them using
 * "nb_unknowns".
 */
/* Obsolete, AP July 31st 95 :
   
   quast old_pip_solve(ps_dep, ps_context, nb_unknowns, min_or_max)
   Psysteme ps_dep, ps_context;
   int nb_unknowns;
   int min_or_max;
   {
     Pvecteur pvect;
     int aux, n, i, res, infinite_num;
     char *com = "essai";
   
     if(min_or_max == PIP_SOLVE_MIN) {
				       ind_min_max = 0;
				       infinite_num = -1;
				     }
     else {
	    ps_dep = converti_psysmin_psysmax(ps_dep, nb_unknowns);
	    ind_min_max = 1;
	    infinite_num = vect_size((ps_dep)->base);
	  }
   
     base_var_ref = base_dup((ps_dep)->base);
     old_base_var = base_dup(base_var_ref);
     for(i = 1, pvect = base_var_ref; i<nb_unknowns; i++) { pvect = pvect->succ; }
     base_ref = pvect->succ;
     pvect->succ = NULL;
     old_base = base_dup(base_ref);
   
     if(! SC_EMPTY_P(ps_context))
       ps_context->base = base_dup(base_ref);
   
     res = ecrit_probleme2(com, ps_dep, ps_context, nb_unknowns,
			   infinite_num, vect_size(base_ref));

     n = 1;
     while(n & 0xff) {
		       int m;
		       m = fork();
		       while(m == -1) {
					char answer;
   
					fprintf(stderr, "Fork failed for PIP\n\t Do you want to retry (y/n) ?\n");
					scanf("%c\n", &answer);
					if(answer == 'y')
					  m = fork();
					else
					  exit(1);
					fprintf(stderr, "\n");
				      }
		       if(m == 0) {
				    execlp(PIP_BIN, PIP_BIN, PIP_OPTION, PIP_IN_FILE, PIP_OUT_FILE, (char *) 0);
				  }

		       wait(&n);
		     }

     if((quayyin = fopen(PIP_OUT_FILE,"r")) == NULL) {
						       fprintf(stderr, "Cannot open file %s\n", PIP_OUT_FILE);
						       exit(1);
						     }
     aux = quayyparse();
   
     return(quast_act);
   }
*/

/*===========================================================================*/
/* quast old2_pip_solve(Psysteme ps_dep, ps_context, int nb_unknowns,
 *                 int min_or_max): Pip resolution.  AL 6/12/93
 *
 * Parameters: ps_dep:     system of constraints on the unknowns
 *                         (and parameters)
 *             ps_context: system of constraints on the parameters
 *             nb_unknowns: number of unknowns in "ps_dep"
 *             int_or_rat:	integer or rational resolution.
 *             min_or_max:  tag for min or max resolution
 *
 * Result: a quast giving the value of the unknowns
 *
 * Note: The basis of "ps_dep" must contain all the unknowns and the
 * parameters, with first the unknowns in order to allow us to catch them using
 * "nb_unknowns".
 *
 * Call to PIP is direct : no input or output file are produced.
 */
/* Obsolete, AP July 31st 95 :

quast old2_pip_solve(ps_dep, ps_context, nb_unknowns, min_or_max)
Psysteme ps_dep, ps_context;
int nb_unknowns, min_or_max;
{
        Pvecteur pvect;
        int     i, infinite_num;

        int     q, bigparm, ni, nvar, nq, non_vide, nparm, nc, p, xq;
        char*   g;
        Tableau *ineq, *context, *ctxt;


	debug_on("PIP_DEBUG_LEVEL");
        debug(5, "pip_solve", "begin\n");
        if (get_debug_level()>4) {
        	debug(5, "pip_solve", "Input Psysteme:\n");
                fprint_psysteme( stderr, ps_dep );
        	debug(5, "pip_solve", "Input Context:\n");
                fprint_psysteme( stderr, ps_context );
                fprintf(stderr, "Number of variables : %d\n", nb_unknowns);
        }

	vect_erase_var(&(ps_dep->base), (Variable) TCST);
	ps_dep->dimension = vect_size((ps_dep)->base);

        if(min_or_max == PIP_SOLVE_MIN) {
                ind_min_max = 0;
                infinite_num = -1;
        }
        else {
                ps_dep = converti_psysmin_psysmax(ps_dep, nb_unknowns);

		vect_erase_var(&(ps_dep->base), (Variable) TCST);
		ps_dep->dimension = vect_size((ps_dep)->base);

		ind_min_max = 1;
                infinite_num = vect_size((ps_dep)->base);
        }

        base_var_ref = base_dup((ps_dep)->base);
        old_base_var = base_dup(base_var_ref); 

        for(i = 1, pvect = base_var_ref; i<nb_unknowns; i++)
                { pvect = pvect->succ; }
        base_ref = pvect->succ;			
        pvect->succ = NULL;			
        old_base = base_dup(base_ref);

        if(! SC_EMPTY_P(ps_context) ) {
                ps_context->base = base_dup(base_ref);
                ps_context->dimension = vect_size( base_ref );
        }

        nvar 	= nb_unknowns;
        nparm 	= ps_dep->dimension - nb_unknowns;
        ni 	= (ps_dep->nb_eq * 2) + ps_dep->nb_ineq;;
        nc 	= ((ps_context == NULL)? 0 :
                         (ps_context->nb_eq * 2) + ps_context->nb_ineq );
        bigparm = infinite_num;
        nq 	= 1;
        debug(5, "pip_solve", "%d  %d  %d  %d  %d  %d\n",
                                nvar, nparm, ni, nc, bigparm, nq );

        limit 	= 0L;
        sol_init();
        tab_init();
        cross_product = 0;
        g 	= tab_hwm();
        ineq 	= sc_to_tableau(ps_dep, nb_unknowns);
        if (ps_context != NULL) {
                context = sc_to_tableau(ps_context, 0);
        }
        else context = NULL;
        xq = p = sol_hwm();

        if (nc) {
                ctxt = expanser(context, nparm, nc, nparm+1, nparm, 0, 0);
                traiter( ctxt, NULL, True, UN, nparm, 0, nc, 0, -1 );
                non_vide = is_not_Nil(p);
                sol_reset(p);
        }
        else non_vide = True;
        if ( non_vide ) {
                traiter( ineq, context, nq, UN, nvar, nparm, ni, nc, bigparm );
                q = sol_hwm();
                init_new_base();
                while((xq = new_sol_edit(xq)) != q);
                sol_reset(p);
        }
        else quast_act = quast_undefined;
        tab_reset(g);

        if (get_debug_level()>5) {
                imprime_quast( stderr, quast_act );
        }
	debug_off();
        return(quast_act);
}
*/


/*=======================================================================*/
/* quast pip_solve_min_with_big(Psysteme ps_dep, ps_context, int nb_unknowns,
 *                 char *big): Pip resolution.     
 *
 * Parameters: ps_dep:     system of constraints on the unknowns
 *                         (and parameters)
 *             ps_context: system of constraints on the parameters
 *             nb_unknowns: number of unknowns in "ps_dep"
 *             big: big parameter 's name.
 *
 * Result: a quast giving the value of the unknowns
 *
 * Note: The basis of "ps_dep" must contain all the unknowns and the
 * parameters, with first the unknowns in order to allow us to catch them
 * using "nb_unknowns".
 *
 * Call to PIP is direct : no input or output file are produced.
 *
 * Note: this function is the same as pip_solve_min() but it has a big
 * parameter given by the use.
 */

quast pip_solve_min_with_big(ps_dep, ps_context, pv_unknowns, big)

 Psysteme   ps_dep, ps_context;
 Pvecteur   pv_unknowns;
 char       *big; 
{
 /* FD variables */
 int        infinite_num;

 /* Pip variables */
 int        q, bigparm, ni, nvar, nq, non_vide, nparm, nc, p, xq;
 char*      g;
 Tableau    *ineq, *context, *ctxt;

 /* AC variables for the big parameter */
 bool    not_found = true;
 list       lbase;
 entity     ent;
 int        nb_unknowns, len_big;

 debug_on("PIP_DEBUG_LEVEL");
 debug(5, "pip_solve_min_with_big", "begin\n");
 /* trace file */
 nb_unknowns = vect_size( pv_unknowns ); 
 if (get_debug_level()>4) 
    {
     debug(5, "pip_solve_min_with_big", "Input Psysteme:\n");
     fprint_psysteme( stderr, ps_dep );
     debug(5, "pip_solve_min_with_big", "Input Context:\n");
     fprint_psysteme( stderr, ps_context );
     fprintf(stderr, "Number of variables : %d\n", nb_unknowns);
    }

 /* We set the env for the kind of resolution desired (i.e. Min) */
 /* and we get the order of the bi parameter called "big" in the */
 /* base of ps_dep                                               */

 /* Normalize base of system ps_dep */
 pv_unknowns = base_normalize( pv_unknowns );
 vect_erase_var( &ps_dep->base, TCST );
 ps_dep->base = vect_add_first(pv_unknowns, ps_dep->base);
 ps_dep->dimension = vect_size( ps_dep->base );

 ind_min_max = 0;
 infinite_num = 1;
 len_big = strlen(big);
 lbase = base_to_list(ps_dep->base);
 while ((not_found) && (lbase != NIL))
    {
     ent = ENTITY(CAR(lbase));
     if (!strncmp(entity_local_name(ent), big, len_big))
         not_found = false; 
     else
	{
	 infinite_num++;
	 lbase = CDR(lbase);
        }
    }
 if (not_found) infinite_num = -1;

 /* Computation of the basis (unknowns and parameters)
  * base of ps_dep is the reference for all unknowns and parameters.
  */
 if (ps_context != NULL) {
     vect_erase_var( &ps_context->base, TCST);
     ps_context->dimension = vect_size( ps_context->base );
     base_var_ref = base_union( ps_dep->base, ps_context->base );
 }
 else base_var_ref = ps_dep->base;
 base_var_ref = vect_add_first(pv_unknowns, base_var_ref);
 base_ref = vect_substract(base_var_ref, pv_unknowns);

 old_base_var = base_dup(base_var_ref);  /* Total base of ps_dep */
 old_base = base_dup(base_ref);

 if (!SC_EMPTY_P(ps_context)) 
    {
     ps_context->base = base_dup(base_ref);
     ps_context->dimension = vect_size( base_ref );
    }

 /* Set PIP variables. Comes from ecrit_probleme2 */
 nvar  = nb_unknowns;
 nparm = ps_dep->dimension - nb_unknowns;
 ni    = (ps_dep->nb_eq * 2) + ps_dep->nb_ineq;;
 nc    = ((ps_context == NULL)? 0 :
                       (ps_context->nb_eq * 2) + ps_context->nb_ineq );
 bigparm = infinite_num;
 nq      = 1;
 debug(5, "pip_solve_min_with_big", "%d  %d  %d  %d  %d  %d\n",\
                        nvar, nparm, ni, nc, bigparm, nq );

 /* Prepare to call PIP */
 limit   = 0L;
 sol_init();
 tab_init();
 cross_product = 0;
 g    = tab_hwm();
 ineq = sc_to_tableau(ps_dep, nb_unknowns);
 if (ps_context != NULL) 
       context = sc_to_tableau(ps_context, 0);
 else  context = NULL;
 xq = p = sol_hwm();

 /* Verification de la non vacuite du contexte */
 if (nc)
    {
     ctxt = expanser(context, nparm, nc, nparm+1, nparm, 0, 0);
     traiter( ctxt, NULL, PIP_SOLVE_RATIONAL, UN, nparm, 0, nc, 0, -1 );
     non_vide = is_not_Nil(p);
     sol_reset(p);
    }
 else non_vide = True;

 if (non_vide) 
    {
     traiter(ineq, context, PIP_SOLVE_RATIONAL, UN, nvar, nparm, ni, nc, bigparm);
     q = sol_hwm();
     init_new_base();
     /* We read solution and put it in global quast_act */
     while((xq = rational_sol_edit(xq)) != q);
     sol_reset(p);
    }
 else quast_act = quast_undefined;
 tab_reset(g);

 if (get_debug_level()>5) imprime_quast(stderr, quast_act);
                
 debug_off();

 return(quast_act);
}

/*=======================================================================*/
/* Pvecteur vect_add_first( pv1, pv2 )				AL 16 02 94
 * 
 * Suppress all variables of pv1 in pv2 and add pv1 before pv2.
 * Keep pv1 order, not pv2's order!
 */
Pvecteur vect_add_first( pv1, pv2 )
Pvecteur pv1, pv2;
{
	Pvecteur pv, pv11, pv22;

	if (pv1 == NULL) return pv2;
	pv11 = vect_reversal( vect_dup( pv1 ) );
	pv22 = vect_dup( pv2 );

        for( pv = pv11 ; pv != NULL; pv = pv->succ) {
		vect_erase_var( &pv22, pv->var );
                if (pv->succ != NULL) continue;
        	pv->succ = pv22;
		break;
        }
	return pv11;
}

/*===========================================================================*/
/* quast pip_solve(Psysteme ps_dep, ps_context, Pvecteur pv_unknowns,
 *                 int int_or_rat, int min_or_max): Pip resolution.  AL 6/12/93
 *
 * Parameters: ps_dep:     system of constraints on the unknowns
 *                         (and parameters)
 *             ps_context: system of constraints on the parameters
 *             pv_unknowns: ordered vector of unknowns in "ps_dep"
 *             int_or_rat:	integer (1)  or rational (0)  resolution.
 *             min_or_max:  tag for min or max resolution
 *
 * Result: a quast giving the value of the unknowns
 *
 * Call to PIP is direct : no input or output file are produced.
 */
quast pip_solve(ps_dep, ps_context, pv_unknowns, int_or_rat, min_or_max)
Psysteme ps_dep, ps_context;
Pvecteur pv_unknowns;
int 	 int_or_rat, min_or_max;
{
	/* FD variables */
	int     infinite_num, nb_unknowns;

	/* Pip variables */
	int     q, bigparm, ni, nvar, nq, non_vide, nparm, nc, p, xq;
	char*   g;
	Tableau *ineq, *context, *ctxt;


	/* Initialization */
	debug_on("PIP_DEBUG_LEVEL");
	debug(5, "pip_solve", "begin\n");
	if (ps_dep == NULL) {
		user_warning("new_pip_solve", "\nInput Psysteme is empty !\n");
		return quast_undefined;
	}
	nb_unknowns = vect_size( pv_unknowns ); 

	/* trace file */
	if (get_debug_level()>4) {
	        debug(5, "pip_solve", "Input Psysteme:\n");
	        fprint_psysteme( stderr, ps_dep );
	        debug(5, "pip_solve", "Input Context:\n");
	        fprint_psysteme( stderr, ps_context );
	        fprintf(stderr, "Number of variables : %d\n", nb_unknowns);
	}


	/* Normalize base of system ps_dep */
	pv_unknowns = base_normalize( pv_unknowns );
	vect_erase_var( &ps_dep->base, TCST );
	ps_dep->base = vect_add_first(pv_unknowns, ps_dep->base);
	ps_dep->dimension = vect_size( ps_dep->base );


	/* We set the env for the kind of resolution desired (Min or Max) */
	if(min_or_max == PIP_SOLVE_MIN) {
	        ind_min_max = 0;
	        infinite_num = -1;
	}
	else {
	        ps_dep = converti_psysmin_psysmax(ps_dep, nb_unknowns);
	        ind_min_max = 1;
	        infinite_num = vect_size((ps_dep)->base);
	}

	/* Computation of the basis (unknowns and parameters)
	 * base of ps_dep is the reference for all unknowns and parameters.
	 */
	if (ps_context != NULL) { 	
		vect_erase_var( &ps_context->base, TCST);
		ps_context->dimension = vect_size( ps_context->base );
		base_var_ref = base_union( ps_dep->base, ps_context->base );
	}
	else base_var_ref = ps_dep->base;
	base_var_ref = vect_add_first(pv_unknowns, base_var_ref);
	base_ref = vect_substract(base_var_ref, pv_unknowns);

	old_base_var = base_dup(base_var_ref);  /* Total base of ps_dep */
	old_base = base_dup(base_ref);

	if(! SC_EMPTY_P(ps_context) ) {
	        ps_context->base = base_dup(base_ref);
	        ps_context->dimension = vect_size( base_ref );
	}

	/* Set PIP variables. Comes from ecrit_probleme2 */
	nvar    = nb_unknowns;
	nparm   = ps_dep->dimension - nb_unknowns;
	ni      = (ps_dep->nb_eq * 2) + ps_dep->nb_ineq;;
	nc      = ((ps_context == NULL)? 0 :
	                 (ps_context->nb_eq * 2) + ps_context->nb_ineq );
	bigparm = infinite_num;
	nq      = int_or_rat;
	debug(5, "pip_solve", "%d  %d  %d  %d  %d  %d\n",
	                        nvar, nparm, ni, nc, bigparm, nq );

	/* Prepare to call PIP */
	limit   = 0L;
	sol_init();
	tab_init();
	cross_product = 0;
	g       = tab_hwm();
	ineq    = sc_to_tableau(ps_dep, nb_unknowns);
	if (ps_context != NULL) {
	        context = sc_to_tableau(ps_context, 0);
	}
	else context = NULL;
	xq = p = sol_hwm();

	/* Verification de la non vacuite du contexte */
	if (nc) {
		ctxt = expanser(context, nparm, nc, nparm+1, nparm, 0, 0);
		traiter( ctxt, NULL, nq, UN, nparm, 0, nc, 0, -1 );
		non_vide = is_not_Nil(p);
		sol_reset(p);
	}
	else non_vide = True;
	if ( non_vide ) {
		traiter( ineq, context, nq, UN, nvar, nparm, ni, nc, bigparm );
		q = sol_hwm();
		init_new_base();
		/* We read solution and put it in global quast_act */
		if( int_or_rat == PIP_SOLVE_INTEGER ) 
			while((xq = integer_sol_edit(xq)) != q);
		else while((xq = rational_sol_edit(xq)) != q);
		sol_reset(p);
	}
	else quast_act = quast_undefined;
	tab_reset(g);

	if (get_debug_level()>5) {
		imprime_quast( stderr, quast_act );
	}
	debug_off();
	return(quast_act);
}
	

/*===================================================================*/
quast pip_integer_min( ps_dep, ps_context, pv_unknowns )
Psysteme 	ps_dep, ps_context;
Pvecteur	pv_unknowns;
{
	return( pip_solve( ps_dep, ps_context, pv_unknowns, 
			PIP_SOLVE_INTEGER, PIP_SOLVE_MIN ));
}

/*===================================================================*/
quast pip_integer_max( ps_dep, ps_context, pv_unknowns )
Psysteme 	ps_dep, ps_context;
Pvecteur	pv_unknowns;
{
	return( pip_solve( ps_dep, ps_context, pv_unknowns, 
			PIP_SOLVE_INTEGER, PIP_SOLVE_MAX ));
}

/*===================================================================*/
quast pip_rational_min( ps_dep, ps_context, pv_unknowns )
Psysteme 	ps_dep, ps_context;
Pvecteur	pv_unknowns;
{
	return( pip_solve( ps_dep, ps_context, pv_unknowns, 
			PIP_SOLVE_RATIONAL, PIP_SOLVE_MIN ));
}

/*===================================================================*/
quast pip_rational_max( ps_dep, ps_context, pv_unknowns )
Psysteme 	ps_dep, ps_context;
Pvecteur	pv_unknowns;
{
	return( pip_solve( ps_dep, ps_context, pv_unknowns, 
			PIP_SOLVE_RATIONAL, PIP_SOLVE_MAX ));
}

/*===================================================================*/
/* void pip_solve_min(Psysteme ps_dep, ps_context, int nb_unknowns):
 * Pip resolution for the minimum.
 */
/* Obsolete, AP July 31st 95 :
   
   quast pip_solve_min(ps_dep, ps_context, nb_unknowns)
   Psysteme ps_dep, ps_context;
   int nb_unknowns;
   {
     return(old2_pip_solve(ps_dep, ps_context, nb_unknowns, PIP_SOLVE_MIN));
   }
   */

/*===================================================================*/
/* void pip_solve_max(Psysteme ps_dep, ps_context, int nb_unknowns):
 * Pip resolution for the maximum.
 */
/* Obsolete, AP July 31st 95 :

   quast pip_solve_max(ps_dep, ps_context, nb_unknowns)
   Psysteme ps_dep, ps_context;
   int nb_unknowns;
   {
   return(old2_pip_solve(ps_dep, ps_context, nb_unknowns, PIP_SOLVE_MAX));
   }
*/

/*===================================================================*/

/* Useful for the sorting of the variables in the system's Pvecteur.  
*/
static Pbase base_for_sort = (Pbase)NULL;

/* function for qsort. 
 * rank_of_variable returns 0 for TCST, hence the strange return
 */
static int compare_variables_in_base(pv1, pv2)
Pvecteur *pv1, *pv2;
{
    int 
	rank_1 = rank_of_variable(base_for_sort, var_of(*pv1)),
	rank_2 = rank_of_variable(base_for_sort, var_of(*pv2));

    message_assert("variable not in global vector", rank_1>=0 && rank_2>=0);

    /* TCST must be last, but given as 0
     */ 
    return((rank_1==0 || rank_2==0) ? rank_1-rank_2 : rank_2-rank_1);
}

/*===================================================================*/
/* void sort_psysteme(Psysteme ps, Pvecteur pv): sorts the system "ps"
 * according to the vector "pv". In order to avoid any side effects,
 * this vector is duplicated into a base "new_base". When the sorting
 * is done, this new base becomes the basis of this system.
 *
 * "pv" MUST contain all the variables that appear in "ps".
 * "pv" DOES NOT HAVE TO contain TCST.
 *
 * modified by FC, 29/12/94.
 * I removed the vect_tri call and the associated memory leaks.
 * sc_vect_sort called instead.
 * maybe the sc base is lost? It depends whether it is pv or not,
 * and what is done with pv by the caller. I would suggest a base_rm.
 */
void sort_psysteme(ps, pv)
Psysteme ps;
Pvecteur pv;
{
    if(SC_EMPTY_P(ps))	return;

    base_for_sort = base_dup(pv);
    sc_vect_sort(ps, compare_variables_in_base);

    ps->dimension = base_dimension(base_for_sort);
    ps->base = base_for_sort;
    base_for_sort = (Pbase)NULL;
}

/*************************************************************************/
