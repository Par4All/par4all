/* package sc : sc_feasibility.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * 
 * This file provides functions to test the feasibility of a system 
 * of constraints. 
 *
 * Arguments of these functions :
 * 
 * - s or sc is the system of constraints.
 * - ofl_ctrl is the way overflow errors are handled
 *     ofl_error == NO_OFL_CTRL
 *               -> overflow errors are not handled
 *     ofl_error == OFL_CTRL
 *               -> overflow errors are handled in the called function
 *     ofl_error == FWD_OFL_CTRL
 *               -> overflow errors must be handled by the calling function
 * - ofl_res is the result of the feasibility test when ofl_error == OFL_CTRL
 *   and there is an overflow error.
 * - integer_p (low_level function only) is a boolean :
 *     integer_p == TRUE to test if there exists at least one integer point 
 *               in the convex polyhedron defined by the system of constraints.
 *     integer_p == FALSE to test if there exists at least one rational point 
 *               in the convex polyhedron defined by the system of constraints.
 *     (This has an impact only upon Fourier-Motzkin feasibility test).
 *
 * Last modified by Beatrice Creusillet, 13/12/94.
 */

#include <string.h>
#include <stdio.h>
#include <setjmp.h>
extern int fprintf();

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"



extern jmp_buf overflow_error;

/* 
 * INTERFACES
 */

boolean sc_rational_feasibility_ofl_ctrl(sc, ofl_ctrl, ofl_res)
Psysteme sc;
int ofl_ctrl;
boolean ofl_res;
{
    return sc_feasibility_ofl_ctrl(sc, FALSE, ofl_ctrl, ofl_res);
}

boolean sc_integer_feasibility_ofl_ctrl(sc,ofl_ctrl, ofl_res)
Psysteme sc;
int ofl_ctrl;
boolean ofl_res;
{
    return sc_feasibility_ofl_ctrl(sc, TRUE, ofl_ctrl, ofl_res);
}




/*
 * LOW LEVEL FUNCTIONS 
 */

boolean sc_feasibility_ofl_ctrl(sc, integer_p, ofl_ctrl, ofl_res)
Psysteme sc;
boolean integer_p;
int ofl_ctrl;
boolean ofl_res;
{
    boolean ok = FALSE;

    if (sc_rn_p(sc))
	ok = TRUE;
    else {
	switch (ofl_ctrl) {
	case OFL_CTRL :
	    ofl_ctrl = FWD_OFL_CTRL;
	    if (setjmp(overflow_error)) {
		ok = ofl_res;
		break;
	    }		
	default:
	    if (sc_nbre_egalites(sc)+sc_nbre_inegalites(sc) >= 
		NB_CONSTRAINTS_MAX_FOR_FM)
		ok = sc_simplexe_feasibility_ofl_ctrl(sc, ofl_ctrl);
	    else 
		ok = sc_fourier_motzkin_feasibility_ofl_ctrl(sc, integer_p, 
							     ofl_ctrl);
	}
    }
    return ok;
}



/* boolean sc_fourier_motzkin_faisabilite_ofl(Psysteme s):
 * test de faisabilite d'un systeme de contraintes lineaires, par projections
 * successives du systeme selon les differentes variables du systeme
 *
 *  resultat retourne par la fonction :
 *
 *  boolean	: TRUE si le systeme est faisable
 *		  FALSE sinon
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme s    : systeme lineaire 
 *
 * Le controle de l'overflow est effectue et traite par le retour 
 * du contexte correspondant au dernier setjmp(overflow_error) effectue.
 */
boolean sc_fourier_motzkin_feasibility_ofl_ctrl(s, integer_p, ofl_ctrl)
Psysteme s;
boolean integer_p;
int ofl_ctrl;
{
    Psysteme s1;
    register Pbase b;
    boolean faisable = TRUE;
    extern jmp_buf overflow_error;

    if (s == NULL)
	return(TRUE);

    s1 = sc_dup(s);
    if ((s1 = sc_kill_db_eg(s1))) {
	/* projection successive selon les  variables du systeme */
	for (b = s1->base;
	     !VECTEUR_NUL_P(b) && faisable;
	     b = b->succ)
	{	      
	    sc_projection_along_variable_ofl_ctrl(&s1, vecteur_var(b), 
						  ofl_ctrl);
	    
	    if (sc_empty_p(s1))
	    {
		faisable = FALSE;
		break;
	    }

	    if (integer_p) {
		if (SC_EMPTY_P(s1 = sc_normalize(s1))) 
		{ 
		    faisable = FALSE; 
		    break;
		}
	    }
	}
	sc_rm(s1);
    }
    else 
	/* sc_kill_db_eg a de'sallouer s1 a` la detection de 
	   sa non-faisabilite */
	faisable = FALSE;

    return(faisable);
}


