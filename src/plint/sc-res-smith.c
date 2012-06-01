/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

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

 /* Package plint (Programmation Lineaire en nombres entiers, i.e. INTeger) */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "matrix.h"

#include "sommet.h"
#include "plint.h"

#define MALLOC(s,t,f) malloc((unsigned)(s))
#define FREE(s,t,f) free((char *)(s))

#define MATRIX 0

/* Psysteme sc_resol_smith(Psysteme ps):
 *  Resolution d'un systeme d'egalites en nombres entiers par la methode 
 *  de Smith.  (c.f. Programmation Lineaire. M.MINOUX. (83))
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme 	   :  systeme lineaire dont le systeme d'inequations est  
 *		      identique a celui du systeme initial.
 * 
 *	      	      le systeme d'equations est remplace par le systeme 
 * 		      d'egalites correspondant a la solution du systeme 
 *		      d'equations du systeme d'egalites initial.   
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme ps     : systeme lineaire 
 */
Psysteme sc_resol_smith(ps)
Psysteme ps;
{

    Psysteme sys=sc_dup(ps);
    int i;

    Pmatrix MAT= MATRIX_UNDEFINED;
    Pmatrix P=MATRIX_UNDEFINED;
    Pmatrix Q=MATRIX_UNDEFINED;
    Pmatrix B=MATRIX_UNDEFINED;
    Pmatrix B2=MATRIX_UNDEFINED;
    Pmatrix D=MATRIX_UNDEFINED;

    Value den=VALUE_ONE;
    int nbl;
    int n; 
    int m;
    bool infaisab = false;
    if (ps) {
	sys = sc_normalize(sc_dup(sys));
	m = sys->dimension;
	n = sys->nb_eq;

	if (m && (n >1) && (sys != NULL)) {

	    MAT = matrix_new(n,m);
	    D = matrix_new(n,m);

	    P = matrix_new(n,n);
	    Q = matrix_new(m,m);

	    B = matrix_new(m,m);
	    B2 = matrix_new(m,m);

#ifdef TRACE
	    printf(" systeme lineaire initial \n");
	    sc_fprint (stdout,sys,noms_var);
#endif

	    sys_mat_conv(sys,MAT,B,n,m);

	    if (sys->egalites != NULL) contraintes_free(sys->egalites);
	    sys->egalites = NULL;

	    MATRIX_DENOMINATOR(B)=VALUE_ONE;
	    MATRIX_DENOMINATOR(MAT) = VALUE_ONE;

	    matrix_nulle(B2);

	    matrix_smith(MAT,P,D,Q);

	    matrix_assign(D,MAT);

	    /* Pre-multiplication par la matrice P */

	    matrix_multiply(P,B,B2);
	    matrix_assign(B2,B);


#ifdef TRACE
	    printf (" apres pre-multiplication par P \n");
	    matrix_print(B);
	    matrix_print(D);
#endif

	    nbl = 2;
	    for (i=1;i<=n && i<=m && !infaisab;i++)
	    {
		/*
		 * Division de chaque terme non nul de B par le 
		 terme correspondant de la
		 * diagonale de la matrice D
		 */
		Value matii=MATRIX_ELEM(MAT,i,i);
		if (value_notzero_p(matii))
		    if (value_zero_p(value_mod(MATRIX_ELEM(B,i,1), matii)))
			value_division(MATRIX_ELEM(B,i,1),matii);
		    else infaisab = true;
		
		else
		{
		    /*
		     * Si un terme diagonal est nul, on verifie 
		     que la variable correspondante est
		     * bien nulle, i.e. que son coefficient dans 
		     B est bien zero et on
		     * ajoute une variable non contrainte au systeme.
		     * En effet, l'equation "0 * x = 0"  
		     ==> la variable x est non contrainte
		     */
		    if (value_zero_p(MATRIX_ELEM(B,i,1)))
		    {   MATRIX_ELEM(B,i,nbl) = den;
			nbl++;
		    }
		    else
			/*
			 * si la variable est non nulle ==> il y a une erreur ==> systeme infaisable
			 */
			infaisab = true;

		}
	    }


	    if (infaisab) {
		matrix_nulle(B);
		sc_rm(sys);
		sys = NULL;
#ifdef TRACE
		printf (" systeme infaisable en nombres entiers \n");
#endif
	    }
	    else {

#ifdef TRACE
		printf (" apres division par les elements diagonaux de D \n");
		matrix_print(B);
#endif

		/* ajout des variables non contraintes  */
		if (m>n)
		 for (i=n+1; i<=m; i++,nbl++)
			MATRIX_ELEM(B,i,nbl) = den;
		nbl -=2;

		/* Pre-multiplication par la matrice Q */

		matrix_multiply(Q,B,B2);
		matrix_assign(B2,B);
#ifdef TRACE
		printf (" apres pre-multiplication par Q \n");
		matrix_print(B);
#endif

		matrix_normalizec(B);

		/* conversion en systeme lineaire */
#ifdef TRACE
		printf (" conversion en systeme lineaire \n");
#endif
		mat_sys_conv(sys,B,m,m,nbl);

	    }

	    FREE(MAT,MATRIX,"smith");
	    FREE(P,MATRIX,"smith");
	    FREE(Q,MATRIX,"smith");
	    FREE(B,MATRIX,"smith");
	    FREE(B2,MATRIX,"smith");
	}

    }
#ifdef TRACE
    sc_fprint(stdout,sys,noms_var);
#endif

    return(sys);
}
