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

 /* package plint */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "matrix.h"

#include "ray_dte.h"
#include "sommet.h"

/* pour recuperer les declarations des fonctions de conversion de
 * sc en liste de sommets et reciproquement, bien que ca casse le
 * DAG des types de donnees
 */

#include "polyedre.h"

#include "plint.h"

#define MALLOC(s,t,f) malloc((unsigned)(s))
#define FREE(s,t,f) free((char *)(s))


/* void var_posit(Psysteme ps, int B[], int m, int nbl):
 * Recherche des inegalites que doivent respecter les variables
 * non contraintes pour que les variables de depart soient positives
 *
 * La matrice B passee en parametres est celle calculee a l'aide 
 * de la fonction "matrice_smith"
 *
 * Les parametres de la fonction :
 *
 *!Psysteme ps  : systeme lineaire d'inegalites
 * int 	B[]	: matrice  de dimension (m,m+1)  correspondant a la matrice 
 *		  solution du systeme d'egalites du systeme lineaire
 * int  n	: nombre de lignes de la matrice
 * int  m 	: nombre de colonnes de la matrice
 * int  nbl 	: nombre de variables non contraintes ajoutees a la matrice
 */
void var_posit(ps,B,m,nbl)
Psysteme ps;
Pmatrix B;
int m;
int nbl;
{
    Pvecteur pv=NULL;
    Pcontrainte pc=NULL;
    Pcontrainte cp = NULL;
    int i,j;
    Value den = VALUE_ONE;
    int sgn_den;
    ps->dimension = 0;
    ps->base = NULL;

    for (i = 1; i <= nbl; i++)
	(void) creat_new_var(ps);

    den =MATRIX_DENOMINATOR(B);
    sgn_den = value_sign(den);
    if (den) {
	for (i=1;i<=m; i++) {
	    Pbase b = ps->base;
	    Value tmp = MATRIX_ELEM(B,i,1);

	    cp = contrainte_new();

	    if (sgn_den==1) value_oppose(tmp);
	    pv = vect_new(TCST,tmp);

	    for (j=1;j<=nbl && b!=VECTEUR_NUL;j++, b = b->succ)
	    {
		tmp = MATRIX_ELEM(B,i,j+1);
		if (sgn_den==1) value_oppose(tmp);
		vect_chg_coeff(&pv, vecteur_var(b), tmp);
	    }
	    assert(j>nbl);

	    cp->vecteur = pv;
	    cp->vecteur = vect_clean(cp->vecteur);
	    cp->succ = pc;
	    pc = cp;
	}
    }
    ps->egalites = NULL;
    ps->inegalites = pc;
    ps->nb_ineq = nbl;
}

/* Psysteme smith_int(Psysteme ps):
 * Resolution d'un systeme d'egalites en nombres entiers par la methode de
 * Smith et recherche du systeme lineaire que doit verifier les nouvelles
 * variables non contraintes du systeme pour que les variables de depart
 * soient positives.
 *
 *  resultat retourne par la fonction :
 *
 *  Psysteme 	   : le systeme lineaire que doit verifier les  
 *		    variables non contraintes 
 *		    
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme ps     : systeme lineaire 
 */
Psysteme smith_int(ps)
Psysteme ps;
{

    Psysteme sys=sc_dup(ps);
    int i;
    Pmatrix MAT= MATRIX_UNDEFINED;
    Pmatrix MATN=MATRIX_UNDEFINED;
    Pmatrix P=MATRIX_UNDEFINED;
    Pmatrix PN=MATRIX_UNDEFINED;
    Pmatrix PN2=MATRIX_UNDEFINED;
    Pmatrix Q=MATRIX_UNDEFINED;
    Pmatrix QN=MATRIX_UNDEFINED;
    Pmatrix QN2=MATRIX_UNDEFINED;
    Pmatrix B=MATRIX_UNDEFINED;
    Pmatrix B2=MATRIX_UNDEFINED;
    Value den=VALUE_ONE;
    int nbl;
    int nblg = sys->nb_eq;     /* nombre de lignes du systeme   */
    int nbv = sys->dimension;     /* nombre de variables du systeme */
    int n;                     /* nombre d'equations du systeme   */
    int m;                     /* nombre de variables du systeme  */
    int n_min,m_min;           /* numero de la ligne et de la colonne correspondant 
				  au plus petit entier non nul appartenant a la 
				  partie triangulaire superieure de la matrice   */
    int level = 0;
    bool trouve = false;
    bool stop = false;
    bool infaisab = false;

    if(ps)
	sys = sc_normalize(sc_dup(sys));
    if (nbv && nblg && sys)
    {
	MAT = matrix_new(nblg,nbv);

	MATN = matrix_new(nblg,nbv);

	P = matrix_new(nblg,nblg);
	PN = matrix_new(nblg,nblg);
	PN2 =matrix_new(nblg,nblg );

	Q =matrix_new(nbv,nbv);
	QN =matrix_new(nbv,nbv );
	QN2 = matrix_new(nbv,nbv );

	B = matrix_new(nbv,(nbv+1));
	B2 = matrix_new(nbv,(nbv+1));

#ifdef TRACE
	printf(" systeme lineaire initial \n");
	sc_fprint (stdout,sys,noms_var);
#endif

	/* Initialisation des parametres */
	n = sys->nb_eq;
	m = sys->dimension;

	sys_mat_conv(sys,MAT,B,n,m);

	if (sys->egalites != NULL)
	    contraintes_free(sys->egalites);
	sys->egalites = NULL;

	MATRIX_DENOMINATOR(B)= VALUE_ONE;
	MATRIX_DENOMINATOR(MAT) = VALUE_ONE;

	matrix_nulle(B);

	matrix_identity(PN,0);
	matrix_identity(QN,0);
	matrix_identity(P,0);
	matrix_identity(Q,0);

	while (!stop) {
	    matrix_min(MAT,&n_min,&m_min,level);
	    if ((((n_min==1 + level) || (m_min==1+level)) && !trouve ) ||
		( (n_min >1 +level) || (m_min >1 +level))) {
		if (n_min >1 + level)
		{
		    matrix_nulle(P);
		    matrix_perm_col(P,n_min,level);

		    matrix_multiply(P,MAT,MATN);
		    matrix_multiply(P,PN,PN2);

		    matrix_assign(MATN,MAT);
		    matrix_assign(PN2,PN);
		}

#ifdef TRACE
		printf (" apres alignement du plus petit element a la premiere colonne \n");
		matrix_print(MAT);
#endif

		if (m_min >1+level)
		{
		    matrix_nulle(Q);
		    matrix_perm_line(Q,m_min,level);

		    matrix_multiply(MAT,Q,MATN);
		    matrix_multiply(QN,Q,QN2);

		    matrix_assign(MATN,MAT);
		    matrix_assign(QN2,QN);
		}

#ifdef TRACE
		printf (" apres alignement du plus petit element a la premiere ligne \n");
		matrix_print(MAT);
#endif

		if (m_min>0 && n_min >0) {
		    if(matrix_col_el(MAT,level)) {
			matrix_maj_col(MAT,P,level);

			matrix_multiply(P,MAT,MATN);
			matrix_multiply(P,PN,PN2);

			matrix_assign(MATN,MAT);
			matrix_assign(PN2,PN);
		    }

#ifdef TRACE
		    printf("apres division par A%d%d des termes de la %d-ieme colonne \n",level+1,level+
			   1,level+
			   1);
		    matrix_print(MAT);
#endif

		    if(matrix_line_el(MAT,level)) {
			matrix_maj_line(MAT,Q,level);

			matrix_multiply(MAT,Q,MATN);
			matrix_multiply(QN,Q,QN2);

			matrix_assign(MATN,MAT);
			matrix_assign(QN2,QN);
		    }
#ifdef TRACE
		    printf("apres division par A%d%d des termes de la %d-ieme ligne \n",level+1,level+
			   1,level+1);
		    matrix_print(MAT);
#endif

		}
		trouve = true;
	    }
	    else {
		if (!n_min || !m_min)
		    stop = true;
		else
		{
		    level++;
		    trouve = false;
		}
	    }
	}
#ifdef TRACE

	printf (" la  matrice D apres transformation est la suivante :");
	matrix_print(MAT);

	printf (" la matrice P est \n");
	matrix_print(PN);

	printf (" la matrice Q est \n");
	matrix_print(QN);

#endif

	/* Pre-multiplication par la matrice P */
	matrix_multiply(PN,B,B2);
	matrix_assign(B2,B);

#ifdef TRACE
	printf (" apres pre-multiplication par P \n");
	matrix_print(B);
#endif

	nbl = 2;

	for (i=1;i<=n && i<=m && !infaisab;i++)	{
	    /* Division de chaque terme non nul de B par le terme
	       correspondant de la diagonale de la matrice D */
	    if (value_notzero_p(MATRIX_ELEM(MAT,i,i))) {
		if (value_zero_p(value_mod(MATRIX_ELEM(B,i,1),
					   MATRIX_ELEM(MAT,i,i))))
		    value_division(MATRIX_ELEM(B,i,1),
				   MATRIX_ELEM(MAT,i,i));
		else 
		    infaisab = true;
	    }
	    else {
		/* Si un terme diagonal est nul, on verifie que la variable
		   correspondante est bien nulle, i.e. que son coefficient
		   dans B est bien zero et on ajoute une variable
		   non contrainte au systeme.

		   En effet, l'equation "0 * x = 0"  ==> 
		                   "la variable x est non contrainte" */
		if (value_zero_p(MATRIX_ELEM(B,i,1))) {
		    MATRIX_ELEM(B,i,nbl) = den;
		    nbl++;
		}
		else
		    /* si la variable est non nulle ==> il y a une erreur
		       ==> systeme infaisable */
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
	    if (m>n) {
		for (i=n+1; i<=m; i++,nbl++)
		    MATRIX_ELEM(B,i,nbl) = den;
	    }
	    nbl -= 2;
	    /* Pre-multiplication par la matrice Q */
	    matrix_multiply(QN,B,B2);
	    matrix_assign(B2,B);

#ifdef TRACE
	    printf (" apres pre-multiplication par Q \n");
	    matrix_print(B);
#endif

	    matrix_normalizec(B);
	    /* recherche des contraintes lineaires que doivent respecter les 
	       variables supplementaires pour que les variables de depart 
	       soient positives  */
	    var_posit(sys,B,m,nbl);
	}
	FREE(MAT,MATRIX,"smith");
	FREE(MATN,MATRIX,"smith");
	FREE(P,MATRIX,"smith");
	FREE(PN,MATRIX,"smith");
	FREE(PN2,MATRIX,"smith");
	FREE(Q,MATRIX,"smith");
	FREE(QN,MATRIX,"smith");
	FREE(QN2,MATRIX,"smith");
	FREE(B,MATRIX,"smith");
	FREE(B2,MATRIX,"smith");
    }
#ifdef TRACE
    sc_fprint(stdout,sys,noms_var);
#endif

    return(sys);
}

/* bool syst_smith(Psysteme ps):
 *  Test de faisabilite d'un systeme lineaire en nombres entiers positifs par
 *  resolution du systeme par la methode de Smith. 
 *
 *  Le resultat n'est pas toujours bon. Il existe des cas ou la fonction ne 
 *  detecte pas l'infaisabilite du systeme en nombres entiers, mais il sera
 *  du moins dans ce cas faisable en nombres reels.   
 *
 *  resultat retourne par la fonction :
 *
 *  boolean	   : true   si le systeme lineaire a une solution entiere
 *  		     false  si le systeme lineaire n'a pas de solution 
 *		     entiere   
 *
 *  Les parametres de la fonction :
 *
 *  Psommet ps     : systeme lineaire 
*/
bool syst_smith(ps)
Psysteme ps;
{
    Psysteme sys2 = NULL;
    Psysteme sys_cond_posit= NULL;
    Psommet som1 = NULL;
    bool is_faisab = true;
#ifdef TRACE 
    printf (" ** syst_smith - test de faisabilite d'un systeme avec Smith \n");
#endif

    if ((ps->egalites != NULL) || (ps->inegalites != NULL)) {

	if ( (sys2 =sc_normalize(sc_dup(ps))) !=  NULL) {

	    Pvecteur lvbase = NULL;
	    int nb_som = 0;
	    int nbvars;
	    Pbase b = BASE_NULLE;

	    nb_som = ps->nb_eq + ps->nb_ineq;
	    nbvars = ps->dimension;
	    b = base_dup(ps->base);

	    /*
	     * transformation du systeme lineaire sous la forme d'un
	     * Psommet
	     */
	    som1 = sys_som_conv(sys2,&nb_som);
	    sc_rm(sys2);

	    /*
	     * ajout des variables d'ecart et transformation des
	     * inegalites du systeme en egalites.
	     */
	    som1 = var_ecart_sup(som1,nb_som,&lvbase,&nbvars,&b);

	    if ((sys2 = som_sys_conv(som1)) != NULL) {
		sys2->dimension = nbvars;
		sys2->base = b;
	    }

	    /* resolution du systeme par la methode de Smith et
	     * recherche du systeme de contraintes
	     * (sys_cond_posit) que doit verifier les variables non
	     * contraintes pour que les variables de depart soient
	     * positives.
	     */
	    sys_cond_posit = smith_int(sys2);
	    sc_rm(sys2);

	    /* Test de faisabilite du systeme de contraintes obtenu. */
	    if  ((sys2 = sc_normalize(sc_dup(sys_cond_posit))) != NULL) 
		/*
		 * On utilise le test de faisabilite en reels, au lieu
		 * d'utiliser le test de faisabilite en entiers ==> on obtient
		 * des resultats un peu moins bon, mais un gain de temps
		 * appreciable.
		 */
		is_faisab = sc_faisabilite(sys2);
	    else
		is_faisab = false; 
	    sc_rm(sys_cond_posit);
	}
	else
	    is_faisab = false;

#ifdef TRACE
	if (is_faisab)
	    printf (" -- smith_int ==> systeme faisable \n");
	else printf (" -- smith_int ==> systeme non faisable \n");
#endif
    }
    return(is_faisab);
}
