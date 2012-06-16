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

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"

/* pour recuperer les declarations des fonctions de conversion de
 * sc en liste de sommets et reciproquement, bien que ca casse le
 * DAG des types de donnees
 */
#include "ray_dte.h"
#include "polyedre.h"
#include "matrix.h"

#include "plint.h"

#define MALLOC(s,t,f) malloc(s)

/* Psommet plint_pas(Psommet sys1, Psommet fonct, Pvecteur * lvbase,
 *                   int * nb_som, int * nbvars, Pbase *b):
 * Algorithme des congruences decroissantes 
 * (Michel Minoux - Programmation mathematique - 
 *                  theorie et algorithmes- tome 2. Dunod. p22. )
 * 
 *  resultat retourne par la fonction :
 *
 *  Psommet        : systeme lineaire modifie
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys1   : systeme lineaire 
 *  Psommet fonct  : fonction economique du  programme lineaire
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme
 *  int nbvars 	   : nombre de variables du systeme
 *  Pbase *b   : liste des variables du systeme
 */
Psommet plint_pas(sys1,fonct,lvbase,nb_som,nbvars,b)
Psommet sys1;
Psommet fonct;
Pvecteur *lvbase;
int *nb_som;
int *nbvars;
Pbase *b;
{

    double d1;
    int no_som =0;
    Variable var2 = 0;
    Value in1;
    bool non_fin = true;
    bool non_stop = true;
    bool degenere = false;

#ifdef TRACE

    Psysteme sys =NULL;
    printf(" * algorithme de GOMORY  \n");
#endif

    sys1 = primal_pivot(sys1,lvbase,*nb_som,fonct);

    if (sys1!=NULL && fonct != NULL)
    {
	if ((sol_entiere(sys1,*lvbase,*nb_som) == false) || 
	    (sol_positive(sys1,*lvbase,*nb_som) == false) || const_negative(sys1))
	{


	    while ((sys1 != NULL) && 
		   (dual_pivot_pas(&sys1,lvbase,*nb_som,fonct,nbvars,b) == true));
	    if (sys1 != NULL && fonct != NULL)
	    {
		if ((sol_entiere(sys1,*lvbase,*nb_som)==false))
		{
		    Value f0 = VALUE_ZERO,fden;

		    while (non_fin)
		    {
			f0 = value_uminus(vect_coeff(TCST,fonct->vecteur));
			fden = fonct->denominateur;
			d1 = VALUE_TO_DOUBLE(f0) / VALUE_TO_DOUBLE(fden);
			in1 = value_div(f0,fden);

			/* test de degenerescence */
			if ((d1 != VALUE_TO_DOUBLE(in1)) || 
			    value_zero_p(in1) ||
			    (cout_nul(fonct,*lvbase,*nbvars,*b) ==
			     false) || degenere) {
			    degenere = false;

			    while (non_stop)
			    {
				Psommet eq_gom = NULL;
				Psommet eq_coup = NULL;

				/* recherche d'une variable de base non entiere,
				   puis ajout d'une coupe de Gomory  */

				if (((eq_gom = gomory_eq(&sys1,*lvbase,*nb_som,&no_som,&var2)) != NULL) && (sys1 !=NULL)) {
				    if ((eq_coup = gomory_trait_eq(eq_gom,var2)) != NULL)
					sommet_add(&sys1,eq_coup,nb_som);
				    else {
					sommets_rm(sys1);
					sys1 = NULL;
					non_fin = false;
					non_stop = false;
				    }
#ifdef TRACE
				    sys = som_sys_conv(sys1);
				    sc_fprint(stdout,sys,noms_var);
#endif
				}
				else {
				    /* il n'y a plus de variable non entiere */
				    non_stop = false ;
				    non_fin = false;
				}
				if ((sys1 != NULL) && non_stop )
				{
				    /* on effectue un pas de l'algorithme dual du simplexe */
				    /* ajout des variables d'ecart   */
				    sys1 = var_ecart_sup (sys1,*nb_som,lvbase,nbvars,b);
				    if (dual_pivot_pas(&sys1,lvbase,*nb_som,fonct,nbvars,b) != false) {
					if  (sol_entiere(sys1,*lvbase,*nb_som) == true)	{
					    /* la solution est entiere mais pas positive */
					    if (ligne_pivot(sys1,*nb_som,&no_som) != NULL) {
						while ((sys1 != NULL)
						       && (dual_pivot_pas(&sys1,lvbase,*nb_som,fonct,nbvars,b) == true))
						    ;

						if ((sol_entiere(sys1,*lvbase,*nb_som) == true) && (sol_positive(sys1,*lvbase,*nb_som) == true))
						{
#ifdef TRACE
						    printf(" - Gomory - on a une solution optimale \n");
#endif
						    non_fin = false;
						    non_stop = false;
						}
						else {
						    if (sys1 == NULL) {
							non_fin = false;
							non_stop = false;
						    }
						}
					    }
					    else {
#ifdef TRACE					
						printf (" - Gomory - on a une solution optimale \n");
#endif
						non_fin = false;
						non_stop = false;
					    }
					}
				    }
				    else {
					/* on a une solution positive */
					if (sys1 == NULL)
					{
					    non_stop = false;
					    non_fin = false;
					}
					else {
					    if (sol_entiere(sys1,*lvbase,*nb_som) == true)
					    {
#ifdef TRACE
						printf (" -  Gomory - on a une solution entiere \n");
#endif
						non_stop = false;
						non_fin = false;
					    }
					}
				    }
				}
#ifdef TRACE
				sys = som_sys_conv(sys1);
				sc_fprint(stdout,sys,noms_var);
#endif
			    }
			}
			else
			{
			    /* cas de degenerescence   */
			    degenere = true;
			    if (plint_degen(&sys1,fonct,nb_som,lvbase,nbvars,b) == false)
			    {
				if (sys1!= NULL)
				{
#ifdef TRACE
				    printf (" - Gomory - on a une solution reelle \n");
				    sys = som_sys_conv(sys1);
				    sc_fprint(stdout,sys,noms_var);
#endif
				}
				non_fin = false;
			    }
			}
		    }
		}
		else {
		    /* on a une sol. entiere et positive a la fin de */
		    /* l'execution de l'algorithme dual du simplexe  */

#ifdef TRACE
		    printf (" - Gomory - on a une solution  \n");
#endif
		}
	    }
#ifdef TRACE
	    else printf (" - Gomory - le systeme est non realisable - fin !!\n");
#endif
	}
	else {
	    /* on a une solution entiere et positive a la fin */
	    /* de l'execution de l'algorithme primal du simplexe */
#ifdef TRACE
	    printf (" - Gomory - on a une solution optimale \n");
	    sys = som_sys_conv(sys1);
	    sc_fprint(stdout,sys,noms_var);
#endif

	}
    }
    else {
#ifdef TRACE
	printf (" -  Gomory - pas de sol. reelle ==> pas de sol. entiere \n");
#endif
    }
    return (sys1);
}

/* void plint_degen(Psommet *sys, Psommet fonct, int *nb_som,
 *                  Pvecteur * lvbase, int nbvars,  Pbase * b):
 * Cas de degenerescence au cours de l'algorithme des
 * congruences decroissantes
 *
 *  resultat retourne par la fonction :
 *
 *  Le systeme initial est modifie.
 *  S'il est vide, c'est que le systeme est non faisable.
 *  Sinon on ajoute une contrainte supplementaire permettant la 
 *  non degenerescence du programme lineaire.
 *
 *  Les parametres de la fonction :
 *
 *  Psommet sys    : systeme lineaire 
 *  Psommet fonct  : fonction economique du  programme lineaire
 *  Pvecteur lvbase: liste des variables de base du systeme
 *  int     nb_som : nombre de contraintes du systeme
 *  int nbvars 	   : nombre de variables du systeme
 *  Pbase b : liste des  variables du systeme
 */
bool plint_degen(sys,fonct,nb_som,lvbase,nbvars,b)
Psommet *sys;
Psommet fonct;
int  *nb_som;
Pvecteur *lvbase;
int *nbvars;
Pbase *b;
{
    Psommet fonct2=NULL;
    Psommet eq_coupe=NULL;
    Psommet sys1 = sommets_dupc(*sys);
    Pvecteur eq_for_ndegen=NULL;
    Pvecteur lvhb_de_cnnul=NULL;
    Pvecteur pv=NULL;
    Pvecteur pv2=VECTEUR_NUL;
    Pvecteur explv=NULL;
    Variable var_entrant;
    register int i;
    int exnbv;
    int exnbs = *nb_som;
    bool result = true;

#ifdef TRACE
    printf(" ** Gomory - cas de degenerescence \n");
#endif

    /* construction de la liste des variables hors base de cout non nul*/
    lvhb_de_cnnul = vect_dup(fonct->vecteur);

    for (pv= *lvbase;pv!=NULL;pv=pv->succ)
    {
	vect_chg_coeff(&lvhb_de_cnnul,pv->var,0);
    }

    vect_chg_coeff(&lvhb_de_cnnul,TCST,0);
    oter_lvbase(sys1,lvhb_de_cnnul);
    exnbv = *nbvars;
    explv = vect_dup(*lvbase);

    /* construction d'une nouvelle fonction economique pour le systeme
       dont on a elimine les variables h.base de cout non nul */

    fonct2 = (Psommet)MALLOC(sizeof(Ssommet),SOMMET,"plint_degen");
    fonct2->denominateur = VALUE_ONE;
    fonct2->vecteur = vect_new(vecteur_var(*b),VALUE_ONE);
    for (i =1,pv2 = (*b)->succ ; i< *nbvars && !VECTEUR_NUL_P(pv2); 
	 i++, pv2=pv2->succ)
	vect_add_elem(&(fonct2->vecteur),vecteur_var(pv2),VALUE_ONE);

    for (pv= lvhb_de_cnnul;pv!=NULL;pv=pv->succ)
   vect_chg_coeff(&(fonct2->vecteur),pv->var,VALUE_ZERO);
  
    vect_chg_coeff(&(fonct2->vecteur),TCST,VALUE_ZERO);

    fonct2->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"plint_degen");
    *(fonct2->eq_sat) = 0;
    fonct2->succ = NULL;

    /* recherche d'une solution entiere pour ce nouveau systeme */
    if (sys1 != NULL)
sys1 = plint_pas(sys1,fonct2,lvbase,nb_som,nbvars,b);
  if (sys1 != NULL)

    {
#ifdef TRACE
	printf (" -- Gomory degen. - le pb. a une sol. optimale \n ");
#endif
	result = false;
	vect_rm(explv);
    }
    else {
	/* le systeme n'admet aucune solution entiere  */
	*nbvars = exnbv;
	*nb_som = exnbs;
	vect_rm(*lvbase);
	*lvbase = explv;
	/*   ajout inegalite permettant la non-degenerescence du 
	     programme lineaire   */

	eq_for_ndegen = vect_dup(lvhb_de_cnnul);
	for (pv = eq_for_ndegen; pv!= NULL; pv = pv->succ)
	    vect_chg_coeff(&pv,pv->var,VALUE_MONE);
	vect_chg_coeff(&eq_for_ndegen,TCST,VALUE_ONE);
	eq_coupe = (Psommet) MALLOC(sizeof(Ssommet),SOMMET,"plint_degen");
	eq_coupe->vecteur = eq_for_ndegen;
	eq_coupe->eq_sat = (int *)MALLOC(sizeof(int),INTEGER,"plint_degen");
	*(eq_coupe->eq_sat) = -1;
	eq_coupe->denominateur = VALUE_ONE;
	eq_coupe->succ=NULL;
	sommet_add(sys,eq_coupe,nb_som);

	*sys = var_ecart_sup(*sys,*nb_som,lvbase,nbvars,b);

	/* on recherche la variable pivot et on effectue le pivot  */
	var_entrant = var_pivotd(eq_coupe,fonct);

	if (var_entrant != NULL)
	{

	    pivoter (*sys,eq_coupe,var_entrant,fonct);
	    lvbase_ote_no_ligne(1,lvbase);
	    lvbase_add(var_entrant,1,lvbase);
	}
	else 
	{
#ifdef TRACE
	    printf (" -- Gomory degen. - le pb. n'est pas borne \n");
#endif
	    result = false;
	    sommets_rm(*sys);
	    *sys = NULL;
	}
    }
    vect_rm(lvhb_de_cnnul);
    sommets_rm(sys1);
    sommets_rm(fonct2);

    return(result);
}

/* Psysteme plint(Psysteme first_sys, Psommet fonct, Psolution *sol_fin):
 * resolution d'un systeme lineaire en nombres entiers positifs par 
 * l'algorithme general des congruences decroissantes 
 * (c.f. Programmation Mathematique - tome 2. M.MINOUX (83))
 *
 *  
 *  resultat retourne par la fonction :
 *
 *  Psysteme       : systeme lineaire donnant la solution de base du 
 *		     systeme ,si celui ci est realisable.
 *		     Dans ce cas, son cout optimal vaut
 *		     la valeur (avec le signe - ) de la constante de la 
 *		     fonction economique et la solution de base est 
 *		     exprimee sous la forme d'une Psolution.
 *			
 *  NULL		   : si le systeme initial n'est pas realisable. 
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme first_syst  : systeme lineaire initial
 *  Psommet fonct       : fonction economique du  programme lineaire
 */
Psysteme plint(first_sys,fonct,sol_fin)
Psysteme first_sys;
Psommet fonct;
Psolution *sol_fin;
{
    Psommet sys1 = NULL;
    Psysteme sys2 = NULL;
    Psysteme syst_res = NULL;
    Pvecteur lvbase = NULL;
    int nb_som = 0;
    int nbvars;
    Pbase b = BASE_NULLE;

#ifdef TRACE
    Psolution sol = NULL;

    printf(" * algorithme des congruences decroissantes global \n");
    printf (" - GOMORY - le nb. de variables du systeme est %d \n",first_sys->dimension);
#endif
    if (first_sys) {
	nbvars = first_sys->dimension;
	b = base_dup(first_sys->base);
	

	if ((first_sys->egalites != NULL) || (first_sys->inegalites != NULL))
	{

	    sys2 =sc_normalize(sc_dup(first_sys));
	    if (sys2 != (Psysteme) NULL && syst_smith(sys2))
		sys1 = sys_som_conv(sys2,&nb_som);

#ifdef TRACE
	    printf (" - GOMORY - le nb. de contraintes est %d \n",nb_som);
#endif
	    /* ajout des variables d'ecart   */
	    if ((sys1 = eq_in_ineq(&sys1,&nb_som,&lvbase)) != (Psommet) NULL) {
		sys1 = var_ecart_sup(sys1,nb_som,&lvbase,&nbvars,&b);
		sys1 = primal_pivot(sys1,&lvbase,nb_som,fonct);
	    }

	    sys1 = plint_pas(sys1,fonct,&lvbase,&nb_som,&nbvars,&b);

	    if (sys1 && fonct) {
		*sol_fin = sol_finale(sys1,lvbase,nb_som);

#ifdef TRACE
		for (sol = *sol_fin; sol != NULL; sol = sol->succ)
		    printf (" %s == %f; ",noms_var(sol->var),(double)(sol->val / sol->denominateur));
		printf (" \n ");
		printf (" - GOMORY - la fonct. economique vaut %d\n ",vect_coeff(TCST,fonct->vecteur)/fonct->denominateur);

#endif

	    }
	    if (sys1 == NULL && fonct != NULL)  fonct->vecteur = NULL;

	    if ((syst_res = som_sys_conv(sys1)) != NULL) {
		syst_res->base = b;
		syst_res->dimension = nbvars;
	    }
	}
	sommets_rm(sys1);
	vect_rm(lvbase);
	sc_rm(sys2);
    }
    return(syst_res);
}
