 /* package polyedre */
 
 /* diverses fonctions de calcul de saturations
  *
  * Malik Imadache, Francois Irigoin
  *
  * Modifie par Francois Irigoin:
  *  - reprise des includes
  */

#include <stdio.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "sommet.h"
#include "ray_dte.h"
#include "sg.h"

#include "polyedre.h"

#include "saturation.h"

#define MALLOC(s,t,f) malloc((unsigned)(s))

/* int satur_som(Psommet s, Pcontrainte eq): saturation d'une contrainte eq
 * (implicitement une inegalite) par un sommet s
 *
 * D'apres Halbwachs page 9 section 1.3.1, un point X (cas general d'un sommet)
 * sature une contrainte si A . X = B
 *
 * Ici, on calcule A. X*denominateur(X) - B*denominateur(X). Attention, le
 * terme constant B a ete passe en terme gauche: A X - B = 0. D'ou le
 * '+' COEEF_CST
 *
 * Il faut donc tester satur_som()==0 pour savoir si le sommet ad_som
 * sature l'inegalite eq.
 *
 * il faut tester satur_som <= 0 pour savoir si le sommet s verifie
 * l'inegalite eq
 */
int satur_som(s,eq)
Psommet s;
Pcontrainte eq;
{
    int saturation;
    saturation =   vect_prod_scal(s->vecteur,eq->vecteur) 
	   + (COEFF_CST(eq) * s->denominateur);
    return(saturation);
}

/* int satur_vect(Pvecteur r, Pcontrainte eq): saturation d'une contrainte eq
 * (implicitement une inegalite)
 * par un rayon r (represente ici uniquement par son vecteur)
 *
 * D'apres Halbwachs page 9 section 1.3.1, un rayon r sature une contrainte
 * si A . r = 0
 *
 * Le test de saturation est donc satur_vect(r,eq)==0
 *
 * Le rayon r respecte l'inegalite eq si satur_vect(r,eq) <= 0 car les
 * inegalites sont mises sous la forme A . X <= B
 */
int satur_vect(r,eq)
Pvecteur r;
Pcontrainte eq;
{ 
    return( vect_prod_scal(r,eq->vecteur)); 
}

/* void ad_s_eq_sat(Psommet ns, Pcontrainte eq_vues, int nb_vues, int nb_eq):
 *
 * on a un nouveau sommet ns pour lequel on veut calculer les nb_vues
 * premieres saturations pour les premieres equations de la liste eq_vues
 * le resultat est dans un tableau (ns->eq_sat) ou la valeur de saturation
 * dans une case d'indice i est associee a l'equation de rang i dans la
 * liste donnee
 *
 * nb_eq est le nombre total d'equation de la liste, l'allocation de eq_sat
 * est faite avec un tel nombre car il se peut que ce sommet soit
 * conserve et que de nouvelles saturation soient a calculer pour lui.
 * on ne calcule pas non plus toutes les saturations car ce sommet peut
 * etre detruit sans les utiliser
 * Modification : ajout de test eg!=NULL dans borne for (YY, 22/10/91) 
 */
void ad_s_eq_sat(ns,eq_vues,nb_vues,nb_eq)
Psommet ns;
Pcontrainte eq_vues;
int nb_vues,nb_eq;
{
    int i;
    Pcontrainte eq;
    if (nb_eq==0) return;
    ns->eq_sat = (int *) MALLOC(nb_eq * sizeof(int),TSATX,"as_s_eq_sat");
    /******** BB, 92.06: cleaned following code ***********/
    /* old code:
     * for (i=0,eq=eq_vues;i<=nb_vues,eq!=NULL;i++,eq=eq->succ)
     *     (ns->eq_sat)[i] = satur_som(ns,eq);
     */
    for (i=0,eq=eq_vues; /*i<=nb_vues,*/ eq!=NULL;i++,eq=eq->succ) {
	(ns->eq_sat)[i] = satur_som(ns,eq);
    }
    /* end of cleaning */
}

/* void ad_r_eq_sat(Pray_dte nr, Pcontrainte eq_vues, int nb_vues, int nb_eq):
 *
 * on a un nouveau rayon nr pour lequel on veut calculer les nb_vues
 * premieres saturations pour les premieres equations de la liste eq_vues
 * le resultat est dans un tableau (nr->eq_sat) ou la valeur de saturation
 * dans une case d'indice i est associee a l'equation de rang i dans la
 * liste donnee
 *
 * nb_eq est le nombre total d'equation de la liste, l'allocation de eq_sat
 * est faite avec un tel nombre car il se peut que ce sommet soit
 * conserve et que de nouvelles saturation soient a calculer pour lui.
 * on ne calcule pas non plus toutes les saturations car ce sommet peut
 * etre detruit sans les utiliser
 * Modification : ajout de test eg!=NULL dans le borne de for (YY, 22/10/91) 
 */
void ad_r_eq_sat(nr,eq_vues,nb_vues,nb_eq)
Pray_dte nr;
Pcontrainte eq_vues;
int nb_vues,nb_eq;
{
    int i;
    Pcontrainte eq;
    if (nb_eq==0) return;
    nr->eq_sat = (int *) MALLOC(nb_eq * sizeof(int),TSATX,"ad_r_eq_sat");
    /******** BB, 92.06: cleaned following code ***********/
    /* old code:
     * for (i=0,eq=eq_vues;i<=nb_vues,eq!=NULL;i++,eq=eq->succ)
     *      (nr->eq_sat)[i] = satur_vect(nr->vecteur,eq);
     */
    for (i=0,eq=eq_vues; /*i<=nb_vues,*/ eq!=NULL;i++,eq=eq->succ) {
	(nr->eq_sat)[i] = satur_vect(nr->vecteur,eq);
    }
    /* end of cleaning */
}

/* void d_sat_s_inter(Ps_sat_inter tab_sat, int nb_s, Psommet lsoms,
 *                    Pcontrainte eq, int num_eq):
 * initialisation de la structure de donnees locale a l'intersection de
 * SG et Demi-espace, on calcule les saturations afin de savoir les
 * combinaisons a faire
 *
 * on n'oublie pas d'affecter le champ eq_sat des sommets
 * pour l'elimination de redondance
 */
void d_sat_s_inter(tab_sat,nb_s,lsoms,eq,num_eq)
Ss_sat_inter *tab_sat;
int nb_s;
Psommet lsoms;
Pcontrainte eq;
int num_eq;
{
    int i; Psommet s;

    for (i=0,s=lsoms;i<nb_s;i++,s=s->succ) {
	tab_sat[i].val_sat = satur_som(s,eq);
	tab_sat[i].sv = satur_vect(s->vecteur,eq);
	tab_sat[i].som_sat = s;
	(s->eq_sat)[num_eq] = tab_sat[i].val_sat;
    }
}

/* void d_sat_vect_inter(Ps_sat_inter tab_sat, int nb_vect, Pray_dte list_v,
 *                    Pcontrainte eq, int num_eq):
 * initialisation de la structure de donnees locale a l'intersection de
 * SG et Demi-espace, on calcule les saturations afin de savoir les
 * combinaisons a faire
 *
 * on n'oublie pas d'affecter le champ eq_sat des sommets
 * pour l'elimination de redondance
 */
void d_sat_vect_inter(tab_sat,nb_vect,list_v,eq,num_eq)
Svect_sat_inter *tab_sat;
int nb_vect;
Pray_dte list_v;
Pcontrainte eq;
int num_eq;
{
    int i;
    Pray_dte v;

    for (i=0,v=list_v;i<nb_vect;i++,v=v->succ) {
	tab_sat[i].val_sat = satur_vect(v->vecteur,eq);
	tab_sat[i].vect_sat = v;
	(v->eq_sat)[num_eq] = tab_sat[i].val_sat;
    }
}

/* void sat_s_inter(Ps_sat_inter tab_sat, int nb_s, Psommet lsoms, 
 *                  Pcontrainte eq):
 * meme usage que les fonctions precedentes sauf qu'ici on n'utilise pas
 *	l'elimination de redondance -> intersection avec un hyperplan
 *	(rappel : on n'elimine pas les redondances sur les egalites)
 */
void sat_s_inter(tab_sat,nb_s,lsoms,eq)
Ss_sat_inter *tab_sat;
int nb_s;
Psommet lsoms;
Pcontrainte eq;
{
    int i;
    Psommet s;
    for (i=0,s=lsoms;i<nb_s;i++,s=s->succ) {
	tab_sat[i].val_sat = satur_som(s,eq);
	tab_sat[i].sv = satur_vect(s->vecteur,eq);
	tab_sat[i].som_sat = s;
    }
}

/* void sat_vect_inter(Pvect_sat_inter tab_sat, int nb_vect, Pray_dte liste_v,
 *                     Pcontrainte eq):
 */
void sat_vect_inter(tab_sat,nb_vect,list_v,eq)
Svect_sat_inter *tab_sat;
int nb_vect;
Pray_dte list_v;
Pcontrainte eq;
{
    int i;
    Pray_dte v;

    for (i=0,v=list_v;i<nb_vect;i++,v=v->succ) {
	tab_sat[i].val_sat = satur_vect(v->vecteur,eq);
	tab_sat[i].vect_sat = v;
    }
}

/* void free_inter(Ps_sat_inter sat_ss
 *                 Pvect_sat_inter sat_rs,
 *                 Pvect_sat_inter sat_ds,
 *                 int nb_ss,
 *                 int nb_rs,
 *                 int nb_ds): 
 * elimination d'elements generateurs devenus inutiles
 * dans l'intersection d'un polyedre avec un hyperplan
 */
void free_inter(sat_ss,sat_rs,sat_ds,nb_ss,nb_rs,nb_ds)
Ss_sat_inter *sat_ss;
Svect_sat_inter *sat_rs,*sat_ds;
int nb_ss,nb_rs,nb_ds;
{
    int i;
    for (i=0;i<nb_ss;i++) {
	if (sat_ss[i].val_sat != 0) sommet_rm(sat_ss[i].som_sat);
    }
    for (i=0;i<nb_rs;i++) {
	if (sat_rs[i].val_sat != 0) ray_dte_rm(sat_rs[i].vect_sat);
    }
    for (i=0;i<nb_ds;i++) {
	if (sat_ds[i].val_sat != 0) ray_dte_rm(sat_ds[i].vect_sat);
    }
}

/* void free_demi_inter(Ps_sat_inter sat_ss,
 *                      Pvect_sat_inter sat_rs,
 *                      int nb_ss,
 *                      int nb_rs):
 * elimination lors de l'intersection de SG avec un demi-espace
 */
void free_demi_inter(sat_ss,sat_rs,nb_ss,nb_rs)
Ss_sat_inter *sat_ss;
Svect_sat_inter *sat_rs;
int nb_ss,nb_rs;
{
    int i;
    for (i=0;i<nb_ss;i++) {
	if (sat_ss[i].val_sat > 0) SOMMET_RM(sat_ss[i].som_sat,
					     "free_demi_inter");
    }
    for (i=0;i<nb_rs;i++) {
	if (sat_rs[i].val_sat > 0)
	    RAY_DTE_RM(sat_rs[i].vect_sat,"free_demi_inter");
    }
}
