 /* operateurs d'intersection de polyedres et d'hyperplan
  * (ainsi que pour les intersections avec des demi-espaces)
  *
  * Malik Imadache
  * 
  * Pour une intersection avec un demi-espace nous ajoutons ici
  * les fonctions qui n'eliminent pas la redondance, nous rappelons que
  * la fonction associee eliminant les redondances ne sera utilisee que pour
  * la recherche de systeme generateur par intersections successives.
  * (c'est la fonction new_sg_e du fichier inter-demi.c)
  * En ce qui concerne les intersections avec des hyperplans, nous n'elimi-
  * jamais la redondance du fait que le nombre d'egalite est reduit et que
  * les elements generateurs alors crees ne sont pas nombreux
  *
  * Modifications par Francois Irigoin, le 24 mars 1989:
  *  - reprise des includes
  *  - reformattages divers
  *  - ajout d'un troisieme argument aux MALLOCs
  *  - suppression de variables non utilisees
  *  - addition d'un appel a sc_normalize a la fin de add_ineg_poly
  */

#include <stdio.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"

#include "polyedre.h"

#include "saturation.h"

#define MALLOC(s,t,f) malloc((unsigned)(s))
#define FREE(p,t,f) free((char *)(p))

/* fonctions d'intersection de polyedre avec un demi-espace
 * sans elimination de redondance
 *
 * leur structure est la meme que celle  des fonctions decrites
 * dans inter-demi.c
 */

/* Stsg_soms ns_demi_red(Ss_sat_inter * sat_s,
 *                       Svect_sat_inter *sat_r,
 *                       Svect_sat_inter *sat_d,
 *                       int nb_s, int nb_r, int nb_d):
 *
 * creation des sommets de cette (laquelle?) intersection
 */
Stsg_soms ns_demi_red(sat_s,sat_r,sat_d,nb_s,nb_r,nb_d)
Ss_sat_inter *sat_s;
Svect_sat_inter *sat_r;
Svect_sat_inter *sat_d;
int nb_s;
int nb_r;
int nb_d;
{
    int s1;
    Stsg_soms result;

    result = new_s_hyp_red(sat_s,sat_r,sat_d,nb_s,nb_r,nb_d);
    for (s1=0;s1<nb_s;s1++) {

	/* il faut aussi retenir tous les sommets de saturation negative */

	if (sat_s[s1].val_sat < 0) {
	    if (som_in_liste(sat_s[s1].som_sat,result.ssg))
		SOMMET_RM(sat_s[s1].som_sat,"ns_demi_red5");
	    else {
		sat_s[s1].som_sat->succ = result.ssg;
		result.ssg = sat_s[s1].som_sat;
		result.nb_s++;
	    }
	}
    }
    return(result);
}

/**** fonctions de construction de nouveaux rayons ****/


/* Stsg_vects ns_demi_red(Svect_sat_inter *sat_r,
 *                        Svect_sat_inter *sat_d,
 *                        int nb_r, int nb_d):
 *
 * creation des rayons de cette (laquelle?) intersection
 */
Stsg_vects nr_demi_red(sat_r,sat_d,nb_r,nb_d)
Svect_sat_inter *sat_r;
Svect_sat_inter *sat_d;
int nb_r;
int nb_d;
{
    int r1;		/* reperes dans sat_r des rayons a considerer */
    int d;		/* reperes dans sat_r(d) de l'element a considerer */
    Stsg_vects result;

    result = new_r_hyp_red(sat_r,sat_d,nb_r,nb_d);
    for (r1=0;r1<nb_r;r1++) {

	/* tt rayon de saturation negative est aussi a retenir */

	if (sat_r[r1].val_sat < 0) {
	    if (rd_in_liste(sat_r[r1].vect_sat,result.vsg))
		RAY_DTE_RM(sat_r[r1].vect_sat,"nr_demi_red4");
	    else {
		sat_r[r1].vect_sat->succ = result.vsg;
		result.vsg = sat_r[r1].vect_sat;
		result.nb_v++;
	    }
	}
    }

    /* il ne faut pas non plus oublier de recuperer toutes les demi-droites
       qui correspondent aux droites de saturation negative ou aux opposees
       des droites de saturation positive
       */

    for (d=0;d<nb_d;d++) {
	if (sat_d[d].val_sat != 0) {
	    sat_d[d].vect_sat->succ = result.vsg;
	    result.vsg = sat_d[d].vect_sat;
	    if (sat_d[d].val_sat > 0)
		(void) vect_multiply((sat_d[d].vect_sat)->vecteur,-1);
	    (sat_d[d].vect_sat)->eq_sat = NULL;
	    result.nb_v++;
	}
    }
    return(result);
}


/* Ptsg nsg_e_red(Ptsg p_sg, Pcontrainte eq): calcul du systeme generateur
 * de l'intersection d'un demi_espace donne par son equation et d'un
 * polyedre donne par son systeme generateur
 *
 * ATTENTION:
 * le systeme generateur donne, p_sg, est remplace par le systeme resultant 
 */
Ptsg nsg_e_red(p_sg,eq) 
Ptsg p_sg;
Pcontrainte eq;
{
    Ss_sat_inter *sat_s;/* tableau de structures ( saturation,&sommet ) */

    Svect_sat_inter *sat_r,*sat_d;
    /* tableaux de structures ( saturation, &vecteur ) */

    int nb_s = sg_nbre_sommets(p_sg);
    int nb_r = sg_nbre_rayons(p_sg);
    int nb_d = sg_nbre_droites(p_sg);/* nombre d'elements generateurs */

    Stsg_soms new_soms;
    Stsg_vects new_rays,new_dtes;/* structures 
				   (nb_elems_construits,listes_new_elems) */

    /* allocation memoire pour ces tableaux */

    sat_s = (Ss_sat_inter *) 
	MALLOC(nb_s * sizeof(Ss_sat_inter),SAT_TAB,"nsg_e_red");
    sat_r = (Svect_sat_inter *) 
	MALLOC(nb_r * sizeof(Svect_sat_inter),SAT_TAB,"nsg_e_red");
    sat_d = (Svect_sat_inter *) 
	MALLOC(nb_d * sizeof(Svect_sat_inter),SAT_TAB,"nsg_e_red");

    /* initialisation de ces tableaux et calcul des saturations */

    sat_s_inter(sat_s,nb_s,(p_sg->soms_sg).ssg,eq);
    sat_vect_inter(sat_r,nb_r,(p_sg->rays_sg).vsg,eq);
    sat_vect_inter(sat_d,nb_d,(p_sg->dtes_sg).vsg,eq);

    /* construction des elements generateurs */

    new_soms = ns_demi_red(sat_s,sat_r,sat_d,nb_s,nb_r,nb_d);
    new_dtes = mk_new_d(sat_d,nb_d);
    new_rays = nr_demi_red(sat_r,sat_d,nb_r,nb_d);

    /* elimination des anciens elements generateurs inutils */
    /* i.e.   ceux    de     saturation     non       nulle */

    free_demi_inter(sat_s,sat_r,nb_s,nb_r); 
    if (sat_s!=NULL) FREE((char *)sat_s,SAT_TAB,"new_sg_e");
    if (sat_r!=NULL) FREE((char *)sat_r,SAT_TAB,"new_sg_e");
    if (sat_d!=NULL) FREE((char *)sat_d,SAT_TAB,"new_sg_e");

    /* on raccroche les listes resultantes a la structure donnee */

    p_sg->soms_sg = new_soms;
    p_sg->rays_sg = new_rays;
    p_sg->dtes_sg = new_dtes;

    return(p_sg);
}

/* void add_ineg_poly(Ppoly p, Pcontrainte eq): ajout d'une inegalite eq
 * a un polyedre p, qui est modifie; eq est perdue;
 *
 * Si p->sc et eq sont "normalises", la creation d'une egalite par
 * la conjonction de -eq dans p->sc et de eq 
 */
void add_ineg_poly(p,eq)
Ppoly p;
Pcontrainte eq;
{
    Pcontrainte ineg1,ineg2;

    /* voyons si l'inegalite a ajouter ne forme pas une egalite
       avec une des inegalites deja presentes
       */
    for (ineg1=ineg2=poly_inegalites(p);
	 ineg1!=NULL;
	 ineg2=ineg1,ineg1=ineg1->succ) {
	if (contrainte_oppos(ineg1,eq)) {
	    /* si oui: une inegalite de moins (la soeur) et une egalite
	       de plus */
	    /* destruction de l'inegalite inutile */
	    if (ineg1==ineg2) poly_inegalites(p) = ineg1->succ;
	    else ineg2->succ = ineg1->succ;
	    CONTRAINTE_RM(ineg1,"add_ineg_poly");
	    poly_nbre_inegalites(p)--;
	    /* ajout de la nouvelle egalite, ex-inegalite eq, a p */
	    eq->succ = poly_inegalites(p);
	    poly_inegalites(p) = eq;
	    poly_nbre_egalites(p)++;
	    return;
	}
    }

    /* si non: une inegalite de plus */
    eq->succ = poly_inegalites(p);
    poly_inegalites(p) = eq;
    poly_nbre_inegalites(p)++;
    /* normalisation facultative */
    p->sc = sc_normalize(sc_dup(p->sc));
}

/* inter_demi: intersection d'un polyedre et d'un demi-espace
 * -> polyedre
 *
 * NB : cette fonction est destructrice
 *      l'equation est de plus integree dans le polyedre
 *
 * lors de l'intersection d'un polydre avec un demi-espace ou un hyperplan
 * il faut rajouter une contrainte au systeme de contraintes et calculer
 * un nouveau systeme generateur
 */
Ppoly inter_demi(p,eq)
Ppoly p;
Pcontrainte eq;
{
    if (poly_nbre_sommets(p)==0) return(p);
    add_ineg_poly(p,eq);
    p->sg = nsg_e_red(p->sg,eq);
    return(p);
}

/* inter_hyp: intersection d'un polyedre et d'un hyperplan
 * -> polyedre
 *
 * NB : cette fonction est destructrice
 *      l'equation est de plus integree dans le polyedre
 */
Ppoly inter_hyp(p,eq)
Ppoly p;
Pcontrainte eq;
{
    if (poly_nbre_sommets(p)==0) return(p);
    eq->succ = poly_inegalites(p);
    poly_inegalites(p) = eq;
    poly_nbre_egalites(p)++;
    p->sg = new_sg_h_red(p->sg,eq);
    return(p);
}

/**************************************************************************/
/*     OPERATEURS         D'INTERSECTIONS        POUR         L'ANALYSE   */
/**************************************************************************/

Ppoly av_demi_inter(p,eq,dim)	/* *p est modifie et eq integree dans *p */
Ppoly p;
Pcontrainte eq;
int dim;
{
    if (p!=NULL) {			/* polyedre non neutre */
	return(inter_demi(p,eq));
    } else {
	return(sc_to_poly(sc_make((Pcontrainte)NULL,eq)));
    }
}

Ppoly ar_demi_inter(p,eq)	/* *p est modifie et eq integree dans *p */
Ppoly p;
Pcontrainte eq;
{
    if (p!=NULL) {			/* polyedre non neutre */
	return(inter_demi(p,eq));
    } else {
	return(NULL);
    }
}

Ppoly av_hyp_inter(p,eq,dim)	/* *p est modifie et eq integree dans *p */
Ppoly p;
Pcontrainte eq;
int dim;
{
    if (p!=NULL) {			/* polyedre non neutre */
	return(inter_hyp(p,eq));
    } else {
	return(sc_to_poly(sc_make(eq,(Pcontrainte)NULL)));
    }
}

Ppoly ar_hyp_inter(p,eq)	/* *p est modifie et eq integree dans *p */
Ppoly p;
Pcontrainte eq;
{
    if (p!=NULL) {			/* polyedre non neutre */
	return(inter_hyp(p,eq));
    } else {
	return(NULL);
    }
}
