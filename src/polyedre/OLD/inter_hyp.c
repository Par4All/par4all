 /* intersection d'un polyedre par un hyperplan
  *
  * Malik Imadache
  *
  * Les commentaires sont ecrits ici d'apres ceux de l'intersection d'un 
  * polyedre avec un demi-espace (demi-inter.c). La structure des      
  * fonctions y est la meme, on pourra donc y trouver plus de details  
  * pour l'implementation                                              
  *
  * Modifie par Francois Irigoin
  *  - reprise des includes
  *  - reformattage divers
  *  - ajout d'un troisieme argument aux MALLOCs
  *  - remplacement de TSOM par SOMMET
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
#define FREE(p,t,f) free((char *)(p))
 
/* new_s_hyp_red: construction des nouveaux sommets pour l'intersection
 * d'un hyperplan avec un polyedre
 */
Stsg_soms new_s_hyp_red(sat_s,sat_r,sat_d,nb_s,nb_r,nb_d)
Ss_sat_inter *sat_s;
Svect_sat_inter *sat_r,*sat_d;
int nb_s,nb_r,nb_d;
{
    int cpt_soms = 0;	/* compteur des sommets crees */
    Psommet list_new=0; /* pointeur de la tete de liste des nouveaux sommets */
    Psommet ns;		/* pointeur vers le sommet nouveau */
    int s1,s2;		/* reperes dans sat_s des sommets a considerer */
    int r,d;		/* reperes dans sat_r(d) de l'element a considerer */
    Stsg_soms result;
    for (s1=0;s1<nb_s;s1++) {

	/* tt sommet de saturation nulle pour l'hyperplan est element generateur   */
	/* de l'intersection                                                       */

	if (sat_s[s1].val_sat == 0) {
	    if (som_in_liste(sat_s[s1].som_sat,list_new))
		SOMMET_RM(sat_s[s1].som_sat,"new_s_hyp1");
	    else {
		sat_s[s1].som_sat->succ = list_new;
		list_new = sat_s[s1].som_sat;
		cpt_soms++;
	    }
	}
	else {
	    for (s2=s1+1;s2<nb_s;s2++) {

		/* tt couple de sommets de saturations opposees donne   par combinaison   */
		/* un sommet de l'intersection                                            */

		if (sat_s[s1].val_sat * sat_s[s2].val_sat < 0) {
		    ns = (Psommet) 
			MALLOC(sizeof(Ssommet),SOMMET,"new-s_hyp_red");
		    if (sat_s[s1].val_sat < 0) {
			new_ss(ns,sat_s[s1],sat_s[s2]);
		    } else 
			new_ss(ns,sat_s[s2],sat_s[s1]);
		    if (som_in_liste(ns,list_new))
			SOMMET_RM(ns,"new_s_hyp2");
		    else {
			ns->succ = list_new;
			list_new = ns;
			cpt_soms++;
		    }
		}
	    }
	    for (r=0;r<nb_r;r++) {

		/* tt couple de SxR dont le sommet et le rayon ont des saturations opposees */
		/* pour l'hyperplan donne par combinaison un sommet de l'intersection     */

		if (sat_s[s1].val_sat * sat_r[r].val_sat < 0) {
		    ns = (Psommet) 
			MALLOC(sizeof(Ssommet),SOMMET,"new_s_hyp_red");
		    new_s(ns,sat_s[s1],sat_r[r]);
		    if (som_in_liste(ns,list_new))
			SOMMET_RM(ns,"new_s_hyp3");
		    else {
			ns->succ = list_new;
			list_new = ns;
			cpt_soms++;
		    }
		}
	    }
	    for (d=0;d<nb_d;d++) {

		/* tt couple de SxD avec une droite de saturation non nulle pour l'hyperplan */
		/* donne un sommet de l'intersection                                         */

		if (sat_d[d].val_sat != 0) {
		    ns = (Psommet) 
			MALLOC(sizeof(Ssommet),SOMMET,"new-s_hyp_red");
		    new_s(ns,sat_s[s1],sat_d[d]);
		    if (som_in_liste(ns,list_new))
			SOMMET_RM(ns,"new_s_hyp4");
		    else {
			ns->succ = list_new;
			list_new = ns;
			cpt_soms++;
		    }
		}
	    }
	}
    }
    result.nb_s = cpt_soms;
    result.ssg = list_new;
    return(result);
}

/**** fonctions de construction de nouveaux rayons ****/

/* new_r_hyp_red:  structure identique a celle de la fonction precedente */
Stsg_vects new_r_hyp_red(sat_r,sat_d,nb_r,nb_d)
Svect_sat_inter *sat_r,*sat_d;
int nb_r,nb_d;
{
    int cpt_rays = 0;
    Pray_dte list_new=0;/* pointeur de la tete de liste des nouveaux rayons */
    Pray_dte nr;/* pointeur vers le rayon nouveau */
    int r1,r2;	/* reperes dans sat_r des rayons a considerer */
    int d;	/* reperes dans sat_r(d) de l'element a considerer */
    Stsg_vects result;

    for (r1=0;r1<nb_r;r1++) {

	/* rayons de saturation nulle a retenir */

	if (sat_r[r1].val_sat == 0) {
	    if (rd_in_liste(sat_r[r1].vect_sat,list_new))
		RAY_DTE_RM(sat_r[r1].vect_sat,"new_r_hyp1");
	    else {
		sat_r[r1].vect_sat->succ = list_new;
		list_new = sat_r[r1].vect_sat;
		cpt_rays++;
	    }
	}
	else {
	    for (r2=r1+1;r2<nb_r;r2++) {

		/* tt couple de rayons de saturations opposees donne un rayon de          */
		/* l'intersection                                                         */

		if (sat_r[r1].val_sat * sat_r[r2].val_sat < 0) {
		    /*
		     * nr = (Pray_dte) 
		     *	MALLOC(sizeof(Sray_dte),RAY_DTE,"new_r_hyp_red");
		     */
		    nr = ray_dte_new();
		    if (sat_r[r1].val_sat < 0) {
			new_rr(nr,sat_r[r2],sat_r[r1]);
		    } else
			new_rr(nr,sat_r[r1],sat_r[r2]);
		    if (rd_in_liste(nr,list_new))
			RAY_DTE_RM(nr,"new_r_hyp2");
		    else {
			nr->succ = list_new;
			list_new = nr;
			cpt_rays++;
		    }
		}
	    }

	    /* tt couple de RxD avec une droite de saturation non nulle donne un rayon */

	    for (d=0;d<nb_d;d++) {
		if (sat_d[d].val_sat != 0) {
		    /* Modification : Alexis Platonoff, 6 juillet 1990
		     *   (cf. 20 lignes plus haut)
		     * Il faut completement initialiser la variable!!
		     * Sinon, assert() dans new_err() provoque une erreur.
		     * Avant cette modification, il y avait :
		     *     nr = (Pray_dte) 
		     *	    MALLOC(sizeof(Sray_dte),RAY_DTE,"new_r_hyp_red");
		     */
		    nr = ray_dte_new();

		    if (sat_d[d].val_sat > 0) {
			new_rr(nr,sat_d[d],sat_r[r1]);
		    } else
			new_rr(nr,sat_r[r1],sat_d[d]);
		    if (rd_in_liste(nr,list_new))
			RAY_DTE_RM(nr,"new_r_hyp3");
		    else {
			nr->succ = list_new;
			list_new = nr;
			cpt_rays++;
		    }
		}
	    }
	}
    }
    result.nb_v = cpt_rays;
    result.vsg = list_new;
    return(result);
}

/**** fonctions de construction de nouvelles droites ****/

/* nk_new_d: structure identique a celle des deux fonctions precedentes
 *
 * cette fonction sert aussi dans inter-demi.c
 */
Stsg_vects mk_new_d(sat_d,nb_d)
Svect_sat_inter *sat_d;
int nb_d;
{
    int cpt_dtes = 0;
    Pray_dte list_new = 0;/* pointeur de la tete de liste des nouveaux droites */
    Pray_dte nd;/* pointeur vers le droite nouveau */
    int d1,d2;	/* reperes dans sat_r des droites a considerer */
    Stsg_vects result;

    for (d1=0;d1<nb_d;d1++) {

	/* tte droite de saturation nulle est une droite de l'intersection */

	if (sat_d[d1].val_sat == 0) {
	    if (rd_in_liste(sat_d[d1].vect_sat,list_new))
		RAY_DTE_RM(sat_d[d1].vect_sat,"new_d1");
	    else {
		sat_d[d1].vect_sat->succ = list_new;
		list_new = sat_d[d1].vect_sat;
		cpt_dtes++;
	    }
	}
	else {
	    for (d2=d1+1;d2<nb_d;d2++) {

		/* tt couple de droites de saturations non nulles donnent une droite */
		/* de l'intersection                                                 */

		if (sat_d[d2].val_sat !=0) {
		    nd = ray_dte_new();
		    /*
		     * nd = (Pray_dte) MALLOC(sizeof(Sray_dte),RAY_DTE,"mk_new_d");
		     */
		    new_rr(nd,sat_d[d2],sat_d[d1]);
		    if (rd_in_liste(nd,list_new)||(nd->vecteur==NULL))
			RAY_DTE_RM(nd,"new_d2");
		    else {
			nd->succ = list_new;
			list_new = nd;
			cpt_dtes++;
		    }
		}
	    }
	}
    }
    result.nb_v = cpt_dtes;
    result.vsg = list_new;
    return(result);
}

/*************************************************************************/
/*			SYSTEME GENERATEUR DE L'INTERSECTION		 */
/*			D'UN    HYPERPLAN   DONNE   PAR  SON		 */
/*			EQUATION  ET  D'UN  POLYEDRE   DONNE		 */
/*			PAR    SON    SYSTEME     GENERATEUR		 */
/*************************************************************************/

/* new_sg_h_red:  version redondante servant pour l'operateur
 * des noeuds de test 
 *
 * ATTENTION: le systeme generateur donne est remplace par le systeme
 * resultant 
 */
Ptsg new_sg_h_red(p_sg,eq) 
Ptsg p_sg;
Pcontrainte eq;
{
    Ss_sat_inter *sat_s;/* tableau de structures ( saturation,&sommet ) */

    Svect_sat_inter *sat_r,*sat_d;
    /* tableaux de structures ( saturation, &vecteur ) */

    int nb_s = sg_nbre_sommets(p_sg);
    int nb_r = sg_nbre_rayons(p_sg);
    int nb_d = sg_nbre_droites(p_sg);	/* nombre d'elements generateurs */

    Stsg_soms new_soms;
    Stsg_vects new_rays,new_dtes;	/* structures 
					   (nb_elems_construits,listes_new_elems) */

    /* allocation memoire pour ces tableaux */

    sat_s = (Ss_sat_inter *) 
	MALLOC(nb_s * sizeof(Ss_sat_inter),SAT_TAB,"new_sg_h_red");
    sat_r = (Svect_sat_inter *) 
	MALLOC(nb_r * sizeof(Svect_sat_inter),SAT_TAB,"new_sg_h_red");
    sat_d = (Svect_sat_inter *) 
	MALLOC(nb_d * sizeof(Svect_sat_inter),SAT_TAB,"new_sg_h-red");

    /* initialisation de ces tableaux et calcul des saturations */

    sat_s_inter(sat_s,nb_s,(p_sg->soms_sg).ssg,eq);
    sat_vect_inter(sat_r,nb_r,(p_sg->rays_sg).vsg,eq);
    sat_vect_inter(sat_d,nb_d,(p_sg->dtes_sg).vsg,eq);

    /* construction des elements generateurs */

    new_soms = new_s_hyp_red(sat_s,sat_r,sat_d,nb_s,nb_r,nb_d);
    new_rays = new_r_hyp_red(sat_r,sat_d,nb_r,nb_d);
    new_dtes = mk_new_d(sat_d,nb_d);

    /* elimination des anciens elements generateurs inutils */
    /* i.e.   ceux    de     saturation     non       nulle */

    free_inter(sat_s,sat_r,sat_d,nb_s,nb_r,nb_d);
    if (sat_s!=NULL) FREE((char *)sat_s,SAT_TAB,"new_sg_h");
    if (sat_r!=NULL) FREE((char *)sat_r,SAT_TAB,"new_sg_h");
    if (sat_d!=NULL) FREE((char *)sat_d,SAT_TAB,"new_sg_h");

    /* on raccroche les listes resultantes a la structure donnee */

    p_sg->soms_sg = new_soms;
    p_sg->rays_sg = new_rays;
    p_sg->dtes_sg = new_dtes;

    return(p_sg);
}
