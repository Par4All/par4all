 /* intersection d'un polyedre avec un demi-espace
  *
  * Malik Imadache
  *
  * Modifie par Francois Irigoin, le 30 mars 1989:
  *  - reprise des includes
  *  - modification des appels a MALLOC (ajout d'un troisieme argument)
  *  - suppression d'une declaration de vect_mult (incompatible et inutile)
  *    et d'une declaration de ch_sgn (inutile)
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

/* creation de nouveaux sommets pour une intersection de polyedre avec un
 * demi-espace en eliminant les redondances.
 * 
 * le polyedre est deja obtenu par succession d'intersections de
 * demi-espaces.
 * 
 * Rappel : pour le calcul de systeme generateur par intersections
 * successives nous eliminons les redondances sinon on a une explosion du
 * nombre d'elements generateurs; dans le cas des operateurs de tests ou
 * une unique intersection est faite nous n'eliminons pas les redondances
 * dans l'operation d'intersection; cette remarque est aussi valable pour
 * l'intersection d'un polyedre avec un hyperplan.
 */
Stsg_soms new_s_demi(sat_s,sat_r,sat_d,nb_s,nb_r,nb_d,eq_vues,nb_vues,nb_eq)
Ss_sat_inter *sat_s;            /* ces 3 pointeurs definissent des cellules  */
Svect_sat_inter *sat_r,*sat_d;  /* avec un pointeur sur un element generateur*/
                               /* une valeur de saturation par rapport a la */
                               /* contrainte d'intersection et un pointeur  */
                               /* sur la cellule suivante                   */
                               /* Ils donnent une reference au SG du poly.  */
                               /* avec lequel l'intersection est effectuee. */
int nb_s,nb_r,nb_d,nb_vues,nb_eq;
                               /* ces entiers donnent les nombres d'elements */
                               /* generateur du polyedre avec lequel on fait */
                               /* l'intersection et le nombre d'inequations  */
                               /* avec lesquelles on a deja fait des         */
                               /* intersections successives et enfin le      */
                               /* nombre total d'intersections a faire       */
Pcontrainte eq_vues;
                               /* nous avons ici la liste des premieres      */
                               /* inegalites avec lesquelles des intersec-   */
                               /* tions ont ete realisees et qui definissent */
                               /* le polyedre avec lequel nous faisons       */
                               /* l'actuelle intersection                    */
{
    Psommet s;
    Stsg_soms result;	/* structure avec un nombre de sommets et leur liste */

    result = ns_demi_red(sat_s,sat_r,sat_d,nb_s,nb_r,nb_d);

    /* ajout les saturations des inegalites pour les nouveaux sommets */

    for (s=result.ssg; s!=NULL; s=s->succ)
	if ((s->eq_sat) == NULL) ad_s_eq_sat(s,eq_vues,nb_vues,nb_eq);

    /* tous les nouveaux sommets ont leurs saturations calculees,
     * il faut eliminer les redondances entre sommets
     */

    result.ssg = red_s_inter(result.ssg,&(result.nb_s),nb_vues);
    return(result);
}



/**** fonctions de construction de nouveaux rayons ****/

/* la structure de cette fonction est identique a la precedente         */

Stsg_vects new_r_demi(sat_r,sat_d,nb_r,nb_d,eq_vues,nb_vues,nb_eq)
Svect_sat_inter *sat_r,*sat_d;
int nb_r,nb_d,nb_vues,nb_eq;
Pcontrainte eq_vues;
{
    Pray_dte nr;			/* pointeur vers le rayon nouveau */
    Stsg_vects result;

    /* calcul de la liste des nouveaux rayons */

    result = nr_demi_red(sat_r, sat_d, nb_r, nb_d);

    /* calcul des saturations des nouveaux elements */

    for (nr=result.vsg; nr!=NULL; nr=nr->succ)
	if ((nr->eq_sat) == NULL) ad_r_eq_sat(nr,eq_vues,nb_vues,nb_eq);

    /* elimination de redondances */

    result.vsg = red_r_inter(result.vsg,&(result.nb_v),nb_vues);
    return(result);
}

/* calcul du systeme generateur de l'intersection d'un demi_espace
 * donne par son equation et d'un polyedre donne par son systeme
 * generateur
 *
 * ATTENTION: le systeme generateur donne est remplace par le systeme
 * resultant 
 *
 * Note sur les arguments:
 *  - eq_vues et nb_vues sont les inegalites deja 
 *    intersectees, elles servent a calculer des saturations
 *    pour eliminer des redondances
 *  - nb_eq est le nombre total d'inequations avec lesquelles
 *    on fait le polyedre.
 */
Ptsg new_sg_e(p_sg, eq, eq_vues, nb_vues, nb_eq) 
Ptsg p_sg;
Pcontrainte eq,eq_vues;
int nb_vues,nb_eq;
{
    Ss_sat_inter *sat_s; /* tableau de structures ( saturation,&sommet ) */
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
	MALLOC(nb_s * sizeof(Ss_sat_inter),SAT_TAB,"new_sg_e");
    sat_r = (Svect_sat_inter *) 
	MALLOC(nb_r * sizeof(Svect_sat_inter),SAT_TAB,"new_sg_e");
    sat_d = (Svect_sat_inter *) 
	MALLOC(nb_d * sizeof(Svect_sat_inter),SAT_TAB,"new_sg_e");
    
    /* initialisation de ces tableaux et calcul des saturations */
    /* voir satur.c a ce sujet                                  */

    d_sat_s_inter(sat_s,nb_s,(p_sg->soms_sg).ssg,eq,nb_vues);
    d_sat_vect_inter(sat_r,nb_r,(p_sg->rays_sg).vsg,eq,nb_vues);
    sat_vect_inter(sat_d,nb_d,(p_sg->dtes_sg).vsg,eq);
    
    /* construction des elements generateurs */
    /* mk_new_d se trouve dans inter_hyp.c   */
    
    new_soms = new_s_demi(sat_s,sat_r,sat_d,nb_s,nb_r,nb_d,eq_vues,
			  nb_vues,nb_eq);
    new_dtes = mk_new_d(sat_d,nb_d);
    new_rays = new_r_demi(sat_r,sat_d,nb_r,nb_d,eq_vues,nb_vues,nb_eq);
    
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
