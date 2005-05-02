 /* Manipulation des structures de donnees contenant les saturations
  * lors du calcul de l'intersection de deux polyedres
  *
  * Malik Imadache, Francois Irigoin
  */

#include <stdio.h>

#include "assert.h"
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

#define MALLOC(s,t,f) malloc(s)

char * malloc();

/* void new_ss(Psommet ns, Ss_sat_inter sat_s1, Ss_sat_inter sat_s2):
 * construction de sommet pour inter-demi-espace par composition de 
 * sommets de saturations opposees
 */
void new_ss(ns, sat_s1, sat_s2)
Psommet ns;
Ss_sat_inter sat_s1,sat_s2;
{
    int d;
    if ((d = ((sat_s2.som_sat)->denominateur) * (sat_s1.sv) 
	 - ((sat_s1.som_sat)->denominateur) * (sat_s2.sv))<0) {
	ns->denominateur = -d;
 	ns->vecteur = vect_cl2(-(sat_s1.val_sat),(sat_s2.som_sat)->vecteur,
			       (sat_s2.val_sat), (sat_s1.som_sat)->vecteur);
	sommet_normalize(ns);
    }
    else {
	ns->denominateur = d;
 	ns->vecteur = vect_cl2(sat_s1.val_sat,(sat_s2.som_sat)->vecteur,
			       -sat_s2.val_sat,(sat_s1.som_sat)->vecteur);
	sommet_normalize(ns);
    }
    ns->eq_sat = NULL;
}

/* void new_s(Psommet ns, Ss_sat_inter sat_so, Svect_sat_inter sat_v):
 * construction de vecteur pour inter-demi-espace (ou hyperplan) 
 * par composition d'un sommet et d'un rayon ou d'une droite
 */
void new_s(ns, sat_so, sat_v)
Psommet ns;
Ss_sat_inter sat_so;
Svect_sat_inter sat_v;
{
    int d;

    d = ((sat_so.som_sat)->denominateur) * (sat_v.val_sat) ;
    if (d<0) {
	ns->denominateur = -d;
	ns->vecteur = vect_cl2(-(sat_v.val_sat),(sat_so.som_sat)->vecteur,
			       (sat_so.val_sat),(sat_v.vect_sat)->vecteur);
	sommet_normalize(ns);
    } 
    else
    {
	ns->denominateur = d;
	ns->vecteur = vect_cl2((sat_v.val_sat),(sat_so.som_sat)->vecteur,
			       -(sat_so.val_sat),(sat_v.vect_sat)->vecteur);
	sommet_normalize(ns);
    }
    ns->eq_sat = NULL;
}

/* void new_rr(Pray_dte nr, Svect_sat_inter sat_r1, Svect_sat_inter sat_r2):
 * construction de rayon pour inter-demi-espace par composition de 
 * rayons de saturations opposees
 */
void new_rr(nr,sat_r1,sat_r2)
Pray_dte nr;
Svect_sat_inter sat_r1,sat_r2;	
{
    assert(nr->vecteur==NULL);

    nr->vecteur = 
	vect_cl2(sat_r1.val_sat,(sat_r2.vect_sat)->vecteur,
		 -sat_r2.val_sat,(sat_r1.vect_sat)->vecteur);
    ray_dte_normalize(nr);
    nr->eq_sat = NULL;
}
