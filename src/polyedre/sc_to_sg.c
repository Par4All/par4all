 /* package sur les polyedres
  *
  * Francois Irigoin
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

int sc_to_sg_debug_level = 0;
#define ifdebug(n) if((n)<sc_to_sg_debug_level)

/* Ptsg sc_to_sg(Psysteme syst, int dim): calcul d'un systeme generateur
 * d'un polyedre rationnel a partir d'un systeme de contraintes lineaires
 * par intersections successives;
 *
 * Si syst==NULL, il s'agit de l'espace tout entier et un sg avec dim
 * droites est renvoye.
 *
 * Une elimination de redondance est effectuee sur l'argument syst
 * aussi bien que sur le systeme generateur renvoye
 *
 * Si syst est non faisable, je ne sais pas ce qui arrive (FI).
 *
 * Note: cette fonction est rangee dans la bibliotheque polyedre pour ne
 * pas entrainer une dependance inutile entre sg et sc.
 *
 * Ancien nom: mk_sg()
 *
 * Modifications:
 *  - suppression du 2eme argument qui en sert plus a cause de la base
 *    (FI, 18/12/89)
 *  - ajout d'une elimination de redondance suite a l'apparition de 185
 *    sommets la ou j'en attendais 8 (FI, 19/09/90); d'ou un effet de bord
 *    a priori inutile sur l'argument
 */                     
Ptsg sc_to_sg(syst)
Psysteme syst;
{
    Spoly p;
    Ptsg sg;
    Pcontrainte eq;
    int nb_vues;
    int dim = syst->dimension;

    ifdebug(8) {
	(void) fprintf(stderr,"sc_to_sg: begin\n");
	(void) fprintf(stderr,"sc_to_sg: dim = %d, systeme de contrainte =\n",
		       dim);
	sc_fprint(stderr,syst,variable_dump_name);
    }

    /* le point de depart des intersections successives est Rn     */

    sg = mk_rn(syst->base);

    ifdebug(8) {
	(void) fprintf(stderr,"sc_to_sg: le sg initial est Rn\n");
	sg_fprint(stderr, sg, variable_dump_name);
    }

    /* nous verifions ici que le polyedre que l'on veut completer n'est pas
     * l'element initial de resolution
     */

    if (syst!=NULL) {

	/* nous faisons ici de l'elimination de redondances a 
	 * chaque intersection
	 * avec des demi-espaces; ainsi nous commencons par calculer les
	 * saturations de notre unique sommet initial de Rn pour ensuite si on
	 * le conserve dans l'intersection comparer ses saturations avec celles
	 * des autres sommets de l'intersection. 
	 * A chaque etape, tous les sommets
	 * et rayons auront de meme leurs saturations entretenues pour les
	 * etapes suivantes
	 */

	ad_s_eq_sat((sg->soms_sg).ssg,syst->inegalites,0,
		    sc_nbre_inegalites(syst));

	/* intersections successives avec les inegalites */
	ifdebug(8) 
	    (void) fprintf(stderr,"sc_to_sg: traitement des inegalites\n");
	for (eq=syst->inegalites,nb_vues=0;eq != NULL;eq=eq->succ,nb_vues++) {
	    sg = new_sg_e(sg,eq,syst->inegalites,nb_vues,
			  sc_nbre_inegalites(syst));
/*	    (void) fprintf(stderr,"sc_to_sg: %d inegalite\n",nb_vues); */
/*            sg_fprint(stderr, sg, variable_dump_name); */
	}

	ifdebug(8) {
	    (void) fprintf(stderr,"sc_to_sg: apres les inegalites\n");
	    sg_fprint(stderr, sg, variable_dump_name);
	}

	/* continuons avec les egalites */
	ifdebug(8) (void) fprintf(stderr,
				  "sc_to_sg: traitement des egalites\n");
	for (eq=syst->egalites,nb_vues=0;eq != NULL;eq=eq->succ,nb_vues++) {
	    sg = new_sg_h_red(sg,eq);
/*	    (void) fprintf(stderr,"sc_to_sg: %d egalite\n",nb_vues); */
/*            sg_fprint(stderr, sg, variable_dump_name); */
	}
    }

    ifdebug(8) {
	(void) fprintf(stderr,"sc_to_sg: apres les egalites\n");
	sg_fprint(stderr, sg, variable_dump_name);
	(void) fprintf(stderr,"sc_to_sg: end\n");
    }

    /* redundancy elimination */
    p.sc = syst;
    p.sg = sg;
    elim_red(&p);

    return(sg);
}
