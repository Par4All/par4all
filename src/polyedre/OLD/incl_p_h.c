/* test d'inclusion d'un polyedre dans un hyperplan definie par une egalite
 *
 * Malik Imadache
 *
 * Modifie par Francois Irigoin:
 *  - reprise des includes
 */

#include <stdio.h>

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

/* boolean inclus_poly_hyp(Ppoly p, Pcontrainte eg): test d'inclusion
 * d'un polyedre dans un hyperplan donne par une egalite
 */
boolean inclus_poly_hyp(p,eg)
Ppoly p;
Pcontrainte eg;
{
    Psommet s;
    Pray_dte rd;
    if (p==NULL) return(TRUE);

    /* si le polyedre est l'element initial de lapropagation => oui */
    /* sinon il faut que tous les elements generateurs du polyedre  */
    /* soient contenus dans l'hyperplan                             */

    else {
	for ( s=poly_sommets(p); s!=NULL ; s=s->succ) {
	    if (0 != satur_som(s,eg)) return(FALSE);
	}
	for ( rd=poly_rayons(p); rd!=NULL ; rd=rd->succ) {
	    if (0 != satur_vect(rd->vecteur,eg)) return(FALSE);
	}
	for ( rd=poly_droites(p); rd!=NULL ; rd=rd->succ) {
	    if (0 != satur_vect(rd->vecteur,eg)) return(FALSE);
	}
	return(TRUE);
    }
}
