 /* package sc */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Psysteme sc_oppose(Psysteme ps):
 * calcul pour un systeme de contraintes sans egalites du systeme de
 * contraintes dont les inegalites sont les negations des inegalites
 * originelles; attention, cela ne calcule pas le polyedre complementaire!
 *
 * Un systeme non trivial risque fort d'etre transforme en systeme non
 * faisable.
 *
 * pour chaque inegalite donnee AX <= B, on construit une inegalites "opposee"
 * AX > B ie -AX <= -B
 */
Psysteme sc_oppose(ps)
Psysteme ps;
{
    Pcontrainte eq;

    if (ps->nb_eq != 0) {
	(void) fprintf(stderr,"sc_oppose: systeme contenant des egalites\n");
	abort();
    }

    for (eq = ps->inegalites; eq != (Pcontrainte )NULL; eq = eq->succ)
	(void) vect_multiply(eq->vecteur, VALUE_MONE);

    return(ps);
}
