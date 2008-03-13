 /* package contrainte - operations sur les listes de contraintes
  */

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"


/* boolean contrainte_in_liste(Pcontrainte c, Pcontrainte lc): test de
 * l'appartenance d'une contrainte c a une liste de contrainte lc
 *
 * Les contrainte sont supposees normalisees (coefficients reduits par leur
 * PGCD, y compris les termes constants).
 *
 * On considere que la contrainte nulle, qui ne represente aucune contrainte
 * (elle est toujours verifiee) appartient a toutes les listes de contraintes.
 *
 */
boolean contrainte_in_liste(c,lc)
Pcontrainte c;
Pcontrainte lc;
{
    Pcontrainte c1;

    assert(!CONTRAINTE_UNDEFINED_P(c));

    if (CONTRAINTE_NULLE_P(c))
	return(TRUE);

    for (c1=lc; !CONTRAINTE_UNDEFINED_P(c1); c1=c1->succ) {
	if (vect_equal((c1->vecteur),(c->vecteur))) {
	    return(TRUE);
	}
    }
    return(FALSE);
}

/* boolean egalite_in_liste(Pcontrainte eg, Pcontrainte leg): test si une
 * egalite appartient a une liste d'egalites
 *
 * Une egalite peut avoir ete multipliee par moins 1 mais ses coefficients,
 * comme les coefficients des egalites de la liste, doivent avoir ete
 * reduits par leur PGCD auparavant
 *
 * Ancien nom: vect_in_liste1()
 */
boolean egalite_in_liste(v,listev)
Pcontrainte v;
Pcontrainte listev;
{
    Pcontrainte v1;

    if (v->vecteur == NULL) return(TRUE);
    for (v1=listev;v1!=NULL;v1=v1->succ) {
	if (vect_equal((v1->vecteur),(v->vecteur)) ||
	    vect_oppos((v1->vecteur),(v->vecteur))) {
	    return(TRUE);
	}
    }
    return(FALSE);
}

/* int nb_elems_list(Pcontrainte list): nombre de contraintes se trouvant
 * dans une liste de contraintes
 *
 * Ancien nom: nb_elems_eq()
 */
int nb_elems_list(list)
Pcontrainte list;
{
	int i;

	for(i=0;list!=NULL;i++,list=list->succ)
	    ;

	return(i);
}
