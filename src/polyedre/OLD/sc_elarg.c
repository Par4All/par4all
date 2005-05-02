 /* package polyedre: elargissement de deux systemes lineaires
  *
  * Ce module est range dans le package polyedre bien qu'il soit utilisable
  * en n'utilisant que des systemes lineaires (package sc) parce qu'il
  * utilise lui-meme des routines sur les polyedres.
  *
  * Francois Irigoin, Janvier 1990
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

/* Psysteme sc_elarg(s1, s2): calcul d'une representation par systeme
 * lineaire de l'elarg convexe des polyedres definis par les systemes
 * lineaires s1 et s2
 *
 * s = elargissement(s1, s2);
 * return s;
 *
 * s1 et s2 ne sont pas modifies. Ils doivent tous les deux avoir au moins
 * une base.
 *
 * Il faudrait traiter proprement les cas particuliers SC_RN et SC_EMPTY
 */
Psysteme sc_elarg(s1, s2)
Psysteme s1;
Psysteme s2;
{
    Pbase b;
    Pvecteur coord;
    Ppoly p1;
    Ppoly p2;
    Ppoly p;
    Psysteme s;

    assert(!SC_UNDEFINED_P(s1) && !SC_UNDEFINED_P(s2));

    /* duplication de s1 et de s2 */
    s1 = sc_dup(s1);
    s2 = sc_dup(s2);

    /* calcul d'une base unique pour s1 et s2 */
    b = s1->base;
    for(coord=s2->base; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	b = vect_add_variable(b, vecteur_var(coord));
    }
    vect_rm(s2->base);
    s2->base = vect_dup(b);

    if(SC_RN_P(s1)) {
	s = s1;
	sc_rm(s2);
    }
    else if(SC_RN_P(s2)) {
	s = s2;
	sc_rm(s1);
    }
    else if(SC_EMPTY_P(s1)) {
	assert(FALSE);
	s = s2;
	sc_rm(s1);
    }
    else if(SC_EMPTY_P(s2)) {
	assert(FALSE);
	s = s1;
	sc_rm(s2);
    }
    else {
	/* cas general */
	/* conversion en polyedres */
	p1 = sc_to_poly(s1);
	p2 = sc_to_poly(s2);

	/* calcul de l'elargissement: p2 est renvoye modifie dans p,
	   p1 est cense ne pas etre touche */
	p = elarg(p1, p2);
	poly_rm(p1);

	/* recuperation du systeme lineaire et desallocation du polyedre */
	s = p->sc;
	p->sc = SC_UNDEFINED;
	poly_rm(p);
    }

    return s;
}
