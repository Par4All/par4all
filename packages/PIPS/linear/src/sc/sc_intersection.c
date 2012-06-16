/*

  $Id$

  Copyright 1989-2012 MINES ParisTech

  This file is part of Linear/C3 Library.

  Linear/C3 Library is free software: you can redistribute it and/or modify it
  under the terms of the GNU Lesser General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  Linear/C3 Library is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU Lesser General Public License for more details.

  You should have received a copy of the GNU Lesser General Public License
  along with Linear/C3 Library.  If not, see <http://www.gnu.org/licenses/>.

*/

/* package sc */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "assert.h"
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Psysteme sc_fusion(Psysteme s1, Psysteme s2): fusion de deux systemes
 * de contraintes afin d'obtenir un systeme contenant les egalites
 * et les inegalites des deux systemes donnes; aucune elimination de
 * redondance n'est effectuee.
 *
 * attention: les deux systemes donnes, s1 et s2, sont detruits.
 *
 * Note: sc_fusion() est considere comme obsolete; voir sc_intersection()
 * et sc_append().
 *
 * The base is not updated.
 */
Psysteme sc_fusion(Psysteme s1, Psysteme s2)
{
    Pcontrainte eq;

    if (SC_UNDEFINED_P(s1))
      return s2;

    if (sc_rn_p(s1)) {
      free(s1);
      return s2;
    }

    if (SC_UNDEFINED_P(s2))
      return s1;

    if (sc_rn_p(s2)) {
      free(s2);
      return s1;
    }

    if (s1->nb_eq != 0)
    {
	for (eq = s1->egalites;
	     eq->succ != (Pcontrainte) NULL;
	     eq = eq->succ)
	    ;
	eq->succ = s2->egalites,
	s1->nb_eq += s2->nb_eq;
    }
    else
	s1->egalites = s2->egalites,
	s1->nb_eq = s2->nb_eq;

    if (s1->nb_ineq != 0)
    {
	for (eq = s1->inegalites;
	     eq->succ != (Pcontrainte)NULL;
	     eq = eq->succ)
	    ;
	eq->succ = s2->inegalites,
	s1->nb_ineq += s2->nb_ineq;
    } 
    else 
	s1->inegalites = s2->inegalites,
	s1->nb_ineq = s2->nb_ineq;

    s2->inegalites = NULL;
    s2->egalites = NULL;
    sc_rm(s2);

    return(s1);
}

/* Psysteme sc_intersection(Psysteme s1, Psysteme s2, Psysteme s3):
 * calcul d'un systeme de contraintes s1 representant l'intersection
 * des polyedres convexes definis par les systemes de contraintes s2 et s3.
 *
 * s1 := intersection(s2,s3);
 * return s1;
 *
 * Les listes d'egalites et d'inegalites de s2 et de s3 sont simplement
 * concatenees. Aucun test de redondance n'est effectuee.
 * Cependant, les valeurs particulieres SC_RN (element neutre) et SC_EMPTY
 * (element absorbant) sont traitees de maniere a donner des resultats
 * corrects.
 * 
 * Attention: le systeme s1 est d'abord vide de ses egalites et inegalites
 * s'il est different de s2 et de s3; sinon, les modifications sont faites
 * en place, sur s2 ou sur s3.
 */
Psysteme sc_intersection(Psysteme s1, Psysteme s2, Psysteme s3)
{
    if(s1==s2) {
	s1 = sc_append(s2,s3);
    }
    else if(s1==s3) {
	s1 = sc_append(s3,s2);
    }
    else {
	if(!SC_EMPTY_P(s1)) {
	    /* on pourrait se contenter de desallouer les deux listes
	       d'egalites et d'inegalites a condition d'avoir un sc_copy() */
	    sc_rm(s1);
	}
	s1 = sc_copy(s2);
	s1 = sc_append(s1,s3);
    }
    return(s1);
}

/* Psysteme sc_append(Psysteme s1, Psysteme s2): calcul de l'intersection
 * des polyedres definis par s1 et par s2 sous forme d'un systeme de
 * contraintes et renvoi de s1
 *
 * s1 := intersection(s1,s2)
 * return s1;
 *
 * Attention, SC_RN est un element neutre et SC_EMPTY un element absorbant.
 * SC_UNDEFINED devrait aussi etre teste proprement.
 *
 * Modifications:
 *  - mise a jour de la base de s1 (FI, 2/1/90)
 */
Psysteme sc_append(s1,s2)
Psysteme s1;
Psysteme s2;
{
    Pcontrainte c;

    if(SC_RN_P(s1))
	/* ne rien faire et renvoyer s2 */
	s1 = sc_copy(s2);
    else if(SC_RN_P(s2))
	/* ne rien faire et renvoyer s1 */
	;
    else if(SC_EMPTY_P(s1)) {
	/* ne rien faire et renvoyer s1 */
	/* on ne devrait jamais passer la dans le futur proche car
	   SC_EMPTY==SC_RN */
	assert(false);
	;
    }
    else if(SC_EMPTY_P(s2)) {
	/* desallouer s1 et renvoyer SC_EMPTY */
	assert(false);
	sc_rm(s1);
	s1 = SC_EMPTY;
    }
    else {

	/* ni s1 ni s2 ne sont des systemes particuliers */
	for(c = sc_egalites(s2); c != (Pcontrainte) NULL; c = c->succ) {
	    sc_add_egalite(s1,contrainte_copy(c));
	}
	for(c = sc_inegalites(s2); c != (Pcontrainte) NULL; c = c->succ) {
	    sc_add_inegalite(s1,contrainte_copy(c));
	}

	/* update s1 basis with s2's vectors */
	base_append(&s1->base, s2->base);
	s1->dimension = vect_size(s1->base);
    }

    return s1;
}


/* Psysteme sc_safe_intersection(Psysteme s1, Psysteme s2, Psysteme s3)
 * input    : 
 * output   : calcul d'un systeme de contraintes s1 representant l'intersection
 *            des polyedres convexes definis par les systemes de contraintes s2 et s3.
 *        
 *            s1 := intersection(s2,s3);
 *            return s1;
 *
 * modifies : s1 is modified if it is not undefined.
 * comment  :	
 * calcul d'un systeme de contraintes s1 representant l'intersection
 * des polyedres convexes definis par les systemes de contraintes s2 et s3.
 *
 * Les listes d'egalites et d'inegalites de s2 et de s3 sont simplement
 * concatenees. Aucun test de redondance n'est effectuee.
 * Cependant, les valeurs particulieres sc_rn (element neutre) et sc_empty
 * (element absorbant) sont traitees de maniere a donner des resultats
 * corrects ( different de sc_intercestion).
 * 
 * Attention: le systeme s1 est d'abord vide de ses egalites et inegalites
 * s'il est different de s2 et de s3; sinon, les modifications sont faites
 * en place, sur s2 ou sur s3.
 */
Psysteme sc_safe_intersection(s1,s2,s3)
Psysteme s1;
Psysteme s2;
Psysteme s3;
{
    if(s1==s2) {
	s1 = sc_safe_append(s2,s3);
    }
    else if(s1==s3) {
	s1 = sc_safe_append(s3,s2);
    }
    else {
	if(!SC_UNDEFINED_P(s1)) {
	    /* on pourrait se contenter de desallouer les deux listes
	       d'egalites et d'inegalites a condition d'avoir un sc_copy() */
	    sc_rm(s1);
	}
	s1 = sc_copy(s2);
	s1 = sc_safe_append(s1,s3);
    }
    return(s1);
}

/* Psysteme sc_safe_append(Psysteme s1, Psysteme s2)
 * input    :
 * output   : calcul de l'intersection des polyedres definis par s1 et
 *            par s2 sous forme d'un systeme de contraintes et renvoi de s1.
 *
 *            s1 := intersection(s1,s2)
 *            return s1;
 * modifies : s1.
 * comment  : sc_rn et sc_empty sont traite's de manie`re particulie`re pour tenir
 *            compte de leur se'mantique.
 *
 */
Psysteme sc_safe_append(s1,s2)
Psysteme s1;
Psysteme s2;
{
    Pcontrainte c;
    Pbase b;
    Pvecteur coord;

    assert(!SC_UNDEFINED_P(s1) && !SC_UNDEFINED_P(s2));

    if(sc_rn_p(s1)) {
	/* ne rien faire et renvoyer s2 apre`s mise a` jour de la base */
	sc_rm(s1);
	s1 = sc_copy(s2);
    }
    else if(sc_rn_p(s2))
	/* ne rien faire et renvoyer s1 apre`s mise a` jour de la base */
	;
    else if(sc_empty_p(s1))
	/* ne rien faire et renvoyer s1 apre`s mise a` jour de la base */
	;
    else if(sc_empty_p(s2)) {
	/* ne rien faire et renvoyer s2 apre`s mise a` jour de la base */
	sc_rm(s1);
	s1 = sc_copy(s2);
    }
    else {
	/* ni s1 ni s2 ne sont des systemes particuliers :
	 * on ajoute a` s1 les e'galite's et ine'galite's de s2 */
	for(c = sc_egalites(s2); c != (Pcontrainte) NULL; c = c->succ) {
	    sc_add_egalite(s1,contrainte_copy(c));
	}
	for(c = sc_inegalites(s2); c != (Pcontrainte) NULL; c = c->succ) {
	    sc_add_inegalite(s1,contrainte_copy(c));
	}
    }

    /* update s1 basis with s2's vectors */
    b = s1->base;
    for(coord = s2->base; !VECTEUR_NUL_P(coord); coord = coord->succ) {
	Variable v = vecteur_var(coord);
	b = vect_add_variable(b, v);
    }
    s1->base = b;
    s1->dimension = vect_size(b);

    return(s1);
}


/* bool sc_intersection_empty_p_ofl(ps1, ps2) 
 * input    : two polyhedra
 * output   : true if their intersection is empty, false otherwise.
 * modifies : nothing
 * comment  : calls sc_faisabilite_ofl in order to trap overflow errors
 *            BC, 1994.	
 */
bool sc_intersection_empty_p_ofl(ps1, ps2)
Psysteme ps1, ps2;
{
    Psysteme ps = SC_UNDEFINED;
    bool result;
    ps1 = sc_copy(ps1);
    ps2 = sc_copy(ps2);
    ps = sc_safe_intersection(ps,ps1,ps2);
    result = !(sc_faisabilite_ofl(ps));
    sc_rm(ps1);
    sc_rm(ps2);
    sc_rm(ps);
    ps1 = ps2 = ps = NULL;
    return result;
}

/* returns the common subsystem if appropriate...
 * s1 and s2 are modified accordingly.
 */
Psysteme extract_common_syst(Psysteme s1, Psysteme s2)
{
  Pcontrainte 
    e = extract_common_constraints(&s1->egalites, &s2->egalites, true),
    i = extract_common_constraints(&s1->inegalites, &s2->inegalites, false);

  sc_fix(s1);
  sc_fix(s2);

  return sc_make(e, i);
}
