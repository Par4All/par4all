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

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Psysteme new_loop_bound(Psysteme scn, Pbase base_index) 
 * computation of the new iteration space (given the loop bounds) 
 * in the new basis G
 *
 * scn is destroyed
 *
 * if the list of indexes is empty, the system is assumed to
 * be a list of constraints that have to be verified for the set
 * of solutions not to be empty, so scn is returned.
 *
 * Reference about the algorithm PPoPP'91
 */
Psysteme new_loop_bound
  (Psysteme scn,
   Pbase base_index)
{
    Psysteme
	result = NULL,
	condition = NULL,
	enumeration = NULL;

    algorithm_row_echelon_generic(scn, base_index, 
				  &condition, &enumeration, false);

    sc_rm(scn);
    scn = NULL;
    result = sc_fusion(condition, enumeration);

    return result;
}

/* Psysteme get_other_constraints(psyst, vars)
 * Psysteme *psyst;
 * Pbase vars;
 *
 *         IN: vars
 *     IN/OUT: psyst
 *        OUT: returned system
 * 
 * returns the constraints that do not contain any of the variables listed
 * in Pbase vars, which are removed from the original Psysteme *psyst.
 * if the original system is undefined, the same thing is returned.
 *
 * (c) FC 16/05/94
 */
Psysteme get_other_constraints(psyst, vars)
Psysteme *psyst;
Pbase vars;
{
    Pcontrainte
	egothers = (Pcontrainte) NULL,
	inothers = (Pcontrainte) NULL,
	egsyst = (Pcontrainte) NULL,
	insyst = (Pcontrainte) NULL;

    assert(!SC_UNDEFINED_P(*psyst));

    if (sc_empty_p(*psyst))
	return(sc_empty(base_difference(sc_base(*psyst), vars)));
    /* else 
     */
    Pcontrainte_separate_on_vars(sc_egalites(*psyst), 
				 vars, &egsyst, &egothers);
    Pcontrainte_separate_on_vars(sc_inegalites(*psyst), 
				 vars, &insyst, &inothers);
	
    /* result in built and syst is modified.
     */
    *psyst = (sc_rm(*psyst), sc_make(egsyst, insyst));
    return(sc_make(egothers, inothers));
}

/*----------------------------------------------------------
 *
 * ALGORITHM ROW ECHELON from Ancourt and Irigoin, PPoPP'91
 *
 * The algorithm is slightly different: 
 * 
 * conditions are taken out of the system built.
 *
 * void algorithm_row_echelon(syst, scans, pcondition, penumeration)
 * Psysteme syst;
 * Pbase scans;
 * Psysteme *pcondition, *penumeration;
 *
 *     IN: syst, scans
 *    OUT: pcondition, penumeration
 *
 * (c) FC 16/05/94
 */

/* each variable should be at least within one <= and one >=;
 * scn IS NOT modified.
 */
void 
algorithm_row_echelon_generic(
    Psysteme scn,           /* initial system, which is not touched. */
    Pbase base_index,       /* enumeration variables, from outer to inner. */
    Psysteme *pcondition,   /* returned condition (what remains from scn). */
    Psysteme *penumeration, /* returned enumeration system. */
    bool redundancy      /* whether to allow outwards redundancy. */)
{
    int i, dimension = vect_size(base_index);
    Psysteme ps_interm, ps_project, ps_tmp;
    Pbase reverse_base;
    Pvecteur pb;
    Pcontrainte	ineq = NULL,
	*c = (Pcontrainte*) malloc(sizeof(Pcontrainte)*(dimension+1));

    if (VECTEUR_NUL_P(base_index)) 
    {
	*penumeration = sc_rn(NULL);
	*pcondition = sc_dup(scn);
	return;
    }

    ps_project = sc_dup(scn);
    sc_transform_eg_in_ineg(ps_project);
    ps_project = sc_sort_constraints(ps_project, base_index);
    ps_project = sc_elim_redond(ps_project);

    if (SC_UNDEFINED_P(ps_project))
	ps_project = sc_empty(base_difference(sc_base(scn), base_index));

    if (sc_empty_p(ps_project))
    {
	*penumeration = sc_empty(base_index);
	*pcondition = ps_project;
	return;
    }

    reverse_base = base_reversal(base_index);
    
    for (pb=reverse_base, i=dimension;
	 !VECTEUR_NUL_P(pb);
	 pb=pb->succ, i--)
    {
	c[i] = contrainte_dup_extract(sc_inegalites(ps_project), pb->var);

	ps_project = sc_projection(ps_project, pb->var);
	vect_erase_var(&sc_base(ps_project), pb->var);
	ps_project->dimension--;
	ps_project = sc_triang_elim_redund(ps_project, base_index); 
    }

    if (ps_project==NULL || sc_empty_p(ps_project))
    {
	*penumeration = sc_empty(base_index);
	*pcondition = ps_project;
	return;
    }

    c[0] = contrainte_dup_extract(sc_inegalites(ps_project), NULL);
    sc_rm(ps_project);

    for (i=0; i<dimension+1; i++)
	ineq = contrainte_append(c[i], ineq);

    ps_interm = sc_make(NULL, ineq);

    /* returns a a la Omega system, with no innerward redundancy.
     * in fact this is just a quick approximation of that property.
     * the way the system is computed may keep some outerward redundancies,
     * because a constraint may be redundant with some not yet computed
     * projection...
     */
    if (redundancy)
    {
	*pcondition = get_other_constraints(&ps_interm, base_index);
	sc_transform_ineg_in_eg(*pcondition); 
	*penumeration = ps_interm;

	return;
    }

    /* ELSE remove redundancy in the system...
     *
     * include the original system again, to recover simple
     * constraints that may have been removed.
     * May not be interesting...
     */
    ps_tmp =  sc_dup(scn);
    sc_transform_eg_in_ineg(ps_tmp);
    //assert(sc_consistent_p(ps_tmp));
    //assert(sc_consistent_p(ps_interm));
    // sc_fusion() is declared obsolete: sc_append() and
    // sc_intersection() are suggested as replacements
    //ps_interm = sc_fusion(ps_interm, ps_tmp);
    ps_interm = sc_append(ps_interm, ps_tmp);
    //assert(sc_consistent_p(ps_interm));
    ps_interm = sc_triang_elim_redund(ps_interm, base_index); 

    *pcondition = get_other_constraints(&ps_interm, base_index);
    sc_transform_ineg_in_eg(*pcondition); 
    *penumeration = ps_interm;

    base_rm(reverse_base), free(c);

    /*  what is returned must be ok.
     */
    assert(!SC_UNDEFINED_P(*pcondition) && !SC_UNDEFINED_P(*penumeration));
}

/* see comments above. 
 */
void 
algorithm_row_echelon(
    Psysteme scn,
    Pbase base_index,
    Psysteme *pcondition,
    Psysteme *penumeration)
{
  algorithm_row_echelon_generic
    (scn, base_index, pcondition, penumeration, false);  
}

void sc_set_row_echelon_redundancy(bool b __attribute__ ((unused)))
{
  return;
}


/*----------------------------------------------------------
 *
 * ALGORITHM TILING from Ancourt and Irigoin, PPoPP'91
 *
 * The algorithm is slightly different: 
 * 
 * constraints may appear thru the projections that do not contain the
 * desired loop variables. These constraints are taken out of the
 * systems. The inner ones are reinjected to help for the outer loop, and
 * those of the outer loop are stored as conditions to be checked before
 * the loop nest. The intuition is that if these constraints are violated,
 * the polyhedron is empty, and the loop nest *must* be avoided.
 * 
 * Corinne ANCOURT, Fabien COELHO, Apr 1 94.
 *
 * - may be improved by recognizing equalities?
 *   then deducable variables could be rebuilt.
 * - there should be no equalities on the scanners...
 *
 * void algorithm_tiling(syst, outer, inner, 
 *      pcondition, ptile_enum, piter_enum)
 * Psysteme syst;
 * Pbase outer, inner;
 * Psysteme *pcondition, *ptile_enum, *piter_enum;
 *
 *      IN: syst, outer, inner
 *     OUT: pcondition, ptile_enum, piter_enum
 */
void algorithm_tiling
  (Psysteme syst,
   Pbase outer,
   Pbase inner,
   Psysteme *pcondition,
   Psysteme *ptile_enum,
   Psysteme *piter_enum)
{
    Psysteme sc = SC_UNDEFINED,	transfer = SC_UNDEFINED;
    Pbase b = BASE_NULLE;

    /* tiles iterations enumeration row echelon
     */
    algorithm_row_echelon_generic(syst, inner, &transfer, piter_enum, true);

    if (sc_empty_p(transfer))
    {
	sc_rm(transfer), sc_rm(*piter_enum),
	*piter_enum = sc_empty(BASE_NULLE),
	*ptile_enum = sc_empty(BASE_NULLE),
	*pcondition = sc_empty(BASE_NULLE);
	return;
    }

    /* project variables
     */
    for(b=inner, sc=sc_safe_intersection(sc_rn(BASE_NULLE), syst, transfer); 
	b!=BASE_NULLE; 
	sc=sc_projection(sc, var_of(b)), b=b->succ);

    sc_rm(transfer);
    sc_nredund(&sc);

    /* tiles enumeration row echelon
     */
    algorithm_row_echelon_generic(sc, outer, pcondition, ptile_enum, true);

    if (sc_empty_p(*pcondition))
    {
	sc_rm(*piter_enum), sc_rm(*ptile_enum),
	*piter_enum = sc_empty(BASE_NULLE),
	*ptile_enum = sc_empty(BASE_NULLE);
	return;
    }

    /* clean bases
     */
    sc_base(*ptile_enum)=(base_rm(sc_base(*ptile_enum)), BASE_NULLE),
    sc_creer_base(*ptile_enum);

    sc_base(*piter_enum)=(base_rm(sc_base(*piter_enum)), BASE_NULLE),
    sc_creer_base(*piter_enum);
}

/*   that is all
 */
