#include <sys/stdtypes.h> 
#include <stdio.h>
#include <malloc.h>
#include <assert.h>
extern int fprintf();

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
Psysteme new_loop_bound(scn, base_index)
Psysteme scn;
Pbase base_index;
{
/*
    Psysteme PD;
    Psysteme ps_project;
    Psysteme ps1;
    Pvecteur pb;
 
    if (VECTEUR_NUL_P(base_index)) return(scn);

    PD = sc_new();
    scn = sc_normalize(scn);
    ps_project = sc_dup(scn);
    
    for (pb = base_reversal(base_index); 
	 !VECTEUR_NUL_P(pb) && pb->succ!= NULL; 
	 pb = pb->succ)
    {
	ps_project = sc_projection(ps_project,pb->var);	
	ps_project = sc_triang_elim_redond(ps_project,base_index);

	ps1 = sc_dup(ps_project);
	PD = sc_fusion(PD,ps1);
	PD = sc_triang_elim_redond(PD,base_index);
    }

    PD = sc_fusion(scn,PD);
    PD = sc_triang_elim_redond(PD,base_index);

    return PD;
*/
    Psysteme
	result = NULL,
	condition = NULL,
	enumeration = NULL;

/*    extern char* entity_local_name();
    fprintf(stderr, "[new_loop_bound] input:\n");
    base_fprint(stderr, base_index, entity_local_name);
    sc_fprint(stderr, scn, entity_local_name); */

    algorithm_row_echelon(scn, base_index, &condition, &enumeration);

    sc_rm(scn);
    result = sc_fusion(condition, enumeration);

/*    fprintf(stderr, "[new_loop_bound] output:\n");
    sc_fprint(stderr, result, entity_local_name); */

    return(result);
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
    Psysteme
	others = SC_UNDEFINED;
    Pcontrainte
	egothers = (Pcontrainte) NULL,
	inothers = (Pcontrainte) NULL,
	egsyst = (Pcontrainte) NULL,
	insyst = (Pcontrainte) NULL;

    if (!SC_UNDEFINED_P(*psyst))
    {
	Pcontrainte_separate_on_vars
	    (sc_egalites(*psyst), vars, &egsyst, &egothers);
	Pcontrainte_separate_on_vars
	    (sc_inegalites(*psyst), vars, &insyst, &inothers);
	
	/*
	 * result in built and syst is modified.
	 */
	others = sc_make(egothers, inothers);
	*psyst = (sc_rm(*psyst), sc_make(egsyst, insyst));
    }
	
    return(others);
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
 * the equalities are assumed to have been translated into inequalities;
 * scn IS NOT modified.
 *
 * !!! there should be no equalities on the scanners (base_index)
 */
void algorithm_row_echelon(scn, base_index, pcondition, penumeration)
Psysteme scn;
Pbase base_index;
Psysteme *pcondition, *penumeration;
{
    int 
	i, dimension = vect_size(base_index);
    Psysteme 
	ps_interm, ps_project;
    Pbase
	reverse_base;
    Pvecteur
	pb;
    Pcontrainte
	ineq = NULL,
	*c = (Pcontrainte*) malloc(sizeof(Pcontrainte)*(dimension+1));

    /* check for equalities on the scanners... 
     */
    if (!constraints_without_vars(sc_egalites(scn), base_index))
	fprintf(stderr, "[algorithm_row_echelon] eq with scanner found\n"),
	abort();

    if (VECTEUR_NUL_P(base_index)) 
    {
	*penumeration = sc_rn(NULL);
	*pcondition = sc_dup(scn);
	return;
    }

    ps_project = sc_dup(scn);
    ps_project = sc_sort_constraints(ps_project, base_index);
    ps_project = sc_elim_redond(ps_project);

    if (ps_project==NULL || sc_empty_p(ps_project))
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
	/* the real redundancy elimination should be used ? */
	ps_project = sc_triang_elim_redond(ps_project, base_index);
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

    /*
    fprintf(stderr, "intermediate redundant system:\n");
    sc_fprint(stderr, ps_interm, *variable_default_name);
    */

    /*  include the original system again, to recover simple
     *  constraints that may have been removed. May not be interesting...
     */
    ps_interm = sc_fusion(ps_interm, sc_dup(scn));
    ps_interm = sc_triang_elim_redond(ps_interm, base_index); 

    *pcondition = get_other_constraints(&ps_interm, base_index);
    *penumeration = ps_interm;

    base_rm(reverse_base), free(c);
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
void algorithm_tiling(syst, outer, inner, 
		      pcondition, ptile_enum, piter_enum)
Psysteme syst;
Pbase outer, inner;
Psysteme *pcondition, *ptile_enum, *piter_enum;
{
    Psysteme
	sc = SC_UNDEFINED,
	transfer = SC_UNDEFINED;
    Pbase
	b = BASE_NULLE;

    /*
     * tiles iterations enumeration row echelon
     */
    algorithm_row_echelon(syst, inner, &transfer, piter_enum);

    if (SC_UNDEFINED_P(*piter_enum) || SC_UNDEFINED_P(transfer))
    {
	sc_rm(transfer),
	*piter_enum = sc_empty(BASE_NULLE),
	*ptile_enum = sc_empty(BASE_NULLE),
	*pcondition = sc_rn(BASE_NULLE);
	return;
    }

    /*
     * project variables
     */
    for(b=inner, sc=sc_safe_intersection(sc_rn(BASE_NULLE), syst, transfer); 
	b!=BASE_NULLE; 
	sc=sc_projection(sc, var_of(b)), b=b->succ);

    sc_rm(transfer);
    sc_nredund(&sc);

    /*
     * tiles enumeration row echelon
     */
    algorithm_row_echelon(sc, outer, pcondition, ptile_enum);

    if (SC_UNDEFINED_P(*ptile_enum) || SC_UNDEFINED_P(*pcondition))
    {
	sc_rm(*piter_enum),
	sc_rm(*pcondition),
	*piter_enum = sc_empty(BASE_NULLE),
	*ptile_enum = sc_empty(BASE_NULLE),
	*pcondition = sc_rn(BASE_NULLE);
	return;
    }

    /*
     * clean bases
     */
    sc_base(*ptile_enum)=(base_rm(sc_base(*ptile_enum)), BASE_NULLE),
    sc_creer_base(*ptile_enum);

    sc_base(*piter_enum)=(base_rm(sc_base(*piter_enum)), BASE_NULLE),
    sc_creer_base(*piter_enum);
}

/*
 *   that is all
 */

