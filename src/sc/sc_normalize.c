 /* package sc */

#include <string.h>
#include <stdio.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Psysteme sc_normalize(Psysteme ps): normalisation d'un systeme d'equation
 * et d'inequations lineaires en nombres entiers ps, en place.
 *
 * Normalisation de chaque contrainte, i.e. division par le pgcd des
 * coefficients (cf. ?!? )
 *
 * Verification de la non redondance de chaque contrainte avec les autres:
 *
 * Pour les egalites, on elimine une equation si on a un systeme d'egalites
 * de la forme :
 * 
 *   a1/    Ax - b == 0,            ou  b1/        Ax - b == 0,              
 *          Ax - b == 0,                           b - Ax == 0,              
 * 
 * ou c1/ 0 == 0	 
 * 
 * Pour les inegalites, on elimine une inequation si on a un systeme de
 * contraintes de la forme :
 * 
 *   a2/    Ax - b <= c,             ou   b2/     0 <= const  (avec const >=0)
 *          Ax - b <= c             
 * 
 *   ou  c2/   Ax == b,	
 *             Ax <= c        avec b <= c,
 * 
 *   ou  d2/    Ax <= b,
 *              Ax <= c    avec c >= b ou b >= c
 * 
 * sc_normalize retourne NULL quand la normalisation a montre que le systeme
 * etait non faisable
 *
 * FI: a revoir de pres; devrait retourner SC_EMPTY en cas de non faisabilite
 */
Psysteme sc_normalize(ps)
Psysteme ps;
{
    Pcontrainte eq;
    boolean is_sc_fais = TRUE;

    ps = sc_kill_db_eg(ps);
    if (ps) {
	for (eq = ps->egalites;
	     (eq != NULL) && is_sc_fais;
	     eq=eq->succ) {
	    /* normalisation de chaque equation */
	    if (eq->vecteur)    {
		vect_normalize(eq->vecteur);
		if ((is_sc_fais = egalite_normalize(eq))== TRUE)
		    is_sc_fais = sc_elim_simple_redund_with_eq(ps,eq);
	    }
	}
	for (eq = ps->inegalites;
	     (eq!=NULL) && is_sc_fais;
	     eq=eq->succ) {
	    if (eq->vecteur)    {
		vect_normalize(eq->vecteur);
		if ((is_sc_fais = inegalite_normalize(eq))== TRUE)
		    is_sc_fais = sc_elim_simple_redund_with_ineq(ps,eq);
	    }
	}

	ps = sc_kill_db_eg(ps);
	sc_elim_empty_constraints(ps, TRUE);
	sc_elim_empty_constraints(ps, FALSE);
    }

    if (!is_sc_fais) 
	sc_rm(ps), ps=NULL;
    
    return(ps);
}

/*
 * ??? could be improve by rewriting *_elim_redond so that only
 * (in)eq may be removed?
 *
 * FC 02/11/94
 */

Psysteme sc_add_normalize_eq(ps, eq)
Psysteme ps;
Pcontrainte eq;
{
    Pcontrainte c;

    if (!eq->vecteur) return(ps);

    vect_normalize(eq->vecteur);
    if (egalite_normalize(eq))
    {
	c = ps->egalites,
	ps->egalites = eq,
	eq->succ = c;
	ps->nb_eq++;

	if (!sc_elim_simple_redund_with_eq(ps, eq))
	{
	    sc_rm(ps);
	    return(NULL);
	}

	sc_rm_empty_constraints(ps, TRUE);
    }

    return(ps);
}

Psysteme sc_add_normalize_ineq(ps, ineq)
Psysteme ps;
Pcontrainte ineq;
{
    Pcontrainte c;

    if (!ineq->vecteur) return(ps);

    vect_normalize(ineq->vecteur);
    if (inegalite_normalize(ineq))
    {
	c = ps->inegalites,
	ps->inegalites = ineq,
	ineq->succ = c;
	ps->nb_ineq++;

	if (!sc_elim_simple_redund_with_ineq(ps, ineq))
	{
	    sc_rm(ps);
	    return(NULL);
	}

	sc_rm_empty_constraints(ps, FALSE);
    }

    return(ps);
}

/* Psysteme sc_safe_normalize(Psysteme ps)
 * output   : ps, normalized.
 * modifies : ps.
 * comment  : when ps is not feasible, returns sc_empty.	
 */
Psysteme sc_safe_normalize(ps)
Psysteme ps;
{
    Pbase ps_base = base_dup(sc_base(ps));

    if (sc_rn_p(ps))
	return(ps);
    else {        
	ps = sc_normalize(ps);
	if (ps == NULL)
	    return sc_empty(ps_base);
	else return(ps);
    }
}
