 /* package sc */

#include <string.h>
#include <stdio.h>
#include <assert.h>
/* #include <values.h> */
#include <limits.h>

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

/* Psysteme sc_normalize2(Psysteme ps): normalisation d'un systeme d'equation
 * et d'inequations lineaires en nombres entiers ps, en place.
 *
 * Normalisation de chaque contrainte, i.e. division par le pgcd des
 * coefficients (cf. ?!? )
 *
 * Propagation des constantes definies par les equations dans les
 * inequations. E.g. N==1.
 *
 * Selection des variables de rang inferieur quand il y a ambiguite: N==M.
 * M is used wherever possible.
 *
 * Selection des variables eliminables exactement. E.g. N==4M. N is
 * substituted by 4M wherever possible. Ceci permet de raffiner les
 * constantes dans les inegalites.
 *
 * Les equations de trois variables ou plus ne sont pas utilisees pour ne
 * pas rendre les inegalites trop complexes.
 *
 * Verification de la non redondance de chaque contrainte avec les autres
 *
 * Les contraintes sont normalisees par leurs PGCDs.  Les constantes sont
 * propagees dans les inegalites.  Les paires de variables equivalentes
 * sont propagees dans les inegalites en utilisant la variable de moindre
 * rang dans la base.
 *
 * Pour les egalites, on elimine une equation si on a un systeme d'egalites
 * de la forme :
 * 
 *   a1/    Ax - b == 0,            ou  b1/        Ax - b == 0,              
 *          Ax - b == 0,                           b - Ax == 0,              
 * 
 * ou c1/ 0 == 0	 
 * 
 * Si on finit avec b==0, la non-faisabilite est detectee.
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
 * Les doubles inegalites syntaxiquement equivalentes a une egalite sont
 * detectees: Ax <= b, Ax >= b
 *
 * Si deux inegalites sont incompatibles, la non-faisabilite est detectee:
 * b <= Ax <= c et c < b.
 *
 * sc_normalize retourne NULL/SC_EMPTY quand la normalisation a montre que
 * le systeme etait non faisable.
 *
 * Une grande partie du travail est effectue dans sc_elim_db_constraints()
 *
 * FI: a revoir de pres; devrait retourner SC_EMPTY en cas de non faisabilite
 *
 */
Psysteme sc_normalize2(ps)
Psysteme ps;
{
  Pcontrainte eq;

  ps = sc_elim_double_constraints(ps);
  sc_elim_empty_constraints(ps, TRUE);
  sc_elim_empty_constraints(ps, FALSE);

  if (!SC_UNDEFINED_P(ps)) {
    Pbase b = sc_base(ps);

    /* Eliminate variables linked by a two-term equation. Preserve integer
       information or choose variable with minimal rank in basis b if some
       ambiguity exists. */
    for (eq = ps->egalites; (!SC_UNDEFINED_P(ps) && eq != NULL); eq=eq->succ) {
      Pvecteur veq = contrainte_vecteur(eq);
      if(((vect_size(veq)==2) && (vect_coeff(TCST,veq)==VALUE_ZERO))
	 || ((vect_size(veq)==3) && (vect_coeff(TCST,veq)!=VALUE_ZERO))) {
	Pbase bveq = make_base_from_vect(veq);
	Variable v1 = vecteur_var(bveq);
	Variable v2 = vecteur_var(vecteur_succ(bveq));
	Variable v = VARIABLE_UNDEFINED;
	Value a1 = value_abs(vect_coeff(v1, veq));
	Value a2 = value_abs(vect_coeff(v2, veq));

	if(a1==a2) {
	  /* Then, after normalization, a1 and a2 must be one */
	  if(rank_of_variable(b, v1) < rank_of_variable(b, v2)) {
	    v = v2;
	  }
	  else {
	    v = v1;
	  }
	}
	else if(value_one_p(a1)) {
	  v = v1;
	}
	else if(value_one_p(a2)) {
	  v = v2;
	}
	if(VARIABLE_DEFINED_P(v)) {
	  /* An overflow is unlikely... but it should be handled here
	     I guess rather than be subcontracted? */
	  sc_simple_variable_substitution_with_eq_ofl_ctrl(ps, eq, v, OFL_CTRL);
	}
      }
    }
  }

  if (!SC_UNDEFINED_P(ps)) {
    /* Propagate constant definitions, only once although a triangular
       system might require n steps is the equations are in the worse order */
    for (eq = ps->egalites; (!SC_UNDEFINED_P(ps) && eq != NULL); eq=eq->succ) {
      Pvecteur veq = contrainte_vecteur(eq);
      if(((vect_size(veq)==1) && (vect_coeff(TCST,veq)==VALUE_ZERO))
	 || ((vect_size(veq)==2) && (vect_coeff(TCST,veq)!=VALUE_ZERO))) {
	Variable v = term_cst(veq)? vecteur_var(vecteur_succ(veq)) : vecteur_var(veq);
	Value a = term_cst(veq)? vecteur_val(vecteur_succ(veq)) : vecteur_val(veq);

	if(value_one_p(a) || value_mone_p(a) || vect_coeff(TCST,veq)==VALUE_ZERO
	   || value_mod(a,vect_coeff(TCST,veq))==VALUE_ZERO) {
	  /* An overflow is unlikely... but it should be handled here
	     I guess rather than be subcontracted. */
	  sc_simple_variable_substitution_with_eq_ofl_ctrl(ps, eq, v, OFL_CTRL);
	}
	else {
	  sc_rm(ps);
	  ps = SC_UNDEFINED;
	}
      }
    }

    ps = sc_elim_double_constraints(ps);
    sc_elim_empty_constraints(ps, TRUE);
    sc_elim_empty_constraints(ps, FALSE);
  }
    
  return(ps);
}

/*
 * ??? could be improved by rewriting *_elim_redond so that only
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

    if (!sc_rn_p(ps) && !sc_empty_p(ps))
    {
	Pbase ps_base = base_dup(sc_base(ps));	
	ps = sc_normalize(ps);
	if (ps == SC_EMPTY)
	    ps = sc_empty(ps_base);
	else 
	    base_rm(ps_base);
    }
    return(ps);
}

static Psysteme sc_rational_feasibility(Psysteme sc)
{

    if(!sc_rational_feasibility_ofl_ctrl((sc), OFL_CTRL,TRUE)) {
	sc_rm(sc);
	sc = SC_EMPTY;
    }
    return sc;
}

/* Psysteme sc_strong_normalize(Psysteme ps)
 *
 * Apply sc_normalize first. Then solve the equations in a copy
 * of ps and propagate in equations and inequations.
 *
 * Flag as redundant equations 0 == 0 and inequalities 0 <= k
 * with k a positive integer constant when they appear.
 *
 * Flag the system as non feasible if any equation 0 == k or any inequality
 * 0 <= -k with k a strictly positive constant appears.
 *
 * Then, we'll have to deal with remaining inequalities...
 *
 * Argument ps is not modified by side-effect but it is freed for
 * backward compatability. SC_EMPTY is returned for
 * backward compatability.
 *
 * The code is difficult to understand because a sparse representation
 * is used. proj_ps is initially an exact copy of ps, with the same constraints
 * in the same order. The one-to-one relationship between constraints
 * must be maintained when proj_ps is modified. This makes it impossible to
 * use most routines available in Linear.
 *
 * Note: this is a redundancy elimination algorithm a bit too strong
 * for sc_normalize.c...
 */
Psysteme sc_strong_normalize(Psysteme ps)
{
    Psysteme new_ps =
	sc_strong_normalize_and_check_feasibility
	    (ps, (Psysteme (*)(Psysteme)) NULL);

    return new_ps;
}

Psysteme sc_strong_normalize3(Psysteme ps)
{
    /*
    Psysteme new_ps =
	sc_strong_normalize_and_check_feasibility
	    (ps, sc_elim_redund);
	    */
    Psysteme new_ps =
	sc_strong_normalize_and_check_feasibility
	    (ps, sc_rational_feasibility);

    return new_ps;
}

Psysteme sc_strong_normalize_and_check_feasibility
(Psysteme ps,
 Psysteme (*check_feasibility)(Psysteme))
{

    Psysteme new_ps = SC_UNDEFINED;
    Psysteme proj_ps = SC_UNDEFINED;
    boolean feasible_p = TRUE;
    Psysteme ps_backup = sc_dup(ps);
    /*
    fprintf(stderr, "[sc_strong_normalize]: Begin\n");
    */

  
    CATCH(overflow_error) 
	{
	    /* CA */
	    fprintf(stderr,"overflow error in  normalization\n"); 
	    new_ps=ps_backup;
	}
    TRY 
	{
	    if(!SC_UNDEFINED_P(ps)) {
		if(!SC_EMPTY_P(ps = sc_normalize(ps))) {
		    Pcontrainte eq = CONTRAINTE_UNDEFINED;
		    Pcontrainte ineq = CONTRAINTE_UNDEFINED;
		    Pcontrainte proj_eq = CONTRAINTE_UNDEFINED;
		    Pcontrainte next_proj_eq = CONTRAINTE_UNDEFINED;
		    Pcontrainte proj_ineq = CONTRAINTE_UNDEFINED;
		    Pcontrainte new_eq = CONTRAINTE_UNDEFINED;
		    Pcontrainte new_ineq = CONTRAINTE_UNDEFINED;
		    
		    /*
		      fprintf(stderr,
		      "[sc_strong_normalize]: After call to sc_normalize\n");
		    */
		    
		    /* We need an exact copy of ps to have equalities
		     * and inqualities in the very same order
		     */
		    new_ps = sc_dup(ps);
		    proj_ps = sc_dup(new_ps);
		    sc_rm(new_ps);
		    new_ps = sc_make(NULL, NULL);
		    
		    /*
		      fprintf(stderr, "[sc_strong_normalize]: Input system %x\n",
		      (unsigned int) ps);
		      sc_dump(ps);
		      fprintf(stderr, "[sc_strong_normalize]: Copy system %x\n",
		      (unsigned int) ps);
		      sc_dump(proj_ps);
		    */
		    
		    /* Solve the equalities */
		    for(proj_eq = sc_egalites(proj_ps),
			    eq = sc_egalites(ps); 
			!CONTRAINTE_UNDEFINED_P(proj_eq);
			eq = contrainte_succ(eq)) {
			
			/* proj_eq might suffer in the substitution... */
			next_proj_eq = contrainte_succ(proj_eq);
			
			if(egalite_normalize(proj_eq)) {
			    if(CONTRAINTE_NULLE_P(proj_eq)) {
				/* eq is redundant */
				;
			    }
			    else {
				Pcontrainte def = CONTRAINTE_UNDEFINED;
				/* keep eq */
				Variable v = TCST;
				Pvecteur pv;
				
				new_eq = contrainte_dup(eq);
				sc_add_egalite(new_ps, new_eq);
				/* use proj_eq to eliminate a variable */
				
				/* Let's use a variable with coefficient 1 if
				 * possible
				 */
				for( pv = contrainte_vecteur(proj_eq);
				     !VECTEUR_NUL_P(pv);
				     pv = vecteur_succ(pv)) {
				    if(!term_cst(pv)) {
					v = vecteur_var(pv);
					if(value_one_p(vecteur_val(pv))) {
					    break;
					}
				    }
				}
				assert(v!=TCST);
				
				/* A softer substitution is needed in order to
				 * preserve the relationship between ps and proj_ps
				 */ 
				/*
				  if(sc_empty_p(proj_ps =
				  sc_variable_substitution_with_eq_ofl_ctrl
				  (proj_ps, proj_eq, v, NO_OFL_CTRL))) {
				  feasible_p = FALSE;
				  break;
				  }
				  else {
				  ;
				  }
				*/
				
				/* proj_eq itself is going to be modified in proj_ps.
				 * use a copy!
				 */
				def = contrainte_dup(proj_eq);
				proj_ps = 
				    sc_simple_variable_substitution_with_eq_ofl_ctrl
				    (proj_ps, def, v, NO_OFL_CTRL);
				contrainte_rm(def);
				/*
				  int contrainte_subst_ofl_ctrl(v,def,c,eq_p, ofl_ctrl)
				*/
			    }
			}
			else {
			    /* The system is not feasible. Stop */
			    feasible_p = FALSE;
			    break;
			}
			
			/*
			  fprintf(stderr,
			  "Print the three systems at each elimination step:\n");
			  fprintf(stderr, "[sc_strong_normalize]: Input system %x\n",
			  (unsigned int) ps);
			  sc_dump(ps);
			  fprintf(stderr, "[sc_strong_normalize]: Copy system %x\n",
			  (unsigned int) proj_ps);
			  sc_dump(proj_ps);
			  fprintf(stderr, "[sc_strong_normalize]: New system %x\n",
			  (unsigned int) new_ps);
			  sc_dump(new_ps);
			*/
			
			proj_eq = next_proj_eq;
		    }
		    assert(!feasible_p ||
			   (CONTRAINTE_UNDEFINED_P(eq) && CONTRAINTE_UNDEFINED_P(ineq)));
		    
		    /* Check the inequalities */
		    for(proj_ineq = sc_inegalites(proj_ps),
			    ineq = sc_inegalites(ps);
			feasible_p && !CONTRAINTE_UNDEFINED_P(proj_ineq);
			proj_ineq = contrainte_succ(proj_ineq),
			    ineq = contrainte_succ(ineq)) {
			
			if(inegalite_normalize(proj_ineq)) {
			    if(contrainte_constante_p(proj_ineq)
			       && contrainte_verifiee(proj_ineq, FALSE)) {
				/* ineq is redundant */
				;
			    }
			    else {
				int i;
				i = sc_check_inequality_redundancy(proj_ineq, proj_ps);
				if(i==0) {
				    /* keep ineq */
				    new_ineq = contrainte_dup(ineq);
				    sc_add_inegalite(new_ps, new_ineq);
				}
				else if(i==1) {
				    /* ineq is redundant with another inequality:
				     * destroy ineq to avoid the mutual elimination of
				     * two identical constraints
				     */
				    eq_set_vect_nul(proj_ineq);
				}
				else if(i==2) {
				    feasible_p = FALSE;
				    break;
				}
				else {
				    assert(FALSE);
				}
			    }
			}
			else {
			    /* The system is not feasible. Stop */
			    feasible_p = FALSE;
			    break;
			}
		    }
		    
		    /*
		      fprintf(stderr,
		      "Print the three systems after inequality normalization:\n");
		      fprintf(stderr, "[sc_strong_normalize]: Input system %x\n",
		      (unsigned int) ps);
		      sc_dump(ps);
		      fprintf(stderr, "[sc_strong_normalize]: Copy system %x\n",
		      (unsigned int) proj_ps);
		      sc_dump(proj_ps);
		      fprintf(stderr, "[sc_strong_normalize]: New system %x\n",
		      (unsigned int) new_ps);
		      sc_dump(new_ps);
		    */
		    
		    /* Check redundancy between residual inequalities */
		    
		    /* sc_elim_simple_redund_with_ineq(ps,ineg) */
		    
		    /* Well, sc_normalize should not be able to do much here! */
		    /*
		      new_ps = sc_normalize(new_ps);
		      feasible_p = (!SC_EMPTY_P(new_ps));
		    */
		}
		else {
		    /*
		      fprintf(stderr,
		      "[sc_strong_normalize]:"
		      " Non-feasibility detected by sc_normalize\n");
		    */
		    feasible_p = FALSE;
		}
	    }
	    else {
		/*
		  fprintf(stderr,
		  "[sc_strong_normalize]: Empty system as input\n");
		*/
		feasible_p = FALSE;
	    }
	    
	    if(feasible_p && check_feasibility != (Psysteme (*)(Psysteme)) NULL) {
		proj_ps = check_feasibility(proj_ps);
		feasible_p = !SC_EMPTY_P(proj_ps);
	    }
	    
	    if(!feasible_p) {
		sc_rm(new_ps);
		new_ps = SC_EMPTY;
	    }
	    else {
		sc_base(new_ps) = sc_base(ps);
		sc_base(ps) = BASE_UNDEFINED;
		sc_dimension(new_ps) = sc_dimension(ps);
		assert(sc_weak_consistent_p(new_ps));
	    }
	    
	    sc_rm(proj_ps);
	    sc_rm(ps);
	    sc_rm(ps_backup);
	    /*
	      fprintf(stderr, "[sc_strong_normalize]: Final value of new system %x:\n",
	      (unsigned int) new_ps);
	      sc_dump(new_ps);
	      fprintf(stderr, "[sc_strong_normalize]: End\n");
	    */
	
	    UNCATCH(overflow_error);
	}
    return new_ps;
}
    
/* Psysteme sc_strong_normalize2(Psysteme ps)
 *
 * Apply sc_normalize first. Then solve the equations in
 * ps and propagate substitutions in equations and inequations.
 *
 * Flag as redundant equations 0 == 0 and inequalities 0 <= k
 * with k a positive integer constant when they appear.
 *
 * Flag the system as non feasible if any equation 0 == k or any inequality
 * 0 <= -k with k a strictly positive constant appears.
 *
 * Then, we'll have to deal with remaining inequalities...
 *
 * Argument ps is modified by side-effect. SC_EMPTY is returned for
 * backward compatability if ps is not feasible.
 *
 * Note: this is a redundancy elimination algorithm a bit too strong
 * for sc_normalize.c... but it's not strong enough to qualify as
 * a normalization procedure.
 */
Psysteme sc_strong_normalize2(Psysteme ps)
{

#define if_debug_sc_strong_normalize_2 if(FALSE)

    Psysteme new_ps = sc_make(NULL, NULL);
    boolean feasible_p = TRUE;

    Psysteme ps_backup = sc_dup(ps);
    CATCH(overflow_error) 
	{
	    /* CA */
	    fprintf(stderr,"overflow error in  normalization\n"); 
	    new_ps=ps_backup;
	}
    TRY 
	{
	    if_debug_sc_strong_normalize_2 {
		fprintf(stderr, "[sc_strong_normalize2]: Begin\n");
	    }
	    
	    if(!SC_UNDEFINED_P(ps)) {
		if(!SC_EMPTY_P(ps = sc_normalize(ps))) {
		    Pcontrainte eq = CONTRAINTE_UNDEFINED;
		    Pcontrainte ineq = CONTRAINTE_UNDEFINED;
		    Pcontrainte next_eq = CONTRAINTE_UNDEFINED;
		    Pcontrainte new_eq = CONTRAINTE_UNDEFINED;
		    
		    if_debug_sc_strong_normalize_2 {
			fprintf(stderr,
				"[sc_strong_normalize2]: After call to sc_normalize\n");
			fprintf(stderr, "[sc_strong_normalize2]: Input system %p\n",
				ps);
			sc_dump(ps);
		    }
		    
		    /* Solve the equalities */
		    for(eq = sc_egalites(ps); 
			!CONTRAINTE_UNDEFINED_P(eq);
			eq = next_eq) {
			
			/* eq might suffer in the substitution... */
			next_eq = contrainte_succ(eq);
			
			if(egalite_normalize(eq)) {
			    if(CONTRAINTE_NULLE_P(eq)) {
				/* eq is redundant */
				;
			    }
			    else {
				Pcontrainte def = CONTRAINTE_UNDEFINED;
				Variable v = TCST;
				Pvecteur pv;
				
				/* keep eq */
				new_eq = contrainte_dup(eq);
				sc_add_egalite(new_ps, new_eq);
				
				/* use eq to eliminate a variable */
				
				/* Let's use a variable with coefficient 1 if
				 * possible
				 */
				for( pv = contrainte_vecteur(eq);
				     !VECTEUR_NUL_P(pv);
				     pv = vecteur_succ(pv)) {
				    if(!term_cst(pv)) {
					v = vecteur_var(pv);
					if(value_one_p(vecteur_val(pv))) {
					    break;
					}
				    }
				}
				assert(v!=TCST);
				
				/* A softer substitution is used
				 */ 
				/*
				  if(sc_empty_p(ps =
				  sc_variable_substitution_with_eq_ofl_ctrl
				  (ps, eq, v, OFL_CTRL))) {
				  feasible_p = FALSE;
				  break;
				  }
				  else {
				  ;
				  }
				*/
				
				/* eq itself is going to be modified in ps.
				 * use a copy!
				 */
				def = contrainte_dup(eq);
				ps = 
				    sc_simple_variable_substitution_with_eq_ofl_ctrl
				    (ps, def, v, NO_OFL_CTRL);
				contrainte_rm(def);
				/*
				  int contrainte_subst_ofl_ctrl(v,def,c,eq_p, ofl_ctrl)
				*/
			    }
			}
			else {
			    /* The system is not feasible. Stop */
			    feasible_p = FALSE;
			    break;
			}
			
			if_debug_sc_strong_normalize_2 {
			    fprintf(stderr,
				    "Print the two systems at each elimination step:\n");
			    fprintf(stderr, "[sc_strong_normalize2]: Input system %p\n",
				    ps);
			    sc_dump(ps);
			    fprintf(stderr, "[sc_strong_normalize2]: New system %p\n",
				    new_ps);
			    sc_dump(new_ps);
			}
			
		    }
		    assert(!feasible_p ||
			   (CONTRAINTE_UNDEFINED_P(eq) && CONTRAINTE_UNDEFINED_P(ineq)));
		    
		    /* Check the inequalities */
		    feasible_p = !SC_EMPTY_P(ps = sc_normalize(ps));
		    
		    if_debug_sc_strong_normalize_2 {
			fprintf(stderr,
				"Print the three systems after inequality normalization:\n");
			fprintf(stderr, "[sc_strong_normalize2]: Input system %p\n",
				ps);
			sc_dump(ps);
			fprintf(stderr, "[sc_strong_normalize2]: New system %p\n",
				new_ps);
			sc_dump(new_ps);
		    }
		}
		else {
		    if_debug_sc_strong_normalize_2 {
			fprintf(stderr,
				"[sc_strong_normalize2]:"
				" Non-feasibility detected by first call to sc_normalize\n");
		    }
		    feasible_p = FALSE;
		}
	    }
	    else {
		if_debug_sc_strong_normalize_2 {
		    fprintf(stderr,
			    "[sc_strong_normalize2]: Empty system as input\n");
		}
		feasible_p = FALSE;
	    }
	    
	    if(!feasible_p) {
		sc_rm(new_ps);
		new_ps = SC_EMPTY;
	    }
	    else {
		base_rm(sc_base(new_ps));
		sc_base(new_ps) = base_dup(sc_base(ps));
		sc_dimension(new_ps) = sc_dimension(ps);
		/* copy projected inequalities left in ps */
		new_ps = sc_safe_append(new_ps, ps);
		/* sc_base(ps) = BASE_UNDEFINED; */
		assert(sc_weak_consistent_p(new_ps));
	    }
	    
	    sc_rm(ps);
	    sc_rm(ps_backup);
	    if_debug_sc_strong_normalize_2 {
		fprintf(stderr,
			"[sc_strong_normalize2]: Final value of new system %p:\n",
			new_ps);
		sc_dump(new_ps);
		fprintf(stderr, "[sc_strong_normalize2]: End\n");
	    }
	UNCATCH(overflow_error);
	}    
    return new_ps;
}

/* Psysteme sc_strong_normalize4(Psysteme ps,
 *                               char * (*variable_name)(Variable))
 */
Psysteme sc_strong_normalize4(Psysteme ps, char * (*variable_name)(Variable))
{
    /*
    Psysteme new_ps =
	sc_strong_normalize_and_check_feasibility2
	    (ps, sc_normalize, variable_name, VALUE_MAX);
	    */

    Psysteme new_ps =
	sc_strong_normalize_and_check_feasibility2
	    (ps, sc_normalize, variable_name, 2);

    return new_ps;
}

Psysteme sc_strong_normalize5(Psysteme ps, char * (*variable_name)(Variable))
{
    /* Good, but pretty slow */
    /*
    Psysteme new_ps =
	sc_strong_normalize_and_check_feasibility2
	    (ps, sc_elim_redund, variable_name, 2);
	    */

    Psysteme new_ps =
	sc_strong_normalize_and_check_feasibility2
	    (ps, sc_rational_feasibility, variable_name, 2);

    return new_ps;
}

/* Psysteme sc_strong_normalize_and_check_feasibility2
 * (Psysteme ps,
 *  Psysteme (*check_feasibility)(Psysteme),
 *  char * (*variable_name)(Variable),
 * int level)
 *
 * Same as sc_strong_normalize2() but equations are used by increasing
 * order of their numbers of variables, and, within one equation,
 * the lexicographic minimal variables is chosen among equivalent variables.
 *
 * Equations with more than "level" variables are not used for the 
 * substitution. Unless level==VALUE_MAX.
 *
 * Finally, an additional normalization procedure is applied on the
 * substituted system. Another stronger normalization can be chosen
 * to benefit from the system size reduction (e.g. sc_elim_redund).
 * Or a light one to benefit from the inequality simplifications due
 * to equation solving (e.g. sc_normalize).
 */
Psysteme sc_strong_normalize_and_check_feasibility2
(Psysteme ps,
 Psysteme (*check_feasibility)(Psysteme),
 char * (*variable_name)(Variable),
 int level)
{

#define if_debug_sc_strong_normalize_and_check_feasibility2 if(FALSE)

  Psysteme new_ps = sc_make(NULL, NULL);
  boolean feasible_p = TRUE;

  Psysteme ps_backup = sc_dup(ps);
  CATCH(overflow_error) 
    {
      /* CA */
      fprintf(stderr,"overflow error in  normalization\n"); 
      new_ps=ps_backup;
    }
  TRY 
    {
      if_debug_sc_strong_normalize_and_check_feasibility2 {
	fprintf(stderr, 
		"[sc_strong_normalize_and_check_feasibility2]"
		" Input system %p\n", ps);
	sc_dump(ps);
      }
	    
      if(SC_UNDEFINED_P(ps)) {
	if_debug_sc_strong_normalize_and_check_feasibility2 {
	  fprintf(stderr,
		  "[sc_strong_normalize_and_check_feasibility2]"
		  " Empty system as input\n");
	}
	feasible_p = FALSE;
      }
      else if(SC_EMPTY_P(ps = sc_normalize(ps))) {
	if_debug_sc_strong_normalize_and_check_feasibility2 {
	  fprintf(stderr,
		  "[sc_strong_normalize_and_check_feasibility2]:"
		  " Non-feasibility detected by first call to sc_normalize\n");
	}
	feasible_p = FALSE;
      }
      else {
	Pcontrainte eq = CONTRAINTE_UNDEFINED;
	Pcontrainte ineq = CONTRAINTE_UNDEFINED;
	Pcontrainte next_eq = CONTRAINTE_UNDEFINED;
	Pcontrainte new_eq = CONTRAINTE_UNDEFINED;
	int nvar;
	int neq = sc_nbre_egalites(ps);
		
	if_debug_sc_strong_normalize_and_check_feasibility2 {
	  fprintf(stderr,
		  "[sc_strong_normalize_and_check_feasibility2]"
		  " Input system after normalization %p\n", ps);
	  sc_dump(ps);
	}
		
		
	/* 
	 * Solve the equalities (if any)
	 *
	 * Start with equalities with the smallest number of variables
	 * and stop when all equalities have been used and or when
	 * all equalities left have too many variables.
	 */
	for(nvar = 1;
	    feasible_p && neq > 0 && nvar <= level /* && sc_nbre_egalites(ps) != 0 */;
	    nvar++) {
	  for(eq = sc_egalites(ps); 
	      feasible_p && !CONTRAINTE_UNDEFINED_P(eq);
	      eq = next_eq) {
			
	    /* eq might suffer in the substitution... */
	    next_eq = contrainte_succ(eq);
			
	    if(egalite_normalize(eq)) {
	      if(CONTRAINTE_NULLE_P(eq)) {
				/* eq is redundant */
		;
	      }
	      else {
	        /* Equalities change because of substitutions.
		 * Their dimensions may go under the present
		 * required dimension, nvar. Hence the non-equality
		 * test.
		 */
		int d = vect_dimension(contrainte_vecteur(eq));
				
		if(d<=nvar) {
		  Pcontrainte def = CONTRAINTE_UNDEFINED;
		  Variable v = TCST;
		  Variable v1 = TCST;
		  Variable v2 = TCST;
		  Variable nv = TCST;
		  Pvecteur pv;
				    
		  /* keep eq */
		  new_eq = contrainte_dup(eq);
		  sc_add_egalite(new_ps, new_eq);
				    
		  /* use eq to eliminate a variable */
				    
		  /* Let's use a variable with coefficient 1 if
		   * possible. Among such variables,
		   * choose the lexicographically minimal one.
		   */
		  v1 = TCST;
		  v2 = TCST;
		  for( pv = contrainte_vecteur(eq);
		       !VECTEUR_NUL_P(pv);
		       pv = vecteur_succ(pv)) {
		    if(!term_cst(pv)) {
		      nv = vecteur_var(pv);
		      v2 = (v2==TCST)? nv : v2;
		      if (value_one_p(vecteur_val(pv))) {
			if(v1==TCST) {
			  v1 = nv;
			}
			else {
			  /* v1 = TCST; */
			  v1 =
			    (strcmp(variable_name(v1),
				    variable_name(nv))>=0)
			    ? nv : v1;
			}
		      }
		    }
		  }
		  v = (v1==TCST)? v2 : v1;
		  /* because of the !CONTRAINTE_NULLE_P() test */
		  assert(v!=TCST);
				    
		  /* eq itself is going to be modified in ps.
		   * use a copy!
		   */
		  def = contrainte_dup(eq);
		  ps = 
		    sc_simple_variable_substitution_with_eq_ofl_ctrl
		    (ps, def, v, NO_OFL_CTRL);
		  contrainte_rm(def);
		}
		else {
		  /* too early to use this equation eq */
		  /* If there any hope to use it in the future?
		   * Yes, if its dimension is no more than nvar+1
		   * because one of its variable might be substituted.
		   * If more variable are substituted, it's dimension
		   * is going to go down and it will be counted later...
		   * Well this is not true, it will be lost:-(
		   */
		  if(d<=nvar+1) {
		    neq++;
		  }
		  else {
				/* to be on the safe side till I find a better idea... */
		    neq++;
		  }
		}
	      }
	    }
	    else {
	      /* The system is not feasible. Stop */
	      feasible_p = FALSE;
	      break;
	    }
			
	    /* This reaaly generates a lot of about on real life system! */
	    /*
	      if_debug_sc_strong_normalize_and_check_feasibility2 {
	      fprintf(stderr,
	      "Print the two systems at each elimination step:\n");
	      fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: Input system %x\n",
	      (unsigned int) ps);
	      sc_dump(ps);
	      fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: New system %x\n",
	      (unsigned int) new_ps);
	      sc_dump(new_ps);
	      }
	    */
			
	    /* This is a much too much expensive transformation
	     * in an innermost loop!
	     *
	     * It cannot be used as a convergence test.
	     */
	    /* feasible_p = (!SC_EMPTY_P(ps = sc_normalize(ps))); */
			
	  }
		    
	  if_debug_sc_strong_normalize_and_check_feasibility2 {
	    fprintf(stderr,
		    "Print the two systems at each nvar=%d step:\n", nvar);
	    fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: Input system %p\n",
		    ps);
	    sc_dump(ps);
	    fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: New system %p\n",
		    new_ps);
	    sc_dump(new_ps);
	  }
	}
	sc_elim_empty_constraints(new_ps,TRUE);
	sc_elim_empty_constraints(ps,TRUE);
	assert(!feasible_p ||
	       (CONTRAINTE_UNDEFINED_P(eq) && CONTRAINTE_UNDEFINED_P(ineq)));
		
	/* Check the inequalities */
	assert(check_feasibility != (Psysteme (*)(Psysteme)) NULL);
		
	feasible_p = feasible_p && !SC_EMPTY_P(ps = check_feasibility(ps));
		
	if_debug_sc_strong_normalize_and_check_feasibility2 {
	  fprintf(stderr,
		  "Print the three systems after inequality normalization:\n");
	  fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: Input system %p\n",
		  ps);
	  sc_dump(ps);
	  fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: New system %p\n",
		  new_ps);
	  sc_dump(new_ps);
	}
      }
	    
      if(!feasible_p) {
	sc_rm(new_ps);
	new_ps = SC_EMPTY;
      }
      else {
	base_rm(sc_base(new_ps));
	sc_base(new_ps) = base_dup(sc_base(ps));
	sc_dimension(new_ps) = sc_dimension(ps);
	/* copy projected inequalities left in ps */
	new_ps = sc_safe_append(new_ps, ps);
	/* sc_base(ps) = BASE_UNDEFINED; */
	if (!sc_weak_consistent_p(new_ps)) 
	{ 
	  fprintf(stderr, 
		  "[sc_strong_normalize_and_check_feasibility2]: "
		  "Input system %p\n", ps);
	  sc_dump(ps);
	  fprintf(stderr, 
		  "[sc_strong_normalize_and_check_feasibility2]: "
		  "New system %p\n", new_ps);
	  sc_dump(new_ps);
	  /* assert(sc_weak_consistent_p(new_ps)); */
	  assert(FALSE);
	}
      }
	    
      sc_rm(ps);
      sc_rm(ps_backup);
      if_debug_sc_strong_normalize_and_check_feasibility2 
	{
	  fprintf(stderr,
		  "[sc_strong_normalize_and_check_feasibility2]: Final value of new system %p:\n",
		  new_ps);
	  sc_dump(new_ps);
	  fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: End\n");
	}
	    
      UNCATCH(overflow_error);
    }
  return new_ps;
}
