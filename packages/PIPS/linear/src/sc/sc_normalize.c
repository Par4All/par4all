/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

  This file is part of PIPS.

  PIPS is free software: you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  any later version.

  PIPS is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.

  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with PIPS.  If not, see <http://www.gnu.org/licenses/>.

*/

 /* package sc
  *
  * Normalization, which include some redundacy elimination
 */

#ifdef HAVE_CONFIG_H
    #include "config.h"
#endif

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
/* #include <values.h> */
#include <limits.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Bounded normalization relies on numerical bounds l and b for vector
 * x, such that l<=x<=b.
 *
 * However, lower and upper bounds are not always available for all
 * components x_i of x. And 0 is a perfectly valid bound. Hence, the
 * bound information cannot be carried by only two vectors. Four
 * vectors are used, two basis vectors, lb and ub, used to specifiy if
 * a bound is available, and two vectors l and b to contain the bounds.
 *
 * Note that l and u are not usual vectors. The constant bounds appear
 * as variable coefficients. For instance, l=2u+3v, together with
 * lb=u+v+w, means that the lower bounds for x_u, x_v and x_w are 2, 3
 * and 0.
 *
 * Note also that it might be useful to separate the
 * sc_bounded_normalization into two functions, one to create the
 * bounding box and one to eliminate constraints according to the
 * bounding box. For instance, when a system is projected along
 * several dimensions, the bounding box could be computed initially
 * and reused at each projection stage. The advantage is to preserve
 * information that may be killed by a redundancy test. The
 * inconvenient is not to take advantage of the newly created
 * constraints. The best might be to update the bouding box at each
 * stage.
 *
 * FI: I do not want to introduce a new data structure to represent a
 * bounding box, and I do not like the idea of keeping independently
 * four vectors.
 */

/* Auxiliary function for sc_bounded_normalization()
 *
 * Update bound information for variable var. A lower or upper bound_p
 * and bound_base_p must be passer regardless of lower_p. lower_p is
 * only used to know how to tighten the bound if it is already
 * defined. The information to use is:
 *
 * lower_p? var <= nb : var>=nb
 *
 * It is not possible to detect an empty system here
 */
static void update_lower_or_upper_bound(Pvecteur * bound_p,
					Pvecteur * bound_base_p,
					Variable var,
					Value nb, /* new bound */
					bool lower_p)
{
  if(vect_coeff(var, *bound_base_p)!=0) {
    /* A bound is already known. It must be updated if the new one is better. */
    Value pb = vect_coeff(var, *bound_p); /* previous bound */
    if(lower_p) { /* bigger is better */
      if(nb>pb) { /* update bound */
	vect_add_elem(bound_p, var, value_minus(nb,pb));
      }
      else {
	/* The current constraint is redundant */
	;
      }
    }
    else { /* smaller is better */
      if(nb<pb) { /* update bound */
	vect_add_elem(bound_p, var, value_minus(nb,pb));
      }
      else {
	/* The current constraint is redundant */
	;
      }
    }
  }
  else {
    /* No bound is known yet */
    base_add_dimension(bound_base_p, var);
    vect_add_elem(bound_p, var, nb);
  }
}

/* Updates upper and lower bounds, (ubound_p, ubound_base_p) and
 * (lbound_p, lbound_base_p), with equations var==nb
 *
 * Do not test for nb==0. This would increase the code size a lot for
 * a very limited benefit.
 *
 * This function returns a boolean to signal an empy interval,
 * i.e. a non-feasible system.
 *
 * Auxiliary function of sc_bounded_normalization()
 */
static bool update_lower_and_upper_bounds(Pvecteur * ubound_p,
					  Pvecteur * ubound_base_p,
					  Pvecteur * lbound_p,
					  Pvecteur * lbound_base_p,
					  Variable var,
					  Value nb) /* new bound */
{
  bool empty_p = false;

  if(var==TCST) {
    /* Some constraint like 0==0, i.e. the NULL vector, or 0==3 an impossible constraint */
    // abort();
    if(!value_zero_p(nb))
      empty_p = true;
  }
  else {
    /* Update the upper bound */
    if(vect_coeff(var,*ubound_base_p)!=0) {
      /* A pre-existing upper bound exists */
      Value ob = vect_coeff(var, *ubound_p);
	if(value_gt(ob, nb)) {
	  /* The new bound nb is stricter and consistent with the preexisting upper bound */
	  vect_add_elem(ubound_p, var, value_minus(nb,ob));
	}
	else if(value_eq(ob,nb))
	  ;
	else {
	  empty_p = true;
	  //abort();
	  ; /* ignore the non feasability */
	  /* but maintain consistency to avoid a later abort */
	  //vect_add_elem(ubound_p, var, value_minus(nb,ob));
	}
    }
    else {
      base_add_dimension(ubound_base_p, var);
      vect_add_elem(ubound_p, var, nb);
    }

    /* Update the lower bound, almost identical, but for the compatibility check */
    if(vect_coeff(var,*lbound_base_p)!=0) {
      /* A lower bound has already been defined */
      Value ob = vect_coeff(var, *lbound_p);
	if(value_lt(ob, nb)) {
	  /* The new bound nb is stricter and consistent with the preexisting lower bound */
	  vect_add_elem(lbound_p, var, value_minus(nb,ob));
	}
	else if(value_eq(ob,nb))
	  ;
	else {
	  empty_p = true;
	  /* ps is empty with contradictory equations and inequalities
	     supposedly trapped earlier */
	  // abort();
	  ; /* ignore the non feasability */
	  /* but maintain consistency to avoid a later abort */
	  //vect_add_elem(lbound_p, var, value_minus(nb,ob));
	}
    }
    else {
      base_add_dimension(lbound_base_p, var);
      vect_add_elem(lbound_p, var, nb);
    }
    if(!empty_p && vect_coeff(var, *lbound_p)!=vect_coeff(var,*ubound_p))
      abort();
  }
  return empty_p;
}

/* Eliminate trivially redundant integer constraint using a O(n x d^2)
 * algorithm, where n is the number of constraints and d the
 * dimension. And possibly detect an non feasible constraint system
 * ps.
 *
 * This function must not be used to decide emptyness when checking
 * redundancy with Fourier-Motzkin because this may increase the
 * initial rational convex polyhedron. No integer point is added, but
 * rational points may be added, which may lead to an accuracy loss
 * when a convex hull is performed.
 *
 * Principle: If two constant vectors, l et u, such that l<=x<=u,
 * where x is the vector representing all variables, then the bound b
 * of any constraint a.x<=b can be compared to a.k where k_i=u_i if
 * a_i is positive, and k_i=l_i elsewhere. The constraint can be
 * eliminated if a.k<=b.
 *
 * It is not necessary to have upper and lower bounds for all
 * components of x to compute the redundancy condition. It is
 * sufficient to meet the condition:
 *
 *  forall i s.t. a_i>0 \exists u_i and forall i s.t. a_i<0 \exists b_i
 *
 * The complexity is O(n x d) where n is the number of constraints and
 * d the dimension, vs O(n^2 x d^2) for some other normalization
 * functions.
 *
 * The normalization is not rational. It assumes that only integer
 * points are relevant. For instance, 2x<=3 is reduced to x<=1. It
 * would be possible to use the same function in rational, but the
 * divisions should be removed, the bounding box would require six
 * vectors, with two new vectors used to keep the variable
 * coefficients, here 2, and the comparisons should be replaced by
 * rational comparisons. Quite a lot of changes, although the general
 * structure would stay the same.
 *
 * This function was developped to cope successfully with
 * Semantics/type03. The projection performed in
 * transformer_intra_to_inter() explodes without this function.
 *
 * Note that the upper and lower bounds, u and l, are stored in Pvecteur in an
 * unusual way. The coefficients are used to store the constants.
 *
 * Note also that constant terms are stored in the lhs in linear, but
 * they are computed here as rhs. In other words, x - 2 <=0 is the
 * internal storage, but the upper bound is 2 as in x<=2.
 *
 * Note that the simple inequalities used to compute the bounding box
 * cannot be eliminated. Hence, their exact copies are also
 * preserved. Another redundancy test is necessary to get rid of
 * them. They could be eliminated when computing the bounding box if
 * the function updating the bounds returned the information. Note
 * that non feasability cannot be checked because the opposite bound
 * vectors are not passed. If they were, then no simple return code
 * would do.
 *
 * The same is true for equations. But the function updating the
 * bounds cannot return directly two pieces of information: the
 * constraint system is empty or the constraint is redundant.
 *
 * This function could be renamed sc_bounded_redundancy_elimination()
 * and be placed into one of the two files, sc_elim_redund.c or
 * sc_elim_simple_redund.c. It just happened that redundancy
 * elimination uses normalization and gdb led me to
 * sc_normalization() when I tried to debug Semantics/type03.c.
 *
 * This function could be split into three functions, one to compute
 * the bounding box, one to simplify a constraint system according to
 * a bounding box and the third one calling those two. This could be
 * useful when projecting a system because the initial bounding box
 * remains valid all along the projection, no matter that happens to
 * the initial constraint. However, the bounding box might be improved
 * with the projections... Since the bounding box computation is fast
 * enough, the function was not split and the transformer (see PIPS)
 * projection uses it at each stage.
 *
 * However, I could not think of a simple data structure to store the
 * bounding box as a unique object. With more variables, it would be
 * better to use hash tables to link variables to their lower and
 * upper bound and to their value when they are constant. Hence the
 * use of four sparse vectors as explained above.
 */
Psysteme sc_bounded_normalization(Psysteme ps)
{
  /* Compute the trivial upper and lower bounds for the systeme
     basis. Since we cannot distinguish between "exist" and "0", we
     need two extra basis to know if the bound exists or not */
  //Pbase b = sc_base(ps);
  Pvecteur u = VECTEUR_NUL;
  Pvecteur l = VECTEUR_NUL;
  Pbase ub = BASE_NULLE;
  Pbase lb = BASE_NULLE;
  Pcontrainte eq = CONTRAINTE_UNDEFINED; /* can be an equation or an inequality */
  bool empty_p = false;

  /* First look for bounds in equalities, although they may have been
     exploited otherwise */
  for(eq = sc_egalites(ps); !CONTRAINTE_UNDEFINED_P(eq); eq = contrainte_succ(eq)) {
    Pvecteur v = contrainte_vecteur(eq);
    int n = vect_size(v);
    Value k;
    if(n==1) {
      Variable var = vecteur_var(v);
      update_lower_and_upper_bounds(&u, &ub, &l, &lb, var, VALUE_ZERO);
    }
    else if(n==2 && (k=vect_coeff(TCST,v))!=0) {
      Variable var = vecteur_var(v);
      Value c = vecteur_val(v);

      if(var==TCST) {
	Pvecteur vn = vecteur_succ(v);
	var = vecteur_var(vn);
	c = vecteur_val(vn);
      }

      /* FI: I do not trust the modulo operator */
      if(c<0) {
	c = -c;
	k = -k;
      }

      Value r = modulo(k,c);
      if(r==0) {
	Value b_var = -value_div(k,c);
	empty_p = update_lower_and_upper_bounds(&u, &ub, &l, &lb, var, b_var);
      }
      else {
	/* ps is empty with two contradictory equations supposedly trapped earlier */
	empty_p = true;;
      }
    }
  }

  if(!empty_p) {
    /* Secondly look for bounds in inequalities */
    for(eq=sc_inegalites(ps); !CONTRAINTE_UNDEFINED_P(eq); eq = contrainte_succ(eq)) {
      Pvecteur v = contrainte_vecteur(eq);
      int n = vect_size(v);
      Value k = VALUE_ZERO;
      if(n==1) {
	Variable var = vecteur_var(v);
	if(var!=TCST) {
	  /* The variable is bounded by zero */
	  Value c = vecteur_val(v);
	  if(c>0) /* upper bound */
	    update_lower_or_upper_bound(&u, &ub, var, VALUE_ZERO, c>0);
	  else /* lower bound */
	    update_lower_or_upper_bound(&l, &lb, var, VALUE_ZERO, c>0);
	}
      }
      else if(n==2 && (k=vect_coeff(TCST,v))!=0) {
	Variable var = vecteur_var(v);
	Value c = vecteur_val(v);
	if(var==TCST) {
	  Pvecteur vn = vecteur_succ(v);
	  var = vecteur_var(vn);
	  c = vecteur_val(vn);
	}
	/* FI: I not too sure how div and pdiv operate nor on what I
	   need here... This explains why the divisions are replicated
	   after the test on the sign of the coefficient. */
	if(value_pos_p(c)) { /* upper bound */
	  Value b_var = -value_pdiv(k,c);
	  update_lower_or_upper_bound(&u, &ub, var, b_var, value_pos_p(c));
	}
	else { /* lower bound */
	  Value b_var = -value_pdiv(k,c);
	  update_lower_or_upper_bound(&l, &lb, var, b_var, value_pos_p(c));
	}
      }
    }

    /* Check that the bounding box is not empty because a lower bound
       is strictly greater than the corresponding upper bound. This
       could be checked above each time a new bound is defined or
       redefined. */
    Pvecteur vc;
    for(vc=ub; !VECTEUR_UNDEFINED_P(vc) && !empty_p; vc=vecteur_succ(vc)) {
      Variable var = vecteur_var(vc);
      Value upper = vect_coeff(var, u);
      if(value_notzero_p(vect_coeff(var, lb))) {
	Value lower = vect_coeff(var, l);
	if(lower>upper)
	  empty_p = true;
      }
    }

    /* The upper and lower bounds should be printed here for debug */
    if(false && (!VECTEUR_NUL_P(ub) || !VECTEUR_NUL_P(lb)) ) {
      if(empty_p) {
	fprintf(stderr, "sc_bounded_normalization: empty bounding box\n");
      }
      else {
	fprintf(stderr, "sc_bounded_normalization: base for upper bound and upper bound:\n");
	vect_dump(ub);
	vect_dump(u);
	fprintf(stderr, "sc_bounded_normalization: base for lower bound and lower bound:\n");
	vect_dump(lb);
	vect_dump(l);
	fprintf(stderr, "sc_bounded_normalization: constraints found:\n");
	/* Impression par intervalle, avec verification l<=u quand l et u
	   sont tous les deux disponibles */
	Pvecteur vc;
	for(vc=ub; !VECTEUR_UNDEFINED_P(vc); vc=vecteur_succ(vc)) {
	  Variable var = vecteur_var(vc);
	  Value upper = vect_coeff(var, u);
	  if(vect_coeff(var, lb)!=0) {
	    Value lower = vect_coeff(var, l);
	    if(lower<upper)
	      fprintf(stderr, "%d <= %s <= %d\n", (int) lower, variable_dump_name(var),
		      (int) upper);
	    else if(lower==upper)
	      fprintf(stderr, "%s == %d\n", variable_dump_name(var), (int) upper);
	    else /* lower>upper */
	      abort(); // should have been filtered above
	  }
	  else {
	    fprintf(stderr, "%s <= %d\n", variable_dump_name(var), (int) upper);
	  }
	}
	for(vc=lb; !VECTEUR_UNDEFINED_P(vc); vc=vecteur_succ(vc)) {
	  Variable var = vecteur_var(vc);
	  Value lower = vect_coeff(var, l);
	  if(vect_coeff(var, ub)==0) {
	    fprintf(stderr, "%d <= %s \n", (int) lower, variable_dump_name(var));
	  }
	}
      }
    }

    if(!empty_p) {
      /* Check inequalities for redundancy with respect to ub and lb, if
	 ub and lb contain a minum of information */
      if(base_dimension(ub)+base_dimension(lb) >=1) {
	for(eq=sc_inegalites(ps);
	    !CONTRAINTE_UNDEFINED_P(eq) && !empty_p; eq = contrainte_succ(eq)) {
	  Pvecteur v = contrainte_vecteur(eq);
	  int n = vect_size(v);
	  Pvecteur vc;
	  Value nlb = VALUE_ZERO; /* new lower bound: useful to check feasiblity */
	  Value nub = VALUE_ZERO; /* new upper bound */
	  Value ob = VALUE_ZERO; /* old bound */
	  bool lb_failed_p = false; /* a lower bound cannot be estimated */
	  bool ub_failed_p = false; /* the bounding box does not contain
				    enough information to check
				    redundancy with the upper bound */
	  /* Try to compute the bounds nub and nlb implied by the bounding box */
	  /* We should not stop on double failure but go to the end
	     and substitute constant variables on the fiable,
	     computing a delta value to be added to the constant term
	     and setting the variable coefficient to zero; we should
	     have not only lb and ub but also cb, the base for the
	     constant terms */
	  for(vc=v; !VECTEUR_NUL_P(vc) && !(ub_failed_p&&lb_failed_p);
	      vc = vecteur_succ(vc)) {
	    Variable var = vecteur_var(vc);
	    Value c = vecteur_val(vc);
	    if(var==TCST) { /* I assume the constraint to be consistent
			       with only one coefficient per dimension */
	      value_assign(ob, value_uminus(c)); /* move the bound to the
						    right hand side of the
						    inequality */
	    }
	    /* We could check here if var has a constant value */
	    else {
	      if(value_pos_p(c)) { /* an upper bound of var is needed */
		if(vect_coeff(var, ub)!=0) { /* the value macro should be used */
		  Value ub_var = vect_coeff(var, u);
		  value_addto(nub, value_mult(c, ub_var));
		}
		else {
		  ub_failed_p = true;
		}
		if(vect_coeff(var, lb)!=0) {
		  Value lb_var = vect_coeff(var, l);
		  value_addto(nlb, value_mult(c, lb_var));
		  //fprintf(stderr, "c=%lld, lb_var=%lld, nlb=%d\n", c, lb_var, nlb);
		}
		else {
		  lb_failed_p = true;
		}
	      }
	      else { /* c<0 : a lower bound is needed for var*/
		if(vect_coeff(var, lb)!=0) {
		  Value lb_var = vect_coeff(var, l);
		  value_addto(nub, value_mult(c, lb_var));
		}
		else {
		  ub_failed_p = true;
		}
		if(vect_coeff(var, ub)!=0) {
		  Value ub_var = vect_coeff(var, u);
		  value_addto(nlb, value_mult(c, ub_var));
		  //fprintf(stderr, "c=%lld, ub_var=%lld, nlb=%d\n", c, ub_var, nlb);
		}
		else {
		  lb_failed_p = true;
		}
	      }
	    }
	  }
	  /* We should update here the constant term with delta, the
	     value accumulated with the constant terms */

	  /* If the new bound nub is tighter, nub <= ob, ob-nub >= 0 */
	  if(!ub_failed_p) {
	    /* Do not destroy the constraints defining the bounding box */
	    bool posz_p = value_posz_p(value_minus(ob,nub));
	    bool pos_p = value_pos_p(value_minus(ob,nub));
	    if( (n>2 && posz_p)
		|| (n==2 && value_zero_p(vect_coeff(TCST, v)) && posz_p)
		|| (n==2 && value_notzero_p(vect_coeff(TCST, v)) && pos_p)
		|| (n==1 && value_zero_p(vect_coeff(TCST, v)) && pos_p)) {
	      /* The constraint eq is redundant */
	      if(false) {
		fprintf(stderr, "Redundant constraint:\n");
		vect_dump(v);
	      }
	      eq_set_vect_nul(eq);
	    }
	    else { /* Preserve this constraint */
	      ;
	    }
	  }

	  /* The new estimated lower bound must be less than the upper
	     bound of the constraint, or the system is empty.*/
	  if(!lb_failed_p) { /* the estimated lower bound, nlb, must
				be less or equal to the effective
				upper bound ob: nlb <= ob, 0<=ob-nlb */
	    /* if ob==nlb, an equation has been detected but it is
	       hard to exploit here */
	    bool pos_p = value_posz_p(value_minus(ob,nlb));
	    if(!pos_p) /* to have a nice breakpoint location */
	      empty_p = true;
	  }
	}
      }
    }
  }

  if(empty_p) {
    Psysteme ns = sc_empty(sc_base(ps));
    sc_base(ps) = BASE_NULLE;
    sc_rm(ps);
    ps = ns;
  }
  return ps;
}

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
 * Il manque une elimination de redondance particuliere pour traiter
 * les booleens. Si on a deux vecteurs constants, l et u, tels que
 * l<=x<=u, alors la borne b de n'importe quelle contrainte a.x<=b
 * peut etre comparee a a.k ou k_i=u_i si a_i est positif et k_i=l_i
 * sinon. La contrainte a peut etre eliminee si a.k <= b. Si des
 * composantes de x n'aparaissent dans aucune contrainte, on ne
 * dispose en consequence pas de bornes, mais ca n'a aucune
 * importance. On veut donc avoir une condition pour chaque
 * contrainte: forall i s.t. a.i!=0 \exists l_i and b_i
 * s.t. l_i<=x_i<=b_i. Voir sc_bounded_normalization().
 *
 * sc_normalize retourne NULL quand la normalisation a montre que le systeme
 * etait non faisable
 *
 * BC: now returns sc_empty when not feasible to avoid unecessary copy_base
 * in caller.
 * 
 * FI: should check the input for sc_empty_p()
 */
Psysteme sc_normalize(Psysteme ps)
{
    Pcontrainte eq;
    bool is_sc_fais = true;

    /* I do not want to disturb every pass of PIPS that uses directly
     * or indirectly sc_normalize. Since sc_bounded_normalization()
     * has been developped specifically for transformer_projection(),
     * it is called directly from the function implementing
     * transformer_projection(). But it is not sufficient for type03.
     */
    static int francois=1;
    if(francois==1)
      ps = sc_bounded_normalization(ps);

    /* FI: this takes (a lot of) time but I do not know why: O(n^2)? */
    ps = sc_safe_kill_db_eg(ps);

    if (ps && !sc_empty_p(ps) && !sc_rn_p(ps)) {
	for (eq = ps->egalites;
	     (eq != NULL) && is_sc_fais;
	     eq=eq->succ) {
	    /* normalisation de chaque equation */
	    if (eq->vecteur)    {
		vect_normalize(eq->vecteur);
		if ((is_sc_fais = egalite_normalize(eq))== true)
		    is_sc_fais = sc_elim_simple_redund_with_eq(ps,eq);
	    }
	}
	for (eq = ps->inegalites;
	     (eq!=NULL) && is_sc_fais;
	     eq=eq->succ) {
	    if (eq->vecteur)    {
		vect_normalize(eq->vecteur);
		if ((is_sc_fais = inegalite_normalize(eq))== true)
		    is_sc_fais = sc_elim_simple_redund_with_ineq(ps,eq);
	    }
	}

	ps = sc_safe_kill_db_eg(ps);
	if (!sc_empty_p(ps))
	  {
	    sc_elim_empty_constraints(ps, true);
	    sc_elim_empty_constraints(ps, false);
	  }
    }

    if (!is_sc_fais)
      {
	/* FI: piece of code that will appear in many places...  How
	 * can we call it: empty_sc()? sc_make_empty()? sc_to_empty()?
	 */
	Psysteme new_ps = sc_empty(sc_base(ps));
	sc_base(ps) = BASE_UNDEFINED;
	sc_rm(ps);
	ps = new_ps;
      }
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
 * Verification de la non redondance de chaque contrainte avec les autres.
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
  sc_elim_empty_constraints(ps, true);
  sc_elim_empty_constraints(ps, false);

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
	   || value_mod(vect_coeff(TCST,veq), a)==VALUE_ZERO) {
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
    sc_elim_empty_constraints(ps, true);
    sc_elim_empty_constraints(ps, false);
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

	sc_rm_empty_constraints(ps, true);
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

	sc_rm_empty_constraints(ps, false);
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

  ps = sc_normalize(ps);
  return(ps);
}

static Psysteme sc_rational_feasibility(Psysteme sc)
{

    if(!sc_rational_feasibility_ofl_ctrl((sc), OFL_CTRL,true)) {
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

    Psysteme volatile new_ps = SC_UNDEFINED;
    Psysteme proj_ps = SC_UNDEFINED;
    bool feasible_p = true;
    /* Automatic variables read in a CATCH block need to be declared volatile as
     * specified by the documentation*/
    Psysteme volatile ps_backup = sc_copy(ps);
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
		    new_ps = sc_copy(ps);
		    proj_ps = sc_copy(new_ps);
		    sc_rm(new_ps);
		    new_ps = sc_make(NULL, NULL);
		    
		    /*
		      fprintf(stderr, "[sc_strong_normalize]: Input system %x\n",
		      (unsigned int) ps);
		      sc_default_dump(ps);
		      fprintf(stderr, "[sc_strong_normalize]: Copy system %x\n",
		      (unsigned int) ps);
		      sc_default_dump(proj_ps);
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
				
				new_eq = contrainte_copy(eq);
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
				  feasible_p = false;
				  break;
				  }
				  else {
				  ;
				  }
				*/
				
				/* proj_eq itself is going to be modified in proj_ps.
				 * use a copy!
				 */
				def = contrainte_copy(proj_eq);
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
			    feasible_p = false;
			    break;
			}
			
			/*
			  fprintf(stderr,
			  "Print the three systems at each elimination step:\n");
			  fprintf(stderr, "[sc_strong_normalize]: Input system %x\n",
			  (unsigned int) ps);
			  sc_default_dump(ps);
			  fprintf(stderr, "[sc_strong_normalize]: Copy system %x\n",
			  (unsigned int) proj_ps);
			  sc_default_dump(proj_ps);
			  fprintf(stderr, "[sc_strong_normalize]: New system %x\n",
			  (unsigned int) new_ps);
			  sc_default_dump(new_ps);
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
			       && contrainte_verifiee(proj_ineq, false)) {
				/* ineq is redundant */
				;
			    }
			    else {
				int i;
				i = sc_check_inequality_redundancy(proj_ineq, proj_ps);
				if(i==0) {
				    /* keep ineq */
				    new_ineq = contrainte_copy(ineq);
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
				    feasible_p = false;
				    break;
				}
				else {
				    assert(false);
				}
			    }
			}
			else {
			    /* The system is not feasible. Stop */
			    feasible_p = false;
			    break;
			}
		    }
		    
		    /*
		      fprintf(stderr,
		      "Print the three systems after inequality normalization:\n");
		      fprintf(stderr, "[sc_strong_normalize]: Input system %x\n",
		      (unsigned int) ps);
		      sc_default_dump(ps);
		      fprintf(stderr, "[sc_strong_normalize]: Copy system %x\n",
		      (unsigned int) proj_ps);
		      sc_default_dump(proj_ps);
		      fprintf(stderr, "[sc_strong_normalize]: New system %x\n",
		      (unsigned int) new_ps);
		      sc_default_dump(new_ps);
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
		    feasible_p = false;
		}
	    }
	    else {
		/*
		  fprintf(stderr,
		  "[sc_strong_normalize]: Empty system as input\n");
		*/
		feasible_p = false;
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
		/* FI: test added for breakpoint placement*/
		if(!sc_weak_consistent_p(new_ps))
		  assert(sc_weak_consistent_p(new_ps));
	    }

	    sc_rm(proj_ps);
	    sc_rm(ps);
	    sc_rm(ps_backup);
	    /*
	      fprintf(stderr, "[sc_strong_normalize]: Final value of new system %x:\n",
	      (unsigned int) new_ps);
	      sc_default_dump(new_ps);
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

#define if_debug_sc_strong_normalize_2 if(false)

    Psysteme new_ps = sc_make(NULL, NULL);
    bool feasible_p = true;

    /* Automatic variables read in a CATCH block need to be declared volatile as
     * specified by the documentation*/
    Psysteme volatile ps_backup = sc_copy(ps);

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
			sc_default_dump(ps);
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
				new_eq = contrainte_copy(eq);
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
				  feasible_p = false;
				  break;
				  }
				  else {
				  ;
				  }
				*/
				
				/* eq itself is going to be modified in ps.
				 * use a copy!
				 */
				def = contrainte_copy(eq);
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
			    feasible_p = false;
			    break;
			}
			
			if_debug_sc_strong_normalize_2 {
			    fprintf(stderr,
				    "Print the two systems at each elimination step:\n");
			    fprintf(stderr, "[sc_strong_normalize2]: Input system %p\n",
				    ps);
			    sc_default_dump(ps);
			    fprintf(stderr, "[sc_strong_normalize2]: New system %p\n",
				    new_ps);
			    sc_default_dump(new_ps);
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
			sc_default_dump(ps);
			fprintf(stderr, "[sc_strong_normalize2]: New system %p\n",
				new_ps);
			sc_default_dump(new_ps);
		    }
		}
		else {
		    if_debug_sc_strong_normalize_2 {
			fprintf(stderr,
				"[sc_strong_normalize2]:"
				" Non-feasibility detected by first call to sc_normalize\n");
		    }
		    feasible_p = false;
		}
	    }
	    else {
		if_debug_sc_strong_normalize_2 {
		    fprintf(stderr,
			    "[sc_strong_normalize2]: Empty system as input\n");
		}
		feasible_p = false;
	    }
	    
	    if(!feasible_p) {
		sc_rm(new_ps);
		new_ps = SC_EMPTY;
	    }
	    else {
		base_rm(sc_base(new_ps));
		sc_base(new_ps) = base_copy(sc_base(ps));
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
		sc_default_dump(new_ps);
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

#define if_debug_sc_strong_normalize_and_check_feasibility2 if(false)

  Psysteme new_ps = sc_make(NULL, NULL);
  bool feasible_p = true;
  /* Automatic variables read in a CATCH block need to be declared volatile as
   * specified by the documentation*/
  Psysteme volatile ps_backup = sc_copy(ps);

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
	sc_default_dump(ps);
      }
	    
      if(SC_UNDEFINED_P(ps)) {
	if_debug_sc_strong_normalize_and_check_feasibility2 {
	  fprintf(stderr,
		  "[sc_strong_normalize_and_check_feasibility2]"
		  " Empty system as input\n");
	}
	feasible_p = false;
      }
      else if(SC_EMPTY_P(ps = sc_normalize(ps))) {
	if_debug_sc_strong_normalize_and_check_feasibility2 {
	  fprintf(stderr,
		  "[sc_strong_normalize_and_check_feasibility2]:"
		  " Non-feasibility detected by first call to sc_normalize\n");
	}
	feasible_p = false;
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
	  sc_default_dump(ps);
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
		  new_eq = contrainte_copy(eq);
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
		  def = contrainte_copy(eq);
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
	      feasible_p = false;
	      break;
	    }
			
	    /* This reaaly generates a lot of about on real life system! */
	    /*
	      if_debug_sc_strong_normalize_and_check_feasibility2 {
	      fprintf(stderr,
	      "Print the two systems at each elimination step:\n");
	      fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: Input system %x\n",
	      (unsigned int) ps);
	      sc_default_dump(ps);
	      fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: New system %x\n",
	      (unsigned int) new_ps);
	      sc_default_dump(new_ps);
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
	    sc_default_dump(ps);
	    fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: New system %p\n",
		    new_ps);
	    sc_default_dump(new_ps);
	  }
	}
	sc_elim_empty_constraints(new_ps,true);
	sc_elim_empty_constraints(ps,true);
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
	  sc_default_dump(ps);
	  fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: New system %p\n",
		  new_ps);
	  sc_default_dump(new_ps);
	}
      }
	    
      if(!feasible_p) {
	sc_rm(new_ps);
	new_ps = SC_EMPTY;
      }
      else {
	base_rm(sc_base(new_ps));
	sc_base(new_ps) = base_copy(sc_base(ps));
	sc_dimension(new_ps) = sc_dimension(ps);
	/* copy projected inequalities left in ps */
	new_ps = sc_safe_append(new_ps, ps);
	/* sc_base(ps) = BASE_UNDEFINED; */
	if (!sc_weak_consistent_p(new_ps)) 
	{ 
	  fprintf(stderr, 
		  "[sc_strong_normalize_and_check_feasibility2]: "
		  "Input system %p\n", ps);
	  sc_default_dump(ps);
	  fprintf(stderr, 
		  "[sc_strong_normalize_and_check_feasibility2]: "
		  "New system %p\n", new_ps);
	  sc_default_dump(new_ps);
	  /* assert(sc_weak_consistent_p(new_ps)); */
	  assert(false);
	}
      }
	    
      sc_rm(ps);
      sc_rm(ps_backup);
      if_debug_sc_strong_normalize_and_check_feasibility2 
	{
	  fprintf(stderr,
		  "[sc_strong_normalize_and_check_feasibility2]: Final value of new system %p:\n",
		  new_ps);
	  sc_default_dump(new_ps);
	  fprintf(stderr, "[sc_strong_normalize_and_check_feasibility2]: End\n");
	}
	    
      UNCATCH(overflow_error);
    }
  return new_ps;
}
