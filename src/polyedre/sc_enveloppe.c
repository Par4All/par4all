 /* package polyedre: enveloppe convexe de deux systemes lineaires
  *
  * Ce module est range dans le package polyedre bien qu'il soit utilisable
  * en n'utilisant que des systemes lineaires (package sc) parce qu'il
  * utilise lui-meme des routines sur les polyedres.
  *
  * Francois Irigoin, Janvier 1990
  */

#include <stdio.h>
#include <stdlib.h>

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

/* Psysteme sc_enveloppe(s1, s2): calcul d'une representation par systeme
 * lineaire de l'enveloppe convexe des polyedres definis par les systemes
 * lineaires s1 et s2
 *
 * s = enveloppe(s1, s2);
 * return s;
 *
 * s1 et s2 ne sont pas modifies. Ils doivent tous les deux avoir au moins
 * une base.
 *
 * Il faudrait traiter proprement les cas particuliers SC_RN et SC_EMPTY
 */
/*
Psysteme sc_enveloppe(s1, s2)
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

    s1 = sc_dup(s1);
    s2 = sc_dup(s2);

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
	p1 = sc_to_poly(s1);
	p2 = sc_to_poly(s2);

	p = env(p1, p2);

	s = p->sc;
	p->sc = SC_UNDEFINED;
	poly_rm(p);
    }

    return s;
}
*/


/* Psysteme sc_enveloppe_chernikova_ofl_ctrl(Psysteme s1, s2, int ofl_ctrl)
 * input    : 
 * output   : 
 * modifies : s1 and s2 are NOT modified.
 * comment  : s1 and s2 must have a basis.
 * 
 * s = enveloppe(s1, s2);
 * return s;
 *
 * calcul d'une representation par systeme
 * lineaire de l'enveloppe convexe des polyedres definis par les systemes
 * lineaires s1 et s2
 */
Psysteme sc_enveloppe_chernikova_ofl_ctrl(s1, s2, ofl_ctrl)
Psysteme s1;
Psysteme s2;
int ofl_ctrl;
{
    Psysteme s = SC_UNDEFINED;
    boolean catch_performed = FALSE;
    /* mem_spy_begin(); */

    assert(!SC_UNDEFINED_P(s1) && !SC_UNDEFINED_P(s2));

    switch (ofl_ctrl) 
    {
    case OFL_CTRL :
	ofl_ctrl = FWD_OFL_CTRL;
	catch_performed = TRUE;
	CATCH(overflow_error) {
	    /* 
	     *   PLEASE do not remove this warning.
	     *
	     *   BC 24/07/95
	     */
	    fprintf(stderr, "[sc_enveloppe_chernikova_ofl_ctrl] "
		    "arithmetic error occured\n" );
	    s = sc_rn(base_dup(sc_base(s1)));
	    break;
	}		
    default: {
    
	    if (SC_RN_P(s2) || sc_rn_p(s2) || sc_dimension(s2)==0
		|| sc_empty_p(s1) || !sc_faisabilite_ofl(s1)) 
	    {
		Psysteme sc2 = sc_dup(s2);
		sc2 = sc_elim_redond(sc2);
		s = SC_UNDEFINED_P(sc2)? sc_empty(base_dup(sc_base(s2))): 
		    sc2;
	    }
	    else 
		if (SC_RN_P(s1) ||sc_rn_p(s1) || sc_dimension(s1)==0   
		    || sc_empty_p(s2) || !sc_faisabilite_ofl(s2)) 
		{
		    Psysteme sc1 = sc_dup(s1);
		    sc1 = sc_elim_redond(sc1);
		    s = SC_UNDEFINED_P(sc1)? 
			sc_empty(base_dup(sc_base(s1))): sc1;
		}
		else 
		{
		    /* calcul de l'enveloppe convexe */
		    s = sc_convex_hull(s1,s2);
		    /* printf("systeme final \n"); sc_dump(s);  */
		}
	    if (catch_performed)
		UNCATCH(overflow_error);
	}
    }
    /* mem_spy_end("sc_enveloppe_chernikova_ofl_ctrl"); */
    return s;
}

Psysteme sc_enveloppe_chernikova(s1, s2)
Psysteme s1, s2;
{
  return sc_enveloppe_chernikova_ofl_ctrl((s1), (s2), OFL_CTRL);
} 


/* Psysteme sc_fast_convex_hull(Psysteme s1, Psysteme s2):
 *
 * exploit constraints shared by s1 and s2 to decrease the space
 * dimension and the number of constraints before calling the
 * effective convex hull function.
 *
 * Simplified view of the algorithm (in fact equalities and inequalities
 * are treated differently):
 *
 * Let s1 and s2 be sets of constraints
 * Build s0 = intersection(s1, s2)
 * Build s1' = s1 - s0
 * Build s2' = s2 - s0
 * Compute h' = convex_hull(s1', s2')
 * Build h = intersection(h', s0)
 * Free h', s1', s2'
 * Return h
 *
 * Francois Irigoin, 9 August 1992
 *
 * Note: This algorithm is simply wrong.
 * Let s1 = {i==1, i==2} and s2 = {i==3, i==2}.
 * Constraint i==2 can be removed. 
 * The convex hull of i==1 and i==3 is 1 <= i <= 3.
 * Its intersection with i == 2 is i == 2.
 */
/*
Psysteme sc_fast_convex_hull(s1, s2)
Psysteme s1;
Psysteme s2;
{
    Psysteme s0 = sc_new();
    Psysteme s1p = sc_new();
    Psysteme s2p = sc_new();
    Psysteme hp = SC_UNDEFINED;
    Pcontrainte eq;
    Pbase b;
    int d;
    extern char * dump_value_name();

    assert(!SC_UNDEFINED_P(s1) && !SC_UNDEFINED_P(s2));

    for(eq = sc_egalites(s1); !CONTRAINTE_UNDEFINED_P(eq); eq = eq->succ) {
	if(egalite_in_liste(eq, sc_egalites(s2))) {
	    sc_add_egalite(s0, contrainte_dup(eq));
	}
	else if(contrainte_in_liste(eq, sc_inegalites(s2))) {
	    sc_add_inegalite(s0, contrainte_dup(eq));
	}
	else {
	    sc_add_egalite(s1p, contrainte_dup(eq));
	}
    }

    for(eq = sc_inegalites(s1); !CONTRAINTE_UNDEFINED_P(eq); eq = eq->succ) {
	if(contrainte_in_liste(eq, sc_inegalites(s2)) ||
	   contrainte_in_liste(eq, sc_egalites(s2))) {
	    sc_add_inegalite(s0, contrainte_dup(eq));
	}
	else {
	    sc_add_inegalite(s1p, contrainte_dup(eq));
	}
    }

    for(eq = sc_egalites(s2); !CONTRAINTE_UNDEFINED_P(eq); eq = eq->succ) {
	if(contrainte_in_liste(eq, sc_inegalites(s0)) ||
	   egalite_in_liste(eq, sc_egalites(s0))) {
	    ;
	}
	else {
	    sc_add_egalite(s2p, contrainte_dup(eq));
	}
    }

    for(eq = sc_inegalites(s2); !CONTRAINTE_UNDEFINED_P(eq); eq = eq->succ) {
	if(contrainte_in_liste(eq, sc_inegalites(s0)) ||
	   contrainte_in_liste(eq, sc_egalites(s0))) {
	    ;
	}
	else {
	    sc_add_inegalite(s2p, contrainte_dup(eq));
	}
    }

    sc_creer_base(s1p);
    sc_creer_base(s2p);
    b = base_union(sc_base(s1p), sc_base(s2p));
    d = base_dimension(b);
    vect_rm(sc_base(s1p));
    vect_rm(sc_base(s2p));
    sc_base(s1p) = base_dup(b);
    sc_dimension(s1p) = d;
    sc_base(s2p) = b;
    sc_dimension(s2p) = d;
    sc_base(s0) = base_union(sc_base(s1), sc_base(s2));

    hp = sc_enveloppe(s1p, s2p);

    hp = sc_append(hp, s0);

    sc_rm(s0);
    sc_rm(s1p);
    sc_rm(s2p);

    return hp;
}
*/

/* Psysteme sc_fast_convex_hull(Psysteme s1, Psysteme s2):
 *
 * exploit constraints shared by s1 and s2 to decrease the space
 * dimension and the number of constraints before calling the
 * effective convex hull function.
 *
 * Simplified view of the algorithm (in fact equalities and inequalities
 * are treated differently):
 *
 * Let s1 and s2 be sets of constraints
 * Build s0 = intersection(s1, s2)
 * Build s1' = s1 - s0
 * Build s2' = s2 - s0
 * Compute h' = convex_hull(s1', s2')
 * Build h = intersection(h', s0)
 * Free h', s1', s2'
 * Return h
 *
 * Francois Irigoin, 9 August 1992
 *
 * Note: as false as above. 
 */
/*
Psysteme sc_fast_enveloppe_chernikova_ofl_ctrl(s1, s2, ofl_ctrl)
Psysteme s1;
Psysteme s2;
int ofl_ctrl;
{
    Psysteme s0 = sc_new();
    Psysteme s1p = sc_new();
    Psysteme s2p = sc_new();
    Psysteme hp = SC_UNDEFINED;
    Pcontrainte eq;
    Pbase b;
    int d;
    extern char * dump_value_name();

    assert(!SC_UNDEFINED_P(s1) && !SC_UNDEFINED_P(s2));

    for(eq = sc_egalites(s1); !CONTRAINTE_UNDEFINED_P(eq); eq = eq->succ) {
	if(egalite_in_liste(eq, sc_egalites(s2))) {
	    sc_add_egalite(s0, contrainte_dup(eq));
	}
	else if(contrainte_in_liste(eq, sc_inegalites(s2))) {
	    sc_add_inegalite(s0, contrainte_dup(eq));
	}
	else {
	    sc_add_egalite(s1p, contrainte_dup(eq));
	}
    }

    for(eq = sc_inegalites(s1); !CONTRAINTE_UNDEFINED_P(eq); eq = eq->succ) {
	if(contrainte_in_liste(eq, sc_inegalites(s2)) ||
	   contrainte_in_liste(eq, sc_egalites(s2))) {
	    sc_add_inegalite(s0, contrainte_dup(eq));
	}
	else {
	    sc_add_inegalite(s1p, contrainte_dup(eq));
	}
    }

    for(eq = sc_egalites(s2); !CONTRAINTE_UNDEFINED_P(eq); eq = eq->succ) {
	if(contrainte_in_liste(eq, sc_inegalites(s0)) ||
	   egalite_in_liste(eq, sc_egalites(s0))) {
	    ;
	}
	else {
	    sc_add_egalite(s2p, contrainte_dup(eq));
	}
    }

    for(eq = sc_inegalites(s2); !CONTRAINTE_UNDEFINED_P(eq); eq = eq->succ) {
	if(contrainte_in_liste(eq, sc_inegalites(s0)) ||
	   contrainte_in_liste(eq, sc_egalites(s0))) {
	    ;
	}
	else {
	    sc_add_inegalite(s2p, contrainte_dup(eq));
	}
    }

    sc_creer_base(s1p);
    sc_creer_base(s2p);
    b = base_union(sc_base(s1p), sc_base(s2p));
    d = base_dimension(b);
    vect_rm(sc_base(s1p));
    vect_rm(sc_base(s2p));
    sc_base(s1p) = base_dup(b);
    sc_dimension(s1p) = d;
    sc_base(s2p) = b;
    sc_dimension(s2p) = d;
    sc_base(s0) = base_union(sc_base(s1), sc_base(s2));
    sc_dimension(s0) = vect_size(s0->base);

    hp = sc_enveloppe_chernikova_ofl_ctrl(s1p, s2p, ofl_ctrl);

    hp = sc_safe_append(hp, s0);

    sc_rm(s0);
    sc_rm(s1p);
    sc_rm(s2p);

    return hp;
}
*/

/* Psysteme sc_common_projection_convex_hull(Psysteme s1, Psysteme s2):
 *
 * exploits equalities shared by s1 and s2 to decrease the space
 * dimension and the number of constraints before calling the
 * effective convex hull function.
 *
 * Simplified view of the algorithm:
 *
 * Let s1 and s2 be sets of constraints, eq1 and eq2 their sets of equalities
 * Build eq0 = intersection(eq1, eq2)
 * Build s1' = projection of s1 along eq0
 * Build s2' = projection of s2 along eq0
 * Compute h' = convex_hull(s1', s2')
 * Build h = intersection(h', s0)
 * Free h', s1', s2'
 * Return h
 *
 * Francois Irigoin, 20 November 1995
 *
 * Note:  it would be better to compute the convex hull of eq1 and eq2 using 
 * an Hermite form
 *
 * Note: missing -1 coeff...
 */
Psysteme
sc_common_projection_convex_hull_with_base_ordering
(Psysteme s1, Psysteme s2, void (*sort_base_func)(Pbase *))
{
#define ifdebug if(FALSE)
    Psysteme s0 = sc_new();
    Psysteme s1p = sc_dup(s1);
    Psysteme s2p = sc_dup(s2);
    Psysteme hp = SC_UNDEFINED;
    Pcontrainte eq;
    Pbase b;
    int d;
    boolean feasible_p = TRUE;
    boolean feasible_p1 = TRUE;
    boolean feasible_p2 = TRUE;

    assert(!SC_UNDEFINED_P(s1) && !SC_UNDEFINED_P(s2));

    /* 
    ifdebug {
    (void) fprintf(stderr, "sc_common_projection_convex_hull: begin\ns1:\n");
    sc_dump(s1);
    (void) fprintf(stderr, "s2:\n");
    sc_dump(s2);
    }
    */

    for(eq = sc_egalites(s1p); !CONTRAINTE_UNDEFINED_P(eq) && feasible_p; eq = eq->succ) {
	if(egalite_in_liste(eq, sc_egalites(s2p))) {
	    if(egalite_normalize(eq)) {
		if(CONTRAINTE_NULLE_P(eq)) {
		    /* eq is redundant in s1, ignore it */
		    ;
		}
		else {
		    Pcontrainte def = CONTRAINTE_UNDEFINED;
		    Pcontrainte new_eq = CONTRAINTE_UNDEFINED;
		    Variable v = TCST;
		    Pvecteur pv;

		    /* Keep eq in s0 */

		    new_eq = contrainte_dup(eq);
		    sc_add_egalite(s0, new_eq);

		    /* Use eq to eliminate a variable */

		    /* Let's use a variable with coefficient 1 if
		     * possible.
		     */
		    for( pv = contrainte_vecteur(eq);
			!VECTEUR_NUL_P(pv);
			pv = vecteur_succ(pv)) {
			if(!term_cst(pv)) {
			    v = vecteur_var(pv);
			    if(vecteur_val(pv)==VALUE_ONE) {
				break;
			    }
			}
		    }
		    assert(v!=TCST);

		    /* eq itself is going to be modified in proj_ps.
		     * use a copy!
		     */
		    def = contrainte_dup(eq);
		    s1p = 
			sc_simple_variable_substitution_with_eq_ofl_ctrl
			    (s1p, def, v, NO_OFL_CTRL);
		    s2p = 
			sc_simple_variable_substitution_with_eq_ofl_ctrl
			    (s2p, def, v, NO_OFL_CTRL);
		    contrainte_rm(def);
		}
	    }
	    else {
		/* The system is not feasible. Stop */
		feasible_p = FALSE;
		break;
	    }
	}
    }

    /* Keep track of the full bases for the convex hull */
    sc_base(s0) = base_union(sc_base(s1), sc_base(s2));
    sc_dimension(s0) = vect_size(s0->base);

    /* 
    ifdebug {
    (void) fprintf(stderr, "sc_common_projection_convex_hull: common equalities\ns1p:\n");
    sc_dump(s0);
    }
    */

    feasible_p1 = feasible_p && !SC_EMPTY_P(s1p = sc_normalize(s1p));
    feasible_p2= !SC_EMPTY_P(s2p = sc_normalize(s2p));

    if(feasible_p1 && feasible_p2) {
	/* Perform a convex hull */

	/* reduce the dimension of s1p and s2p as much as possible */
	base_rm(sc_base(s1p));
	sc_base(s1p) = BASE_UNDEFINED;
	base_rm(sc_base(s2p));
	sc_base(s2p) = BASE_UNDEFINED;
	sc_creer_base(s1p);
	sc_creer_base(s2p);
	b = base_union(sc_base(s1p), sc_base(s2p));
	sort_base_func(&b);
	d = base_dimension(b);
	vect_rm(sc_base(s1p));
	vect_rm(sc_base(s2p));
	sc_base(s1p) = base_dup(b);
	sc_dimension(s1p) = d;
	sc_base(s2p) = b;
	sc_dimension(s2p) = d;

	/*
	ifdebug {
	(void) fprintf(stderr, "sc_common_projection_convex_hull: new systems\ns1p:\n");
	sc_dump(s1p);
	(void) fprintf(stderr, "s2p:\n");
	sc_dump(s2p);
        }
	 */

	hp = sc_enveloppe_chernikova_ofl_ctrl(s1p, s2p, OFL_CTRL);

	/*
	ifdebug {
	(void) fprintf(stderr, "sc_common_projection_convex_hull: small enveloppe\nhp:\n");
	sc_dump(hp);
        }
	 */

	hp = sc_safe_append(hp, s0);

    }
    else if (feasible_p2) {
	/* if(sc_feasible_p(s2p)) { */
	if(sc_rational_feasibility_ofl_ctrl(s2p, OFL_CTRL, TRUE)) {
	    hp = s2p;
	    s2p = SC_UNDEFINED;
	    sc_base(hp) = sc_base(s0);
	}
	else {
	    hp = sc_empty(sc_base(s0));
	}
	sc_dimension(hp) = sc_dimension(s0);
	sc_base(s0) = BASE_UNDEFINED;
    }
    else if (feasible_p1) {
	/* if(sc_feasible_p(s1p)) { */
	if(sc_rational_feasibility_ofl_ctrl(s1p, OFL_CTRL, TRUE)) {
	    hp = s1p;
	    s1p = SC_UNDEFINED;
	    sc_base(hp) = sc_base(s0);
	}
	else {
	    hp = sc_empty(sc_base(s0));
	}
	sc_dimension(hp) = sc_dimension(s0);
	sc_base(s0) = BASE_UNDEFINED;
    }
    else {
	hp = sc_empty(sc_base(s0));
	sc_dimension(hp) = sc_dimension(s0);
	sc_base(s0) = BASE_UNDEFINED;
    }

    sc_rm(s0);
    sc_rm(s1p);
    sc_rm(s2p);

    return hp;
}

void no_base_sort(Pbase *pbase)
{
    return;
}

Psysteme sc_common_projection_convex_hull(Psysteme s1, Psysteme s2)
{
  return sc_common_projection_convex_hull_with_base_ordering
    (s1, s2, no_base_sort);
}

/* call chernikova with compatible base.
 */
static Psysteme actual_convex_union(Psysteme s1, Psysteme s2)
{
  Psysteme s;

  /* same common base! */
  Pbase b1 = sc_base(s1), b2 = sc_base(s2), bu = base_union(b1, b2);
  int d1 = sc_dimension(s1), d2 = sc_dimension(s2), du = vect_size(bu);

  sc_base(s1) = bu;
  sc_dimension(s1) = du;
  sc_base(s2) = bu;
  sc_dimension(s2) = du;

  /* call chernikova directly.
     sc_common_projection_convex_hull improvements have already been included.
  */
  s = sc_enveloppe_chernikova(s1, s2);

  /* restaure initial base */
  sc_base(s1) = b1;
  sc_dimension(s1) = d1;
  sc_base(s2) = b2;
  sc_dimension(s2) = d2;

  base_rm(bu);

  return s;
}

/* implements FC basic idea of simple fast cases...
 * returns s1 v s2. 
 * s1 and s2 are not touched.
 * The bases should be minimum for best efficiency!
 * a common base is rebuilt in actual_convex_union.
 * other fast cases may be added?
 */
Psysteme elementary_convex_union(Psysteme s1, Psysteme s2)
{
  if (sc_empty_p(s1) || sc_empty_p(s2))
    return sc_empty(base_union(sc_base(s1), sc_base(s2)));
  
  if (sc_rn_p(s1) || sc_rn_p(s2) || 
      !vect_common_variables_p(sc_base(s1), sc_base(s2)))
    return sc_rn(base_union(sc_base(s1), sc_base(s2)));

  /*
  if (sc_dimension(s1)==1 && sc_dimension(s2)==1 && 
      sc_base(s1)->var == sc_base(s2)->var)
  {
    // fast computation...
  }
  */

  return actual_convex_union(s1, s2);
}
