/* 
 * $Id$
 * 
 * This file provides functions to test the feasibility of a system 
 * of constraints. 
 *
 * Arguments of these functions :
 * 
 * - s or sc is the system of constraints.
 * - ofl_ctrl is the way overflow errors are handled
 *     ofl_ctrl == NO_OFL_CTRL
 *               -> overflow errors are not handled
 *     ofl_ctrl == OFL_CTRL
 *               -> overflow errors are handled in the called function
 *     ofl_ctrl == FWD_OFL_CTRL
 *               -> overflow errors must be handled by the calling function
 * - ofl_res is the result of the feasibility test when ofl_ctrl == OFL_CTRL
 *   and there is an overflow error.
 * - integer_p (low_level function only) is a boolean :
 *     integer_p == TRUE to test if there exists at least one integer point 
 *               in the convex polyhedron defined by the system of constraints.
 *     integer_p == FALSE to test if there exists at least one rational point 
 *               in the convex polyhedron defined by the system of constraints.
 *     (This has an impact only upon Fourier-Motzkin feasibility test).
 *
 * Last modified by Beatrice Creusillet, 13/12/94.
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>
#include <malloc.h>

#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "sc-private.h"

/* 
 * INTERFACES
 */

boolean 
sc_rational_feasibility_ofl_ctrl(sc, ofl_ctrl, ofl_res)
Psysteme sc;
int ofl_ctrl;
boolean ofl_res;
{
    return sc_feasibility_ofl_ctrl(sc, FALSE, ofl_ctrl, ofl_res);
}

boolean 
sc_integer_feasibility_ofl_ctrl(sc,ofl_ctrl, ofl_res)
Psysteme sc;
int ofl_ctrl;
boolean ofl_res;
{
    return sc_feasibility_ofl_ctrl(sc, TRUE, ofl_ctrl, ofl_res);
}

/*
 * LOW LEVEL FUNCTIONS 
 */

/*  just a test to improve the Simplex/FM decision.
 * c is a list of constraints, equalities or inequalities
 * pc is the number of constraints in the list
 * pv is the number of non-zero coefficients in the system
 *
 * pc and pv MUST be initialized. They are multiplied by weight.
 */
static void 
decision_data(c, pc, pv, weight)
Pcontrainte c;
int *pc, *pv, weight;
{
    Pvecteur v;
    
    for(; c!=NULL; c=c->succ)
    {
	v=c->vecteur;
	if (v!=NULL) 
	{
	    (*pc)+=weight;
	    for(; v!=NULL; v=v->succ) 
		if (var_of(v)!=TCST) (*pv)+=weight;
	}
    }
}

/* chose the next variable in base b for projection in system s.
 * tries to avoid Fourier potential explosions when combining inequalities.
 * - if there are equalities, chose the var with the min |coeff| (not null)
 * - if there are only inequalities, chose the var that will generate the
 *   minimum number of constraints with pairwise combinations.
 * - if ineq is TRUE, consider variables even if no equalities.
 *
 * (c) FC 21 July 1995
 */
static Variable
chose_variable_to_project_for_feasability(Psysteme s, Pbase b, boolean ineq)
{
  Pcontrainte c = sc_egalites(s);
  Pvecteur v;
  Variable var = NULL;
  Value val;
  int size = vect_size(b);

  ifscdebug(8)
    {
      fprintf(stderr, "[chose_variable_to_project_for_feasability] b/s:\n");
      vect_fprint(stderr, b, default_variable_to_string);
      sc_fprint(stderr, s, default_variable_to_string);
    }

  if (size==1) return var_of(b);
  assert(size>1);
    
  if (c)
  {
    /* find the lowest coeff 
     */
    Variable minvar = TCST;
    Value minval = VALUE_ZERO;

    for (; c; c=c->succ)
    {
      for (v = contrainte_vecteur(c); v; v=v->succ)
      {
	var = var_of(v);

	if (var!=TCST)
	{
	  val = value_abs(val_of(v));
	  if ((value_notzero_p(minval) && value_lt(val,minval))
	      || value_zero_p(minval)) 
	    minval = val, minvar = var;
		     
	  if (value_one_p(minval)) return minvar;
	}
      }
    }

    /* shouldn't find empty equalities ?? */
    /* assert(minvar!=TCST); */
    var = minvar;
  }
  
  if (!var && ineq)
  {
    /* only inequalities, reduce the explosion
     */
    int i;
    two_int_infop t = (two_int_infop) malloc(2*size*sizeof(int));
    Pbase tmp;
    int min_new;

    c = sc_inegalites(s);

    /* initialize t
     */
    for (i=0; i<size; i++) t[i][0]=0, t[i][1]=0;

    /* t[x][0 (resp. 1)] = number of negative (resp. positive) coeff
     */
    for (; c; c=c->succ)
    {
      for (v = contrainte_vecteur(c); v; v=v->succ)
      {
	var = var_of(v); 
	if (var!=TCST)
	{
	  ifscdebug(9) 
	    fprintf(stderr, "%s\n", default_variable_to_string(var));

	  for (i=0, tmp=b; tmp && var_of(tmp)!=var; 
	       i++, tmp=tmp->succ);
	  assert(tmp);
		    
	  t[i][value_posz_p(val_of(v))]++;
	}
      }
    }

    /* t[x][0] = number of combinations, i.e. new created constraints.
     */
    for (i=0; i<size; i++) t[i][0] *= t[i][1];

    for (tmp=b->succ, var=var_of(b), min_new=t[0][0], i=1;
	 min_new && i<size; 
	 i++, tmp=tmp->succ) {
      if (t[i][0]<min_new)
	min_new = t[i][0], var=var_of(tmp); 
    }

    free(t);
  }

  ifscdebug(8)
    fprintf(stderr, "[chose_variable_to_project_for_feasability] "
	    "suggesting %s\n", default_variable_to_string(var));

  return var;
}

/* project in s1 (which is modified) 
 * using equalities or both equalities and inequalities.
*/
static boolean sc_fm_project_variables
(Psysteme s1, boolean integer_p, boolean use_eq_only, int ofl_ctrl)
{
  Pbase b = base_dup(sc_base(s1));
  Variable var;
  boolean faisable = TRUE;
    
  while (b && faisable)
  {
    var = chose_variable_to_project_for_feasability(s1, b, !use_eq_only);

    /* if use_eq_only */
    if (!var) break;

    vect_erase_var(&b, var);

    ifscdebug(8)
      {
	fprintf(stderr, 
		"[sc_fm_project_variables] system before %s projection:\n", 
		var);
	sc_fprint(stderr, s1, default_variable_to_string);
      }
	    
    sc_projection_along_variable_ofl_ctrl(&s1, var, ofl_ctrl);

    ifscdebug(8)
    {
      fprintf(stderr, 
	      "[sc_fm_project_variables] system after projection:\n");
      sc_fprint(stderr, s1, default_variable_to_string);
    }
	    
    if (sc_empty_p(s1))
    {
      faisable = FALSE;
      break;
    }

    if (integer_p) {
      s1 = sc_normalize(s1);
      if (SC_EMPTY_P(s1)) 
      { 
	faisable = FALSE; 
	break;
      }
    }
  }  

  base_rm(b);
  return faisable;
}

#define SIMPLEX_METHOD		1
#define FM_METHOD		0
#define PROJECT_EQ_METHOD	2
#define NO_PROJ_METHOD		0

static boolean internal_sc_feasibility
  (Psysteme sc, int method, boolean integer_p, int ofl_ctrl)
{
  Psysteme sw = NULL;
  boolean ok = TRUE;

  if (method & PROJECT_EQ_METHOD)
  {
    sw = sc_dup(sc);
    ok = sc_fm_project_variables(sw, integer_p, TRUE, ofl_ctrl);
  }

  /* maybe the S/FM should be chosen again as #ref has changed... */
  if (ok) 
  {
    if (method & SIMPLEX_METHOD)
    {
      ok = sc_simplexe_feasibility_ofl_ctrl(sw? sw: sc, ofl_ctrl);
    }
    else 
    {
      ok = sc_fourier_motzkin_feasibility_ofl_ctrl
	(sw? sw: sc, integer_p, ofl_ctrl);
    }
  }
  
  if (sw) sc_rm(sw);

  return ok;
}

boolean 
sc_feasibility_ofl_ctrl(sc, integer_p, ofl_ctrl, ofl_res)
Psysteme sc;
boolean integer_p;
int ofl_ctrl;
boolean ofl_res;
{
  int
    method = 0,
    n_var = sc->dimension,
    n_cont_eq = 0, n_ref_eq = 0,
    n_cont_in = 0, n_ref_in = 0;
  boolean 
    ok = FALSE,
    catch_performed = FALSE;

  if (sc_rn_p(sc)) /* shortcut */
    return TRUE;

  decision_data(sc_egalites(sc), &n_cont_eq, &n_ref_eq, 2);
  decision_data(sc_inegalites(sc), &n_cont_in, &n_ref_in, 1);

  /* else
   */
  switch (ofl_ctrl) 
  {
  case OFL_CTRL :
    ofl_ctrl = FWD_OFL_CTRL;
    catch_performed = TRUE;
    CATCH(overflow_error) 
      {
	ok = ofl_res;
	catch_performed = FALSE;
	/* 
	 *   PLEASE do not remove this warning.
	 *
	 *   FC 30/01/95
	 */
	fprintf(stderr, "[sc_feasibility_ofl_ctrl] "
		"arithmetic error (%s[%d,%deq/%dref,%din/%dref]) -> %s\n",
		method&SIMPLEX_METHOD ? "Simplex" : "Fourier-Motzkin", 
		n_var, n_cont_eq, n_ref_eq, n_cont_in, n_ref_in,
		ofl_res ? "TRUE" : "FALSE");
	break;
      }		
  default: /* anyway, try */
    {
      /* a little discussion about what to decide (FC, 05/07/2000)
       *
       * - FM is good at handling equalities which are simply projected, 
       *   but may explode with many inequalities when they are combined.
       *   it is quite fast with few inequalities anyway.
       *
       * - SIMPLEX switches every eq to 2 inequalities, adding hyperplanes.
       *   thus it is not that good with equalities. Maybe equalities could
       *   be handled as such but I don't think that it is the case.
       *   it is quite slow with small systems because of the dense matrix
       *   to build and manipulate.
       *
       * suggestion implemented here: 
       *  1/ project equalities as much as possible,
       *     if there are many of them... ???
       *  2/ chose between FM and SIMPLEX **after** that?
       */
      /* use_simplex = (n_cont_in >= NB_CONSTRAINTS_MAX_FOR_FM || 
		     (n_cont_in>=10 && n_ref_in>2*n_cont_in));
		     (use_simplex && n_cont_eq >= 20) => proj */
      
      if (n_cont_in >= NB_CONSTRAINTS_MAX_FOR_FM ||
	  (n_cont_in>=10 && n_ref_in>2*n_cont_in))
      {
	method = SIMPLEX_METHOD;
	if (n_cont_eq >= 20)
	  method &= PROJECT_EQ_METHOD;
      }
      else
      {
	method = FM_METHOD;
      }

      /* STATS
	 extern void init_log_timers(void);
	 extern void get_string_timers(char **, char **);
	 for (method=0; method<4; method++) {
	 char * t1, * t2;
	 init_log_timers(); */

      ok = internal_sc_feasibility(sc, method, integer_p, ofl_ctrl);

	/* STATS 
	   get_string_timers(&t1, &t2);
	   fprintf(stderr, "FEASIBILITY %d %d %d %d %d %d %s",
	   n_cont_eq, n_ref_eq, n_cont_in, n_ref_in, method, ok, t1); } */

    }
  }

  if (catch_performed)
    UNCATCH(overflow_error);
  
  return ok;
}

/* boolean sc_fourier_motzkin_faisabilite_ofl(Psysteme s):
 * test de faisabilite d'un systeme de contraintes lineaires, par projections
 * successives du systeme selon les differentes variables du systeme
 *
 *  resultat retourne par la fonction :
 *
 *  boolean	: TRUE si le systeme est faisable
 *		  FALSE sinon
 *
 *  Les parametres de la fonction :
 *
 *  Psysteme s    : systeme lineaire 
 *
 * Le controle de l'overflow est effectue et traite par le retour 
 * du contexte correspondant au dernier CATCH(overflow_error) effectue.
 */
boolean 
sc_fourier_motzkin_feasibility_ofl_ctrl(s, integer_p, ofl_ctrl)
Psysteme s;
boolean integer_p;
int ofl_ctrl;
{
  Psysteme s1;
  boolean faisable = TRUE;

  if (s == NULL) return TRUE;
  s1 = sc_dup(s);

  ifscdebug(8)
    {
      fprintf(stderr, "[sc_fourier_motzkin_feasibility_ofl_ctrl] system:\n");
      sc_fprint(stderr, s1, default_variable_to_string);
    }

  s1 = sc_elim_db_constraints(s1);

  if (s1 != NULL)
  {
    /* a small basis if possible... (FC).
     */
    base_rm(sc_base(s1));
    sc_creer_base(s1);

    faisable = sc_fm_project_variables(s1, integer_p, FALSE, ofl_ctrl);
	
    sc_rm(s1);
  }
  else 
    /* sc_kill_db_eg a de'sallouer s1 a` la detection de 
       sa non-faisabilite */
    faisable = FALSE;

  return faisable;
}
