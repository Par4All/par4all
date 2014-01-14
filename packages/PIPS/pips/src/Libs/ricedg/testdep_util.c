/*

 $Id$

 Copyright 1989-2014 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif
/* here is a collection of function intended to create and manipulate "di"
 variables and test of dependence. di variables are pseudo variables
 created by PIPS and not accessible to the user that represent the
 distance between two dependent statement iterations.  the sign of this
 distance is sufficient for Kennedy's way of parallelizing programs, but
 the exact value migth be of interest for other algorithms such as
 systolic algorithms.

 Written by Remi, Yi-Qing; reorganized by Yi-Qing (18/09/91)
 */

#include "local.h"

/* To deal with overflow errors occuring during the projection
 * of a Psysteme along a variable */

/* The tables of di variables, li variables and ds variables.
 *
 * Variable DiVars[i-1] or LiVars[i-1] is associated to the loop at nesting
 * level i. A di variable represents the difference in iteration number
 * between the two references considered.
 *
 * The variable DsiVars[i] is associated to the ith element in the list
 * of scalar variables modified in the loops
 */
entity DiVars[MAXDEPTH];
entity LiVars[MAXDEPTH];
entity DsiVars[MAXSV];

/* This function creates di variables. There are MAXDEPTH di variables
 which means that programs with more than MAXDEPTH nested loops cannot be
 parallelized by pips.

 @param l is the nesting level of the variable to create
 */
entity MakeDiVar(int l) {
  entity e;
  string s;
  /* Create a variable of this format: "d#X" */
  asprintf(&s, "%s%sd#%d", DI_VAR_MODULE_NAME, MODULE_SEP_STRING, l);

  if((e = gen_find_tabulated(s, entity_domain)) == entity_undefined)
    e = make_entity(s, type_undefined, storage_undefined, value_undefined);
  else
    free(s);

  return (e);
}

/* This functions looks up a di variable of nesting level l in table
 DiVars. di variables are created if they do not exist.  */
entity GetDiVar(l)
  int l; {
  entity e;

  if(l < 1 || l > MAXDEPTH)
    user_error("parallelize", "too many nested loops\n");

  if((e = DiVars[l - 1]) == (entity)0) {
    int i;
    for (i = 0; i < MAXDEPTH; i++)
      DiVars[i] = MakeDiVar(i + 1);

    e = DiVars[l - 1];
  }

  return (e);
}

/* This function creates li variables(thee ith loop index variable).

 There are MAXDEPTH li variables which means that programs with more
 than MAXDEPTH nested loops cannot be parallelized by pips.

 @param l is the nesting level of the variable to create */
entity MakeLiVar(int l) {
  entity e;
  string s;
  /* Create a variable of this format: "l#X" */
  asprintf(&s, "%s%sl#%d", DI_VAR_MODULE_NAME, MODULE_SEP_STRING, l);

  if((e = gen_find_tabulated(s, entity_domain)) == entity_undefined)
    e = make_entity(s, type_undefined, storage_undefined, value_undefined);
  else
    free(s);

  return (e);
}

/*
 this functions looks up a li variable of nesting level l in table
 LiVars. li variables are created if they do not exist.
 */
entity GetLiVar(l)
  int l; {
  entity e;

  if(l < 1 || l > MAXDEPTH)
    user_error("parallelize", "too many nested loops\n");

  if((e = LiVars[l - 1]) == (entity)0) {
    int i;
    for (i = 0; i < MAXDEPTH; i++)
      LiVars[i] = MakeLiVar(i + 1);

    e = LiVars[l - 1];
  }

  return (e);
}

/* This function creates dsi variables.

 There are MAXSV dsi variables which means that programs with more than
 MAXSV scalar variables cannot be parallelized by PIPS.

 @param l means to create Dsi[l] variable
 */
entity MakeDsiVar(int l) {
  entity e;
  string s;
  /* Create a variable of this format: "ds#XXXX" */
  asprintf(&s, "%s%sds#%.4d", DI_VAR_MODULE_NAME, MODULE_SEP_STRING, l);

  if((e = gen_find_tabulated(s, entity_domain)) == entity_undefined)
    e = make_entity(s, type_undefined, storage_undefined, value_undefined);
  else
    free(s);

  return (e);
}

/*
 this functions looks up a dsi variable of the lth varable in table
 DsiVars. dsi variables are created if they do not exist.
 */
entity GetDsiVar(l)
  int l; {
  entity e;

  if(l < 1 || l > MAXSV)
    user_error("parallelize", "too many scalar variables\n");

  if((e = DsiVars[l - 1]) == (entity)0) {
    int i;
    for (i = 0; i < MAXSV; i++)
      DsiVars[i] = MakeDsiVar(i + 1);

    e = DsiVars[l - 1];
  }

  return (e);
}

/* 
 this function returns the nesting level of a given di variable e.
 */
int DiVarLevel(e)
  entity e; {
  int i;

  for (i = 0; i < MAXDEPTH; i++)
    if(e == DiVars[i])
      return (i + 1);

  return (0);
}


/*
 this function replaces each occurrence of variable e in system s by
 (e+dl) where dl is the di variable of nesting level l.

 l is the nesting level.

 e is the variable to replace.

 s is the system where replacements are to be done.

 li is the numerical value of the loop increment expression.
 */
void sc_add_di(l, e, s, li)
  int l;entity e;Psysteme s;int li; {
  Variable v = (Variable)GetDiVar(l);
  Value vli = int_to_value(li);
  Pcontrainte pc;

  for (pc = s->egalites; pc != NULL; pc = pc->succ) {
    Value ve = vect_coeff((Variable)e, pc->vecteur);
    value_product(ve, vli);
    vect_add_elem(&(pc->vecteur), v, ve);
  }

  for (pc = s->inegalites; pc != NULL; pc = pc->succ) {
    Value ve = vect_coeff((Variable)e, pc->vecteur);
    value_product(ve, vli);
    vect_add_elem(&(pc->vecteur), v, ve);
  }
}

/*
 this function replaces each occurrence of variable e in system s by
 (e+dsl) where dsl is the dsi variable of the lth element in the list of scalar variable.

 l is the order of e in the list.

 e is the variable to replace.

 s is the system where replacements are to be done.
 */
void sc_add_dsi(l, e, s)
  int l;entity e;Psysteme s; {
  Variable v = (Variable)GetDsiVar(l);
  Pcontrainte pc;
  for (pc = s->egalites; pc != NULL; pc = pc->succ) {
    vect_add_elem(&(pc->vecteur), v, vect_coeff((Variable)e, pc->vecteur));
  }

  for (pc = s->inegalites; pc != NULL; pc = pc->succ) {
    vect_add_elem(&(pc->vecteur), v, vect_coeff((Variable)e, pc->vecteur));
  }
}



/*
 this function projects a system on a set of di variables. this set is
 defined by cl, the common nesting level of the two array references
 being tested: only di variables whose nesting level is less or equal
 than cl are kept in the projected system.

 projection is done by eliminating variables (Fourier-Motzkin) which are
 not part of the set.

 cl is the common nesting level.

 s is the system to project. s is modified.
 */

int sc_proj_on_di(cl, s)
  int cl;Psysteme s; {
  Pbase coord;

  for (coord = s->base; !VECTEUR_NUL_P(coord); coord = coord->succ) {
    Variable v = vecteur_var(coord);
    int l = DiVarLevel((entity)v);
    
    if(l <= 0 || l > cl) {
      debug(8,
            "ProjectOnDi",
            "projection sur %s\n",
            entity_local_name((entity)v));
      if(SC_EMPTY_P(s = sc_projection_pure(s, v))) {
        debug(8, "ProjectOnDi", "infaisable\n");
        return (false);
      }
      debug(8, "ProjectOnDi", "faisable\n");
    }
  }

  return (true);
}



/* Pbase MakeDibaseinorder(int n) make a base of D#1 ... D#n in order of 
 * D#1-> D#2, ...-> D#n.
 */
Pbase MakeDibaseinorder(n)
  int n; {
  Pbase Dibase = BASE_NULLE;
  int i;

  for (i = 1; i <= n; i++) {
    Dibase = vect_add_variable(Dibase, (Variable)GetDiVar(n - i + 1));
  }
  return (Dibase);
}

int FindMaximumCommonLevel(n1, n2)
  cons *n1, *n2; {
  int cl = 0;

  while(n1 != NIL && n2 != NIL) {
    if(LOOP(CAR(n1)) != LOOP(CAR(n2)))
      break;
    n1 = CDR(n1);
    n2 = CDR(n2);
    cl += 1;
  }

  return (cl);
}


/* Management of loop counters */

#define ILCMAX 100000

static int ilc = ILCMAX;

void ResetLoopCounter() {
  ilc = 0;
}

/* Create a new loop counter variable */
entity MakeLoopCounter() {
  entity e;
  string s;
  //static char lcn[] = "lc#XXXXXX";

  while(1) {
    /* Create a variable of this format: "lc#XXXXXX" */
    asprintf(&s,
             "%s%slc#%06d",
             LOOP_COUNTER_MODULE_NAME,
             MODULE_SEP_STRING,
             ilc);

    if((e = gen_find_tabulated(s, entity_domain)) == entity_undefined) {
      pips_debug(8, "loop counter is %s\n", s);
      return make_entity(strdup(s),
                         type_undefined,
                         storage_undefined,
                         value_undefined);
    } else
      free(s);

    /* Try a new free one: */
    if((ilc += 1) == ILCMAX)
      break;
  }

  pips_internal_error("too many loop counters");
  return (entity_undefined);
}


/* int dep_type(action ac1,action ac2) 
 * This function test the type of the dependence. ac1, ac2 are the action of 
 * two references.The representations of the result are as follows.
 *  0 ---- def-use dependence
 *  1 ---- use-def dependence
 *  2 ---- def-def dependence
 *  3 ---- use-use dependence (added in 20/01/92)
 * FI->YY: we also have use-use dependence (also called input dependence);
 * there is no reason to abort here; input dependences should just be
 * ignored for parallelization, but not for tiling or cache optimization
 */

int dep_type(ac1, ac2)
  action ac1, ac2; {
  if(action_write_p(ac1) && action_read_p(ac2))
    return (0);
  else if(action_read_p(ac1) && action_write_p(ac2))
    return (1);
  else if(action_write_p(ac1) && action_write_p(ac2))
    return (2);
  else if(action_read_p(ac1) && action_read_p(ac2))
    return (3);
  else
    pips_internal_error("A undefined chain ---chains fault");

  /* to please gcc */
  return -1;
}

/* int sc_proj_optim_on_di_ofl(cl, sc)  
 *
 * This function projects a system onto a set of di variables. This set is
 * defined by cl, the common nesting level of the two array references
 * being tested: only di variables whose nesting level is less than or equal to
 * cl are kept in the projected system (i.e. outermost loops).
 *
 * The projection is performed by first eliminating variables in the
 * equations. Variables whose coefficients are 1 or -1 are considered first. 
 * (in such case it's integer elimination). Remaining inequalities are
 * projected by Fourier-Motzkin elimination.
 *
 * cl is the common nesting level.
 * sc is the system to project. sc is modified but psc always points to
 *    a consistent Psysteme on return (i.e. it's up to the caller to free it).
 *    *psc on return is sc_empty() if *psc on entry turns out to be 
 *    non-feasible.
 * a long jump buffer must have been initialized to handle overflows
 * The value returned is true if the system is feasible, false otherwise.
 */
int sc_proj_optim_on_di_ofl(int cl, Psysteme *psc)
{
  Pbase coord;
  Pvecteur pv = VECTEUR_NUL;
  Variable v;
  int l;
  int res;

  pips_debug(6, "begin\n");

  /* find the set of variables to be eliminated */
  for (coord = (*psc)->base; !VECTEUR_NUL_P(coord); coord = coord->succ) {
    v = vecteur_var(coord);
    l = DiVarLevel((entity)v);
    
    if(l <= 0 || l > cl) /* find one */
      vect_add_elem(&pv, v, 1);
  }

  ifdebug(6) {
    fprintf(stderr, "The list of variables to be eliminated is :\n");
    vect_debug(pv);
  }
  is_test_Di = true;

  pips_assert("Dependence system *psc is consistent before projection",
	      sc_consistent_p(*psc));

  *psc = sc_projection_optim_along_vecteur_ofl(*psc, pv);

  pips_assert("Dependence system *psc is consistent after projection",
	      sc_consistent_p(*psc));

  if(sc_empty_p(*psc))
    res = false;
  else
    res = true;


  vect_rm(pv);


  pips_assert("Dependence system *psc is consistent after empty test",
	      sc_consistent_p(*psc));
  pips_debug(6, "end\n");

  return res;
}

/* bool sc_faisabilite_optim (Psysteme sc) :
 *
 * Test system sc feasibility by successive projections
 * along all variables in its basis.
 *
 * carry out the projection with function sc_projection_optim_along_vecteur().
 *
 * sc_normalize() is called here before the projection, which means
 * that sc may be deallocated
 * 
 *  result  :
 *
 *  boolean	: true if system is faisable
 *		  false else
 *
 * Modification:
 *  - call to sc_rm() added when sc_projection_optim_along_vecteur_ofl()
 *    returns sc_empty; necessary to have a consistent interface: when FALSE
 *    is returned, sc always has been freed.
 */
bool sc_faisabilite_optim(sc)
  Psysteme sc; {
  debug(6, "sc_faisabilite_optim", "begin\n");
  sc = sc_normalize(sc);
  if(!sc_empty_p(sc)) {
    /* Automatic variables read in a CATCH block need to be declared volatile as
     * specified by the documentation*/
    Psysteme volatile sc1 = sc_dup(sc);
    is_test_Di = false;

    CATCH(overflow_error) {
      pips_debug(7, "overflow error, returning TRUE. \n");
      sc_rm(sc1);
      debug(6, "sc_faisabilite_optim", "end\n");
      return (true);

    } TRY {
      Pbase base_sc1 = base_dup(sc1->base);
      sc1 = sc_projection_optim_along_vecteur_ofl(sc1, base_sc1);
      base_rm(base_sc1);
      if(sc_empty_p(sc1)) {
        debug(7, "sc_faisabilite_optim", "system not feasible\n");
        debug(6, "sc_faisabilite_optim", "end\n");
        sc_rm(sc);
        UNCATCH(overflow_error);
        return (false);
      } else {
        sc_rm(sc1);
        debug(7, "sc_faisabilite_optim", "system feasible\n");
        debug(6, "sc_faisabilite_optim", "end\n");
        UNCATCH(overflow_error);
        return (true);
      }
    }
  } else {
    debug(7, "sc_faisabilite_optim", "normalized system not feasible\n");
    debug(6, "sc_faisabilite_optim", "end\n");
    sc_rm(sc);
    return (false);
  }
}

/* bool combiner_ofl_with_test(Psysteme sc, Variable v): 
 *
 * the copy of combiner() adding the test of suffisants conditions of integer 
 * combination. 
 *
 * It returns true if exact
 */
static bool combiner_ofl_with_test(Psysteme sc, Variable v)
{
  Psysteme sc1;
  Pcontrainte pos, neg, nul;
  Pcontrainte pc, pcp, pcn;
  Value c;
  int nnul, np, nn, i;

  pc = sc->inegalites;

  if(pc == NULL) {
    FMComp[0]++;
    return (true);
  }

  sc1 = sc_dup(sc);
  pos = neg = nul = NULL;
  nnul = 0;
  while(pc != NULL) {
    Pcontrainte pcs = pc->succ;

    if(value_pos_p(c = vect_coeff(v,pc->vecteur))) {
      pc->succ = pos;
      pos = pc;
    }
    else if(value_neg_p(c)) {
      pc->succ = neg;
      neg = pc;
    }
    else {
      pc->succ = nul;
      nul = pc;
      nnul += 1;
    }

    pc = pcs;
  }
  sc->inegalites = NULL;
  sc->nb_ineq = 0;

  np = nb_elems_list(pos);
  nn = nb_elems_list(neg);

  if((i = np * nn) <= 16)
    FMComp[i]++;
  else
    FMComp[17]++;

  for (pcp = pos; pcp != NULL; pcp = pcp->succ) {
    for (pcn = neg; pcn != NULL; pcn = pcn->succ) {
      bool int_comb_p = true;
      Pcontrainte pcnew =
          sc_integer_inequalities_combination_ofl_ctrl(sc1,
                                                       pcp,
                                                       pcn,
                                                       v,
                                                       &int_comb_p,
                                                       FWD_OFL_CTRL);

      if(contrainte_constante_p(pcnew)) {
        if(contrainte_verifiee(pcnew, false)) {
          contrainte_free(pcnew);
        } else {
          contraintes_free(pos);
          contraintes_free(neg);
          contraintes_free(nul);
          contraintes_free(pcnew);
          sc_rm(sc1);
          return (false);
        }
      } else {
        pcnew->succ = nul;
        nul = pcnew;
        nnul += 1;
        if(!int_comb_p) {
          if(is_test_exact)
            is_test_exact = false;
          if(is_test_inexact_fm == false)
            is_test_inexact_fm = true;
        }
      }
    }
  }

  /* apres les combinaisons eliminer les elements devenus inutiles */
  contraintes_free(pos);
  contraintes_free(neg);

  /* mise a jour du systeme */
  sc->inegalites = nul;
  sc->nb_ineq = nnul;
  sc_rm(sc1);
  return (true);
}

/* Psysteme sc_projection_optim_along_vecteur_ofl(Psysteme sc, Pvecteur pv)
 *
 * This fonction returns the projected system resulting of the
 * SUCCESSIVE projections of the system sc along the variables
 * contained in vecteur pv. Variables are only more or less projected
 * according to their order in pv.
 *
 * The projection is done first by eliminating variables constrained
 * by equations. The variables whose coefficients are 1 (or -1?) are
 * considered first (in such case it is a valide integer elimination),
 * then variables appearing in equations with non unit coefficient,
 * and finally all left over variables in pv using Fourier-Motzkin
 * elimination. At each step, the order in pv is used.
 *
 * If the system sc is not faisable, SC_EMPTY is returned.
 *
 * FI: this function could be moved to linear... but for the debugging
 * stuff. Also, it could be broken into three parts. And the number of
 * return statements should be reduced to one.
 */
Psysteme sc_projection_optim_along_vecteur_ofl( Psysteme sc, Pvecteur pv)
{
  Pcontrainte eq;
  Pvecteur pv1, prv, pve;
  Variable v;
  Value coeff;
  int syst_size_init;
  Pbase base_sc = base_dup(sc->base);
  Pbase lb = BASE_NULLE;

  pips_assert("The input constraint system is consistent", sc_consistent_p(sc));
  ifdebug(7) {
    pips_debug(7, "All variables to project: ");
    vect_dump(pv);
  }

  pve = vect_dup(pv);
  Pvecteur pve_to_free = pve;

  do {
    /* The elimination of variables using  equations */
    if(pve != NULL && sc->nb_eq != 0) {

      /* First,carry out the integer elimination possible */
      pips_debug(7, "carry out the integer elimination by equation:\n");

      prv = NULL;
      pv1 = pve;
      while(!VECTEUR_NUL_P(pv1) && (sc->nb_eq != 0)) {
	v = pv1->var;
	eq = contrainte_var_min_coeff(sc->egalites, v, &coeff, false);
	if((eq == NULL) || value_notone_p(coeff)) {
	  prv = pv1;
	  pv1 = pv1->succ;
	} else {
	  ifdebug(7) {
	    fprintf(stderr, "eliminate %s by :", entity_local_name((entity)v));
	    egalite_debug(eq);
	  }
	  /* coeff == 1, do this integer elimination for variable
	   * v with the others contrainte(equations and inequations */
	  sc = sc_variable_substitution_with_eq_ofl_ctrl(sc, eq, v, FWD_OFL_CTRL);

	  if(sc_empty_p(sc)) {
	    if(is_test_Di)
	      NbrTestProjEqDi++;
	    else
	      NbrTestProjEq++;
	    pips_debug(7, "projection infaisable\n");
	    for (pv1 = pv; pv1 != NULL; pv1 = pv1->succ) {
	      v = pv1->var;
	      sc_base_remove_variable(sc, v);
	    }
	    base_rm(base_sc);
	    vect_rm(pve_to_free);

	    pips_debug(7, "Return 1: empty\n");
	    return (sc);
	  }
	  sc = sc_normalize(sc);

	  if(sc_empty_p(sc)) { // FI: another normalization step is going to occur here
	    if(is_test_Di)
	      NbrTestProjEqDi++;
	    else
	      NbrTestProjEq++;
	    pips_debug(7, "normalisation infaisable\n");

	    for (pv1 = pv; pv1 != NULL; pv1 = pv1->succ) {
	      v = pv1->var;
	      sc_base_remove_variable(sc, v);
	    }

	    vect_rm(pve_to_free);

	    pips_debug(7, "Return 2: empty\n");
	    return (sc);

	  }

	  // The removal is delayed till the end of the function. sc is
	  // still consistent. However, too many variables might be left if an
	  // intermediate return occurs
	  // sc_base_remove_variable(sc, v);
	
	  ifdebug(7) {
	    fprintf(stderr, "projected normalised system is:\n");
	    sc_syst_debug(sc);
	  }

	  /* remove v from the list of variables to be projected pve */
	  if(prv == NULL) /* it's in head */
	    pve = pv1 = pv1->succ;
	  else {
	    prv->succ = pv1->succ;
	    free(pv1);
	    pv1 = prv->succ;
	  }
	}
      }
    }

    pips_assert("sc is consistent after the first use of equations",
		sc_consistent_p(sc));
    ifdebug(7) {
      pips_debug(7, "Variables left to project after the first step: ");
      vect_dump(pv);
    }

    /* carry out the non-exact elimination if necessary and possible
       using other equations */
    if(pve != NULL && sc->egalites != NULL) {
      pips_debug(7, "carry out the non integer elimination by equation:\n");
      pv1 = pve;
      prv = NULL;
      while((sc->egalites != 0) && (pv1 != NULL)) {
	v = pv1->var;
	eq = contrainte_var_min_coeff(sc->egalites, v, &coeff, true);
	if(eq == NULL && pv1 != NULL) {
	  prv = pv1;
	  pv1 = pv1->succ;
	} else {
	  if(eq != NULL) {
	    /* find a variable which appears in the equations, eliminate it*/
	    ifdebug(7) {
	      fprintf(stderr, "eliminate %s by :", entity_local_name((entity)v));
	      egalite_debug(eq);
	    }
	    if(is_test_inexact_eq == false)
	      is_test_inexact_eq = true;
	    if(is_test_exact)
	      is_test_exact = false;

	    sc = sc_variable_substitution_with_eq_ofl_ctrl(sc,
							   eq,
							   v,
							   FWD_OFL_CTRL);
	    if(sc_empty_p(sc)) {
	      if(is_test_Di)
		NbrTestProjEqDi++;
	      else
		NbrTestProjEq++;
	      pips_debug(7, "projection-infaisable\n");

	      for (pv1 = pv; pv1 != NULL; pv1 = pv1->succ) {
		v = pv1->var;
		sc_base_remove_variable(sc, v);
	      }
	      base_rm(base_sc);
	      vect_rm(pve_to_free);

	      pips_debug(7, "Return 3: empty\n");
	      return sc;
	    }

	    sc = sc_normalize(sc);

	    if(sc_empty_p(sc)) { // FI: should be sc_is_empty_p... or
	      // the normalization is repeated...
	      pips_debug(7, "normalization detects a non-feasible constraint system\n");
	      if(is_test_Di)
		NbrTestProjEqDi++;
	      else
		NbrTestProjEq++;

	      for (pv1 = pv; pv1 != NULL; pv1 = pv1->succ) {
		v = pv1->var;
		sc_base_remove_variable(sc, v);
	      }

	      vect_rm(pve_to_free);

	      pips_debug(7, "Return 4: empty\n");
	      return sc;
	    }

	    // FI: the removal is delayed to end of this function, but
	    // sc is nevertheless consistent
	    // sc_base_remove_variable(sc, v);
	    ifdebug(7) {
	      fprintf(stderr, "projected normalised system is:\n");
	      sc_syst_debug(sc);
	    }
	    /*eliminate v in the list of variables pve*/
	    if(prv == NULL) /* it's in head */
	      pve = pv1 = pv1->succ;
	    else
	      prv->succ = pv1 = pv1->succ;
	  }
	}
      }
    }

    pips_assert("sc is consistent after the second use of equations",
		sc_consistent_p(sc));
    ifdebug(7) {
      pips_debug(7, "Variables left to project with Fourier-Motzkin: ");
      vect_dump(pve);
    }

    /* carry out the elimination using Fourier-Motzkin for the other variables */

    if(pve != NULL) {
      pv1 = pve;

      while(pv1 != NULL) {

	NbrProjFMTotal++;
	syst_size_init = nb_elems_list(sc->inegalites);
	v = pv1->var;

	ifdebug(7) {
	  fprintf(stderr, "eliminate %s by F-M\n", entity_local_name((entity)v));
	  pips_debug(7, "is_test_exact before: ");
	  if(is_test_exact)
	    fprintf(stderr, "%s\n", "exact");
	  else
	    fprintf(stderr, "%s\n", "not exact");
	}

	if(combiner_ofl_with_test(sc, v) == false) {
	  /* detection of non feasability of Psysteme sc */
	  if(is_test_Di)
	    NbrTestProjFMDi++;
	  else
	    NbrTestProjFM++;
	  sc_rm(sc);
	  sc = sc_empty(base_sc);
	  for (pv1 = pv; pv1 != NULL; pv1 = pv1->succ) {
	    v = pv1->var;
	    sc_base_remove_variable(sc, v);
	  }

	  vect_rm(pve_to_free);

	  pips_debug(7, "Return 5: empty\n");
	  return sc;
	}

	sc = sc_normalize(sc);

	if(sc_empty_p(sc)) {
	  if(is_test_Di)
	    NbrTestProjFMDi++;
	  else
	    NbrTestProjFM++;
	  pips_debug(7, "normalisation-infaisable\n");
	  for (pv1 = pv; pv1 != NULL; pv1 = pv1->succ) {
	    v = pv1->var;
	    sc_base_remove_variable(sc, v);
	  }

	  vect_rm(pve_to_free);

	  pips_debug(7, "Return 6: empty\n");
	  return sc;
	}

	/* 	    sc->inegalites = contrainte_sort(sc->inegalites, sc->base, BASE_NULLE, */
	/* 					     true, false); */

	/* 	    ifdebug(8) { */
	/* 		debug(8, "", "Sorted system :\n"); */
	/* 		sc_syst_debug(sc); */
	/* 	    } */

	build_sc_nredund_2pass_ofl_ctrl(&sc, FWD_OFL_CTRL);

	ifdebug(7) {
	  pips_debug(7, "is_test_exact after: ");
	  if(is_test_exact)
	    fprintf(stderr, "%s\n", "exact");
	  else
	    fprintf(stderr, "%s\n", "not exact");
	  fprintf(stderr, "projected normalised system is:\n");
	  sc_syst_debug(sc);
	}
	if(nb_elems_list(sc->inegalites) <= syst_size_init)
	  NbrFMSystNonAug++;
	pv1 = pv1->succ;
      }
    }

    pips_assert("sc is consistent before base update", sc);

    /* Are we done? This used to be always true, but sc_normalize()
       may promote inequalities as equations. Since the projection
       steps may reveal some implicit equations, we are not sure here
       that all projections have been performed by the three steps
       (equations, equations, inequalities).

       See for instance the test dependence in SAC/sgemm for such an example.
    */
    Pbase mb = sc_to_minimal_basis(sc);
    lb = base_intersection(mb, pv);
    vect_rm(mb);
    
  } while(!BASE_NULLE_P(lb)); // No need to free lb since it is empty

  /* Update the base and the dimension of the constraint system sc */

  sc->nb_ineq = nb_elems_list(sc->inegalites);
  for (pv1 = pv; pv1 != NULL; pv1 = pv1->succ) {
    v = pv1->var;
    sc_base_remove_variable(sc, v);
  }

  pips_assert("sc is consistent after base update", sc);

  /* Clean up auxiliary data structures */
  vect_rm(pve_to_free);
  base_rm(base_sc);

  pips_debug(7, "final return: faisable\n");

  return sc;
}

/* void sc_minmax_of_variable_optim(Psysteme ps, Variable var, Value *pmin, *pmax):
 * examine un systeme pour trouver le minimum et le maximum d'une variable
 * apparaissant dans ce systeme par projection a la Fourier-Motzkin.
 * la procedure retourne la valeur false si le systeme est infaisable et
 * true sinon
 *
 * le systeme ps est detruit.
 * 
 */
bool sc_minmax_of_variable_optim(ps, var, pmin, pmax)
  Psysteme volatile ps;Variable var;Value *pmin, *pmax; {
  Value val;
  Pcontrainte pc;
  Pbase b;
  Pvecteur pv = NULL;

  *pmax = VALUE_MAX;
  *pmin = VALUE_MIN;

  if(sc_value_of_variable(ps, var, &val) == true) {
    *pmin = val;
    *pmax = val;
    return true;
  }

  /* projection sur toutes les variables sauf var */
  for (b = ps->base; !VECTEUR_NUL_P(b); b = b->succ) {
    Variable v = vecteur_var(b);
    if(v != var) {
      vect_add_elem(&pv, v, 1);
    }
  }

  CATCH(overflow_error) {
    debug(6,
          "sc_minmax_of_variable_optim",
          " overflow error, returning INT_MAX and INT_MIN. \n");
    *pmax = INT_MAX;
    *pmin = INT_MIN;
  } TRY {

    ps = sc_projection_optim_along_vecteur_ofl(ps, pv);
    if(sc_empty_p(ps)) {
      sc_rm(ps);
      UNCATCH(overflow_error);
      return false;
    }
    if(sc_empty_p(ps = sc_normalize(ps))) {
      sc_rm(ps);
      UNCATCH(overflow_error);
      return false;
    }

    if(sc_value_of_variable(ps, var, &val) == true) {
      *pmin = val;
      *pmax = val;
      UNCATCH(overflow_error);
      return true;
    }

    for (pc = ps->inegalites; pc != NULL; pc = pc->succ) {
      Value cv = vect_coeff(var, pc->vecteur);
      Value cc = value_uminus(vect_coeff(TCST, pc->vecteur));

      if(value_pos_p(cv)) {
        /* cette contrainte nous donne une borne max */
        Value bs = value_pdiv(cc,cv);
        if(value_lt(bs,*pmax))
          *pmax = bs;
      } else if(value_neg_p(cv)) {
        /* cette contrainte nous donne une borne min */
        Value bi = value_pdiv(cc,cv);
        if(value_gt(bi,*pmin))
          *pmin = bi;
      }
    }

    UNCATCH(overflow_error);
    vect_rm(pv);
  }

  if(value_lt(*pmax,*pmin))
    return false;

  sc_rm(ps);

  return true;
}

/* Psysteme sc_invers(Psysteme ps):
 * calcul un systeme des contraintes qui est l'invers du systeme initial. 
 * pour chaque element b dans le base initial, remplace b par -b dans 
 * le systeme initial.  
 */
Psysteme sc_invers(ps)
  Psysteme ps; {
  Pbase b;
  Pcontrainte eq;
  Variable v;

  for (b = ps->base; !VECTEUR_NUL_P(b); b = b->succ) {
    v = vecteur_var(b);
    for (eq = ps->egalites; eq != NULL; eq = eq->succ)
      vect_chg_var_sign(&eq->vecteur, v);
    
    for (eq = ps->inegalites; eq != NULL; eq = eq->succ)
      vect_chg_var_sign(&eq->vecteur, v);
  }
  return (ps);
}

/* void vect_chg_var_sign(Pvecteur *ppv, Variable var)
 * changement de signe de la coordonnee var du vecteur *ppv 
 */
void vect_chg_var_sign(ppv, var)
  Pvecteur *ppv;Variable var; {
  Pvecteur pvcour;

  for (pvcour = (*ppv); pvcour != NULL; pvcour = pvcour->succ)
    if(pvcour->var == var)
      value_oppose(pvcour->val);

  return;
}

/* Ppoly sc_poly_enveloppe(s1, s2): calcul d'une representation par polyedre 
 * de l'enveloppe convexe des polyedres definis par les systemes
 * lineaires s1 et s2
 *
 * p = enveloppe(s1, s2);
 * return p;
 *
 * s1 et s2 ne sont pas modifies. Ils doivent tous les deux avoir au moins
 * une base.
 *
 * Il faudrait traiter proprement les cas particuliers SC_RN et SC_EMPTY
 */
/*
Ppoly
sc_poly_enveloppe(s1, s2)
Psysteme s1;
Psysteme s2;
{
    Pbase b;
    Pvecteur coord;
    Ppoly p1;
    Ppoly p2;
    Ppoly p;
    Psysteme s = SC_UNDEFINED;

    assert(!SC_UNDEFINED_P(s1) || !SC_UNDEFINED_P(s2));

    if(SC_EMPTY_P(s1)) {
  s = s2;
    }
    else if(SC_EMPTY_P(s2)) {
  s = s1;
    }

    if (s != SC_UNDEFINED) {
  p  = sc_to_poly(s);
  return (p);
    }
    else {
  s1 = sc_dup(s1);
  s2 = sc_dup(s2);

  b = s1->base;
  for(coord=s2->base; !VECTEUR_NUL_P(coord); coord = coord->succ) {
      b = vect_add_variable(b, vecteur_var(coord));
  }
  vect_rm(s2->base);
  s2->base = vect_dup(b);

  p1 = sc_to_poly(s1);
  p2 = sc_to_poly(s2);

  p = env(p1, p2);
  return (p);
    }
}
*/
