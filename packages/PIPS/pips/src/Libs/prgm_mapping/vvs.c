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

/* Name     : substitution.c
 * Package  : prgm_mapping
 * Author   : Platonoff Alexis
 * Date     : 4 october 1993
 * Historic :
 * Documents:
 * Comments : 
 * This file contains the functions for manipulating substitutions.
 */

/* Ansi includes 	*/
#include<stdio.h>

/* Newgen includes 	*/
#include "genC.h"

/* C3 includes 		*/
#include "boolean.h"
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "union.h"
#include "matrice.h"
#include "matrix.h"

/* Pips includes 	*/
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "graph.h"
#include "paf_ri.h"
#include "text.h"
#include "text-util.h"
#include "paf-util.h"
#include "prgm_mapping.h"

/* Macro functions  	*/
#define VV_VAR(x) var_val_variable(x)
#define VV_COEFF(x) expression_to_int(var_val_value(x))
#define VV_VAL_(x) normalized_linear_(expression_normalized(var_val_value(x)))
#define VV_VAL(x) normalized_linear(expression_normalized(var_val_value(x)))
#define MAKE_VV(var, coeff, val)  make_var_val(var, \
        make_expression(make_syntax(is_syntax_call, \
				    make_call(make_integer_constant_entity(coeff), \
					      NIL)), \
			make_normalized(is_normalized_linear, val)))

/* Global variables 	*/

/* Internal variables 	*/

/* ======================================================================== */
Ppolynome my_vecteur_to_polynome(pv)
Pvecteur pv;
{
  Ppolynome pp;

  if(VECTEUR_NUL_P(pv))
    pp = POLYNOME_NUL;
  else {
    Pvecteur vec;

    pp = NULL;
    for(vec = pv; vec != NULL; vec = vec->succ) {
      Variable var = vecteur_var(vec);
      float val = (float) vecteur_val(vec);
      Ppolynome newpp;

      newpp = make_polynome(val, var, 1);
      polynome_succ(newpp) = pp;
      pp = newpp;
    }
  }

  return(pp);
}


/* ======================================================================== */
list make_vvs(var, coeff, val)
entity var;
int coeff;
Pvecteur val;
{
  return(CONS(VAR_VAL, MAKE_VV(var, coeff, val), NIL));
}


/* ======================================================================== */
var_val dup_vv(vv)
var_val vv;
{
  return(MAKE_VV(VV_VAR(vv), VV_COEFF(vv), vect_dup((Pvecteur) VV_VAL(vv))));
}


/* ======================================================================== */
list dup_vvs(vvs)
list vvs;
{
  list new_vvs = NIL, l;

  for(l = vvs; !ENDP(l); POP(l)) {
    var_val v = VAR_VAL(CAR(l));

    new_vvs = gen_nconc(new_vvs, CONS(VAR_VAL, dup_vv(v), NIL));
  }
  return(new_vvs);
}


/* ======================================================================== */
void fprint_vvs(fp, vvs)
FILE *fp;
list vvs;
{
 if(vvs == NULL)
   fprintf(fp, "Var_vals Empty\n");
 for( ; !ENDP(vvs); POP(vvs))
    fprint_vv(fp, VAR_VAL(CAR(vvs)));
}


/* ======================================================================== */
void fprint_vv(fp, vv)
FILE *fp;
var_val vv;
{
  if(expression_syntax(var_val_value(vv)) == syntax_undefined)
    fprintf(fp, "%s --> ", entity_local_name(VV_VAR(vv)));
  else {
    int coeff = VV_COEFF(vv);
    if(coeff == 1)
      fprintf(fp, "%s --> ", entity_local_name(VV_VAR(vv)));
    else if(coeff == -1)
      fprintf(fp, "-%s --> ", entity_local_name(VV_VAR(vv)));
    else
      fprintf(fp, "%d.%s --> ", coeff, entity_local_name(VV_VAR(vv)));
  }

  pu_vect_fprint(fp, (Pvecteur) VV_VAL(vv));
}


/* ======================================================================== */
/*
 * list vvs_on_vvs(list vv1, vv2)
 *
 * Applies "vv1" on "vv2", i.e. replaces in the rhs of "vv2" the variables
 * appearing in the lhs of "vv1" by their value. It does not add the
 * substitutions of "vv1" in the list.
 *
 * For each sub of "vv1", it walks through all the sub of "vv2": for a given
 * sub of "vv1" (c1.e1 = pv1) and a given sub of "vv2" (c2.e2 = pv2),
 * there are two cases:
 *
 *	1. the lhs are equal, so a new sub is computed from the following
 *	equality: (p/c1).pv1 = (p/c2).pv2 = 0, with p = lcm(c1,c2).
 *
 *	2. the lhs are not equal, so pv2 may be modified if e1 appears in it:
 *	pv2 = cc2.e1 + pv0
 *	=> pv2 = (cc2/g)*pv1 + (c1/g)*pv0, with g = gcd(cc2, c1)
 *	
 */
list vvs_on_vvs(vv1, vv2)
list vv1, vv2;
{
  list l1, l2, new_vvs = gen_concatenate(vv2, NIL);

  for(l1 = vv1; !ENDP(l1); POP(l1)) {
    var_val v1 = VAR_VAL(CAR(l1));
    entity e1 = VV_VAR(v1);
    int c1 = VV_COEFF(v1);
    Pvecteur pv1 = (Pvecteur) VV_VAL(v1);

    for(l2 = new_vvs; !ENDP(l2); POP(l2)) {
      var_val v2 = VAR_VAL(CAR(l2));
      entity e2 = VV_VAR(v2);
      int c2 = VV_COEFF(v2);
      Pvecteur pv2 = (Pvecteur) VV_VAL(v2);

      if(same_entity_p(e1, e2)) {
        list il, el, aux_vvs;
	Psysteme aps1 = sc_new(), aps2;
	Pvecteur pv;
	int p;

	p = ppcm(c1, c2);
	pv = vect_cl2_ofl_ctrl((p/c1), pv1, -(p/c2), pv2, NO_OFL_CTRL);

	il = vecteur_to_list(pv);
	sc_add_egalite(aps1, contrainte_make(pv));

	aps2 = elim_var_with_eg(aps1, &il, &el);
	aux_vvs = make_vvs_from_sc(aps2, el);

	if(ENDP(aux_vvs))
	  return(NIL);
	if(!ENDP(CDR(aux_vvs)))
	  user_error("vvs_on_vvs", "There are more than one vv\n");

	CDR(aux_vvs) = CDR(l2);
	CDR(l2) = aux_vvs;
      }
      else {
	int cc2, g;
        if( (cc2 = vect_coeff((Variable) e1, pv2)) != 0 ) {
	  g = pgcd(cc2, c1);
          vect_erase_var(&pv2, (Variable) e1);
          VV_VAL_(v2) = newgen_Pvecteur
	      (vect_cl2_ofl_ctrl((c1/g), pv2, (cc2/g), pv1, NO_OFL_CTRL));
	}
      }
    }
  }
  return(new_vvs);
}


/* ======================================================================== */
/*
 * list compose_vvs(list vv1, vv2)
 *
 * Applies "vv1" on "vv2" AND "vv2" on "vv1", i.e. replaces in the rhs of
 * "vv2" the variables appearing in the lhs of "vv1" by their value AND
 * vice-versa.
 *
 * For each sub of "vv1", it walks through all the sub of "vv2": for a given
 * sub of "vv1" (c1.e1 = pv1) and a given sub of "vv2" (c2.e2 = pv2),
 * there are two cases:
 *
 *      1. the lhs are equal, so a new sub is computed from the following
 *      equality: (p/c1).pv1 = (p/c2).pv2 = 0, with p = lcm(c1,c2).
 *
 *      2. the lhs are not equal, so pv2 may be modified if e1 appears in it:
 *      pv2 = cc2.e1 + pv0
 *      => pv2 = (cc2/g)*pv1 + (c1/g)*pv0, with g = gcd(cc2, c1)
 *
 * Note: we work upon copies of "vv1" and "vv2".
 */
list compose_vvs(vv1, vv2)
list vv1, vv2;
{
  list l1, l2, p2, cvv1, cvv2;

  cvv1 = dup_vvs(vv1);
  cvv2 = dup_vvs(vv2);

  if(cvv1 == NIL) {
    if(cvv2 == NIL)
      return(NIL);
    else
      return(cvv2);
  }
  if(cvv2 == NIL)
    return(cvv1);
    
  for(l1 = cvv1; !ENDP(l1); POP(l1)) {
    var_val v1 = VAR_VAL(CAR(l1));
    entity e1 = VV_VAR(v1);
    int c1 = VV_COEFF(v1);
    Pvecteur pv1 = (Pvecteur) VV_VAL(v1);

    p2 = NIL;
    for(l2 = cvv2; !ENDP(l2); POP(l2)) {
      var_val v2 = VAR_VAL(CAR(l2));
      entity e2 = VV_VAR(v2);
      int c2 = VV_COEFF(v2);
      Pvecteur pv2 = (Pvecteur) VV_VAL(v2);

      if(same_entity_p(e1, e2)) {
        list il, el, aux_vvs;
	Psysteme aps1 = sc_new(), aps2;
	Pvecteur pv;
	int p;

        p = ppcm(c1,c2);
	pv = vect_cl2_ofl_ctrl((p/c1), pv1, -(p/c2), pv2, NO_OFL_CTRL);

	il = vecteur_to_list(pv);
	sc_add_egalite(aps1, contrainte_make(pv));

	aps2 = elim_var_with_eg(aps1, &il, &el);
	aux_vvs = sc_to_vvs(aps2, el);

        /* If "pv" is not NULL and "aux_vvs" is NULL, it means that
	 * elim_var_with_eg() did not work.
	 */
	if(ENDP(aux_vvs)) {
	  /* "v1 == v2", so "v2" is remove from "cvv2" */
	  if(p2 == NIL)
	    cvv2 = CDR(l2);
	  else
	    CDR(p2) = CDR(l2);
	}
	else if(!ENDP(CDR(aux_vvs)))
	  user_error("compose_vvs", "There are more than one vv\n");
        else {
          VAR_VAL(CAR(l2)) = VAR_VAL(CAR(aux_vvs));
	  p2 = l2;
	}
      }
      else {
	int cc2;
        if( (cc2 = vect_coeff((Variable) e1, pv2)) != 0 ) {
	  int g = pgcd(cc2, c1);
          vect_erase_var(&pv2, (Variable) e1);
          VV_VAL_(v2) = newgen_Pvecteur
	      (vect_cl2_ofl_ctrl((c1/g), pv2, (cc2/g), pv1, NO_OFL_CTRL));
	}
        p2 = l2;
      }
    }
  }
  for(l1 = cvv2; !ENDP(l1); POP(l1)) {
    var_val v1 = VAR_VAL(CAR(l1));
    entity e1 = VV_VAR(v1);
    int c1 = VV_COEFF(v1);
    Pvecteur pv1 = (Pvecteur) VV_VAL(v1);
    for(l2 = cvv1; !ENDP(l2); POP(l2)) {
      var_val v2 = VAR_VAL(CAR(l2));
      entity e2 = VV_VAR(v2);
      Pvecteur pv2 = (Pvecteur) VV_VAL(v2);

      if(same_entity_p(e1, e2)) {
	  user_error("compose_vvs", "Entities should be different\n");
      }
      else {
	int cc2;
	if( (cc2 = vect_coeff((Variable) e1, pv2)) != 0 ) {
	  int g = pgcd(cc2, c1);
	  vect_erase_var(&pv2, (Variable) e1);
	  VV_VAL_(v2) = newgen_Pvecteur
	      (vect_cl2_ofl_ctrl((c1/g), pv2, (cc2/g), pv1, NO_OFL_CTRL));
	}
      }
    }
  }
 return(gen_nconc(cvv1, cvv2));
}


/* ======================================================================== */
/* list sc_to_vvs(sc, var_l): returns the substitution equivalent to
 * the system given in argument knowing that the substituted variables are
 * contained in "var_l", in the same order. Then, their must be as many var
 * in "var_l" as there are equalities in "sc".
 */
list sc_to_vvs(sc, var_l)
Psysteme sc;
list var_l;
{
 list vvs = NIL;

 Pcontrainte pc;
 Pvecteur pv_elim;
 
 for(pc = sc->egalites ; pc != NULL; pc = pc->succ, var_l = CDR(var_l))
   {
    bool var_not_found = true;
    Pvecteur pv = pc->vecteur;
    entity var = ENTITY(CAR(var_l));

    for( ; !VECTEUR_NUL_P(pv) && var_not_found; )
      {
       if( ((pv->val == 1) || (pv->val == -1)) && (pv->var != TCST) )
         {
          if(same_entity_p((entity) pv->var, var))
             var_not_found = false;
          else
             pv = pv->succ;
         }
       else
          pv = pv->succ;
      }
    if(! var_not_found)
      {
       /* We have: val.var + v = 0 => var = -val.v, with val in {1, -1}
        * however: V = val.var + v => v = V - val.var
        * so: var = -val(V - val.var) => var = -val.V + (val)^2.var => var = -val.V + var
	*/
       pv_elim = vect_cl2_ofl_ctrl((pv->val)*(-1), pc->vecteur, 1,
				   vect_new(pv->var, 1),
				   NO_OFL_CTRL);

       vvs = compose_vvs(vvs, CONS(VAR_VAL, MAKE_VV(var, 1, pv_elim), NIL));
      }
    else
       user_error("sc_to_vvs", "The var must in the equality\n");
   }
 return(vvs);
}


/* ======================================================================== */
Psysteme vvs_to_sc(vvs)
list vvs;
{
 Psysteme ps = sc_new();
 
 for( ; !ENDP(vvs) ; POP(vvs)) {
    var_val vv = VAR_VAL(CAR(vvs));
    entity v = VV_VAR(vv);
    int c = VV_COEFF(vv);
    Pvecteur pv = (Pvecteur) VV_VAL(vv);

    vect_add_elem(&pv, (Variable) v, -c);
    sc_add_egalite(ps, contrainte_make(pv));
 }
 sc_creer_base(ps);
 return(ps);
}


/* ======================================================================== */
bool vvs_faisabilite(vvs)
list vvs;
{
  Psysteme ps;
  ps = vvs_to_sc(vvs);
  return(sc_rational_feasibility_ofl_ctrl(ps, NO_OFL_CTRL, true));
}

typedef bool (*argh)(Pvecteur*, Pvecteur*);

/* ======================================================================== */
/*
 * Ppolynome vvs_on_polynome(list vvs, Ppolynome pp)
 *
 * applies a substitution "sub" on a polynome "pp".
 */
Ppolynome vvs_on_polynome(vvs, pp)
list vvs;
Ppolynome pp;
{
 list avvs, used_vars;

  if(vvs == NIL)
    return(pp);

  /* Special treatment if there is only one sub */
  if(CDR(vvs) == NIL) {
    var_val vv = VAR_VAL(CAR(vvs));
    Variable var = (Variable) var_val_variable(vv);
    int c = VV_COEFF(vv);
    Pvecteur pv = (Pvecteur) VV_VAL(vv);
    Ppolynome ppv = vecteur_to_polynome(pv);

    if(c != 1)
      user_error("vvs_on_polynome", "Coeff is not 1\n");
    else
      pp = prototype_var_subst(pp, var, ppv);

    return(pp);
  }

 used_vars = base_to_list(polynome_used_var(pp, (argh) pu_is_inferior_var));

 for(avvs = vvs; !ENDP(avvs) && !ENDP(used_vars); POP(avvs)) {
    var_val vv = VAR_VAL(CAR(avvs));
    entity var = VV_VAR(vv);
    int c = VV_COEFF(vv);
    list prec, curr;
    bool found = false;

    if(c != 1)
      user_error("vvs_on_polynome", "Coeff is not 1\n");

    for(curr = used_vars, prec = NIL; !ENDP(curr) && (!found); POP(curr)) {
      if(same_entity_p(var, ENTITY(CAR(curr)))) {
	found = true;
	if(prec == NIL)
	  used_vars = CDR(used_vars);
	else
	  CDR(prec) = CDR(curr);
      }
      prec = curr;
    }

    if(found) {
      Pvecteur pv = (Pvecteur) VV_VAL(vv);
      Ppolynome ppv = vecteur_to_polynome(pv);

      pp = prototype_var_subst(pp, (Variable) var, ppv);
   }
 }

 return(pp);
}


/* ======================================================================== */
/* Pvecteur vvs_on_vecteur(list vvs, Pvecteur pv): applies the
 * substitution "s" on the vecteur "pv". The substitution is done directly
 * on "pv" ; so, it is modified and is the returned value.
 */
Pvecteur vvs_on_vecteur(vvs, pv)
list vvs;
Pvecteur pv;
{
  list avvs;

  if(ENDP(vvs))
    return(pv);

  for(avvs = vvs; !ENDP(avvs) ; POP(avvs)) {
    var_val vv = VAR_VAL(CAR(avvs));
    entity vve = VV_VAR(vv);
    int vvc = VV_COEFF(vv);
    Pvecteur vvpv = (Pvecteur) VV_VAL(vv);
    Value val = vect_coeff((Variable) vve, pv);

    if(vvc != 1)
      user_error("vvs_on_vecteur", "Coeff is not 1\n");

    /* We substitute "vvpv" to "vve" in "pv". */
    if(val != 0) {
      /* We delete the occurence of "vve"... */
      vect_erase_var(&pv, (Variable) vve);

      /* ... and we add "val*vvpv" to "pv". */
      pv = vect_cl_ofl_ctrl(pv, val, vvpv, NO_OFL_CTRL);
    }
  }
  return(pv);
}


/* ======================================================================== */
/* Psysteme vvs_on_systeme(list vvs, Psysteme ps): applies the
 * substitution "s" on the system "ps". The substitution is done directly on
 * "ps" ; so, it is modified and is the returned value.
 */
Psysteme vvs_on_systeme(vvs, ps)
list vvs;
Psysteme ps;
{
  Psysteme new_ps = ps;
  list avvs;

  if(ENDP(vvs))
    return(ps);

  for(avvs = vvs; !ENDP(avvs) ; POP(avvs)) {
    var_val vv = VAR_VAL(CAR(avvs));
    entity e = VV_VAR(vv);
    int c = VV_COEFF(vv);
    Pvecteur pv = (Pvecteur) VV_VAL(vv);

    /* We substitute "pv" to "c.e" in ps. */
    substitute_var_with_vec(new_ps, e, c, pv);
  }

  return(new_ps);
}


/* ======================================================================== */
/*
 * list make_vvs_from_sc(Psysteme ps_aux, list var_l): returns a list of
 * substitutions (var_val) computed with "ps_aux" and "var_l".
 *
 * Only the equations of "ps_aux" are considered and there must be as many
 * equations as there are variables in "var_l".
 * Also, the list equations and of variables must be ordered so as to have the
 * following conditions: one variable MUST appear in the corresponding
 * equation (i.e., it has a non zero coefficient) and MUST NOT appear in all
 * the others equations.
 *
 * Example: ps_aux = {l5 + l2 - l3 == 0, l3 + l4 - l1 == 0}
 *          var_l = {l2, l1}
 *
 * the function returns the substitution {l2 <-- l3 - l5, l1 <-- l3 + l4}
 *
 * Note: better than sc_to_vvs(), above.
 */
list make_vvs_from_sc(ps_aux, var_l)
Psysteme ps_aux;
list var_l;
{
  list new_vvs = NIL, l;
  Pcontrainte pc = ps_aux->egalites;

  for(l = var_l; !ENDP(l); POP(l), pc = pc->succ) {
    entity var = ENTITY(CAR(l));
    Pvecteur vec = pc->vecteur, new_vec;
    int val = vect_coeff((Variable) var, vec);
    var_val vv;

    if(val == 0)
      user_error("make_vvs_from_sc", "Value should not be equal to zero\n");

    /* We remove "var" from "vec" */
    new_vec = vect_del_var(vec, (Variable) var);

    /* We have: val.var + new_vec == 0. So: abs(val).var = -sign(val).new_vec
     */
    if(val < 0)
      val = 0-val;
    else
      vect_chg_sgn(new_vec);

/*
    vv_syn = syntax_undefined;
    if(val != 1) {
      vv_syn = make_syntax(is_syntax_call,
                           make_call(make_integer_constant_entity(val), NIL));
    }
    vv = make_var_val(var, make_expression(vv_syn,
					   make_normalized(is_normalized_linear,
                                                           new_vec)));
*/
    vv = MAKE_VV(var, val, new_vec);

    new_vvs = gen_nconc(new_vvs, CONS(VAR_VAL, vv, NIL));
  }
  return(new_vvs);
}


/* ======================================================================== */
/* list plc_make_vvs_with_vector(Pvecteur v): computes a substitution
 * from a vector representing the following equality: v = 0
 *
 * We do a Gauss-Jordan elimination on "v", the equation is transformed into
 * a substitution.
 *
 * For example, with the equation:
 *      C1 - C2 = 0
 *
 * we can eliminate one of the two variables C1 or C2), the substitution may
 * be:
 *       C2 <-- C1
 */
list plc_make_vvs_with_vector(v)
Pvecteur v;
{
  list vvs = NIL, lc;
  bool var_not_found;
  entity crt_var = entity_undefined;
  int crt_val = 0;
  Pvecteur pv_elim;

  /* we sort the variables of this vector in order to try the elimination in
   * the following order: first the CONST_COEFF, second the PARAM_COEFF,
   * third the INDEX_COEFF, last the MU_COEFF.
   */

  lc = general_merge_sort(vecteur_to_list(v), compare_coeff);

  /* We look, in lc, for a variable that we can eliminate in v, i.e. with a
   * coefficient equal to 1 or -1.
   */
  var_not_found = true;
  for(; (lc != NIL) && var_not_found; lc = CDR(lc)) {
    crt_var = ENTITY(CAR(lc));
    crt_val = (int) vect_coeff((Variable) crt_var, v);
    if((crt_val == 1) || (crt_val == -1))
      var_not_found = false;
  }
  if(! var_not_found) {
    pv_elim = vect_cl2_ofl_ctrl((crt_val)*(-1), v, 1,
				vect_new((Variable) crt_var, 1),
				NO_OFL_CTRL);
    vvs = make_vvs(crt_var, 1, pv_elim);
  }

  if(get_debug_level() > 5) {
     fprintf(stderr, "[plc_make_vvs_with_dist] \t\t\tDist vvs:\n");
     fprint_vvs(stderr, vvs);
     fprintf(stderr, "\n");
  }

  return(vvs);
}

