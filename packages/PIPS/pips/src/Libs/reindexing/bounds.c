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
/************************************************************************/
/* Name     : bounds.c
 * Package  : reindexing
 * Author   : Alexis Platonoff
 * Date     : March 1995
 * Historic :
 *
 * Documents: SOON
 * Comments : This file contains the functions dealing with the bounds of
 * the variables and the loops.
 */

/* Ansi includes 	*/
#include <stdio.h>

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
#include "matrice.h"
#include"union.h"
#include "matrix.h"
#include "sparse_sc.h"

/* Pips includes 	*/
#include "boolean.h"
#include "ri.h"
#include "constants.h"
#include "ri-util.h"
#include "misc.h"
#include "complexity_ri.h"
#include "database.h"
#include "graph.h"
#include "dg.h"
#include "paf_ri.h"
#include "parser_private.h"
#include "property.h"
#include "reduction.h"
#include "text.h"
#include "text-util.h"
#include "tiling.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "static_controlize.h"
#include "paf-util.h"
#include "pip.h"
#include "array_dfg.h"
#include "prgm_mapping.h"
#include "conversion.h"
#include "scheduling.h"
#include "reindexing.h"

/*====================================================================*/
/* void set_array_declaration(entity var_to_decl, list lrange)
 *
 * Set the dimensions of entity "var_to_decl" with the ranges of
 * "lrange". */

void set_array_declaration(var_to_decl, lrange)
entity var_to_decl;
list lrange;
{
  list lr, ldims;
  variable var;

  var = type_variable(entity_type(var_to_decl));
  ldims = NIL;

  for(lr = lrange; !ENDP(lr); POP(lr)) {
    expression lb, ub, st;
    range ra = RANGE(CAR(lr));
    
    lb = range_lower(ra);
    ub = range_upper(ra);
    st = range_increment(ra);
    
    if(!expression_equal_integer_p(st, 1))
      user_error("set_array_declaration", "\n Increment diff de 1\n");
    
    ldims = gen_nconc(ldims, CONS(DIMENSION, make_dimension(lb, ub),
				  NIL));
  }
  variable_dimensions(var) = ldims;
}


#define IS_LOWER 0
#define IS_UPPER 1

/*====================================================================*/
/*
 * expression constraint_to_bound(Pcontrainte pc, entity ent, list lvar,
 *                               list lrange, int lower_or_upper,
 *                               int array_or_loop)
 *
 * Build a bound of the variable "ent" (named "e" in the following) from a
 * constraint "pc".
 *
 * This constraint may contains other variables from the list "lvar" which
 * have to be eliminated if we are in the case of ARRAY BOUNDS
 * ("array_or_loop") using their bounds (in "lrange"). See make_bounds(),
 * below.
 *
 * The nature (lower or upper) of the bound to compute is given by
 * "lower_or_upper" (lower: 0, upper: 1).
 *
 * For example, let us compute the lower bound of e with the constraint
 * (e-i+n-2 <= 0), knowing that (1 <= i <= MAX(n,m)). The elimination of i
 * is done by replacing it by n. We obtain the following lower bound : e
 * <= 2
 *
 * If the bound of a variable that have to be substitute is expressed via
 * a MIN or MAX function, we have to make a bound that will have as many
 * expressions  as there are arguments in this MIN or MAX function. This
 * variable is then successively replaced by each of these arguments.
 *
 * */
expression constraint_to_bound(pc, ent, lvar, lrange, lower_or_upper,
			       array_or_loop)
Pcontrainte pc;
entity ent;
list lvar, lrange;
int lower_or_upper, array_or_loop;
{
  expression bound = expression_undefined;
  Pvecteur vect;
  Value val;
  list llv, llr;
  Pcontrainte lvect;
  
  vect = vect_dup(pc->vecteur);

  /* "val" corresponds to the absolute value of the coefficient of "e" in
   * the constraint. */
  val = vect_coeff((Variable) ent, vect);
  value_absolute(val);

  /* "vect" represents the term of the constraints that does not contains
   * "e". We have : (e+vect <= 0) or (-e+vect <= 0), whether it is a lower
   * or upper bound. */
  vect = vect_del_var(vect, (Variable) ent);

  /* In the case of the upper bound, we have to change the sign of "vect"
   * in order to have a positive coefficient for "e". */
  if(lower_or_upper == IS_UPPER)
    vect_chg_sgn(vect);

  /* We build this Pcontrainte which will be used as a list of
   * Pvecteur. */
  lvect = contrainte_make(vect);
  
  if (get_debug_level() > 6) {
    fprintf(stderr, "\nlvect :");
    vecteur_fprint(stderr, lvect, pu_variable_name);
    fprintf(stderr, "\n");
  }
  
  /* In the case of ARRAY BOUNDS, we substitute in "vect" all the var of
   * "lvar" that have a non zero coeff by their min or max value, which is
   * given by "lrange". */
  if(array_or_loop == IS_ARRAY_BOUNDS) {
    for(llv = lvar, llr = lrange; (!ENDP(llv)) && (!ENDP(llr));
	POP(llv), POP(llr)) {
      entity var = ENTITY(CAR(llv));
      range cr = RANGE(CAR(llr));
      Value coeff;
    
      if(value_notzero_p(coeff = vect_coeff((Variable) var, vect))) {
	Pvecteur new_vect;
	expression cexp;
	normalized new_nor;
	int sign = value_sign(coeff);

	/* We replace the current var by whether its lower or upper bound,
	 * it depends on the sign of its coefficient. It also depends on the
	 * kind of bound we have to build. */
	if(lower_or_upper == IS_LOWER) {
	  if(sign ==1)
	    cexp = range_lower(cr);
	  else
	    cexp = range_upper(cr);
	}
	else			/* lower_or_upper == IS_UPPER */ {
	  if(sign==-1)
	    cexp = range_lower(cr);
	  else
	    cexp = range_upper(cr);
	}

	if (get_debug_level() > 6) {
	  fprintf(stderr, "Subs of %s by %s\n",
		  entity_local_name(var),
		  words_to_string(words_expression(cexp)));
	}

	/* This expression by which we replace the current var can be of two
	 * forms. The first, a single linear expression, is just
	 * substitute. The second, a MAX or MIN expression has to be switch
	 * in as many expressions as there are linear expressions in this
	 * expression.
	 * */
	new_nor = NORMALIZE_EXPRESSION(cexp);
	if(normalized_tag(new_nor) == is_normalized_linear) {

	  if (get_debug_level() > 6) {
	    fprintf(stderr, "SINGLE expression\n");
	  }

	  /* We apply the substitution on all the vectors of our list. */
	  new_vect = (Pvecteur) normalized_linear(new_nor);
	  for(pc = lvect; pc != NULL; pc = pc->succ) {
	    pc->vecteur = vect_var_subst(pc->vecteur, (Variable) var,
					 vect_dup(new_vect));
	  }

	}
	else {
	  if(min_or_max_expression_p(cexp)) {
	    list args, la;

	    args = call_arguments(syntax_call(expression_syntax(cexp)));

	    if (get_debug_level() > 6) {
	      fprintf(stderr, "MAX or MIN expression of %d args\n",
		      gen_length(args)); 
	    }

	    /* For each vector of our list, we duplicate it in order to take
	     * into account all the arguments of this MIN or MAX
	     * function. */
	    for(pc = lvect; pc != NULL; pc = pc->succ) {
	      Pvecteur pcv = pc->vecteur;
	      Pcontrainte pcsucc = pc->succ;

	      if (get_debug_level() > 6) {
		fprintf(stderr, "Subs in:\n");
		pu_vect_fprint(stderr, pcv);
	      }

	      /* We walk through all these arguments. */
	      for(la = args; !ENDP(la); POP(la)) {
		expression arg = EXPRESSION(CAR(la));
		new_nor = NORMALIZE_EXPRESSION(arg);
		new_vect = (Pvecteur) normalized_linear(new_nor);

		/* This "arg" is not the last one, so we duplicate our
		 * vector before substituting. */
		pcv = pc->vecteur;
		if(CDR(la) != NIL) {
		  Pcontrainte npc = contrainte_make(vect_dup(pcv));
		  pc->succ = npc;
		  npc->succ = pcsucc;
		}
		pc->vecteur = vect_var_subst(pcv, (Variable) var,
					     vect_dup(new_vect));
		
		if (get_debug_level() > 6) {
		  Psysteme aux_ps = sc_make(lvect, NULL);

		  fprintf(stderr, "Subs with %s gives:\n",
			  words_to_string(words_expression(arg)));
		  pu_vect_fprint(stderr, pc->vecteur);

		  fprintf(stderr, "\tlvect : \n");
		  fprint_psysteme(stderr, aux_ps);
		  fprintf(stderr, "\n");
		}

		/* This is not the last argument, so the successor is our
		 * current vector which has been just duplicated. */
		if(CDR(la) != NIL) 
		  pc = pc->succ;
	      }
	    }
	  }
	  else
	    user_error("constraint_to_bound",
		       "\nWe want a linear bound\n");
	}
      }
    }
  }

  /* We now build the expression that represent the bound of our variable
   * "e". This expression is a single linear one if "lvect" contains only
   * one vector, otherwise, it is a MIN or MAX expression, depending on
   * whether it is a lower or upper bound. */
  if(lvect->succ == NULL)
    bound = make_rational_exp(lvect->vecteur, val);
  else {
    call ca;
    list args = NIL;
	 
    for(pc = lvect; pc != NULL; pc = pc->succ)
      ADD_ELEMENT_TO_LIST(args, EXPRESSION,
			  make_rational_exp(pc->vecteur, val));
    if(lower_or_upper == IS_LOWER)
      ca = make_call(entity_intrinsic("MIN"), args);
    else /* lower_or_upper == IS_UPPER */
      ca = make_call(entity_intrinsic("MAX"), args);
    bound = make_expression(make_syntax(is_syntax_call, ca),
			    normalized_undefined);
  }
  return(bound);
}

/*====================================================================*/
/*
 * expression bound_compute(Psysteme sc, entity ent, list lvar,
 *                          list lrange, int lower_or_upper
 *                         int array_or_loop) 
 *
 * Computes the lower or upper bound of a given variable "ent" from a
 * system of constraints "sc". Each constraint should contains a reference
 * to this variable and is treated separatly with constraint_to_bound(),
 * see above.
 *
 */

expression bound_compute(sc, ent, lvar, lrange, lower_or_upper, array_or_loop)
Psysteme sc;
entity ent;
list lvar, lrange;
int lower_or_upper, array_or_loop;
{
  expression bound = expression_undefined;
  Pcontrainte inequ;
  
  if (get_debug_level() > 6) {
    if(lower_or_upper == IS_LOWER)
      fprintf(stderr, "\nIN LOWER\n");
    else
      fprintf(stderr, "\nIN UPPER\n");
  }
  
  if(SC_UNDEFINED_P(sc))
    user_error("bound_compute", "Undefined systeme\n");

  for(inequ = sc->inegalites; inequ != NULL; inequ = inequ->succ) {
    if(bound == expression_undefined)
      bound = constraint_to_bound(inequ, ent, lvar, lrange,
				  lower_or_upper, array_or_loop);
    else {
      int min_or_max;
      if(lower_or_upper == IS_LOWER)
	min_or_max = IS_MAX;
      else
	min_or_max = IS_MIN;
      bound = merge_expressions(bound,
				constraint_to_bound(inequ, ent, lvar,
						    lrange,
						    lower_or_upper,
						    array_or_loop),
				min_or_max);
    }
  }
  
  return(bound);
}


/*======================================================================*/
/* range make_bounds(ps, ent, array_or_loop, list lvar, list lrange)
 *
 * Builds the bounds of variable "ent" from the psysteme
 * "ps". "array_or_loop" gives which kind of bounds we want:
 * IS_LOOP_BOUNDS or IS_ARRAY_BOUNDS. For the first, these bounds will be
 * used in a loop, so we just have to get the constraints of "ps" and use
 * them to build these bounds (in the bounds of a given loop may appear
 * other loop indices from englobing loops). On the contrary, for the
 * second, these bounds will be used in an array dimension declaration, so
 * we first have to eliminate the other variables (in "lvar") using their
 * bounds (in "lrange") before building these bounds.
 *
 * AP 95/01/20 */

range make_bounds(ps, ent, array_or_loop, lvar, lrange)
Psysteme     ps;
entity       ent;
int array_or_loop;
list lvar, lrange;
{
  Pcontrainte  cont;
  expression   upper, lower, incr;
  Psysteme     sc_upper = sc_new(), sc_lower = sc_new();
  list         llr;
  
  if (get_debug_level() > 5) {
    fprintf(stderr,"\nBEGIN make_bounds : %s", entity_local_name(ent));
    fprintf(stderr,"\n\tKind : %d", array_or_loop);
    fprint_psysteme(stderr, ps);
    fprintf(stderr, "List of vars : ");
    fprint_entity_list(stderr, lvar);
    fprintf(stderr, "\nList of ranges :\n");
    for(llr = lrange; !ENDP(llr); POP(llr)) {
      fprintf(stderr, "\t%s\n",
	      words_to_string(words_range(RANGE(CAR(llr)))));
    }
  }
  
  if (SC_UNDEFINED_P(ps))
    user_error("make_bounds", "\nUndefined system\n");

  /* the psysteme should have two constraints, one for the lb and the
   * other for the ub, but we don't know which one is what.  In fact, it
   * may have more constraints, in which case the lower or/and upper
   * bounds are expressed by MIN or MAX functions. */
  cont = ps->inegalites;
  while (cont != NULL) {
    if (value_posz_p(vect_coeff((Variable) ent, cont->vecteur)))
    { sc_add_inegalite(sc_upper, contrainte_dup(cont)); }
    else
    { sc_add_inegalite(sc_lower, contrainte_dup(cont)); }
    cont = cont->succ;
  }
    
  if (get_debug_level() > 6) {
    fprintf(stderr, "\n LOWER ps\n");
    fprint_psysteme(stderr, sc_lower);
    fprintf(stderr, "\n UPPER ps\n");
    fprint_psysteme(stderr, sc_upper);
  }

  /* We compute these lower and upper bounds. */
  lower = bound_compute(sc_lower, ent, lvar, lrange, IS_LOWER,
			array_or_loop);
  upper = bound_compute(sc_upper, ent, lvar, lrange, IS_UPPER,
			array_or_loop);

  if (get_debug_level() > 6) {
    fprintf(stderr, "\nEND make_bounds : %s\n", entity_local_name(ent));
    fprintf(stderr, " Lower bound : %s\n",
	    words_to_string(words_expression(lower)));
    fprintf(stderr, " Upper bound : %s\n\n",
	    words_to_string(words_expression(upper)));
  }

  incr = int_to_expression(1);
  
  return(make_range(lower, upper, incr));
}


/*======================================================================*/
/* void get_bounds_expression(sys, lt, lb, ub)
 *
 * From the list of Psysteme psys, we determine the value of each lower
 * and upper bound of the local time.
 *
 * Lower bounds are in the inequalities of "sys" ; upper bounds are in the
 * equalities.
 *
 * AC 94/06/13
 *
 * Lower and upper bounds are now in the inequalities. So, first, we have
 * to sort these ineq in two groups, one for the lower bound and one for
 * the upper bound.
 *
 * AP 95/01/30 */

void get_bounds_expression(sys, lt, lb, ub)

     Psyslist     sys;
     list         lt, *lb, *ub;
{
  Psyslist     p = sys;
  Psysteme     sc, sc_lower = sc_new(), sc_upper = sc_new();
  Pcontrainte  cont;
  Pvecteur     vect;
  expression   lower = expression_undefined, upper = expression_undefined;
  Value        val;
  list         l = lt;
  entity       ent;
  
  for (l = lt; l != NIL; l = l->cdr)
    {
      sc = p->psys;
      ent = ENTITY(CAR(l));
      
      if (get_debug_level() > 6) {
	fprintf(stderr,"\nSysteme :");
	fprint_psysteme(stderr, sc);
      }
      
      /* Sort the inequalities */
      cont = sc->inegalites;
      while (cont != NULL) {
	if (value_posz_p(vect_coeff((Variable) ent, cont->vecteur)))
	{ sc_add_inegalite(sc_upper, contrainte_dup(cont)); }
	else
	{ sc_add_inegalite(sc_lower, contrainte_dup(cont)); }
	cont = cont->succ;
      }

      if (sc_lower->nb_ineq == 1)
	/* Only one lower bound */
	{
	  vect = vect_dup((sc_lower->inegalites)->vecteur);
	  val = vect_coeff((Variable) ent, vect);
	  value_absolute(val);
	  vect = vect_del_var(vect, (Variable) ent);
	  lower = make_rational_exp(vect, val);
	}
      else if (sc_lower->nb_ineq > 1)	{
	/* More than one lower bound, build a max */
	list llower = NIL;
	for (cont = sc_lower->inegalites; cont != NULL; cont = cont->succ) {
	  vect = vect_dup(cont->vecteur);
	  val = vect_coeff((Variable) ent, vect);
	  value_absolute(val);
	  vect = vect_del_var(vect, (Variable) ent);
	  llower = CONS(EXPRESSION, make_rational_exp(vect, val),
			llower);
	}  
	lower = make_expression(make_syntax(is_syntax_call,
					    make_call(entity_intrinsic("MAX"),
						      llower)),
				normalized_undefined);
      }
      else
	user_error("get_bounds_expression", "\n No lower bound\n");
      
      if (sc_upper->nb_ineq == 1) {
	/* Only one upper bound */
	vect = vect_dup((sc_upper->inegalites)->vecteur);
	val = vect_coeff((Variable) ent, vect);
	value_absolute(val);
	vect = vect_del_var(vect, (Variable) ent);
	vect_chg_sgn(vect);
	upper = make_rational_exp(vect, val);
      }
      else if(sc_upper->nb_ineq > 1) {
	/* More than one upper bound, build a min */
	list lupper = NIL;
	for (cont = sc_upper->inegalites; cont != NULL; cont = cont->succ) {
	  vect = vect_dup(cont->vecteur);
	  val = vect_coeff((Variable) ent, vect);
	  value_absolute(val);
	  vect = vect_del_var(vect, (Variable) ent);
	  vect_chg_sgn(vect);
	  lupper = CONS(EXPRESSION,
			make_rational_exp(vect, val), lupper);
	}
	upper = make_expression(make_syntax(is_syntax_call,
					    make_call(entity_intrinsic("MIN"),
						      lupper)),
				normalized_undefined);
      }
      else
	{
	  user_error("get_bounds_expression", "\n No upper bound\n");
	}
      ADD_ELEMENT_TO_LIST((*lb), EXPRESSION, lower);
      ADD_ELEMENT_TO_LIST((*ub), EXPRESSION, upper);

      if (get_debug_level() > 6) {
	fprintf(stderr,
		"\nNew lb and ub expressions :\n\tLB: %s\n\tUB: %s\n",
		words_to_string(words_expression(lower)),
		words_to_string(words_expression(upper)));
      }

    }
}


/*=========================================================================*/
/* Psyslist separate_variables(ps, l)
 *
 * We have a psystem ps containing all variables in l, and we want to have
 * a list of psystem where each system is containing the variable of its
 * rank.
 *
 *    Ex: variable t1, t2, p1.
 *  => we will have 3 systems, ps1, ps2, ps3.
 *     ps1 = f(n, t1)  (n structure parameters);
 *     ps2 = f(t1, n, t2);
 *     ps3 = f(p1, t1, t2);
 *
 * We also separate the bounds and put the constraints on the upper bound
 * in the equalities and the constraints on the lower bound in the
 * inequalities.
 *
 * AC 94/04/05 */

Psyslist separate_variables(ps, l, sp, c)
Psysteme     ps, *sp;
list         l;
int          c;
{
  list         lp, lr = gen_nreverse(l);
  Psyslist     lsys = NULL, lsys_aux;
  entity       ent;
  Pcontrainte  cont;
  int          i;
  Psysteme     ps_aux;
  
  (*sp) = SC_UNDEFINED;
  
  if(l != NIL) {
    for (lp = lr; lp != NIL; lp = lp->cdr) {
      Psysteme   cps = sc_new();
      ent = ENTITY(CAR(lp));
      for (cont = ps->inegalites; cont != NULL; cont = cont->succ) {
	if (base_contains_variable_p(cont->vecteur, (Variable) ent)) {
          if (value_neg_p(vect_coeff((Variable) ent, cont->vecteur)))
            { sc_add_inegalite(cps, contrainte_dup(cont)); }
          else
            { sc_add_egalite(cps, contrainte_dup(cont)); }
          cont->vecteur = VECTEUR_NUL;
	}
      }
      for (cont = ps->egalites; cont != NULL; cont = cont->succ) {
	Pvecteur pv = cont->vecteur;
	if (base_contains_variable_p(pv, (Variable) ent)) {
	  if (value_neg_p(vect_coeff((Variable) ent, pv))) {
	    sc_add_inegalite(cps, contrainte_make(vect_dup(pv)));
	    vect_chg_sgn(pv);
	    sc_add_egalite(cps, contrainte_make(vect_dup(pv))); 
	  }
	  else {
	    sc_add_egalite(cps, contrainte_make(vect_dup(pv))); 
	    vect_chg_sgn(pv);
	    sc_add_inegalite(cps, contrainte_make(vect_dup(pv)));
	  }
	  cont->vecteur = VECTEUR_NUL;
	}
      }
      sc_normalize(cps);
      lsys = add_sc_to_sclist(cps, lsys);
    }
    
    if (get_debug_level() > 5) {
      fprintf(stderr,"\nListe de psystemes construite :");
      sl_fprint(stderr,lsys,entity_local_name);
    }

    lsys_aux = lsys;
    for (i = 1; i <= c; i++) {
      lsys_aux = lsys_aux->succ;
    }

    for (; lsys_aux != NULL; lsys_aux = lsys_aux->succ) {
      ps_aux = lsys_aux->psys;

      /* add inequalities on the max */
      while (ps_aux->egalites != NULL) {
	cont = ps_aux->egalites;
	ps_aux->egalites = (ps_aux->egalites)->succ;
	cont->succ = NULL;
	sc_add_inegalite(ps_aux, cont);
      }
      (*sp) = sc_append(*sp, sc_dup(ps_aux));
    }
  }
    
  return(lsys); 
}

/*=========================================================================*/
/* Psyslist separate_variables_2(ps, l): we have a psystem ps containing all
 * variables in l, and we want to have a list of psystem where each system
 * is containing the variable of its rank.
 *    Ex: variable t1, t2, p1.
 *  => we will have 3 systems, ps1, ps2, ps3.
 *     ps1 = f(n, t1)  (n structure parameters);
 *     ps2 = f(t1, n, t2);
 *     ps3 = f(p1, t1, t2);
 *
 * same function as above but we do not distinguish the min from the max.
 *
 * AC 94/04/05
 */

Psyslist separate_variables_2(ps, l, sp, c)

 Psysteme     ps, *sp;
 list         l;
 int          c;
{
 list         lp, lr = gen_nreverse(l);
 Psyslist     lsys = NULL, lsys_aux, lsys_aux2;
 entity       ent;
 Pcontrainte  cont;
 int          i;
 Psysteme     ps_aux;

 (*sp) = SC_UNDEFINED;

 for (lp = lr; lp != NIL; lp = lp->cdr)
   {
    Psysteme   cps = sc_new();
    ent = ENTITY(CAR(lp));
    for (cont = ps->inegalites; cont != NULL; cont = cont->succ)
      {
       if (base_contains_variable_p(cont->vecteur, (Variable) ent))
	 {
          sc_add_inegalite(cps, contrainte_dup(cont));
          cont->vecteur = VECTEUR_NUL;
	 }
      }
    sc_normalize(cps);
    lsys = add_sc_to_sclist(cps, lsys);
   }

 if (get_debug_level() > 5)
   {
    fprintf(stderr,"\nListe de psystemes construite 2:");
    sl_fprint(stderr,lsys,entity_local_name);
   }

 if (c == 1)
   {
    lsys_aux = lsys->succ;

    for (; lsys_aux != NULL; lsys_aux = lsys_aux->succ)
      {
       ps_aux = lsys_aux->psys;

       /* add inequalities on the max */
       while (ps_aux->egalites != NULL)
         {
          cont = ps_aux->egalites;
          ps_aux->egalites = (ps_aux->egalites)->succ;
          cont->succ = NULL;
          sc_add_inegalite(ps_aux, cont);
         }
       (*sp) = sc_append(*sp, sc_dup(ps_aux));
      }
   }
 else
   {
    lsys_aux = lsys->succ;
    lsys_aux2 = lsys;
    for (i = 2; i <= c; i++)
      {
       lsys_aux = lsys_aux->succ;
       lsys_aux2 = lsys_aux2->succ;
      }

    for (; lsys_aux != NULL; lsys_aux = lsys_aux->succ)
      {
       ps_aux = lsys_aux->psys;
       /* add inequalities on the max */
       while (ps_aux->egalites != NULL)
         {
          cont = ps_aux->egalites;
          ps_aux->egalites = (ps_aux->egalites)->succ;
          cont->succ = NULL;
          sc_add_inegalite(ps_aux, cont);
         }   
       (*sp) = sc_append(*sp, sc_dup(ps_aux));
      }
   }

 if (get_debug_level() > 5)
   {
    fprintf(stderr,"\nListe de psystemes construite en final 2:");
    sl_fprint(stderr,lsys,entity_local_name);
   }

 return(reverse_psyslist(lsys)); 
}
