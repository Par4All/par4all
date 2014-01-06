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
 /*
  * transformer package - convex hull computation
  *
  * Francois Irigoin, 21 April 1990
  */

#include <stdio.h>

#include "genC.h"
#include "linear.h"

#include "ri.h"
#include "ri-util.h"

#include "misc.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

/* temporarily, for ifdebug */
#include "transformer.h"

/* This function used to have side effects on arguments t1 and t2 */
static transformer transformer_convex_hulls
(transformer t1, transformer t2, Psysteme (*method)(Psysteme, Psysteme))
{
  transformer t = transformer_undefined;


  pips_debug(6,"begin\n");
  ifdebug(6) {
    pips_debug(6, "convex hull t1 (%p):\n", t1);
    dump_transformer(t1) ;
  }
  ifdebug(6) {
    pips_debug(6, "convex hull t2 (%p):\n", t2);
    dump_transformer(t2) ;
  }

  /* If one of the transformers is empty, you do not want to union
   * the arguments
   */
  if(transformer_empty_p(t1)) {
    t = transformer_dup(t2);
  }
  else if(transformer_empty_p(t2)) {
    t = transformer_dup(t1);
  }
  else {
    Psysteme r1 = sc_dup((Psysteme) predicate_system(transformer_relation(t1)));
    Psysteme r2 = sc_dup((Psysteme) predicate_system(transformer_relation(t2)));
    Psysteme r;
    Pbase b1;
    Pbase b2;
    Pbase b1_min = sc_to_minimal_basis(r1);
    Pbase b2_min = sc_to_minimal_basis(r2);
    Pbase b;
    /* Pbase b_min; */ /* Could be used if useless dimensions are not
       detected at lower level(s) */
    int eq_count1 = 0;
    int eq_count2 = 0;
    int ineq_count1 = 0;
    int ineq_count2 = 0;

    t = transformer_identity();
    transformer_arguments(t) = 
      arguments_union(transformer_arguments(t1),
		      transformer_arguments(t2));

    /* add implicit equality constraints in r1 and r2 */
    /* These equalities should only be added if the variable explicitly
       appears in at least one constraint in the other constraint
       system. Mathematical proof?. */
    FOREACH(ENTITY, a,  transformer_arguments(t1)) {
      if(!entity_is_argument_p(a, transformer_arguments(t2))) {
	entity a_new = entity_to_new_value(a);
	if(base_contains_variable_p(b1_min, (Variable) a_new)) {
	  entity a_old = entity_to_old_value(a);
	  Pvecteur eq = vect_new((Variable) a_new, -1);
	  vect_chg_coeff(&eq, (Variable) a_old, 1);
	  r2 = sc_equation_add(r2, contrainte_make(eq));
	  eq_count2++;
	  if(basic_logical_p(variable_basic(type_variable(entity_type(a))))) {
	    /* add implicit constraints for bool variables */
	    Pvecteur ineq1 = vect_new((Variable) a_new, VALUE_ONE);
	    Pvecteur ineq2 = vect_new((Variable) a_new, VALUE_MONE);

	    vect_add_elem(&ineq1, TCST, VALUE_MONE);
	    r2 = sc_inequality_add(r2, contrainte_make(ineq1));
	    r2 = sc_inequality_add(r2, contrainte_make(ineq2));
	    ineq_count2++;
	  }
	}
      }
    }
    base_rm(b1_min);
    FOREACH(ENTITY, a, transformer_arguments(t2)) {
      if(!entity_is_argument_p(a, transformer_arguments(t1))) {
	entity a_new = entity_to_new_value(a);
	if(base_contains_variable_p(b2_min, (Variable) a_new)) {
	  entity a_old = entity_to_old_value(a);
	  Pvecteur eq = vect_new((Variable) a_new, -1);
	  vect_chg_coeff(&eq, (Variable) a_old, 1);
	  r1 = sc_equation_add(r1, contrainte_make(eq));
	  eq_count1++;
	  if(basic_logical_p(variable_basic(type_variable(entity_type(a))))) {
	    /* add implicit constraints for bool variables */
	    Pvecteur ineq1 = vect_new((Variable) a_new, VALUE_ONE);
	    Pvecteur ineq2 = vect_new((Variable) a_new, VALUE_MONE);

	    vect_add_elem(&ineq1, TCST, VALUE_MONE);
	    r1 = sc_inequality_add(r1, contrainte_make(ineq1));
	    r1 = sc_inequality_add(r1, contrainte_make(ineq2));
	    ineq_count1++;
	  }
	}
      }
    }
    base_rm(b2_min);

    pips_debug(6,"Number of equations added to t1: %d, to t2: %d\n"
	       "Number of inequalities added to t1: %d, to t2: %d\n", 
	       eq_count1, eq_count2, ineq_count1, ineq_count2);

    /* update bases using their "union"; convex hull has to be computed 
       relatively to ONE space */
    b1 = r1->base;
    b2 = r2->base;
    b = base_union(b1, b2);
    base_rm(b1);
    base_rm(b2);
    /* b is duplicated because it may be later freed by (*method)()
     * FI->CA: To be changed when (*method)() is cleaned up
     */
    sc_base(r1) = base_dup(b);
    /* please, no sharing between Psysteme's */
    sc_base(r2) = base_dup(b);
    sc_dimension(r1) = base_dimension(b);
    sc_dimension(r2) = sc_dimension(r1);

    /* meet operation (with no side-effect on arguments r1 and r2) */
    r = (* method)(r1, r2);

    /* There is no way to distinguish between SC_RN and SC_EMPY since
       both are defined as NULL */
    if(SC_EMPTY_P(r)) {
      /* FI: this could be eliminated if SC_EMPTY was really usable; 27/5/93 */
      /* and replaced by a SC_UNDEFINED_P() and pips_error() */
      r = sc_empty(BASE_NULLE);
    }
    else {
      base_rm(b);
      b = BASE_NULLE;
    }

    sc_rm(r1);
    sc_rm(r2);

    if(sc_empty_p(r)) {
      /* To eliminate the arguments in case r is really empty */
      t = empty_transformer(t);
    }
    else
      predicate_system(transformer_relation(t)) = r;

  }

  ifdebug(6) {
    pips_debug(6, "convex hull, t (%p):\n", t);
    dump_transformer(t) ;
  }

  pips_debug(6, "end\n");

  return t;
}

/* transformer transformer_convex_hull(t1, t2): compute convex hull for t1
 * and t2; t1 and t2 are slightly modified to give them the same basis; else
 * convex hull means nothing; some of the work is duplicated in sc_enveloppe;
 * however their "relation" fields are preserved; the whole thing is pretty
 * badly designed; shame on Francois! FI, 24 August 1990
 */
transformer transformer_convex_hull(transformer t1, transformer t2)
{
  /* return transformer_convex_hulls(t1, t2, sc_enveloppe);  */
  /* return transformer_convex_hulls(t1, t2, sc_enveloppe_chernikova); */
  /* return transformer_convex_hulls(t1, t2, sc_common_projection_convex_hull);
   */
/*   t1 = transformer_normalize(t1, 4); */
/*   t2 = transformer_normalize(t2, 4); */
  t1 = transformer_normalize(t1, 2);
  t2 = transformer_normalize(t2, 2);
  return transformer_convex_hulls(t1, t2, cute_convex_union);
}

/* I removed this because I do not want to port the polyedre library
 * to use "Value". If you want this function, do the port! FC 07/96
 */
/*
transformer transformer_fast_convex_hull(t1, t2)
transformer t1;
transformer t2;
{
    return transformer_convex_hulls(t1, t2, sc_fast_convex_hull);
}
*/

/*
transformer transformer_chernikova_convex_hull(t1, t2)
transformer t1;
transformer t2;
{
    return transformer_convex_hulls(t1, t2, sc_enveloppe_chernikova);
}
*/
