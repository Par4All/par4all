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
/* Functions to deal with transformer lists.
 *
 * Tansformer lists are used to delay convex hulls and to refine the
 * precondition computation of loops by putting aside the identity
 * transformer.
 *
 * If there is an identity transformer in the list, it is supposed to
 * be the first one in the list.
 *
 * However, some control paths are almost identity transformers
 * because the store is apparently unchanged. However, they contain
 * predicates on the possible values. Although they have no arguments,
 * and hence, the store is left unchanged, they are different from the
 * identity transformer bebcause the relation is smaller.
 *
 * So it should be up to the function using the transformer list to
 * decide how to cope with transformers that restrict the identity
 * transition.
 */
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h>

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "properties.h"

#include "misc.h"
/* Must not be used, beware of library cycles: #include "semantics.h" */

#include "transformer.h"

/* Union of two lists

   If the list includes the identity transformer, it must be the first
   in the list and be nowehere else

   It is not clear if transformer lists will be stored in hash
   tables... Hence I do not know if the input list should be freed,
   reused or left unshared. To be conservative, no alias is created.
 */
list merge_transformer_lists(list tl1, list tl2)
{
  list ntl = NIL;
  list ntl1 = NIL;
  list ntl2 = NIL;

  if(ENDP(tl1))
    ntl = gen_full_copy_list(tl2);
  else if(ENDP(tl2))
    ntl = gen_full_copy_list(tl1);
  else {
    /* Do we have to worry about different bases in transformers? */
    transformer t1 = TRANSFORMER(CAR(tl1));
    transformer t2 = TRANSFORMER(CAR(tl2));

    /* Too much information is sometimes lost with this
       simplification */
    /*
    if(ENDP(transformer_arguments(t1))) {
      free_transformer(t1);
      t1 = transformer_identity();
    }

    if(ENDP(transformer_arguments(t2))) {
      free_transformer(t2);
      t2 = transformer_identity();
    }
    */

    if(transformer_identity_p(t1) || transformer_identity_p(t2)) {
      ntl = CONS(TRANSFORMER, transformer_identity(), NIL);
    }
    if(transformer_identity_p(t1))
      ntl1 = gen_full_copy_list(CDR(tl1));
    else
      ntl1 = gen_full_copy_list(tl1);
    if(transformer_identity_p(t2))
      ntl2 = gen_full_copy_list(CDR(tl2));
    else
      ntl2 = gen_full_copy_list(tl2);
    ntl1 = gen_nconc(ntl1, ntl2);
    ntl = gen_nconc(ntl, ntl1);
  }

  ifdebug(1) {
    int ntll = gen_length(ntl);
    int tl1l = gen_length(tl1);
    int tl2l = gen_length(tl2);
    pips_assert("The new list is about the sum of the input lists.\n",
		ntll>=tl1l+tl2l-1 && ntll<=tl1l+tl2l);
    pips_assert("The new transformer list is legal",
		check_transformer_list(ntl));
  }
  return ntl;
}


/* What do we want to impose?
 *
 * 1. Only one identity transformer
 *
 * 2. Common basis?
 *
 * 3. No empty transformer
 *
 */
bool check_transformer_list(list tl)
{
  bool identity_p = false;
  bool one_p = false; // useless for the time being
  one_p=one_p;

  if(ENDP(tl)) {
    /* The empty transformer list is used to represent the empty
       transformer... */
    ;
  }
  else {
    FOREACH(TRANSFORMER, tf, tl) {
      if(transformer_identity_p(tf)) {
	if(identity_p) {
	  one_p = false;
	  pips_internal_error("Two identity transformers in one list.");
	}
	else {
	  identity_p = true;
	  one_p = true;
	}
      }
    }
    if(identity_p) {
      /* It must be the first one */
      if(!transformer_identity_p(TRANSFORMER(CAR(tl))))
	pips_internal_error("The identity transformer is not the list header.");
    }

    FOREACH(TRANSFORMER, tf, tl) {
      if(transformer_empty_p(tf))
	pips_internal_error("An empty transformer has been found.");
    }
  }

  return true;
}

/* each transformer of tl1 must be combined with a transformer of
   tl2, including the identity transformer. If an identity
   transformer is generated and if identity transformers are always
   first in the list, it will again be first in the returned list. */
list combine_transformer_lists(list tl1, list tl2)
{
  list ntl = NIL;
  int n1 = gen_length(tl1);
  int n2 = gen_length(tl2);
  int en = 0;
  int nn = -1;

  pips_assert("tl1 is OK\n", check_transformer_list(tl1));
  pips_assert("tl2 is OK\n", check_transformer_list(tl2));

  FOREACH(TRANSFORMER, t1, tl1) {
    FOREACH(TRANSFORMER, t2, tl2) {
      transformer nt = transformer_combine(copy_transformer(t1), t2);

      // transformer_empty_p() is not strong enough currently
      // It does not detect that k<= 2 && k>=3 is empty...
      nt = transformer_normalize(nt, 2);
      if(!transformer_empty_p(nt))
	ntl = CONS(TRANSFORMER, nt, ntl);
      else
	en++;
    }
  }
  ntl = gen_nreverse(ntl);

  nn = gen_length(ntl);
  pips_assert("ntl is OK\n", check_transformer_list(ntl));
  pips_assert("nn is n1*n2-en", nn==n1*n2-en);

  return ntl;
}

/* each transformer of tl1 must be applied to each precondition of
   tl2, including the identity transformer. If an identity
   transformer is generated and if identity transformers are always
   first in the list, it will again be first in the returned
   list. Empty preconditions are not preserved in the returned
   list. An empty list is unfeasible.

   if exclude_p==false, return U_i1 U_i2 apply(t_i1, p_i2);
   else return U_i1 U_i2!=i1 apply(t_i1, p_i2);
 */
list apply_transformer_lists_generic(list tl1, list tl2, bool exclude_p)
{
  list ntl = NIL;
  int n1 = gen_length(tl1);
  int n2 = gen_length(tl2);
  int en = 0; // number of empty preconditions generated
  int sn = 0; // number of excluded/skipped preconditions
  int nn = -1; // number of preconditions in the result
  int i1 = 0;

  // FI: not true is option keep_p has been used to maintain
  //|tl_1|==|tl_2| which is useful when exclude_p is TRUE
  if(!exclude_p) {
    pips_assert("tl1 is OK\n", check_transformer_list(tl1));
    pips_assert("tl2 is OK\n", check_transformer_list(tl2));
  }

  FOREACH(TRANSFORMER, t1, tl1) {
    int i2 = 0;
    i1++;
    FOREACH(TRANSFORMER, t2, tl2) {
      i2++;
      if(i1!=i2 && exclude_p) {
	transformer nt = transformer_apply(t1, t2);

	if(!transformer_empty_p(nt))
	  ntl = CONS(TRANSFORMER, nt, ntl);
	else
	  en++;
      }
      else if(exclude_p)
	sn++;
    }
  }
  ntl = gen_nreverse(ntl);

  nn = gen_length(ntl);
  pips_assert("nn is n1*n2-en", nn==n1*n2-en-sn);
  //FI: there is no guarantee here that the identity transformer is
  //not returned multiple times... although it would make sense if the
  //input lists are properly sorted.
  pips_assert("ntl is OK\n", check_transformer_list(ntl));

  return ntl;
}

list apply_transformer_lists(list tl1, list tl2)
{
  return apply_transformer_lists_generic(tl1, tl2, false);
}

list apply_transformer_lists_with_exclusion(list tl1, list tl2)
{
  return apply_transformer_lists_generic(tl1, tl2, true);
}

/* Eliminate empty transformers and keep at most one identity
 * transformer, placed as first list element.
 *
 * check_transformer_list(ntfl) should be TRUE.
 *
 * tfl is fully destroyed (to  avoid memory leaks and nightmares); to
 * be more efficient, the transformers moved from the input list into
 * the output list should be detached from the input list and then the
 * input list could be fully destroyed without having to copy any
 * transformers; but FOREACH operates at too high a level for this.
 */
list clean_up_transformer_list(list tfl)
{
  list ntfl = NIL;
  bool identity_p = false;

  FOREACH(TRANSFORMER, tf, tfl) {
    bool tf_identity_p = transformer_identity_p(tf);
    if(!tf_identity_p && !transformer_empty_p(tf))
      ntfl = CONS(TRANSFORMER, copy_transformer(tf), ntfl);
    identity_p = identity_p || tf_identity_p;
  }
  gen_full_free_list(tfl);
  ntfl = gen_nreverse(ntfl);
  if(identity_p)
    ntfl = CONS(TRANSFORMER, transformer_identity(), ntfl);
  return ntfl;
}

/* Transformer two transformers into a correct transformer list
 *
 * Could be generalized to any number of transformers using a
 * varargs... and more thinking.
 *
 * Two transformers are obtained for loops that may be skipped or
 * entered and for tests whose condition is not statically decidable.
 */
list two_transformers_to_list(transformer t1, transformer t2)
{
  list tl = NIL;
  if(transformer_empty_p(t1)) {
    if(transformer_empty_p(t2)) {
      tl = NIL;
    }
    else {
      tl = CONS(TRANSFORMER, t2, NIL);
    }
  }
  else {
    if(transformer_empty_p(t2)) {
      tl = CONS(TRANSFORMER, t1, NIL);
    }
    else {

      /* This is a very dangerous step that should not always be
	 taken. It is useful to ease the detection of identity
	 paths, but it looses a lot of information. So almost
	 identity path might simply be better identified elsewhere */
      /*
      if(ENDP(transformer_arguments(t1))) {
	free_transformer(t1);
	t1 = transformer_identity();
      }

      if(ENDP(transformer_arguments(t2))) {
	free_transformer(t2);
	t2 = transformer_identity();
      }
      */

      if(transformer_identity_p(t1)) {
	if(transformer_identity_p(t2)) {
	  tl = CONS(TRANSFORMER, t1, NIL);
	  free_transformer(t2);
	}
	else {
	  tl = CONS(TRANSFORMER, t1,
		    CONS(TRANSFORMER, t2, NIL));
	}
      }
      else {
	if(transformer_identity_p(t2)) {
	  tl = CONS(TRANSFORMER, t2,
		    CONS(TRANSFORMER, t1, NIL));
	}
	else {
	  tl = CONS(TRANSFORMER, t1,
		    CONS(TRANSFORMER, t2, NIL));
	}
      }
    }
  }
  return tl;
}

/* Reduce the transformer list with the convex hull operator.
 *
 * If active_p is true, skip transformers that do not update the
 * state. Beyond the identity transformer, any transformer without
 * arguments does not really update the state, although it may
 * restrict it.
 *
 * A new transformer is always allocated. The transformers in the
 * transformer list ltl are freed.
 */
transformer generic_transformer_list_to_transformer(list ltl, bool active_p)
{
  transformer ltf = transformer_undefined; // list transformer

  if(ENDP(ltl))
    ltf = transformer_empty();
  else {
    list ctl = ltl;
    /* Look for the first useful transformer in the list */
    FOREACH(TRANSFORMER, tf, ltl) {
      if(!active_p || !ENDP(transformer_arguments(tf))) {
	ltf = copy_transformer(tf);
	free_transformer(tf);
	POP(ctl);
	break;
      }
      POP(ctl);
    }
    if(ENDP(ctl)) {
      if(transformer_undefined_p(ltf))
	/* Only range conditions have been found: the store is
	   restricted but not changed. */
	ltf = transformer_identity();
    }
    else {
      /* Take care of the following useful transformers */
      while(!ENDP(ctl)) {
	/* Look for the next useful transformer in the list */
	FOREACH(TRANSFORMER, tf, ctl) {
	  if(!active_p || !ENDP(transformer_arguments(tf))) {
	    transformer ntf = copy_transformer(tf);
	    transformer ptf = ltf;
	    ltf = transformer_convex_hull(ptf, ntf);
	    free_transformer(ntf);
	    free_transformer(ptf);
	    POP(ctl);
	    break;
	  }
	  POP(ctl);
	}
      }
    }
  }

  return ltf;
}

/* Reduce the transformer list ltl to one transformer using the convex
 *  hull operator.
 */
transformer transformer_list_to_transformer(list ltl)
{
  return generic_transformer_list_to_transformer(ltl, false);
}

/* Reduce the sublist of active transformers in the transformer list
 * ltl to one transformer using the convex hull operator. An active
 * transformer is a transformer with argument(s): at least one value
 * is changed.
 *
 * Note: a hidden identity transformer such as T(i) {i==i#init} is not
 * detected.
 */
transformer active_transformer_list_to_transformer(list ltl)
{
  return generic_transformer_list_to_transformer(ltl, true);
}

// Remove all inactive transformers from ltl and generate a new list
// with copied elements
list transformer_list_to_active_transformer_list(list tl)
{
  list atl = NIL;

  FOREACH(TRANSFORMER, tf, tl) {
    if(!ENDP(transformer_arguments(tf)))
      atl = CONS(TRANSFORMER, copy_transformer(tf), atl);
  }

  atl = gen_nreverse(atl);

  return atl;
}


/* Compute the precondition of a loop whose body transformers T_i are
 * in transformer list tl and whose condition is modelized by
 * transformer c_t. The precondition of iteration 0 is p_0.
 *
 * We need a developped formulae for P*=(U_i T_i)^*P_0... to postpone
 * the convex hulls as much as possible
 *
 * For instance, and this is a heuristics:
 *
 * P_0 is known as pre_init
 *
 * P_1 = U_i T_i(P_0)
 *
 * P_2 = U_i U_j T_i(T_j(P_0))
 *
 * P_3^s = U_i T_i^+(P_0)  --- only one path is used
 *
 * P_3^+ = U_i U_j!=i T^+_i(T_j(T^*(P_1)))  --- at least two
 *                                              paths are used
 *
 * which would make more sense when i and j are in
 * [0..1]. Note that in P_3, T_j and T^+_i could be recomputed
 * wrt P_1 instead of pre_fuzzy... which is not provided
 *
 * Maybe T^*=(U_i T_i)^* could/should be computed as (U_i T_i*)* but
 * it is not clear how all the useful conditions could be taken into
 * account.
 *
 * A more accurate approach would use several developped formulae for
 * P* and an intersection of their results.
 */
static transformer
transformer_list_closure_to_precondition_depth_two(list tl,
						   transformer c_t,
						   transformer p_0)
{
  list ntl = transformers_combine(gen_full_copy_list(tl), c_t);

  list p_1_l = transformers_apply(ntl, p_0);
  transformer p_1 = transformer_list_to_transformer(p_1_l);

  list t_2_l = combine_transformer_lists(ntl, ntl);
  list p_2_l = transformers_apply(t_2_l, p_0);
  transformer p_2 = transformer_list_to_transformer(p_2_l);

  list itcl = transformers_derivative_fix_point(ntl); // individual
						      // transformer closures
  itcl = transformers_combine(itcl, c_t);
  list itcl_plus = gen_full_copy_list(itcl); // to preserve ictl
  itcl_plus = one_to_one_transformers_combine(itcl_plus, ntl);
  list p_3_l = transformers_apply(itcl_plus, p_0);
  transformer p_3 = transformer_list_to_transformer(gen_full_copy_list(p_3_l));

  transformer t_star = transformer_undefined;
  if(!get_bool_property("SEMANTICS_USE_DERIVATIVE_LIST")) {
  // Not satisfying: works only for the whole space, not for subpaces
  // left untouched
    t_star = active_transformer_list_to_transformer(itcl);
  }
  else
    t_star = transformer_list_transitive_closure(itcl);
  transformer p_4_1 = transformer_apply(t_star, p_1); // one + * iteration
  list p_4_2_l = transformers_apply_and_keep_all(ntl, p_4_1); // another iteration
  pips_assert("itcl_plus and p_4_2_l have the same numer of elements",
	      gen_length(itcl_plus)==gen_length(p_4_2_l));
  list p_4_3_l = apply_transformer_lists_with_exclusion(itcl_plus, p_4_2_l);
  transformer p_4 = transformer_list_to_transformer(p_4_3_l);

  transformer p_star = transformer_undefined;

  ifdebug(8) {
    pips_debug(8, "p_0:\n");
    print_transformer(p_0);
    pips_debug(8, "p_1:\n");
    print_transformer(p_1);
    pips_debug(8, "p_2:\n");
    print_transformer(p_2);
    pips_debug(8, "p_3:\n");
    print_transformer(p_3);
    pips_debug(8, "p_4:\n");
    print_transformer(p_4);
  }

  // reduce p_0, p_1, p_2, p_3 and p_4 to p_star
  transformer p_01 = transformer_convex_hull(p_0, p_1);
  transformer p_012 = transformer_convex_hull(p_01, p_2);
  transformer p_0123 = transformer_convex_hull(p_012, p_3);
  p_star = transformer_convex_hull(p_0123, p_4);

  ifdebug(8) {
    pips_debug(8, "p_star:\n");
    print_transformer(p_star);
  }

  // Clean up all intermediate variables
  gen_full_free_list(ntl);
  //gen_full_free_list(p_1_l);
  free_transformer(p_1);
  //gen_full_free_list(t_2_l);
  //gen_full_free_list(p_2_l);
  free_transformer(p_2);
  //gen_full_free_list(itcl);
  gen_full_free_list(itcl_plus);
  //gen_full_free_list(p_3_l);
  free_transformer(p_3);
  free_transformer(t_star);
  free_transformer(p_4_1);
  gen_full_free_list(p_4_2_l);
  //gen_full_free_list(p_4_3_l);
  free_transformer(p_4);
  free_transformer(p_01);
  free_transformer(p_012);
  free_transformer(p_0123);

  return p_star;
}

/* Compute the precondition of a loop whose body transformers T_i are
 * in transformer list tl and whose condition is modelized by
 * transformer c_t. The precondition of iteration 0 is p_0.
 *
 * We need a developped formulae for P*=(U_i T_i)^*P_0... to postpone
 * the convex hulls as much as possible
 *
 * For instance, and this is a heuristics:
 *
 * P_0 is known as pre_init
 *
 * P_l = U_i T_i(P_0)
 *
 * P_2 = U_i U_j T_i(T_j(P_0))
 *
 * P_3^s = U_i T_i^+(P_0)  --- only one path is used
 *
 * P_3^+ = U_i U_j!=i T^+_i(T_j(T^*(P_1)))  --- at least two
 *                                              paths are used
 *
 * which would make more sense when i and j are in
 * [0..1]. Note that in P_3, T_j and T^+_i could be recomputed
 * wrt P_1 instead of pre_fuzzy... which is not provided
 *
 * Maybe T^*=(U_i T_i)^* could/should be computed as (U_i T_i*)* but
 * it is not clear how all the useful conditions could be taken into
 * account.
 *
 * A more accurate approach would use several developped formulae for
 * P* and an intersection of their results.
 */
static transformer
transformer_list_closure_to_precondition_max_depth(list tl,
						   transformer c_t,
						   transformer p_0)
{
  list ntl = transformers_combine(gen_full_copy_list(tl), c_t);
  //pips_assert("ntl is OK", check_transformer_list(ntl));

  list p_1_l = transformers_apply(ntl, p_0);
  transformer p_1 = transformer_list_to_transformer(p_1_l);

  list t_2_l = combine_transformer_lists(ntl, ntl);
  list p_2_l = transformers_apply(t_2_l, p_0);
  transformer p_2 = transformer_list_to_transformer(p_2_l);

  list itcl = transformers_derivative_fix_point(ntl); // individual
						      // transformer closures
  itcl = transformers_combine(itcl, c_t);
  list itcl_plus = gen_full_copy_list(itcl); // to preserve ictl
  itcl_plus = one_to_one_transformers_combine(itcl_plus, ntl);
  list p_3_l = transformers_apply(itcl_plus, p_0);
  transformer p_3 = transformer_list_to_transformer(gen_full_copy_list(p_3_l));

  transformer t_star = transformer_undefined;
  if(!get_bool_property("SEMANTICS_USE_DERIVATIVE_LIST")) {
  // Not satisfying: works only for the whole space, not for subpaces
  // left untouched
    t_star = active_transformer_list_to_transformer(itcl);
  }
  else
    t_star = transformer_list_transitive_closure(itcl);
  transformer p_4_1 = transformer_apply(t_star, p_1); // one + * iteration
  list p_4_2_l = transformers_apply_and_keep_all(ntl, p_4_1); // another iteration
  pips_assert("itcl_plus and p_4_2_l have the same numer of elements",
	      gen_length(itcl_plus)==gen_length(p_4_2_l));
  list p_4_3_l = apply_transformer_lists_with_exclusion(itcl_plus, p_4_2_l);
  transformer p_4 = transformer_list_to_transformer(p_4_3_l);

  transformer p_star = transformer_undefined;

  ifdebug(8) {
    pips_debug(8, "p_0:\n");
    print_transformer(p_0);
    pips_debug(8, "p_1:\n");
    print_transformer(p_1);
    pips_debug(8, "p_2:\n");
    print_transformer(p_2);
    pips_debug(8, "p_3:\n");
    print_transformer(p_3);
    pips_debug(8, "p_4:\n");
    print_transformer(p_4);
  }

  // reduce p_0, p_1, p_2, p_3 and p_4 to p_star
  transformer p_01 = transformer_convex_hull(p_0, p_1);
  transformer p_012 = transformer_convex_hull(p_01, p_2);
  transformer p_0123 = transformer_convex_hull(p_012, p_3);
  p_star = transformer_convex_hull(p_0123, p_4);

  ifdebug(8) {
    pips_debug(8, "p_star:\n");
    print_transformer(p_star);
  }

  // Clean up all intermediate variables
  gen_full_free_list(ntl);
  //gen_full_free_list(p_1_l);
  free_transformer(p_1);
  //gen_full_free_list(t_2_l);
  //gen_full_free_list(p_2_l);
  free_transformer(p_2);
  //gen_full_free_list(itcl);
  gen_full_free_list(itcl_plus);
  //gen_full_free_list(p_3_l);
  free_transformer(p_3);
  free_transformer(t_star);
  free_transformer(p_4_1);
  gen_full_free_list(p_4_2_l);
  //gen_full_free_list(p_4_3_l);
  free_transformer(p_4);
  free_transformer(p_01);
  free_transformer(p_012);
  free_transformer(p_0123);

  return p_star;
}

/* Relay to select a heuristic */
transformer transformer_list_closure_to_precondition(list tl,
						     transformer c_t,
						     transformer p_0)
{
  transformer p_star = transformer_undefined;
  const char* h = get_string_property("SEMANTICS_LIST_FIX_POINT_OPERATOR");

  if(strcmp(h, "depth_two")) {
    p_star = transformer_list_closure_to_precondition_depth_two(tl, c_t, p_0);
  }
    else if(strcmp(h, "max_depth")) {
    p_star = transformer_list_closure_to_precondition_max_depth(tl, c_t, p_0);
  }
  else
    pips_user_error("Unknown value \"%s\" for property "
		    "\"SEMANTICS_LIST_FIX_POINT_OPERATOR\"", h);
  return p_star;
}

/* Returns a new list of newly allocated projected transformers. If a
   value of a variable in list proj appears in t of tl, it is
   projected. New transformers are allocated to build the projection
   list*/
list transformer_list_safe_variables_projection(list tl, list proj)
{
  list ptl = NIL;
  FOREACH(TRANSFORMER, t, tl) {
    list apl = NIL; // actual projection list
    transformer nt = copy_transformer(t);
    FOREACH(ENTITY, v, proj) {
      if(entity_is_argument_p(v, transformer_arguments(t))) {
	entity ov = entity_to_old_value(v);
	apl = CONS(ENTITY, ov, apl);
      }
      apl = CONS(ENTITY, v, apl);
    }
    nt = safe_transformer_projection(nt, apl);
    ptl = CONS(TRANSFORMER, nt, ptl);
    gen_free_list(apl);
  }
  ptl = gen_nreverse(ptl);
  return ptl;
}

/* Returns the list of variables modified by at least one transformer
   in tl */
list transformer_list_to_argument(list tl)
{
  list vl = NIL;

  FOREACH(TRANSFORMER, t, tl) {
    FOREACH(ENTITY, v, transformer_arguments(t)) {
      if(!gen_in_list_p(v, vl))
	vl = CONS(ENTITY, v, vl);
    }
  }

  vl = gen_nreverse(vl);

  return vl;
}

/* build a sublist sl of the transformer list tl with transformers that
   modify the value of variable v */
list transformer_list_with_effect(list tl, entity v)
{
  list sl = NIL;

  FOREACH(TRANSFORMER, t, tl){
    if(entity_is_argument_p(v, transformer_arguments(t)))
      sl = CONS(TRANSFORMER, t, sl);
  }
  sl = gen_nreverse(sl);
  return sl;
}

/* returns the list of variables in vl which are not modified by
   transformers belonging to tl-tl_v. tl_v is assumed to be a subset
   of tl. */
list transformer_list_preserved_variables(list vl, list tl, list tl_v)
{
  list pvl = NIL;

  FOREACH(ENTITY, v, vl) {
    bool found_p = false;
    FOREACH(TRANSFORMER, t, tl) {
      if(!gen_in_list_p(t, tl_v)) {
	if(entity_is_argument_p(v, transformer_arguments(t))) {
	  found_p = true;
	  break;
	}
      }
    }
    if(!found_p) {
      pvl = CONS(ENTITY, v, pvl);
    }
  }

  pvl = gen_nreverse(pvl);

  return pvl;
}

/* When some variables are not modified by some transformers, use
   projections on subsets to increase the number of identity
   transformers and to increase the accuracy of the loop precondition.

   The preconditions obtained with the different projections are
   intersected.

   FI: this may be useless when derivatives are computed before the
   convex union. No. This was due to a bug in the computation of list
   of derivaties.

   FI: this should be mathematically useless but it useful when a
   heuristic is used to compute the invariant. The number of
   transitions is reduced and hence a limited number of combinations
   is more likely to end up with a precise result.
 */
transformer transformer_list_multiple_closure_to_precondition(list tl,
							      transformer c_t,
							      transformer p_0)
{
  transformer prec = transformer_list_closure_to_precondition(tl, c_t, p_0);

  if(get_bool_property("SEMANTICS_USE_LIST_PROJECTION")) {
    list al = transformer_arguments(prec);
    list vl = transformer_list_to_argument(tl); // list of modified
    // variables
    // FI: this is too strong vl must only be included in transformer_arguments(prec)
    //pips_assert("all modified variables are argument of the global precondition",
    //arguments_equal_p(vl, transformer_arguments(prec)));
    FOREACH(ENTITY, v, vl) {
      // FI: the same projection could be obtained for different
      // variables v and the computation should not be performed again
      // but I choose not to memoize past lists tl_v
      list tl_v = transformer_list_with_effect(tl, v);

      if(gen_length(tl_v)<gen_length(tl)) {
	list keep_v = transformer_list_preserved_variables(vl, tl, tl_v);
	list proj_v = arguments_difference(vl, keep_v);
	list ptl_v = transformer_list_safe_variables_projection(tl_v, proj_v);
	transformer p_0_v
	  = safe_transformer_projection(copy_transformer(p_0), proj_v);
	transformer c_t_v
	  = safe_transformer_projection(copy_transformer(c_t), proj_v);
	transformer prec_v
	  = transformer_list_closure_to_precondition(ptl_v, c_t_v, p_0_v);
	transformer_arguments(prec_v) = gen_copy_seq(al); // memory leak
	prec = transformer_intersection(prec, prec_v); // memory leak

	free_transformer(prec_v);
	free_transformer(c_t_v);
	free_transformer(p_0_v);
	gen_full_free_list(ptl_v);
	gen_free_list(keep_v);
      }
      gen_free_list(tl_v);
    }

    gen_free_list(vl);
  }
  pips_assert("prec is consistent", transformer_consistency_p(prec));
  return prec;
}

/* Allocate a new constraint system sc(dx) with dx=x'-x and t(x,x')
 *
 * FI: this function should/might be located in fix_point.c
 */
Psysteme transformer_derivative_constraints(transformer t)
{
  Psysteme sc = sc_copy(predicate_system(transformer_relation(t)));
  /* sc is going to be modified and returned */
  /* Do not handle variable which do not appear explicitly in constraints! */
  Pbase b = sc_to_minimal_basis(sc);
  Pbase bv = BASE_NULLE; /* basis vector */

  /* Compute constraints with difference equations */

  for(bv = b; !BASE_NULLE_P(bv); bv = bv->succ) {
    entity oldv = (entity) vecteur_var(bv);

    /* Only generate difference equations if the old value is used */
    if(old_value_entity_p(oldv)) {
      entity var = value_to_variable(oldv);
      entity newv = entity_to_new_value(var);
      entity diffv = entity_to_intermediate_value(var);
      Pvecteur diff = VECTEUR_NUL;
      Pcontrainte eq = CONTRAINTE_UNDEFINED;

      diff = vect_make(diff,
		       (Variable) diffv, VALUE_ONE,
		       (Variable) newv,  VALUE_MONE,
		       (Variable) oldv, VALUE_ONE,
		       TCST, VALUE_ZERO);

      eq = contrainte_make(diff);
      sc = sc_equation_add(sc, eq);
    }
  }

  ifdebug(8) {
    pips_debug(8, "with difference equations=\n");
    sc_fprint(stderr, sc, (char * (*)(Variable)) external_value_name);
  }

  /* Project all variables but differences to get T' */

  sc = sc_projection_ofl_along_variables(sc, b);

  ifdebug(8) {
    pips_debug(8, "Non-homogeneous constraints on derivatives=\n");
    sc_fprint(stderr, sc, (char * (*)(Variable)) external_value_name);
  }

  return sc;
}

/* Computation of an upper approximation of a transitive closure using
 * constraints on the discrete derivative for a list of
 * transformers. Each transformer is used to compute its derivative
 * and the derivatives are unioned by convex hull.
 *
 * The reason for doing this is D(T1) U D(T2) == D(T1 U T2) but the
 * complexity is lower
 *
 * See http://www.cri.ensmp.fr/classement/doc/A-429.pdf
 *
 * Implicit equations, n#new - n#old = 0, are added because they are
 * necessary for the convex hull.
 *
 * Intermediate values are used to encode the differences. For instance,
 * i#int = i#new - i#old
 */
/* This code was cut-and-pasted from
   transformer_derivative_fix_point() but is more general and subsume
   it */
/* transformer transformer_derivative_fix_point(transformer tf)*/
transformer transformer_list_generic_transitive_closure(list tfl, bool star_p)
{
  transformer tc_tf = transformer_identity();

  ifdebug(8) {
    pips_debug(8, "Begin for transformer list %p:\n", tfl);
    print_transformers(tfl);
  }

  if(ENDP(tfl)) {
    if(star_p) {
    /* Since we compute the * transitive closure and not the +
       transitive closure, the fix point is the identity. */
      tc_tf = transformer_identity();
    }
    else {
      tc_tf = transformer_empty();
    }
  }
  else {
    // Pbase ib = base_dup(sc_base(sc)); /* initial and final basis */
    Pbase ib = BASE_NULLE;
    Pbase diffb = BASE_NULLE; /* basis of difference vectors */
    Pbase bv = BASE_NULLE;
    Pbase b = BASE_NULLE;

    /* Compute the global argument list and the global base b */
    list gal = NIL;
    FOREACH(TRANSFORMER, t, tfl) {
      list al = transformer_arguments(t);
      /* Cannot use arguments_union() because a new list is allocated */
      FOREACH(ENTITY, e, al)
	gal = arguments_add_entity(gal, e);

      // FI: this copy is almost entirely memory leaked
      Psysteme sc = sc_copy(predicate_system(transformer_relation(t)));
      // redundant with call to transformer_derivative_constraints(t)
      Pbase tb = sc_to_minimal_basis(sc);
      Pbase nb = base_union(b, tb);
      base_rm(b); // base_union() allocates a new base
      b = nb;
    }

    /* For each transformer t in list tl
     *
     *   compute its derivative constraint system
     *
     *   add the equations for the unchanged variables
     *
     *   compute its convex hull with the current value of sc if sc is
     *   already defined
     */
    Psysteme sc = SC_UNDEFINED;
    FOREACH(TRANSFORMER, t, tfl) {
      Psysteme tsc = transformer_derivative_constraints(t);
      FOREACH(ENTITY,e,gal) {
	if(!entity_is_argument_p(e, transformer_arguments(t))) {
	  /* Add corresponding equation */
	  entity diffv = entity_to_intermediate_value(e);
	  Pvecteur diff = VECTEUR_NUL;
	  Pcontrainte eq = CONTRAINTE_UNDEFINED;

	  diff = vect_make(diff,
			   (Variable) diffv, VALUE_ONE,
			   TCST, VALUE_ZERO);

	  eq = contrainte_make(diff);
	  tsc = sc_equation_add(tsc, eq);
	}
      }
      if(SC_UNDEFINED_P(sc))
	sc = tsc;
      else {
	/* This could be optimized by using the convex hull of a
	   Psystemes list and by keeping the dual representation of
	   the result instead of converting it several time back
	   and forth. */
	Psysteme nsc = cute_convex_union(sc, tsc);
	sc_free(sc);
	sc = nsc;
      }
    }

    /* Multiply the constant terms by the iteration number ik and add a
       positivity constraint for the iteration number ik and then
       eliminate the iteration number ik to get T*(dx). */
    entity ik = make_local_temporary_integer_value_entity();
    //Psysteme sc_t_prime_k = sc_dup(sc);
    //sc_t_prime_k = sc_multiply_constant_terms(sc_t_prime_k, (Variable) ik);
    sc = sc_multiply_constant_terms(sc, (Variable) ik, star_p);
    //Psysteme sc_t_prime_star = sc_projection_ofl(sc_t_prime_k, (Variable) ik);
    sc = sc_projection_ofl(sc, (Variable) ik);
    if(SC_EMPTY_P(sc)) {
      sc = sc_empty(BASE_NULLE);
    }
    else {
      sc->base = base_remove_variable(sc->base, (Variable) ik);
      sc->dimension--;
      // FI: I do not remember nor find how to get rid of local values...
      //sc_rm(sc);
      //sc = sc_t_prime_star;
    }

    ifdebug(8) {
      pips_debug(8, "All invariants on derivatives=\n");
      sc_fprint(stderr, sc, (char * (*)(Variable)) external_value_name);
    }

    /* Difference variables must substituted back to differences
     * between old and new values.
     */

    for(bv = b; !BASE_NULLE_P(bv); bv = bv->succ) {
      entity oldv = (entity) vecteur_var(bv);

      /* Only generate difference equations if the old value is used */
      if(old_value_entity_p(oldv)) {
	entity var = value_to_variable(oldv);
	entity newv = entity_to_new_value(var);
	entity diffv = entity_to_intermediate_value(var);
	Pvecteur diff = VECTEUR_NUL;
	Pcontrainte eq = CONTRAINTE_UNDEFINED;

	diff = vect_make(diff,
			 (Variable) diffv, VALUE_ONE,
			 (Variable) newv,  VALUE_MONE,
			 (Variable) oldv, VALUE_ONE,
			 TCST, VALUE_ZERO);

	eq = contrainte_make(diff);
	sc = sc_equation_add(sc, eq);
	diffb = base_add_variable(diffb, (Variable) diffv);
	ib = base_add_variable(ib, (Variable) oldv);
	ib = base_add_variable(ib, (Variable) newv);
      }
    }

    ifdebug(8) {
      pips_debug(8,
		 "All invariants on derivatives with difference variables=\n");
      sc_fprint(stderr, sc, (char * (*)(Variable)) external_value_name);
    }

    /* Project all difference variables */

    sc = sc_projection_ofl_along_variables(sc, diffb);

    ifdebug(8) {
      pips_debug(8, "All invariants on differences=\n");
      sc_fprint(stderr, sc, (char * (*)(Variable)) external_value_name);
    }

    /* The full basis must be used again */
    base_rm(sc_base(sc)), sc_base(sc) = BASE_NULLE;
    sc_base(sc) = ib;
    sc_dimension(sc) = vect_size(ib);
    base_rm(b), b = BASE_NULLE;

    ifdebug(8) {
      pips_debug(8, "All invariants with proper basis =\n");
      sc_fprint(stderr, sc, (char * (*)(Variable)) external_value_name);
    }

    /* Plug sc back into tc_tf */
    predicate_system(transformer_relation(tc_tf)) = sc;
    transformer_arguments(tc_tf) = gal;

  }
  /* That's all! */

  ifdebug(8) {
    pips_debug(8, "Transitive closure tc_tf=\n");
    fprint_transformer(stderr, tc_tf, (get_variable_name_t) external_value_name);
    transformer_consistency_p(tc_tf);
    pips_debug(8, "end\n");
  }

  return tc_tf;
}

/* Compute (U tfl)* */
transformer transformer_list_transitive_closure(list tfl)
{
  return transformer_list_generic_transitive_closure(tfl, true);
}

/* Compute (U tfl)+ */
transformer transformer_list_transitive_closure_plus(list tfl)
{
  return transformer_list_generic_transitive_closure(tfl, false);
}

/* Internal recursive function. Should be used as
 * transformer_list_combination(). As long as n is not zero, choose
 * all unused transformation numbers in past and call yourself
 * recursively after updating past and ct. past is a bit field. Each
 * bit of past is set to 1 initially because no transformer has been
 * used yet. It is reset to 0 when the transformation has been
 * used. The selected transformer is combined to ct.
 *
 * @param tn is the number of transformers that can be chosen to be
 * combined
 *
 * @param ta[tn] is an array containing the tn transformers to combine
 *
 * @param n is the number of combinations to perform. It is less than
 * tn.
 *
 * @param ct is the current combination of past transformers
 *
 * @param past is a bit field used to keep track of transformers
 * alreasy used to build ct
 *
 * @return the list of all non-empty combinations of n transformers
 */
static list transformer_list_add_combination(int tn,
					     transformer ta[tn],
					     int n,
					     transformer ct,
					     int past)
{
  list cl = NIL;

  if(n>0) {
    int k = 1; // to select a non-zero bit in past
    int ti; // transformation index
    bool found_p = false;
    for(ti=0; ti<n;ti++) {
      if(k&past) {
	// this transformation is selectable, because it has not been
	// selected yet
	int npast = past ^ k; // mark it as selected
	transformer t = ta[ti]; // to ease debugging
	transformer nct = transformer_combine(copy_transformer(ct), t);
	// Necessary before checking emptiness
	nct = transformer_normalize(nct, 2);
	// Check the sequence feasability
	if(!transformer_empty_p(nct)) {
	  list nl = transformer_list_add_combination(tn, ta, n-1, nct, npast);
	  cl = gen_nconc(cl, nl);
	}
	found_p = true;
	free_transformer(nct);
      }
      k <<= 1;
    }
    pips_assert("At least one transformation has been found", found_p);
  }
  else {
    // The recursion is over, ct contains the right number of
    // transformer combinations
    cl = CONS(TRANSFORMER, ct, NIL);
  }
  return cl;
}

/* compute all combinations of n different transformers t from
 * transformer list tl. No check is performed on the content of list
 * tl. Empty transformers and identity transformers should have been
 * removed before this call.
 *
 * @param tl: a non-empty list of transformers
 *
 * int n: a strictly positive integer, smaller than the length of tl
 *
*/
static list __attribute__ ((unused)) transformer_list_combination(list tl, int n)
{
  list cl = NIL;
  int tn = gen_length(tl);

  pips_assert("n is smaller than the number of transformers",
	      n <= tn && n>=1 && n<=31);

  transformer ta[tn]; // build a transformer array
  int r = 0;

  FOREACH(TRANSFORMER, t, tl) {
    ta[r++] = t;
  }

  // Initialize recurrence
  transformer ct = transformer_identity();
  cl = transformer_list_add_combination(tn, ta, n, ct, -1);
  free_transformer(ct);

  return cl;
}
