/*

  $Id:$

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
  bool identity_p = FALSE;
  bool one_p = FALSE; // useless for the time being

  if(ENDP(tl))
    /* The empty transformer list could be used to represent the
       empty transformer... */
    pips_internal_error("Empty transformer list");

  FOREACH(TRANSFORMER, tf, tl) {
    if(transformer_identity_p(tf)) {
      if(identity_p) {
	one_p = FALSE;
	pips_internal_error("Two identity transformers in one list.");
      }
      else {
	identity_p = TRUE;
	one_p = TRUE;
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

  return TRUE;
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
      transformer nt = transformer_combine(t1, t2);

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
  bool identity_p = FALSE;

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
  return generic_transformer_list_to_transformer(ltl, FALSE);
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
  return generic_transformer_list_to_transformer(ltl, TRUE);
}
