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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif
#include <stdio.h>

#include "arithmetique.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"

#include "misc.h"
#include "transformer.h"
#include "semantics.h"

/* Return true if statement s is reachable according to its precondition. */
static bool parametric_statement_feasible_p(statement s,
				bool empty_p(transformer))
{
  transformer pre;
  bool feasible_p;

  pre = load_statement_precondition(s);

  ifdebug(6) {
    int so = statement_ordering(s);
    debug(6, "parametric_statement_feasible_p",
	  "Begin for statement %d (%d,%d) and precondition %p\n",
	  statement_number(s),
	  ORDERING_NUMBER(so), ORDERING_STATEMENT(so), pre);
  }

  feasible_p = !empty_p(pre);

  debug(6, "parametric_statement_feasible_p", " End with feasible_p = %d\n",
	feasible_p);

  return feasible_p;
}

/* Return false if precondition of statement s is transformer_empty() */
bool statement_weakly_feasible_p(statement s)
{
  transformer pre = load_statement_precondition(s);
  /* bool feasible_p = !transformer_empty_p(pre); */

  /* FI: this test is much stronger than I intended. I just wanted
   * to check that the predicate of pre was exactly sc_empty()!
   *
   * Now I'm afraid to change it. suppress_dead_code isn't that slow
   * on ocean. I'm not sure that I could validate Transformations.
   */

  Psysteme sc = predicate_system(transformer_relation(pre));

  bool feasible_p = !sc_empty_p(sc);

  return feasible_p;
}

/* Return true if statement s is reachable according to its precondition. */
bool statement_feasible_p(statement s)
{
  bool feasible_p = parametric_statement_feasible_p(s, transformer_empty_p);
  return feasible_p;
}

/* Return true if statement s is reachable according to its precondition. */
bool statement_strongly_feasible_p(statement s)
{
  bool feasible_p =
    parametric_statement_feasible_p(s, transformer_strongly_empty_p);
  return feasible_p;
}


/* A range cannot be tested exactly wrt a precondition
 * You can try to prove that it is empty or you can
 * try to prove that it is not empty. In both case, you may fail
 * and be unable to decide emptiness or non-emptiness.
 */
bool empty_range_wrt_precondition_p(range r, transformer p)
{
  bool empty = false;

  empty = check_range_wrt_precondition(r, p, true);

  return empty;
}

bool non_empty_range_wrt_precondition_p(range r, transformer p)
{
  bool non_empty = false;

  non_empty = check_range_wrt_precondition(r, p, false);

  return non_empty;
}

/* FI: this function is outdated because it does not compute the
   transformers for the range expressions, because it does not take
   side effects into account and ... We might need a
   range_to_transformer() or loop_range_to_transformer() instead and
   check that its range is empty. */
bool check_range_wrt_precondition(range r, transformer p, bool check_empty)
{
  bool check = true;
  expression lb_e = range_lower(r);
  expression ub_e = range_upper(r);
  expression incr_e = range_increment(r);
  normalized lb_n = NORMALIZE_EXPRESSION(lb_e);
  normalized ub_n = NORMALIZE_EXPRESSION(ub_e);
  normalized incr_n = NORMALIZE_EXPRESSION(incr_e);
  int incr_lb = 0;
  int incr_ub = 0;

  pips_debug(8, "begins for check %s\n", check_empty? "empty" : "non-empty");

  if(normalized_undefined_p(lb_n)
     || normalized_undefined_p(lb_n)
     || normalized_undefined_p(lb_n)) {
    user_error("check_range_wrt_precondition",
	       "Expression should have been normalized. "
	       "Maybe you have DOALL statements in your "
	       "supposedly sequential source code!\n");
  }

  if(normalized_linear_p(lb_n)
     && normalized_linear_p(ub_n)
     && normalized_linear_p(incr_n)) {

    /* gather information about the increment sign */
    expression_and_precondition_to_integer_interval(incr_e, p, &incr_lb, &incr_ub);

    if(incr_lb==0 && incr_ub==0) {
      user_error("check_range_wrt_precondition",
		 "Range with illegal zero increment\n");
    }

    if(incr_lb<=incr_ub) {
      Pvecteur lb_v = vect_dup((Pvecteur) normalized_linear(lb_n));
      Pvecteur ub_v = vect_dup((Pvecteur) normalized_linear(ub_n));
      Pcontrainte ci = CONTRAINTE_UNDEFINED;

      if(check_empty) {
	/* Try to prove that no iterations are performed */
	if(incr_lb>=1) {
	  Pvecteur ni = vect_substract(lb_v, ub_v);

	  ci = contrainte_make(ni);
	}
	else if(incr_ub<=-1) {
	  Pvecteur ni = vect_substract(ub_v, lb_v);

	  ci = contrainte_make(ni);
	}
	else {
	  /* Without information about the increment sign,
	   * you cannot make a decision. Should we accept
	   * increments greater or equal to zero? Lesser
	   * or equal to zero?
	   */
	  check = false;
	}
      }
      else {
	/* Try to prove that at least one iteration is performed */
	if(incr_lb>=1) {
	  Pvecteur ni = vect_substract(ub_v, lb_v);

	  vect_add_elem(&ni, TCST, (Value) 1);
	  ci = contrainte_make(ni);
	}
	else if(incr_ub<=-1) {
	  Pvecteur ni = vect_substract(lb_v, ub_v);

	  vect_add_elem(&ni, TCST, (Value) 1);
	  ci = contrainte_make(ni);
	}
	else {
	  /* Without information about the increment sign,
	   * you cannot make a decision. Should we accept
	   * increments greater or equal to zero? Lesser
	   * or equal to zero?
	   */
	  check = false;
	}
      }

      if(check) {
	/* No decision has been made yet */
	/* a numerical range may be empty although no information is available */
	Psysteme s = transformer_undefined_p(p) ?
	  sc_make(CONTRAINTE_UNDEFINED, CONTRAINTE_UNDEFINED) :
	  sc_dup((Psysteme) predicate_system(transformer_relation(p)));

	s = sc_inequality_add(s, ci);

	ifdebug(8) {
	  debug(8, "check_range_wrt_precondition",
		"Test feasibility for system:\n");
	  sc_fprint(stderr, s, (char * (*)(Variable)) dump_value_name);
	}

	/* s = sc_strong_normalize4(s, (char * (*)(Variable)) entity_local_name); */
	s = sc_strong_normalize5(s, (char * (*)(Variable)) entity_local_name);
	/*s = sc_elim_redund(s);*/
	ifdebug(8) {
	  debug(8, "check_range_wrt_precondition",
		"System after normalization:\n");
	  sc_fprint(stderr, s, (char * (*)(Variable)) dump_value_name);
	}

	if(SC_EMPTY_P(s)) {
	  check = true;
	}
	else {
	  sc_rm(s);
	  check = false;
	}
      }
    }
    else {
      debug(8, "check_range_wrt_precondition",
	    "The loop is never executed because it is in a dead code section\n");
      check = check_empty;
    }
  }
  else {
    debug(8, "check_range_wrt_precondition",
	  "No decision can be made because the increment sign is unknown\n");
    check = false;
  }

  debug(8, "check_range_wrt_precondition",
	"ends with check=%s for check_empty=%s\n",
	bool_to_string(check), bool_to_string(check_empty));

  return check;
}

/* A condition cannot be tested exactly wrt a precondition You can try to
 * prove that it is always true (because it is never false) or you can try
 * to prove that it is always false (because it is never true). In both
 * case, you may fail and be unable to decide emptiness or non-emptiness.  */
bool condition_true_wrt_precondition_p(expression c, transformer p)
{
  bool empty = false;

  empty = check_condition_wrt_precondition(c, p, true);

  return empty;
}

bool condition_false_wrt_precondition_p(expression c, transformer p)
{
  bool non_empty = false;

  non_empty = check_condition_wrt_precondition(c, p, false);

  return non_empty;
}

bool check_condition_wrt_precondition(expression c,
				      transformer pre,
				      bool check_true)
{
  bool check = true;
  transformer twc = transformer_dup(pre);

  pips_debug(8,	"begins for check %s\n",
	bool_to_string(check_true));

  if(check_true) {
    twc = transformer_add_condition_information(twc, c, pre, false);
  }
  else {
    /* Check that is is always false in a store s such that p(s) */
    twc = transformer_add_condition_information(twc, c, pre, true);
  }
  check = transformer_empty_p(twc);

  pips_debug(8,	"ends with check=%s for check_true=%s\n",
	bool_to_string(check), bool_to_string(check_true));

  return check;
}

/* Evaluate expression e in context p, assuming that e is an integer
 * expression. If p is empty, return an empty interval.
 *
 * Could be more general, I'm lazy (FI).
 */
void
expression_and_precondition_to_integer_interval(expression e,
						transformer p,
						int * plb,
						int * pub)
{
  normalized n = NORMALIZE_EXPRESSION(e);

  if(normalized_linear_p(n)) {
    Pvecteur v = (Pvecteur) normalized_linear(n);
    if(vect_constant_p(v)) {
      if(VECTEUR_NUL_P(v)) {
	*plb = 0;
      }
      else {
	Value vi = vect_coeff(TCST, v);
	*plb = VALUE_TO_INT(vi);
      }
      *pub = *plb;
    }
    else if(vect_size(v) == 1) {
      Psysteme s = transformer_undefined_p(p) ?
	sc_make(CONTRAINTE_UNDEFINED, CONTRAINTE_UNDEFINED) :
	sc_dup((Psysteme) predicate_system(transformer_relation(p)));
      Value lb = VALUE_ZERO, ub = VALUE_ZERO;
      entity var = (entity) vecteur_var(v);

      if(sc_minmax_of_variable(s, (Variable) var,
			       &lb, &ub)) {
	*plb = value_min_p(lb)? INT_MIN : VALUE_TO_INT(lb);
	*pub = value_max_p(ub)? INT_MAX : VALUE_TO_INT(ub);
      }
      else {
	/* precondition p is not feasible */
	*plb = 1;
	*pub = 0;
      }
    }
    else {
      /* OK, we could do something: add a pseudo-variable
       * equal to expression e and check its min abd max values
       */
      *plb = INT_MIN;
      *pub = INT_MAX;
    }
  }
  else {
    /* we are not handling an affine integer expression */
    *plb = INT_MIN;
    *pub = INT_MAX;
  }

}

/* Could be used for bool expressions too? Extended to any kind of expression? */
void integer_expression_and_precondition_to_integer_interval(expression e,
							     transformer p,
							     int * plb,
							     int * pub)
{
  type t = expression_to_type(e);
  entity tmp = make_local_temporary_value_entity(t);
  transformer et = integer_expression_to_transformer(tmp, e, p, true);

  /* If expression e is transformer-wise side-effect free (i.e. the ABSTRACT store is not modified)*/
  if(!transformer_undefined_p(et) && ENDP(transformer_arguments(et))) {
    transformer tp = transformer_range_intersection(transformer_dup(p), et);
    Psysteme s = transformer_undefined_p(tp) ?
      sc_make(CONTRAINTE_UNDEFINED, CONTRAINTE_UNDEFINED) :
      sc_dup((Psysteme) predicate_system(transformer_relation(tp)));
    Value lb = VALUE_ZERO, ub = VALUE_ZERO;

    if(sc_minmax_of_variable(s, (Variable) tmp, 
			     &lb, &ub)) {
      *plb = value_min_p(lb)? INT_MIN : VALUE_TO_INT(lb);
      *pub = value_max_p(ub)? INT_MAX : VALUE_TO_INT(ub);
    }
    else {
      /* precondition p is not feasible */
      *plb = 1;
      *pub = 0;
    }
  }
  else {
    *plb = INT_MIN;
    *pub = INT_MAX;
  }
}
