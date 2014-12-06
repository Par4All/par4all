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
 * Integer constants calculated by preconditions are replaced by their value.
 * Expressions are evaluated to (ICOEF*SUBEXPR + ISHIFT) in order to perform
 * some simplifications. When ICOEF==0, SUBEXPR must be undefined.
 */

/* Hypotheses pour l'implementation:

Toute fonction d'evaluation partielle retourne eformat_undefined
lorsqu'elle n'a rien evalue (ex: lors d'un appel d'une fonction
externe).

eformat.expr NE DOIT JAMAIS partager de structures avec le code. Par
contre, une expression detachee du code peut appartenir a eformat.expr.

Lorsqu'une expression est detachee du code, il faut prendre garde a la
remplacer par expression_undefined. Sinon, le free (dans
regenerate_expression()) causera des degas!

Si une information est ajoutee a eformat_undefined, alors l'expression
est RECOPIEE. Pourtant, eformat.simpler reste false et l'expression
d'origine n'est pas freee, car une seule information ne permet aucune
simplification. A partir de la prise en compte de la seconde
information, des qu'eformat est simplife', alors eformat.simpler
devient vrai. L'expression d'origine sera alors free'e lorsque
regenerate_expression().

Des que l'evaluation n'est plus possible, il faut regenerer l'expression.

Note FI: now NORMALIZE_EXPRESSION() is also used to simplify
expressions and sub-expressions in partial_eval_expression() because
automatic program transformations sometimes generate kind of stupid
code such as "i-i+1" or "512-1". When a simplification occurs, no
feedback is provided and the linearized version of the expression is
assumed "simpler". Hence, "simpler" now means "may be simpler". See
comments below in partial_eval_expression(). Also, I did not take time
to understand the invariant for expression allocation and free. I'm
very likely to have introduced memory leaks via the changes in
partial_eval_expression().

Note FI: the interface is based on Psystemes instead of transformers,
which makes maintenance and evolution harder.
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "text.h"

#include "text-util.h"
#include "database.h"
#include "resources.h"
#include "control.h"
#include "ri-util.h"
#include "effects-util.h"
#include "pipsdbm.h"
#include "misc.h"
#include "properties.h"

#include "effects-generic.h"
#include "effects-simple.h" /* for print_effects() */
#include "transformer.h"
#include "semantics.h" /* for module_to_value_mappings() */

#include "arithmetique.h"

#include "expressions.h"


static eformat_t  eformat_undefined = {expression_undefined, 1, 0, false};
/* when formating is useless (ie. = (1 * expr + 0)) */

/* Set of enclosing loop indices
 *
 * This set is maintained to avoid useless partial evaluation of loop indices
 * which are very unlikely to be partially evaluable. Loops with only one
 * iteration can be removed by dead code elimination.
 *
 * This set is implemented as a list because the loop nest depth is expected
 * small.
 *
 * It would be nice to take inductive variables into account. Their evaluation
 * is quite long in ocean. We could use the loop transformer instead of the
 * loop index to populate live_loop_indices. But it would be harder to regenerate
 * the set when leaving the loop, unless a copy is made on entrance.
 *
 * A stack is not too attractive, as it should be all visited to make sure
 * a variable is not a live loop index or inductive variable.
 *
 * A multiset might make loop exit easier, but each membership test will be
 * longer.
 */

static list live_loop_indices = list_undefined;
static statement  partial_eval_current_statement =statement_undefined;

void set_live_loop_indices()
{
  pips_assert("set_live_loop_indices", live_loop_indices==list_undefined);
  live_loop_indices = NIL;
}

void reset_live_loop_indices()
{
  /* The index set should be empty when leaving partial eval */
  pips_assert("reset_live_loop_indices", ENDP(live_loop_indices));
  if(!ENDP(live_loop_indices)) {
    free_arguments(live_loop_indices);
  }
  live_loop_indices = list_undefined;
}

void dump_live_loop_indices()
{
  pips_assert("set_live_loop_indices", live_loop_indices!=list_undefined);
  dump_arguments(live_loop_indices);
}

static bool live_loop_index_p(entity i)
{
  pips_assert("set_live_loop_indices", live_loop_indices!=list_undefined);
  return entity_is_argument_p(i, live_loop_indices);
}

static void add_live_loop_index(entity i)
{
  pips_assert("add_live_index",!live_loop_index_p(i));
  live_loop_indices = gen_nconc(live_loop_indices,
				CONS(ENTITY, i, NIL));
}

static void rm_live_loop_index(entity i)
{
  pips_assert("set_live_loop_indices", live_loop_indices!=list_undefined);
  live_loop_indices = arguments_rm_entity(live_loop_indices, i);
}


void init_use_proper_effects(const char* module_name)
{
  set_proper_rw_effects((statement_effects)
			db_get_memory_resource(DBR_PROPER_EFFECTS, module_name, true));
  pips_assert("init_use_proper_effects", !proper_rw_effects_undefined_p());
}

/* returns proper effects associated to statement stmt */
effects stmt_to_fx(statement stmt, statement_effects fx_map)
{
  effects fx;

  pips_assert("stmt is defined", stmt != statement_undefined);

  pips_debug(9,
	"Look for effects for statement at %p (ordering %td, number %td):\n",
	stmt, statement_ordering(stmt), statement_number(stmt));

  fx = apply_statement_effects(fx_map, stmt);
  ifdebug(5)
    {
      print_effects(effects_effects(fx));
    }

  return(fx);
}

bool entity_written_p(entity ent, effects fx)
{
  if(fx==effects_undefined)
    pips_internal_error("effects undefined");

  MAPL(ftl, {
      effect ft = EFFECT(CAR(ftl));
      if( ENDP(reference_indices(effect_any_reference(ft)))
	  && same_entity_p(ent, reference_variable(effect_any_reference(ft)))
	  && action_write_p(effect_action(ft)) 
	  && action_kind_store_p(effect_action_kind(ft)) )
	return(true);
    }, effects_effects(fx));

  return(false);
}


void init_use_preconditions(const char* module_name)
{
  set_precondition_map( (statement_mapping)
			db_get_memory_resource(DBR_PRECONDITIONS, module_name, true) );
  pips_assert("init_use_preconditions",
	      get_precondition_map() != hash_table_undefined);
  if(get_debug_level()==9) {
    transformer_map_print();
  }
}

/*
  cf. load_statement_transformer() in semantics/dbm_interface.c
  */
Psysteme stmt_prec(statement stmt)
{
  transformer t;

  pips_assert("stmt_prec", stmt != statement_undefined);

  pips_debug(9,
	"Look for preconditions for statement at %p (ordering %td, number %td):\n",
	stmt, statement_ordering(stmt), statement_number(stmt));

  t = load_statement_precondition(stmt);

  if(t==(transformer) HASH_UNDEFINED_VALUE) t = transformer_undefined;

  /* pips_assert("stmt_prec", t != transformer_undefined);*/

  return(t==transformer_undefined ?
	 SC_UNDEFINED :
	 (Psysteme)predicate_system(transformer_relation(t)) );

}

void transformer_map_print(void)
{
  FILE * f =stderr;
  hash_table htp = get_precondition_map();

  hash_table_print_header (htp,f);

  HASH_MAP(k, v, {
      fprintf(f, "\nFor statement at %p (ordering %td, number %td):\n",
	      k,
	      statement_ordering((statement) k),
	      statement_number((statement) k));
      print_transformer((transformer) v);
    },
    htp);
}

bool eformat_equivalent_p(eformat_t ef1, eformat_t ef2)
{
  /* should not require anything about expr */
  return( ef1.expr == ef2.expr /* ie expression_eq(ef1.expr, ef2.expr) */
	  && ef1.icoef == ef2.icoef
	  && ef1.ishift == ef2.ishift );
}

void print_eformat(eformat_t ef, char *name)
{
  (void) printf("eformat %s = %d x EXPR + %d, %ssimpler, with EXPR:\n",
		name, ef.icoef, ef.ishift, (ef.simpler ? "" : "NOT "));
  print_expression(ef.expr);
}

void partial_eval_expression_and_regenerate(expression *ep, Psysteme ps, effects fx)
{
  eformat_t ef;

  ef = partial_eval_expression(*ep, ps, fx);

  ifdebug(5)
    print_eformat(ef, "before regenerate");

  regenerate_expression(&ef, ep);

  ifdebug(5) {
    if(!expression_consistent_p(*ep)) {
      pips_internal_error("bad evaluation");
    }
    else {
      print_eformat(ef, "after regenerate");
    }
  }
}

eformat_t partial_eval_expression_and_copy(expression expr, Psysteme ps, effects fx)
{
  eformat_t ef;

  ef = partial_eval_expression(expr, ps, fx);

  if(eformat_equivalent_p(ef,eformat_undefined)) {
    ef.expr = expr;
  }

  return(ef);
}

eformat_t partial_eval_expression(expression e, Psysteme ps, effects fx)
{
  eformat_t ef;
  expression ne = e;
  normalized n;

  unnormalize_expression(e);
  NORMALIZE_EXPRESSION(e);
  n = expression_normalized(e);

  /* In case the expression is affine, use its affine form

     FI: it would be better to test if ne is really simpler than e: it
     should contain fewer operators (1+1->2) or fewer references
     (n->1).
 */
  if(normalized_linear_p(n)) {
    Pvecteur pv = normalized_linear(n);
    ne = make_vecteur_expression(pv);
    ef = partial_eval_syntax(ne, ps, fx);
    if(!ef.simpler /*&& !ef.icoef==0 && !ef.ishift==0*/) {
      /* FI: it may be simpler because the simplification may be
	 performed by normalize_expression() */
      ef.simpler = true;
      if(expression_undefined_p(ef.expr) && ef.icoef!=0)
	ef.expr = ne;
      //ef.expr = ne;
      //ef.icoef = 1;
      //ef.ishift = 0;
    }
  }
  else
    ef = partial_eval_syntax(ne, ps, fx);

  return ef;
}

eformat_t partial_eval_syntax(expression e, Psysteme ps, effects fx)
{
  eformat_t ef;
  syntax s = expression_syntax(e);

  switch (syntax_tag(s)) {
  case is_syntax_reference:
    ef = partial_eval_reference(e, ps, fx);
    break;
  case is_syntax_range:
    ef = eformat_undefined;
    break;
  case is_syntax_call:
    ef = partial_eval_call_expression(e, ps, fx);
    break;
  case is_syntax_cast: {
    cast c = syntax_cast(s);

    partial_eval_expression_and_regenerate(&(cast_expression(c)), ps, fx);
    ef = eformat_undefined;
    break;
  }
  case is_syntax_sizeofexpression: {
    sizeofexpression soe = syntax_sizeofexpression(s);

    if(sizeofexpression_expression_p(soe)) {
      partial_eval_expression_and_regenerate(&(sizeofexpression_expression(soe)), ps, fx);
    }
    else if(get_bool_property("EVAL_SIZEOF")) {
        type t = sizeofexpression_type(soe);
        int tms = type_memory_size(t);
        update_expression_syntax(e,make_syntax_call(
                    make_call(int_to_entity(tms),NIL)
                    )
                );
    }
    ef = eformat_undefined;
    break;
  }
  case is_syntax_subscript: {
    subscript sub = syntax_subscript(s);
    partial_eval_expression_and_regenerate(&(subscript_array(sub)), ps, fx);
    for(list iter=subscript_indices(sub);!ENDP(iter);POP(iter))
        partial_eval_expression_and_regenerate((expression*)REFCAR(iter), ps, fx);

    /*
      MAPL(ce, {
      partial_eval_expression_and_regenerate(&(EXPRESSION(CAR(ce))), ps, fx);
      }, el);
    */

    ef = eformat_undefined;
    break;
  }
  case is_syntax_application: {
    application a = syntax_application(s);

    partial_eval_expression_and_regenerate(&(application_function(a)), ps, fx);

    /*
      MAPL(ce, {
      partial_eval_expression_and_regenerate(&(EXPRESSION(CAR(ce))), ps, fx);
      }, al);
    */
    ef = eformat_undefined;
    break;
  }
  case is_syntax_va_arg:
    ef =  eformat_undefined;
    break;
  default:
    pips_internal_error("case default");
    abort();
  }

  if (get_debug_level()==9)
    print_eformat(ef, "after partial_eval_syntax");

  return(ef);
}

eformat_t partial_eval_reference(expression e, Psysteme ps, effects fx)
{
  reference r;
  entity var;
  Pbase base_min = BASE_UNDEFINED;

  pips_assert("partial_eval_reference",
	      syntax_reference_p(expression_syntax(e)));
  r = syntax_reference(expression_syntax(e));
  var = reference_variable(r);

  if(reference_indices(r) != NIL) {
    MAPL(li, {
	expression expr = EXPRESSION(CAR(li));

	partial_eval_expression_and_regenerate(&expr, ps, fx);
	EXPRESSION_(CAR(li)) = expr;
      }, reference_indices(r));

    debug(9, "partial_eval_reference", "Array elements not evaluated\n");
    return(eformat_undefined);
  }

  /* FI: this test may be wrong for enum variables? Does it matter?
     what can you do with enum anyway? Well, they can be replaced by
     their values */
  if(!type_variable_p(entity_type(var)) ||
     !basic_int_p(variable_basic(type_variable(ultimate_type(entity_type(var))))) ) {
    pips_debug(9, "Reference to a non-scalar-integer variable %s cannot be evaluated\n",
	       entity_name(var));
    return(eformat_undefined);
  }

  if (SC_UNDEFINED_P(ps)) {
    pips_debug(9, "No precondition information\n");
    pips_internal_error("Probably corrupted precondition");
    return(eformat_undefined);
  }

  if(entity_written_p(var, fx)) {
    /* entity cannot be replaced */
    debug(9, "partial_eval_reference",
	  "Write Reference to variable %s cannot be evaluated\n",
	  entity_name(var));
    return(eformat_undefined);
  }

  if(live_loop_index_p(var)) {
    pips_debug(9, "Index %s cannot be evaluated\n", entity_name(var));
    return(eformat_undefined);
  }

  /* faire la Variable */
  /* verification de la presence de la variable dans ps */
  base_min = sc_to_minimal_basis(ps);
  if(base_contains_variable_p(base_min, (Variable) var)) {
    bool feasible;
    Value min, max;
    Psysteme ps1 = sc_dup(ps);

    /* feasible = sc_minmax_of_variable(ps1, (Variable) var, &min, &max); */
    feasible = sc_minmax_of_variable2(ps1, (Variable) var, &min, &max);
    if (! feasible) {
      pips_user_warning("Not feasible system:"
		   " module contains some dead code.\n");
    }
    if ( value_eq(min,max) ) {
      eformat_t ef;

      /* var is constant and has to be replaced */
      ifdebug(9) {
	pips_debug(9, "Constant to replace: \n");
	print_expression(e);
      }

      ef.icoef = 0;
      ef.ishift = VALUE_TO_INT(min);
      ef.expr = expression_undefined;
      ef.simpler = true;
      return(ef);

      /*		new_expr=int_to_expression((int)min); */
      /* replace expression_normalized(e) with
	 expression_normalized(new_expr) */
      /*		free_normalized(expression_normalized(e));
			expression_normalized(e) = expression_normalized(new_expr);
			expression_normalized(new_expr) = normalized_undefined; */

      /* replace expression_syntax(e) with
	 expression_syntax(new_expr) */
      /*
	free_syntax(expression_syntax((e)));
	expression_syntax(e) = expression_syntax(new_expr);
	expression_syntax(new_expr) = syntax_undefined;

	free_expression(new_expr);




	if ( get_debug_level() == 9) {
	debug(9, "partial_eval_reference",
	"Constant replaced by expression: \n");
	print_to_expressionession(e);
	expression_consistent_p(e);
	pips_assert("partial_eval_reference",
	syntax_call_p(expression_syntax(e)));
	} */
    }
    /*	    return(entity_initial(call_function(syntax_call(expression_syntax(e)))));
     */
    return(eformat_undefined);
  }
  base_rm(base_min);
  return(eformat_undefined);
}

void partial_eval_call_and_regenerate(call ca, Psysteme ps, effects fx)
{
  //list le = list_undefined;
  eformat_t ef = partial_eval_call(ca, ps, fx);

  pips_assert("ca is a defined call", ca!= call_undefined);

  /* FI: if the call is an operator, it is not part of the
     simplification; e.g. "3+5;" */
  /*
  for(le=call_arguments(ca); !ENDP(le); POP(le)) {
    expression exp = EXPRESSION(CAR(le));

    partial_eval_expression_and_regenerate(&exp, ps, fx);
    EXPRESSION_(CAR(le))= exp;
  }
  */
  regenerate_call(&ef, ca);
}


eformat_t partial_eval_call(call ec, Psysteme ps, effects fx)
{
  entity func;
  value vinit;
  eformat_t ef;
  func = call_function(ec);
  vinit = entity_initial(func);

  switch (value_tag(vinit)) {
  case is_value_intrinsic:
  case is_value_unknown: {
    /* it might be an intrinsic function */
    cons *la = call_arguments(ec);
    size_t nbargs = gen_length(la);
    switch(nbargs) {
        case 1:
            ef = partial_eval_unary_operator(func, la, ps, fx);break;
        case 2:
            ef = partial_eval_binary_operator(func, la, ps, fx);break;
        default:
            {
                MAPL(le, partial_eval_expression_and_regenerate((expression*)&EXPRESSION_(CAR(le)), ps, fx), call_arguments(ec) );
                ef = eformat_undefined;
            }
    }
  } break;
  case is_value_constant:
    if(integer_constant_p(func, &ef.ishift)) {
      ef.icoef = 0;
      ef.expr = expression_undefined;
      ef.simpler = false;
    }
    else ef = eformat_undefined;
    break;
  case is_value_symbolic:
    if(integer_symbolic_constant_p(func, &ef.ishift)) {
      ef.icoef = 0;
      ef.expr = expression_undefined;
      ef.simpler = true;
    }
    else ef = eformat_undefined;
    break;
  case is_value_code:
    {
        /* FI: The actual aruments are not partially evaluated?
         * SG: no it's not, I fixed this
	 *
	 * FI: Actually, the parameter mode should be checked. And the
	 * actual effects.
	 *
	 * Note FI: the result obtained for the call to fx in
	 * Transformations/eval.c is correct, but I do not understand
	 * why. Actual parameters must be evaluated somewhere else.
         */
      list ce;
      for(ce = call_arguments(ec); !ENDP(ce); POP(ce))
        {
	  expression eparam = EXPRESSION(CAR(ce));
	  if(c_language_module_p(func)) {
	    /* value passing */
	    partial_eval_expression_and_regenerate(&eparam,ps,fx);
	    EXPRESSION_(CAR(ce)) = eparam;
	  }
	  else if(fortran_language_module_p(func)) {
	    /* the partial evaluation could be further improved by
	       checking if there is a write effect on the
	       corresponding formal parameter */
	    if(false && expression_reference_p(eparam))
	      /* This is dealt for using fx when dealing with a
		 reference */
	      ; // in doubt, do nothing
	    else {
	      partial_eval_expression_and_regenerate(&eparam,ps,fx);
	      EXPRESSION_(CAR(ce)) = eparam;
	    }
	  }
	  else {
	    pips_internal_error("Unexpected programming language");
	  }
        }

        ef = eformat_undefined;
    } break;
  default:
    pips_internal_error("Default case reached.");
  }
  return(ef);
}

eformat_t partial_eval_call_expression(expression exp, Psysteme ps, effects fx)
{
  call ec;
  //entity func;
  //value vinit;
  eformat_t ef;

  pips_assert("The expression is a call",
	      syntax_call_p(expression_syntax(exp)));
  ec = syntax_call(expression_syntax(exp));

  ef = partial_eval_call(ec, ps,fx);

  return ef;
}

eformat_t partial_eval_unary_operator(entity func, cons *la, Psysteme ps, effects fx)
{
  eformat_t ef;
  expression *sub_ep;

  pips_assert("one argument", gen_length(la)==1);
  sub_ep = /*&EXPRESSION(CAR(la));*/ (expression*) REFCAR(la);

  if (ENTITY_UNARY_MINUS_P(func)) {
    ef = partial_eval_expression_and_copy(*sub_ep, ps, fx);

    if(ef.icoef==0
       || ((ef.icoef<0 || ef.icoef>1)
	   && (ef.ishift<=0))
       ) {
      ef.simpler= true;
    }

    ef.icoef= -(ef.icoef);
    ef.ishift= -(ef.ishift);
  }
  else if(ENTITY_ADDRESS_OF_P(func)) {
      ef = partial_eval_expression_and_copy(*sub_ep, ps, fx);
      if(ef.icoef!=0) // it means we should not generate a constant now
          partial_eval_expression_and_regenerate(sub_ep, ps, fx);
      ef = eformat_undefined;
  }
  else {
    /* operator can be a pre/post inc/dec C operator */
    partial_eval_expression_and_regenerate(sub_ep, ps, fx);
    ef = eformat_undefined;
  }
  return(ef);
}


#define PERFORM_ADDITION 1
#define PERFORM_SUBTRACTION 2
#define PERFORM_MULTIPLICATION 3
#define PERFORM_DIVISION 4
#define PERFORM_C_DIVISION 14
#define PERFORM_POWER 5
#define PERFORM_MODULO 6
#define PERFORM_C_MODULO 16
#define PERFORM_MINIMUM 7
#define PERFORM_MAXIMUM 8

eformat_t partial_eval_mult_operator(expression *ep1,
				     expression *ep2,
				     Psysteme ps,
				     effects fx)
{
  eformat_t ef, ef1, ef2;

  ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
  ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

  if(ef1.icoef==0 && ef2.icoef==0) {
    ef.icoef=0;
    ef.expr=expression_undefined;
    ef.ishift= ef1.ishift * ef2.ishift;
    ef.simpler= true;
  }
  else if(ef1.icoef!=0 && ef2.icoef!=0) {
    if(ef2.icoef!=1 && ef2.ishift==0) {
      expression *ep;
      /* exchange ef1 and ef2 (see later) */
      ef=ef2; ef2=ef1; ef1=ef; ef= eformat_undefined;
      ep=ep2; ep2=ep1; ep1=ep;
    }
    if(ef1.icoef!=1 && ef1.ishift==0) {
      ef.simpler= ef1.simpler;
      ef.icoef= ef1.icoef;
      regenerate_expression(&ef2, ep2);
      ef.expr= MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
			      ef1.expr, *ep2);
      ef.ishift= 0;
    }
    else { /* cannot optimize */
      regenerate_expression(&ef1, ep1);
      regenerate_expression(&ef2, ep2);

      ef= eformat_undefined;
    }
  }
  else {
    if(ef2.icoef==0) {
      expression *ep;
      /* exchange ef1 and ef2 (see later) */
      ef=ef2; ef2=ef1; ef1=ef; ef= eformat_undefined;
      ep=ep2; ep2=ep1; ep1=ep;
    }
    /* here we know that ef1.ecoef==0 and ef2.ecoef!=0 */
    if(ef1.ishift==0) {
      ef.icoef= 0;
      ef.expr= expression_undefined;
      ef.ishift= 0;
      ef.simpler= true;
      regenerate_expression(&ef2, ep2);
    }
    else {
      ef.icoef= ef1.ishift * ef2.icoef;
      ef.expr= ef2.expr;
      ef.ishift= ef1.ishift * ef2.ishift;
      ef.simpler= (ef1.ishift==1 || ef2.icoef!=1
		   || ef1.simpler || ef2.simpler);
    }
  }

  return ef;
}

eformat_t partial_eval_plus_or_minus_operator(int token,
					      expression *ep1,
					      expression *ep2,
					      Psysteme ps,
					      effects fx)
{
  eformat_t ef;
  /* Automatic tools sometimes generate source code like "i - i" */
  /* Could be improved with a commutative_expression_equal_p() if
     cases arise */
  if(expression_equal_p(*ep1, *ep2)) {
    if(token==PERFORM_SUBTRACTION) {
      ef.simpler = true;
      ef.expr = expression_undefined; //int_to_expression(0);
      ef.icoef = 0;
      ef.ishift = 0;
    }
    else if(token==PERFORM_ADDITION) {
      ef.simpler = true;
      /* FI: no idea of the expression should be copied or not, let's
	 play safe. */
      /* Here we should go down to see if *ep1 can be partially
	 evaluated */
      eformat_t ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
      ef.expr = ef1.expr;
      ef.icoef = 2*ef1.icoef;
      ef.ishift = 2*ef1.ishift;
      /* FI: here I should get rid of ef1... */
    }
  }
  else {
    eformat_t ef1, ef2;

    ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
    ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

    /* generate ef.icoef and ef.expr */
    if( (ef1.icoef==ef2.icoef || ef1.icoef==-ef2.icoef)
	&& (ef1.icoef<-1 || ef1.icoef>1) ) {
      /* factorize icoef */
      ef.simpler=true;
      if( (token==PERFORM_ADDITION && ef1.icoef==ef2.icoef)
	  || (token==PERFORM_SUBTRACTION && ef1.icoef==-ef2.icoef) ) {
	/* addition */
	ef.expr= MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
				ef1.expr,
				ef2.expr);
	ef.icoef= ef1.icoef;
      }
      else if( (ef1.icoef>1)
	       && (token==PERFORM_SUBTRACTION ? (ef2.icoef>0) : (ef2.icoef<0)) ) {
	/* substraction e1-e2 */
	ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				ef1.expr,
				ef2.expr);
	ef.icoef= ef1.icoef;
      }
      else {
	/* substraction e2-e1 */
	ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				ef2.expr,
				ef1.expr);
	ef.icoef= -ef1.icoef;
      }
    }
    else if(ef1.icoef!=0 && ef2.icoef!=0) {
      int c1 = ef1.icoef;
      int c2 = (token==PERFORM_SUBTRACTION ? -ef2.icoef : ef2.icoef);
      expression e1= generate_monome((c1>0 ? c1: -c1), ef1.expr);
      expression e2= generate_monome((c2>0 ? c2: -c2), ef2.expr);

      /* generate without factorize, but for -1? */
      ef.simpler= (ef1.simpler || ef2.simpler); /* not precise ?? */
      if(c1*c2>0) {
	ef.expr= MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
				e1, e2);
	ef.icoef= (c1>0 ? 1 : -1);
      }
      else if(c1>0) {
	ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				e1, e2);
	ef.icoef= 1;
      }
      else {
	ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				e2, e1);
	ef.icoef= 1;
      }
    }
    else {
      ef.simpler= (ef1.simpler || ef2.simpler);
      if(ef1.icoef==0) {
	/* CA (9/9/97) condition <0 added in order to simplify
	   also expression like (J)+(-1) in (J-1)    */
	if(ef1.ishift<=0)
	  ef.simpler=true;
	ef.expr=ef2.expr;
	ef.icoef=(token==PERFORM_SUBTRACTION ? -ef2.icoef : ef2.icoef);
      }
      else if(ef2.icoef==0) {
	/* FI: simplification i++ + 0? I do not understand Corinne's
	   code above. ef.ishift is generated below. I force simpler
	   because ef2.icoef==0 can result from a simplification */
	ef.expr = ef1.expr;
	ef.icoef = ef1.icoef;
	ef.simpler=true;
      }
      else {
	if(ef2.ishift<=0)
	  ef.simpler=true;
	ef.expr=ef1.expr;
	ef.icoef=ef1.icoef;
      }
    }

    /* generate ef.ishift */
    if  ((ef1.icoef==0 || ef1.ishift!=0)
	 && (ef2.icoef==0 || ef2.ishift!=0))
      {
	/* simplify shifts */
	ef.simpler= true;
      }

    ef.ishift= (token==PERFORM_SUBTRACTION ?
		ef1.ishift-ef2.ishift : ef1.ishift+ef2.ishift);
  }
  return ef;
}

eformat_t partial_eval_plus_operator(expression *ep1,
				     expression *ep2,
				     Psysteme ps,
				     effects fx)
{
  eformat_t ef;

  ef = partial_eval_plus_or_minus_operator(PERFORM_ADDITION,
					   ep1, ep2, ps, fx);

  return ef;
}

eformat_t partial_eval_plus_c_operator(expression *ep1,
				       expression *ep2,
				       Psysteme ps,
				       effects fx)
{
  eformat_t ef = eformat_undefined;
  basic b1 = basic_of_expression(*ep1);
  basic b2 = basic_of_expression(*ep2);
  if( !basic_pointer_p(b1) && !basic_pointer_p(b2) )
    {
      ef = partial_eval_plus_operator(ep1,ep2,ps,fx);
    }
  else {
    partial_eval_expression_and_regenerate(ep1,ps,fx);
    partial_eval_expression_and_regenerate(ep2,ps,fx);
  }
  free_basic(b1);
  free_basic(b2);

  return ef;
}

eformat_t partial_eval_minus_operator(expression *ep1,
				      expression *ep2,
				      Psysteme ps,
				      effects fx)
{
  eformat_t ef;

  ef = partial_eval_plus_or_minus_operator(PERFORM_SUBTRACTION,
					   ep1, ep2, ps, fx);

  return ef;
}

eformat_t partial_eval_minus_c_operator(expression *ep1,
					expression *ep2,
					Psysteme ps,
					effects fx)
{
  eformat_t ef = eformat_undefined;
  basic b1 = basic_of_expression(*ep1);
  basic b2 = basic_of_expression(*ep2);
  if( !basic_pointer_p(b1) && !basic_pointer_p(b2) )
    {
      ef = partial_eval_minus_operator(ep1,ep2,ps,fx);
    }
  else {
    partial_eval_expression_and_regenerate(ep1,ps,fx);
    partial_eval_expression_and_regenerate(ep2,ps,fx);
  }
  free_basic(b1);
  free_basic(b2);
  return ef;
}

eformat_t partial_eval_div_or_mod_operator(int token,
					   expression *ep1,
					   expression *ep2,
					   Psysteme ps,
					   effects fx)
{
  eformat_t ef, ef1, ef2;

  ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
  ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

  if( ef2.icoef==0 && ef2.ishift == 0 )
    user_error("partial_eval_div_or_mod_operator",
	       "division by zero!\n");
  if( token==PERFORM_DIVISION && ef2.icoef==0
      && (ef1.ishift % ef2.ishift)==0
      && (ef1.icoef % ef2.ishift)==0 ) {
    /* integer division does NOT commute with in any */
    /* multiplication -> only performed if "exact" */
    ef.simpler= true;
    ef.icoef= ef1.icoef / ef2.ishift;
    ef.ishift= ef1.ishift / ef2.ishift;
    ef.expr= ef1.expr;
  }
  else if(ef1.icoef==0 && ef2.icoef==0) {
    ef.simpler= true;
    ef.icoef= 0;
    ef.expr= expression_undefined;
    if (token==PERFORM_DIVISION) { /* refer to Fortran77 chap 6.1.5 */
      /* FI->SG: FORTRAN_DIV, SIGN_EQ, FORTRAN_MOD,... have been left in
	 transformations-local.h when this code was moved into
	 expressions. I haven't checked if they are used
	 elsewhere. Could the expression library not dependent on
	 transformations? */
      ef.ishift= FORTRAN_DIV(ef1.ishift, ef2.ishift);
    }
    else {
      /* FI: C and Fortran modulo and division operators seem in fact
	 equivalent, using negative modulo to maintain the equation

	 a == (a/b)*b+a%b

	 instead of a positive remainder, i.e. modulo.
      */
      if(token==PERFORM_MODULO)
	ef.ishift= FORTRAN_MOD(ef1.ishift, ef2.ishift);
      else if(token==PERFORM_C_MODULO)
	ef.ishift= C_MODULO(ef1.ishift, ef2.ishift);
      else
	pips_internal_error("Unexpected tocken");
    }
  }
  else {
    regenerate_expression(&ef1, ep1);
    regenerate_expression(&ef2, ep2);
    ef= eformat_undefined;
  }
  return ef;
}

eformat_t partial_eval_div_operator(expression *ep1,
				    expression *ep2,
				    Psysteme ps,
				    effects fx)
{
  eformat_t ef;

  ef = partial_eval_div_or_mod_operator(PERFORM_DIVISION,
					ep1, ep2, ps, fx);

  return ef;
}

eformat_t partial_eval_mod_operator(expression *ep1,
				    expression *ep2,
				    Psysteme ps,
				    effects fx)
{
  eformat_t ef;

  ef = partial_eval_div_or_mod_operator(PERFORM_MODULO,
					ep1, ep2, ps, fx);

  return ef;
}

eformat_t partial_eval_c_mod_operator(expression *ep1,
				    expression *ep2,
				    Psysteme ps,
				    effects fx)
{
  eformat_t ef;

  ef = partial_eval_div_or_mod_operator(PERFORM_C_MODULO,
					ep1, ep2, ps, fx);

  return ef;
}

eformat_t partial_eval_min_or_max_operator(int token,
					   expression *ep1,
					   expression *ep2,
					   Psysteme ps,
					   effects fx)
{
    eformat_t ef;
    bool ok = false;
    {
        expression fake = make_op_exp(MINUS_OPERATOR_NAME,copy_expression(*ep1),copy_expression(*ep2));
        transformer tr = transformer_range(load_statement_precondition(partial_eval_current_statement));
        intptr_t lb,ub;
        if(precondition_minmax_of_expression(fake,tr,&lb,&ub))
        {
            if(lb>=0) /* ep1-ep2 >= 0 -> min(ep1,ep2) == ep2 and max(ep1,ep2) == ep1 */
            {
                ef= partial_eval_expression_and_copy(*(token==PERFORM_MAXIMUM?ep1:ep2),ps,fx);
                ef.simpler=true;
                ok=true;
            }
            else if(ub <= 0 ) /* ep1-ep2 <= 0 -> min(ep1,ep2) == ep1 and  max(ep1,ep2) == ep2 */
            {
                ef= partial_eval_expression_and_copy(*(token==PERFORM_MAXIMUM?ep2:ep1),ps,fx);
                ef.simpler=true;
                ok=true;
            }
        }
    }

    if(!ok) {
        eformat_t ef1, ef2;
  ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
  ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

  if( ef1.icoef == 0 && ef2.icoef == 0 ) {
    ef.icoef = 0;
    ef.ishift = (token==PERFORM_MAXIMUM)? MAX(ef1.ishift,ef2.ishift):
      MIN(ef1.ishift,ef2.ishift);
    ef.expr = expression_undefined;
    ef.simpler = true;
  }
  else {
    regenerate_expression(&ef1, ep1);
    regenerate_expression(&ef2, ep2);
    ef = eformat_undefined;
  }
    }

  return ef;
}

eformat_t partial_eval_min_operator(expression *ep1,
				    expression *ep2,
				    Psysteme ps,
				    effects fx)
{
  eformat_t ef;

  ef = partial_eval_min_or_max_operator(PERFORM_MINIMUM,
					ep1, ep2, ps, fx);

  return ef;
}

eformat_t partial_eval_max_operator(expression *ep1,
				    expression *ep2,
				    Psysteme ps,
				    effects fx)
{
  eformat_t ef;

  ef = partial_eval_min_or_max_operator(PERFORM_MAXIMUM,
					ep1, ep2, ps, fx);

  return ef;
}

eformat_t partial_eval_power_operator(expression *ep1,
				      expression *ep2,
				      Psysteme ps,
				      effects fx)
{
  eformat_t ef, ef1, ef2;

  ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
  ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

  if( ef1.icoef == 0 && ef2.icoef == 0 && ef2.ishift >= 0) {
    ef.icoef = 0;
    ef.ishift = ipow(ef1.ishift, ef2.ishift);
    ef.expr = expression_undefined;
    ef.simpler = true;
  }
  else {
    regenerate_expression(&ef1, ep1);
    regenerate_expression(&ef2, ep2);
    ef = eformat_undefined;
  }

  return ef;
}

/* FI: a better job could be done by distinguishing between the
   different kinds of operators. For instance "a+=0" or "b*=1;" could
   be simplified. For the time being, we simplify the two
   sub-expressions but not the current expression. */
eformat_t partial_eval_update_operators(expression *ep1 __attribute__ ((__unused__)),
					expression *ep2,
					Psysteme ps,
					effects fx)
{
  eformat_t ef, ef1, ef2;

  /* You do not want to change "n = 1; n = 1;" into n = 1; 1 = 1;" */
  if(!(expression_reference_p(*ep1)
       && reference_scalar_p(expression_reference(*ep1)))) {
    ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
    regenerate_expression(&ef1, ep1);
  }

  ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);
  regenerate_expression(&ef2, ep2);

  ef = eformat_undefined;

  return ef;
}

static struct perform_switch {
  string operator_name;
  eformat_t (*binary_operator)(expression *, expression *, Psysteme, effects);
} binary_operator_switch[] = {
  {PLUS_OPERATOR_NAME, partial_eval_plus_operator},
  {PLUS_C_OPERATOR_NAME, partial_eval_plus_c_operator},
  {MINUS_OPERATOR_NAME, partial_eval_minus_operator},
  {MINUS_C_OPERATOR_NAME, partial_eval_minus_c_operator},
  {MULTIPLY_OPERATOR_NAME, partial_eval_mult_operator},
  {DIVIDE_OPERATOR_NAME, partial_eval_div_operator},
  {POWER_OPERATOR_NAME, partial_eval_power_operator},
  {MODULO_OPERATOR_NAME, partial_eval_mod_operator},
  {C_MODULO_OPERATOR_NAME, partial_eval_c_mod_operator},
  {MIN0_OPERATOR_NAME, partial_eval_min_operator},
  {MIN_OPERATOR_NAME, partial_eval_min_operator},
  {MAX0_OPERATOR_NAME, partial_eval_max_operator},
  {MAX_OPERATOR_NAME, partial_eval_max_operator},
  {ASSIGN_OPERATOR_NAME, partial_eval_update_operators},
  {MULTIPLY_UPDATE_OPERATOR_NAME, partial_eval_update_operators},
  {DIVIDE_UPDATE_OPERATOR_NAME, partial_eval_update_operators},
  {PLUS_UPDATE_OPERATOR_NAME, partial_eval_update_operators},
  {MINUS_UPDATE_OPERATOR_NAME, partial_eval_update_operators},
  {RIGHT_SHIFT_UPDATE_OPERATOR_NAME, partial_eval_update_operators},
  {LEFT_SHIFT_UPDATE_OPERATOR_NAME, partial_eval_update_operators},
  {BITWISE_OR_UPDATE_OPERATOR_NAME, partial_eval_update_operators},
  {0 , 0}
};

eformat_t partial_eval_binary_operator(entity func,
				       cons *la,
				       Psysteme ps,
				       effects fx)
{
  eformat_t ef;
  expression *ep1, *ep2;
  int i = 0;
  eformat_t (*binary_partial_eval_operator)(expression *,
					    expression *,
					    Psysteme,
					    effects) = 0;

  pips_assert("partial_eval_binary_operator", gen_length(la)==2);
  ep1= (expression*) REFCAR(la);
  ep2= (expression*) REFCAR(CDR(la));

  while (binary_operator_switch[i].operator_name!=NULL) {
    if (strcmp(binary_operator_switch[i].operator_name,
	       entity_local_name(func))==0) {
      binary_partial_eval_operator =
	binary_operator_switch[i].binary_operator;
      break;
    }
    i++;
  }

  if (binary_partial_eval_operator!=0)
    ef = binary_partial_eval_operator (ep1, ep2, ps, fx);
  else {
    partial_eval_expression_and_regenerate(ep1, ps, fx);
    partial_eval_expression_and_regenerate(ep2, ps, fx);
    ef = eformat_undefined;
  }
  /* some more optimization if a neutral element has been generated */
  entity neutral = operator_neutral_element(func);
  if(!entity_undefined_p(neutral)) {
      if(same_entity_p(expression_to_entity(*ep1),neutral)) {
          ef=partial_eval_expression(*ep2,ps,fx);
          if(!ef.simpler) /*use some partial eval dark magic */
          {
              ef.simpler=true;
              ef.icoef=1;
              ef.expr=*ep2;
          }
      }
      else if(same_entity_p(expression_to_entity(*ep2),neutral)) {
          ef=partial_eval_expression(*ep1,ps,fx);
          if(!ef.simpler) /*use some partial eval dark magic */
          {
              ef.simpler=true;
              ef.icoef=1;
              ef.expr=*ep1;
          }
      }
  }

  return ef;
}

eformat_t partial_eval_binary_operator_old(entity func,
					   cons *la,
					   Psysteme ps,
					   effects fx)
{
  eformat_t ef, ef1, ef2;
  expression *ep1, *ep2;
  int token= -1;

  pips_assert("partial_eval_binary_operator", gen_length(la)==2);
  ep1= (expression*) REFCAR(la);
  ep2= (expression*) REFCAR(CDR(la));

  if (strcmp(entity_local_name(func), MINUS_OPERATOR_NAME) == 0) {
    token = PERFORM_SUBTRACTION;
  }
  else if (strcmp(entity_local_name(func), PLUS_OPERATOR_NAME) == 0) {
    token = PERFORM_ADDITION;
  }
  else if (strcmp(entity_local_name(func), MULTIPLY_OPERATOR_NAME) == 0) {
    token = PERFORM_MULTIPLICATION;
  }
  else if (strcmp(entity_local_name(func), DIVIDE_OPERATOR_NAME) == 0) {
    /* FI: The C divide operator may be defined differently for negative
       integers */
    token = PERFORM_DIVISION;
  }
  else if (strcmp(entity_local_name(func), MODULO_OPERATOR_NAME) == 0) {
    token = PERFORM_MODULO;
  }
  else if (strcmp(entity_local_name(func), C_MODULO_OPERATOR_NAME) == 0) {
    /* FI: The C modulo operator may be defined differently for negative
       integers */
    token = PERFORM_MODULO;
  }

  if ( token==PERFORM_ADDITION || token==PERFORM_SUBTRACTION ) {
    ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
    ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

    /* generate ef.icoef and ef.expr */
    if( (ef1.icoef==ef2.icoef || ef1.icoef==-ef2.icoef)
	&& (ef1.icoef<-1 || ef1.icoef>1) ) {
      /* factorize */
      ef.simpler=true;
      if( (token==PERFORM_ADDITION && ef1.icoef==ef2.icoef)
	  || (token==PERFORM_SUBTRACTION && ef1.icoef==-ef2.icoef) ) {
	/* addition */
	ef.expr= MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
				ef1.expr,
				ef2.expr);
	ef.icoef= ef1.icoef;
      }
      else if( (ef1.icoef>1)
	       && (token==PERFORM_SUBTRACTION ? (ef2.icoef>0) : (ef2.icoef<0)) ) {
	/* substraction e1-e2 */
	ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				ef1.expr,
				ef2.expr);
	ef.icoef= ef1.icoef;
      }
      else {
	/* substraction e2-e1 */
	ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				ef2.expr,
				ef1.expr);
	ef.icoef= -ef1.icoef;
      }
    }
    else if(ef1.icoef!=0 && ef2.icoef!=0) {
      int c1 = ef1.icoef;
      int c2 = (token==PERFORM_SUBTRACTION ? -ef2.icoef : ef2.icoef);
      expression e1= generate_monome((c1>0 ? c1: -c1), ef1.expr);
      expression e2= generate_monome((c2>0 ? c2: -c2), ef2.expr);
      /* generate without factorize */
      ef.simpler= (ef1.simpler || ef2.simpler); /* not precise ?? */
      if(c1*c2>0) {
	ef.expr= MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME),
				e1, e2);
	ef.icoef= (c1>0 ? 1 : -1);
      }
      else if(c1>0) {
	ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				e1, e2);
	ef.icoef= 1;
      }
      else {
	ef.expr= MakeBinaryCall(entity_intrinsic(MINUS_OPERATOR_NAME),
				e2, e1);
	ef.icoef= 1;
      }
    }
    else {
      ef.simpler= (ef1.simpler || ef2.simpler);
      if(ef1.icoef==0) {
	if(ef1.ishift==0) ef.simpler=true;
	ef.expr=ef2.expr;
	ef.icoef=(token==PERFORM_SUBTRACTION ? -ef2.icoef : ef2.icoef);
      }
      else {
	if(ef2.ishift==0) ef.simpler=true;
	ef.expr=ef1.expr;
	ef.icoef=ef1.icoef;
      }
    }

    /* generate ef.ishift */
    if ( (ef1.icoef==0 || ef1.ishift!=0)
	 && (ef2.icoef==0 || ef2.ishift!=0) ) {
      /* simplify shifts */
      ef.simpler= true;
    }
    ef.ishift= (token==PERFORM_SUBTRACTION ?
		ef1.ishift-ef2.ishift : ef1.ishift+ef2.ishift);
  }
  else if( token==PERFORM_MULTIPLICATION ) {
    ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
    ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

    if(ef1.icoef==0 && ef2.icoef==0) {
      ef.icoef=0;
      ef.expr=expression_undefined;
      ef.ishift= ef1.ishift * ef2.ishift;
      ef.simpler= true;
    }
    else if(ef1.icoef!=0 && ef2.icoef!=0) {
      if(ef2.icoef!=1 && ef2.ishift==0) {
	expression *ep;
	/* exchange ef1 and ef2 (see later) */
	ef=ef2; ef2=ef1; ef1=ef; ef= eformat_undefined;
	ep=ep2; ep2=ep1; ep1=ep;
      }
      if(ef1.icoef!=1 && ef1.ishift==0) {
	ef.simpler= ef1.simpler;
	ef.icoef= ef1.icoef;
	regenerate_expression(&ef2, ep2);
	ef.expr= MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
				ef1.expr, *ep2);
	ef.ishift= 0;
      }
      else { /* cannot optimize */
	regenerate_expression(&ef1, ep1);
	regenerate_expression(&ef2, ep2);

	ef= eformat_undefined;
      }
    }
    else {
      if(ef2.icoef==0) {
	expression *ep;
	/* exchange ef1 and ef2 (see later) */
	ef=ef2; ef2=ef1; ef1=ef; ef= eformat_undefined;
	ep=ep2; ep2=ep1; ep1=ep;
      }
      /* here we know that ef1.ecoef==0 and ef2.ecoef!=0 */
      if(ef1.ishift==0) {
	ef.icoef= 0;
	ef.expr= expression_undefined;
	ef.ishift= 0;
	ef.simpler= true;
	regenerate_expression(&ef2, ep2);
      }
      else {
	ef.icoef= ef1.ishift * ef2.icoef;
	ef.expr= ef2.expr;
	ef.ishift= ef1.ishift * ef2.ishift;
	ef.simpler= (ef1.ishift==1 || ef2.icoef!=1
		     || ef1.simpler || ef2.simpler);
      }
    }
  }
  else if(token==PERFORM_DIVISION || token==PERFORM_MODULO) {
    ef1 = partial_eval_expression_and_copy(*ep1, ps, fx);
    ef2 = partial_eval_expression_and_copy(*ep2, ps, fx);

    if( ef2.icoef==0 && ef2.ishift == 0 )
      user_error("partial_eval_binary_operator",
		 "division by zero!\n");
    if( token==PERFORM_DIVISION && ef2.icoef==0
	&& (ef1.ishift % ef2.ishift)==0
	&& (ef1.icoef % ef2.ishift)==0 ) {
      /* integer division does NOT commute with in any */
      /* multiplication -> only performed if "exact" */
      ef.simpler= true;
      ef.icoef= ef1.icoef / ef2.ishift;
      ef.ishift= ef1.ishift / ef2.ishift;
      ef.expr= ef1.expr;
    }
    else if(ef1.icoef==0 && ef2.icoef==0) {
      ef.simpler= true;
      ef.icoef= 0;
      ef.expr= expression_undefined;
      if (token==PERFORM_DIVISION) { /* refer to Fortran77 chap 6.1.5 */
	ef.ishift= FORTRAN_DIV(ef1.ishift, ef2.ishift);
      }
      else { /* tocken==PERFORM_MODULO */
	ef.ishift= FORTRAN_MOD(ef1.ishift, ef2.ishift);
      }
    }
    else {
      regenerate_expression(&ef1, ep1);
      regenerate_expression(&ef2, ep2);
      ef= eformat_undefined;
    }
  }
  else {
    partial_eval_expression_and_regenerate(ep1, ps, fx);
    partial_eval_expression_and_regenerate(ep2, ps, fx);
    ef= eformat_undefined;
  }
  return(ef);
}

/* in order to regenerate expression from eformat.;
 *
 * optimized so that it can be called for any compatible ef and *ep;
 * result in *ep.
 */
void regenerate_expression(eformat_t *efp, expression *ep)
{
  if(eformat_equivalent_p(*efp,eformat_undefined)) {
    /* nothing to do because expressions are the same */
  }
  else if(!get_bool_property("PARTIAL_EVAL_ALWAYS_SIMPLIFY") && !efp->simpler) {
    /* simply free efp->expr */
    /* ******commented out for debug******* */
    //free_expression(efp->expr);
    efp->expr= expression_undefined; /* useless */
  }
  else {
    expression tmp_expr;

    /* *ep must be freed */
    /* ?? ******commented out for debug******* */
    /* free_expression(*ep); */

    if(efp->icoef != 0) {
      /* check */
      pips_assert("regenerate_expression",
		  efp->expr != expression_undefined);

      if(efp->icoef == 1) {
	tmp_expr= efp->expr;
      }
      else if(efp->icoef == -1) {
	/* generate unary_minus */
	tmp_expr= MakeUnaryCall(entity_intrinsic(UNARY_MINUS_OPERATOR_NAME),
				efp->expr);
      }
      else {
	/* generate product */
	tmp_expr= MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
				 int_to_expression(efp->icoef),
				 efp->expr);
      }

      if(efp->ishift != 0) {
	/* generate addition or substraction */
	string operator = (efp->ishift>0) ? PLUS_OPERATOR_NAME
	  : MINUS_OPERATOR_NAME;
	tmp_expr= MakeBinaryCall(entity_intrinsic(operator),
				 tmp_expr,
				 int_to_expression(ABS(efp->ishift)));
      }
    }
    else {
      /* check */
      pips_assert("the expression is undefined",
		  efp->expr == expression_undefined);
      /* final expression is constant efp->ishift */
      tmp_expr= int_to_expression(efp->ishift);
    }

    /* replace *ep by tmp_expr */
    *ep = tmp_expr;
  }
}

/* We are likely to end up in trouble because the regenerated
   expression may not be a call; for instance "n+0" is converted into
   a reference to n... */
void regenerate_call(eformat_t *efp, call ca)
{
  expression e = expression_undefined;
  regenerate_expression(efp, &e);
  if(expression_undefined_p(e)) {
    /* Nothing to do */
    ;
  }
  else if(expression_call_p(e)) {
    call nc = syntax_call(expression_syntax(e));
    //list al = call_arguments(ca);
    call_function(ca) = call_function(nc);
    call_arguments(ca) = call_arguments(nc);
    // gen_full_free_list(al);
  }
  else if(expression_reference_p(e)) {
    /* We are in trouble... */
    list el = expression_to_proper_effects(e);
    /* Any memory write effect? */
    if(effects_all_read_p(el)) {
      call_function(ca) = entity_intrinsic(CONTINUE_FUNCTION_NAME);
      call_arguments(ca) = NIL;
    }
    else {
      /* Do not change the initial call */
      free_expression(e);
    }
  }
  else {
    /* We are even more in trouble */
    pips_internal_error("Unexpected case.");
  }
}

expression generate_monome(int coef, expression expr)
{
  if(coef==0) {
    pips_assert("generate_monome", expr==expression_undefined);
    return(int_to_expression(0));
  }
  pips_assert("generate_monome", expr!=expression_undefined);
  if(coef==1) {
    return(expr);
  }
  if(coef==-1) {
    return(MakeUnaryCall(entity_intrinsic(UNARY_MINUS_OPERATOR_NAME),
			 expr));
  }
  return(MakeBinaryCall(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
			int_to_expression(coef),
			expr));
}

/* assumes conditions checked by partial_eval_declarations()
 *
 * partial evaluation of expressions in dimensions and of
 * initialization expression.
 */

void partial_eval_declaration(entity v, Psysteme pre_sc, effects fx)
{
  type vt = entity_type(v);
  list dl = variable_dimensions(type_variable(vt));
  expression ie = variable_initial_expression(v);

  /* See if the dimensions can be simplified */
  FOREACH(DIMENSION, d, dl) {
    partial_eval_expression_and_regenerate(&dimension_lower(d), pre_sc, fx);
    partial_eval_expression_and_regenerate(&dimension_upper(d), pre_sc, fx);
  }

  /* See if the initialization expression can be simplified */
  if(!expression_undefined_p(ie)) {
    /* FI: Lots of short cuts about side effects: pre and fx should be
       recomputed between each partial evaluation by the caller... */
    partial_eval_expression_and_regenerate(&ie, pre_sc, fx);
    entity_initial(v) = make_value_expression(ie);
    /* Too bad for the memory leak of entity_initial() but I'm afraid
       the expression might be shared*/
  }
}


void partial_eval_declarations(list el, Psysteme pre_sc, effects fx)
{
  FOREACH(ENTITY, e, el) {
    if(variable_entity_p(e) && entity_variable_p(e))
      partial_eval_declaration(e, pre_sc, fx);
  }
}

/**
 * apply partial eval on each statement
 * we cannot recurse on something other than a statement
 * because we use the effects & preconditions attached to the statement
 * @param stmt statement to partial_eval
 *
 * It is assumed that sub-expressions do not have side effects because
 * the same precondition is used for all of them.
 *
 * It would be simpler to transform instruction calls into instruction
 * expression to have fewer cases to handle. Case instruction
 * expression was introduced for C and made instruction call
 * obsolete. The same is true for value: as expressions have been
 * added, constant became redundant.
 */
void partial_eval_statement(statement stmt)
{
  instruction inst = statement_instruction(stmt);
  statement_effects fx_map = get_proper_rw_effects();
  pips_assert("internal current_statement was reseted",statement_undefined_p(partial_eval_current_statement));
  partial_eval_current_statement=stmt;

  pips_debug(8, "begin with tag %d\n",  instruction_tag(inst));

  if(declaration_statement_p(stmt)) {
    list dl = statement_declarations(stmt);
    partial_eval_declarations(dl, stmt_prec(stmt),stmt_to_fx(stmt,fx_map));
  }

  switch(instruction_tag(inst)) {
  case is_instruction_block :
    {
      /* This is no longer useful with the new representation of C
	 declarations. */
      if(false) {
	FOREACH(ENTITY,e,statement_declarations(stmt)) {
	  value v = entity_initial(e);
	  if(value_expression_p(v))
	    partial_eval_expression_and_regenerate(&value_expression(v),
						   stmt_prec(stmt),
						   stmt_to_fx(stmt,fx_map));
	}
      }
    } break;
  case is_instruction_test :
    {
      test t = instruction_test(inst);
      partial_eval_expression_and_regenerate(&test_condition(t),
					     stmt_prec(stmt),
					     stmt_to_fx(stmt,fx_map));
      ifdebug(9) {
	print_statement(stmt);
	pips_assert("stmt is consistent", statement_consistent_p(stmt));
      }
    } break;
  case is_instruction_loop :
    {
      loop l = instruction_loop(inst);
      range r = loop_range(l);
      effects fx = stmt_to_fx(stmt,fx_map);
      Psysteme sc = stmt_prec(stmt);

      /* Assuming no side-effects in loop bounds, sc is constant... */

      partial_eval_expression_and_regenerate(&range_lower(r),
					     sc,
					     fx);
      partial_eval_expression_and_regenerate(&range_upper(r),
					     sc,
					     fx);
      partial_eval_expression_and_regenerate(&range_increment(r),
					     sc,
					     fx);
      add_live_loop_index(loop_index(l));
      rm_live_loop_index(loop_index(l));

      ifdebug(9) {
	print_statement(stmt);
	pips_assert("stmt is consistent", statement_consistent_p(stmt));
      }
    } break;
  case is_instruction_forloop :
    {
      forloop fl = instruction_forloop(inst);

      partial_eval_expression_and_regenerate(&forloop_initialization(fl),
					     stmt_prec(stmt),
					     stmt_to_fx(stmt,fx_map));
      // FI: wrong precondition!
      partial_eval_expression_and_regenerate(&forloop_condition(fl),
					     stmt_prec(stmt),
					     stmt_to_fx(stmt,fx_map));
      // FI: wrong precondition!
      partial_eval_expression_and_regenerate(&forloop_increment(fl),
					     stmt_prec(stmt),
					     stmt_to_fx(stmt,fx_map));
      //add_live_loop_index(loop_index(l));
      //rm_live_loop_index(loop_index(l));

      if(get_debug_level()>=9) {
	print_text(stderr, text_statement(entity_undefined, 0, stmt, NIL));
	pips_assert(__func__, statement_consistent_p(stmt));
      }
    } break;
  case is_instruction_whileloop :
    {
      /* The whileloop precondition cannot be used to evaluate the
	 while condition. It must be unioned with the body postcondition.
	 partial_eval_expression_and_regenerate(&whileloop_condition(l),
	 stmt_prec(stmt),
	 stmt_to_fx(stmt,fx_map));
      */
      /* Also, two kinds of while must be handled */
      /* Short term fix... we might as well not try anything for
	 the while condition */
      /* partial_eval_expression_and_regenerate(&whileloop_condition(l),
	 SC_UNDEFINED, stmt_to_fx(stmt,fx_map));
      */

      ifdebug(9) {
	print_statement(stmt);
	pips_assert("stmt is consistent", statement_consistent_p(stmt));
      }
    } break;
  case is_instruction_call :
    {
      partial_eval_call_and_regenerate(instruction_call(inst),
				       stmt_prec(stmt),
				       stmt_to_fx(stmt,fx_map));
    } break;
  case is_instruction_goto:
    break;
  case is_instruction_unstructured:
    break;
  case is_instruction_expression:
    partial_eval_expression_and_regenerate(&instruction_expression(inst),
					   stmt_prec(stmt),
					   stmt_to_fx(stmt,fx_map));
    break;
  default :
    pips_internal_error("Bad instruction tag %d", instruction_tag(inst));
  }
  partial_eval_current_statement=statement_undefined;
}

/* Top-level function
 */

bool partial_eval(const char* module_name)
{
  entity module;
  statement module_statement;

  /* be carrefull not to get any mapping before the code */
  /* DBR_CODE will be changed: argument "pure" is true because
     partial_eval() *modifies* DBR_CODE. */
  /* still bugs in dbm because effects are stored on disc after this phase */

  set_current_module_entity(local_name_to_top_level_entity(module_name));
  module = get_current_module_entity();

  set_current_module_statement(
			       (statement) db_get_memory_resource(DBR_CODE, module_name, true));

  module_statement= get_current_module_statement();

  init_use_proper_effects(module_name); /* uses set_proper_effects_map */

  /* preconditions may need to print preconditions for debugging purposes */
  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));

  module_to_value_mappings(module);

  init_use_preconditions(module_name);

  set_live_loop_indices();

  debug_on("PARTIAL_EVAL_DEBUG_LEVEL");
  gen_recurse(module_statement,statement_domain,gen_true,partial_eval_statement);
  debug_off();

  /* Reorder the module, because new statements may have been generated. */
  module_reorder(module_statement);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,module_statement);

  reset_live_loop_indices();
  reset_precondition_map();
  reset_cumulated_rw_effects();
  reset_proper_rw_effects();
  reset_current_module_entity();
  reset_current_module_statement();
  free_value_mappings();

  return true;
}


/*------------------------------

  END OF FILE

--------------------------------*/




