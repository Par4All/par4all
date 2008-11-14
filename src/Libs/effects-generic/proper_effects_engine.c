/* package generic effects :  Be'atrice Creusillet 5/97
 *
 * File: proper_effects_engine.c
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains the generic functions necessary for the computation of 
 * all types of proper effects and proper references.
 *
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"

#include "linear.h"
#include "ri.h"
#include "database.h"

#include "misc.h"
#include "ri-util.h"
#include "control.h"
#include "constants.h"
#include "misc.h"
#include "text-util.h"
#include "text.h"
#include "makefile.h"

#include "properties.h"
#include "pipsmake.h"

#include "transformer.h"
#include "semantics.h"
#include "pipsdbm.h"
#include "resources.h"

#include "effects-generic.h"

/* For debuging

static void debug_ctxt(string s, transformer t)
{
  Psysteme p;
  fprintf(stderr, "context %p at %s\n", t, s);
  if (transformer_undefined_p(t))
    fprintf(stderr, "UNDEFINED...");
  else
    {
      p = predicate_system(transformer_relation(t));
      fprintf(stderr, "%p: %d/%d\n", p, sc_nbre_egalites(p), sc_nbre_inegalites(p));
      sc_syst_debug(p);
      assert(sc_weak_consistent_p(p));
    }
}
*/

/************************************************ TO CONTRACT PROPER EFFECTS */

static bool contract_p = TRUE;

void
set_contracted_proper_effects(bool b)
{
    contract_p = b;
}

/**************************************** LOCAL STACK FOR LOOP RANGE EFFECTS */

/* Effects on loop ranges have to be added to inner statements to model 
 * control dependances (see loop filter for PUSH).
 */

DEFINE_LOCAL_STACK(current_downward_cumulated_range_effects, effects)

void proper_effects_error_handler()
{
    error_reset_effects_private_current_stmt_stack();
    error_reset_effects_private_current_context_stack();
    error_reset_current_downward_cumulated_range_effects_stack();
}

static list
cumu_range_effects()
{
      list l_cumu_range = NIL;

      if(! current_downward_cumulated_range_effects_empty_p())
      {
	  l_cumu_range =
	      effects_dup(effects_effects
			  (current_downward_cumulated_range_effects_head()));
      }
      return(l_cumu_range);
}

static void
free_cumu_range_effects()
{
    if(! current_downward_cumulated_range_effects_empty_p())
	free_effects(current_downward_cumulated_range_effects_head());
}


/************************************************************** EXPRESSSIONS */

/* list generic_proper_effects_of_range(range r, context)
 * input    : a loop range (bounds and stride) and the context.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list 
generic_proper_effects_of_range(range r)
{
    list le;
    expression el = range_lower(r);
    expression eu = range_upper(r);
    expression ei = range_increment(r);

    pips_debug(5, "begin\n");

    le = generic_proper_effects_of_expression(ei);
    le = gen_nconc(generic_proper_effects_of_expression(eu), le);
    le = gen_nconc(generic_proper_effects_of_expression(el), le);

    pips_debug(5, "end\n");
    return(le);
}

/* effects of a reference that is written */
/* list proper_effects_of_lhs(reference ref)
 * input    : a reference that is written.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list 
generic_proper_effects_of_lhs(reference ref)
{    
    list le = NIL;
    list inds = reference_indices(ref);
    transformer context = effects_private_current_context_head();


    pips_debug(3, "begin\n");

    if (! (*empty_context_test)(context))
    {
	le = CONS(EFFECT, 
		  (*reference_to_effect_func)(ref,
					      make_action(is_action_write, UU)),
		  NIL);

	if (! ENDP(inds)) 
	    le = gen_nconc(le, generic_proper_effects_of_expressions(inds));

	(*effects_precondition_composition_op)(le, context);
    } 

  
    pips_debug(3, "end\n");
    return(le);
}

static list generic_proper_effects_of_a_subscripted_lhs(reference ra, list inds, effect * pe)
{
  /* Simple pattern: I assume something like p[3] or
     p[3][4][5]... where p is a pointer and not an array. */

  list le = NIL;
  effect me = effect_undefined; /* main effect */
  /* generate a direct write on the variable although it is not an array */
  //entity p = reference_variable(ra);
  list ie = NIL;
  reference mr = reference_undefined;

  pips_internal_error("Function not used anymore\n");

  le = generic_proper_effects_of_reference(ra);

  /* add the subscript effects */
  MAP(EXPRESSION, ind, {
      ie = gen_nconc(ie, generic_proper_effects_of_expression(ind));
    }, inds);
  le = gen_nconc(le, ie);

  /* add the subscript to the effect or to the region in a generic way... */
  //mr = make_reference(p, gen_full_copy_list(inds));
  mr = copy_reference(ra);
  reference_indices(mr) = gen_nconc(reference_indices(mr), gen_full_copy_list(inds));
  me = make_effect(make_cell_reference(mr),
		   make_action_write(),
		   make_addressing_pre(),
		   make_approximation_must(),
		   make_descriptor_none()); /* Not generic, but fixed later */
  *pe = me;

  ifdebug(8) {
    extern void print_effect(effect);
    pips_debug(8, "End with main effect:\n");
    print_effect(me);
    pips_debug(8, "And le: ");
    (*effects_prettyprint_func)(le);
  }

  return le;
}

static list generic_proper_effects_of_a_point_to_lhs(reference r1, expression e2, effect * pe)
{
  /* The pattern is r1->e2, which is equivalent to (*e1).e2 and mapped
     onto (*e1[inds1])[rank(e2)]. This is too complex as you would
     need pre and post-indexing. We take post-indexing in account
     only: We deal only with the pattern e1==p where p is not
     subscripted */
  list le = NIL;
  list ind1 = reference_indices(r1);
  entity p = reference_variable(r1);

  if(ENDP(ind1)) {
    /* No indices for */
    /* An indirect write effect can be generated */
    /* If there were indices, the semantics will be p[i]->a, (*p[i])[a]  */
    reference mr = reference_undefined;
    effect me = effect_undefined;
    syntax s2 = expression_syntax(e2);
    entity f = reference_variable(syntax_reference(s2));
    int rank = -1;

    pips_assert("e2 is a reference", syntax_reference_p(s2));

    /* There are no subscript effects for r1 */
    /* There are not effect for e2 */

    /* set the main write effect to indirect post-index */
    rank =  entity_field_rank(f);
    mr = make_reference(p,CONS(EXPRESSION, int_to_expression(rank), NIL)); /* NOT GENERIC */
    me = make_effect(make_cell_reference(mr),
		     make_action_write(),
		     make_addressing_post(),
		     make_approximation_must(),
		     make_descriptor_none()); /* NOT GENERIC */
    *pe = me;
  }
  else {
    *pe = effect_undefined;
  }

  return le;
}

/* This function is not used */
static list generic_proper_effects_of_a_dereferencing_lhs(expression e1)
{
  /* Pattern: *e */
  list le = NIL;
  syntax s1 = expression_syntax(e1);

  if(syntax_reference_p(s1)) {
    /* An indirect write effect can be generated */
    /* If there are indices, the semantics will be *(p[i]) and not (*p)[i] */
    reference r1 = syntax_reference(s1);
    list lr = generic_proper_effects_of_lhs(r1);

    /* set the main write effect to indirect post-indexed: assume *(p)[i++]) or *p++ */
    le = lr;
    le = gen_nconc(le, generic_proper_effects_of_expression(e1));
  }
  else {
    /* let's give up with complex address computations: generate a write effect to anywhere */
    pips_user_warning("A write effect to anywhere should be generated\n");
    ;
    /* And add other effects.. */
    le = gen_nconc(le, generic_proper_effects_of_expression(e1));
  }

  return le;
}

/* Return a pointer to a new effect for s.e2 */
static list generic_proper_effects_of_a_field_lhs(reference r, expression e2, effect * pe)
{
  /* Simple basic pattern handled: s.f */
  /* Reference to a field */
  syntax s2 = expression_syntax(e2);
  entity f = entity_undefined; /* field */
  int rank = -1; /* rank of the field */
  effect me = effect_undefined; /* main effect */
  reference mr = reference_undefined;
  list le = NIL;
  list indices = reference_indices(r);

  pips_assert("The expression is a reference to a structure field",
	      syntax_reference_p(s2));

  f = reference_variable(syntax_reference(s2));
  rank = entity_field_rank(f);

  /* make a copy of the reference so as not to touch the source
     code via a persistant reference, preference */
  mr = copy_reference(r);
  reference_consistent_p(mr);
  reference_indices(mr) = gen_nconc(reference_indices(mr),
				    CONS(EXPRESSION, int_to_expression(rank), NIL));
  me = make_effect(make_cell_reference(mr),
		   make_action_write(),
		   make_addressing_index(),
		   make_approximation_must(),
		   make_descriptor_none()); /* This is not generic */
  effect_consistent_p(me);

  ifdebug(8) {
    extern void print_effect(effect);
    pips_debug(8, "Effects for a structure field \"\%s\" of rank %d: ",
	       entity_user_name(f), rank);
    print_effect(me);
    pips_debug(8, "\n");
  }

  /* Take care of effects in the reference indices */
  MAP(EXPRESSION, exp, {
      le = gen_nconc(le, generic_proper_effects_of_expression(exp));
    }, indices);

  *pe = me;
  return le;
}

/* Go down along the first argument till you find a reference or a
   dereferencing and build the effect e by side effects as well as the
   auxiliary effect list on the way back up*/
static list generic_proper_effects_of_call_in_lhs(entity op, list args, effect * pe)
{
  list le = NIL;

  if(ENTITY_FIELD_P(op)) {
    expression e1 = EXPRESSION(CAR(args));
    syntax s1 = expression_syntax(e1);
    expression e2 = EXPRESSION(CAR(CDR(args)));

    if(syntax_reference_p(s1)) {
      reference r1 = syntax_reference(s1);
      le = generic_proper_effects_of_a_field_lhs(r1, e2, pe);
    }
    else if(syntax_call_p(s1)) {
      call c1 = syntax_call(s1);
      entity op1 = call_function(c1);
      list args1 = call_arguments(c1);

      /* We must go down recursively first */
      le = generic_proper_effects_of_call_in_lhs(op1, args1, pe);

      if(!effect_undefined_p(*pe)) {
	/* update pe */
	syntax s2 = expression_syntax(e2);
	reference r2 = syntax_reference(s2);
	entity f = reference_variable(r2);
	int rank = entity_field_rank(f);
	reference mr = effect_any_reference(*pe);

	pips_assert("e2 is a reference", syntax_reference_p(s2));

	/* THIS IS NOT GENERIC */
	reference_indices(mr) = gen_nconc(reference_indices(mr),
					  CONS(EXPRESSION, int_to_expression(rank), NIL));
      }
    }
  }
  else if(ENTITY_POINT_TO_P(op)) {
    /* Any kind of complex expressions may appear here. But since the
       field in e2 implies a postindexing, e1 has to be simple because
       we cannot support both pre- and post-indexing */
    expression e1 = EXPRESSION(CAR(args));
    syntax s1 = expression_syntax(e1);
    expression e2 = EXPRESSION(CAR(CDR(args)));

    if(syntax_reference_p(s1)) {
      reference r1 = syntax_reference(s1);
      le = generic_proper_effects_of_a_point_to_lhs(r1, e2, pe);
      le = generic_proper_effects_of_expression(e1);
    }
    else {
      le = generic_proper_effects_of_expression(e1);
      le = generic_proper_effects_of_expression(e2);
      *pe = effect_undefined;
    }
  }
  else if(ENTITY_DEREFERENCING_P(op)) {
    /* Any kind of complex expressions may appear here */
    expression e1 = EXPRESSION(CAR(args));
    syntax s1 = expression_syntax(e1);

    if(syntax_call_p(s1)) {
      call c1 = syntax_call(s1);
      entity op = call_function(c1);
      list nargs = call_arguments(c1);

      le = generic_proper_effects_of_call_in_lhs(op, nargs, pe);
      //le = generic_proper_effects_of_expression(e1);
      /* we have to modify pe if it is defined? No we do not know how
	 to deal with expressions except *(p->in[i].b etc...) */
      //*pe = effect_undefined;
    }
    else if(syntax_reference_p(s1)) {
      reference mr = copy_reference(syntax_reference(s1));
      /* Keep addressing open to further indexation if possible */
      effect me = make_effect(make_cell_reference(mr),
			      make_action_write(),
			      ENDP(reference_indices(mr))?
			      make_addressing_post() : make_addressing_pre(),
			      make_approximation_must(),
			      make_descriptor_none());
      list ind1 = reference_indices(syntax_reference(s1));
      *pe = me;
      
      /* Take care of effects in the reference indices */
      MAP(EXPRESSION, exp, {
	  le = gen_nconc(le, generic_proper_effects_of_expression(exp));
	}, ind1);
      /* take care fo the pointer itself: it must be read */
      le = gen_nconc(le, generic_proper_effects_of_reference(syntax_reference(s1)));
    }
    else {
      pips_internal_error("Something wrong in RI or missing");
    }

    /* If pe is defined and of kind index/direct addressing mode, then we have a pre-indexation */
    if(!effect_undefined_p(*pe)) {
      addressing a = effect_addressing(*pe);

      if(addressing_index(a)) {
	/* If s1 is a call, then the dereferencing happens after the potential indexation: pre.
	 If s1 is an indexed reference, same thing: pre 
	 If s1 is a scalar reference, then post indexing is possible */
	effect_addressing(*pe) = (syntax_reference_p(s1)
				  && ENDP(reference_indices(syntax_reference(s1)))) ?
	  make_addressing_post() : make_addressing_pre();
	free_addressing(a);
      }
    }
    else {
      free_effect(*pe);
      *pe = effect_undefined;
    }
  } /* End for dereferencing */
  else {
    /* This may happen within a dereferencing argument to compute an address */
    /* No hope (yet) to identify a main effect as in *(p+q-r) or *(b?p:q) */
    *pe = effect_undefined;

    if(ENTITY_CONDITIONAL_P(op)) {
      expression cond = EXPRESSION(CAR(args));
      expression et = EXPRESSION(CAR(CDR(args)));
      expression ef = EXPRESSION(CAR(CDR(CDR(args))));

      list lc = generic_proper_effects_of_expression(cond);
      list lt = generic_proper_effects_of_expression(et);
      list lf = generic_proper_effects_of_expression(ef);

      le = (*effects_test_union_op)(lt, lf, effects_same_action_p);
      le = (*effects_union_op)(le, lc, effects_same_action_p);
    }
    else {
      MAP(EXPRESSION, exp, {
	  le = gen_nconc(le, generic_proper_effects_of_expression(exp));
	}, args);
    }
  }

  ifdebug(8) {
    pips_debug(8, "End with le=\n");
    (*effects_prettyprint_func)(le);
  }

  return le;
}

/* Go down along the first argument till you find a reference or a
   dereferencing and build the effect e by side effects as well as the
   auxiliary effect list on the way back up*/
list generic_proper_effects_of_complex_lhs(expression exp, effect * pmwe, effect * pmre, int lhs_p)
{
  list le = NIL;
  syntax s = expression_syntax(exp);
  bool finished_p = FALSE;
  expression s_exp = expression_undefined;
  extern void print_effect(effect);
  reference mr = reference_undefined;
  reference mre = reference_undefined;

  /* First step: see if we should recurse or not. Set s_exp if yes. If not, set mr. */

  if(syntax_reference_p(s)) {
    /* Do not recurse any longer: the basis of the address expression is found */
    mr = copy_reference(syntax_reference(s));
    mre = copy_reference(syntax_reference(s));
      
    /* take care fo the object itself: it must be read, most of the
       times. Also take care of its indices */
    /* We know it must be read because we are dealing with a complex
       lhs, except when we have a field operator only... So, this read effect mre
       will have to be taken care of later. */
    le = generic_proper_effects_of_expressions(reference_indices(syntax_reference(s)));
    finished_p = TRUE;
  }
  else if(syntax_call_p(s)) {
    call c = syntax_call(s);
    entity op = call_function(c);
    list args = call_arguments(c);
    /* FI: we assume there it at least one argument */
    
    if(gen_length(args)==0) {
      /* Problem with *(1) which is syntactically legal; could also happend with hardware*/
      pips_user_warning("Constant in a lhs expression: \"\%s\"\n",
			words_to_string(words_expression(exp)));
      /* Will be converted into an anywhere effect by the caller */
      mr = reference_undefined;
      finished_p = TRUE;
    }
    else if(ENTITY_FIELD_P(op) || ENTITY_POINT_TO_P(op)) {
      s_exp = EXPRESSION(CAR(args));
    }
    else if(ENTITY_DEREFERENCING_P(op)) {
      s_exp = EXPRESSION(CAR(args));
      if(expression_call_p(s_exp)) {
	call s_c = syntax_call(expression_syntax(s_exp));
	entity s_op = call_function(s_c);
	list s_args = call_arguments(s_c);

	if(ENTITY_PLUS_C_P(s_op)) {
	  /* This might be tractable if arg1 is a reference to a
	     pointer. For instance, *(p+q-r) can be converted to p[q-r] */
	  expression e1 = EXPRESSION(CAR(s_args));
	  syntax s1 = expression_syntax(e1);
	  expression e2 = EXPRESSION(CAR(CDR(s_args)));
	  pips_user_warning("Not fully and correctly implemented yet\n");

	  if(syntax_reference_p(s1)) {
	    reference r1 = syntax_reference(s1);
	    entity v1 = reference_variable(r1);
	    type t1 = ultimate_type(entity_type(v1));
	    if(type_variable_p(t1)) {
	      variable vt1 = type_variable(t1);
	      basic b1 = variable_basic(vt1);
	      if(basic_pointer_p(b1)) {
		mr = make_reference(v1, CONS(EXPRESSION,e2,NIL));
		mre = make_reference(v1, NIL);
		le = generic_proper_effects_of_expression(e1);
		le = gen_nconc(le, generic_proper_effects_of_expression(e2));
		finished_p = TRUE;
	      }
	    }
	  }
	  if(!finished_p) {
	    le = generic_proper_effects_of_expression(exp);
	    mr = reference_undefined;
	    finished_p = TRUE;
	  }
	}
	/* Other functions to process: p++, ++p, p--, --p */
	else if(ENTITY_POST_INCREMENT_P(s_op) || ENTITY_POST_DECREMENT_P(s_op)) {
	  expression e1 = EXPRESSION(CAR(s_args));
	  syntax s1 = expression_syntax(e1);
	  reference r1 = syntax_reference(s1);
	  entity v1 = reference_variable(r1);

	  /* YOU DO NOT WANT TO GO DOWN RECURSIVELY. DO AS FOR C_PLUS ABOVE: p[0]! */

	  pips_assert("The argument is a reference", syntax_reference_p(s1));

	  /* This seems OK for a scalar. How about an indexed reference? */

	  le = generic_proper_effects_of_expression(EXPRESSION(CAR(args)));
	  mr = make_reference(v1, CONS(EXPRESSION, int_to_expression(0), NIL));
	  mre = copy_reference(r1);
	  /* DO NOT go down recursively with this new s_exp since the
	     incrementation or decrementation can be ignored for the
	     dereferencing. */
	  finished_p = TRUE;
	}
	else if(ENTITY_PRE_INCREMENT_P(s_op) || ENTITY_PRE_DECREMENT_P(s_op)) {
	  expression e1 = EXPRESSION(CAR(s_args));
	  syntax s1 = expression_syntax(e1);
	  reference r1 = syntax_reference(s1);
	  reference nr1 = reference_undefined;

	  /* YOU DO NOT WANT TO GO DOWN RECURSIVELY. DO AS FOR C_PLUS ABOVE */

	  pips_assert("The argument is a reference", syntax_reference_p(s1));
	  pips_assert("The reference is scalar", ENDP(reference_indices(r1)));

	  le = generic_proper_effects_of_expression(EXPRESSION(CAR(args)));
	  nr1 = copy_reference(r1);
	  if(ENTITY_PRE_INCREMENT_P(s_op))
	    reference_indices(nr1) = CONS(EXPRESSION, int_to_expression(1), NIL);
	  else
	    reference_indices(nr1) = CONS(EXPRESSION, int_to_expression(-1), NIL);

	  /* Too bad for the memory leaks involved... This s_exp
	     should be freed at exit. */
	  mr = nr1;
	  mre = copy_reference(r1);
	  finished_p = TRUE;
	}
	else {
	  /* do nothing, go down recursively to handle other calls */
	  ;
	}
      }
      else {
	/* This is not a call, go down recursively */
	;
      }
    }
    else {
      /* failure: a user function is called to return a structure or an address */
      pips_user_warning("PIPS does not know how to handle precisely this %s: \"%s\"\n",
			lhs_p? "lhs":"rhs",
			words_to_string(words_expression(exp)));
      /* FI: This comes too late. down in the recursion. The effect of
	 the other sub-expressions won't be computed because we've set
	 up finish_p==TRUE and *pmwe == effect_undefined */
      le = generic_proper_effects_of_expression(exp);

      finished_p = TRUE;
    }
  }
  else if(syntax_subscript_p(s)) {
    s_exp = subscript_array(syntax_subscript(s));
  }

  if(finished_p) {
    if(reference_undefined_p(mr)) {
      *pmwe = effect_undefined;
    }
    else {
      /* Keep addressing open to further indexation if possible */
      effect me = make_effect(make_cell_reference(mr),
			      lhs_p?make_action_write():make_action_read(),
			      make_addressing_index(),
			      make_approximation_must(),
			      make_descriptor_none());
      *pmwe = me;

      /* Can mre be undefined when mr is defined?*/
      if(!reference_undefined_p(mre)) {
	/* Read effect to generate for point_to and for dereferencing */
	effect mre = make_effect(make_cell_reference(copy_reference(mr)),
				 make_action_read(),
				 make_addressing_index(),
				 make_approximation_must(),
				 make_descriptor_none());

	*pmre = mre;
      }
      else {
	pips_debug(8, "mr is defined but not mre\n");
      }
    }
  }
  else {
    /* go down */
    le = gen_nconc(le, generic_proper_effects_of_complex_lhs(s_exp, pmwe, pmre, lhs_p));

    ifdebug(8) {
      if(effect_undefined_p(*pmwe)) {
	pips_debug(8, "\nReturn with *pmwe:\n");
	fprintf(stderr, "EFFECT UNDEFINED\n");
      }
      else {
	pips_debug(8, "And *pmwe (addressing mode %d):\n",
		   addressing_tag(effect_addressing(*pmwe)));
	print_effect(*pmwe);
      }
      if(effect_undefined_p(*pmre)) {
	pips_debug(8, "And *pmre:\n");
	fprintf(stderr, "EFFECT UNDEFINED\n");
      }
      else {
	pips_debug(8, "And *pmre (addressing mode %d):\n",
		   addressing_tag(effect_addressing(*pmre)));
	print_effect(*pmre);
      }
      pips_debug(8, "And le :\n");
      (*effects_prettyprint_func)(le);
    }

    if(!effect_undefined_p(*pmwe)) {
      /* Let's try to refine *pmwe with the current expression, the
	 current operator if any and the current second expression
	 when it exists */
      reference mr = effect_any_reference(*pmwe);
      list mr_inds = reference_indices(mr);
      addressing ad = effect_addressing(*pmwe);

      if(syntax_reference_p(s)) {
	pips_internal_error("A reference should lead to the finished state\n");
      }
      else if(syntax_call_p(s)) {
	call c = syntax_call(s);
	entity op = call_function(c);
	list args = call_arguments(c);

	if(ENTITY_FIELD_P(op)) {
	  expression e2 = EXPRESSION(CAR(CDR(args)));
	  syntax s2 = expression_syntax(e2);
	  reference r2 = syntax_reference(s2);
	  entity f = reference_variable(r2);
	  int rank = entity_field_rank(f);

	  pips_assert("e2 is a reference", syntax_reference_p(s2));

	  /* Can we extend *pmwe? Yes, if its addressing is direct
	     or post since we are add a subscript */
	  if(addressing_index_p(ad) || addressing_post_p(ad)) {
	    reference_indices(mr) = gen_nconc(reference_indices(mr),
					      CONS(EXPRESSION, int_to_expression(rank), NIL));
	    // addressing is left unchanged
	    finished_p = TRUE;
	  }
	}
	else if(ENTITY_POINT_TO_P(op)) {
	  /* Since the field in e2 implies a postindexing, *pmwe has to
	     be direct or post because we cannot support pre- and
	     post-indexing simultanesouly */
	  expression e2 = EXPRESSION(CAR(CDR(args)));
	  syntax s2 = expression_syntax(e2);

	  pips_assert("e2 is a reference", syntax_reference_p(s2));

	  /* We add a dereferencing and a subscript: no previous
	     dereferencing or subscript is possible. */
	  if(addressing_index_p(ad) && ENDP(mr_inds)) {
	    entity f = reference_variable(syntax_reference(s2));
	    int rank = entity_field_rank(f);
	    expression new_int = int_to_expression(rank);

	    reference_indices(mr) = gen_nconc(reference_indices(mr), 
					      CONS(EXPRESSION, new_int, NIL));
	    addressing_tag(ad) = is_addressing_post;
	    finished_p = TRUE;
	  }

	  /* add effects due to e2 */
	  le = gen_nconc(le, generic_proper_effects_of_expression(e2));

	  /* A read must to the main variable must be added to le */
	  pips_debug(8, "Add *pmre to le\n");
	  le = gen_nconc(le, CONS(EFFECT, copy_effect(*pmre), NIL));
	}
	else if(ENTITY_DEREFERENCING_P(op)) {
	  /* Any kind of complex expressions may appear here. But only
	     one level of indirection is supported: *pmwe must be direct
	     addressing; if not indexing is used yet, post-indexing is
	     still possible; if not, this is pre-indexing */

	  if(addressing_index_p(ad)) {
	    if(ENDP(mr_inds))
	      addressing_tag(ad) = is_addressing_post;
	    else
	      addressing_tag(ad) = is_addressing_pre;
	    finished_p = TRUE;
	  }
	  /* A read must to the main variable must be added to le */
	  pips_debug(8, "Add *pmre to le\n");
	  le = gen_nconc(le, CONS(EFFECT, copy_effect(*pmre), NIL));
	}
	else {
	  pips_internal_error("Unexpected call to \"\%s\"\n", entity_name(op));
	}
      }
      else if(syntax_subscript_p(s)) {
	/* If current addressing is:
	 *  - direct with no indexing: pre
	 *  - direct with indexing: pre
	 *  - pre : pre
	 *  - post: impossible to combine pre and post
	 */
	subscript ss = syntax_subscript(s);
	list ind = subscript_indices(ss);
      
	if(!addressing_pre_p(ad)) {
	  reference_indices(mr) = gen_nconc(reference_indices(mr), gen_full_copy_list(ind));
	  // addressing mode is unchanged
	  finished_p = TRUE;
	}
	else {
	  pips_debug(8, "Give up on *pmwe because of addressing\n");
	}

	/* Take care of effects in ind */
	MAP(EXPRESSION, exp, {
	    le = gen_nconc(le, generic_proper_effects_of_expression(exp));
	  }, ind);

	/* take care of the pointer itself: it must be read, but this
	   must have been done much earlier when the main reference in
	   the lhs has been found */
      }
      else {
	/* we should be finished already because we do not know how to
	   handle these constructs and we knew that before going down
	   and up. */
	pips_internal_error("Something wrong in RI or missing");
      }
    } /* end of !effect_undefined_p(*pmwe) */
  } /* */
    
  if(!finished_p && !effect_undefined_p(*pmwe)) {
    /* The sub-effect could not be refined */
    free_effect(*pmwe);
    *pmwe = effect_undefined;
  }

  ifdebug(8) {
    pips_debug(8, "End with le=\n");
    (*effects_prettyprint_func)(le);
    if(effect_undefined_p(*pmwe)) {
      pips_debug(8, "And *pmwe:\n");
      fprintf(stderr, "EFFECT UNDEFINED\n");
    }
    else {
      pips_debug(8, "And *pmwe (addressing mode %d):\n",
		 addressing_tag(effect_addressing(*pmwe)));
      print_effect(*pmwe);
    }
    if(effect_undefined_p(*pmre)) {
      pips_debug(8, "And *pmre:\n");
      fprintf(stderr, "EFFECT UNDEFINED\n");
    }
    else {
      pips_debug(8, "And *pmre (addressing mode %d):\n",
		 addressing_tag(effect_addressing(*pmre)));
      print_effect(*pmre);
    }
  }

  return le;
}

static list generic_proper_effects_of_subscript_in_lhs(subscript sub, effect * pe)
{
  expression a = subscript_array(sub); /* address expression */
  list inds = subscript_indices(sub);
  syntax sa = expression_syntax(a);
  list le = NIL;
  void print_effect(effect);

  pips_internal_error("Function not used anymore\n");

  if(syntax_reference_p(sa)) {
    reference ra = syntax_reference(sa);
    effect e = make_effect(cell_undefined, action_undefined, addressing_undefined,
			   approximation_undefined, descriptor_undefined);

    le = generic_proper_effects_of_a_subscripted_lhs(ra, inds, &e);
    // FI: this should be done at a higher level
    //pips_user_warning("If defined, effect e should be added to le");
    //if(!effect_undefined_p(e)) {
    //	le = gen_nconc(CONS(EFFECT, e, NIL), le);
    //}
  }
  else if(syntax_subscript_p(sa)) {
    /* Go down */
    subscript sub1 = syntax_subscript(sa);

    pips_debug(8, "Go down...\n");

    le = generic_proper_effects_of_subscript_in_lhs(sub1, pe);

    ifdebug(8) {
      pips_debug(8, "Come back up with pe \%sdefined\n",
		 effect_undefined_p(*pe)? "un" : "");
      if(!effect_undefined_p(*pe)) {
	print_effect(*pe);
      }
    }

    if(!effect_undefined_p(*pe)) {
      reference ra1 = effect_any_reference(*pe);
      /* Take care of subscript linked to sub and update again *pe */
      /* FI: I'm lazy (or in a hurry)... Stuff is redone and must be
	 dumped right away */
      list lr = generic_proper_effects_of_a_subscripted_lhs(ra1, inds, pe);
      gen_free_list(lr);

      ifdebug(8) {
	pips_debug(8, "Update on the way up with pe \%sdefined\n",
		   effect_undefined_p(*pe)? "un" : "");
	if(!effect_undefined_p(*pe)) {
	  print_effect(*pe);
	}
      }
    }
  }
  else if(syntax_call_p(sa)) {
    call c = syntax_call(sa);
    entity op = call_function(c);
    list nargs = call_arguments(c);

    le = generic_proper_effects_of_call_in_lhs(op, nargs, pe);

    if(!effect_undefined_p(*pe)) {
      //reference r = effect_any_reference(*pe);
      addressing ad = effect_addressing(*pe);
      int t = addressing_tag(ad);
      //list lr = generic_proper_effects_of_a_subscripted_lhs(r, inds, pe);
      if(!effect_undefined_p(*pe)) {
	addressing na = effect_addressing(*pe);
	int nt = addressing_tag(na);
	/* FI: A simple assignment would do the same, but I want to
	   keep track of these changes, at least with the debugger */
	if(t!=nt)
	  addressing_tag(na) = t;
      }
    }
  }
  else {
    /* We do not know what to do or how to express this within our lattice */
    *pe = anywhere_effect(make_action_write());
  }

  ifdebug(8) {
    pips_debug(8, "End with le=\n");
    (*effects_prettyprint_func)(le);
    pips_debug(8, "and pe=\n");
    if(effect_undefined_p(*pe))
      fprintf(stderr, " UNDEFINED \n");
    else
      print_effect(*pe);
  }

  return le;
}


list generic_proper_effects_of_any_lhs(expression lhs)
{
  return generic_proper_effects_of_address_expression(lhs, TRUE);
}

/* FI: lhs should be subtituted by ae */
list generic_proper_effects_of_address_expression(expression lhs, int write_p)
{
  list le = NIL;
  syntax s = expression_syntax(lhs);

  if (syntax_reference_p(s)) {
    if(write_p)
      le = generic_proper_effects_of_lhs(syntax_reference(s));
    else
      pips_internal_error("Case not taken into account");
  }
  else if(syntax_call_p(s) || syntax_subscript_p(s)) {
    effect e = effect_undefined; /* main write effect */
    effect re = effect_undefined; /* main read effect */
    effect ge = effect_undefined; /* generic effect */

    /* Look for a main write effect of the lhs */
    le = generic_proper_effects_of_complex_lhs(lhs, &e, &re, write_p);

    if(!effect_undefined_p(re)) {
      /* Copies of re were used to deal with complex addressing. The
	 data structure is no longer useful */
      free_effect(re);
    }

    if(!effect_undefined_p(e)) {
      reference r = effect_any_reference(e);
      transformer context = effects_private_current_context_head();

     /* Generate a proper decriptor in a generic way */
      ge = (*reference_to_effect_func)(r, write_p?make_action_write():make_action_read());
      /* FI: memory leak of a CONS */
      (*effects_precondition_composition_op)(CONS(EFFECT, ge, NIL), context);
      effect_addressing(ge) = copy_addressing(effect_addressing(e));
      effect_approximation(ge) = copy_approximation(effect_approximation(e));
    }
    else {
      /* add an anywhere effect */
      ge = anywhere_effect(write_p? make_action_write() : make_action_read());
    }
    le = CONS(EFFECT, ge, le);
    ifdebug(8) {
      pips_debug(8, "Effect for a call:\n");
      (*effects_prettyprint_func)(le);
    }
  }
  else if(syntax_cast_p(s)) {
    pips_user_error("use of cast expressions as lvalues is deprecated\n");
  }
  else if(syntax_sizeofexpression_p(s)) {
    pips_user_error("sizeof cannot be a lhs\n");
  }
  else if(syntax_application_p(s)) {
    /* I assume this not any more possible than a standard call */
    pips_user_error("use of indirect function call as lhs is not allowed\n");
  }
  else
    pips_internal_error("lhs is not a reference and is not handled yet: syntax tag=%d\n",
			syntax_tag(s));

  ifdebug(8) {
    pips_debug(8, "End with le=\n");
    (*effects_prettyprint_func)(le);
    fprintf(stderr, "\n");
  }

  return le;
}

/* list generic_proper_effects_of_reference(reference ref)
 * input    : a reference that is read.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list 
generic_proper_effects_of_reference(reference ref)
{
  list le = NIL;
  entity v = reference_variable(ref);

  /* structure fields are referenced, not called, altough they are constant... */
  if(!entity_field_p(v)) {
    list inds = reference_indices(ref);
    transformer context;

    /* CA: lazy, because in the in region backward translation of formal
     * parameters that are not simple references on the caller side,
     * this stuff may be called without proper context.
     */
    if (effects_private_current_context_empty_p())
      context = transformer_undefined;
    else {
	context = effects_private_current_context_head();
      }

    pips_debug(3, "begin\n");
    
    if (! (*empty_context_test)(context))
      {	
	le = CONS(EFFECT, 
		  (*reference_to_effect_func)(ref,
					      make_action(is_action_read, UU)),
		  NIL);

	if (! ENDP(inds)) 
	  le = gen_nconc(le, generic_proper_effects_of_expressions(inds));


	(*effects_precondition_composition_op)(le, context);
      }
  }
  pips_debug(3, "end\n");
  return(le);
}


/* TO VERIFY !!!!!!!!!!!!!*/
list 
generic_proper_effects_of_subscript(subscript s)
{
    list inds = subscript_indices(s);
    list le = NIL;
    transformer context;

    if (effects_private_current_context_empty_p())
	context = transformer_undefined;
    else
      {
	context = effects_private_current_context_head();
      }


    pips_debug(3, "begin\n");
    
    if (! (*empty_context_test)(context))
    {	
      le = generic_proper_effects_of_expression(subscript_array(s));
      
      if (! ENDP(inds)) 
	le = gen_nconc(le, generic_proper_effects_of_expressions(inds));
      

	(*effects_precondition_composition_op)(le, context);
    }

    pips_debug(3, "end\n");
    return(le);
}

list generic_proper_effects_of_application(application a __attribute__((__unused__)))
{
  list le = NIL;

  /* Add code here */
  pips_user_warning("Effect of indirect calls not implemented\n");

  return(le);
}

/* list generic_proper_effects_of_syntax(syntax s)
 * input    : 
 * output   : 
 * modifies : 
 * comment  :	
 */
list 
generic_proper_effects_of_syntax(syntax s)
{
    list le = NIL;

    pips_debug(5, "begin\n");

    switch(syntax_tag(s))
      {
      case is_syntax_reference:
	le = generic_proper_effects_of_reference(syntax_reference(s));
	break;
      case is_syntax_range:
        le = generic_proper_effects_of_range(syntax_range(s));
        break;
      case is_syntax_call:
        le = generic_r_proper_effects_of_call(syntax_call(s));
        break;
      case is_syntax_cast: 
	le = generic_proper_effects_of_expression(cast_expression(syntax_cast(s)));
	break;
      case is_syntax_sizeofexpression:
	{
	  sizeofexpression se = syntax_sizeofexpression(s);
	  if (sizeofexpression_expression_p(se))
	    le = generic_proper_effects_of_expression(sizeofexpression_expression(se));
	  break;
	}
      case is_syntax_subscript:
	{
	  expression exp = make_expression(s, normalized_undefined);
	  le = generic_proper_effects_of_address_expression(exp, FALSE);
	  expression_syntax(exp) = syntax_undefined;
	  free_expression(exp);
	break;
	}
      case is_syntax_application:
	le = generic_proper_effects_of_application(syntax_application(s));
	break;
    default:
        pips_internal_error("unexpected tag %d\n", syntax_tag(s));
    }

    ifdebug(8)
    {
	pips_debug(8, "Proper effects of expression \"%s\":\n",
		   words_to_string(words_syntax(s)));
	(*effects_prettyprint_func)(le);
    }

    pips_debug(5, "end\n");
    return(le);
}

/* list proper_effects_of_expression(expression e)
 * inputgeneric_    : an expression and the current context
 * output   : the correpsonding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list 
generic_proper_effects_of_expression(expression e)
{
  list le = generic_proper_effects_of_syntax(expression_syntax(e));

  /* keep track of proper effects associated to sub-expressions if required.
   */
  if (!expr_prw_effects_undefined_p())
  {
    /* in IO lists, the effects are computed twice, 
     * once as LHS, once as a REFERENCE...
     * so something may already be in. Let's skip it.
     * I should investigate further maybe. FC.
     */
    if (!bound_expr_prw_effects_p(e))
      store_expr_prw_effects(e, make_effects(gen_full_copy_list(le)));
  }

  return le;
}

/* list generic_proper_effects_of_expressions(list exprs)
 * input    : a list of expressions and the current context.
 * outpproper_ut   : the correpsonding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list 
generic_proper_effects_of_expressions(list exprs)
{
    list le = NIL;

    pips_debug(5, "begin\n");

    MAP(EXPRESSION, exp,
	/* le may be long... */
	le = gen_nconc(generic_proper_effects_of_expression(exp), le),
	exprs);

    pips_debug(5, "end\n");
    return(le);
}


static list 
generic_proper_effects_of_external(entity func, list args)
{
    list le = NIL;
    char *func_name = module_local_name(func);

    pips_debug(4, "translating effects for %s\n", func_name);

    if (! entity_module_p(func)) 
    {
	pips_error("proper_effects_of_external", 
		   "%s: bad function\n", func_name);
    }
    else 
    {
	list func_eff;
	transformer context;

        /* Get the in summary effects of "func". */	
	func_eff = (*db_get_summary_rw_effects_func)(func_name);
	/* Translate them using context information. */
	context = effects_private_current_context_head();
	le = (*effects_backward_translation_op)(func, args, func_eff, context);
    }
    return le;  
}

/* list proper_effects_of_call(call c, transformer context, list *plpropreg)
 * input    : a call, which can be a call to a subroutine, but also
 *            to an function, or to an intrinsic, or even an assignement.
 *            And a pointer that will be the proper effects of the call; NIL,
 *            except for an intrinsic (assignment or real FORTRAN intrinsic).
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list 
generic_r_proper_effects_of_call(call c)
{
    list le = NIL;
    entity e = call_function(c);
    tag t = value_tag(entity_initial(e));
    string n = module_local_name(e);
    list pc = call_arguments(c);

    pips_debug(2, "begin for %s\n", entity_local_name(e));

    switch (t)
    {
    case is_value_code:
        pips_debug(5, "external function %s\n", n);
        le = generic_proper_effects_of_external(e, pc);
        break;

    case is_value_intrinsic:
        pips_debug(5, "intrinsic function %s\n", n);
        le = generic_proper_effects_of_intrinsic(e, pc);
        break;

    case is_value_symbolic:
	pips_debug(5, "symbolic\n");
	break;

    case is_value_constant:
	pips_debug(5, "constant\n");
        break;

    case is_value_unknown:
	if (get_bool_property("HPFC_FILTER_CALLEES"))
	    /* hpfc specials are managed here... */
	    le = NIL;
	else
	    pips_internal_error("unknown function %s\n", entity_name(e));
        break;

    default:
        pips_internal_error("unknown tag %d\n", t);
    }

    pips_debug(2, "end\n");

    return(le);
}


/**************************************************************** STATEMENTS */

static void 
proper_effects_of_call(call c)
{
    list l_proper = NIL;
    statement current_stat = effects_private_current_stmt_head();
    instruction inst = statement_instruction(current_stat);
    list l_cumu_range = cumu_range_effects();

    /* Is the call an instruction, or a sub-expression? */
    if (instruction_call_p(inst) && (instruction_call(inst) == c))
    {
	pips_debug(2, "Effects for statement%03zd:\n",
		   statement_ordering(current_stat)); 
	l_proper = generic_r_proper_effects_of_call(c);
	l_proper = gen_nconc(l_proper, effects_dup(l_cumu_range));
		
	if (contract_p)
	    l_proper = proper_effects_contract(l_proper);
	ifdebug(2) {
	  pips_debug(2, "Proper effects for statement%03zd:\n",
		     statement_ordering(current_stat));  
	  (*effects_prettyprint_func)(l_proper);
	  pips_debug(2, "end\n");
	}

	store_proper_rw_effects_list(current_stat, l_proper);
    }
}

/* judt to handle one kind of instruction */
proper_effects_of_expression_instruction(instruction i)
{
  list l_proper = NIL;
  statement current_stat = effects_private_current_stmt_head();
  instruction inst = statement_instruction(current_stat);
  list l_cumu_range = cumu_range_effects();

  /* Is the call an instruction, or a sub-expression? */
  if (instruction_expression_p(i)) {
    expression ie = instruction_expression(i);
    syntax is = expression_syntax(ie);

    if(syntax_cast_p(is)) {
      expression ce = cast_expression(syntax_cast(is));
      syntax sc = expression_syntax(ce);

      if(syntax_call_p(sc)) {
	call c = syntax_call(sc);

	pips_debug(2, "Effects for expression instruction in statement%03zd:\n",
		   statement_ordering(current_stat)); 

	l_proper = generic_r_proper_effects_of_call(c);
	l_proper = gen_nconc(l_proper, effects_dup(l_cumu_range));
		
	if (contract_p)
	  l_proper = proper_effects_contract(l_proper);
	ifdebug(2) {
	  pips_debug(2, "Proper effects for statement%03zd:\n",
		     statement_ordering(current_stat));  
	  (*effects_prettyprint_func)(l_proper);
	  pips_debug(2, "end\n");
	}

	store_proper_rw_effects_list(current_stat, l_proper);
      }
      else {
	pips_internal_error("Cast case not implemented\n");
	  }
    }
    else {
      pips_internal_error("Instruction expression case not implemented\n");
	}
  }
}

static void 
proper_effects_of_unstructured(unstructured u __attribute__((__unused__)))
{
    statement current_stat = effects_private_current_stmt_head();
    store_proper_rw_effects_list(current_stat,NIL);
}

static bool
loop_filter(loop l)
{
    list l_proper = generic_proper_effects_of_range(loop_range(l));
    list l_eff = cumu_range_effects();
    
    l_eff = gen_nconc(l_proper, l_eff);
    current_downward_cumulated_range_effects_push(make_effects(l_eff));
    return(TRUE);
}

static void proper_effects_of_loop(loop l)
{
    statement current_stat = effects_private_current_stmt_head();
    list l_proper = NIL;
    list l_cumu_range = NIL;

    entity i = loop_index(l);
    range r = loop_range(l);

    list li = NIL, lb = NIL;

    pips_debug(2, "Effects for statement%03zd:\n",
	       statement_ordering(current_stat)); 

    free_cumu_range_effects();
    current_downward_cumulated_range_effects_pop();
    l_cumu_range = cumu_range_effects();
    
    /* proper_effects first */

    /* Effects of loop on loop index.
     * loop index is must-written but may-read because the loop might
     * execute no iterations.
     */
    /* FI, RK: the may-read effect on the index variable is masked by
     * the initial unconditional write on it (see standard page 11-7, 11.10.3);
     * if masking is not performed, the read may prevent privatization
     * somewhere else in the module (12 March 1993)
     */
    /* Parallel case
     *
     * as I need the same effects on a parallel loop to remove
     * unused private variable in rice/codegen.c, I put the
     * same code to compute parallel loop proper effects.
     * this may not be correct, but I should be the only one to use
     * such a feature. FC, 23/09/93
     */
    li = generic_proper_effects_of_lhs(make_reference(i, NIL));

    /* effects of loop bound expressions. */
    lb = generic_proper_effects_of_range(r);

    l_proper = gen_nconc(li, lb);
    l_proper = gen_nconc(l_proper, effects_dup(l_cumu_range));
  
    ifdebug(2)
    {
	pips_debug(2, "Proper effects for statement%03zd:\n",
		   statement_ordering(current_stat));  
	(*effects_prettyprint_func)(l_proper);
	pips_debug(2, "end\n");
    }

    if (contract_p)
	l_proper = proper_effects_contract(l_proper);

    store_proper_rw_effects_list(current_stat, l_proper);
}

static void proper_effects_of_forloop(forloop l)
{
    statement current_stat = effects_private_current_stmt_head();
    list l_proper = NIL;
    list l_cumu_range = NIL;

    //    entity i = loop_index(l);
    // range r = loop_range(l);

    list li = NIL, lc = NIL, linc = NIL;

    pips_debug(2, "Effects for statement%03zd:\n",
	       statement_ordering(current_stat)); 

    // What is this about? See Fabien...
    // free_cumu_range_effects();
    //current_downward_cumulated_range_effects_pop();
    //l_cumu_range = cumu_range_effects();
    
    /* proper_effects first */

    li = generic_proper_effects_of_expression(forloop_initialization(l));

    /* effects of condition expression */
    lc = generic_proper_effects_of_expression(forloop_condition(l));
    /* effects of incrementation expression  */
    linc = generic_proper_effects_of_expression(forloop_increment(l));

    l_proper = gen_nconc(li, lc);
    l_proper = gen_nconc(l_proper, linc);
    l_proper = gen_nconc(l_proper, effects_dup(l_cumu_range));
  
    ifdebug(2)
    {
	pips_debug(2, "Proper effects for statement%03zd:\n",
		   statement_ordering(current_stat));  
	(*effects_prettyprint_func)(l_proper);
	pips_debug(2, "end\n");
    }

    if (contract_p)
	l_proper = proper_effects_contract(l_proper);

    store_proper_rw_effects_list(current_stat, l_proper);
}

static void proper_effects_of_while(whileloop w)
{
    statement current_stat = effects_private_current_stmt_head();
    list /* of effect */ l_proper = 
	generic_proper_effects_of_expression(whileloop_condition(w));
    store_proper_rw_effects_list(current_stat, l_proper);
}

static void proper_effects_of_test(test t)
{
    list l_proper=NIL;
    statement current_stat = effects_private_current_stmt_head();
    list l_cumu_range = cumu_range_effects();

    pips_debug(2, "Effects for statement%03zd:\n",
	       statement_ordering(current_stat)); 

    /* effects of the condition */
    l_proper = generic_proper_effects_of_expression(test_condition(t));
    l_proper = gen_nconc(l_proper, effects_dup(l_cumu_range));
    
    ifdebug(2)
    {
	pips_debug(2, "Proper effects for statement%03zd:\n",
		   statement_ordering(current_stat));  
	(*effects_prettyprint_func)(l_proper);
	pips_debug(2, "end\n");
    }

    if (contract_p)
	l_proper = proper_effects_contract(l_proper);
    store_proper_rw_effects_list(current_stat, l_proper);
}

static void proper_effects_of_sequence(sequence block __attribute__((__unused__)))
{
    statement current_stat = effects_private_current_stmt_head();   
    store_proper_rw_effects_list(current_stat, NIL);
}

static bool stmt_filter(statement s)
{
  pips_debug(1, "Entering statement %03zd :\n", statement_ordering(s));
  effects_private_current_stmt_push(s);
  effects_private_current_context_push((*load_context_func)(s));
  return(TRUE);
}

static void proper_effects_of_statement(statement s)
{
    if (!bound_proper_rw_effects_p(s)) 
     { 
 	pips_debug(2, "Warning, proper effects undefined, set to NIL\n"); 
 	store_proper_rw_effects_list(s,NIL);	 
     } 
    effects_private_current_stmt_pop();
    effects_private_current_context_pop();

    pips_debug(1, "End statement%03zd :\n", statement_ordering(s));
  
}

void proper_effects_of_module_statement(statement module_stat)
{    
    make_effects_private_current_stmt_stack();
    make_effects_private_current_context_stack();
    make_current_downward_cumulated_range_effects_stack();
    pips_debug(1,"begin\n");
    
    gen_multi_recurse
	(module_stat, 
	 statement_domain, stmt_filter, proper_effects_of_statement,
	 sequence_domain, gen_true, proper_effects_of_sequence,
	 test_domain, gen_true, proper_effects_of_test,
	 /* Reached only through syntax (see expression rule) */
	 call_domain, gen_true, proper_effects_of_call,
	 loop_domain, loop_filter, proper_effects_of_loop,
	 whileloop_domain, gen_true, proper_effects_of_while,
	 forloop_domain, gen_true, proper_effects_of_forloop,
	 unstructured_domain, gen_true, proper_effects_of_unstructured,
         /* Just to retrieve effects of instructions with kind
	    expression since they are ruled out by the next clause */
	 instruction_domain, gen_true, proper_effects_of_expression_instruction,
	 expression_domain, gen_false, gen_null, /* NOT THESE CALLS */
	 NULL); 

    pips_debug(1,"end\n");
    free_effects_private_current_stmt_stack();
    free_effects_private_current_context_stack();
    free_current_downward_cumulated_range_effects_stack();
}

bool proper_effects_engine(char *module_name)
{    
    /* Get the code of the module. */
    set_current_module_statement( (statement)
		      db_get_memory_resource(DBR_CODE, module_name, TRUE));

    set_current_module_entity(module_name_to_entity(module_name));

    (*effects_computation_init_func)(module_name);

    /* Compute the effects or references of the module. */
    init_proper_rw_effects();
  
    debug_on("PROPER_EFFECTS_DEBUG_LEVEL");
    pips_debug(1, "begin\n");

    proper_effects_of_module_statement(get_current_module_statement()); 

    pips_debug(1, "end\n");
    debug_off();

    (*db_put_proper_rw_effects_func)(module_name, get_proper_rw_effects());

    reset_current_module_entity();
    reset_current_module_statement();
    reset_proper_rw_effects();

    (*effects_computation_reset_func)(module_name);
    
    return(TRUE);
}

/* compute proper effects for both expressions and statements
   WARNING: the functions are set as a side effect.
 */
void 
expression_proper_effects_engine(
    string module_name,
    statement current)
{    
    (*effects_computation_init_func)(module_name);

    init_proper_rw_effects();
    init_expr_prw_effects();
  
    debug_on("PROPER_EFFECTS_DEBUG_LEVEL");
    pips_debug(1, "begin\n");

    proper_effects_of_module_statement(current); 

    pips_debug(1, "end\n");
    debug_off();

    (*effects_computation_reset_func)(module_name);
}
