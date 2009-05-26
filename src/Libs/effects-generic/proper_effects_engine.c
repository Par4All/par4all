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

/* list generic_proper_effects_of__read_reference(reference ref, bool written_p)
 * input    : a reference and a boolean true if it is a written reference.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  : effects of a reference that is either read or written. 	
 */
list 
generic_proper_effects_of_reference(reference ref, bool written_p)
{
  list le = NIL;
  transformer context = effects_private_current_context_head();
  entity v = reference_variable(ref);

  pips_debug(3, "begin\n");
  /* structure fields are referenced, not called, altough they are constant... */
  if(!entity_field_p(v)) 
    {
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
      
      if (! (*empty_context_test)(context)) 
	{
	  action ac;
	  effect eff;
	  
	  ac = written_p ? make_action(is_action_write, UU) : 
	    make_action(is_action_read, UU);
	  eff = (*reference_to_effect_func)(ref, ac);
	  /* if the effect is undefined, shouldn't we have an anywhere effect ?*/ 
	  le = effect_undefined_p(eff) ? NIL : CONS(EFFECT, eff, NIL);
	  
	  if (! ENDP(inds))
	    le = gen_nconc(le, generic_proper_effects_of_expressions(inds));
	  
	  (*effects_precondition_composition_op)(le, context);
	}
      
      pips_debug(3, "end\n");
    }
  return(le);
}


/* list generic_proper_effects_of_read_reference(reference ref)
 * input    : a reference that is read.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  : effects of a reference that is read 	
 */
list 
generic_proper_effects_of_read_reference(reference ref)
{
  list le = NIL;

  le = generic_proper_effects_of_reference(ref, FALSE);

  return(le);
}

/* list proper_effects_of_written_reference(reference ref)
 * input    : a reference that is written.
 * output   : the corresponding list of effects.
 * modifies : nothing.
 * comment  : effects of a reference that is written	
 */
list 
generic_proper_effects_of_written_reference(reference ref)
{    
  list le = NIL;
 
  le = generic_proper_effects_of_reference(ref, TRUE);
  
  return(le);
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

  pips_debug(3, "begin for expression : %s\n", 
	     words_to_string(words_expression(exp)));

  /* First step: see if we should recurse or not. Set s_exp if yes. 
     If not, set mr. */

  pips_debug(4, "First, see if we should recurse\n");
  if(syntax_reference_p(s)) 
    {
      pips_debug(4, "this is a reference; let's compute the effects of the indices.\n");

      /* Do not recurse any longer: the basis of the address expression 
	 is found */
      mr = copy_reference(syntax_reference(s));
      mre = copy_reference(syntax_reference(s));
      
      /* take care fo the object itself: it must be read, most of the
	 times. Also take care of its indices */
      /* We know it must be read because we are dealing with a complex
	 lhs, except when we have a field operator only... So, this read
	 effect mre will have to be taken care of later. */
      le = generic_proper_effects_of_expressions
	(reference_indices(syntax_reference(s)));
      finished_p = TRUE;
    }
  else if(syntax_call_p(s)) 
    {
      call c = syntax_call(s);
      entity op = call_function(c);
      list args = call_arguments(c);
      /* FI: we assume there it at least one argument */
      pips_debug(4, "This is a call\n");
      
      if(gen_length(args)==0) 
	{
	  /* Problem with *(1) which is syntactically legal; 
	     could also happend with hardware*/
	  pips_user_warning("Constant in a lhs expression: \"\%s\"\n",
			    words_to_string(words_expression(exp)));
	  /* Will be converted into an anywhere effect by the caller */
	  mr = reference_undefined;
	  finished_p = TRUE;
	}
      else if(ENTITY_FIELD_P(op) || ENTITY_POINT_TO_P(op)) 
	{
	  pips_debug(4, "Call is a field or a point to operator\n");
	  s_exp = EXPRESSION(CAR(args));
	}
      else if(ENTITY_DEREFERENCING_P(op)) 
	{
	  pips_debug(4, "Call is a dereferencing operator \n");
	  s_exp = EXPRESSION(CAR(args));
	  
	  if(expression_call_p(s_exp)) {
	    call s_c = syntax_call(expression_syntax(s_exp));
	    entity s_op = call_function(s_c);
	    list s_args = call_arguments(s_c);
	    
	    pips_debug(4,"The dereferenced expression is a call itself (%s)\n",
		       entity_local_name(s_op));
	    ifdebug(8)
	      {
		pips_debug(8,"with arguments : \n");
		 FOREACH(EXPRESSION,x,s_args)
		   {
		     print_expression(x);
		   }
	      }
	    
	    /* MINUS_C should be handled as well BC */
	    /* all this stuff should be moved in another function BC */
	    if(ENTITY_PLUS_C_P(s_op)) 
	      {
		/* case *(e1+e2) */
		/* This might be tractable if e1 is a reference to a
		   pointer. For instance, *(p+q-r) can be converted to p[q-r] */
		expression e1 = EXPRESSION(CAR(s_args));
		syntax s1 = expression_syntax(e1);
		expression e2 = EXPRESSION(CAR(CDR(s_args)));
		
		if(syntax_reference_p(s1)) 
		  {
		    reference r1 = syntax_reference(s1);
		    entity v1 = reference_variable(r1);
		    type t1 = ultimate_type(entity_type(v1));
		    if(type_variable_p(t1)) 
		      {
			variable vt1 = type_variable(t1);
			basic b1 = variable_basic(vt1);
			if(basic_pointer_p(b1))
			  {
			    /*normalized ne2 = NORMALIZE_EXPRESSION(e2);*/
			    syntax s2 = expression_syntax(e2);

			    /* deal with case *(p+(i=exp))
			     * the effect is equivalent to an effect on *(p+exp)
			     */
			    if (syntax_call_p(s2))
			      {
				call s2_c = syntax_call(s2);
				entity s2_op = call_function(s2_c);
				list s2_args = call_arguments(s2_c);
				if (ENTITY_ASSIGN_P(s2_op))
				  {
				    e2 =  EXPRESSION(CAR(CDR(s2_args)));
				  }
			      }
			    
			    /* We should verify here that e2 does not contain
			     * a call to an external or io function.
			     * In this case, we should generate an unbounded
			     * dimension. BC.
			     */
			    mr = make_reference(v1, CONS(EXPRESSION,e2,NIL));
			    mre = make_reference(v1, NIL);
			    le = generic_proper_effects_of_expression(e1);
			    le = gen_nconc(le, generic_proper_effects_of_expression(e2));
			    finished_p = TRUE;
			  } /* if(basic_pointer_p(b1)) */
		      }
		  }
		if(!finished_p) 
		  {
		  le = generic_proper_effects_of_expression(exp);
		  mr = reference_undefined;
		  finished_p = TRUE;
		  }
	      }
	    /* Other functions to process: p++, ++p, p--, --p */
	    else if(ENTITY_POST_INCREMENT_P(s_op) || 
		    ENTITY_POST_DECREMENT_P(s_op)) 
	      {
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
	    else if(ENTITY_PRE_INCREMENT_P(s_op) || 
		    ENTITY_PRE_DECREMENT_P(s_op)) 
	      {
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
	    else 
	      {
		/* do nothing, go down recursively to handle other calls */
		;
	      }
	  }
	  else 
	    {
	      /* This is not a call, go down recursively */
	      pips_debug(4,"The dereferenced expression is not a call itself : we go down recursively\n");
	      ;
	    }
	}
      else 
	{
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
  else if(syntax_cast_p(s)) 
    {
      pips_debug(4, "This is a cast\n");
      /* FI: The cast has an impact o pointer arithmetic. I do not know
	 how to take it into account. */
      pips_user_warning("Cast impact on pointer arithmetic and indexing is ignored\n");
      s_exp = cast_expression(syntax_cast(s));
    }
  else if(syntax_subscript_p(s)) 
    {
      pips_debug(4,"This is a subscript\n");
      s_exp = subscript_array(syntax_subscript(s));
    }
  else if(syntax_va_arg_p(s)) 
    {
      pips_debug(4,"This is a va_arg\n");
      /* The built-in can return a pointer which is dereferenced */
      /* va_args is read... */
      finished_p = TRUE;
    }
  else 
    {
      /* sizeofexpression, application. va_arg */
      pips_internal_error("Unexpected case\n");
    }

  if(finished_p) 
    {

      pips_debug(3, "Stop recursing\n");
      if(reference_undefined_p(mr)) 
	{
	  pips_debug(4, "mr is undefined\n");
	  *pmwe = effect_undefined;
	}
      else 
	{
	  effect me;
	  
	  pips_debug(4,"mr is defined\n");

	  me = (*reference_to_effect_func)
	    (mr, lhs_p?make_action_write():make_action_read());
	  *pmwe = me;

	  /* Can mre be undefined when mr is defined?*/
	  if(!reference_undefined_p(mre)) 
	    {
	      /* Read effect to generate for point_to and for dereferencing */
	      effect mree;

	      pips_debug(4,"mre is defined\n");
	      mree = 
		(*reference_to_effect_func)(mre, make_action_read());
	      *pmre = mree;
	    }
	  else 
	    {
	      pips_debug(8, "mr is defined but not mre\n");
	    }
	}
    }
  else 
    {
      pips_debug(8, "We go down recursively\n");
      le = gen_nconc
	(le, generic_proper_effects_of_complex_lhs(s_exp, pmwe, pmre, lhs_p));
      
      pips_debug(8,"Returning from recursive call of generic_proper_effects_of_complex_lhs : \n");
      
      if(!effect_undefined_p(*pmwe)) 
	{
	  /* Let's try to refine *pmwe with the current expression, the
	     current operator if any and the current second expression
	     when it exists */
	  reference mr = effect_any_reference(*pmwe);
	  list mr_inds = reference_indices(mr);
	  
	  pips_debug(4, "*pmwe is defined : try to refine it\n");

	  if(syntax_reference_p(s)) 
	    {
	      pips_internal_error("A reference should lead to the finished state\n");
	    }
	  else if(syntax_call_p(s)) 
	    {
	      call c = syntax_call(s);
	      entity op = call_function(c);
	      list args = call_arguments(c);
	      
	      if(ENTITY_FIELD_P(op)) 
		{
		  expression e2 = EXPRESSION(CAR(CDR(args)));
		  syntax s2 = expression_syntax(e2);
		  reference r2 = syntax_reference(s2);
		  entity f = reference_variable(r2);
		  int rank = entity_field_rank(f);
		  
		  pips_assert("e2 is a reference", syntax_reference_p(s2));
		  
		  pips_debug(4, "It's a field operator\n");

		  /* we extend *pmwe by adding a dimension corresponding
		  * to the rank of the field */
		  effect_add_field_dimension(*pmwe,rank);
		  finished_p = TRUE;
		}
	      else if(ENTITY_POINT_TO_P(op)) 
		{
		  /* Since the field in e2 implies a postindexing, *pmwe has to
		     be direct or post because we cannot support pre- and
		     post-indexing simultanesouly */
		  expression e2 = EXPRESSION(CAR(CDR(args)));
		  syntax s2 = expression_syntax(e2);
		  entity f;
		  int rank;

		  pips_assert("e2 is a reference", syntax_reference_p(s2));

		  f = reference_variable(syntax_reference(s2));
		  rank = entity_field_rank(f);
		  
		  pips_debug(4, "It's a point to operator\n");
		  
		  /* We add a dereferencing and a subscript */
		   effect_add_dereferencing_dimension(* pmwe);
		   effect_add_field_dimension(*pmwe,rank);
		   finished_p = TRUE;
		  
		  /* add effects due to e2 */
		  le = gen_nconc(le, generic_proper_effects_of_expression(e2));
		  
		  /* A read must to the main variable must be added to le,
		     unless an array is used as pointer; possibly an array
		     with only one element... */
		  if(effect_undefined_p(*pmre))
		    pips_debug(8, "Do not add *pmre to le\n");
		  else 
		    {
		      pips_debug(8, "Add *pmre to le\n");
		      le = gen_nconc(le, CONS(EFFECT, copy_effect(*pmre), NIL));
		    }
		}
	      else if(ENTITY_DEREFERENCING_P(op)) 
		{

		  pips_debug(4,"It's a dereferencing operator\n");
		  
		  effect_add_dereferencing_dimension(* pmwe);
		  finished_p = true;
		  /* A read must to the main variable must be added to le */
		  pips_debug(8, "Add *pmre to le\n");
		  le = gen_nconc(le, CONS(EFFECT, copy_effect(*pmre), NIL));
		}
	      else 
		{
		  pips_internal_error("Unexpected call to \"\%s\"\n", entity_name(op));
		}
	    }
	  else if(syntax_subscript_p(s)) 
	    {
	      subscript ss = syntax_subscript(s);
	      list ind = subscript_indices(ss);

	      pips_debug(4, "It's a subscript\n");
	      
	      /* We add the corresponding dimensions to the effect*pmwe
	       * and read effects on each subscript index
	       */
	      FOREACH(EXPRESSION, ind_exp, ind)
		{
		  (*effect_add_expression_dimension_func)(*pmwe, ind_exp);
		  le = gen_nconc(le, generic_proper_effects_of_expression(ind_exp));
		  
		}
	      finished_p = true;
	      
	      /* take care of the pointer itself: it must be read, but this
		 must have been done much earlier when the main reference in
		 the lhs has been found */
	    }
      else 
	{
	  /* we should be finished already because we do not know how to
	     handle these constructs and we knew that before going down
	     and up. */
	  pips_internal_error("Something wrong in RI or missing");
	}
	} /* end of !effect_undefined_p(*pmwe) */
    } /* */
  
  if(!finished_p && !effect_undefined_p(*pmwe)) 
    {
      /* The sub-effect could not be refined */
      /* Should'nt we replace pmwe with an anywhere effect ? */
      free_effect(*pmwe);
      *pmwe = effect_undefined;
    }
  
  ifdebug(8) 
    {
      pips_debug(8, "End with le=\n");
      (*effects_prettyprint_func)(le);
      if(effect_undefined_p(*pmwe)) {
	pips_debug(8, "And *pmwe:\n");
	fprintf(stderr, "EFFECT UNDEFINED\n");
      }
      else 
	{
	  pips_debug(8, "And *pmwe :\n");
	  print_effect(*pmwe);
	}
      if(effect_undefined_p(*pmre)) 
	{
	  pips_debug(8, "And *pmre:\n");
	  fprintf(stderr, "EFFECT UNDEFINED\n");
	}
      else 
	{
	  pips_debug(8, "And *pmre \n");
	  print_effect(*pmre);
	}
    }
  
  return le;
}


list generic_proper_effects_of_any_lhs(expression lhs)
{
  return generic_proper_effects_of_address_expression(lhs, TRUE);
}

/* FI: lhs should be subtituted by ae */
list generic_proper_effects_of_address_expression(expression addexp, int write_p)
{
  list le = NIL;
  syntax s = expression_syntax(addexp);


  pips_debug(5, "begin for expression : %s\n", 
	     words_to_string(words_expression(addexp)));

  switch (syntax_tag(s))
    {
    case is_syntax_reference:
      {
	pips_debug(5, "reference case\n");
	if(write_p)
	  le = generic_proper_effects_of_written_reference(syntax_reference(s));
	else
	  pips_internal_error("Case not taken into account");
	break;
      }
    case is_syntax_call:
      pips_debug(5, "call case\n");
    case is_syntax_subscript:
      {
	effect e = effect_undefined; /* main data read-write effect: p[*] */
	effect re = effect_undefined; /* main pointer read effect: p */
	effect ge = effect_undefined; /* generic effect */
      
	pips_debug(5, "call or subscript case\n");
	/* Look for a main read-write effect of the lhs and for its 
	   secondary effects */
	le = generic_proper_effects_of_complex_lhs(addexp, &e, &re, write_p);
      
	if(!effect_undefined_p(re)) {
	  /* Copies of re were used to deal with complex addressing. The
	     data structure is no longer useful */
	  free_effect(re);
	}

	if(!effect_undefined_p(e)) 
	  {
	    transformer context = effects_private_current_context_head();
	    le = CONS(EFFECT, e, le);	    
	    (*effects_precondition_composition_op)(le, context);
	    
	  }
	else 
	  {
	    /* add an anywhere effect */
	    ge = anywhere_effect
	      (write_p? make_action_write() : make_action_read());
	    le = CONS(EFFECT, ge, le);
	    
	  }

	ifdebug(8) {
	  pips_debug(8, "Effect for a call:\n");
	  (*effects_prettyprint_func)(le);
	}
	break;
      }
    case is_syntax_cast:
      {
	pips_user_error("use of cast expressions as lvalues is deprecated\n");
	break;
      }
    case is_syntax_sizeofexpression:
      {
	pips_user_error("sizeof cannot be a lhs\n");	
	break;
      }
    case is_syntax_application:
      {
	/* I assume this not any more possible than a standard call */
	pips_user_error
	  ("use of indirect function call as lhs is not allowed\n");
	break;
      }
    default:
      pips_internal_error
	("lhs is not a reference and is not handled yet: syntax tag=%d\n",
	 syntax_tag(s));
      
    } /* end switch */


  ifdebug(8) {
    pips_debug(8, "End with le=\n");
    (*effects_prettyprint_func)(le);
    fprintf(stderr, "\n");
  }

  return le;
}


/* TO VERIFY !!!!!!!!!!!!!*/
/* UNUSED ? */
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



/* list generic_proper_effects_of_expression(expression e)
 * input : an expression
 * output   : the correpsonding list of effects.
 * modifies : nothing.
 * comment  :	
 */
list 
generic_proper_effects_of_expression(expression e)
{
  list le = NIL;
  syntax s;

  pips_debug(3, "begin\n");

  s = expression_syntax(e);

  switch(syntax_tag(s))
    {
    case is_syntax_reference:
      le = generic_proper_effects_of_read_reference(syntax_reference(s));
      break;
    case is_syntax_range:
      le = generic_proper_effects_of_range(syntax_range(s));
      break;
    case is_syntax_call:
      {
	entity op = call_function(syntax_call(s));

	/* first the case of an adressing operator : this could also be done
	 * by calling generic_r_proper_effects_of_call, but then the expression
	 * is lost and must be rebuild later to call 
	 * g_p_e_of_address_expression.
	 */
	if (ENTITY_FIELD_P(op) || 
	    ENTITY_POINT_TO_P(op) || 
	    ENTITY_DEREFERENCING_P(op))
	  le = generic_proper_effects_of_address_expression(e, FALSE);
	else
	  le = generic_r_proper_effects_of_call(syntax_call(s));
	break;
      }
    case is_syntax_cast: 
      le = generic_proper_effects_of_expression(cast_expression(syntax_cast(s)));
      break;
    case is_syntax_sizeofexpression:
      {
	sizeofexpression se = syntax_sizeofexpression(s);
	if (sizeofexpression_expression_p(se)) 
	  {
	    /* FI: If the type of the reference is a dependent type, this
	       may imply the reading of some expressions... See for
	       instance type_supporting_entities()? Is sizeof(a[i]) ok? */
	    /* The type of the variable is read, not the variable itself.*/
	    /* le = generic_proper_effects_of_expression(sizeofexpression_expression(se)); */
	  ;
	  }
	break;
      }
    case is_syntax_subscript:
      {
	le = generic_proper_effects_of_address_expression(e, FALSE);
	break;
      }
    case is_syntax_application:
      le = generic_proper_effects_of_application(syntax_application(s));
      break;
    case is_syntax_va_arg: 
      {
	list al = syntax_va_arg(s);
	sizeofexpression ae = SIZEOFEXPRESSION(CAR(al));
	le = generic_proper_effects_of_expression
	  (sizeofexpression_expression(ae));
	break;
      }
    default:
      pips_internal_error("unexpected tag %d\n", syntax_tag(s));
    }
  
  ifdebug(8)
    {
	pips_debug(8, "Proper effects of expression \"%s\":\n",
		   words_to_string(words_syntax(s)));
	(*effects_prettyprint_func)(le);
    }
  
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

bool check_sdfi_effects_p(entity func, list func_sdfi)
{
  list ce = list_undefined;
  bool check_p = TRUE;
  type ut = ultimate_type(entity_type(func));

  pips_assert("func is a function", type_functional_p(ut));

  /* Check the SDFI effects */
  for(ce = func_sdfi; !ENDP(ce); POP(ce)) {
    effect eff = EFFECT(CAR(ce));
    reference r = effect_any_reference(eff);
    entity v = reference_variable(r);

    if(formal_parameter_p(v)) {
      storage s = entity_storage(v);
      formal fs = storage_formal(s);
      int rank = formal_offset(fs);
      entity called_function = formal_function(fs);

      if(called_function!=func) {
	fprintf(stderr, "Summary effect %p for function \"%s\" refers to "
		"formal parameter \"%s\" of function \"%s\"\n",
		eff, entity_name(func), entity_name(v), entity_name(called_function));
	check_p = FALSE;
      }

      if(rank>gen_length(functional_parameters(type_functional(ut)))) {
	fprintf(stderr, "Formal parameter \"%s\" is ranked %d out of %zd!\n",
		entity_name(v), rank, gen_length(functional_parameters(type_functional(ut))));
	check_p = FALSE;
      }
    }
  }
  return check_p;
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

	if(!check_sdfi_effects_p(func, func_eff))
	  pips_internal_error("SDFI effects for \"%s\" are corrupted in the data base\n",
			      entity_name(func));

	/* Translate them using context information. */
	context = effects_private_current_context_head();
	le = (*effects_backward_translation_op)(func, args, func_eff, context);

	if(!check_sdfi_effects_p(func, func_eff))
	  pips_internal_error("SDFI effects for \"%s\" have been corrupted by the translation\n",
			      entity_name(func));
    }
    return le;  
}

/* list proper_effects_of_call(call c, transformer context, list *plpropreg)
 * @return the list of effects found.
 * @param c, a call, which can be a call to a subroutine, but also
 * to an function, or to an intrinsic, or even an assignement.
 * And a pointer that will be the proper effects of the call; NIL,
 * except for an intrinsic (assignment or real FORTRAN intrinsic).
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
  type uet = ultimate_type(entity_type(e));

  pips_debug(2, "begin for %s\n", entity_local_name(e));

  if(type_functional_p(uet)) {
    switch (t) {
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
      break;
    }
  }
  else if(type_variable_p(uet)) {
    /* We could be less optimistic even when no information about the function called is known.
     *
     * We could look up all functions with the same type and make the union of their effects.
     *
     * We could assume that all parameters are read.
     *
     * We could assume that all pointers are used to produce indirect write.
     */
    pips_user_warning("Effects of call thru functional pointers are ignored\n");
  }
  else if(type_statement_p(uet)) {
    le = NIL;
  }
  else {
    pips_internal_error("Unexpected case\n");
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
      pips_debug(2, "Effects for statement %03zd:\n",
		   statement_ordering(current_stat)); 

	l_proper = generic_r_proper_effects_of_call(c);

	l_proper = gen_nconc(l_proper, effects_dup(l_cumu_range));
		
	if (contract_p)
	    l_proper = proper_effects_contract(l_proper);

	ifdebug(2) {
	  pips_debug(2, "Proper effects for statement %03zd:\n",
		     statement_ordering(current_stat));  
	  (*effects_prettyprint_func)(l_proper);
	  pips_debug(2, "end\n");
	}

	store_proper_rw_effects_list(current_stat, l_proper);
    }
}

/* just to handle one kind of instruction, expressions which are not calls */
static void proper_effects_of_expression_instruction(instruction i)
{
  list l_proper = NIL;
  statement current_stat = effects_private_current_stmt_head();
  //instruction inst = statement_instruction(current_stat);
  list l_cumu_range = cumu_range_effects();

  /* Is the call an instruction, or a sub-expression? */
  if (instruction_expression_p(i)) {
    expression ie = instruction_expression(i);
    syntax is = expression_syntax(ie);
    call c = call_undefined;

    switch (syntax_tag(is)) 
      {
      case is_syntax_cast :
	{
	expression ce = cast_expression(syntax_cast(is));
	syntax sc = expression_syntax(ce);

	if(syntax_call_p(sc)) {
	  c = syntax_call(sc);
	  l_proper = generic_r_proper_effects_of_call(c);
	}
	else {
	  pips_internal_error("Cast case not implemented\n");
	}
	
	break;
	}
      case is_syntax_call :
	{
	  /* This may happen when a loop is unstructured by the controlizer */
	  c = syntax_call(is);
	  l_proper = generic_r_proper_effects_of_call(c);
	  break;
	}
      case is_syntax_application :
	{
	  /* This may happen when a structure field contains a pointer to
	     a function. We do not know which function is is... */
	  application a = syntax_application(is);
	  expression fe = application_function(a);
	  
	  /* Effect to find which function it it */
	  l_proper = generic_proper_effects_of_expression(fe);
	  /* More effects should be added to take the call site into account */
	  /* Same as for pointer-based call: use type, assume worst case,... */
	  /* A new function is needed to retrieve all functions with a
	     given signature. Then the effects of all the candidates must
	     be unioned. */
	  pips_user_warning("Effects of call site using a function pointer in "
			    "a structure are ignored for the time being\n");
	  break;
	}
      default :
	pips_internal_error("Instruction expression case not implemented\n");
    }
     
    pips_debug(2, "Effects for expression instruction in statement%03zd:\n",
	       statement_ordering(current_stat)); 

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
    li = generic_proper_effects_of_written_reference(make_reference(i, NIL));

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
