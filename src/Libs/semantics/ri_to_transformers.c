 /* semantical analysis
  *
  * phasis 1: compute transformers from statements and statements effects
  *
  * For (simple) interprocedural analysis, this phasis should be performed
  * bottom-up on the call tree.
  *
  * Francois Irigoin, April 1990
  *
  * $Id$
  *
  * $Log: ri_to_transformers.c,v $
  * Revision 1.61  2003/07/24 10:54:37  irigoin
  * More debugging messages, more normalization steps, handling of
  * unstructured statements moved into usntructured.c.
  *
  * Revision 1.60  2001/10/22 15:42:16  irigoin
  * Intraprocedural preconditions can be propagated along transformers to
  * refine them.
  *
  * Revision 1.59  2001/07/24 13:21:55  irigoin
  * Formatting added
  *
  * Revision 1.58  2001/07/24 13:18:00  irigoin
  * Cleanup of test_to_transformer() to handle side effects in test conditions
  *
  * Revision 1.57  2001/07/19 17:58:09  irigoin
  * Two bug fixes + reformatting with a smaller indent
  *
  * Revision 1.56  2001/07/13 15:02:58  irigoin
  * Restructured version with separate processing of loops and
  * expressions. Multitype version.
  *
  * Revision 1.55  2001/02/07 18:14:21  irigoin
  * New C format + support for recomputing loop fixpoints with precondition information
  *
  * Revision 1.54  2000/11/23 17:17:31  irigoin
  * Function moved into unstructured.c, typing in debugging statement,
  * consistency checks
  *
  * Revision 1.53  2000/11/03 17:15:07  irigoin
  * Declarations and references are trusted or not. Better handling of unstructured.
  *
  * Revision 1.52  2000/07/20 14:29:42  coelho
  * cleaner.
  *
  * Revision 1.51  2000/07/20 14:25:26  coelho
  * leaks--
  *
  * Revision 1.50  2000/07/20 13:55:40  irigoin
  * cleaning...
  *
  * Revision 1.49  2000/05/25 08:37:55  coelho
  * no more successor when adding an eq of ineq.
  *
  * Revision 1.48  1999/01/07 16:44:14  irigoin
  * Bug fix in user_call_to_transformer() to handle aliasing between two formal parameters. See spice02.f in Validation.
  *
  * Revision 1.47  1999/01/07 07:52:32  irigoin
  * Bug fix in minmax_to_transformer()
  *
  */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"
#include "database.h"
#include "linear.h"
#include "ri.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "constants.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"

#include "misc.h"

#include "properties.h"

#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "transformer.h"

#include "semantics.h"

extern Psysteme sc_projection_by_eq(Psysteme sc, Pcontrainte eq, Variable v);

transformer effects_to_transformer(list e) /* list of effects */
{
  /* algorithm: keep only write effects on variables with values */
  list args = NIL;
  Pbase b = VECTEUR_NUL;
  Psysteme s = sc_new();

  MAP(EFFECT, ef, 
  {
    reference r = effect_reference(ef);
    action a = effect_action(ef);
    entity v = reference_variable(r);
    
    if(action_write_p(a) && entity_has_values_p(v)) 
    {
      entity new_val = entity_to_new_value(v);
      args = arguments_add_entity(args, new_val);
      b = vect_add_variable(b, (Variable) new_val);
    }
  },
      e);
  
  s->base = b;
  s->dimension = vect_size(b);

  return make_transformer(args, make_predicate(s));
}

transformer filter_transformer(transformer t, list e) 
{
  /* algorithm: keep only information about scalar variables with values
   * appearing in effects e and store it into a newly allocated transformer
   */
  Pbase b = VECTEUR_NUL;
  Psysteme s = SC_UNDEFINED;
  Psysteme sc = predicate_system(transformer_relation(t));
  list args = NIL;
  Psysteme sc_restricted_to_variables_transitive_closure(Psysteme, Pbase);

  MAP(EFFECT, ef, 
  {
    reference r = effect_reference(ef);
    /* action a = effect_action(ef); */
    entity v = reference_variable(r);
    
    if(/* action_write_p(a) && */ entity_has_values_p(v)) {
      /* I do not know yet if I should keep old values... */
      entity new_val = entity_to_new_value(v);
      b = vect_add_variable(b, (Variable) new_val);
      
      if(entity_is_argument_p(v, transformer_arguments(t))) {
	args = arguments_add_entity(args, v);
      }
    }
  },
      e);
  
  /* FI: I should check if sc is sc_empty but I haven't (yet) found a
     cheap syntactic test */
  s = sc_restricted_to_variables_transitive_closure(sc, b);

  return make_transformer(args, make_predicate(s));
}


/* Recursive Descent in Data Structure Statement */

/* SHARING : returns the transformer stored in the database. Make a copy 
 * before using it. The copy is not made here because the result is not 
 * always used after a call to this function, and it would create non 
 * reachable structures. Another solution would be to store a copy and free 
 * the unused result in the calling function but transformer_free does not 
 * really free the transformer. Not very clean. 
 * BC, oct. 94 
 */

static transformer 
block_to_transformer(list b, transformer pre)
{
  statement s;
  transformer btf = transformer_undefined;
  transformer stf = transformer_undefined;
  transformer post = transformer_undefined;
  transformer next_pre = transformer_undefined;
  list l = b;

  pips_debug(8,"begin\n");

  if(ENDP(l))
    btf = transformer_identity();
  else {
    s = STATEMENT(CAR(l));
    stf = statement_to_transformer(s, pre);
    post = transformer_safe_apply(stf, pre);
    post = transformer_safe_normalize(post, 4);
    btf = transformer_dup(stf);
    for (POP(l) ; !ENDP(l); POP(l)) {
      s = STATEMENT(CAR(l));
      if(!transformer_undefined_p(next_pre))
	free_transformer(next_pre);
      next_pre = post;
      stf = statement_to_transformer(s, next_pre);
      post = transformer_safe_apply(stf, next_pre);
      post = transformer_safe_normalize(post, 4);
      btf = transformer_combine(btf, stf);
      btf = transformer_normalize(btf, 4);
      ifdebug(1) 
	pips_assert("btf is a consistent transformer", 
		    transformer_consistency_p(btf));
	pips_assert("post is a consistent transformer if pre is defined", 
		    transformer_undefined_p(pre)
		    || transformer_consistency_p(post));
    }
    free_transformer(post);
  }

  pips_debug(8, "end\n");
  return btf;
}

list 
effects_to_arguments(list fx) /* list of effects */
{
  /* algorithm: keep only write effects on scalar variable with values */
  list args = NIL;

  MAP(EFFECT, ef, 
  {
    reference r = effect_reference(ef);
    action a = effect_action(ef);
    entity e = reference_variable(r);
	
    if(action_write_p(a) && entity_has_values_p(e)) {
      args = arguments_add_entity(args, e);
    }
  },
      fx);

  return args;
}


static transformer 
test_to_transformer(test t, transformer pre, list ef) /* effects of t */
{
  statement st = test_true(t);
  statement sf = test_false(t);
  transformer tf;

  /* EXPRESSION_TO_TRANSFORMER() SHOULD BE USED MORE EFFECTIVELY */

  pips_debug(8,"begin\n");

  if(pips_flag_p(SEMANTICS_FLOW_SENSITIVE)) {
    expression e = test_condition(t);
    /* Ideally, they should be initialized with the current best
       precondition, intraprocedural if nothing else better is
       available. This function's profile as well as most function
       profiles in ri_to_transformers should be modifed. */
    transformer tftwc = transformer_undefined_p(pre)?
      transformer_identity() :
      precondition_to_abstract_store(pre);
    transformer context = transformer_dup(tftwc);
    transformer tffwc = transformer_dup(tftwc);
    transformer post_tftwc = transformer_undefined;
    transformer post_tffwc = transformer_undefined;
    list ta = NIL;
    list fa = NIL;

    /*
    tftwc = transformer_dup(statement_to_transformer(st));
    tffwc = transformer_dup(statement_to_transformer(sf));
    */

    tftwc = precondition_add_condition_information(tftwc, e, context, TRUE);
    ifdebug(8) {
      pips_debug(8, "tftwc before transformer_temporary_value_projection %p:\n", tftwc);
      (void) print_transformer(tftwc);
    }
    tftwc = transformer_temporary_value_projection(tftwc);
    reset_temporary_value_counter();
    ifdebug(8) {
      pips_debug(8, "tftwc before transformer_apply %p:\n", tftwc);
      (void) print_transformer(tftwc);
    }
    post_tftwc = transformer_apply(statement_to_transformer(st, tftwc), tftwc);
    ifdebug(8) {
      pips_debug(8, "tftwc after transformer_apply %p:\n", tftwc);
      (void) print_transformer(tftwc);
      pips_debug(8, "post_tftwc after transformer_apply %p:\n", post_tftwc);
      (void) print_transformer(post_tftwc);
    }

    tffwc = precondition_add_condition_information(tffwc, e, context, FALSE);
    tffwc = transformer_temporary_value_projection(tffwc);
    reset_temporary_value_counter();
    post_tffwc = transformer_apply(statement_to_transformer(sf, tffwc), tffwc);

    ifdebug(8) {
      pips_debug(8, "post_tftwc before transformer_convex_hull %p:\n", post_tftwc);
      (void) print_transformer(post_tftwc);
      pips_debug(8, "post_tffwc after transformer_apply %p:\n", post_tffwc);
      (void) print_transformer(post_tffwc);
    }
    tf = transformer_convex_hull(post_tftwc, post_tffwc);
    transformer_free(context);
    transformer_free(tftwc);
    transformer_free(tffwc);
    transformer_free(post_tftwc);
    transformer_free(post_tffwc);
    free_arguments(ta);
    free_arguments(fa);
  }
  else {
    transformer id = transformer_identity();
    (void) statement_to_transformer(st, id);
    (void) statement_to_transformer(sf, id);
    tf = effects_to_transformer(ef);
    free_transformer(id);
  }

  debug(8,"test_to_transformer","end\n");
  return tf;
}

transformer 
intrinsic_to_transformer(
    entity e, list pc, transformer pre, list ef) /* effects of intrinsic call */
{
  transformer tf = transformer_undefined;

  pips_debug(8, "begin\n");

  if(ENTITY_ASSIGN_P(e)) {
    tf = any_assign_to_transformer(pc, ef, pre);
  }
  else if(ENTITY_STOP_P(e))
    tf = transformer_empty();
  else
    tf = effects_to_transformer(ef);

  pips_debug(8, "end\n");

  return tf;
}

static transformer user_call_to_transformer(entity, list, list);

static transformer 
call_to_transformer(call c, transformer pre, list ef) /* effects of call c */
{
  transformer tf = transformer_undefined;
  entity e = call_function(c);
  cons *pc = call_arguments(c);
  tag tt;

  pips_debug(8,"begin with precondition %p\n", pre);

  switch (tt = value_tag(entity_initial(e))) {
  case is_value_code:
    /* call to an external function; preliminary version:
       rely on effects */
    pips_debug(5, "external function %s\n", entity_name(e));
    if(get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
      tf = user_call_to_transformer(e, pc, ef);
      reset_temporary_value_counter();
    }
    else
      tf = effects_to_transformer(ef);
    break;
  case is_value_symbolic:
  case is_value_constant:
    tf = transformer_identity();
    break;
  case is_value_unknown:
    pips_internal_error("function %s has an unknown value\n", entity_name(e));
    break;
  case is_value_intrinsic:
    pips_debug(5, "intrinsic function %s\n", entity_name(e));
    tf = intrinsic_to_transformer(e, pc, pre, ef);
    break;
  default:
    pips_internal_error("unknown tag %d\n", tt);
  }
  pips_assert("transformer tt is consistent", 
	      transformer_consistency_p(tf)); 

  pips_debug(8,"Transformer before intersection with precondition, tf=%p\n", tf);
  ifdebug(8) {
    (void) print_transformer(tf);
  }

  /* Add information from pre. Invariant information is easy to
     use. Information about initial values, that is final values in pre,
     can also be used. */
  tf = transformer_safe_domain_intersection(tf, pre);
  pips_debug(8,"After intersection and before normalization with tf=%p\n", tf);
  ifdebug(8) {
    (void) print_transformer(tf);
  }
  pips_debug(8,"with precondition pre=%p\n", pre);
  ifdebug(8) {
    (void) print_transformer(pre);
  }
  tf = transformer_normalize(tf, 4);

  pips_debug(8,"end after normalization with tf=%p\n", tf);
  ifdebug(8) {
    (void) print_transformer(tf);
  }

  return(tf);
}

transformer 
user_function_call_to_transformer(
				  entity e, /* a value */
				  expression expr) /* a call to a function */
{
  syntax s = expression_syntax(expr);
  call c = syntax_call(s);
  entity f = call_function(c);
  list pc = call_arguments(c);
  transformer t_caller = transformer_undefined;
  basic rbt = basic_of_call(c);
  list ef = expression_to_proper_effects(expr);

  pips_debug(8, "begin\n");
  pips_assert("s is a call", syntax_call_p(s));

  /* if(basic_int_p(rbt)) { */
  if(basic_equal_p(rbt, variable_basic(type_variable(entity_type(e))))) {
    string fn = module_local_name(f);
    entity rv = global_name_to_entity(fn, fn);
    entity orv = entity_undefined;
    Psysteme sc = SC_UNDEFINED;
    Pcontrainte c = CONTRAINTE_UNDEFINED;
    Pvecteur eq = VECTEUR_NUL;
    transformer t_equal = transformer_undefined;

    pips_assert("rv is defined",
		!entity_undefined_p(rv));

    /* Build a transformer reflecting the call site */
    t_caller = user_call_to_transformer(f, pc, ef);

    ifdebug(8) {
      pips_debug(8, "Transformer %p for callee %s:\n",
		 t_caller, entity_local_name(f));
      dump_transformer(t_caller);
    }

    /* Build a transformer representing the equality of
     * the function value to e
     */
    eq = vect_make(eq,
		   (Variable) e, VALUE_ONE,
		   (Variable) rv, VALUE_MONE,
		   TCST, VALUE_ZERO);
    c = contrainte_make(eq);
    sc = sc_make(c, CONTRAINTE_UNDEFINED);
    t_equal = make_transformer(NIL,
			       make_predicate(sc));

    /* Consistency cannot be checked on a non-local transformer */
    /* pips_assert("t_equal is consistent",
       transformer_consistency_p(t_equal)); */

    ifdebug(8) {
      pips_debug(8,
		 "Transformer %p for equality of %s with %s:\n",
		 t_equal, entity_local_name(e), entity_name (rv));
      dump_transformer(t_equal);
    }

    /* Combine the effect of the function call and of the equality */
    t_caller = transformer_combine(t_caller, t_equal);
    free_transformer(t_equal);

    /* Get rid of the temporary representing the function's value */
    orv = global_new_value_to_global_old_value(rv);
    t_caller = transformer_filter(t_caller,
				  CONS(ENTITY, rv, CONS(ENTITY, orv, NIL)));


    ifdebug(8) {
      pips_debug(8,
		 "Final transformer %p for assignment of %s with %s:\n",
		 t_caller, entity_local_name(e), entity_name(rv));
      dump_transformer(t_caller);
    }

    /* FI: e is added in arguments because user_call_to_transformer()
     * uses effects to make sure arrays and non scalar integer variables
     * impact is taken into account
     */
    /*
      transformer_arguments(t_caller) =
      arguments_rm_entity(transformer_arguments(t_caller), e);
    */

    /* FI, FI: il vaudrait mieux ne pas eliminer e d'abord1 */
    /* J'ai aussi des free a decommenter */
    /*
      if(ENDP(transformer_arguments(t_caller))) {
      transformer_arguments(t_caller) = 
      gen_nconc(transformer_arguments(t_caller), CONS(ENTITY, e, NIL));
      }
      else {
      t_caller = transformer_value_substitute(t_caller, rv, e);
      }
    */
    /* Not checkable with temporary variables
       pips_assert("transformer t_caller is consistent", 
       transformer_consistency_p(t_caller));
    */
  }
  else {
    pips_assert("transformer t_caller is undefined", 
		transformer_undefined_p(t_caller));
  }

  gen_free_list(ef);

  pips_debug(8, "end with t_caller=%p\n", t_caller);

    
  return t_caller;
}

/* transformer translation
 */
transformer 
transformer_intra_to_inter(
    transformer tf,
    list le)
{
  cons * lost_args = NIL;
  /* Filtered TransFormer ftf */
  transformer ftf = transformer_dup(tf);
  /* cons * old_args = transformer_arguments(ftf); */
  Psysteme sc = SC_UNDEFINED;
  Pbase b = BASE_UNDEFINED;
  Pbase eb = BASE_UNDEFINED;

  pips_debug(8,"begin\n");
  pips_debug(8,"argument tf=%p\n",ftf);
  ifdebug(8) (void) dump_transformer(ftf);

  /* get rid of tf's arguments that do not appear in effects le */

  /* build a list of arguments to suppress */
  /* FI: I do not understand anymore why corresponding old values do not have
   * to be suppressed too (6 July 1993)
   *
   * FI: because only read arguments are eliminated, non? (12 November 1995)
   *
   * FI: the resulting intermediate transformer is not consistent (18 July 2003)
   */
  /*
  MAPL(ca, 
  {entity e = ENTITY(CAR(ca));
  if(!effects_write_entity_p(le, e) &&
     !storage_return_p(entity_storage(e))) 
    lost_args = arguments_add_entity(lost_args, e);
  },
       old_args);
  */
  /* get rid of them */
  /* ftf = transformer_projection(ftf, lost_args); */

  /* free the temporary list of entities */
  /*
  gen_free_list(lost_args);
  lost_args = NIL;

  debug(8,"transformer_intra_to_inter","after first filtering ftf=%x\n",ftf);
  ifdebug(8) (void) dump_transformer(ftf);
  */

  /* get rid of local read variables */

  /* FI: why not use this loop to get rid of *all* local variables, read or written? */

  sc = (Psysteme) predicate_system(transformer_relation(ftf));
  b = sc_base(sc);
  for(eb=b; !BASE_UNDEFINED_P(eb); eb = eb->succ) {
    entity e = (entity) vecteur_var(eb);

    if(e != (entity) TCST) {
      entity v = value_to_variable(e);

      /* Variables with no impact on the caller world are eliminated.
       * However, the return value associated to a function is preserved.
       */
      if( ! effects_read_or_write_entity_p(le, v) &&
	  !storage_return_p(entity_storage(v)) &&
	  !entity_constant_p(v)) {
	lost_args = arguments_add_entity(lost_args, e);
      }
    }
  }

  /* get rid of them */
  ftf = transformer_projection(ftf, lost_args);

  /* free the temporary list of entities */
  gen_free_list(lost_args);
  lost_args = NIL;

  debug(8,"transformer_intra_to_inter","return ftf=%x\n",ftf);
  ifdebug(8) (void) dump_transformer(ftf);
  debug(8,"transformer_intra_to_inter","end\n");

  return ftf;
}

/* Effects are necessary to clean up the transformer t_caller. For
   instance, an effect on variable X may not be taken into account in
   t_callee but it may be equivalenced thru a common to a variable i which
   is analyzed in the caller. If X is written, I value is lost. See
   Validation/equiv02.f. */

static transformer 
user_call_to_transformer(
    entity f,
    list pc,
    list ef)
{
  transformer t_callee = transformer_undefined;
  transformer t_caller = transformer_undefined;
  transformer t_effects = transformer_undefined;
  entity caller = entity_undefined;
  list all_args = list_undefined;

  pips_debug(8, "begin\n");
  pips_assert("f is a module", entity_module_p(f));

  if(!get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
    /*
      pips_user_warning(
      "unknown interprocedural transformer for %s\n",
      entity_local_name(f));
    */
    t_caller = effects_to_transformer(ef);
  }
  else {
    /* add equations linking formal parameters to argument expressions
       to transformer t_callee and project along the formal parameters */
    /* for performance, it  would be better to avoid building formals
       and to inline entity_to_formal_parameters */
    /* it wouls also be useful to distinguish between in and out
       parameters; I'm not sure the information is really available
       in a field ??? */
    list formals = module_to_formal_analyzable_parameters(f);
    list formals_new = NIL;
    cons * ce;

    t_callee = load_summary_transformer(f);

    ifdebug(8) {
      Psysteme s = 
	(Psysteme) predicate_system(transformer_relation(t_callee));
      pips_debug(8, "Transformer for callee %s:\n", 
		 entity_local_name(f));
      dump_transformer(t_callee);
      sc_fprint(stderr, s, (char * (*)(Variable)) dump_value_name);
    }

    t_caller = transformer_dup(t_callee);

    /* take care of analyzable formal parameters */

    for( ce = formals; !ENDP(ce); POP(ce)) {
      entity fp = ENTITY(CAR(ce));
      int r = formal_offset(storage_formal(entity_storage(fp)));
      expression expr = find_ith_argument(pc, r);

      if(expr == expression_undefined)
	pips_user_error("not enough args for %d formal parm."
			" %s in call to %s from %s\n",
			r, entity_local_name(fp), entity_local_name(f),
			get_current_module_entity());
      else {
	/* type checking. You already know that fp is a scalar variable */
	type tfp = entity_type(fp);
	basic bfp = variable_basic(type_variable(tfp));
	basic bexpr = basic_of_expression(expr);

	if(!basic_equal_p(bfp, bexpr)) {
	  pips_user_warning("Type incompatibility\n(formal %s/actual %s)"
			    "\nfor formal parameter %s (rank %d)"
			    "\nin call to %s from %s\n",
			    basic_to_string(bfp), basic_to_string(bexpr),
			    entity_local_name(fp), r, module_local_name(f),
			    module_local_name(get_current_module_entity()));
	  continue;
	}
      }

      if(entity_is_argument_p(fp, transformer_arguments(t_callee))) {
	/* formal parameter e is modified. expr must be a reference */
	syntax sexpr = expression_syntax(expr);

	if(syntax_reference_p(sexpr)) {
	  entity ap = reference_variable(syntax_reference(sexpr));

	  if(entity_has_values_p(ap)) {
	    Psysteme s = (Psysteme) predicate_system(transformer_relation(t_caller));
	    entity ap_new = entity_to_new_value(ap);
	    entity ap_old = entity_to_old_value(ap);

	    if(base_contains_variable_p(s->base, (Variable) ap_new)) {
	      pips_user_error(
			      "Variable %s seems to be aliased thru variable %s"
			      " at a call site to %s in %s\n"
			      "PIPS semantics analysis assumes no aliasing as"
			      " imposed by the Fortran standard.\n",
			      entity_name(fp),
			      entity_name(value_to_variable(ap_new)),
			      module_local_name(f),
			      get_current_module_name());
	    }
	    else { /* normal case: ap_new==fp_new, ap_old==fp_old */
	      entity fp_new = external_entity_to_new_value(fp);
	      entity fp_old = external_entity_to_old_value(fp);

	      t_caller = transformer_value_substitute
		(t_caller, fp_new, ap_new);
	      t_caller = transformer_value_substitute
		(t_caller, fp_old, ap_old);
	    }
	  }
	  else { /* Variable ap is not analyzed. The information about fp
                    will be lost. */
	    ;
	  }
	}
	else {
	  /* Attemps at modifying a value: expr is call, fp is modified */
	  /* Actual argument is not a reference: it might be a user error!
	   * Transformers do not carry the may/must information.
	   * A check with effect list ef should be performed...
	   *
	   * FI: does effect computation emit a MUST/MAYwarning?
	   */
	  entity fp_new = external_entity_to_new_value(fp);
	  entity fp_old = external_entity_to_old_value(fp);
	  list args = arguments_add_entity(arguments_add_entity(NIL, fp_new), fp_old);
			
	  pips_user_warning("value (!) might be modified by call to %s\n"
			    "%dth formal parameter %s\n",
			    entity_local_name(f), r, entity_local_name(fp));
	  t_caller = transformer_filter(t_caller, args);
	  free_arguments(args);
	}
      }
      else {
	/* Formal parameter fp is not modified. Add fp == expr, if possible. */
	/* We should evaluate expr under a precondition pre... which has
	   not been passed down. We set pre==tf_undefined. */
	entity fp_new = external_entity_to_new_value(fp);
	transformer t_expr = any_expression_to_transformer(fp_new, expr,
							   transformer_undefined,
							   FALSE);

	if(!transformer_undefined_p(t_expr)) {
	  t_expr = transformer_temporary_value_projection(t_expr);
	  /* temporary value counter cannot be reset because other
             temporary values may be in use in a case the user call is a
             user function call */
	  /* reset_temporary_value_counter(); */
	  t_caller = transformer_safe_image_intersection(t_caller, t_expr);
	  free_transformer(t_expr);
	}
      }
    }
  
    pips_debug(8, "Before formal new values left over are eliminated\n");
    ifdebug(8)   dump_transformer(t_caller);
	

    /* formal new and old values left over are eliminated */
    MAPL(ce,{entity e = ENTITY(CAR(ce));
    entity e_new = external_entity_to_new_value(e); 
    formals_new = CONS(ENTITY, e_new, formals_new);
    /* test to insure that entity_to_old_value exists */
    if(entity_is_argument_p(e_new, 
			    transformer_arguments(t_caller))) {
      entity e_old = external_entity_to_old_value(e);
      formals_new = CONS(ENTITY, e_old, formals_new);
    }},
	 formals);
		 
    t_caller = transformer_filter(t_caller, formals_new);
		 
    free_arguments(formals_new);
    free_arguments(formals);
  }

  ifdebug(8) {
    Psysteme s = predicate_system(transformer_relation(t_caller));
    pips_debug(8,
	       "After binding formal/real parameters and eliminating formals\n");
    dump_transformer(t_caller);
    sc_fprint(stderr, s, (char * (*)(Variable)) dump_value_name);
  }

  /* take care of global variables */
  caller = get_current_module_entity();
  translate_global_values(caller, t_caller);

  /* FI: are invisible variables taken care of by translate_global_values()? 
   * Yes, now...
   * A variable may be invisible because its location is reached
   * thru an array or thru a non-integer scalar variable in the
   * current module, for instance because a COMMON is defined
   * differently. A variable whose location is not reachable
   * in the current module environment is considered visible.
   */

  ifdebug(8) {
    pips_debug(8, "After replacing global variables\n");
    dump_transformer(t_caller);
  }

  if(!transformer_empty_p(t_caller)) {
  /* Callee f may have read/write effects on caller's scalar
   * integer variables thru an array and/or non-integer variables.
   */
  t_effects = effects_to_transformer(ef);
  all_args = arguments_union(transformer_arguments(t_caller),
			     transformer_arguments(t_effects));
  /*
    free_transformer(t_effects);
    gen_free_list(transformer_arguments(t_caller));
  */
  transformer_arguments(t_caller) = all_args;
  /* The relation basis must be updated too */
  MAP(ENTITY, v, {
    Psysteme sc = (Psysteme) predicate_system(transformer_relation(t_caller));
    sc_base_add_variable(sc, (Variable) v);
  }, transformer_arguments(t_effects));
  }
  else {
    pips_user_warning("Call to %s seems never to return."
		      " This may be due to an infinite loop in %s,"
		      " or to a systematic exit in %s,"
		      " or to standard violations (see previous messages)\n",
		      module_local_name(f),
		      module_local_name(f),
		      module_local_name(f));
  }
    
  ifdebug(8) {
    pips_debug(8,
	       "End: after taking all scalar effects in consideration %p\n",
	       t_caller);
    dump_transformer(t_caller);
  }

  /* The return value of a function is not yet projected. */
  pips_assert("transformer t_caller is consistent",
	      transformer_weak_consistency_p(t_caller));

  return t_caller;
}


/* transformer assigned_expression_to_transformer(entity e, expression
 * expr, list ef): returns a transformer abstracting the effect of
 * assignment e = expr when possible, transformer_undefined otherwise.
 *
 * Note: it might be better to distinguish further between e and expr
 * and to return a transformer stating that e is modified when e
 * is accepted for semantics analysis.
 *
 */
transformer 
assigned_expression_to_transformer(
    entity v,
    expression expr,
    transformer pre)
{
  transformer tf = transformer_undefined;

  pips_debug(8, "begin\n");

  if(entity_has_values_p(v)) {
    entity v_new = entity_to_new_value(v);
    entity v_old = entity_to_old_value(v);
    entity tmp = make_local_temporary_value_entity(entity_type(v));
    list tf_args = CONS(ENTITY, v, NIL);

    tf = any_expression_to_transformer(tmp, expr, pre, TRUE);
    reset_temporary_value_counter();
    if(!transformer_undefined_p(tf)) {
      tf = transformer_value_substitute(tf, v_new, v_old);
      tf = transformer_value_substitute(tf, tmp, v_new);
      tf = transformer_temporary_value_projection(tf);
      transformer_arguments(tf) = tf_args;
    }
  }
  else {
    /* vect_rm(ve); */
    tf = transformer_undefined;
  }

  pips_debug(8, "end with tf=%p\n", tf);

  return tf;
}

transformer integer_assign_to_transformer(expression lhs,
					  expression rhs,
					  transformer pre,
					  list ef) /* effects of assign */
{
  /* algorithm: if lhs and rhs are linear expressions on scalar integer
     variables, build the corresponding equation; else, use effects ef
       
     should be extended to cope with constant integer division as in
     N2 = N/2
     because it is used in real program; inequalities should be
     generated in that case 2*N2 <= N <= 2*N2+1
       
     same remark for MOD operator
       
     implementation: part of this function should be moved into
     transformer.c
  */

  transformer tf = transformer_undefined;
  normalized n = NORMALIZE_EXPRESSION(lhs);

  pips_debug(8,"begin\n");

  if(normalized_linear_p(n)) {
    Pvecteur vlhs = (Pvecteur) normalized_linear(n);
    entity e = (entity) vecteur_var(vlhs);

    if(entity_has_values_p(e) /* && integer_scalar_entity_p(e) */) {
      /* FI: the initial version was conservative because
       * only affine scalar integer assignments were processed
       * precisely. But non-affine operators and calls to user defined
       * functions can also bring some information as soon as
       * *some* integer read or write effect exists
       */
      /* check that *all* read effects are on integer scalar entities */
      /*
	if(integer_scalar_read_effects_p(ef)) {
	tf = assigned_expression_to_transformer(e, rhs, ef);
	}
      */
      /* Check that *some* read or write effects are on integer 
       * scalar entities. This is almost always true... Let's hope
       * assigned_expression_to_transformer() returns quickly for array
       * expressions used to initialize a scalar integer entity.
       */
      if(some_integer_scalar_read_or_write_effects_p(ef)) {
	tf = assigned_expression_to_transformer(e, rhs, pre);
      }
    }
  }
  /* if some condition was not met and transformer derivation failed */
  if(tf==transformer_undefined)
    tf = effects_to_transformer(ef);

  pips_debug(6,"return tf=%lx\n", (unsigned long)tf);
  ifdebug(6) (void) print_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

transformer any_scalar_assign_to_transformer(entity v,
					     expression rhs,
					     list ef, /* effects of assign */
					     transformer pre) /* precondition */
{
  transformer tf = transformer_undefined;

  if(entity_has_values_p(v)) {
    entity v_new = entity_to_new_value(v);
    entity v_old = entity_to_old_value(v);
    entity tmp = make_local_temporary_value_entity(entity_type(v));

    tf = any_expression_to_transformer(tmp, rhs, pre, TRUE);

    if(!transformer_undefined_p(tf)) {

      pips_debug(9, "A transformer has been obtained:\n");
      ifdebug(9) dump_transformer(tf);

      if(entity_is_argument_p(v, transformer_arguments(tf))) {
	/* Is it standard compliant? The assigned variable is modified by the rhs. */
	transformer teq = simple_equality_to_transformer(v, tmp, TRUE);
	string s = words_to_string(words_syntax(expression_syntax(rhs)));

	pips_user_warning("Variable %s in lhs is uselessly updated by the rhs '%s'\n",
			  entity_local_name(v), s);

	free(s);

	tf = transformer_combine(tf, teq);
	free_transformer(teq);
      }
      else {
	/* Take care of aliasing */
	entity v_repr = value_to_variable(v_new);

	/* tf = transformer_value_substitute(tf, v_new, v_old); */
	tf = transformer_value_substitute(tf, v_new, v_old);

	pips_debug(9,"After substitution v_new=%s -> v_old=%s\n",
	      entity_local_name(v_new), entity_local_name(v_old));

	tf = transformer_value_substitute(tf, tmp, v_new);

	pips_debug(9,"After substitution tmp=%s -> v_new=%s\n",
	      entity_local_name(tmp), entity_local_name(v_new));

	transformer_add_modified_variable(tf, v_repr);
      }
    }
    if(!transformer_undefined_p(tf)) {
      tf = transformer_temporary_value_projection(tf);
      pips_debug(9, "After temporary value projection, tf=%p:\n", tf);
      ifdebug(9) dump_transformer(tf);
    }
    reset_temporary_value_counter();
  }

  if(transformer_undefined_p(tf))
    tf = effects_to_transformer(ef);

  return tf;
}

transformer any_assign_to_transformer(list args, /* arguments for assign */
				      list ef, /* effects of assign */
				      transformer pre) /* precondition */
{
  transformer tf = transformer_undefined;
  expression lhs = EXPRESSION(CAR(args));
  expression rhs = EXPRESSION(CAR(CDR(args)));
  syntax slhs = expression_syntax(lhs);

  pips_assert("2 args to assign", CDR(CDR(args))==NIL);

  /* The lhs must be a scalar reference to perform an interesting analysis */
  if(syntax_reference_p(slhs)) {
    reference rlhs = syntax_reference(slhs);
    if(ENDP(reference_indices(rlhs))) {
      entity v = reference_variable(rlhs);
      tf = any_scalar_assign_to_transformer(v, rhs, ef, pre); 
    }
  }

  /* if some condition was not met and transformer derivation failed */
  if(tf==transformer_undefined)
    tf = effects_to_transformer(ef);

  pips_debug(6,"return tf=%p\n", tf);
  ifdebug(6) (void) print_transformer(tf);
  pips_debug(8,"end\n");
  return tf;
}

static transformer 
instruction_to_transformer(
			   instruction i,
			   transformer pre,
			   cons * e) /* effects associated to instruction i */
{
  transformer tf = transformer_undefined;
  test t;
  loop l;
  call c;
  whileloop wl;

  debug(8,"instruction_to_transformer","begin\n");

  switch(instruction_tag(i)) {
  case is_instruction_block:
    tf = block_to_transformer(instruction_block(i), pre);
    break;
  case is_instruction_test:
    t = instruction_test(i);
    tf = test_to_transformer(t, pre, e);
    break;
  case is_instruction_loop:
    l = instruction_loop(i);
    tf = loop_to_transformer(l, pre, e);
    break;
  case is_instruction_whileloop:
    wl = instruction_whileloop(i);
    tf = whileloop_to_transformer(wl, pre, e);
    break;
  case is_instruction_goto:
    pips_error("instruction_to_transformer",
	       "unexpected goto in semantics analysis");
    tf = transformer_identity();
    break;
  case is_instruction_call:
    c = instruction_call(i);
    tf = call_to_transformer(c, pre, e);
    break;
  case is_instruction_unstructured:
    tf = unstructured_to_transformer(instruction_unstructured(i), pre, e);
    break ;
  default:
    pips_error("instruction_to_transformer","unexpected tag %d\n",
	       instruction_tag(i));
  }
  pips_debug(9, "resultat:\n");
  ifdebug(9) (void) print_transformer(tf);
  pips_debug(8, "end\n");
  return tf;
}


transformer statement_to_transformer(
				     statement s,
				     transformer spre) /* stmt precondition */
{
  instruction i = statement_instruction(s);
  list e = NIL;
  transformer t = transformer_undefined;
  transformer te = transformer_undefined;
  transformer pre = transformer_undefined;

  pips_debug(8,"begin for statement %03d (%d,%d) with precondition %p:\n",
	     statement_number(s), ORDERING_NUMBER(statement_ordering(s)), 
	     ORDERING_STATEMENT(statement_ordering(s)), spre);
  ifdebug(8) {
    pips_assert("The statement and its substatements are fully defined",
		all_statements_defined_p(s));
    (void) print_transformer(spre);
  }

  pips_assert("spre is a consistent precondition",
	      transformer_consistent_p(spre));

  if(get_bool_property("SEMANTICS_COMPUTE_TRANSFORMERS_IN_CONTEXT")) 
    pre = transformer_undefined_p(spre)? transformer_identity() : 
    transformer_range(spre);
  else
    pre = transformer_undefined;

  pips_assert("pre is a consistent precondition",
	      transformer_consistent_p(pre));

  pips_debug(8,"Range precondition pre:\n");
  ifdebug(8) {
    (void) print_transformer(pre);
  }

  e = load_cumulated_rw_effects_list(s);
  t = load_statement_transformer(s);

  /* it would be nicer to control warning_on_redefinition */
  if (t == transformer_undefined) {
    t = instruction_to_transformer(i, pre, e);

    /* add array references information using proper effects */
    if(get_bool_property("SEMANTICS_TRUST_ARRAY_REFERENCES")) {
      transformer_add_reference_information(t, s);
      /* t = transformer_normalize(t, 0); */
    }
    /* t = transformer_normalize(t, 7); */
    t = transformer_normalize(t, 4);

    if(!transformer_consistency_p(t)) {
      int so = statement_ordering(s);
      (void) fprintf(stderr, "statement %03d (%d,%d):\n",
		     statement_number(s),
		     ORDERING_NUMBER(so), ORDERING_STATEMENT(so));
      /* (void) print_transformer(load_statement_transformer(s)); */
      (void) print_transformer(t);
      dump_transformer(t);
      pips_internal_error("Inconsistent transformer detected\n");
    }
    ifdebug(1) {
      pips_assert("Transformer is internally consistent",
		  transformer_internal_consistency_p(t));
    }

    store_statement_transformer(s, t);
  }
  else {
    user_warning("statement_to_transformer","redefinition for statement %03d (%d,%d)\n",
		 statement_number(s), ORDERING_NUMBER(statement_ordering(s)), 
		 ORDERING_STATEMENT(statement_ordering(s)));
    pips_internal_error("transformer redefinition");
  }

  ifdebug(1) {
    int so = statement_ordering(s);
    transformer stf = load_statement_transformer(s);

    (void) fprintf(stderr, "statement %03d (%d,%d), transformer %p:\n",
		   statement_number(s),
		   ORDERING_NUMBER(so), ORDERING_STATEMENT(so),
		   stf);
    (void) print_transformer(stf);
    pips_assert("same pointer", stf==t);
  }

  /* If i is a loop, the expected transformer can be more complex (see
     nga06) because the stores transformer is later used to compute the
     loop body precondition. It cannot take into account the exit
     condition. */
  if(instruction_loop_p(i)) {
    /* likely memory leak:-(. te should be allocated in both test
       branches and freed at call site but I program everything under
       the opposite assumption */
    /* The refined transformer may be lost or stored as a block
       transformer is the loop is directly surrounded by a bloc or used to
       compute the transformer of the surroundings blokcs */
    te = refine_loop_transformer(t, pre, instruction_loop(i));
  }
  else {
    te = t;
  }

  free_transformer(pre);

  ifdebug(8) {
    pips_assert("The statement and its substatements are still fully defined",
		all_statements_defined_p(s));
  }

  pips_debug(8,"end for statement %03d (%d,%d) with t=%p and te=%p\n",
	     statement_number(s), ORDERING_NUMBER(statement_ordering(s)), 
	     ORDERING_STATEMENT(statement_ordering(s)), t, te);

  return te;
}
