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
  /* algorithm: keep only write effects on integer scalar variable */
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
  /* algorithm: keep only information about integer scalar variables
   * appearing in effects e and store it into a newly allocated transformer
   */
  Pbase b = VECTEUR_NUL;
  Psysteme s = SC_UNDEFINED;
  Psysteme sc = predicate_system(transformer_relation(t));
  list args = NIL;
  
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
 * always used after a call to this function, and itwould create non 
 * reachable structures. Another solution would be to store a copy and free 
 * the unused result in the calling function but transformer_free does not 
 * really free the transformer. Not very clean. 
 * BC, oct. 94 
 */

static transformer 
block_to_transformer(list b)
{
    statement s;
    transformer btf;
    transformer stf = transformer_undefined;
    list l = b;

    pips_debug(8,"begin\n");

    if(ENDP(l))
	btf = transformer_identity();
    else {
	s = STATEMENT(CAR(l));
	stf = statement_to_transformer(s);
	btf = transformer_dup(stf);
	for (POP(l) ; !ENDP(l); POP(l)) {
	    s = STATEMENT(CAR(l));
	    stf = statement_to_transformer(s);
	    btf = transformer_combine(btf, stf);
	    ifdebug(1) 
	      pips_assert("consistent transformer", 
			  transformer_consistency_p(btf));
	}
    }

    pips_debug(8, "end\n");
    return btf;
}

static void 
unstructured_to_transformers(unstructured u)
{
  list blocs = NIL ;
  control ct = unstructured_control(u) ;
  
  pips_debug(8,"begin\n");
  
  /* There is no need to compute transformers for unreachable code,
   * using CONTROL_MAP, but this may create storage and prettyprinter
   * problems because of the data structure inconsistency.
   */
  CONTROL_MAP(c, {
    statement st = control_statement(c) ;
    (void) statement_to_transformer(st) ;
  }, ct, blocs) ;
  
  gen_free_list(blocs) ;
  
  pips_debug(8,"end\n");
}

/* This function is also used when computing preconditions if the exit
   node is not reached. It assumes that transformers for all statements in
   the unstructured have already been computed. */
transformer 
unstructured_to_global_transformer(
    unstructured u)
{
  /* Assume any reachable node is executed at each iteration. A fix-point
     of the result can be used to approximate the node preconditions. Some
     nodes can be discarded because they do not modify the store such as
     IF statements (always) and CONTINUE statements (if they do not link
     the entry and the exit nodes). */
  
  list nodes = NIL;
  /* Entry node */
  control entry_node = unstructured_control(u);
  control exit_node = unstructured_exit(u);
  transformer tf_u = transformer_empty();
  transformer fp_tf_u = transformer_undefined;
  
  pips_debug(8,"begin\n");

  FORWARD_CONTROL_MAP(c, {
    statement st = control_statement(c);
    /* transformer_convex_hull has side effects on its arguments:-( */
    /* Should be fixed now, 29 June 2000 */
    /* transformer tf_st = copy_transformer(load_statement_transformer(st)); */
    transformer tf_st = load_statement_transformer(st);
    transformer tf_old = tf_u;
    
    if(statement_test_p(st)) {
      /* Any side effect? */
      if(!ENDP(transformer_arguments(tf_st))) {
	tf_u = transformer_convex_hull(tf_old, tf_st); /* test */
	free_transformer(tf_old);
      }
    }
    else {
      if(continue_statement_p(st)) {
	if(gen_find_eq(entry_node, control_predecessors(c))!=chunk_undefined
	   && gen_find_eq(exit_node, control_successors(c))!=chunk_undefined) {
	  tf_u = transformer_convex_hull(tf_old, tf_st); /* continue */
	  free_transformer(tf_old);
	}
      }
      else {
	tf_u = transformer_convex_hull(tf_old, tf_st); /* other */
	free_transformer(tf_old);
      }
    }
    
  }, entry_node, nodes) ;
  
  gen_free_list(nodes) ;
  
  /* fp_tf_u = transformer_derivative_fix_point(tf_u); */
  /* Some of the fix-point operators are bugged because they drop part
     of the basis. The problem was not fixed in the fix-point
     computation but in whileloop handling:-(. The derivative version
     should be ok. */
  /* transformer_basic_fix_point() is not defined in fix_point.c:
     dropping all constraints is correct!; Hence,
     transformer_fix_point_operator is not initialized unless
     SEMANTICS_FIX_POINT is set... and this is not the default option.  To
     be redesigned... */
  /* fp_tf_u = (*transformer_fix_point_operator)(tf_u); */
  fp_tf_u = transformer_derivative_fix_point(tf_u);
  
  ifdebug(8) {
    pips_debug(8,"Result for one step tf_u:\n");
    print_transformer(tf_u);
    pips_debug(8,"Result for fix-point fp_tf_u:\n");
    print_transformer(fp_tf_u);
  }
  
  pips_debug(8,"end\n");
  
  return fp_tf_u;
}

static transformer 
unstructured_to_transformer(unstructured u, list e) /* effects */
{
    transformer ctf;
    transformer tf;
    control c;

    pips_debug(8,"begin\n");

    pips_assert("unstructured_to_transformer", u!=unstructured_undefined);

    c = unstructured_control(u);
    if(control_predecessors(c) == NIL && control_successors(c) == NIL) {
	/* there is only one statement in u; no need for a fix-point */
	pips_debug(8,"unique node\n");
	ctf = statement_to_transformer(control_statement(c));
	tf = transformer_dup(ctf);
    }
    else {
      /* Do not try anything clever! God knows what may happen in
	 unstructured code. Transformer tf is not computed recursively
	 from its components but directly derived from effects e.
	 Transformers associated to its components are then computed
	 independently, hence the name unstructured_to_transformerS
	 instead of unstructured_to_transformer */
      statement exit = control_statement(unstructured_exit(u));
      
      pips_debug(8,"complex: based on effects\n");
      
      unstructured_to_transformers(u);
      
      /* if(load_statement_transformer(exit)!=transformer_undefined) { */
	/* The exit node has been reached */
	/* tf = effects_to_transformer(e); */
	/* tf = unstructured_to_global_transformer(u); */
      /* } */
      /* else { */
	/* Never ending loop in unstructured... unless a call to STOP
	   occurs */
	/* tf = transformer_empty(); */
      /* } */
      tf = unstructured_to_accurate_transformer(u);
    }

    pips_debug(8,"end\n");

    return tf;
}

list 
effects_to_arguments(list fx) /* list of effects */
{
    /* algorithm: keep only write effects on integer scalar variable */
    list args = NIL;

    MAP(EFFECT, ef, 
    {
	reference r = effect_reference(ef);
	action a = effect_action(ef);
	entity e = reference_variable(r);
	
	if(action_write_p(a) && entity_integer_scalar_p(e)) {
	    args = arguments_add_entity(args, e);
	}
    },
	fx);

    return args;
}


/* The transformer associated to a DO loop does not include the exit 
 * condition because it is used to compute the precondition for any 
 * loop iteration.
 *
 * There is only one attachment for the unbounded transformer and
 * for the bounded one.
 */

static transformer 
loop_to_transformer(loop l, list e) /* effects of loop l */
{
    /* loop transformer tf = tfb* or tf = tfb+ or ... */
    transformer tf;
    /* loop body transformer */
    transformer tfb;

    entity i = loop_index(l);
    range r = loop_range(l);
    expression incr = range_increment(r);
    Pvecteur v_incr = VECTEUR_UNDEFINED;
    statement s = loop_body(l);

    debug(8,"loop_to_transformer","begin\n");

    if(pips_flag_p(SEMANTICS_FIX_POINT)) {
	/* compute the loop body transformer */
	tfb = transformer_dup(statement_to_transformer(s));
	/* it does not contain the loop index update
	   the loop increment expression must be linear to find inductive 
	   variables related to the loop index */
	if(!VECTEUR_UNDEFINED_P(v_incr = expression_to_affine(incr))) {
	    if(entity_has_values_p(i))
		tfb = transformer_add_loop_index(tfb, i, v_incr);
	    else
		user_warning("loop_to_transformer", 
			     "non-integer loop index %s?\n",
			     entity_local_name(i));
	}

	/* compute tfb's fix point according to pips flags */
	if(pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
	    tf = transformer_halbwachs_fix_point(tfb);
	}
	else {
	    /* FI: it might have been easier to pass tf as an argument to
	     * transformer_equality_fix_point() and to update it with
	     * new equations...
	     */
	    /* transformer ftf = transformer_equality_fix_point(tfb); */
	    transformer ftf = (* transformer_fix_point_operator)(tfb);
	    normalized nlb = NORMALIZE_EXPRESSION(range_lower(r));

	    if(*transformer_fix_point_operator==transformer_equality_fix_point) {
		/* The result must be fixed... because
                   transformer_equality_fix_point() looses some vital
                   information */
		Psysteme fsc = predicate_system(transformer_relation(ftf));
		Psysteme sc = SC_UNDEFINED;
		Pcontrainte eq = CONTRAINTE_UNDEFINED;
		Pbase new_b = BASE_UNDEFINED;
	    
		tf = effects_to_transformer(e);
		sc = (Psysteme) predicate_system(transformer_relation(tf));

		/* compute the basis for tf and ftf */

		/* FI: just in case.
		 * I do not understand why sc_base(fsc) is not enough.
		 * I do not understand why I used effects_to_transformer() instead
		 * of transformer_indentity()...
		 */
		new_b = base_union(sc_base(fsc), sc_base(sc));
		base_rm(sc_base(sc));
		sc_base(sc) = new_b;
		sc_dimension(sc) = base_dimension(new_b);

		/* add equations from ftf to tf */
		for(eq = sc_egalites(fsc); !CONTRAINTE_UNDEFINED_P(eq); ) {
		    Pcontrainte neq;

		    neq = eq->succ;
		    eq->succ = NULL;
		    sc_add_egalite(sc, eq);
		    eq = neq;
		}

		/* add inequalities from ftf to tf */
		for(eq = sc_inegalites(fsc); !CONTRAINTE_UNDEFINED_P(eq); ) {
		    Pcontrainte neq;
		    
		    neq = eq->succ;
		    eq->succ = NULL;
		    sc_add_inegalite(sc, eq);
		    eq = neq;
		}

		/* FI: I hope that inequalities will be taken care of some day! */
		/* Well, in June 1997.. */

		sc_egalites(fsc) = CONTRAINTE_UNDEFINED;
		sc_inegalites(fsc) = CONTRAINTE_UNDEFINED;
		free_transformer(ftf);
	    }
	    else {
		tf = ftf;
	    }

	    ifdebug(8) {
		pips_debug(8, "intermediate fix-point tf=\n");
		fprint_transformer(stderr, tf, external_value_name);
	    }

	    /* add initialization for the loop index variable */
	    /* FI: this seems to be all wrong because a transformer cannot
	     * state anything about its initial state...
	     *
	     * Also, sc basis should be updated!
	     *
	     * I change my mind: let's use the lower bound anyway since it
	     * make sense as soon as i_init is eliminated in the transformer
	     */
	    if(entity_has_values_p(i) && normalized_linear_p(nlb)) {
		Psysteme sc = (Psysteme) predicate_system(transformer_relation(tf));
		Pcontrainte eq = CONTRAINTE_UNDEFINED;
		Pvecteur v_lb = vect_dup(normalized_linear(nlb));
		Pbase b_tmp, b_lb = make_base_from_vect(v_lb); 
		entity i_init = entity_to_old_value(i);

		vect_add_elem(&v_lb, (Variable) i_init, VALUE_MONE);
		eq = contrainte_make(v_lb);
		/* The new variables in eq must be added to sc; otherwise,
		 * further consistency checks core dump. bc.
		 */
		/* sc_add_egalite(sc, eq); */
		/* The call to sc_projection_with_eq frees eq */
		sc = sc_projection_by_eq(sc, eq, (Variable) i_init);
		b_tmp = sc_base(sc);
		sc_base(sc) = base_union(b_tmp, b_lb);
		sc_dimension(sc) = base_dimension(sc_base(sc));
		base_rm(b_tmp);
		base_rm(b_lb);
		if(SC_RN_P(sc)) {
		    /* FI: a NULL is not acceptable; I assume that we cannot
		     * end up with a SC_EMPTY...
		     */
		    predicate_system_(transformer_relation(tf)) =
			newgen_Psysteme
			(sc_make(CONTRAINTE_UNDEFINED, CONTRAINTE_UNDEFINED));
		}
		else
		    predicate_system_(transformer_relation(tf)) = 
			newgen_Psysteme(sc);
	    }

	    ifdebug(8) {
		debug(8, "loop_to_transformer", "full fix-point tf=\n");
		fprint_transformer(stderr, tf, external_value_name);
		debug(8, "loop_to_transformer", "end\n");
	    }

	}
	/* we have a problem here: to compute preconditions within the
	   loop body we need a tf using private variables; to return
	   the loop transformer, we need a filtered out tf; only
	   one hook is available in the ri..; let'a assume there
	   are no private variables and that if they are privatizable
	   they are not going to get in our way */
    }
    else {
	/* basic cheap version: do not use the loop body transformer and
	   avoid fix-points; local variables do not have to be filtered out
	   because this was already done while computing effects */

	(void) statement_to_transformer(s);
	tf = effects_to_transformer(e);
    }

    ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","loop_to_transformer",
		       "resultat tf =");
	(void) (void) print_transformer(tf);
    }
    debug(8,"loop_to_transformer","end\n");
    return tf;
}

/* This function computes the effect of K loop iteration, with K positive.
 * This function does not take the loop exit into account because its result
 * is used to compute the precondition of the loop body.
 * Hence the loop exit condition only is added when preconditions are computed.
 * This is confusing when transformers are prettyprinted with the source code.
 */

static transformer 
whileloop_to_transformer(whileloop l, list e) /* effects of whileloop l */
{
    /* loop transformer tf = tfb* or tf = tfb+ or ... */
    transformer tf;
    /* loop body transformer */
    transformer tfb;
    expression cond = whileloop_condition(l);
    statement s = whileloop_body(l);

    debug(8,"whileloop_to_transformer","begin\n");

    if(pips_flag_p(SEMANTICS_FIX_POINT)) {
	/* compute the whileloop body transformer */
	tfb = transformer_dup(statement_to_transformer(s));

	/* If the while entry condition is usable, it must be added
	 * on the old values
	 */
	tfb = transformer_add_condition_information(tfb, cond, TRUE);

	/* compute tfb's fix point according to pips flags */
	if(pips_flag_p(SEMANTICS_INEQUALITY_INVARIANT)) {
	    tf = transformer_halbwachs_fix_point(tfb);
	}
	else if (transformer_empty_p(tfb)) {
	  /* The loop is never entered */
	  tf = transformer_identity();
	}
	else {
	    transformer ftf = (* transformer_fix_point_operator)(tfb);

	    if(*transformer_fix_point_operator==transformer_equality_fix_point) {
		Psysteme fsc = predicate_system(transformer_relation(ftf));
		Psysteme sc = SC_UNDEFINED;
	    
		/* Dirty looking fix for a fix point computation error:
		 * sometimes, the basis is restricted to a subset of
		 * the integer scalar variables. Should be useless with proper
		 * fixpoint opertors.
		 */
		tf = effects_to_transformer(e);
		sc = (Psysteme) predicate_system(transformer_relation(tf));

		sc = sc_append(sc, fsc);

		free_transformer(ftf);
	    }
	    else {
		tf = ftf;
	    }

	    ifdebug(8) {
		pips_debug(8, "intermediate fix-point tf=\n");
		fprint_transformer(stderr, tf, external_value_name);
	    }

	}
	/* we have a problem here: to compute preconditions within the
	   whileloop body we need a tf using private variables; to return
	   the loop transformer, we need a filtered out tf; only
	   one hook is available in the ri..; let'a assume there
	   are no private variables and that if they are privatizable
	   they are not going to get in our way */
    }
    else {
	/* basic cheap version: do not use the whileloop body transformer and
	   avoid fix-points; local variables do not have to be filtered out
	   because this was already done while computing effects */

	(void) statement_to_transformer(s);
	tf = effects_to_transformer(e);
    }

    ifdebug(8) {
	(void) fprintf(stderr,"%s: %s\n","whileloop_to_transformer",
		       "resultat tf =");
	(void) (void) print_transformer(tf);
    }
    debug(8,"whileloop_to_transformer","end\n");
    return tf;
}

static transformer 
test_to_transformer(test t, list ef) /* effects of t */
{
    statement st = test_true(t);
    statement sf = test_false(t);
    transformer tf;

    debug(8,"test_to_transformer","begin\n");

    /* the test condition cannot be used to improve transformers
       it may be used later when propagating preconditions 
       Francois Irigoin, 15 April 1990 

       Why?
       Francois Irigoin, 31 July 1992

       Well, you can benefit from STOP statements.
       But you do not know if the variable values in
       the condition are the new or the old values...
       Francois Irigoin, 8 November 1995

       */

    if(pips_flag_p(SEMANTICS_FLOW_SENSITIVE)) {
	/*
	tft = statement_to_transformer(st);
	tff = statement_to_transformer(sf);
	tf = transformer_convex_hull(tft, tff);
	*/
	expression e = test_condition(t);
	transformer tftwc;
	transformer tffwc;
	list ta = NIL;
	list fa = NIL;

	tftwc = transformer_dup(statement_to_transformer(st));
	tffwc = transformer_dup(statement_to_transformer(sf));

	/* Look for variables modified in one branch only */
	ta = arguments_difference(transformer_arguments(tftwc),
				  transformer_arguments(tffwc));
	fa = arguments_difference(transformer_arguments(tffwc),
				  transformer_arguments(tftwc));

	MAPL(ca, {
	    entity v = ENTITY(CAR(ca));

	    tffwc = transformer_add_identity(tffwc, v);
	}, ta);

	MAPL(ca, {
	    entity v = ENTITY(CAR(ca));

	    tftwc = transformer_add_identity(tftwc, v);
	}, fa);

	tftwc = transformer_add_condition_information(tftwc, e, TRUE);
	tffwc = transformer_add_condition_information(tffwc, e, FALSE);

	tf = transformer_convex_hull(tftwc, tffwc);
	transformer_free(tftwc);
	transformer_free(tffwc);
	free_arguments(ta);
	free_arguments(fa);
    }
    else {
	(void) statement_to_transformer(st);
	(void) statement_to_transformer(sf);
	tf = effects_to_transformer(ef);
    }

    debug(8,"test_to_transformer","end\n");
    return tf;
}

static transformer assign_to_transformer(list, list);

static transformer 
intrinsic_to_transformer(
    entity e, list pc, list ef) /* effects of intrinsic call */
{
    transformer tf;

    debug(8,"intrinsic_to_transformer","begin\n");

    if(ENTITY_ASSIGN_P(e))
	tf = assign_to_transformer(pc, ef);
    else if(ENTITY_STOP_P(e))
	tf = transformer_empty();
    else
	tf = effects_to_transformer(ef);

    debug(8,"intrinsic_to_transformer","end\n");

    return tf;
}

static transformer user_call_to_transformer(entity, list, list);

static transformer 
call_to_transformer(call c, list ef) /* effects of call c */
{
    transformer tf = transformer_undefined;
    entity e = call_function(c);
    cons *pc = call_arguments(c);
    tag tt;

    pips_debug(8,"begin\n");

    switch (tt = value_tag(entity_initial(e))) {
      case is_value_code:
	/* call to an external function; preliminary version:
	   rely on effects */
	pips_debug(5, "external function %s\n", entity_name(e));
	if(get_bool_property(SEMANTICS_INTERPROCEDURAL))
	    tf = user_call_to_transformer(e, pc, ef);
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
	tf = intrinsic_to_transformer(e, pc, ef);
	break;
      default:
	pips_internal_error("unknown tag %d\n", tt);
    }
    pips_assert("transformer tt is consistent", 
		transformer_consistency_p(tf)); 


    pips_debug(8,"end\n");

    return(tf);
}

/* Effects ef re needed here to use user_call_to_transformer()
 * although the general idea is to return an undefined transformer
 * on failure rather than a transformer derived from effects
 */

static transformer 
user_function_call_to_transformer(
    entity e, 
    expression expr,
    list ef)
{
    syntax s = expression_syntax(expr);
    call c = syntax_call(s);
    entity f = call_function(c);
    list pc = call_arguments(c);
    transformer t_caller = transformer_undefined;
    basic rbt = basic_of_call(c);

    pips_debug(8, "begin\n");
    pips_assert("user_function_call_to_transformer", syntax_call_p(s));

    if(basic_int_p(rbt)) {
	string fn = module_local_name(f);
	entity rv = global_name_to_entity(fn, fn);
	entity orv = entity_undefined;
	entity e_new = entity_to_new_value(e);
	cons * tf_args = CONS(ENTITY, e_new, NIL);
	Psysteme sc = SC_UNDEFINED;
	Pcontrainte c = CONTRAINTE_UNDEFINED;
	Pvecteur eq = VECTEUR_NUL;
	transformer t_assign = transformer_undefined;

	pips_assert("user_function_call_to_transformer",
		    !entity_undefined_p(rv));

	/* Build a transformer reflecting the call site */
	t_caller = user_call_to_transformer(f, pc, ef);

	ifdebug(8) {
	    pips_debug(8, "Transformer %p for callee %s:\n",
		       t_caller, entity_local_name(f));
	    dump_transformer(t_caller);
	}

	/* Build a transformer representing the assignment of
	 * the function value to e
	 */
	eq = vect_make(eq,
		       (Variable) entity_to_new_value(e), VALUE_ONE,
		       (Variable) rv, VALUE_MONE,
		       TCST, VALUE_ZERO);
	c = contrainte_make(eq);
	sc = sc_make(c, CONTRAINTE_UNDEFINED);
	/* FI: I do not understand why this is useful since the basis
	 * does not have to have old values that do not appear in
	 * predicates... See transformer_consistency_p()
	 *
	 * But it proved useful for the call to foo3 in funcside.f
	 */
	sc_base_add_variable(sc, (Variable) entity_to_old_value(e));
	t_assign = make_transformer(tf_args,
				    make_predicate(sc));

	/* Consistency cannot be checked on a non-local transformer */
	/* pips_assert("user_function_call_to_transformer",
	   transformer_consistency_p(t_assign)); */

	ifdebug(8) {
	    debug(8, "user_function_call_to_transformer", 
		  "Transformer %p for assignment of %s with %s:\n",
		  t_assign, entity_local_name(e), entity_name (rv));
	    dump_transformer(t_assign);
	}

	/* Combine the effect of the function call and of the assignment */
	t_caller = transformer_combine(t_caller, t_assign);
	free_transformer(t_assign);

	/* Get rid of the temporary representing the function's value */
	orv = global_new_value_to_global_old_value(rv);
	t_caller = transformer_filter(t_caller,
				      CONS(ENTITY, rv, CONS(ENTITY, orv, NIL)));


	ifdebug(8) {
	    debug(8, "user_function_call_to_transformer", 
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
	pips_assert("transformer t_caller is consistent", 
		    transformer_consistency_p(t_caller));
    }
    else {
	pips_assert("transformer t_caller is undefined", 
		    transformer_undefined_p(t_caller));
    }

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
    cons * old_args = transformer_arguments(ftf);
    Psysteme sc = SC_UNDEFINED;
    Pbase b = BASE_UNDEFINED;
    Pbase eb = BASE_UNDEFINED;

    debug(8,"transformer_intra_to_inter","begin\n");
    debug(8,"transformer_intra_to_inter","argument tf=%x\n",ftf);
    ifdebug(8) (void) dump_transformer(ftf);

    /* get rid of tf's arguments that do not appear in effects le */

    /* build a list of arguments to suppress */
    /* FI: I do not understand anymore why corresponding old values do not have
     * to be suppressed too (6 July 1993)
     *
     * FI: because only read arguments are eliminated, non? (12 November 1995)
     */
    MAPL(ca, 
     {entity e = ENTITY(CAR(ca));
      if(!effects_write_entity_p(le, e) &&
	       !storage_return_p(entity_storage(e))) 
	  lost_args = arguments_add_entity(lost_args, e);
     },
    old_args);

    /* get rid of them */
    ftf = transformer_projection(ftf, lost_args);

    /* free the temporary list of entities */
    gen_free_list(lost_args);
    lost_args = NIL;

    debug(8,"transformer_intra_to_inter","after first filtering ftf=%x\n",ftf);
    ifdebug(8) (void) dump_transformer(ftf);

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
	       !storage_return_p(entity_storage(v))) {
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
	user_warning("user_call_to_transformer",
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
	list formals = entity_to_formal_integer_parameters(f);
	list formals_new = NIL;
	cons * ce;

	t_callee = load_summary_transformer(f);

	ifdebug(8) {
	    Psysteme s = 
		(Psysteme) predicate_system(transformer_relation(t_callee));
	    pips_debug(8, "Transformer for callee %s:\n", 
		       entity_local_name(f));
	    dump_transformer(t_callee);
	    sc_fprint(stderr, s, dump_value_name);
	}

	t_caller = transformer_dup(t_callee);

	/* take care of formal parameters */
	/* let's start a long, long, long MAPL, so long that MAPL is a pain */
	for( ce = formals; !ENDP(ce); POP(ce)) {
	    entity e = ENTITY(CAR(ce));
	    int r = formal_offset(storage_formal(entity_storage(e)));
	    expression expr;
	    normalized n;

	    if((expr = find_ith_argument(pc, r)) == expression_undefined)
		pips_user_error("not enough args for %d formal parm."
				" %s in call to %s from %s\n",
				r, entity_local_name(e), entity_local_name(f),
				get_current_module_entity());

	    n = NORMALIZE_EXPRESSION(expr);
	    if(normalized_linear_p(n)) {
		Pvecteur v = vect_dup((Pvecteur) normalized_linear(n));
		if(value_mappings_compatible_vector_p(v)) {
		    entity e_new = external_entity_to_new_value(e);
		    entity a_new = entity_undefined;
		    entity a_old = entity_undefined;
		    
		    if(entity_is_argument_p(e_new, 
					    transformer_arguments(t_caller))) {
		      /* e_new and e_old must be replaced by the
			 actual entity argument */
		      entity e_old = external_entity_to_old_value(e);
 ifdebug(8) {
	debug(8, "user_call_to_transformer", 
	      "entity=%s entity_old_value= %s\n",entity_local_name(e),entity_local_name(e_old));
	
	
    }
		      if(vect_size(v) != 1 || vecteur_var(v) == TCST) {
			/* Actual argument is not a reference: it might be a user error!
			 * Transformers do not carry the may/must information.
			 * A check with effect list ef should be performed...
			 *
			 * FI: does effect computation emit a warning?
			 */
			list args = arguments_add_entity(arguments_add_entity(NIL, e_new), e_old);
			
			user_warning("user_call_to_transformer",
				     "value (!) might be modified by call to %s\n%dth formal parameter %s\n",
				     entity_local_name(f), r, entity_local_name(e));
			t_caller = transformer_filter(t_caller, args);
			free_arguments(args);
			
		      }
		      else {
			  Psysteme s = (Psysteme) predicate_system(transformer_relation(t_caller));
			  a_new = entity_to_new_value((entity) vecteur_var(v));
			  a_old = entity_to_old_value((entity) vecteur_var(v));

			  if(base_contains_variable_p(s->base, (Variable) a_new)) {
			      user_error("user_call_to_transformer",
					 "Variable %s seems to be aliased thru variable %s"
					 " at a call site to %s in %s\n"
					 "PIPS semantics analysis assumes no aliasing as"
					 " imposed by the Fortran standard.\n",
					 entity_name(e),
					 entity_name((entity) vecteur_var(v)),
					 module_local_name(f),
					 get_current_module_name());
			  }
			  else {
			      t_caller = transformer_value_substitute
				  (t_caller, e_new, a_new);
			      t_caller = transformer_value_substitute
				  (t_caller, e_old, a_old);
			  }
			
		      }
		    }
		    else { 
			/* simple case: formal parameter e is not
			   modified and can be replaced by actual
			   argument expression */
			vect_add_elem(&v, (Variable) e_new, VALUE_MONE);
			t_caller = transformer_equality_add(t_caller,
							    v);
		    }
		}
		else { 
		    /* formal parameter e has to be eliminated:
		       e_new and e_old will be eliminated */
		    vect_rm(v);
		}
	    }
	}

   ifdebug(8) {
	debug(8, "user_call_to_transformer", 
	      "Before formal new values left over are eliminated\n");
	dump_transformer(t_caller);
	
    }

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
	debug(8, "user_call_to_transformer", 
	      "After binding formal/real parameters\n");
	dump_transformer(t_caller);
	sc_fprint(stderr, s, dump_value_name);
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
	debug(8, "user_call_to_transformer", 
	      "After replacing global variables\n");
	dump_transformer(t_caller);
    }

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
    

    ifdebug(8) {
	debug(8, "user_call_to_transformer", 
	      "End: after taking non-integer scalar effects %p\n",
	      t_caller);
	dump_transformer(t_caller);
    }

    /* pips_assert("transformer t_caller is consistent", transformer_consistency_p(t_caller)); 
     */


    return t_caller;
}

transformer
transformer_add_identity(transformer tf, entity v)
{
    entity v_new = entity_to_new_value(v);
    entity v_old = entity_to_old_value(v);
    Pvecteur eq = vect_new((Variable) v_new, (Value) 1);

    vect_add_elem(&eq, (Variable) v_old, (Value) -1);
    tf = transformer_equality_add(tf, eq);
    transformer_arguments(tf) = 
	arguments_add_entity(transformer_arguments(tf), v_new);

    return tf;
}

static transformer 
affine_to_transformer(entity e, Pvecteur a, bool assignment)
{
    transformer tf = transformer_undefined;
    Pvecteur ve = vect_new((Variable) e, VALUE_ONE);
    entity e_new = entity_to_new_value(e);
    entity e_old = entity_to_old_value(e);
    cons * tf_args = CONS(ENTITY, e_new, NIL);
    /* must be duplicated right now  because it will be
       renamed and checked at the same time by
       value_mappings_compatible_vector_p() */
    Pvecteur vexpr = vect_dup(a);
    Pcontrainte c;
    Pvecteur eq = VECTEUR_NUL;

    debug(8, "affine_to_transformer", "begin\n");

    ifdebug(9) {
	pips_debug(9, "\nLinearized expression:\n");
	vect_dump(vexpr);
    }

    if(!assignment) {
	vect_add_elem(&vexpr, (Variable) e, (Value) 1);

	ifdebug(8) {
	    pips_debug(8, "\nLinearized expression for incrementation:\n");
	    vect_dump(vexpr);
	}
    }

    if(value_mappings_compatible_vector_p(ve) &&
       value_mappings_compatible_vector_p(vexpr)) {
	ve = vect_variable_rename(ve,
				  (Variable) e,
				  (Variable) e_new);
	(void) vect_variable_rename(vexpr,
				    (Variable) e_new,
				    (Variable) e_old);
	eq = vect_substract(ve, vexpr);
	vect_rm(ve);
	vect_rm(vexpr);
	c = contrainte_make(eq);
	tf = make_transformer(tf_args,
		      make_predicate(sc_make(c, CONTRAINTE_UNDEFINED)));
    }
    else {
	vect_rm(eq);
	vect_rm(ve);
	vect_rm(vexpr);
	tf = transformer_undefined;
    }

    debug(8, "affine_to_transformer", "end\n");

    return tf;
}

static transformer 
affine_assignment_to_transformer(entity e, Pvecteur a)
{
    transformer tf = transformer_undefined;

    tf = affine_to_transformer(e, a, TRUE);

    return tf;
}

transformer 
affine_increment_to_transformer(entity e, Pvecteur a)
{
    transformer tf = transformer_undefined;

    tf = affine_to_transformer(e, a, FALSE);

    return tf;
}

static transformer 
modulo_to_transformer(e, expr)
entity e;
expression expr;
{
    transformer tf = transformer_undefined;
    expression arg2 = expression_undefined;
    call c = syntax_call(expression_syntax(expr));

    debug(8, "modulo_to_transformer", "begin\n");
    
    arg2 = find_ith_argument(call_arguments(c), 2);

    if(integer_constant_expression_p(arg2)) {
	int d = integer_constant_expression_value(arg2);
	entity e_new = entity_to_new_value(e);
	Pvecteur ub = vect_new((Variable) e_new, VALUE_ONE);
	Pvecteur lb = vect_new((Variable) e_new, VALUE_MONE);
	Pcontrainte clb = contrainte_make(lb);
	Pcontrainte cub = CONTRAINTE_UNDEFINED;
	cons * tf_args = CONS(ENTITY, e_new, NIL);

	vect_add_elem(&ub, TCST, int_to_value(1-d));
	vect_add_elem(&lb, TCST, int_to_value(d-1));
	cub = contrainte_make(ub);
	clb->succ = cub;
	tf = make_transformer(tf_args,
		make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }

    ifdebug(8) {
	debug(8, "modulo_to_transformer", "result:\n");
	print_transformer(tf);
	debug(8, "modulo_to_transformer", "end\n");
    }

   return tf;
}

static transformer 
iabs_to_transformer(e, expr)
entity e;
expression expr;
{
    transformer tf = transformer_undefined;
    call c = syntax_call(expression_syntax(expr));
    expression arg = EXPRESSION(CAR(call_arguments(c)));
    normalized n = NORMALIZE_EXPRESSION(arg);

    debug(8, "iabs_to_transformer", "begin\n");

    if(normalized_linear_p(n)) {
	entity e_new = entity_to_new_value(e);
	entity e_old = entity_to_old_value(e);
	Pvecteur vlb1 = vect_dup((Pvecteur) normalized_linear(n));
	Pvecteur vlb2 = vect_multiply(vect_dup((Pvecteur) normalized_linear(n)), VALUE_MONE);
	Pcontrainte clb1 = CONTRAINTE_UNDEFINED;
	Pcontrainte clb2 = CONTRAINTE_UNDEFINED;
	cons * tf_args = CONS(ENTITY, e_new, NIL);

	(void) vect_variable_rename(vlb1,
				    (Variable) e_new,
				    (Variable) e_old);

	(void) vect_variable_rename(vlb2,
				    (Variable) e_new,
				    (Variable) e_old);

	vect_add_elem(&vlb1, (Variable) e_new, VALUE_MONE);
	vect_add_elem(&vlb2, (Variable) e_new, VALUE_MONE);
	clb1 = contrainte_make(vlb1);
	clb2 = contrainte_make(vlb2);
	clb1->succ = clb2;
	tf = make_transformer(tf_args,
		make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb1)));
    }

    ifdebug(8) {
	debug(8, "iabs_to_transformer", "result:\n");
	print_transformer(tf);
	debug(8, "iabs_to_transformer", "end\n");
    }

   return tf;
}

static transformer 
integer_divide_to_transformer(e, expr)
entity e;
expression expr;
{
    transformer tf = transformer_undefined;
    call c = syntax_call(expression_syntax(expr));
    expression arg1 = expression_undefined;
    normalized n1 = normalized_undefined;
    expression arg2 = expression_undefined;

    debug(8, "integer_divide_to_transformer", "begin\n");
    
    arg1 = find_ith_argument(call_arguments(c), 1);
    n1 = NORMALIZE_EXPRESSION(arg1);
    arg2 = find_ith_argument(call_arguments(c), 2);

    if(integer_constant_expression_p(arg2) && normalized_linear_p(n1)) {
	int d = integer_constant_expression_value(arg2);
	entity e_new = entity_to_new_value(e);
	entity e_old = entity_to_old_value(e);
	cons * tf_args = CONS(ENTITY, e, NIL);
	/* must be duplicated right now  because it will be
	   renamed and checked at the same time by
	   value_mappings_compatible_vector_p() */
	Pvecteur vlb =
	    vect_multiply(vect_dup(normalized_linear(n1)), VALUE_MONE); 
	Pvecteur vub = vect_dup(normalized_linear(n1));
	Pcontrainte clb = CONTRAINTE_UNDEFINED;
	Pcontrainte cub = CONTRAINTE_UNDEFINED;

	(void) vect_variable_rename(vlb,
				    (Variable) e_new,
				    (Variable) e_old);
	(void) vect_variable_rename(vub,
				    (Variable) e_new,
				    (Variable) e_old);

	vect_add_elem(&vlb, (Variable) e_new, int_to_value(d));
	vect_add_elem(&vub, (Variable) e_new, int_to_value(-d));
	vect_add_elem(&vub, TCST, int_to_value(1-d));
	clb = contrainte_make(vlb);
	cub = contrainte_make(vub);
	clb->succ = cub;
	tf = make_transformer(tf_args,
	       make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }

    ifdebug(8) {
	debug(8, "integer_divide_to_transformer", "result:\n");
	print_transformer(tf);
	debug(8, "integer_divide_to_transformer", "end\n");
    }

    return tf;
}

static transformer 
integer_power_to_transformer(e, expr)
entity e;
expression expr;
{
  transformer tf = transformer_undefined;
  call c = syntax_call(expression_syntax(expr));
  expression arg1 = expression_undefined;
  normalized n1 = normalized_undefined;
  expression arg2 = expression_undefined;
  normalized n2 = normalized_undefined;

  debug(8, "integer_power_to_transformer", "begin\n");
    
  arg1 = find_ith_argument(call_arguments(c), 1);
  n1 = NORMALIZE_EXPRESSION(arg1);
  arg2 = find_ith_argument(call_arguments(c), 2);
  n2 = NORMALIZE_EXPRESSION(arg2);

  if(signed_integer_constant_expression_p(arg2) && normalized_linear_p(n1)) {
    int d = signed_integer_constant_expression_value(arg2);

    if(d%2==0) {
      entity e_new = entity_to_new_value(e);
      entity e_old = entity_to_old_value(e);
      cons * tf_args = CONS(ENTITY, e, NIL);

      if(d==0) {
	/* 1 is assigned unless arg1 equals 0... which is neglected */
	Pvecteur v = vect_new((Variable) e_new, VALUE_ONE);

	vect_add_elem(&v, TCST, VALUE_MONE);
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(contrainte_make(v),
						     CONTRAINTE_UNDEFINED)));
      }
      else if(d>0) {
	/* Does not work because unary minus is not seen as part of a constant */
	/* The expression value must be greater or equal to arg2 and positive */
	/* must be duplicated right now  because it will be
	   renamed and checked at the same time by
	   value_mappings_compatible_vector_p() */
	Pvecteur vlb1 = vect_dup(normalized_linear(n1));
	Pvecteur vlb2 = vect_multiply(vect_dup(normalized_linear(n1)), VALUE_MONE);
	Pcontrainte clb1 = CONTRAINTE_UNDEFINED;
	Pcontrainte clb2 = CONTRAINTE_UNDEFINED;

	(void) vect_variable_rename(vlb1,
				    (Variable) e_new,
				    (Variable) e_old);

	vect_add_elem(&vlb1, (Variable) e_new, VALUE_MONE);
	vect_add_elem(&vlb2, (Variable) e_new, VALUE_MONE);
	clb1 = contrainte_make(vlb1);
	clb2 = contrainte_make(vlb2);
	clb1->succ = clb2;
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb1)));
      }
      else {
	/* d is negative and even */
	entity e_new = entity_to_new_value(e);
	cons * tf_args = CONS(ENTITY, e, NIL);
	Pvecteur vub = vect_new((Variable) e_new, VALUE_ONE);
	Pvecteur vlb = vect_new((Variable) e_new, VALUE_MONE);
	Pcontrainte clb = CONTRAINTE_UNDEFINED;
	Pcontrainte cub = CONTRAINTE_UNDEFINED;

	vect_add_elem(&vub, TCST, VALUE_MONE);
	clb = contrainte_make(vlb);
	cub = contrainte_make(vub);
	clb->succ = cub;
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
      }
    }
    else if(d<0) {
      /* d is negative, arg1 cannot be 0, expression value is -1, 0
	 or 1 */
      entity e_new = entity_to_new_value(e);
      cons * tf_args = CONS(ENTITY, e, NIL);
      Pvecteur vub = vect_new((Variable) e_new, VALUE_MONE);
      Pvecteur vlb = vect_new((Variable) e_new, VALUE_ONE);
      Pcontrainte clb = CONTRAINTE_UNDEFINED;
      Pcontrainte cub = CONTRAINTE_UNDEFINED;

      vect_add_elem(&vub, TCST, VALUE_MONE);
      vect_add_elem(&vlb, TCST, VALUE_MONE);
      clb = contrainte_make(vlb);
      cub = contrainte_make(vub);
      clb->succ = cub;
      tf = make_transformer(tf_args,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }
    else if(d==1) {
	entity e_new = entity_to_new_value(e);
	cons * tf_args = CONS(ENTITY, e, NIL);
	Pvecteur v = vect_dup(normalized_linear(n1));

	vect_add_elem(&v, (Variable) e_new, VALUE_MONE);
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(contrainte_make(v),
						     CONTRAINTE_UNDEFINED)));
    }
  }
  else if(signed_integer_constant_expression_p(arg1)) {
    int d = signed_integer_constant_expression_value(arg1);
    entity e_new = entity_to_new_value(e);

    if(d==0||d==1) {
      /* 0 or 1 is assigned unless arg2 equals 0... which is neglected */
      cons * tf_args = CONS(ENTITY, e, NIL);
      Pvecteur v = vect_new((Variable) e_new, VALUE_ONE);

      vect_add_elem(&v, TCST, int_to_value(-d));
      tf = make_transformer(tf_args,
			    make_predicate(sc_make(contrainte_make(v),
						   CONTRAINTE_UNDEFINED)));
    }
    else if(d > 1) {
      /* the assigned value is positive */
      cons * tf_args = CONS(ENTITY, e, NIL);
      Pvecteur v1 = vect_new((Variable) e_new, VALUE_MONE);
      Pcontrainte c1 = contrainte_make(v1);

      if(normalized_linear_p(n2)) {
	Pvecteur v2 = vect_dup(normalized_linear(n2));
	Pcontrainte c2 = CONTRAINTE_UNDEFINED;

	vect_add_elem(&v2, TCST, VALUE_ONE);
	vect_add_elem(&v2, (Variable) e_new, VALUE_MONE);
	c2 = contrainte_make(v2);
	contrainte_succ(c1) = c2;
      }

      tf = make_transformer(tf_args,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, c1)));
    }
    else if(d == -1) {
      /* The assigned value is 1 or -1 */
      entity e_new = entity_to_new_value(e);
      cons * tf_args = CONS(ENTITY, e, NIL);
      Pvecteur vub = vect_new((Variable) e_new, VALUE_MONE);
      Pvecteur vlb = vect_new((Variable) e_new, VALUE_ONE);
      Pcontrainte clb = CONTRAINTE_UNDEFINED;
      Pcontrainte cub = CONTRAINTE_UNDEFINED;

      vect_add_elem(&vub, TCST, VALUE_MONE);
      vect_add_elem(&vlb, TCST, VALUE_MONE);
      clb = contrainte_make(vlb);
      cub = contrainte_make(vub);
      clb->succ = cub;
      tf = make_transformer(tf_args,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED, clb)));
    }
  }

  ifdebug(8) {
    debug(8, "integer_power_to_transformer", "result:\n");
    print_transformer(tf);
    debug(8, "integer_power_to_transformer", "end\n");
  }

  return tf;
}

static transformer 
minmax_to_transformer(e, expr, minmax)
entity e;
expression expr;
bool minmax;
{
    transformer tf = transformer_undefined;
    call c = syntax_call(expression_syntax(expr));
    expression arg = expression_undefined;
    normalized n = normalized_undefined;
    list cexpr;
    cons * tf_args = CONS(ENTITY, e, NIL);
    Pcontrainte cl = CONTRAINTE_UNDEFINED;

    debug(8, "minmax_to_transformer", "begin\n");

    for(cexpr = call_arguments(c); !ENDP(cexpr); POP(cexpr)) {
	arg = EXPRESSION(CAR(cexpr));
	n = NORMALIZE_EXPRESSION(arg);

	if(normalized_linear_p(n)) {
	    Pvecteur v = vect_dup((Pvecteur) normalized_linear(n));
	    Pcontrainte cv = CONTRAINTE_UNDEFINED;
	    entity e_new = entity_to_new_value(e);
	    entity e_old = entity_to_old_value(e);

	    (void) vect_variable_rename(v,
					(Variable) e,
					(Variable) e_old);
	    vect_add_elem(&v, (Variable) e_new, VALUE_MONE);

	    if(minmax) {
		v = vect_multiply(v, VALUE_MONE);
	    }

	    cv = contrainte_make(v);
	    cv->succ = cl;
	    cl = cv;

	}
    }

    if(CONTRAINTE_UNDEFINED_P(cl) || CONTRAINTE_NULLE_P(cl)) {
	Psysteme sc = sc_make(CONTRAINTE_UNDEFINED, cl);
	entity oldv = entity_to_old_value(e);
	entity newv = entity_to_new_value(e);

	sc_base(sc) = base_add_variable(base_add_variable(BASE_NULLE,
							  (Variable) oldv),
					(Variable) newv);
	sc_dimension(sc) = 2;
	tf = make_transformer(tf_args,
			      make_predicate(sc));
    }
    else {
	/* A miracle occurs and the proper basis is derived from the
	   constraints ( I do not understand why the new and the old value
	   of e both appear... so it may not be necessary for the
	   consistency check... I'm lost, FI, 6 Jan. 1999) */
	tf = make_transformer(tf_args,
			      make_predicate(sc_make(CONTRAINTE_UNDEFINED, cl)));
    }


    ifdebug(8) {
	debug(8, "minmax_to_transformer", "result:\n");
	print_transformer(tf);
	debug(8, "minmax_to_transformer", "end\n");
    }

    return tf;
}

static transformer 
min0_to_transformer(e, expr)
entity e;
expression expr;
{
    return minmax_to_transformer(e, expr, TRUE);
}

static transformer 
max0_to_transformer(e, expr)
entity e;
expression expr;
{
    return minmax_to_transformer(e, expr, FALSE);
}

/* transformer expression_to_transformer(entity e, expression expr, list ef):
 * returns a transformer abstracting the effect of assignment e = expr
 * if entity e and entities referenced in expr are accepted for
 * semantics analysis anf if expr is affine; else returns
 * transformer_undefined
 *
 * Note: it might be better to distinguish further between e and expr
 * and to return a transformer stating that e is modified when e
 * is accepted for semantics analysis.
 *
 * Bugs:
 *  - core dumps if entities referenced in expr are not accepted for
 *    semantics analysis
 *
 * Modifications:
 *  - MOD and / added for very special cases (but not as general as it should be)
 *    FI, 25/05/93
 *  - MIN, MAX and use function call added (for simple cases)
 *    FI, 16/11/95
 */
transformer 
expression_to_transformer(
    entity e,
    expression expr,
    list ef)
{
    transformer tf = transformer_undefined;

    pips_debug(8, "begin\n");

    if(entity_has_values_p(e)) {
        /* Pvecteur ve = vect_new((Variable) e, VALUE_ONE); */
	normalized n = NORMALIZE_EXPRESSION(expr);

	if(normalized_linear_p(n)) {
	    tf = affine_assignment_to_transformer(e,
		       (Pvecteur) normalized_linear(n));
	}
	else if(modulo_expression_p(expr)) {
	    tf = modulo_to_transformer(e, expr);
	}
	else if(divide_expression_p(expr)) {
	    tf = integer_divide_to_transformer(e, expr);
	}
	else if(power_expression_p(expr)) {
	    tf = integer_power_to_transformer(e, expr);
	}
	else if(iabs_expression_p(expr)) {
	    tf = iabs_to_transformer(e, expr);
	}
	else if(min0_expression_p(expr)) {
	    tf = min0_to_transformer(e, expr);
	}
	else if(max0_expression_p(expr)) {
	    tf = max0_to_transformer(e, expr);
	}
	else if(user_function_call_p(expr) 
		&& get_bool_property(SEMANTICS_INTERPROCEDURAL)) {
	    tf = user_function_call_to_transformer(e, expr, ef);
	}
	else {
	    /* vect_rm(ve); */
	    tf = transformer_undefined;
	}
    }

    pips_debug(8, "end with tf=%p\n", tf);

    return tf;
}

static transformer 
assign_to_transformer(list args, /* arguments for assign */
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

    expression lhs = EXPRESSION(CAR(args));
    expression rhs = EXPRESSION(CAR(CDR(args)));
    transformer tf = transformer_undefined;
    normalized n = NORMALIZE_EXPRESSION(lhs);

    pips_debug(8,"begin\n");
    pips_assert("2 args to assign", CDR(CDR(args))==NIL);

    if(normalized_linear_p(n)) {
	Pvecteur vlhs = (Pvecteur) normalized_linear(n);
	entity e = (entity) vecteur_var(vlhs);

	if(entity_has_values_p(e) && integer_scalar_entity_p(e)) {
	    /* FI: the initial version was conservative because
	     * only affine scalar integer assignments were processed
	     * precisely. But non-affine operators and calls to user defined
	     * functions can also bring some information as soon as
	     * *some* integer read or write effect exists
	     */
	    /* check that *all* read effects are on integer scalar entities */
	    /*
	    if(integer_scalar_read_effects_p(ef)) {
		tf = expression_to_transformer(e, rhs, ef);
	    }
	    */
	    /* Check that *some* read or write effects are on integer 
	     * scalar entities. This is almost always true... Let's hope
	     * expression_to_transformer() returns quickly for array
	     * expressions used to initialize a scalar integer entity.
	     */
	    if(some_integer_scalar_read_or_write_effects_p(ef)) {
		tf = expression_to_transformer(e, rhs, ef);
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

static transformer 
instruction_to_transformer(i, e)
instruction i;
cons * e; /* effects associated to instruction i */
{
    transformer tf = transformer_undefined;
    test t;
    loop l;
    call c;
    whileloop wl;

    debug(8,"instruction_to_transformer","begin\n");

    switch(instruction_tag(i)) {
      case is_instruction_block:
	tf = block_to_transformer(instruction_block(i));
	break;
      case is_instruction_test:
	t = instruction_test(i);
	tf = test_to_transformer(t, e);
	break;
      case is_instruction_loop:
	l = instruction_loop(i);
	tf = loop_to_transformer(l, e);
	break;
      case is_instruction_whileloop:
	wl = instruction_whileloop(i);
	tf = whileloop_to_transformer(wl, e);
	break;
      case is_instruction_goto:
	pips_error("instruction_to_transformer",
		   "unexpected goto in semantics analysis");
	tf = transformer_identity();
	break;
      case is_instruction_call:
	c = instruction_call(i);
	tf = call_to_transformer(c, e);
	break;
      case is_instruction_unstructured:
	tf = unstructured_to_transformer(instruction_unstructured(i), e);
	  break ;
      default:
	pips_error("instruction_to_transformer","unexpected tag %d\n",
	      instruction_tag(i));
    }
    debug(9,"instruction_to_transformer","resultat:\n");
    ifdebug(9) (void) print_transformer(tf);
    debug(8,"instruction_to_transformer","end\n");
    return tf;
}


transformer statement_to_transformer(s)
statement s;
{
    instruction i = statement_instruction(s);
    list e = NIL;
    transformer t;

    pips_debug(8,"begin for statement %03d (%d,%d)\n",
	       statement_number(s), ORDERING_NUMBER(statement_ordering(s)), 
	       ORDERING_STATEMENT(statement_ordering(s)));

    e = load_cumulated_rw_effects_list(s);
    t = load_statement_transformer(s);

    /* it would be nicer to control warning_on_redefinition */
    if (t == transformer_undefined) {
	t = instruction_to_transformer(i, e);

	/* add array references information */
	if(get_bool_property("SEMANTICS_TRUST_ARRAY_REFERENCES")) {
	    transformer_add_reference_information(t, s);
	}

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

    pips_debug(8,"end with t=%p\n", t);

    return(t);
}
