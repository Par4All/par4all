 /* Predicate transformer package: sc complexity level */

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
/* #include "semantics.h" */

#include "transformer.h"

/* transformer transformer_combine(transformer t1, transformer t2):
 * compute the composition of transformers t1 and t2 (t1 then t2)
 *
 * t1 := t2 o t1
 * return t1
 *
 * t1 is updated, but t2 is preserved
 */
transformer 
transformer_combine(
    transformer t1,
    transformer t2)
{
    /* algorithm: 
       let a1 be t1 arguments, a2 be t2 arguments,
       let ints be the intersection of a1 and a2
       let r1 be t1 relation and r2 be a copy of t2 relation
       let a be a1 union a2
       rename entities in ints in r1 (new->int) and r2 (old->int)
       rename entities in a2-ints in r1 (new->old)
       build a system b with r1 and r2
       project b along ints
       build t1 with a and b
       */
    cons * a1 = transformer_arguments(t1);
    cons * a2 = transformer_arguments(t2);
    /* Newgen does not generate the proper castings */
    Psysteme r1 = (Psysteme) predicate_system(transformer_relation(t1));
    Psysteme r2 = sc_dup((Psysteme)predicate_system(transformer_relation(t2)));
    /* ints: list of intermediate value entities */
    cons * ints = NIL;
    /* local variable ce2: why don't you use MAPL :-)! */
    cons * ce2;

    debug(8,"transformer_combine","begin\n");

    debug(8,"transformer_combine","arg. t1=%x\n",t1);
    ifdebug(8) (void) dump_transformer(t1);
    /* The consistencies of transformers t1 and t2 cannot be checked with
       respect to the current environment because t1 or t2 may be relative
       to a callee as in user_function_call_to_transformer(). Hence a
       debug level of 10. */
    ifdebug(10) pips_assert("consistent t1", transformer_consistency_p(t1));

    debug(8,"transformer_combine","arg. t2=%x\n",t2);
    ifdebug(8) (void) dump_transformer(t2);
    ifdebug(10) pips_assert("consistent t2", transformer_consistency_p(t2));

    /* build new argument list and rename old and intermediate values,
       as well as new (i.e. unmodified) variables in t1 */

    for(ce2 = a2; !ENDP(ce2); POP(ce2)) {
	entity e2 = ENTITY(CAR(ce2));
	if(entity_is_argument_p(e2, a1)) {
	    /* renaming of intermediate values in r1 and r2 */
	    entity e_int = entity_to_intermediate_value(e2);
	    entity e_old = entity_to_old_value(e2);
	    r1 = sc_variable_rename(r1, (Variable) e2, (Variable) e_int);
	    r2 = sc_variable_rename(r2, (Variable) e_old, (Variable) e_int);
	    ints = arguments_add_entity(ints, e_int);
	}
	else {
	    /* if ever e2 is used as e2#new in r1 it must now be
	       replaced by e2#old */
	    entity e_old = entity_to_old_value(e2);
	    if(base_contains_variable_p(r1->base, (Variable) e2))
		r1 = sc_variable_rename(r1, (Variable) e2, (Variable) e_old);
	    /* e2 must be appended to a1 as new t1's arguments;
	       hopefully we are not iterating on a1; but 
	       entity_is_argument_p() receives a longer argument each time;
	       possible improvements? */
	    a1 = gen_nconc(a1, CONS(ENTITY, e2, NIL));
	}
    }

    /* build global linear system: r1 is destroyed, r2 is preserved
     */
    r1 = sc_append(r1, r2);

    /* ??? The base returned may be empty... FC...
     * boumbadaboum in the projection later on.
     */
    sc_rm(r2);
    r2 = SC_UNDEFINED;
    ifdebug(9) {
	(void) fprintf(stderr, "%s: %s", "[transformer_combine]",
		       "global linear system r1 before projection\n");
	sc_fprint(stderr, r1, dump_value_name);
	sc_dump(r1);
    }
    
    /* get rid of intermediate values, if any.
     * ??? guard added to avoid an obscure bug, but I guess it should
     * never get here with en nil base... FC
     */
    if (sc_base(r1)) {
	MAP(ENTITY, e_temp,
	{
	    if (sc_expensive_projection_p(r1,(Variable) e_temp)) {
		ifdebug(9) {
		    pips_debug(9, "expensive projection on %s with\n",
			       entity_local_name(e_temp));
		    sc_fprint(stderr, r1,entity_local_name);
		}
		sc_elim_var(r1,(Variable) e_temp);
		sc_base_remove_variable(r1,(Variable) e_temp);
		ifdebug(9) {
		    pips_debug(9, "simplified tranformer\n");
		    sc_fprint(stderr, r1,entity_local_name);
		}
	    }
	    else {		   
		CATCH(overflow_error) 
		    {
			/* CA */
			pips_user_warning("overflow error in projection of %s, "
					  "variable eliminated\n",
					  entity_name(e_temp)); 
			r1 = sc_elim_var(r1, (Variable) e_temp);
		    }
		TRY 
		    {
			sc_and_base_projection_along_variable_ofl_ctrl
			    (&r1, (Variable) e_temp, NO_OFL_CTRL);
			UNCATCH(overflow_error);
		    }
		
		if (! sc_empty_p(r1)) {
		    Pbase b = base_dup(sc_base(r1));
		    r1 = sc_normalize(r1);
		    if(SC_EMPTY_P(r1)) 
			r1 = sc_empty(b);
		    else
			base_rm(b);
		}
	    }
	},
	    ints);
    }

    ifdebug(9) {
	pips_debug(9, "global linear system r1 after projection\n");
	sc_fprint(stderr, r1, dump_value_name);
	sc_dump(r1);
    }

    /* get rid of ints */
    gen_free_list(ints);
    ints = NIL;

    /* update t1 */
    transformer_arguments(t1) = a1;
    /* predicate_system(transformer_relation(t1)) = (Psysteme) r1; */
    predicate_system_(transformer_relation(t1)) = r1;

     
    debug(8,"transformer_combine","res. t1=%x\n",t1);
    ifdebug(8) (void) dump_transformer(t1);
    debug(8,"transformer_combine","end\n");
    return t1;
}

/* eliminate (some) redundancy */
transformer
transformer_normalize(transformer t, int level)
{
    Psysteme r = (Psysteme) predicate_system(transformer_relation(t));

    if (!sc_empty_p(r)) {
	Pbase b = base_dup(sc_base(r));
	Pbase r2 = sc_dup(r);
	/* Select one tradeoff between speed and accuracy:
	 * enumerated by increasing speeds according to Beatrice
	 */
			    
	CATCH(overflow_error) 
	    {
		/* CA */
		pips_user_warning("overflow error in  redundancy elimination\n"); 
		r = r2;
	    }
	TRY 
	    {
		switch(level) {
		    
		case 0:
		    /* Our best choice for accuracy, but damned slow on ocean */
		    r = sc_elim_redund(r);
		    break;
		    
		case 1:
		    /* Beatrice's best choice: does not deal with minmax2 (only)
		     * but still requires 74 minutes of real time (55 minutes of CPU time)
		     * for ocean preconditions, when applied to each precondition stored.
		     *
		     * Only 64 s for ocean, if preconditions are not normalized.
		     * But andne, callabsval, dead2, hind, negand, negand2, or,
		     * validation_dead_code are not validated any more. Redundancy
		     * could always be detected in a trivial way after propagating
		     * values from equations into inequalities.
		     */
		    sc_nredund(&r);
		    predicate_system_(transformer_relation(t)) = newgen_Psysteme(r);
		    break;
		    
		case 2:
		    /* Francois' own: does most of the easy stuff.
		     * Fails on mimax2 and sum_prec, but it is somehow
		     * more user-friendly because trivial preconditions are
		     * not destroyed as redundant. It makes you feel safer.
		     *
		     * Result for full precondition normalization on ocean: 114 s
		     * for preconditions, 4 minutes between split ocean.f and
		     * OCEAN.prec
		     */
		    r = sc_strong_normalize(r);
		    break;
		    
		case 5:
		    /* Same plus a good feasibility test
		     */
		    r = sc_strong_normalize3(r);
		    break;
		    
		case 3:
		    /* Similar, but variable are actually substituted
		     * which is sometimes painful when a complex equations
		     * is used to replace a simple variable in a simple
		     * inequality.
		     */
		    r = sc_strong_normalize2(r);
		    break;
		case 6:
		    /* Similar, but variables are substituted if they belong to
		     * a more or less simple equation, and simpler equations
		     * are processed first and a lexicographically minimal
		     * variable is chosen when equivalent variables are
		     * available.
		     */
		    r = sc_strong_normalize4(r, (char * (*)(Variable)) external_value_name);
		    break;
		    
		case 7:
		    /* Same plus a good feasibility test, plus variable selection for elimination,
		     * plus equation selection for elimination
		     */
		    r = sc_strong_normalize5(r, (char * (*)(Variable)) external_value_name);
		    break;
		    
		case 4:
		    /* Pretty lousy: equations are not even used to eliminate redundant 
		     * inequalities!
		     */
		    r = sc_normalize(r);
		    break;
		    
		default:
		    pips_error("transformer_normalize", "unknown level %d\n", level);
		}
		sc_rm(r2);
		UNCATCH(overflow_error);
	    }
			
	if (SC_EMPTY_P(r)) {
	    r = sc_empty(b);
	}
	else 
	    base_rm(b);

	r->dimension = vect_size(r->base);
	predicate_system_(transformer_relation(t)) = newgen_Psysteme(r);
    }
    return t;
}

/* transformer transformer_projection(transformer t, cons * args):
 * projection of t along the hyperplane defined by entities in args;
 * this generate a projection and not a cylinder based on the projection
 *
 * use the most complex/complete redundancy elimination in Linear
 *
 * args is not modified. t is modified by side effects.
 */
transformer 
transformer_projection(t, args)
transformer t;
cons * args;
{
    t = transformer_projection_with_redundancy_elimination(t, args, 
							   sc_elim_redund);
    return t;
}

Psysteme 
no_elim(Psysteme ps)
{
    return ps;
}

transformer 
transformer_projection_with_redundancy_elimination(
    transformer t,
    list args,
    Psysteme (*elim)(Psysteme))
{
    /* Library Linear/sc contains several reundancy elimination functions:
     *  sc_elim_redund()
     *  build_sc_nredund_2pass_ofl_ctrl() --- if it had the same profile...
     *  ...
     * no_elim() is provided here to obtain the fastest possible projection
     */
    list new_args = NIL;
    Psysteme r = (Psysteme) predicate_system(transformer_relation(t));

    if(!ENDP(args))
    {
	list cea;

	/* get rid of unwanted values in the relation r and in the basis */
	for (cea = args ; !ENDP(cea); POP(cea)) {
	    entity e = ENTITY(CAR(cea));
	    Pbase b = base_dup(sc_base(r));
            pips_assert("base contains variable to project...",
                        base_contains_variable_p(b, (Variable) e));
            
	    CATCH(overflow_error) 
	    {
                /* FC */
		pips_user_warning("overflow error in projection of %s, "
				  "variable eliminated\n",
				  entity_name(e)); 
		r = sc_elim_var(r, (Variable) e);
	    }
	    TRY 
	    {
		sc_projection_along_variable_ofl_ctrl
		    (&r,(Variable) e, NO_OFL_CTRL);
		UNCATCH(overflow_error);
	    }

	    sc_base_remove_variable(r,(Variable) e);
	    /* Eliminate redundancy at each projection stage
	     * to avoid explosion of the constraint number
	     */
	    /*
	    if (!sc_empty_p(r) {
		r = elim(r);
		if (SC_EMPTY_P(r)) {
		    r = sc_empty(b);
		    sc_base_remove_variable(r,(Variable) e);
		}
		else base_rm(b);
	    }
	    */
	}

	/* Eliminate redundancy only once projections have all
	 * been performed because redundancy elimination is
	 * expensive and because most variables are exactly 
	 * projected because they appear in at least one equation
	 */
	if (!sc_empty_p(r)) {
	    Pbase b = base_dup(sc_base(r));
	    r = elim(r);
	    if (SC_EMPTY_P(r)) {
		r = sc_empty(b);
	    }
	    else 
		base_rm(b);
	}

	r->dimension = vect_size(r->base);

	/* compute new_args */
	MAPL(ce, { entity e = ENTITY(CAR(ce));
		   if((entity) gen_find_eq(e, args) ==
		      (entity) chunk_undefined) {
		       /* e must be kept if it is not in args */
		       new_args = arguments_add_entity(new_args, e);
		   }},
	     transformer_arguments(t));

	/* update the relation and the arguments field for t */

	/* the relation is updated by side effect FI ?
	 * Maybe not if SC_EMPTY(r) 1 Feb. 94 */
	predicate_system_(transformer_relation(t)) = newgen_Psysteme(r);

	/* replace the old arguments by the new one */
	gen_free_list(transformer_arguments(t));
	transformer_arguments(t) = new_args;
    } 
    return t;
}

/* transformer transformer_apply(transformer tf, transformer pre):
 * apply transformer tf on precondition pre to obtain postcondition post
 *
 * There is (should be!) no sharing between pre and tf. No sharing is
 * introduced between pre or tf and post. Neither pre nor tf are modified.
 */
transformer 
transformer_apply(tf, pre)
transformer tf;
transformer pre;
{
    transformer post;
    transformer copy_pre;

    debug(8,"transformer_apply","begin\n");
    pips_assert("transformer_apply", tf!=transformer_undefined);
    debug(8,"transformer_apply","tf=%x\n", tf);
    ifdebug(8) (void) print_transformer(tf);
    pips_assert("transformer_apply", pre!=transformer_undefined);
    debug(8,"transformer_apply","pre=%x\n", pre);
    ifdebug(8) (void) print_transformer(pre);

    /* post = tf o pre ; pre would be modified by transformer_combine */
    copy_pre = transformer_dup(pre);
    post = transformer_combine(copy_pre, tf);

    pips_assert("transformer_apply", post!=transformer_undefined);
    debug(8,"transformer_apply","post=%x\n", post);
    ifdebug(8) (void) print_transformer(post);
    pips_assert("transformer_apply: unexpected sharing:",post != pre);
    debug(8,"transformer_apply","end\n");
    return post;
}

/* transformer transformer_filter(transformer t, cons * args):
 * projection of t along the hyperplane defined by entities in args;
 * this generate a projection and not a cylinder based on the projection;
 *
 * if the relation associated to t is empty, t is not modified although
 * it should have a basis and this basis should be cleaned up. Since
 * no basis is carried in the current implementation of an empty system,
 * this cannot be performed (FI, 7/12/92).
 *
 * formal argument args is not modified
 *
 * Note: this function is almost equal to transformer_projection();
 * however, entities of args do not all have to appear in t's relation;
 * thus transformer_filter has a larger definition domain than 
 * transformer_projection; on transformer_projection's domain, both
 * functions are equal
 *
 * transformer_projection is useful to get cores when you know all entities
 * in args should appear in the relation.
 */
transformer 
transformer_filter(t, args)
transformer t;
cons * args;
{
    cons * new_args = NIL;
    Psysteme r = (Psysteme) predicate_system(transformer_relation(t));

    if(!ENDP(args) && !SC_EMPTY_P(r)) {
	/* get rid of unwanted values in the relation r and in the basis */
	MAPL(cea, { entity e = ENTITY(CAR(cea));
		    if(base_contains_variable_p(r->base, (Variable) e)) {
			/* r = sc_projection(r, (Variable) e); */
			/*
                        sc_projection_along_variable_ofl_ctrl(&r, (Variable) e,
                                                              NO_OFL_CTRL);  */
			CATCH(overflow_error) 
			    {				    
				/* CA */
				pips_user_warning("overflow error in projection of %s, "
						  "variable eliminated\n",
						  entity_name(e)); 
				r = sc_elim_var(r, (Variable) e);
			    }
			TRY 
			    {
				sc_projection_along_variable_ofl_ctrl
				    (&r, (Variable) e, NO_OFL_CTRL);
				UNCATCH(overflow_error);
			    }
			/*       sc_projection_along_variable_ofl_ctrl(&r, (Variable) e,
				 OFL_CTRL);*/
			sc_base_remove_variable(r,(Variable) e);}},
	args);
	r->dimension = vect_size(r->base);
	
	/* compute new_args */
	/* use functions on arguments instead of in-lining !
	  MAPL(ce, { entity e = ENTITY(CAR(ce));
	  if((entity) gen_find_eq(e, args)== (entity) chunk_undefined) {
	  -- e must be kept if it is not in args --
	  new_args = arguments_add_entity(new_args, e);
	  }},
	  transformer_arguments(t));
	  */
	new_args = arguments_difference(transformer_arguments(t), args);

	/* update the relation and the arguments field for t */

	/* Is the relation updated by side effect?
	 * Yes, in general. No if the system is non feasible
	 */

	predicate_system_(transformer_relation(t)) = newgen_Psysteme(r);

	/* replace the old arguments by the new one */
	free_arguments(transformer_arguments(t));
	transformer_arguments(t) = new_args;
    } 
    return t;
}

/* bool transformer_affect_linear_p(transformer tf, Pvecteur l): returns TRUE
 * if there is a state s such that eval(l, s) != eval(l, tf(s));
 * returns FALSE if l is invariant w.r.t. tf, i.e. for all state s,
 * eval(l, s) == eval(l, tf(s))
 */
bool 
transformer_affect_linear_p(tf, l)
transformer tf;
Pvecteur l;
{
    if (!transformer_undefined_p(tf)){
	list args = transformer_arguments(tf);

	MAP(ENTITY, e, 
	{
	    Value v = vect_coeff((Variable) e, l);
	    if(value_notzero_p(v)) return TRUE;
	},
	    args);
    }

    return FALSE;
}

/* Generates a transformer abstracting a totally unknown modification of
 * the values associated to variables in list le.
 */
transformer 
args_to_transformer(le)
list le; /* list of entities */
{
    transformer tf = transformer_identity();
    cons * args = transformer_arguments(tf);
    Pbase b = VECTEUR_NUL;
    Psysteme s = sc_new();

    MAPL(ce, { 
      entity e = ENTITY(CAR(ce));
      entity new_val = entity_to_new_value(e);

      args = arguments_add_entity(args, new_val);
      b = vect_add_variable(b, (Variable) new_val);
      }, le);

    transformer_arguments(tf) = args;
    s->base = b;
    s->dimension = vect_size(b);
    predicate_system_(transformer_relation(tf)) = s;
    return tf;
}

/* transformer invariant_wrt_transformer(transformer p, transformer tf):
 * keep the invariant part of predicat p wrt tf in a VERY crude way;
 * old and new values related to an entity modified by tf are discarded
 * by projection, regardless of the way they are modified; information
 * that they are modified is preserved; in fact, this is *not* a projection
 * but a cylinder based on the projection.
 *                                                      inf
 * A real fix-point a la Halbwachs should be used p' = UNION(tf^k(p))
 *                                                      k=0
 * or simply one of PIPS loop fix-points.
 *
 * Be careful if tf is not feasible because the result is p itself which may not
 * be what you expect.
 *
 * p is not modified
 */
transformer 
invariant_wrt_transformer(p, tf)
transformer p;
transformer tf;
{
    transformer rtf = args_to_transformer(transformer_arguments(tf));
    transformer inv = transformer_apply(rtf, p);

    free_transformer(rtf);

    return inv;
}

/*transformer transformer_value_substitute(transformer t,
 *                                         entity e1, entity e2):
 * if e2 does not appear in t initially:
 *    replaces occurences of value e1 by value e2 in transformer t's arguments
 *    and relation fields; 
 * else
 *    error
 * fi
 *
 * "e2 must not appear in t initially": this is the general case; the second case
 * may occur when procedure A calls B and C and when B and C share a global variable X
 * which is not seen from A. A may contain relations between B:X and C:X...
 * See hidden.f in Bugs or Validation...
 */
transformer 
transformer_value_substitute(t, e1, e2)
transformer t;
entity e1;
entity e2;
{
    /* updates are performed by side effects */

    cons * a = transformer_arguments(t);
    Psysteme s = (Psysteme) predicate_system(transformer_relation(t));

    pips_assert("transformer_value_substitute",
		e1 != entity_undefined && e2 != entity_undefined);
    /*
    pips_assert("transformer_value_substitute", 
		!base_contains_variable_p(s->base, (Variable) e2));
		*/

    /* update only if necessary */
    if(base_contains_variable_p(s->base, (Variable) e1)) {

	if(!base_contains_variable_p(s->base, (Variable) e2)) {

	    (void) sc_variable_rename(s,(Variable) e1, (Variable)e2);

	    /* rename value e1 in argument a; e1 does not necessarily
	       appear in a because it's not necessarily the new value of
	       a modified variable */
	    MAPL(ce, {entity e = ENTITY(CAR(ce));
		      if( e == e1) ENTITY(CAR(ce)) = e2;},
		 a);
	}
	else {
	    pips_error("transformer_value_substitute",
		       "cannot substitute e1=%s by e2=%s: e2 already in basis\n",
		       entity_name(e1), entity_name(e2));
	}
    }

    return t;
}

/* If TRUE is returned, the transformer certainly is empty.
 * If FALSE is returned,
 * the transformer still might be empty, it all depends on the normalization
 * procedure power. Beware of its execution time!
 */
static bool 
parametric_transformer_empty_p(transformer t,
			       Psysteme (*normalize)(Psysteme,
						     char * (*)(Variable)))
{
    /* FI: the arguments seem to have no impact on the emptiness
     * (i.e. falseness) of t
     */
    predicate pred = transformer_relation(t);
    Psysteme ps = predicate_system(pred);
    Psysteme new_ps = sc_dup (ps);
    bool empty_p = FALSE;

    /* empty_p = !sc_faisabilite(ps); */
    /* empty_p = !sc_rational_feasibility_ofl_ctrl(ps, OFL_CTRL, TRUE); */

    /* Normalize the transformer, use all "reasonnable" equations
     * to reduce the problem
     * size, check feasibility on the projected system
     */
    /* new_ps = normalize(new_ps, (char * (*)(Variable)) external_value_name); */
    /* FI: when dealing with interprocedural preconditions, the value mappings
     * are initialized for the caller but the convex hull, which calls this function,
     * must be performed in the calle value space.
     */
    new_ps = normalize(new_ps, (char * (*)(Variable)) entity_local_name);

    if(SC_EMPTY_P(new_ps)) {
	empty_p = TRUE;
    }
    else {
	sc_rm(new_ps);
	empty_p = FALSE;
    }

    return empty_p;
}

/* If TRUE is returned, the transformer certainly is empty.
 * If FALSE is returned,
 * the transformer still might be empty, but it's not too likely...
 */
bool 
transformer_empty_p(transformer t)
{
    bool empty_p = parametric_transformer_empty_p(t, sc_strong_normalize4);
    return empty_p;
}

/* If TRUE is returned, the transformer certainly is empty.
 * If FALSE is returned,
 * the transformer still might be empty, but it's not likely at all...
 */
bool 
transformer_strongly_empty_p(transformer t)
{
    bool empty_p = parametric_transformer_empty_p(t, sc_strong_normalize5);
    return empty_p;
}
