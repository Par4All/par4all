 /* Predicate transformer package: sc complexity level */

#include <stdio.h>
extern int fprintf();
extern int printf();
#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"
#include "semantics.h"

#include "transformer.h"

/* transformer transformer_combine(transformer t1, transformer t2):
 * compute the composition of transformers t1 and t2 (t1 then t2)
 *
 * t1 := t2 o t1
 * return t1
 *
 * t1 is updated, but t2 is preserved
 */
transformer transformer_combine(t1, t2)
transformer t1;
transformer t2;
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
    ifdebug(8) (void) print_transformer(t1);
    debug(8,"transformer_combine","arg. t2=%x\n",t2);
    ifdebug(8) (void) print_transformer(t2);

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

    /* build global linear system: r1 is destroyed, r2 is preserved */
    r1 = sc_append(r1, r2);
    sc_rm(r2);
    r2 = SC_UNDEFINED;
    if( get_debug_level() >= 9) {
	(void) fprintf(stderr, "%s: %s", "transformer_combine",
		       "global linear system r1 before projection\n");
	sc_fprint(stderr, r1, external_value_name);
	sc_dump(r1);
    }

    /* get rid of intermediate values */
    MAPL(ce_temp, { entity e_temp = ENTITY(CAR(ce_temp));
		    r1 = sc_projection(r1, (Variable) e_temp);
		    r1 = sc_normalize(r1);
		    if(SC_EMPTY_P(r1))
			break;
		    sc_base_remove_variable(r1,(Variable) e_temp);},
	 ints);
    if(SC_EMPTY_P(r1)) {
	/* FI: this could be eliminated if SC_EMPTY was really usable; 27/5/93 */
	Pvecteur v = vect_new(TCST, 1);
	Pcontrainte eq = contrainte_make(v);

	r1 = sc_make(eq, CONTRAINTE_UNDEFINED);
    }
    else 
	r1->dimension = vect_size(r1->base);

    if( get_debug_level() >= 9) {
	(void) fprintf(stderr, "%s: %s", "transformer_combine",
		       "global linear system r1 after projection\n");
	sc_fprint(stderr, r1, external_value_name);
	sc_dump(r1);
    }

    /* get rid of ints */
    gen_free_list(ints);
    ints = NIL;

    /* update t1 */
    transformer_arguments(t1) = a1;
    /* predicate_system(transformer_relation(t1)) = (Psysteme) r1; */
    predicate_system(transformer_relation(t1)) = (char *) r1;

    debug(8,"transformer_combine","res. t1=%x\n",t1);
    ifdebug(8) (void) print_transformer(t1);
    debug(8,"transformer_combine","end\n");
    return t1;
}

/* transformer transformer_projection(transformer t, cons * args):
 * projection of t along the hyperplane defined by entities in args;
 * this generate a projection and not a cylinder based on the projection
 *
 * args is not modified
 */
transformer transformer_projection(t, args)
transformer t;
cons * args;
{
    cons * new_args = NIL;
    Psysteme r = (Psysteme) predicate_system(transformer_relation(t));

    if(!ENDP(args)) {
	cons * cea;

	/* get rid of unwanted values in the relation r and in the basis */
	for (cea = args ; !ENDP(cea); POP(cea)) {
	    entity e = ENTITY(CAR(cea));
	    Pbase b = base_dup(sc_base(r));
	    r = sc_projection(r, (Variable) e);
	    if (r==SC_EMPTY) {
		r = sc_empty(b);
		sc_base_remove_variable(r,(Variable) e);
	    }
	    else {
		sc_base_remove_variable(r,(Variable) e);
		r = sc_elim_redond(r);
		if (SC_EMPTY_P(r)) {
		    r = sc_empty(b);
		    sc_base_remove_variable(r,(Variable) e);
		}
		else base_rm(b);
	    }
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

	/* the relation is updated by side effect FI ? Maybe not if SC_EMPTY(r) 1 Feb. 94 */
	predicate_system(transformer_relation(t)) = r;

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
transformer transformer_apply(tf, pre)
transformer tf;
transformer pre;
{
    transformer post;

    debug(8,"transformer_apply","begin\n");
    pips_assert("transformer_apply", tf!=transformer_undefined);
    debug(8,"transformer_apply","tf=%x\n", tf);
    ifdebug(8) (void) print_transformer(tf);
    pips_assert("transformer_apply", pre!=transformer_undefined);
    debug(8,"transformer_apply","pre=%x\n", pre);
    ifdebug(8) (void) print_transformer(pre);

    /* post = tf o pre ; pre would be modified by transformer_combine */
    post = transformer_combine(transformer_dup(pre), tf);

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
 * args is not modified
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
transformer transformer_filter(t, args)
transformer t;
cons * args;
{
    cons * new_args = NIL;
    Psysteme r = (Psysteme) predicate_system(transformer_relation(t));

    if(!ENDP(args) && !SC_EMPTY_P(r)) {
	/* get rid of unwanted values in the relation r and in the basis */
	MAPL(cea, { entity e = ENTITY(CAR(cea));
		    if(base_contains_variable_p(r->base, (Variable) e)) {
			r = sc_projection(r, (Variable) e);
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

	/* the relation is updated by side effect FI ? */

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
bool transformer_affect_linear_p(tf, l)
transformer tf;
Pvecteur l;
{
    if (!transformer_undefined_p(tf)){
	list args = transformer_arguments(tf);

	MAPL(cef, { entity e = ENTITY(CAR(cef));
		    if(vect_coeff((Variable) e, l)!=0)
			return TRUE;},
	     args);
    }

    return FALSE;
}

/* transformer invariant_wrt_transformer(transformer p, transformer tf):
 * keep the invariant part of predicat p wrt tf in a VERY crude way;
 * old and new values related to an entity modified by tf are discarded
 * by projection, regardless of the way they are modified; information
 * that they are modified is preserved
 *
 * p is modified and returned;
 */
transformer invariant_wrt_transformer(p, tf)
transformer p;
transformer tf;
{
    cons * olds = NIL;

    /* get rid of new values */
    p = transformer_filter(p, transformer_arguments(tf));

    /* get rid of old values */
    MAPL(ce, {entity e = ENTITY(CAR(ce));
	      olds = CONS(ENTITY, entity_to_old_value(e), olds);},
	 transformer_arguments(tf));
    p = transformer_filter(p, olds);

    free_arguments(olds);

    return p;
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
transformer transformer_value_substitute(t, e1, e2)
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
	    pips_error("transformer_value_substitute", "conflict between e1=%s and e2=%s\n",
		       entity_name(e1), entity_name(e2));
	}
    }

    return t;
}

/* Return true if a statement is feasible according to the precondition. */
bool statement_feasible_p(statement s)
{
  transformer pre;
  Psysteme ps;
  predicate pred;

  pre = load_statement_precondition(s);
  if (get_debug_level() >= 7) {
     (void) printf("Precondition 0x%x\n", pre);
  }
  
  pred = transformer_relation(pre);
  
  ps = predicate_system(pred);

  if (get_debug_level() >= 6) {
    (void) printf("C  %s\n", precondition_to_string(pre));
    print_statement(s);
    if (get_debug_level() >= 7) sc_dump(ps);
    (void) printf("\"statement_feasible_p\" Faisabilite' = %d\n\n",
		  sc_faisabilite(ps));
  }

  /* Renvoie l'utilite' : */  
  return sc_faisabilite(ps);
}
