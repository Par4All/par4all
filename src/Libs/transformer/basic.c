 /* transformer package - basic routines
  *
  * Francois Irigoin
  */

#include <stdio.h>

#include "genC.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "transformer.h"

transformer transformer_dup(t_in)
transformer t_in;
{
    /* I should use gen_copy_tree but directly Psysteme is not yet properly
       interfaced with NewGen */
    transformer t_out;
    Psysteme sc;

    pips_assert("transformer_dup", t_in != transformer_undefined);

    t_out = transformer_identity();
    transformer_arguments(t_out) = 
	(cons *) gen_copy_seq(transformer_arguments(t_in));
    sc = (Psysteme) predicate_system(transformer_relation(t_in));
    pips_assert("transformer_dup", !SC_UNDEFINED_P(sc));
    predicate_system(transformer_relation(t_out)) = 
	(char *) sc_dup(sc);

    return t_out;
}

void transformer_free(t)
transformer t;
{
    /* I should use gen_free directly but Psysteme is not yet properly
       interfaced with NewGen */
    Psysteme s;

    pips_assert("transformer_free", t != transformer_undefined);

    s = (Psysteme) predicate_system(transformer_relation(t));
    sc_rm(s);
    predicate_system(transformer_relation(t)) = (char *) SC_UNDEFINED;
    /* gen_free should stop before trying to free a Psysteme and
       won't free entities in arguments because they are tabulated */
    /* commented out for DRET demo */
    /*
    gen_free(t);
    */
    /* end of DRET demo */
}

transformer transformer_identity()
{
    /* return make_transformer(NIL, make_predicate(SC_RN)); */
    /* en fait, on voudrait initialiser a "liste de contraintes vide" */
    return make_transformer(NIL,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED,
						   CONTRAINTE_UNDEFINED))); 
}

transformer transformer_empty()
{
    return make_transformer(NIL,
			    make_predicate(sc_empty(BASE_NULLE)));
}

bool transformer_identity_p(t)
transformer t;
{
    /* no variables are modified; no constraints exist on their values */

    Psysteme s;

    pips_assert("transformer_identity_p", t != transformer_undefined);
    s = (Psysteme) predicate_system(transformer_relation(t));
    return transformer_arguments(t) == NIL && sc_nbre_egalites(s) == 0
	&& sc_nbre_inegalites(s) == 0;
}

/* CHANGE THIS NAME: no loop index please, it's not directly linked
 * to loops!!!
 */

/* transformer transformer_add_loop_index(transformer t, entity i,
 *                                        Pvecteur incr):
 * add the index incrementation expression incr for loop index i to
 * transformer t. 
 *
 * t = intersection(t, i#new = i#old + incr)
 *
 * Pvecteur incr should not be used after a call to transformer_add_index
 * because it is shared by t and modified
 */
transformer transformer_add_loop_index(t, i, incr)
transformer t;
entity i;
Pvecteur incr;
{
    /* Psysteme * ps = 
       &((Psysteme) predicate_system(transformer_relation(t))); */
    Psysteme psyst = (Psysteme) predicate_system(transformer_relation(t));
    Psysteme * ps = &psyst; 
    entity i_old;

    transformer_arguments(t) = arguments_add_entity(transformer_arguments(t),
						    i);
    i_old = entity_to_old_value(i);
    (* ps)->base = vect_add_variable((*ps)->base, (Variable) i);
    (* ps)->base = vect_add_variable((*ps)->base, (Variable) i_old);
    (* ps)->dimension = vect_size((* ps)->base);
    vect_chg_coeff(&incr, (Variable) i, -1);
    vect_chg_coeff(&incr, (Variable) i_old, 1);
    sc_add_egalite(*ps, contrainte_make(incr));

    return t;
}

transformer transformer_constraint_add(tf, i, equality)
transformer tf;
Pvecteur i;
bool equality;
{
    Pcontrainte c;
    Psysteme sc; 
    Pbase old_basis;
    Pbase new_basis;

    pips_assert("transformer_constraint_add", tf != transformer_undefined
		&& tf != (transformer) NULL);

    if(VECTEUR_NUL_P(i)) {
	user_warning("transformer_constraint_add",
		     "trivial constraint 0 %s 0 found: code should be optimized\n",
		     (equality)? "==" : "<=");
	return tf;
    }

    c = contrainte_make(i);
    sc = (Psysteme) predicate_system(transformer_relation(tf));

    if(equality)
	sc_add_egalite(sc,c)
    else
	sc_add_inegalite(sc,c)

    /* maintain consistency, although it's expensive; how about a
       sc_update_base function? Or a proper sc_add_inegalite function? */
    old_basis = sc->base;
    sc->base = (Pbase) VECTEUR_NUL;
    sc_creer_base(sc);
    new_basis = sc->base;
    sc->base = base_union(old_basis, new_basis);
    sc->dimension = base_dimension(sc->base);
    base_rm(new_basis);
    base_rm(old_basis);

    return tf;
}

transformer transformer_inequality_add(tf, i)
transformer tf;
Pvecteur i;
{
    return transformer_constraint_add(tf, i, FALSE);
}

transformer transformer_equality_add(tf, i)
transformer tf;
Pvecteur i;
{
    return transformer_constraint_add(tf, i, TRUE);
}

transformer transformer_equalities_add(tf, eqs)
transformer tf;
Pcontrainte eqs;
{
    /* please, do not introduce any sharing at the Pcontrainte level
       you do not know how they have to be chained in diferent transformers;
       do not introduce any sharing at the Pvecteur level; I'm not
       sure it's so useful, but think of what would happen if one transformer
       is renamed... */
    for(;eqs!=CONTRAINTE_UNDEFINED; eqs = eqs->succ)
	(void) transformer_constraint_add(tf, 
					  vect_dup(contrainte_vecteur(eqs)),
					  TRUE);
    return tf;
}

bool transformer_consistency_p(t)
transformer t;
{
    /* the relation should be consistent and any variable corresponding to an old value
     * should appear in the argument list since an old value cannot (should not) be
     * introduced unless the variable is changed and since every changed variable is
     * in the argument list.
     */
    Psysteme sc = (Psysteme) predicate_system(transformer_relation(t));
    list args = transformer_arguments(t);
    bool consistent = TRUE;

    consistent = sc_consistent_p(sc);

    if(consistent) {
	Pbase b = sc_base(sc);
	Pbase t = BASE_UNDEFINED;

	for( t = b; !BASE_UNDEFINED_P(t) && consistent; t = t->succ) {
	    entity val = (entity) vecteur_var(t);
	    if(old_value_entity_p(val)) {
		entity var = value_to_variable(val);

		consistent = entity_is_argument_p(var, args);
	    }
	}
    }

    pips_assert("transformer_consistency_p", consistent);

    return consistent;
}
