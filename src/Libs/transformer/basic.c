 /* transformer package - basic routines
  *
  * Francois Irigoin
  */

#include <stdio.h>
#include <malloc.h> 

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"

#include "transformer.h"

transformer 
transformer_dup(t_in)
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
    predicate_system_(transformer_relation(t_out)) = sc_dup(sc);

    return t_out;
}


void 
transformer_free(t)
transformer t;
{
    free_transformer(t);
}

void 
old_transformer_free(t)
transformer t;
{
    /* I should use gen_free directly but Psysteme is not yet properly
       interfaced with NewGen */
    Psysteme s;

    pips_assert("transformer_free", t != transformer_undefined);

    s = (Psysteme) predicate_system(transformer_relation(t));
    sc_rm(s);
    predicate_system_(transformer_relation(t)) = SC_UNDEFINED;
    /* gen_free should stop before trying to free a Psysteme and
       won't free entities in arguments because they are tabulated */
    /* commented out for DRET demo */
    /*
    gen_free(t);
    */
    /* end of DRET demo */
}

transformer 
transformer_identity()
{
    /* return make_transformer(NIL, make_predicate(SC_RN)); */
    /* en fait, on voudrait initialiser a "liste de contraintes vide" */
    return make_transformer(NIL,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED,
						   CONTRAINTE_UNDEFINED))); 
}

transformer 
transformer_empty()
{
    return make_transformer(NIL,
			    make_predicate(sc_empty(BASE_NULLE)));
}

bool 
transformer_identity_p(t)
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
transformer 
transformer_add_loop_index(t, i, incr)
transformer t;
entity i;
Pvecteur incr;
{
    /* Psysteme * ps = 
       &((Psysteme) predicate_system(transformer_relation(t))); */
    Psysteme psyst = predicate_system(transformer_relation(t));
    entity i_old;

    transformer_arguments(t) = arguments_add_entity(transformer_arguments(t), i);
    i_old = entity_to_old_value(i);
    psyst->base = vect_add_variable(psyst->base, (Variable) i);
    psyst->base = vect_add_variable(psyst->base, (Variable) i_old);
    psyst->dimension = vect_size(psyst->base);
    vect_chg_coeff(&incr, (Variable) i, -1);
    vect_chg_coeff(&incr, (Variable) i_old, 1);
    psyst = sc_equation_add(psyst, contrainte_make(incr));

    return t;
}

transformer 
transformer_constraint_add(tf, i, equality)
transformer tf;
Pvecteur i;
bool equality;
{
    Pcontrainte c;
    Psysteme sc; 

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

    sc = sc_constraint_add(sc, c, equality);

    return tf;
}

transformer 
transformer_inequality_add(tf, i)
transformer tf;
Pvecteur i;
{
    return transformer_constraint_add(tf, i, FALSE);
}

transformer 
transformer_equality_add(tf, i)
transformer tf;
Pvecteur i;
{
    return transformer_constraint_add(tf, i, TRUE);
}

transformer 
transformer_equalities_add(tf, eqs)
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

/* FI: I do not know if this procedure should always return or fail when
 * an inconsistency is found. For instance, summary transformers for callees
 * are inconsistent with respect to the current module. FC/CA: help...
 *
 * I do not understand why errors are reported only if the debug level is greater
 * than 1. A demo effect?
 *
 * Also, since no precise information about the inconsistency is displayed, a
 * core dump would be welcome to retrieve pieces of information with gdb.
 *
 * But, see final comment... In spite of it, I do not always return any longer.
 */
bool 
transformer_consistency_p(t)
transformer t;
{
#define TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL 0
    /* the relation should be consistent 
     * and any variable corresponding to an old value
     * should appear in the argument list since 
     * an old value cannot (should not) be
     * introduced unless the variable is changed and
     * since every changed variable is
     * in the argument list.
     *
     * Apparently, a variable may appear as an argument but its old value
     * does not have to appear in the basis if it is not required by
     * the constraints. This does not seem very safe to me (FI, 13 Nov. 95)
     */
    Psysteme sc = (Psysteme) predicate_system(transformer_relation(t));
    Pbase b = sc_base(sc);
    list args = transformer_arguments(t);
    bool consistent = TRUE;

    /* The NewGen data structure must be fully defined */
    ifdebug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL)
	consistent = gen_defined_p(t);
    else
	consistent = TRUE;
    if(!consistent)
	debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
	      "transformer_consistency_p", "transformer t is not gen_defined\n");

    /* The predicate must be weakly consistent. Every variable
     * in the constraints must be in the basis (but not the other
     * way round.
     */
    consistent = consistent && sc_weak_consistent_p(sc);
    if(!consistent)
	debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
	      "transformer_consistency_p", "sc is not weekly consistent\n");

    /* If an old value appears in the predicate, the corresponding
     * variable should be an argument of the transformer
     */
    if(consistent) {
	Pbase b = sc_base(sc);
	Pbase t = BASE_UNDEFINED;

	for( t = b; !BASE_UNDEFINED_P(t) && consistent; t = t->succ) {
	    entity val = (entity) vecteur_var(t);

       /* test aliasing between arguments and relations
          high cost testing */
	    ifdebug(8) {  
		boolean aliasing = FALSE;
		string emn =entity_module_name(val);
		string eln =entity_local_name(val);
		list lt =  args;
		entity e;
		for (lt =  args; lt && !aliasing ;POP(lt))
		{
		    e = ENTITY(CAR(lt));
		    consistent = consistent && 
			(same_string_p(entity_local_name(e), eln) ? 
			 same_string_p(entity_module_name(e),emn) 
			 : TRUE);
		    aliasing = aliasing && entity_conflict_p(e,val);
		}
		
		if(!consistent)
		    user_warning("transformer_consistency_p", 
				 "different global variable names in  arguments and basis (%s) \n",
				 eln);
		if (aliasing)
		    pips_error("transformer_consistency_p", 
			       "aliasing between  arguments and basis (%s) \n",
			       entity_name(val));
	    }
	   
	    /* FI: the next test is not safe because val can be
	     * a global value not recognized in the current
	     * context. old_value_entity_p() returns TRUE or FALSE
	     * or pips_error.
	     *
	     * A general version of this routine is needed...
	     * the return value of a function is not recognized as a 
	     * global value by old_value_entity_p
	     */
	    if( !storage_return_p(entity_storage(val))
		&& old_value_entity_p(val)) {
		entity var = value_to_variable(val);

		consistent = entity_is_argument_p(var, args);
		if(!consistent)
		    debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
			  "transformer_consistency_p",
			  "Old value of % s in sc but not in arguments\n",
			  entity_name(var));
	    }
	    /* The constant term should not appear in the basis */
	    if(consistent) {
		consistent = consistent && !term_cst(t);
		if(!consistent)
		    debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
			  "transformer_consistency_p", "TCST in sc basis\n");
	    }
	}
    }

    /* The constant term should not be an argument */
    if(consistent) {
	MAP(ENTITY, e, {
	    consistent = consistent && (e != (entity) TCST);
	}, args);
	if(!consistent)
	    debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
		  "transformer_consistency_p", "TCST appears in arguments\n");
    }

    /* Check that the transformer is compatible with the current value mappings.
     *
     * This is not always true as you may need to import the summary transformer
     * of a callee. Before translation, this check will most likely fail.
     *
     * Debugging step which does not return if an incompatibility is found.
     */

    /* Check that every argument has a value.
     * This is not redundant with the printout procedure which uses
     * entity_minimal_name() and not the value mappings.
     */
    MAP(ENTITY, e, {
	/*
	pips_assert("Argument entity appears in the value mappings",
		    entity_has_values_p(e));
		    */
	if(!entity_has_values_p(e)) {
	    pips_user_warning("No value for argument %s in value mappings\n",
			      entity_name(e));
	    consistent = FALSE;
	}
    }, args);

    /* If a variable appears as argument, its new value must be in the basis
     * See for instance, effects_to_transformer()
     */
    MAP(ENTITY, e, {
	entity v = entity_to_new_value(e);
	/*
	pips_assert("Argument is in the basis", base_contains_variable_p(b, (Variable) v));
	*/
	if(!base_contains_variable_p(b, (Variable) v)) {
	    pips_user_warning("No value for argument %s in relation basis\n",
			      entity_name(e));
	    consistent = FALSE;
	}
    }, args);

    /* FI: let the user react and print info before core dumping */
    /* pips_assert("transformer_consistency_p", consistent); */

    return consistent;
}
