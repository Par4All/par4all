 /* transformer package - basic routines
  *
  * Francois Irigoin
  *
  * $id$
  *
  * $Log: basic.c,v $
  * Revision 1.23  2003/07/24 08:36:20  irigoin
  * Functions empty_transformer(), transformer_weak_consistency_p(),
  * transformer_general_consistency_p(), transformer_add_modified_variable()
  * and move_transformer() added. Plus some reformatting.
  *
  * Revision 1.22  2001/10/22 15:54:36  irigoin
  * reformatting + precondition_to_abstract_store() added, although it is
  * redundant with transformer_range()
  *
  * Revision 1.21  2001/07/19 18:03:21  irigoin
  * Lots of additions for a better handling of multitype transformers
  *
  *
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
    /* FI: I do not reduce transformer_dup() to a macro calling
       copy_transformer() because I do not want to create problems with
       the link edit and because I want to keep the assertion */

    Psysteme sc = SC_UNDEFINED;
    transformer t_out;

    pips_assert("transformer_dup", t_in != transformer_undefined);

    sc = (Psysteme) predicate_system(transformer_relation(t_in));
    pips_assert("transformer_dup", !SC_UNDEFINED_P(sc));
    t_out = copy_transformer(t_in);

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

/* Do not allocate an empty transformer, but transform an allocated
   transformer into an empty_transformer. */
transformer empty_transformer(transformer t)
{
  free_predicate(transformer_relation(t));
  gen_free_list(transformer_arguments(t));
  transformer_arguments(t) = NIL;
  transformer_relation(t) = make_predicate(sc_empty(BASE_NULLE));
  return t;
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
 * incr is supposed to be compatible with the value mappings
 *
 * Pvecteur incr should not be used after a call to transformer_add_index
 * because it is shared by t and modified
 */
transformer 
transformer_add_variable_incrementation(t, i, incr)
transformer t;
entity i;
Pvecteur incr;
{
    /* Psysteme * ps = 
       &((Psysteme) predicate_system(transformer_relation(t))); */
    Psysteme psyst = predicate_system(transformer_relation(t));
    entity i_old = entity_to_old_value(i);
    entity i_new = entity_to_new_value(i);
    entity i_rep = value_to_variable(i_new);

    transformer_arguments(t) = arguments_add_entity(transformer_arguments(t), i_rep);
    psyst->base = vect_add_variable(psyst->base, (Variable) i_new);
    psyst->base = vect_add_variable(psyst->base, (Variable) i_old);
    psyst->dimension = vect_size(psyst->base);
    vect_chg_coeff(&incr, (Variable) i_new, -1);
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

    pips_assert("tf is defined", tf != transformer_undefined
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

/* Warning: */
transformer 
transformer_inequalities_add(transformer tf, Pcontrainte ineqs)
{
  Pcontrainte ineq = CONTRAINTE_UNDEFINED;

    for(ineq = ineqs; !CONTRAINTE_UNDEFINED_P(ineq); ineq = contrainte_succ(ineq))
	(void) transformer_constraint_add(tf, 
					  contrainte_vecteur(ineq),
					  FALSE);
    return tf;
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

bool transformer_argument_consistency_p(transformer t)
{
  return transformer_argument_general_consistency_p(t, FALSE);
}

bool transformer_argument_weak_consistency_p(transformer t)
{
  return transformer_argument_general_consistency_p(t, TRUE);
}

bool transformer_argument_general_consistency_p(transformer t, bool is_weak)
{
  list args = transformer_arguments(t);
  bool consistent = TRUE;
  Psysteme sc = (Psysteme) predicate_system(transformer_relation(t));
  Pbase b = sc_base(sc);

  /* If no final state can be reached, no variable can be changed in between */
  if(sc_empty_p(sc)) {
    consistent = ENDP(args);
    pips_assert("Empty transformer must have no arguments", consistent);
  }
  else if(!is_weak) {
    /* If a variable appears as argument, its new value must be in the basis
     * See for instance, effects_to_transformer()
     */

    MAP(ENTITY, e, {
      entity v = entity_to_new_value(e);
      /*
	pips_assert("Argument is in the basis", base_contains_variable_p(b, (Variable) v));
      */
      if(!base_contains_variable_p(b, (Variable) v)) {
	/* pips_user_warning("No value for argument %s in relation basis\n",
	   entity_name(e)); */
	pips_internal_error("No value for argument %s in relation basis\n",
			    entity_name(e));
	consistent = FALSE;
      }
    }, args);
    pips_assert("Argument variables must have values in basis", consistent);
  }

  return consistent;
}

/* FI: I do not know if this procedure should always return or fail when
 * an inconsistency is found. For instance, summary transformers for callees
 * are inconsistent with respect to the current module. FC/CA: help...
 *
 * I do not understand why errors are reported only if the debug level is greater
 * than 1. A demo effect? No, this routine is coded that way to save time on
 * regular runs.
 *
 * Also, since no precise information about the inconsistency is
 * displayed, a core dump would be welcome to retrieve pieces of
 * information with gdb.  The returned value should always be tested and a
 * call to pips_internal_error() should always be performed if an
 * inconsistency is detected.
 *
 * But, see final comment... In spite of it, I do not always return any longer.  */
bool transformer_consistency_p(t)
transformer t;
{
  return transformer_general_consistency_p(t, FALSE);
}

/* Interprocedural transformers do not meet all conditions. */
bool transformer_weak_consistency_p(t)
transformer t;
{
  return transformer_general_consistency_p(t, TRUE);
}

bool transformer_general_consistency_p(transformer t, bool is_weak)
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
    list args = transformer_arguments(t);
    bool consistent = TRUE;

    /* The NewGen data structure must be fully defined */
    ifdebug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL)
	consistent = transformer_defined_p(t);
    else
	consistent = TRUE;
    if(!consistent)
	debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
	      "transformer_consistency_p", "transformer t is not gen_defined\n");

    /* The predicate must be weakly consistent. Every variable
     * in the constraints must be in the basis (but not the other
     * way round).
     */
    consistent = consistent && sc_weak_consistent_p(sc);
    if(!consistent)
	debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
	      "transformer_consistency_p", "sc is not weakly consistent\n");

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
	     *
	     * old_value_entity_p() is likely to core dump on
	     * interprocedural transformers and preconditions.
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
	    if(!is_weak)
	      consistent = FALSE;
	}
    }, args);

    if(consistent && !is_weak)
      consistent = transformer_argument_consistency_p(t);

    /* FI: let the user react and print info before core dumping */
    /* pips_assert("transformer_consistency_p", consistent); */

    return consistent;
}

/* Same as above but equivalenced variables should not appear in the
   argument list or in the predicate basis. */
bool transformer_internal_consistency_p(transformer t)
{
  Psysteme sc = (Psysteme) predicate_system(transformer_relation(t));
  Pbase b = sc_base(sc);
  Pbase e = BASE_UNDEFINED;
  list args = transformer_arguments(t);
  bool consistent = transformer_consistency_p(t);

  MAP(ENTITY, e, {
    entity v = entity_to_new_value(e);

    if(v!=e) {
      pips_user_warning("New value %s should be the same entity as variable %s"
			" as long as equivalence equations are not added\n", 
			entity_local_name(v), entity_local_name(e));
      pips_assert("Argument must be a value", FALSE);
    }
  }, args);

  for(e=b; !BASE_NULLE_P(e); e = vecteur_succ(e)) {
    entity val = (entity) vecteur_var(e);

    if(!(new_value_entity_p(val) || old_value_entity_p(val)
	 || intermediate_value_entity_p(val))) {
      if(!entity_constant_p(val)) {
	pips_user_warning("Variable %s in basis should be an internal value",
			  entity_local_name(val));
	pips_assert("Basis variables must be an internal value", FALSE);
      }
    }
  }

  return consistent;
}

list transformer_projectable_values(transformer tf)
{
  list proj = NIL;
  Psysteme sc = predicate_system(transformer_relation(tf));
  Pbase b = BASE_UNDEFINED;

  for(b=sc_base(sc); !BASE_NULLE_P(b); b = vecteur_succ(b)) {
    entity v = (entity) vecteur_var(b);

    proj = CONS(ENTITY, v, proj);
  }

  return proj;
}
/* Get rid of all old values and arguments. Argument pre is unchanged and
   result as is allocated. Should be a call to transformer_range(). */
transformer
precondition_to_abstract_store(transformer pre)
{
  transformer as = transformer_dup(pre);

  /* Project all old values */
  as = transformer_projection(as, transformer_arguments(as));

  /* Redefine the arguments */
  gen_free_list(transformer_arguments(as));
  transformer_arguments(as) = NIL;

  return as;
}

transformer transformer_add_modified_variable(
    transformer tf,
    entity var)
{
  /* Should we check that var has values? */
  Psysteme sc =  (Psysteme) predicate_system(transformer_relation(tf));
  Pbase b = sc_base(sc);

  transformer_arguments(tf) = arguments_add_entity(transformer_arguments(tf), var);
  sc_base(sc) = vect_add_variable(b, (Variable) var);
  sc_dimension(sc) = base_dimension(sc_base(sc));

  return tf;
}

/* Move arguments and predicate of t2 into t1, free old arguments and
   predicate of t1, free what's left of t2. This is used to perform a side
   effect on an argument when a function allocates a new transformer to
   return a result. t2 should not be used after a call to move_transformer() */
transformer move_transformer(transformer t1, transformer t2)
{
  pips_assert("t1 is consistent on entry", transformer_consistency_p(t1));
  pips_assert("t2 is consistent on entry", transformer_consistency_p(t2));

  free_arguments(transformer_arguments(t1));
  transformer_arguments(t1) = transformer_arguments(t2);
  transformer_arguments(t2) = NIL;

  sc_rm(predicate_system(transformer_relation(t1)));
  predicate_system(transformer_relation(t1))
    = predicate_system(transformer_relation(t2));
  predicate_system(transformer_relation(t2))= SC_UNDEFINED;

  free_transformer(t2);

  pips_assert("t1 is consistent on exit", transformer_consistency_p(t1));

  return t1;
}
