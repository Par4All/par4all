/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
 /* transformer package - basic routines
  *
  * Francois Irigoin
  */

#include <stdio.h>
#include <stdlib.h> 

#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"

#include "transformer.h"
//#include "alias-classes.h"

transformer transformer_dup(transformer t_in)
{
    /* FI: I do not reduce transformer_dup() to a macro calling
       copy_transformer() because I do not want to create problems with
       the link edit and because I want to keep the assertion */

    Psysteme sc = SC_UNDEFINED;
    transformer t_out;

    pips_assert("transformer t_in is not undefined", t_in != transformer_undefined);

    sc = (Psysteme) predicate_system(transformer_relation(t_in));
    pips_assert("transformer_dup", !SC_UNDEFINED_P(sc));
    t_out = copy_transformer(t_in);

    return t_out;
}


void transformer_free(transformer t)
{
    free_transformer(t);
}

void free_transformers(transformer t, ...) {
  va_list args;

  /* Analyze in args the variadic arguments that may be after t: */
  va_start(args, t);
  /* Since a variadic function in C must have at least 1 non variadic
     argument (here the s), just skew the varargs analysis: */
  do {
    free_transformer(t);
    /* Get the next argument: */
    t = va_arg(args, transformer);
  } while(t!=NULL);
  /* Release the variadic analyzis: */
  va_end(args);
}

void old_transformer_free(transformer t)
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

/* Allocate an identity transformer */
transformer transformer_identity()
{
    /* return make_transformer(NIL, make_predicate(SC_RN)); */
    /* en fait, on voudrait initialiser a "liste de contraintes vide" */
    return make_transformer(NIL,
			    make_predicate(sc_make(CONTRAINTE_UNDEFINED,
						   CONTRAINTE_UNDEFINED)));
}

/* Allocate an empty transformer */
transformer transformer_empty()
{
    return make_transformer(NIL,
			    make_predicate(sc_empty(BASE_NULLE)));
}

/* Do not allocate an empty transformer, but transform an allocated
 * transformer into an empty_transformer.
 *
 * Pretty dangerous because the predicate contained in t may have been
 * deleted before empty_transformer is called. It is not clear that
 * the predicate of t can be freed or not: it is up to the caller to
 * be aware of the problem.
 *
 * This function is risky to use. It can cause either a memory leak if
 * the predicate is not freed, or a double free if the pointer in the
 * transformer is already dangling. If the predicate has already been
 * freed during some processing at the linear level, the caller must
 * update the pointer transformer_relation(t) with:
 *
 * transformer_relation(t) = relation_undefined;
 *
 * before the call.
 */
transformer empty_transformer(transformer t)
{
  free_predicate(transformer_relation(t));
  gen_free_list(transformer_arguments(t));
  transformer_arguments(t) = NIL;
  transformer_relation(t) = make_predicate(sc_empty(BASE_NULLE));
  return t;
}

/* Check that t is an identity function */
bool transformer_identity_p(transformer t)
{
    /* no variables are modified; no constraints exist on their values */

    Psysteme s;

    pips_assert("transformer_identity_p", t != transformer_undefined);
    s = (Psysteme) predicate_system(transformer_relation(t));
    return transformer_arguments(t) == NIL && sc_nbre_egalites(s) == 0
	&& sc_nbre_inegalites(s) == 0;
}

/* Check that transformer t is the canonical representation of an
 * empty transformer.
 *
 * See transformer_empty_p(), transformer_strongly_empty_p() if you
 * need to check that a set of affine constraints is not feasible.
 */
bool transformer_is_empty_p(transformer t)
{
    Psysteme s;

    pips_assert("transformer_identity_p", t != transformer_undefined);
    s = (Psysteme) predicate_system(transformer_relation(t));
    return sc_empty_p(s) && ENDP(transformer_arguments(t));
}

/* Check that transformer t is the canonical representation of the
   whole afine space defined by its basis */
bool transformer_is_rn_p(transformer t)
{
    Psysteme s;

    pips_assert("transformer_identity_p", t != transformer_undefined);
    s = (Psysteme) predicate_system(transformer_relation(t));
    return sc_nbre_egalites(s)==0 && sc_nbre_inegalites(s)==0;
}

/* CHANGE THIS NAME: no loop index please, it's not directly linked
 * to loops!!!
 */

/* Add in tf the information that v is striclty positive, strictly negative or zero. 
 *
 * Assume that v is a value and tf is defined.
 */
transformer transformer_add_sign_information(transformer tf,
					     entity v,
					     int v_sign)
{
  Psysteme psyst = predicate_system(transformer_relation(tf));

  ifdebug(1) {
    pips_assert("Transformer tf is consistent on entrance",
		transformer_consistency_p(tf));
  }

  pips_assert("v is a value", value_entity_p(v) || local_temporary_value_entity_p(v));

  psyst->base = base_add_variable(psyst->base, (Variable) v);
  psyst->dimension = base_dimension(psyst->base);

  if(v_sign!=0) {
    Pvecteur cv;
    Pcontrainte ineq;
    if(v_sign>0) {
      cv = vect_new((Variable) v, VALUE_MONE);
      vect_add_elem(&cv, TCST, VALUE_ONE);
    }
    else {
      /* v_sign<0 */
      cv = vect_new((Variable) v, VALUE_ONE);
      vect_add_elem(&cv, TCST, VALUE_ONE);
    }
    ineq = contrainte_make(cv);
    sc_add_inegalite(psyst, ineq);
  }
  else {
    /* v_sign==0*/
    Pvecteur cv = vect_new((Variable) v, VALUE_ONE);
    Pcontrainte eq = contrainte_make(cv);

    sc_add_egalite(psyst, eq);
  }

  ifdebug(1) {
    pips_assert("Transformer tf is consistent on exit",
		transformer_consistency_p(tf));
  }

  return tf;
}

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

/* Add an update of variable v into t
 *
 * NL : this function is not finish
 *      do the same thing than transformer_add_value_update for the moment
 * TODO
 */
transformer transformer_add_variable_update(transformer t, entity v)
{
  Psysteme psyst = predicate_system(transformer_relation(t));
  entity v_new = entity_to_new_value(v);
  entity v_old = entity_to_old_value(v);
  entity v_rep = value_to_variable(v_new);

  transformer_arguments(t) = arguments_add_entity(transformer_arguments(t), v_rep);
  if(!base_contains_variable_p(psyst->base, (Variable) v_new)) {
    psyst->base = base_add_variable(psyst->base, (Variable) v_new);
  }
  if(!base_contains_variable_p(psyst->base, (Variable) v_old)) {
    psyst->base = base_add_variable(psyst->base, (Variable) v_old);
  }
  psyst->dimension = vect_size(psyst->base);

//
//  pips_assert("before value substitution\n", transformer_consistency_p(t));
////  t = transformer_value_substitute(t, v_new, v_old);
//
//  pips_debug(9, "before tes\nt");
//  if(base_contains_variable_p(psyst->base, (Variable) v_new)) {
//    if(!base_contains_variable_p(psyst->base, (Variable) v_old)) {
//      pips_debug(9, "rename variable\n");
//      (void) sc_variable_rename(psyst,(Variable) v_new, (Variable)v_old);
//    }
//  }
//  pips_assert("after value substitution\n", transformer_consistency_p(t));
  return t;
}

/* Add an update of variable v to t (a value cannot be updated) */
transformer transformer_add_value_update(transformer t, entity v)
{
  entity nv = entity_to_new_value(v);
  entity ov = entity_to_old_value(v);

  if(!transformer_empty_p(t)) {
    Psysteme psyst = predicate_system(transformer_relation(t));

    transformer_arguments(t) = arguments_add_entity(transformer_arguments(t), v);
    if(!base_contains_variable_p(psyst->base, (Variable) nv))
      psyst->base = base_add_variable(psyst->base, (Variable) nv);
    if(!base_contains_variable_p(psyst->base, (Variable) ov))
      psyst->base = base_add_variable(psyst->base, (Variable) ov);
    psyst->dimension = vect_size(psyst->base);
  }

  return t;
}

transformer transformer_constraint_add(tf, i, equality)
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
    return transformer_constraint_add(tf, i, false);
}

transformer 
transformer_equality_add(tf, i)
transformer tf;
Pvecteur i;
{
    return transformer_constraint_add(tf, i, true);
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
					  true);
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
					  false);
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

/* Add an equality between two values (two variables?) */
transformer transformer_add_equality(transformer tf, entity v1, entity v2)
{
  Pvecteur eq = vect_new((Variable) v1, (Value) 1);

  //pips_assert("v1 has values", entity_has_values_p(v1));
  //pips_assert("v2 has values", entity_has_values_p(v2));

  vect_add_elem(&eq, (Variable) v2, VALUE_MONE);
  tf = transformer_equality_add(tf, eq);

  return tf;
}

/* Add an equality between a value and an integer constant: v==cst */
transformer transformer_add_equality_with_integer_constant(transformer tf, entity v, long long int cst)
{
  Pvecteur eq = vect_new((Variable) v, VALUE_MONE);

  //pips_assert("v1 has values", entity_has_values_p(v1));
  //pips_assert("v2 has values", entity_has_values_p(v2));

  vect_add_elem(&eq, TCST, (Value) cst);
  tf = transformer_equality_add(tf, eq);

  return tf;
}

/* Add the equality v1 <= v2 or v1 < v2 */
transformer transformer_add_inequality(transformer tf, entity v1, entity v2, bool strict_p)
{
  Pvecteur eq = vect_new((Variable) v1, VALUE_ONE);

  vect_add_elem(&eq, (Variable) v2, VALUE_MONE);
  if(strict_p)
    vect_add_elem(&eq, TCST, VALUE_ONE);
  tf = transformer_inequality_add(tf, eq);

  return tf;
}

/* Add the inequality v <= cst or v >= cst */
transformer transformer_add_inequality_with_integer_constraint(transformer tf, entity v, long long int cst, bool less_than_p)
{
  Pvecteur eq = vect_new((Variable) v, VALUE_ONE);

  if(less_than_p) {
    eq = vect_new((Variable) v, VALUE_ONE);
    vect_add_elem(&eq, TCST, (Value) -cst);
  }
  else {
    eq = vect_new((Variable) v, VALUE_MONE);
    vect_add_elem(&eq, TCST, (Value) cst);
  }

  tf = transformer_inequality_add(tf, eq);

  return tf;
}

/* Add the inequality v <= a x + cst or v >= a x + cst */
transformer transformer_add_inequality_with_affine_term(transformer tf, entity v, entity x, int a, int cst, bool less_than_p)
{
  Pvecteur eq = vect_new((Variable) v, VALUE_ONE);

  if(less_than_p) {
    eq = vect_new((Variable) v, VALUE_ONE);
    vect_add_elem(&eq, (Variable) x, (Value) -a);
    vect_add_elem(&eq, TCST, (Value) -cst);
  }
  else {
    eq = vect_new((Variable) v, VALUE_MONE);
    vect_add_elem(&eq, TCST, (Value) cst);
    vect_add_elem(&eq, (Variable) x, (Value) a);
  }

  tf = transformer_inequality_add(tf, eq);

  return tf;
}

/* Add the inequality v <= a x or v >= a x */
transformer transformer_add_inequality_with_linear_term(transformer tf, entity v, entity x, int a, bool less_than_p)
{
  return transformer_add_inequality_with_affine_term(tf, v, x, a, VALUE_ZERO, less_than_p);
}

bool transformer_argument_consistency_p(transformer t)
{
  return transformer_argument_general_consistency_p(t, false);
}

bool transformer_argument_weak_consistency_p(transformer t)
{
  return transformer_argument_general_consistency_p(t, true);
}

bool transformer_argument_general_consistency_p(transformer t, bool is_weak)
{
  list args = transformer_arguments(t);
  bool consistent = true;
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
	pips_internal_error("No value for argument %s in relation basis",
			    entity_name(e));
	consistent = false;
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
bool transformer_consistency_p(transformer t)
{
  return transformer_general_consistency_p(t, false);
}
bool transformers_consistency_p(list tl)
{
  bool consistent_p = true;
  FOREACH(TRANSFORMER, t, tl) {
    consistent_p = consistent_p && transformer_general_consistency_p(t, false);
  }
  return consistent_p;
}

/* Interprocedural transformers do not meet all conditions. */
bool transformer_weak_consistency_p(t)
transformer t;
{
  return transformer_general_consistency_p(t, true);
}

bool transformer_general_consistency_p(transformer tf, bool is_weak)
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
    Psysteme sc = (Psysteme) predicate_system(transformer_relation(tf));
    list args = transformer_arguments(tf);
    bool consistent = true;

    /* The NewGen data structure must be fully defined */
    ifdebug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL)
	consistent = transformer_defined_p(tf);
    else
	consistent = true;
    if(!consistent)
	pips_debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
		   "transformer t is not gen_defined\n");

    /* The predicate must be weakly consistent. Every variable
     * in the constraints must be in the basis (but not the other
     * way round).
     */
    consistent = consistent && sc_weak_consistent_p(sc);
    if(!consistent)
	pips_debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
		   "sc is not weakly consistent\n");

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
		bool aliasing = false;
		const char* emn =entity_module_name(val);
		const char* eln =entity_local_name(val);
		list lt =  args;
		entity e;
		for (lt =  args; lt && !aliasing ;POP(lt))
		{
		    e = ENTITY(CAR(lt));
		    consistent = consistent &&
			(same_string_p(entity_local_name(e), eln) ?
			 same_string_p(entity_module_name(e),emn)
			 : true);
		    aliasing = aliasing && entities_may_conflict_p(e,val);
		}

		if(!consistent)
		    pips_user_warning("different global variable names in "
				      "arguments and basis (%s) \n", eln);
		if (aliasing)
		    pips_internal_error("aliasing between  arguments and basis (%s) ",
					entity_name(val));
	    }

	    /* FI: the next test is not safe because val can be
	     * a global value not recognized in the current
	     * context. old_value_entity_p() returns true or FALSE
	     * or pips_error.
	     *
	     * A general version of this routine is needed...  The
	     * return value of a function is not recognized as a
	     * global value by old_value_entity_p
	     *
	     * old_value_entity_p() is likely to core dump on
	     * interprocedural transformers and preconditions.
	     */
	    if( !storage_return_p(entity_storage(val))
		&& old_value_entity_p(val)) {
		entity var = value_to_variable(val);

		consistent = entity_is_argument_p(var, args);
		if(!consistent) {
		  dump_transformer(tf);
		  pips_debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
			     "Old value of %s in sc but not in arguments of transformer %p\n",
			     entity_name(var), tf);
		}
	    }
	    /* The constant term should not appear in the basis */
	    if(consistent) {
		consistent = consistent && !term_cst(t);
		if(!consistent)
		    pips_debug(TRANSFORMER_CONSISTENCY_P_DEBUG_LEVEL,
			       "TCST in sc basis\n");
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
    FOREACH(ENTITY, e, args) {
	/*
	pips_assert("Argument entity appears in the value mappings",
		    entity_has_values_p(e));
		    */
	if(!entity_has_values_p(e)) {
	  /* Values returned by callees may appear in interprocedural
	     transformers */
	  if(!storage_return_p(entity_storage(e))) {
	    pips_user_warning("No value for argument %s in value mappings\n",
			      entity_name(e));
	    if(!is_weak)
	      consistent = false;
	    }
	}
    }

    if(consistent && !is_weak)
      consistent = transformer_argument_consistency_p(tf);

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
      pips_assert("Argument must be a value", false);
    }
  }, args);

  for(e=b; !BASE_NULLE_P(e); e = vecteur_succ(e)) {
    entity val = (entity) vecteur_var(e);

    if(!(new_value_entity_p(val) || old_value_entity_p(val)
	 || intermediate_value_entity_p(val))) {
      if(!entity_constant_p(val)) {
	pips_user_warning("Variable %s in basis should be an internal value",
			  entity_local_name(val));
	pips_assert("Basis variables must be an internal value", false);
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

/* FI: this function does not end up with a consistent transformer
   because the old value is not added to the basis of sc. Also, the
   variable should be transformed into a new value... See next
   function. */
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

/* FI: like the previous function, but supposed to end up with a
   consistent transformer. When the transformer is empty/unfeasible,
   the variable is not added to conform to rules about standard empty
   transformer: how could a variable be updated by a non-existing
   transition?*/
transformer transformer_add_modified_variable_entity(transformer tf,
						     entity var)
{
  if(!transformer_empty_p(tf)) {
    Psysteme sc =  (Psysteme) predicate_system(transformer_relation(tf));
    Pbase b = sc_base(sc);

    if(entity_has_values_p(var)) {
      entity v_new = entity_to_new_value(var);
      entity v_old = entity_to_old_value(var);

      /* FI: it is not well specifived if the argument should be made
	 of new values or of progtram variables because up to now the
	 two are the same, except when printed out. */
      transformer_arguments(tf) = arguments_add_entity(transformer_arguments(tf), var);
      sc_base(sc) = vect_add_variable(b, (Variable) v_new);
      sc_base(sc) = vect_add_variable(sc_base(sc), (Variable) v_old);
      sc_dimension(sc) = base_dimension(sc_base(sc));
    }
    else
      pips_internal_error("Entity \"%s\" has no values.", entity_name(var));
  }

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
