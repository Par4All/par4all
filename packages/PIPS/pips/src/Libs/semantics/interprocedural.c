/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
/* package semantics
 */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"


#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "misc.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "transformer.h"

#include "semantics.h"

/* What follows has to be updated!!!
 *
 * The SUMMARY_PRECONDITION of a module m is built incrementally when
 * the preconditions of its CALLERS are computed. Each time a call site cs
 * to m is encountered, the precondition for cs is augmented with equations
 * between formal and actual integer arguments. The new precondition so
 * obtained is chained to the other preconditions for m.
 *
 * In other word, SUMMARY_PRECONDITION is a list of preconditions containing
 * each a relation between formal and actual parameter. Should it be called
 * SUMMARY_PRECONDITIONS?
 *
 * Before using the list of preconditions for m, each precondition has to:
 *  - be translated to update references to variables in common; if they
 * are visible in m, their entity names caller:x have to be replaced by
 * m:x;
 *  - be projected to eliminate variables which are not recognized by m
 * semantics analysis (i.e. they do not appear in its value mappings);
 *  - be replaced by their unique convex hull, which is the real module
 * precondition.
 *
 * These steps cannot be performed in the caller because the callee's
 * value mappings are unknown.
 *
 * Memory management:
 *  - transformers stored by add_module_call_site_precondition()
 * are copies or modified copies of the precondition argument ;
 *  - transformers returned as module preconditions are not duplicated;
 * it does not seem to matter for the time being because a module
 * precondition should be used only once. Wait for the strange bugs...
 * db_get_memory_resource() could be not PURE.
 *
 * Note: illegal request are made to pipsdbm()
 */

transformer get_module_precondition(m)
entity m;
{
  transformer p;

  pips_assert("get_module_precondition",entity_module_p(m));

  if(db_resource_p(DBR_SUMMARY_PRECONDITION, module_local_name(m)))
    p = (transformer) db_get_memory_resource(DBR_SUMMARY_PRECONDITION,
					     module_local_name(m),
					     true);
  else
    p = transformer_undefined;

  /* FI: this does not work because the summary preconditions is reset
     each time it should be accumulated */
  /*
    if(check_resource_up_to_date(DBR_SUMMARY_PRECONDITION,
    module_local_name(m))) {
    p = (transformer) db_get_memory_resource(DBR_SUMMARY_PRECONDITION,
    module_local_name(m),
    true);
    }
    else {
    p = transformer_undefined;
    }
  */

  return p;
}

void add_module_call_site_precondition(m, p)
entity m;
transformer p;
{
  /* module precondition */
  transformer mp;
  transformer new_mp;
  /* cons * ef = code_effects(value_code(entity_initial(m))); */
  list ef = load_summary_effects(m);

  pips_assert("add_module_call_site_precondition",entity_module_p(m));
  pips_assert("add_module_call_site_precondition",
	      p != transformer_undefined);

  ifdebug(8) {
    pips_debug(8,"begin\n");
    pips_debug(8,"for module %s\n",
	       module_local_name(m));
    pips_debug(8,"call site precondition %p:\n", p);
    /* p might not be printable; it may (should!) contain formal parameters
       of module m */
    dump_transformer(p);
  }

  /* keep only the interprocedural part of p that can be easily used
     by m; this is non optimal because symbolic constants will be
     lost; this is due to value mappings; new and old values should be
     added to the mapping using the module precondition*/
  p = precondition_intra_to_inter(m, p, ef);

  pips_debug(8, "filtered call site precondition:\n");
  ifdebug(8) dump_transformer(p);

  mp = get_module_precondition(m);

  ifdebug(8) {
    if (!transformer_undefined_p(mp)) {
      pips_debug(8, "old module precondition:\n");
      dump_transformer(mp);
    }
    else
      pips_debug(8, "old module precondition undefined\n");
  }

  translate_global_values(get_current_module_entity(), p);

  pips_debug(8, "new module precondition in current frame:\n");
  ifdebug(8) dump_transformer(p);

  if (!transformer_undefined_p(mp)) {

    /* convert global variables in the summary precondition in the
     * local frame as defined by value mappings (FI, 1 February
     * 1994) */

    /* p is returned in the callee's frame; there is no need for a
     * translation; the caller's frame should always contain the
     * callee's frame by definition of effects;unfortunately, I do not
     * remember *why* I added this translation; it was linked to a
     * problem encountered with transformer and "invisible" variables,
     * i.e. global variables which are indirectly changed by a
     * procedure which does not see them; such variables receive an
     * arbitrary existing global name; they may receive different
     * names in different context, because there is no canonical name;
     * each time, summary_precondition and summary_transformer are
     * used, they must be converted in a unique frame, which can only
     * be the frame of the current module.
     *
     * FI, 9 February 1994
     */
    translate_global_values(get_current_module_entity(), mp);

    pips_debug(8, "old module precondition in current frame:\n");
    ifdebug(8) dump_transformer(mp);

    if(transformer_identity_p(mp)) {
      /* the former precondition represents the entire space :
       * the new precondition must also represent the entire space
       * BC, november 1994.
       */
      transformer_free(p);
      new_mp = mp;
    }
    else
      new_mp = transformer_convex_hull(mp, p);

  }
  else {
    /* the former precondition is undefined. The new precondition
     * is defined by the current call site precondition
     * BC, november 1994.
     */
    new_mp = p;
  }

  pips_debug(8, "new module precondition in current frame:\n");
  ifdebug(8) dump_transformer(new_mp);

  DB_PUT_MEMORY_RESOURCE(DBR_SUMMARY_PRECONDITION,
			 strdup(module_local_name(m)),
			 (char*) new_mp );

  pips_debug(8, "end\n");
}



/* returns a module's parameter's list */

list module_to_formal_analyzable_parameters(entity f)
{
  /* get unsorted list of formal analyzable parameters for f by declaration
     filtering; these parameters may not be used by the callee's
     semantics analysis, but we have no way to know it because
     value mappings are not available */

  list formals = NIL;
  list decl = list_undefined;

  pips_assert("f is a module",entity_module_p(f));

  decl = code_declarations(entity_code(f));
  MAPL(ce, {entity e = ENTITY(CAR(ce));
  if(storage_formal_p(entity_storage(e)) &&
     analyzable_scalar_entity_p(e))
    formals = CONS(ENTITY, e, formals);},
       decl);

  return formals;
}

bool
same_analyzable_type_scalar_entity_list_p(list l)
{
  bool result = true;

  if (!ENDP(l)) {
    entity e1 = ENTITY(CAR(l));
    type t1 = entity_type(e1);
    result = analyzable_scalar_entity_p(e1);

    MAP(ENTITY,el, {
      if (result) {
	type t = entity_type(el);
	result = result && analyzable_scalar_entity_p(el)
	  && type_equal_p(t1, t);
      }
    },
	CDR(l));
  }
  return result;
}

/* add_formal_to_actual_bindings(call c, transformer pre, entity caller):
 *
 * pre := pre  U  {f  = expr }
 *                  i       i
 * for all i such that formal f_i is an analyzable scalar variable and
 * as far as expression expr_i is analyzable and of the same type
 */
transformer add_formal_to_actual_bindings(call c, transformer pre, entity caller)
{
  entity f = call_function(c);
  list pc = call_arguments(c);
  list formals = module_to_formal_analyzable_parameters(f);
  cons * ce;

  pips_debug(6, "begin for call to %s from %s, pre=%p\n",
	     module_local_name(f), module_local_name(caller), pre);
  ifdebug(6) dump_transformer(pre);

  pips_assert("f is a module",
	      entity_module_p(f));
  pips_assert("The precondition pre is defined",
	      pre != transformer_undefined);

  /* let's start a long, long, long MAPL, so long that MAPL is a pain */
  for( ce = formals; !ENDP(ce); POP(ce)) {
    entity fp = ENTITY(CAR(ce));
    int r = formal_offset(storage_formal(entity_storage(fp)));
    expression expr = find_ith_argument(pc, r);

    if(expr == expression_undefined)
      pips_user_error("not enough args in call to %s from %s for formal parameter %s of rank %d\n",
		      module_local_name(f),
		      module_local_name(caller),
		      entity_local_name(fp), r);
    else {
      /* type checking. You already know that fp is a scalar variable */
      type tfp = entity_type(fp);
      basic bfp = variable_basic(type_variable(tfp));
      basic bexpr = basic_of_expression(expr);

      if(!same_basic_p(bfp, bexpr)) {
	pips_user_warning("Type incompatibility\n(%s/%s)\nfor formal parameter %s"
			  " (rank %d)\nin call to %s from %s\n",
			  basic_to_string(bfp), basic_to_string(bexpr),
			  entity_local_name(fp), r,
			  module_local_name(f), module_local_name(caller));
      }

      if(basic_tag(bfp)==basic_tag(bexpr)) {
	/* Do not care about side effects on expressions: this is used to map
	   a caller precondition towards a callee summary precondition. */
	entity fp_new = external_entity_to_new_value(fp);
	entity tmp = make_local_temporary_value_entity(entity_type(fp));
	/* tmp must be used instead of fp_new because fp_new does not
           exist in the caller frame */
	transformer t_expr = any_expression_to_transformer(tmp, expr,
							   transformer_undefined,
							   false);

	t_expr = transformer_safe_value_substitute(t_expr, tmp, fp_new);
	/* Likely memory leak for the initial pre */
	pre = transformer_safe_image_intersection(pre, t_expr);
	free_transformer(t_expr);
       }
      else {
	/* ignore assocation */
	pips_debug(6, "Full type incompatibility (%s/%s) for formal parameter %s (rank %d)"
		   " in call to %s from %s\n"
		   "Association ignored",
		   basic_to_string(bfp), basic_to_string(bexpr),
		   entity_local_name(fp), r, module_local_name(f),
		   module_local_name(caller));
      }
    }
  }

  free_arguments(formals);

  ifdebug(6) {
    pips_debug(6, "new pre=%p\n", pre);
    dump_transformer(pre);
    pips_debug(6, "end for call to %s from %s\n", module_local_name(f),
	       module_local_name(caller));
  }

  return pre;
}

/* Take side effects into account:
 *
 * pre := (t(expr )...(t_expr ))(pre)  U  {f  = expr }
 *               n           1              i       i
 * for all i such that formal f_i is an analyzable scalar variable and
 * as far as expression expr_i is analyzable and of the same type.
 *
 * The algorithmic structure has to be different from the previous one.
 *
 * pre is modified by side effects.
 */
transformer new_add_formal_to_actual_bindings(call c, transformer pre, entity caller)
{
  entity f = call_function(c);
  list args = call_arguments(c);

  transformer tf = any_user_call_site_to_transformer(f, args, pre, NIL);
  transformer new_pre = transformer_apply(tf, pre);

  ifdebug(6) {
    pips_debug(6, "new pre=%p\n", new_pre);
    dump_transformer(new_pre);
    pips_debug(6, "end for call to %s from %s\n", module_local_name(f),
	       module_local_name(caller));
  }

  free_transformer(tf);

  return new_pre;
}

transformer precondition_intra_to_inter(entity callee,
					transformer pre,
					list le) // effect list
{
#define DEBUG_PRECONDITION_INTRA_TO_INTER 1
  list values = NIL;
  list lost_values = NIL;
  list preserved_values = NIL;
  Psysteme r;
  Pbase b;
  cons * ca;

  ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER)
    {
      pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
	    "begin for call to %s\nwith precondition:\n",
	    module_local_name(callee));
      /* precondition cannot be printed because equations linking formal
       * parameters have been added to the real precondition
       */
      dump_transformer(pre);
    }

  r = (Psysteme) predicate_system(transformer_relation(pre));

  /* make sure you do not export a (potentially) meaningless old value */
  for( ca = transformer_arguments(pre); !ENDP(ca); POP(ca) )
    {
      entity e = ENTITY(CAR(ca));
      entity e_old;

      /* Thru DATA statements, old values of other modules may appear */
      if(!same_string_p(entity_module_name(e),
			module_local_name(get_current_module_entity()))) {
	pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
	      "entitiy %s not belonging to module %s\n",
	      entity_name(e),
	      module_local_name(get_current_module_entity()));
      }

      e_old  = entity_to_old_value(e);

      if(base_contains_variable_p(sc_base(r), (Variable) e_old))
	lost_values = arguments_add_entity(lost_values,
					   e_old);
    }

  ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER)
    {
      pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
	    "meaningless old value(s):\n");
      dump_arguments(lost_values);
    }

  /* get rid of old_values */
  pre = transformer_projection_with_redundancy_elimination
    (pre, lost_values, /* sc_elim_redund */ /* no_elim */ sc_safe_normalize);

  gen_free_list(lost_values);


  translate_global_values(callee, pre);

  /* get rid of pre's variables that do not appear in effects le */
  /* we should not have to know about these internal objects, Psysteme
     and Pvecteur! */
  lost_values = NIL;
  r = (Psysteme) predicate_system(transformer_relation(pre));
  for(b = r->base; b != NULL; b = b->succ) {
    entity v = (entity) vecteur_var(b);

    if(!entity_constant_p(v))
      values = arguments_add_entity(values, v);
  }

  /* build a list of values to suppress*/
  if(true || fortran_language_module_p(callee)) {
  /* get rid of variables that are not referenced, directly or indirectly,
     by the callee; translate what you can */
  pips_debug(9, "Module effect list:");
  ifdebug(9) print_effects(le);

  for(ca = values; !ENDP(ca);  POP(ca))    {
    entity e = ENTITY(CAR(ca));
    // FI: concrete effects are now killed by abstract effects
    list l_callee =
      (list) concrete_effects_entities_which_may_conflict_with_scalar_entity(le, e);
    // FI: the association with an abstract effect kills the
    // translation process below
    // list l_callee = (list) effects_entities_which_may_conflict_with_scalar_entity(le, e);
    /* For clarity, all cases are presented */
    if(c_language_module_p(callee) && local_entity_of_module_p(e, callee)) {
      /* No need to substitute or eliminate this value */
      /* This is a short term improvement for partial_eval01-02
	 that works only for values local to the callee not for
	 values updated indirectedly by callees of "callee". */
      /* It resulted in many core dumps in array privatization */
	;
    }
    else if (ENDP(l_callee)) {   /* no conflicts */
      lost_values = arguments_add_entity(lost_values, e);
      pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
	    "value %s lost according to effect list\n",
	    entity_name(e));
    }
    else {
      /* list of conflicting entities */
      entity e_callee = ENTITY(CAR(l_callee));
      /* case 1: only one entity*/
      if (gen_length(l_callee)==1) {
	/*  case 1.1: one conflicting integer entity */
	if (analyzable_scalar_entity_p(e_callee)) {
	  if(e_callee != e) {
	    if(type_equal_p(entity_type(e_callee), entity_type(e))) {
	      pre = transformer_value_substitute(pre,
						 e, e_callee);
	      ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER) {
		pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
		      "value %s substituted by %s according to effect list le:\n",
		      entity_name(e), entity_name(e_callee));
		dump_arguments(lost_values);
	      }
	    }
	    else {
	      /* Type mismatch */
	      lost_values = arguments_add_entity(lost_values, e);
	      pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
		    "value %s lost because non analyzable scalar entity\n",
		    entity_name(e));
	    }
	  }
	}
	/* case 1.22: one conflicting non analyzable scalar entity*/
	else {
	  lost_values = arguments_add_entity(lost_values, e);
	  pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
		"value %s lost because non analyzable scalar entity\n",
		entity_name(e));
	}
      }
      else  { /* case 2: at least 2 conflicting entities */
	if (same_analyzable_type_scalar_entity_list_p(l_callee)) {
	  /* case 2.1: all entities have the same type,
	     according to mapping_values the subtitution
	     is made with the first list element e_callee*/
	  if(e_callee != e) {
	    pre = transformer_value_substitute(pre,
					       e, e_callee);
	    ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER) {
	      pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
		    "value %s substituted by %s the first element list according to effect list le:\n",
		    entity_name(e), entity_name(e_callee));
	      dump_arguments(lost_values);
	    }
	  }
	}

	else { /* case 2.2: all entities do not have the same type*/
	  lost_values = arguments_add_entity(lost_values, e);
	  pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
		"value %s lost - list of conflicting entities with different types\n",
		entity_name(e));
	}
      }
    }

  }
  }
  else if(false && c_language_module_p(callee)) {
    /* Get rid of variables local to the caller */
    ;
  }

  preserved_values = arguments_difference(values, lost_values);

  ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER) {
    pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
	  "values lost because they do not appear in the effect list le:\n");
    dump_arguments(lost_values);
    pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
	  "values preserved because they do appear in the effect list le"
	  " and in the transformer basis:\n");
    dump_arguments(preserved_values);
  }

  /* Get rid of unused or untouched variables, even though they may
   * appear as global variables or formal parameters
   *
   * This happens with automatically generated modules and for routine
   * XERCLT in KIVA (Renault) because it has been emptied.
   *
   */
  if(ENDP(preserved_values)) {
    /* No information but feasibility can be preserved */
    if(transformer_empty_p(pre)) {
      /* Get rid of the basis and arguments to define the empty set */
      free_transformer(pre);
      pre = transformer_empty();
    }
    else{
      /* No information: the all value space is OK */
      free_transformer(pre);
      pre = transformer_identity();
    }
  }
  else {
    pre = transformer_projection_with_redundancy_elimination
      (pre, lost_values, /* sc_elim_redund */ /* no_elim */ sc_safe_normalize);
  }

  /* free the temporary list of entities */
  gen_free_list(preserved_values);
  gen_free_list(lost_values);
  gen_free_list(values);

  /* get rid of arguments because they are meaningless for
     a module precondition: v_new == v_old by definition */
  gen_free_list(transformer_arguments(pre));
  transformer_arguments(pre) = NIL;

  ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER) {
    pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
	       "return pre=%p\n",pre);
    dump_transformer(pre);
    pips_debug(DEBUG_PRECONDITION_INTRA_TO_INTER, "end\n");
  }

  return pre;
}

void translate_global_values(entity m, transformer tf)
{
    Psysteme s = (Psysteme) predicate_system(transformer_relation(tf));
    /* a copy of sc_base(s) is needed because translate_global_value()
       modifies it at the same time */
    Pbase b = (Pbase) vect_dup(sc_base(s));
    Pbase bv;

    ifdebug(6) {
	pips_debug(6, "Predicate for tf:\n");
	sc_fprint(stderr, s, (char * (*)(Variable)) dump_value_name);
    }

    for(bv = b; bv != NULL; bv = bv->succ) {
	translate_global_value(m, tf, (entity) vecteur_var(bv));

    }

    base_rm(b);
}

/* Try to convert an value on a non-local variable into an value
 * on a local variable using a guessed name (instead of a location
 * identity: M and N declared as COMMON/FOO/M and COMMON/FOO/N
 * are not identified as a unique variable/location).
 *
 * Mo more true: It might also fail to translate variable C:M into A:M if C is
 * indirectly called from A thru B and if M is not defined in B.
 *
 * This routine is not too safe. It accepts non-translatable variable
 * as input and does not refuse them, most of the time.
 */
void translate_global_value(m, tf, v)
entity m;
transformer tf;
entity v;
{
  storage store = storage_undefined;
  ram r = ram_undefined;
  entity rf = entity_undefined;
  entity section = entity_undefined;

  ifdebug(7) {
    pips_debug(7, "begin v = %s and tf = %p\n", entity_name(v), tf);
  }


  if(v == NULL) {
    pips_internal_error("Trying to translate TCST");
    return;
  }

  /* Filter out constant and values *local* to the current module */
  if(value_entity_p(v) || entity_constant_p(v)) {
    /* FI: to be modified to account for global values that have a name
     * but that should nevertheless be translated on their canonical
     * representant; this occurs for non-visible global variables
     */
    /* FI: to be completed later... 3 December 1993
       entity var = value_to_variable(v);

       pips_debug(7,
       "%s is translated into %s\n",
       entity_name(v), entity_name(e));
       transformer_value_substitute(tf, v, e);
    */

    pips_debug(7, "end: No need to translate %s\n", entity_name(v));
    return;
  }

  /* Filter out old values: they are translated when the new value is
   * encountered, and the new value has to appear if the old value does
   * appear.
   *
   * Instead, old values could be translated into new values and processing
   * could go on...
   *
   * FI, 26 October 1994
   */
  if(global_old_value_p(v)) {
    pips_debug(7, "end: No need to translate %s yet\n",
	       entity_name(v));
    return;
  }

  store = entity_storage(v);

  pips_debug(7, "Trying to translate %s\n", entity_name(v));

  if(!storage_ram_p(store)) {
    if(storage_rom_p(store)) {
      pips_debug(7, "%s is not translatable: store tag %d\n",
		 entity_name(v), storage_tag(store));
      /* Should it be projected? No, this should occur later for
       * xxxx#init variables when the xxxx is translated. Or before if
       * xxxx has been translated
       */
      return;
    }
    else
      if(storage_formal_p(store)) {
	pips_debug(7, "formal %s is not translatable\n",
		   entity_name(v));
	return;
      }
    if(storage_return_p(store)) {
      pips_debug(7, "return %s is not translatable\n",
		 entity_name(v));
      return;
    }
    else
      pips_internal_error("%s is not translatable: store tag %d",
			  entity_name(v), storage_tag(store));
  }

  ifdebug(7) {
    pips_debug(7, "let's do it for v = %s and tf =\n",
	       entity_name(v));
    dump_transformer(tf);
  }

  r = storage_ram(store);
  rf = ram_function(r);
  section = ram_section(r);

  if(rf != m && top_level_entity_p(section)) {
    /* must be a common; dynamic and static area must
       have been filtered out before */
    entity e;
    entity v_init = entity_undefined;
    Psysteme sc = SC_UNDEFINED;
    Pbase b = BASE_UNDEFINED;

    if(top_level_entity_p(v)) {
      // No need to translate
      return;
    }

    /* try to find an equivalent entity by its name
       (whereas we should use locations) */
    /*
      e = FindEntity(module_local_name(m),
      entity_local_name(v));
      e = value_alias(value_to_variable(v));
    */
    e = value_alias(v);
    if(e == entity_undefined) {
      /* no equivalent name found, get rid of v */
      pips_debug(7, "No equivalent for %s in %s: project %s\n",
		 entity_name(v), entity_name(m), entity_name(v));
      user_warning("translate_global_value",
		   "Information about %s lost,\n"
		   "check structure of common /%s/ in modules %s and %s\n",
		   entity_name(v), module_local_name(section), entity_module_name(v),
		   module_local_name(m));
      if(entity_is_argument_p(v, transformer_arguments(tf))) {
	entity v_old = global_new_value_to_global_old_value(v);
	/* transformer_projection(tf, CONS(ENTITY, v_old, NIL)); */
	(void) transformer_filter(tf, CONS(ENTITY, v_old, NIL));
      }
      transformer_projection(tf, CONS(ENTITY, v, NIL));
      return;
    }

    if(!same_scalar_location_p(v, e)) {
      /* no equivalent location found, get rid of v */
      pips_debug(7, "No equivalent location for %s and %s: project %s\n",
		 entity_name(v), entity_name(e), entity_name(v));
      transformer_projection(tf, CONS(ENTITY, v, NIL));
      user_warning("translate_global_value",
		   "Information about %s lost,\n"
		   "check structure of common /%s/ in modules %s and %s\n",
		   entity_name(v), entity_local_name(section), entity_module_name(v),
		   module_local_name(m));
      if(entity_is_argument_p(v, transformer_arguments(tf))) {
	entity v_old = global_new_value_to_global_old_value(v);
	transformer_projection(tf, CONS(ENTITY, v_old, NIL));
      }
      transformer_projection(tf, CONS(ENTITY, v, NIL));
      return;
    }

    if(!type_equal_p(entity_type(v), entity_type(e))) {
      /* no equivalent location found, get rid of v */
      pips_debug(7, "Same location but different types for %s (%s) and %s (%s):"
		 " project both %s and %s\n",
		 entity_name(v), type_to_string(entity_type(v)),
		 entity_name(e), type_to_string(entity_type(e)),
		 entity_name(e), entity_name(v));
      transformer_projection(tf, CONS(ENTITY, v, NIL));
      user_warning("translate_global_value",
		   "Information about %s lost,\n"
		   "check types for variables in common /%s/ in modules %s and %s\n",
		   entity_name(v), entity_module_name(section), entity_module_name(v),
		   module_local_name(m));
      if(entity_is_argument_p(v, transformer_arguments(tf))) {
	entity v_old = global_new_value_to_global_old_value(v);
	transformer_projection(tf, CONS(ENTITY, v_old, NIL));
      }
      if(entity_is_argument_p(e, transformer_arguments(tf))) {
	entity e_old = global_new_value_to_global_old_value(e);
	transformer_filter(tf, CONS(ENTITY, e_old, NIL));
      }
      transformer_filter(tf, CONS(ENTITY, v, NIL));
      transformer_filter(tf, CONS(ENTITY, e, NIL));
      return;
     }

    sc = (Psysteme)
      predicate_system(transformer_relation(tf));
    b = sc_base(sc);
    if(base_contains_variable_p(b, (Variable) e)) {
      /* e has already been introduced and v eliminated;
	 this happens when a COMMON variable is
	 also passed as real argument */
      /* FI: v may still appear in the constraints as in spice.f
	 (Perfect Club) and spice01.f (Validation) */
      Pvecteur subst = vect_new((Variable) v, (Value) 1);
      Pcontrainte eq = CONTRAINTE_UNDEFINED;
      list args = CONS(ENTITY, v, NIL);

      vect_add_elem(&subst, (Variable) e, (Value) -1);
      eq = contrainte_make(subst);
      sc_add_egalite(sc, eq);

      if(entity_is_argument_p(v, transformer_arguments(tf))) {
	entity v_old = global_new_value_to_global_old_value(v);
	entity e_old = global_new_value_to_global_old_value(e);
	Pvecteur subst_old = vect_new((Variable) v_old, (Value) 1);
	Pcontrainte eq_old = CONTRAINTE_UNDEFINED;

       args = CONS(ENTITY, v_old, args);

       vect_add_elem(&subst_old, (Variable) e_old, (Value) -1);
       eq_old = contrainte_make(subst_old);
       sc_equation_add(sc, eq_old);
      }

      pips_debug(7, "%s has already been translated into %s\n",
		 entity_name(v), entity_name(e));
      if(!language_c_p(module_language(m))) {
	user_warning("translate_global_value",
		     "Variable %s is probably aliased with a formal parameter"
		     " by the current call to %s from %s.\n"
		     "This is forbidden by the Fortran 77 standard.\n",
		     entity_name(v), entity_module_name(v), module_local_name(m));
      }
      ifdebug(7) {
	pips_debug(7,
		   "%s should again be translated into %s by projection of %s\n",
		   entity_name(v), entity_name(e), entity_name(v));
	dump_transformer(tf);
      }

      if(entity_is_argument_p(v, transformer_arguments(tf))) {
	transformer_arguments(tf) =
	  arguments_add_entity(transformer_arguments(tf), e);
      }

      tf = transformer_projection(tf, args);
      gen_free_list(args);

      ifdebug(7) {
	pips_debug(7, "After projection of %s\n",
		   entity_name(v));
	dump_transformer(tf);
      }
    }
    else {
      pips_debug(7, "%s is translated into %s\n",
		 entity_name(v), entity_name(e));
      transformer_value_substitute(tf, v, e);
    }

    v_init = (entity)
      gen_find_tabulated(concatenate(entity_name(v),
				     OLD_VALUE_SUFFIX,
				     (char *) NULL),
			 entity_domain);
    if(v_init != entity_undefined) {
      entity e_init = (entity)
	gen_find_tabulated(concatenate(entity_name(e),
				       OLD_VALUE_SUFFIX,
				       (char *) NULL),
			   entity_domain);
      if(e_init == entity_undefined) {
	/* this cannot happen when the summary transformer
	   of a called procedure is translated because
	   the write effect in the callee that is implied
	   by v_init existence must have been passed
	   upwards and must have led to the creation
	   of e_init */
	/* this should not happen when a caller
	   precondition at a call site is transformed
	   into a piece of a summary precondition for
	   the callee because v_init becomes meaningless;
	   at the callee's entry point, by definition,
	   e == e_init; v_init should have been projected
	   before
	*/
	Psysteme r =
	  (Psysteme) predicate_system(transformer_relation(tf));

	if(base_contains_variable_p(sc_base(r), (Variable) v_init))
	  pips_internal_error("Cannot find value %s",
		     strdup(
			    concatenate(
					module_local_name(m),
					MODULE_SEP_STRING,
					entity_local_name(v),
					OLD_VALUE_SUFFIX,
					(char *) NULL)));
	else {
	  /* forget e_init: there is no v_init in tf */
	  ;
	  pips_debug(7, "%s is not used in tf\n",
		     entity_name(v_init));
	}
      }
      else {
	pips_debug(7, "%s is translated into %s\n",
		   entity_name(v), entity_name(e));
	if(transformer_value_substitutable_p(tf, v_init, e_init))
	  transformer_value_substitute(tf, v_init, e_init);
	else {
	  pips_user_error("Unsupported aliasing linked to %s and %s\n",
			  entity_name(v_init), entity_name(e_init));
	}
      }
    }
    else {
      /* there is no v_init to worry about; v is not changed in
	 the caller (or its subtree of callees) */
    }
  }
  else {
    /* this value does not need to be translated */
  }
}

void expressions_to_summary_precondition(pre, le)
transformer pre;
list le;
{
  MAPL(ce, {
    expression e = EXPRESSION(CAR(ce));
    expression_to_summary_precondition(pre, e);
  },
       le)
    }

void expression_to_summary_precondition(pre, e)
transformer pre;
expression e;
{
  syntax s = expression_syntax(e);

  if(syntax_call_p(s)) {
    call c = syntax_call(s);
    call_to_summary_precondition(pre, c);
  }
}

void call_to_summary_precondition(pre, c)
transformer pre;
call c;
{
  entity e = call_function(c);
  tag tt;
  list args = call_arguments(c);
  transformer pre_callee = transformer_undefined;

  pips_debug(8, "begin\n");

  switch (tt = value_tag(entity_initial(e))) {

  case is_value_intrinsic:
    pips_debug(5, "intrinsic function %s\n",
	       entity_name(e));
    /* propagate precondition pre as summary precondition
       of user functions */
    expressions_to_summary_precondition(pre, args);
    break;

  case is_value_code:
    pips_debug(5, "external function %s\n",
	       entity_name(e));
    pre_callee = transformer_dup(pre);
    pre_callee =
      add_formal_to_actual_bindings(c, pre_callee, get_current_module_entity());
    add_module_call_site_precondition(e, pre_callee);
    /* propagate precondition pre as summary precondition
       of user functions */
    expressions_to_summary_precondition(pre, args);
    break;

  case is_value_symbolic:
    /* user_warning("call_to_summary_precondition",
       "call to symbolic %s\n",
       entity_name(e)); */
    break;

  case is_value_constant:
    break;
    user_warning("call_to_summary_precondition",
		 "call to constant %s\n",
		 entity_name(e));

  case is_value_unknown:
    pips_internal_error("unknown function %s",
	       entity_name(e));
    break;

  default:
    pips_internal_error("unknown tag %d", tt);
  }

  pips_debug(8, "end\n");

}

/* This function does everything needed.
 * Called by ICFG with many different contexts.
 */
text
call_site_to_module_precondition_text(
    entity caller,
    entity callee,
    statement s,
    call c)
{
  text result;
  /* summary effects for the callee */
  list seffects_callee = load_summary_effects(callee);
  /* caller preconditions */
  transformer caller_prec = transformer_undefined;
  /* callee preconditions */
  transformer call_site_prec = transformer_undefined;

  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource
			   (DBR_CUMULATED_EFFECTS,
			    module_local_name(caller), true));

  set_semantic_map((statement_mapping)
		   db_get_memory_resource
		   (DBR_PRECONDITIONS,
		    module_local_name(caller),
		    true) );

  /* load caller preconditions */
  caller_prec = transformer_dup(load_statement_semantic(s));

  set_current_module_statement(s);

  /* first, we deal with the caller */
  set_current_module_entity(caller);
  /* create htable for old_values ... */
  module_to_value_mappings(caller);

  /* add to preconditions the links to the callee formal params */
  caller_prec = add_formal_to_actual_bindings (c, caller_prec, caller);
  /* transform the preconditions to make sense for the callee */
  call_site_prec = precondition_intra_to_inter (callee,
						caller_prec,
						seffects_callee);
  call_site_prec = transformer_normalize(call_site_prec, 2);

  /* translate_global_values(e_caller, call_site_prec); */
  free_value_mappings();
  reset_current_module_entity();

  /* Now deal with the callee */
  set_current_module_entity(callee);
  /* Set the htable with its variables because now we work
     in this frame */
  module_to_value_mappings(callee);

  result = text_for_a_transformer(call_site_prec, false);

  reset_current_module_entity();
  reset_current_module_statement();
  reset_cumulated_rw_effects();
  reset_semantic_map();
  free_value_mappings();

  return result;
}

/* Context to compute summary preconditions */

static entity current_caller = entity_undefined;
static transformer current_precondition = transformer_undefined;

static entity current_callee = entity_undefined;
static list summary_effects_of_callee = list_undefined;
static transformer current_summary_precondition = transformer_undefined;
static int number_of_call_sites = -1;

void reset_call_site_number()
{
    number_of_call_sites = 0;
}

int get_call_site_number()
{
    return number_of_call_sites;
}

/* Each time a statement is entered, its precondition is memorized*/
static bool memorize_precondition_for_summary_precondition(statement s)
{
  bool go_down = false;

  if(statement_ordering(s)!=STATEMENT_ORDERING_UNDEFINED) {
    /* There may be or not a precondition associated to
     * this statement, depending on an intermediate disk
     * storage. However, if the ordering of statement s
     * is undefined, statement s is not reachable and the
     * corresponding call site should be ignored, as well
     * as call sites in statements controlled by s
     */
    /* FI: probably a memory leak is started here */
    current_precondition = transformer_range(load_statement_semantic(s));

    pips_assert("current precondition is defined",
		current_precondition!=transformer_undefined);

    go_down = true;
  }
  else {
    current_precondition = transformer_undefined;
    go_down = false;
  }

  if(go_down && declaration_statement_p(s)) {
    list dl = statement_declarations(s);
    void update_summary_precondition_in_declaration(expression e, transformer pre);
    current_precondition =
      propagate_preconditions_in_declarations(dl, current_precondition, update_summary_precondition_in_declaration);
  }

  return go_down;
}

/* Update the current_summary_precondition, if necessary */
static bool process_call_for_summary_precondition(call c)
{
#define PROCESS_CALL_DEBUG_LEVEL 5

  transformer caller_prec = transformer_undefined;
  transformer call_site_prec = transformer_undefined;

  if(call_function(c) != current_callee) {
    return true;
  }


  number_of_call_sites++;

  ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
    pips_debug(PROCESS_CALL_DEBUG_LEVEL,
	       "Begin for module %s with %d call sites with caller %s\n",
	       module_local_name(current_callee), number_of_call_sites,
	       module_local_name(current_caller));
    pips_debug(PROCESS_CALL_DEBUG_LEVEL,
	       "call site precondition %p:\n", current_precondition);
    /* p might not be printable; it may (should!) contain formal parameters
       of module m */
    dump_transformer(current_precondition);
    pips_debug(PROCESS_CALL_DEBUG_LEVEL,
	       "current summary precondition for callee %s, %p:\n",
	       module_local_name(current_callee),
	       current_summary_precondition);
    /* p might not be printable; it may (should!) contain formal parameters
       of module m */
    dump_transformer(current_summary_precondition);
  }

  /* add to call site preconditions the links to the callee formal params */
  caller_prec = new_add_formal_to_actual_bindings
    (c, transformer_dup(current_precondition), current_caller);
  ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
    pips_debug(PROCESS_CALL_DEBUG_LEVEL,
	       "call site precondition in caller %s with bindings %p:\n",
	       module_local_name(current_caller),
	       caller_prec);
    /* caller_prec should not be printable; it should contain
     * formal parameters of module callee
     */
    dump_transformer(caller_prec);
  }

  /* transform the preconditions to make sense for the callee */
  /* Beware: call_site_prec and caller_prec are synonymous */
  call_site_prec =
    precondition_intra_to_inter(current_callee,
				caller_prec,
				summary_effects_of_callee);

  ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
    pips_debug(PROCESS_CALL_DEBUG_LEVEL,
	       "call site precondition with filtered actual parameters:\n");
    dump_transformer(call_site_prec);
  }

  translate_global_values(current_caller, call_site_prec);

  ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
    pips_debug(PROCESS_CALL_DEBUG_LEVEL,
	  "new call site precondition in caller's frame:\n");
    dump_transformer(call_site_prec);
  }

  /* Provoque initialization with an undefined transformer... */
  /*pips_assert("process_call", !transformer_undefined_p(call_site_prec)); */

  if (!transformer_undefined_p(current_summary_precondition)) {

    /* convert global variables in the summary precondition in the
     * caller's frame as defined by value mappings (FI, 1 February 1994)
     */

    /* p is returned in the callee's frame; there is no need for a
     * translation; the caller's frame should always contain the
     * callee's frame by definition of effects;
     *
     * Unfortunately, I do not remember *why* I added this
     * translation; It was linked to a problem encountered with
     * transformer and "invisible" variables, i.e. global variables
     * which are indirectly changed by a procedure which does not see
     * them; such variables receive an arbitrary existing global name;
     * they may receive different names in different context, because
     * there is no canonical name; each time, summary_precondition and
     * summary_transformer are used, they must be converted in a
     * unique frame, which can only be the frame of the current
     * module.  In other words, you have to be in the same environment
     * to be allowed to combine preconditions.
     *
     * FI, 9 February 1994
     *
     * This may be now useless...
     */
    translate_global_values(current_caller,
			    current_summary_precondition);
    ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
      pips_debug(PROCESS_CALL_DEBUG_LEVEL,
	    "old module current summary precondition (%p) in current frame:\n",
	    current_summary_precondition);
      dump_transformer(current_summary_precondition);
    }

    if(transformer_identity_p(current_summary_precondition)) {
      /* the former precondition represents the entire space :
       * the new precondition must also represent the entire space
       * BC, november 1994.
       */
      transformer_free(call_site_prec);
    }
    else {
      transformer new_current_summary_precondition =
	transformer_undefined;
      pips_assert("A new transformer is allocated",
		  current_summary_precondition != call_site_prec);
      new_current_summary_precondition =
	transformer_convex_hull(current_summary_precondition,
				call_site_prec);
      transformer_free(current_summary_precondition);
      current_summary_precondition = new_current_summary_precondition;

    }
  }
  else {
    /* the former precondition is undefined. The new precondition
     * is defined by the current call site precondition
     * BC, november 1994.
     */
    current_summary_precondition = call_site_prec;
  }

  ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
    pips_debug(PROCESS_CALL_DEBUG_LEVEL,
	  "new current summary precondition for module %s in current frame, %p:\n",
	  module_local_name(current_caller),
	  current_summary_precondition);
    dump_transformer(current_summary_precondition);
  }

  /* FI: Let's put the summary_precondition in the callee's frame..
   * Well, it's an illusion because translate_global_values() is
   * not symmetrical. It only can import global values. The summary
   * precondition is left in the last caller's frame. It will
   * have to be translated in callee's frame when used.
   */
  /*
    translate_global_values(current_callee,
    current_summary_precondition);
  */

  return true;
}

#if 0
/* Update the current_summary_precondition, if necessary, for call
   located in the dimension declarations. May be useless because of
   function below... */
static bool process_statement_for_summary_precondition(statement s)
{
  bool ret_p = true;
  pips_internal_error("Not implemented. Should not be called.");
  if(declaration_statement_p(s)) {
    /* Look for call sites in the declarations, but see functions below... */
    //list dl = statement_declarations(s);
    //ret_p = process_call_for_summary_precondition();
  }
  return ret_p;
}
#endif

/* This function is called to deal with call sites located in
   initialization expressions carried by declarations. */
void update_summary_precondition_in_declaration(expression e,
						transformer pre)
{
  current_precondition = pre;
  gen_recurse(e,
	      call_domain,
	      (bool (*)(void *)) process_call_for_summary_precondition,
	      gen_null);
}

/* Update precondition t for callee with preconditions of call sites
   to callee in caller. Call sites are found in the statement of
   caller, but also in its declarations. Return the updated
   precondition t.
*/
transformer update_precondition_with_call_site_preconditions(transformer t,
							     entity caller,
							     entity callee)
{
  statement caller_statement = (statement) db_get_memory_resource
    (DBR_CODE, module_local_name(caller), true);
  /* summary effects for the callee */
  summary_effects_of_callee = load_summary_effects(callee);

  pips_assert("callee is the current module",
	      get_current_module_entity() == callee);

  entity old_entity = get_current_module_entity();
  statement old_statement = get_current_module_statement();

  reset_current_module_entity();
  reset_current_module_statement();

  set_current_module_entity(caller);
  set_current_module_statement(caller_statement);
  current_summary_precondition = t;
  current_caller = caller;
  current_callee = callee;

  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource
			   (DBR_CUMULATED_EFFECTS,
			    module_local_name(caller), true));

  set_semantic_map((statement_mapping)
		   db_get_memory_resource
		   (DBR_PRECONDITIONS,
		    module_local_name(caller),
		    true) );

  module_to_value_mappings(caller);

  /* calls hidden in dimension declarations are not caught because
     entities are not traversed by gen_recurse(). */
  gen_multi_recurse(caller_statement,
		    statement_domain, memorize_precondition_for_summary_precondition, gen_null,
		    call_domain, process_call_for_summary_precondition, gen_null,
		    // FI: to be checked. Should be useless.
		    //statement_domain, process_declaration_for_summary_precondition, gen_null,
		    NULL);

  free_value_mappings();
  reset_current_module_entity();
  reset_current_module_statement();
  reset_cumulated_rw_effects();
  reset_semantic_map();
  set_current_module_entity(old_entity);
  set_current_module_statement(old_statement);

  current_caller = entity_undefined;
  current_callee = entity_undefined;
  current_precondition = transformer_undefined;
  summary_effects_of_callee = list_undefined;

  if( ! transformer_defined_p(current_summary_precondition)) {
    // Problem: the initializer pass does not notify pipsmake that the
    // module list, a.k.a. %ALL, has been changed.
    // I do not know how to fix pipsmake. Add a new exception to
    // restart the computation of the requested resource list? How can
    // we combine this one with the pips_user_error() exception?
    pips_user_error("No summary precondition for module \"%s\". The callgraph "
		    "is probably broken because stubs have been generated."
		    "Please (re)compute the call graph,"
		    " e.g. display CALLGRAPH_FILE.\n",
		    entity_name(caller));
  }

  /* This normalization seems pretty uneffective for fraer01.tpips */
  t = transformer_normalize(current_summary_precondition, 4);
  current_summary_precondition = transformer_undefined;

  return t;
}

/* With value passing, writes on formal parameters are not effective
 * interprocedurally.
 *
 * All new values corresponding to formal arguments of f must be
 * projected out and removed from the arguments list.
 *
 * Performed by side-effect on tf.
*/
transformer value_passing_summary_transformer(entity f, transformer tf)
{
  list al = transformer_arguments(tf); // argument list
  list mfl = NIL; // modified formal list
  list omfl = NIL; // modified formal list
  list tvl = NIL; // temporary value list
  list ctvl = NIL; // current pointer in tvl

  FOREACH(ENTITY, a, al) {
    storage s = entity_storage(a);

    if(storage_formal_p(s)) {
      formal fs = storage_formal(s);
      if(formal_function(fs)==f) {
	entity nav = entity_to_new_value(a);
	entity oav = entity_to_old_value(a);
	mfl = CONS(ENTITY, nav, mfl);
	omfl = CONS(ENTITY, oav, omfl);
      }
    }
  }

  /* The old values cannot be renamed directly after projection,
     because the transformer projection opearator detects an
     inconsistency. */

  /* Rename old values as temporary values in the caller frame. */
  FOREACH(ENTITY, oav, omfl) {
    entity tv = make_local_temporary_value_entity(ultimate_type(entity_type(oav)));

    tvl = CONS(ENTITY, tv, tvl);
    tf = transformer_value_substitute(tf, oav, tv);
  }

  /* Updates the argument list after the projections */
  tf = transformer_projection(tf, mfl);

  /* Rename tmp values as new values in the caller frame. */
  tvl = gen_nreverse(tvl);
  ctvl = tvl;
  /* oav renamed oav1 because of FOREACH macro implementation */
  FOREACH(ENTITY, oav1, omfl) {
    entity tv = ENTITY(CAR(ctvl));
    entity v = value_to_variable(oav1);
    entity nav = entity_to_new_value(v);

    tf = transformer_value_substitute(tf, tv, nav);
    POP(ctvl);
  }

  gen_free_list(tvl);

  return tf;
}

