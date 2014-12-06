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
/* package simple effects :  Be'atrice Creusillet 5/97
 *
 * File: interprocedural.c
 * ~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains functions for the interprocedural computation of simple
 * effects.
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "text.h"

#include "misc.h"
#include "properties.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"

#include "effects-generic.h"
#include "effects-simple.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


/*********************************************************** INITIALIZATION */

void simple_effects_translation_init(entity __attribute__((unused)) callee,
				     list __attribute__((unused)) real_args,
				     bool __attribute__((unused)) backward_p)
{
}

void simple_effects_translation_end()
{
}


void simple_effect_descriptor_interprocedural_translation(effect __attribute__((unused))eff)
{
}

/************************************************************** INTERFACES  */




list /* of effects */
simple_effects_backward_translation(
    entity func,
    list real_args,
    list l_eff,
    transformer context __attribute__ ((unused)))
{
    return summary_to_proper_effects(func, real_args, l_eff);
}

/* To replace the make_sdfi_macro() and its call to macro make_simple_effect() */
effect translate_effect_to_sdfi_effect(effect eff)
{
  // functions that can be pointed by effect_dup_func:
  // simple_effect_dup
  // region_dup
  // copy_effect
  effect ge = (*effect_dup_func)(eff);
  reference ger = effect_any_reference(ge);
  //entity v = reference_variable(ger);
  //type ut = ultimate_type(entity_type(v));

  /* FI: I want to change this function and preserve indices when they
     are constant or unbounded. This is needed to extend the effect
     semantics and to analyze array elements and structure fields and
     pointer dereferencing. A new property would probably be needed to
     avoid a disaster, at least with the validation suite... althoug
     we might be safe with Fortran code...*/

  /*if(reference_indices(ger) != NIL) { */
  /* FI: could be rewritten like the other summarization routines,
     with a check for each subscript expression */
  if(!reference_with_constant_indices_p(ger) && !reference_with_unbounded_indices_p(ger))
  {
    list le = effect_to_sdfi_list(ge);
    ge = EFFECT(CAR(le));
  }


  return ge;
}

/****************************************************************************/

/* list effects_dynamic_elim(list l_reg)
 *
 * input    : a list of effects.
 *
 * output   : a list of effects in which effects of dynamic variables
 *            are removed, and in which dynamic integer scalar
 *            variables are eliminated from the predicate. If
 *            parameters are passed by value, direct effects on scalar
 *            are also removed.
 *
 * modifies : nothing for direct effects; the effects l_reg initially
 *            contains are copied if necessary. Indirect effects may
 *            have to be changed into anywhere effects.
 *
 * comment : this procedure is used to generate summary effects. It is
 *            related to the procedure used to summarize the effects
 *            of a block.
 *
 */
list effects_dynamic_elim(list l_eff)
{
  list l_res = NIL;
  list c_eff = list_undefined;
  bool add_anywhere_write_effect_p = false;
  bool add_anywhere_read_effect_p = false;
  bool value_passing_p = c_module_p(get_current_module_entity());

  for(c_eff = l_eff; !ENDP(c_eff); POP(c_eff)) {
    effect eff = EFFECT(CAR(c_eff));

    entity eff_ent = effect_entity(eff);
    storage eff_s = entity_storage(eff_ent);
    bool ignore_this_effect = false;

    ifdebug(4) {
      pips_debug(4, "current effect for entity \"\%s\":\n",
		 entity_name(eff_ent));
      print_effect(eff);
    }

    if(!anywhere_effect_p(eff)) {
      /* If the reference is a common variable (ie. with storage ram but
       * not dynamic) or a formal parameter, the effect is not ignored.
       */
      switch (storage_tag(eff_s)) {
      case is_storage_return:
	pips_debug(5, "return var ignored (%s)\n", entity_name(eff_ent));
	ignore_this_effect = true;
	break;
      case is_storage_ram:
	{
	  ram r = storage_ram(eff_s);
	  /* FI: heap areas effects should be preserved... */
	  if (dynamic_area_p(ram_section(r)) || heap_area_p(ram_section(r))
	      || stack_area_p(ram_section(r))) {
	    type ut = ultimate_type(entity_type(eff_ent));
	    list sl = reference_indices(effect_any_reference(eff));

	    if(pointer_type_p(ut))
	      if(!ENDP(sl)) {
		/* Can we convert this effect using the pointer
		   initial value? */
		/* FI: this should rather be done after a constant
		   pointer analysis but I'd like to improve quickly
		   results with Effects/fulguro01.c */
		value v = entity_initial(eff_ent);

		if(value_expression_p(v) && !self_initialization_p(eff_ent)) {
		  expression ae = value_expression(v);
		  /* Save the action before the effect may be changed */
		  list nel, fel;

		  /* re-use an existing function... */
		  nel = c_summary_effect_to_proper_effects(eff, ae);
		  /* Should this effect be preserved? */
		  /* Let's hope we do not loop recursively... */
		  fel = effects_dynamic_elim(nel);

		  if(ENDP(fel)) {
		    ignore_this_effect = true;
		  }
		  else
		    {
		      /* c_summary_to_proper_effects can return
			 several effects, but initially it was
			 designed to return a single effect. I have to
			 rebuild effects_dynamic_elim to take that
			 into account. BC.
		      */
		      eff= EFFECT(CAR(fel));
		    }
		  gen_free_list(nel);
		  gen_free_list(fel);

		}
		else {
		  action ac = effect_action(eff);
		  pips_debug(5, "Local pointer \"%s\" is not initialized!\n",
			     entity_name(eff_ent));
		  if(action_write_p(ac))
		    add_anywhere_write_effect_p = true;
		  else
		    add_anywhere_read_effect_p = true;
		  ignore_this_effect = true;
		}
	      }
	      else {
		pips_debug(5, "Local pointer \"%s\" can be ignored\n",
			   entity_name(eff_ent));
		ignore_this_effect = true;
	      }
	    else {
	      pips_debug(5, "dynamic or pointed var ignored (%s)\n",
			 entity_name(eff_ent));
	      ignore_this_effect = true;
	    }
	  }
	  break;
	}
      case is_storage_formal:
	if(value_passing_p) {
	  reference r = effect_any_reference(eff);
	  list inds = reference_indices(r);
	  action ac = effect_action(eff);
	  if(action_write_p(ac) && ENDP(inds))
	    ignore_this_effect = true;
	}
	break;
      case is_storage_rom:
	if(!entity_special_area_p(eff_ent) && !anywhere_effect_p(eff))
	  ignore_this_effect = true;
	break;
	/*  pips_internal_error("bad tag for %s (rom)",
	    entity_name(eff_ent));*/
      default:
	pips_internal_error("case default reached");
      }
    }

    if (! ignore_this_effect)  /* Eliminate dynamic variables. */ {
      /* FI: this macro is not flexible enough */
      /* effect eff_res = make_sdfi_effect(eff); */
      effect eff_res = translate_effect_to_sdfi_effect(eff);
      ifdebug(4) {
	pips_debug(4, "effect preserved for variable \"\%s\": \n",
		   entity_name(effect_variable(eff_res)));
	print_effect(eff_res);
      }
      l_res = CONS(EFFECT, eff_res, l_res);
    }
    else ifdebug(4) {
	pips_debug(4, "effect removed for variable \"\%s\": \n\t %s\n",
		   entity_name(effect_variable(eff)),
		   words_to_string(words_effect(eff)));
      }
  }

  if(add_anywhere_write_effect_p)
    l_res = CONS(EFFECT, make_anywhere_effect(make_action_write_memory()), l_res);
  if(add_anywhere_read_effect_p)
    l_res = CONS(EFFECT, make_anywhere_effect(make_action_read_memory()), l_res);

  l_res = gen_nreverse(l_res);
  return(l_res);
}


/*
  returns a linear expression giving the size of the n dimension sub-array
  of array a
  returns NULL if size of array a is not a linear expression.
*/
static Pvecteur
size_of_array(entity a,int n)
{
    Pvecteur size_nm1, size_n;

    if (n == 0)
        return(vect_new(TCST, VALUE_ONE));

    if ((size_nm1 = size_of_array(a, n-1)) != (Pvecteur) NULL)
    {
        dimension d = entity_ith_dimension(a, n);
        normalized ndl = NORMALIZE_EXPRESSION(dimension_lower(d));
        normalized ndu = NORMALIZE_EXPRESSION(dimension_upper(d));

        if (normalized_linear_p(ndl) && normalized_linear_p(ndu))
	{
            Pvecteur v1 = vect_new(TCST, VALUE_ONE);
            Pvecteur vd = vect_substract(normalized_linear(ndu),
                                         normalized_linear(ndl));
            Pvecteur v = vect_add(vd, v1);

            vect_rm(v1);
            vect_rm(vd);

            size_n = vect_product(&size_nm1, &v);
        }
        else
	{
            vect_rm(size_nm1);
            size_n = (Pvecteur) NULL;
        }
    }
    else {
        size_n = (Pvecteur) NULL;
    }

    return(size_n);
}

/*
  returns a linear expression giving the offset of the n first indices
  of an array reference.
  returns (Pvecteur) -1 if the offset is not a linear expression.
*/
static Pvecteur
offset_of_reference(reference ref, int n)
{
    entity a = reference_variable(ref);
    cons *indices = reference_indices(ref);
    int nindices = gen_length(indices);
    Pvecteur voffset = (Pvecteur) NULL;
    int idim;

    pips_assert("offset_of_reference", nindices == 0 || nindices >= n);

    /* nindices is equal to zero when an array is passed as an argument
       in a call with no indices:
       CALL INIT(MAT, N)
    */

    if (nindices == 0)
        return(voffset);

    for (idim = 1; idim <= n; idim++) {
        bool non_computable = false;
        expression index = reference_ith_index(ref, idim);
        expression bound = dimension_lower(entity_ith_dimension(a, idim));
        normalized ni, nb;

        ni = NORMALIZE_EXPRESSION(index);
        nb = NORMALIZE_EXPRESSION(bound);

        if (normalized_linear_p(ni) && normalized_linear_p(nb))
	{
            Pvecteur vi = normalized_linear(ni);
            Pvecteur vb = normalized_linear(nb);

            if (! vect_equal(vi, vb))
	    {
                Pvecteur v = vect_substract(vi, vb);
                if (idim > 1)
		{
                    Pvecteur s = size_of_array(a, idim-1);
                    if ((v = vect_product(&v, &s)) == NULL)
		    {
                        non_computable = true;
                    }
                }
                if (! non_computable)
		{
                    voffset = vect_cl(voffset, VALUE_ONE, v);
                    vect_rm(v);
                }
            }
        }
        else
	{
            non_computable = true;
        }

        if (non_computable)
	{
            vect_rm(voffset);
            return((Pvecteur) -1);
        }
    }
    return(voffset);
}

/* Generate the unknown subscripts for a variable of type depth d */
list make_unknown_subscript(int d)
{
  list ind = NIL;
  int i = 0;

  for(i=0; i<d; i++) {
    expression e = make_unbounded_expression();
    ind = CONS(EXPRESSION, e, ind);
  }
  return ind;
}


/* list global_effect_translation(effect ef, entity func)
 * input    : a global or static effect from/to subroutine func.
 * output   : a list of effect, translation of effect ef into effects in
 *            the current subroutine name space.
 * modifies : nothing.
 * comment  : The corresponding common might not be declared in the caller.
 *            In this case, we translate the effect into an effect on
 *            variables of the first module found in the common variable list.
 *            BC, october 1995.
 */
static list /* of effect */
global_effect_translation(
    effect ef,
    entity source_func __attribute__ ((unused)),
    entity target_func)
{
    list l_com_ent, l_tmp, l_new_eff = NIL;
    entity eff_ent = effect_entity(ef);
    entity ccommon; /* current common */
    int eff_ent_size, total_size, eff_ent_begin_offset, eff_ent_end_offset;
    effect new_eff;
    bool found = false;

    pips_debug(5, "target function: %s (local name: %s)\n",
	       entity_name(target_func), module_local_name(target_func));
    pips_debug(5,"input effect: \n%s\n", effect_to_string(ef));

    /* If the entity is a top-level entity, no translation;
     * It is the case for variables dexcribing I/O effects (LUNS).
     */
    if (top_level_entity_p(eff_ent))
      //return CONS(EFFECT, make_sdfi_effect(ef), NIL);
	return CONS(EFFECT, translate_effect_to_sdfi_effect(ef), NIL);

    if (effects_package_entity_p(eff_ent))
      return CONS(EFFECT, translate_effect_to_sdfi_effect(ef), NIL);

    /* First, we search if the common is declared in the target function;
     * if not, we have to deterministically choose an arbitrary function
     * in which the common is declared. It will be our reference.
     * By deterministically, I mean that this function shall be chosen whenever
     * we try to translate from this common to a routine where it is not
     * declared.
     */
    ccommon = ram_section(storage_ram(entity_storage(eff_ent)));
    l_com_ent = area_layout(type_area(entity_type(ccommon)));

    pips_debug(5, "common name: %s\n", entity_name(ccommon));

    for( l_tmp = l_com_ent; !ENDP(l_tmp) && !found; l_tmp = CDR(l_tmp) )
    {
	entity com_ent = ENTITY(CAR(l_tmp));
	if (strcmp(entity_module_name(com_ent),
		   module_local_name(target_func)) == 0)
	{
	    found = true;
	}
    }

    /* If common not declared in caller, use the subroutine of the first entity
     * that appears in the common layout. (not really deterministic: I should
     * take the first name in lexical order. BC.
     */
    if(!found)
    {
	entity ent = ENTITY(CAR(l_com_ent));
	target_func = module_name_to_entity(
	    entity_module_name(ent));
	ifdebug(5)
	{
	    pips_debug(5, "common not declared in caller,\n"
		       "\t using %s declarations instead\n",
		       entity_name(target_func));
	}
    }

    /* Second, we calculate the offset and size of the effect entity */
    eff_ent_size = array_size(eff_ent);
    eff_ent_begin_offset = ram_offset(storage_ram(entity_storage(eff_ent)));
    eff_ent_end_offset = eff_ent_begin_offset + eff_ent_size - 1;

    pips_debug(6, "\n\t eff_ent: size = %d, offset_begin = %d,"
	       " offset_end = %d \n",
	      eff_ent_size, eff_ent_begin_offset, eff_ent_end_offset);

    /* then, we perform the translation */
    for(total_size = 0; !ENDP(l_com_ent) && (total_size < eff_ent_size);
	l_com_ent = CDR(l_com_ent))
    {
	entity new_ent = ENTITY(CAR(l_com_ent));

	pips_debug(6, "current entity: %s\n", entity_name(new_ent));

	if (strcmp(entity_module_name(new_ent),
		   module_local_name(target_func)) == 0)
	{
	    int new_ent_size = array_size(new_ent);
	    int new_ent_begin_offset =
		ram_offset(storage_ram(entity_storage(new_ent)));
	    int new_ent_end_offset = new_ent_begin_offset + new_ent_size - 1;

	    pips_debug(6, "\n\t new_ent: size = %d, "
		       "offset_begin = %d, offset_end = %d \n",
		       new_ent_size, new_ent_begin_offset, new_ent_end_offset);

	    if ((new_ent_begin_offset <= eff_ent_end_offset) &&
		(eff_ent_begin_offset <= new_ent_end_offset ))
		/* these entities have elements in common */
	    {
	      type new_t = entity_type(new_ent);
	      int new_d = type_depth(new_t);
	      list ind = make_unknown_subscript(new_d);

		/* if the new entity is entirely contained in the original one
		 */
		if ((new_ent_begin_offset >= eff_ent_begin_offset) &&
		    (new_ent_end_offset <= eff_ent_end_offset))
		{
		    new_eff =
			make_simple_effect
			(make_reference(new_ent, ind), /* ??? memory leak */
			 copy_action(effect_action(ef)),
			 make_approximation
			 (approximation_tag(effect_approximation(ef)), UU));
		}
		/* If they only have some elements in common */
		else
		{
		    new_eff =
			make_simple_effect
			(make_reference(new_ent, ind), /* ??? memory leak */
			 copy_action(effect_action(ef)),
			 make_approximation(is_approximation_may, UU));
		}

		total_size += min (eff_ent_begin_offset,new_ent_end_offset)
		    - max(eff_ent_begin_offset, new_ent_begin_offset) + 1;

		l_new_eff = CONS(EFFECT, new_eff, l_new_eff);
	    }
	}
    }

    ifdebug(5)
    {
	pips_debug(5, "final effects:\n");
	print_effects(l_new_eff);
    }

    return gen_nreverse(l_new_eff);
}

static effect
translate_array_effect(entity called_func, list real_args, reference real_ref,
		       effect formal_effect)
{
    bool bad_reshaping;

    action formal_tac = effect_action(formal_effect);
    tag formal_tap = approximation_tag(effect_approximation(formal_effect));

    entity formal_var = reference_variable(effect_any_reference(formal_effect));
    int formal_ndims = NumberOfDimension(formal_var);

    entity real_var = reference_variable(real_ref);
    int real_ndims = NumberOfDimension(real_var);

    effect real_effect = effect_undefined;

    cons *pc;
    int ipc;

    Psysteme equations;			/* equations entre parametres et
					   arguments */
    Pvecteur size_formal;		/* taille du tableau formel */
    Pvecteur size_real;			/* taille du sous-tableau reel */
    Pvecteur offset;			/* offset de la reference reelle */
    Pvecteur ineq;			/* contrainte a tester:
					   size_formal+offset < size_real */

    /* FI: why give up for two n-D arrays? Because there is no offset
       to compute for the other dimensions, since there is no other
       dimension, I guess. */
    if (formal_ndims >= real_ndims)
	return(effect_undefined);

    /* build equations linking actual arguments and formal parameters */
    equations = sc_new();
    for (ipc = 1, pc = real_args; pc != NIL; pc = CDR(pc), ipc++) {
	expression ra = EXPRESSION(CAR(pc));
	entity pf = find_ith_parameter(called_func, ipc);

	if (entity_integer_scalar_p(pf)) {
	    normalized nra = NORMALIZE_EXPRESSION(ra);

	    if (normalized_linear_p(nra)) {
		Pvecteur v1 = (Pvecteur) normalized_linear(nra);
		Pvecteur v2 = vect_new((Variable) pf, VALUE_ONE);
		sc_add_egalite(equations,
			  contrainte_make(vect_substract(v1, v2)));

	    }
	}
    }

    if (get_debug_level() >= 5) {
	fprintf(stderr, "\nBinding equations: ");
	sc_fprint(stderr, equations, (get_variable_name_t) vect_debug_entity_name);
    }

    /*
      on calcule la taille des tableaux formels et reels,
      l'offset de la reference, et le vecteur de l'inequation.
      */
    size_formal = size_of_array(formal_var, formal_ndims);
    size_real = size_of_array(real_var, formal_ndims);
    offset = offset_of_reference(real_ref, formal_ndims);
    if (get_debug_level() >= 5) {
	fprintf(stderr, "size of formal: ");
	vect_fprint(stderr, size_formal, (get_variable_name_t) vect_debug_entity_name);
	fprintf(stderr, "size of real: ");
	vect_fprint(stderr, size_real, (get_variable_name_t) vect_debug_entity_name);
	if(offset != (Pvecteur) -1) {
	    fprintf(stderr, "offset: ");
	    vect_fprint(stderr, offset, (get_variable_name_t) vect_debug_entity_name);
	}
	else {
	    fprintf(stderr, "offset: could not be computed\n");
	}
    }

    /* Check that inequality real_size <= formal_size + offset - 1
     * is not compatible with the binding equations
     */
    bad_reshaping = true;
    if (size_formal != NULL && size_real != NULL && offset != (Pvecteur) -1)
    {
	Pcontrainte ct = CONTRAINTE_UNDEFINED;
	ineq = size_real;
	vect_add_elem(&ineq,  TCST, VALUE_ONE);
	ineq = vect_cl(ineq, VALUE_MONE, size_formal);
	vect_rm(size_formal);
	ineq = vect_cl(ineq, VALUE_MONE, offset);
	vect_rm(offset);

	/* on ajoute la contrainte au systeme */
	ct =  contrainte_make(ineq);
	sc_add_ineg(equations, ct);

	ifdebug(5) {
	    fprintf(stderr, "contrainte: ");
	    vect_fprint(stderr, ineq, (get_variable_name_t) vect_debug_entity_name);
	    fprintf(stderr, "\nsysteme a prouver: ");
	    sc_fprint(stderr, equations, (get_variable_name_t) vect_debug_entity_name);
	}

	bad_reshaping = false;
	sc_creer_base(equations);
	if (!sc_empty_p((equations = sc_normalize(equations)))) {
	    debug(5, "translate_array_effect",
		  "Test feasability for normalized system\n");
	    bad_reshaping = sc_faisabilite(equations);
	    debug(5, "translate_array_effect",
		  "Test feasability for normalized system: %s\n",
		  bool_to_string(bad_reshaping));
	}
	else {
	    debug(5, "translate_array_effect",
		  "System could not be normalized\n");
	}
	sc_rm(equations);
    }

    if (! bad_reshaping) {
	int i;
	cons *pdims = NIL;

	for (i = 1; i <= formal_ndims; i++) {
	    pdims = gen_nconc(pdims, CONS(EXPRESSION,
					  entity_ith_bounds(real_var, i),
					  NIL));
	}

	if (reference_indices(real_ref) != NIL) {

	    for (i = formal_ndims+1; i <= real_ndims; i++) {
		pdims = gen_nconc(pdims,
				  CONS(EXPRESSION,
				       reference_ith_index(real_ref, i),
				       NIL));
	    }
	}
	else { /* Il faudrait recuperer la declaration du tableau.
		* (03/93,yi-qing) */
	     for (i = formal_ndims+1; i <= real_ndims; i++) {
		pdims = gen_nconc(pdims, CONS(EXPRESSION,
					      int_to_expression(1),
					      NIL));
	    }
	 }

	real_effect = make_simple_effect(
	    make_reference(real_var, pdims), /* ??? memory leak */
	    copy_action(formal_tac),
	    make_approximation(formal_tap, UU));
	pips_debug(5, "good reshaping between %s and %s\n",
	      entity_name(real_var), entity_name(formal_var));
    }
    else {
	user_warning("translate_array_effect",
		     "bad reshaping between %s and %s\n",
		     entity_name(real_var), entity_name(formal_var));
    }

    return(real_effect);
}

static effect
translate_effect(entity called_func, list real_args, reference real_ref,
		 effect formal_effect)
{
    entity formal_var = reference_variable(effect_any_reference(formal_effect));
    entity real_var = reference_variable(real_ref);

    action formal_tac = effect_action(formal_effect);
    tag formal_tap = approximation_tag(effect_approximation(formal_effect));

    effect real_effect;

    ifdebug(8)
    {
	pips_debug(8, "Formal effect: %s\n",
		   words_to_string(words_effect(formal_effect)));

    }
    if (entity_scalar_p(formal_var))
    {
	pips_debug(8, "Scalar formal variable.\n");
	if (entity_scalar_p(real_var) || !ENDP(reference_indices(real_ref)))
	    real_effect = make_simple_effect
		(real_ref,
		 copy_action(formal_tac),
		 make_approximation(formal_tap, UU));
	else
	{
	    reference eff_ref;
	    list eff_ref_indices = NIL;

	    MAP(DIMENSION, dim,
	    {
		expression lo_exp = dimension_lower(dim);

		eff_ref_indices =
		    gen_nconc(eff_ref_indices,
			      CONS(EXPRESSION, lo_exp, NIL));

	    },
		variable_dimensions(type_variable(entity_type(real_var))));
	    eff_ref = make_reference(real_var, eff_ref_indices);
	    real_effect = make_simple_effect(
		eff_ref,
		copy_action(formal_tac),
		make_approximation(formal_tap, UU));
	}
    }
    else {
	pips_debug(8, "Array formal variable: translating.\n");
	real_effect = translate_array_effect(called_func, real_args,
					     real_ref, formal_effect);
    }

    if (real_effect == effect_undefined) {
      int d = type_depth(entity_type(real_var));
      list sl = make_unbounded_subscripts(d);

	pips_debug(8," translation failed\n");
        /* translation failed; returns the whole real arg */
        real_effect = make_simple_effect
	    (make_reference(real_var, sl), /* ??? memory leak */
	     copy_action(formal_tac),
	     make_approximation(is_approximation_may, UU));
    }

    ifdebug(8)
	{
	pips_debug(8, "Real effect: %s\n",
		   words_to_string(words_effect(real_effect)));

	}

    return(real_effect);
}



/* FC: I developped this function afther noticing that the next one was useless for my purpose.
 *
 * FI: it assumes reference passing (check added).
 */
/** @return the list of effect of the callee translated into the caller
 *  name space
 *  @param c, the called function
 *  @param e, the effect to translate
 */
list /* of effect */
summary_effect_to_proper_effect(
    call c,
    effect e)
{
    entity var = effect_variable(e);
    storage st = entity_storage(var);
    list le = NIL;
    entity f = call_function(c);

    if(!parameter_passing_by_reference_p(f)) {
      // not handled case return an empty list
      pips_user_warning("not handled case, need to be implemented\n");
    } else {
      if (storage_formal_p(st)) {
	effect res;
	pips_debug (9, "storage formal case\n");
	/* find the corresponding argument and returns the reference */
	int n = formal_offset(storage_formal(st));
	expression nth = EXPRESSION(gen_nth(n-1, call_arguments(c)));

	pips_assert("expression is a reference or read effect",
		    effect_read_p(e) || expression_reference_p(nth));
	/* FI: a preference is forced here */
	res = make_effect(make_cell_preference(make_preference(expression_reference(nth))),
			  copy_action(effect_action(e)),
			  copy_approximation(effect_approximation(e)),
			  copy_descriptor(effect_descriptor(e)));

	le = CONS(EFFECT, res, NIL);
      }
      else if (storage_ram_p(st)) {
	pips_debug (9, "storage ram case\n");
	le = global_effect_translation (e,
					call_function(c),
					get_current_module_entity());
      }
    }
    return le;
}

/* FC: argh! the translation is done the other way around...
 * from the call effects are derived and checked with summary ones,
 * while I expected the summary proper effects to be translated
 * into proper effects considering the call site...
 * the way it is done makes it useless to any other use
 * (for instance to translate reduction references)
 * Thus I have to develop my own function:-(
 */
list /* of effect */
fortran_summary_to_proper_effects(entity func,
				  list /* of expression */ args,
				  list /* of effect */ func_sdfi)
{
  list pc, le = NIL;
    int ipc;
    list l_formals = module_formal_parameters(func);
    int n_formals = (int) gen_length(l_formals);
    gen_free_list(l_formals);

    pips_debug(3, "effects on formals on call to %s\n", entity_name(func));

    /* Might have been done earlier by the parser thru typing or by
       flint, but let's make sure that the formal and effective
       parameter lists are compatible. */
    check_user_call_site(func, args);

    /* effets of func on formal variables are translated */
    for (pc = args, ipc = 1; ! ENDP(pc) && ipc<=n_formals; pc = CDR(pc), ipc++)
    {
        expression expr = EXPRESSION(CAR(pc));
        syntax sexpr = expression_syntax(expr);
	list la = NULL;

	if (syntax_call_p(sexpr)) {
	    /* To deal with substring real argument */
	    call c = syntax_call(sexpr);
	    entity f = call_function(c);
	    la = call_arguments(c);

	   if  (intrinsic_entity_p(f)
		&& (strcmp(entity_local_name(f), SUBSTRING_FUNCTION_NAME)==0))
	    sexpr = expression_syntax(EXPRESSION(CAR(la)));
	}

	if (syntax_reference_p(sexpr))
	{
            reference real_ref= syntax_reference(sexpr);

            MAP(EFFECT, formal_effect,
		{
		    entity formal_param = effect_entity(formal_effect);

		    if (ith_parameter_p(func, formal_param, ipc))
		    {
			effect real_effect =
			    translate_effect(func, args, real_ref, formal_effect);

			le = CONS(EFFECT, real_effect, le);
		    }
		},
		    func_sdfi);
        }
	else {
	    /* check if there is no must write effect */
	  FOREACH(EFFECT, formal_effect, func_sdfi)
		{
		    entity formal_param = effect_entity(formal_effect);

		    if (ith_parameter_p(func, formal_param, ipc))
		    {
			if (effect_write_p(formal_effect)) {
			    char * term = NULL;

			    switch(ipc) {
			    case 1: term = "st";
				break;
			    case 2: term = "nd";
				break;
			    case 3: term ="rd";
				break;
			    default:
				term = "th";
			    }

			    if (effect_exact_p(formal_effect))
				pips_user_warning
				    ("\nmodule %s called by module %s:\n\twrite"
				     " effect on non-variable actual"
				     " parameter thru %d%s formal parameter %s\n",
				     module_local_name(func),
				     module_local_name
				     (get_current_module_entity()),
				     ipc, term, entity_local_name(formal_param));
			    else
				pips_user_warning
				    ("\nmodule %s called by module %s:\n\t"
				     "possible write effect on non-variable "
				     "actual parameter thru %d%s "
				     "formal parameter %s\n",
				     module_local_name(func),
				     module_local_name
				     (get_current_module_entity()),
				     ipc, term,
				     entity_local_name(formal_param));
			}
		    }
		}

	    /* if everything is fine, then the effects are the read effects
	     * of the expression */
	    le = gen_nconc(generic_proper_effects_of_expression(expr), le);
	}

    }
    pips_debug(3, "effects on statics and globals\n");
/* effets of func on static and global variables are translated */
    if (get_bool_property("GLOBAL_EFFECTS_TRANSLATION"))
    {
      FOREACH(EFFECT, ef, func_sdfi)
	    {
		if (storage_ram_p(entity_storage(effect_entity(ef))))
		    le = gen_nconc(global_effect_translation
				   (ef, func, get_current_module_entity()),le);
	    }
    }
    else
	/* hack for HPFC: no translation of global effects */
    {
	MAP(EFFECT, ef,
	    {
		if (storage_ram_p(entity_storage(effect_entity(ef))))
		    le = CONS(EFFECT, make_sdfi_effect(ef), le);

	    },
		func_sdfi);
    }

    return gen_nreverse(le);
}

/**

 @param l_sum_eff is a list of effects on a C function formal parameter. These
        effects must be vissible from the caller, which means that their
        reference has at leat one index.
 @param real_arg is an expression. It's the real argument corresponding to
        the formal parameter which memory effects are represented by l_sum_eff.
 @param context is the transformer translating the callee's neame space into
        the caller's name space.
 @return a list of effects which are the translation of l_sum_eff in the
         caller's name space.
 */
list c_simple_effects_on_formal_parameter_backward_translation(list l_sum_eff,
    expression real_arg,
    transformer context)
{
  list l_eff = NIL; /* the result */

  if (!ENDP(l_sum_eff))
  {
    syntax real_s = expression_syntax(real_arg);
    type real_arg_t = expression_to_type(real_arg);
    ifdebug(5)
    {
      pips_debug(8, "begin for real arg %s, of type %s and effects :\n",
          words_to_string(words_expression(real_arg,NIL)),
          type_to_string(real_arg_t));
      (*effects_prettyprint_func)(l_sum_eff);
    }

    switch (syntax_tag(real_s))
    {
    case is_syntax_reference:
    {
      reference real_ref = syntax_reference(real_s);
      entity real_ent = reference_variable(real_ref);
      list real_ind = reference_indices(real_ref);

      /* if it's a pointer or a partially indexed array
       * We should do more testing here to check if types
       * are compatible... (see effect_array_substitution ?)
       */
      if (pointer_type_p(real_arg_t) ||
          gen_length(real_ind) < type_depth(entity_type(real_ent)))
      {
        FOREACH(EFFECT, eff, l_sum_eff) {
          reference eff_ref = effect_any_reference(eff);
          //action eff_act = effect_action(eff);
          //list eff_ind = reference_indices(eff_ref);

          pips_debug(8, "pointer type real arg reference\n");

          reference n_eff_ref;
          descriptor d;
          bool exact_translation_p;
          simple_cell_reference_with_value_of_cell_reference_translation(eff_ref, descriptor_undefined,
              real_ref, descriptor_undefined,
              0,
              &n_eff_ref, &d,
              &exact_translation_p);

          if (entity_all_locations_p(reference_variable(n_eff_ref)))
          {
            // functions that can be pointed by reference_to_effect_func:
            // reference_to_simple_effect
            // reference_to_convex_region
            // reference_to_reference_effect
            effect real_eff = (*reference_to_effect_func)(real_ref, effect_action(eff), true);
            type real_eff_type = reference_to_type(real_ref);
            tag act = effect_write_p(eff)? 'w' : 'r' ;
            list l_tmp = generic_effect_generate_all_accessible_paths_effects(real_eff,
                real_eff_type,
                act);
            l_eff = gen_nconc(l_eff, l_tmp);
            free_effect(real_eff);
            free_type(real_eff_type);
            free_reference(n_eff_ref);
          }
          else
          {
            effect n_eff = make_effect(make_cell(is_cell_reference, n_eff_ref),
                copy_action(effect_action(eff)),
                exact_translation_p? copy_approximation(effect_approximation(eff)):
                    make_approximation_may(),
                    make_descriptor(is_descriptor_none,UU));
            l_eff = gen_nconc(l_eff, CONS(EFFECT, n_eff, NIL));
          }
        } /* FOREACH */
      } /*  if (pointer_type_p(real_arg_t)) */
      else
      {
        pips_debug(8, "real arg reference is not a pointer and is not a partially indexed array -> NIL \n");

      } /* else */
      break;
    } /* case is_syntax_reference */
    case is_syntax_subscript:
    {
      bool read_p = false;
      bool write_p = false;
      pips_user_warning("Subscript not supported yet : returning anywhere effect\n");
      FOREACH(EFFECT, eff, l_sum_eff)
      {
        if (effect_read_p(eff)) read_p = true;
        if (effect_write_p(eff)) write_p = true;
      }
      if (read_p)
        l_eff = CONS(EFFECT, make_anywhere_effect(make_action_read_memory()), NIL);
      if (write_p)
        l_eff = CONS(EFFECT, make_anywhere_effect(make_action_write_memory()), l_eff);
      break;
    }
    case is_syntax_call:
    {
      call real_call = syntax_call(real_s);
      entity real_op = call_function(real_call);
      list args = call_arguments(real_call);
      effect n_eff = effect_undefined;

      if (ENTITY_ASSIGN_P(real_op))
      {
        l_eff = c_simple_effects_on_formal_parameter_backward_translation
            (l_sum_eff, EXPRESSION(CAR(CDR(args))), context);
      }
      else if(ENTITY_ADDRESS_OF_P(real_op))
      {
        expression arg1 = EXPRESSION(CAR(args));
        //syntax s1 = expression_syntax(arg1);
        //reference r1 = syntax_reference(s1);
        list l_real_arg = NIL;
        bool anywhere_w_p = false;
        bool anywhere_r_p = false;

        /* first we compute an effect on the argument of the
         * address_of operator (to treat cases like &(n->m))*/
        pips_debug(6, "addressing operator case \n");


        list l_eff1 = NIL;
        l_real_arg =
            generic_proper_effects_of_complex_address_expression
            (arg1, &l_eff1, true);

        pips_debug_effects(6, "base effects :%s\n", l_eff1);

        FOREACH(EFFECT, eff1, l_eff1)
        {
          FOREACH(EFFECT, eff, l_sum_eff) {
            reference eff_ref = effect_any_reference(eff);
            action eff_act = effect_action(eff);

            pips_debug_effect(6, "current effect :%s\n",eff);

            if ((anywhere_r_p && action_read_p(eff_act))
                || (anywhere_w_p && action_write_p(eff_act)))
            {
              pips_debug(6, "no need to translate, "
                  "result is already anywhere\n");
            }
            else
            {
              if (effect_undefined_p(eff1))
              {
                n_eff =  make_anywhere_effect(copy_action(eff_act));
                if (action_read_p(eff_act))
                  anywhere_r_p = true;
                else
                  anywhere_w_p = true;
              }
              else
              {
                reference eff1_ref = effect_any_reference(eff1);
                reference n_eff_ref;
                descriptor d;
                bool exact_translation_p;
                simple_cell_reference_with_address_of_cell_reference_translation(eff_ref, descriptor_undefined,
                    eff1_ref, descriptor_undefined,
                    0,
                    &n_eff_ref, &d,
                    &exact_translation_p);
                n_eff = make_effect(make_cell(is_cell_reference, n_eff_ref),
                    copy_action(effect_action(eff)),
                    exact_translation_p? copy_approximation(effect_approximation(eff)):
                        make_approximation_may(),
                        make_descriptor(is_descriptor_none,UU));

                if (entity_all_locations_p(reference_variable(n_eff_ref)))
                {
                  if (action_read_p(eff_act))
                    anywhere_r_p = true;
                  else
                    anywhere_w_p = true;
                }
              }

              l_eff = gen_nconc(l_eff, CONS(EFFECT, n_eff, NIL));
            }
          } /*  FOREACH(EFFECT, eff, l_sum_eff) */
        } /* FOREACH(EFFECT, eff1, l_eff1) */
        gen_free_list(l_real_arg);
        gen_full_free_list(l_eff1);
      }
      else if(ENTITY_POINT_TO_P(real_op)|| ENTITY_FIELD_P(real_op))
      {
        list l_real_arg = NIL;
        bool anywhere_w_p = false;
        bool anywhere_r_p = false;
        /* first we compute an effect on the real_arg */

        list l_eff1 = NIL;
        l_real_arg = generic_proper_effects_of_complex_address_expression
            (real_arg, &l_eff1, true);

        FOREACH(EFFECT, eff1, l_eff1)
        {
          FOREACH(EFFECT, eff, l_sum_eff) {
            reference eff_ref = effect_any_reference(eff);
            list eff_ind = reference_indices(eff_ref);
            tag eff_act = effect_action_tag(eff);

            if ((anywhere_r_p && eff_act == is_action_read) || (anywhere_w_p && eff_act == is_action_write))
            {
              pips_debug(6, "no need to translate, result is already anywhere\n");
            }
            else
            {
              if (effect_undefined_p(eff1))
              {
                n_eff =
                    make_anywhere_effect(copy_action(effect_action(eff)));
                if (eff_act == is_action_read)
                  anywhere_r_p = true;
                else
                  anywhere_w_p = true;
              }
              else
              {
                // functions that can be pointed by effect_dup_func:
                // simple_effect_dup
                // region_dup
                // copy_effect
                n_eff = (*effect_dup_func)(eff1);
                /* memory leaks ? */
                effect_approximation(n_eff) =
                    copy_approximation(effect_approximation(eff));
                effect_action(n_eff) =
                    copy_action(effect_action(eff));
              }
              /* Then we add the indices of the effect reference */
              /* Well this is valid only in the general case :
               * we should verify that types are compatible.
               */
              FOREACH(EXPRESSION, ind, eff_ind)
              {
                // functions that can be pointed by effect_add_expression_dimension_func:
                // simple_effect_add_expression_dimension
                // convex_region_add_expression_dimension
                (*effect_add_expression_dimension_func)(n_eff, ind);
              }
              l_eff = gen_nconc(l_eff, CONS(EFFECT, n_eff, NIL));
            }
          } /* FOREACH(EFFECT, eff, l_sum_eff) */
        } /* FOREACH(EFFECT, eff1, l_eff1) */
        gen_free_list(l_real_arg);
        gen_full_free_list(l_eff1);
      }
      else  if(ENTITY_DEREFERENCING_P(real_op))
      {
        pips_debug(6, "dereferencing operator case \n");
        /* if it's a pointer or a partially indexed array
         * We should do more testing here to check if types
         * are compatible...
         */
        if (pointer_type_p(real_arg_t) ||
            !ENDP(variable_dimensions(type_variable(real_arg_t))))
        {
          pips_debug(8, "pointer type real arg\n");
          /* first compute the region corresponding to the
           * real argument
           */
          list l_real_eff = NIL;
          list l_real_arg =
              generic_proper_effects_of_complex_address_expression
              (real_arg, &l_real_eff, true);

          pips_debug_effects(6, "base effects :\n", l_real_eff);
          bool anywhere_w_p = false;
          bool anywhere_r_p = false;

          FOREACH(EFFECT, real_eff, l_real_eff)
          {
            FOREACH(EFFECT, eff, l_sum_eff) {
              tag eff_act = effect_action_tag(eff);

              if ((anywhere_r_p && eff_act == is_action_read) || (anywhere_w_p && eff_act == is_action_write))
              {
                pips_debug(6, "no need to translate, result is already anywhere\n");
              }
              else
              {/* this could easily be made generic BC. */
                if(!anywhere_effect_p(real_eff) && store_effect_p(real_eff))
                {
                  reference n_eff_ref;
                  descriptor n_eff_d;
                  effect n_eff;
                  bool exact_translation_p;
                  // functions that can be pointed by effect_dup_func:
                  // simple_effect_dup
                  // region_dup
                  // copy_effect
                  effect init_eff = (*effect_dup_func)(eff);

                  /* and then perform the translation */
                  simple_cell_reference_with_value_of_cell_reference_translation(effect_any_reference(init_eff),
                      effect_descriptor(init_eff),
                      effect_any_reference(real_eff),
                      effect_descriptor(real_eff),
                      0,
                      &n_eff_ref, &n_eff_d,
                      &exact_translation_p);
                  if (entity_all_locations_p(reference_variable(n_eff_ref)))
                  {
                    if (eff_act == is_action_read)
                      anywhere_r_p = true;
                    else
                      anywhere_w_p = true;
                  }
                  n_eff = make_effect(make_cell(is_cell_reference, n_eff_ref),
                      copy_action(effect_action(eff)),
                      exact_translation_p? copy_approximation(effect_approximation(eff)):
                          make_approximation_may(),
                          make_descriptor(is_descriptor_none,UU));
                  l_eff = gen_nconc(l_eff, CONS(EFFECT, n_eff, NIL));

                  free_effect(init_eff);
                }
              }
            }
          } /* FOREACH(EFFECT, real_eff, l_real_eff) */
          gen_free_list(l_real_arg);
          gen_full_free_list(l_real_eff);
        } /*  if (pointer_type_p(real_arg_t)) */
        else
        {
          pips_debug(8, "real arg reference is not a pointer and is not a partially indexed array -> NIL \n");
        } /* else */
        break;
      }
      else if(ENTITY_MALLOC_SYSTEM_P(real_op))
      {
        /* BC : do not generate effects on HEAP */
        /* n_eff = heap_effect(get_current_module_entity(),
         *         copy_action(effect_action(eff)));*/
      }
      else
      {
        l_eff = gen_nconc
            (l_eff,
                c_actual_argument_to_may_summary_effects(real_arg, 'x'));
      }

      if (n_eff != effect_undefined && l_eff == NIL)
        l_eff = CONS(EFFECT,n_eff, NIL);
      break;
    } /* case is_syntax_call */
    case is_syntax_cast :
    {
      pips_debug(5, "cast case\n");
      expression cast_exp = cast_expression(syntax_cast(real_s));
      type cast_t = expression_to_type(cast_exp);
      /* we should test here the compatibility of the casted expression type with
       * the formal entity type. It is not available here, however, I think it's
       * equivalent to test the compatibility with the real arg expression type
       * since the current function is called after testing the compatilibty between
       * the real expression type and the formal parameter type.
       */
      if (types_compatible_for_effects_interprocedural_translation_p(cast_t, real_arg_t))
      {
        l_eff = gen_nconc
            (l_eff,
                c_simple_effects_on_formal_parameter_backward_translation
                (l_sum_eff, cast_exp, context));
      }
      else if (!ENDP(l_sum_eff))
      {
        /* let us at least generate effects on all memory locations reachable from
         * the cast expression
         */
        cast c = syntax_cast(real_s);
        bool read_p = false, write_p = false;
        FOREACH(EFFECT, eff, l_sum_eff)
        {
          if(effect_write_p(eff)) write_p = true;
          else read_p = false;
        }
        tag t = write_p ? (read_p ? 'x' : 'w') : 'r';
        l_eff = gen_nconc
            (l_eff,
                c_actual_argument_to_may_summary_effects(cast_expression(c), t));
      }
      break;
    }
    case is_syntax_sizeofexpression :
    {
      pips_debug(5,"sizeof epxression -> NIL");
      break;
    }
    case is_syntax_va_arg :
    {
      pips_internal_error("va_arg() : should have been treated before");
      break;
    }
    case is_syntax_application :
    {
      pips_internal_error("Application not supported yet");
      break;
    }
    case is_syntax_range :
    {
      pips_user_error("Illegal effective parameter: range\n");
      break;
    }
    default:
      pips_internal_error("Illegal kind of syntax");
      break;
    } /* switch */

    free_type(real_arg_t);
  }

  ifdebug(8)
  {
    pips_debug(8, "end with effects :\n");
    print_effects(l_eff);
  }

  return(l_eff);
}


/* list c_summary_effect_to_proper_effects(effect eff, expression real_arg)
 * input    : a summary effect eff corresponding to a formal parameter,
 *            and the corresponding actual argument at the current call site.
 * output   : a list of new effects in the name space of the caller.
 *            if the translation is not possible, returns a list of effects
 *            on the memory locations possibly pointed to by the actual
 *            argument.
 * modifies : nothing.
 * comment  : Contrarily to the previous version of this function,
 *            does not modify eff.
 * FI: it might be better to return a list of effects including the
 * read implied by the evaluation of real_arg
 * BC : no because there can be several effects for one real arg. This would
 * induce redundant, hence harmful, computations.
 *

 */
list c_summary_effect_to_proper_effects(effect eff, expression real_arg)
{
  list l_eff = NIL; /* the result */

  reference eff_ref = effect_any_reference(eff);
  list eff_ind = reference_indices(eff_ref);

  ifdebug(8)
  {
    pips_debug(8, "begin for real arg %s, and effect :\n",
        words_to_string(words_expression(real_arg,NIL)));
    print_effect(eff);
  }

  /* Whatever the real_arg may be if there is an effect on the sole value of the
   * formal arg, it generates no effect on the caller side.
   */

  if (gen_length(eff_ind) == 0)
  {
    pips_debug(5, "effect on the value of the formal parameter -> NIL\n");
  }
  else
  {
    syntax real_s = expression_syntax(real_arg);
    type real_arg_t = expression_to_type(real_arg);

    pips_debug(5, "type of real argument expression : %s\n",
        type_to_string(real_arg_t));

    switch (syntax_tag(real_s))
    {
    case is_syntax_reference:
    {
      reference real_ref = syntax_reference(real_s);
      entity real_ent = reference_variable(real_ref);
      list real_ind = reference_indices(real_ref);

      /* if it's a pointer or a partially indexed array
       * We should do more testing here to check if types
       * are compatible... (see effect_array_substitution ?)
       */
      if (pointer_type_p(real_arg_t) ||
          gen_length(real_ind) < type_depth(entity_type(real_ent)))
      {
        reference eff_ref = effect_any_reference(eff);
        list eff_ind = reference_indices(eff_ref);

        reference new_ref = copy_reference(real_ref);
        effect new_eff;

        pips_debug(8, "pointer type real arg reference\n");

        /* we add the indices of the effect reference
         * to the real reference */
        pips_debug(8, "effect on the pointed area : \n");
        // functions that can be pointed by reference_to_effect_func:
        // reference_to_simple_effect
        // reference_to_convex_region
        // reference_to_reference_effect
        new_eff = (* reference_to_effect_func)(new_ref,
            copy_action(effect_action(eff)),false);
        FOREACH(EXPRESSION, eff_ind_exp, eff_ind)
        {
          // functions that can be pointed by effect_add_expression_dimension_func:
          // simple_effect_add_expression_dimension
          // convex_region_add_expression_dimension
          (*effect_add_expression_dimension_func)(new_eff, eff_ind_exp);
        }
        l_eff = gen_nconc(l_eff, CONS(EFFECT, new_eff, NIL));
      } /*  if (pointer_type_p(real_arg_t)) */
      else
      {
        pips_debug(8, "real arg reference is not a pointer and is not a partially indexed array -> NIL \n");
      } /* else */
      break;
    } /* case is_syntax_reference */
    case is_syntax_subscript:
    {
      /* I guess this case could be merged with other cases (calls other than adress of, and reference case */
      list l_real_arg_eff= NIL;
      list l_real_arg = NIL;

      /* first we compute an effect on the real argument */
      l_real_arg = generic_proper_effects_of_complex_address_expression
          (real_arg, &l_real_arg_eff, effect_write_p(eff));
      gen_full_free_list(l_real_arg);

      FOREACH(EFFECT, real_arg_eff, l_real_arg_eff)
      {
        if (effect_undefined_p(real_arg_eff))
          real_arg_eff =
              make_anywhere_effect(copy_action(effect_action(eff)));

        if (!anywhere_effect_p(real_arg_eff))
        {
          effect_approximation_tag(real_arg_eff) = effect_approximation_tag(eff);

          /* then we add the indices of the original_effect */
          reference eff_ref = effect_any_reference(eff);
          list eff_ind = reference_indices(eff_ref);
          FOREACH(EXPRESSION, eff_ind_exp, eff_ind)
          {
            // functions that can be pointed by effect_add_expression_dimension_func:
            // simple_effect_add_expression_dimension
            // convex_region_add_expression_dimension
            (*effect_add_expression_dimension_func)
                    (real_arg_eff, eff_ind_exp);
          }
        }
        l_eff = gen_nconc(l_eff, CONS(EFFECT, real_arg_eff, NIL));
      }
      gen_free_list(l_real_arg_eff);
    }
    break;
    case is_syntax_call:
    {
      call real_call = syntax_call(real_s);
      entity real_op = call_function(real_call);
      list args = call_arguments(real_call);
      effect n_eff = effect_undefined;

      if (ENTITY_ASSIGN_P(real_op))
      {
        l_eff = c_summary_effect_to_proper_effects
            (eff, EXPRESSION(CAR(CDR(args))));
      }
      else if(ENTITY_ADDRESS_OF_P(real_op))
      {
        expression arg1 = EXPRESSION(CAR(args));
        list l_real_arg = NIL;
        list l_eff1 = NIL;

        /* first we compute an effect on the argument of the
         * address_of operator (to treat cases like &(n->m))*/
        /* It's very costly because we do not re-use l_real_arg
         * We should maybe scan the real args instead of scanning
         * the effects : we would do this only once per
         * real argument */
        l_real_arg = generic_proper_effects_of_complex_address_expression
            (arg1, &l_eff1, effect_write_p(eff));
        gen_full_free_list(l_real_arg);

        FOREACH(EFFECT, eff1, l_eff1)
        {
          if (effect_undefined_p(eff1))
            n_eff =  make_anywhere_effect(copy_action(effect_action(eff)));
          else
          {
            n_eff = eff1;
            effect_approximation(n_eff) =
                copy_approximation(effect_approximation(eff));
          }
          /* BC : This must certainely be improved
           *      only simple cases are handled .*/
          if(ENDP(reference_indices(effect_any_reference(n_eff))))
          {
            expression first_ind = EXPRESSION(CAR(eff_ind));

            /* The operand of & is a scalar expression */
            /* the first index of eff reference (which must
             * be equal to 0) must not be taken into account.
             * The other indices must be appended to n_eff indices
             */

            pips_assert("scalar case : the first index of eff must be equal to [0]", expression_equal_integer_p(first_ind, 0));
            FOREACH(EXPRESSION, eff_ind_exp, CDR(eff_ind))
            {
              // functions that can be pointed by effect_add_expression_dimension_func:
              // simple_effect_add_expression_dimension
              // convex_region_add_expression_dimension
              (*effect_add_expression_dimension_func)
                      (n_eff, eff_ind_exp);
            }
          }
          else
          {
            expression first_ind = EXPRESSION(CAR(eff_ind));
            reference n_eff_ref = effect_any_reference(n_eff);
            expression last_n_eff_ind =
                EXPRESSION(CAR(gen_last(reference_indices(n_eff_ref))));
            expression n_exp;


            /* The operand of & is subcripted */
            /* The first index of eff must be added to the last
             * index of n_eff (except if it is unbounded),
             * and the remaining indices list
             * be appended to th indices of n_eff
             * could be more generic
             */
            if(!unbounded_expression_p(last_n_eff_ind))
            {
              if(!unbounded_expression_p(first_ind))
              {
                value v;
                n_exp = MakeBinaryCall
                    (entity_intrinsic(PLUS_OPERATOR_NAME),
                        last_n_eff_ind, copy_expression(first_ind));
                /* Then we must try to evaluate the expression */
                v = EvalExpression(n_exp);
                if (! value_undefined_p(v) &&
                    value_constant_p(v))
                {
                  constant vc = value_constant(v);
                  if (constant_int_p(vc))
                  {
                    /* free_expression(n_exp);*/
                    n_exp = int_to_expression(constant_int(vc));
                  }
                }
              }
              else
              {
                n_exp = make_unbounded_expression();
              }
              CAR(gen_last(reference_indices(n_eff_ref))).p
                  = (void *) n_exp;
              /*should we free last_n_eff_ind ? */
            }
            FOREACH(EXPRESSION, eff_ind_exp, CDR(eff_ind))
            {
              // functions that can be pointed by effect_add_expression_dimension_func:
              // simple_effect_add_expression_dimension
              // convex_region_add_expression_dimension
              (*effect_add_expression_dimension_func)
                      (n_eff, eff_ind_exp);
            }
          }
          l_eff = CONS(EFFECT, n_eff, l_eff);
        } /* FOREACH(EFFECT, eff1, l_eff1) */
        gen_free_list(l_eff1);
      }
      else if(ENTITY_POINT_TO_P(real_op)|| ENTITY_FIELD_P(real_op))
      {
        list l_real_arg = NIL;
        list l_eff1 = NIL;
        /* first we compute an effect on the real_arg */
        /* It's very costly because we do not re-use l_real_arg
         * We should maybe scan the real args instead of scanning
         * the effects. */
        l_real_arg = generic_proper_effects_of_complex_address_expression
            (real_arg, &l_eff1, effect_write_p(eff));
        gen_free_list(l_real_arg);

        FOREACH(EFFECT, eff1, l_eff1)
        {
          if (effect_undefined_p(eff1))
            n_eff =  make_anywhere_effect(copy_action(effect_action(eff)));
          else
          {
            n_eff = eff1;
            effect_approximation(n_eff) =
                copy_approximation(effect_approximation(eff));
          }

          /* Then we add the indices of the effect reference */
          /* Well this is valid only in the general case :
           * we should verify that types are compatible.
           */
          FOREACH(EXPRESSION, ind, eff_ind)
          {
            // functions that can be pointed by effect_add_expression_dimension_func:
            // simple_effect_add_expression_dimension
            // convex_region_add_expression_dimension
            (*effect_add_expression_dimension_func)(n_eff, ind);
          }
          l_eff = CONS(EFFECT, n_eff, l_eff);
        }
        gen_free_list(l_eff1);
      }
      else if(ENTITY_MALLOC_SYSTEM_P(real_op)) {
        /* BC : do not generate effects on HEAP */
        /*n_eff = heap_effect(get_current_module_entity(),
                              copy_action(effect_action(eff)));*/
      }
      else {
        /* We do not know what to do with the initial value */
        l_eff = effect_to_list(make_anywhere_effect(copy_action(effect_action(eff))));
      }

      if (n_eff != effect_undefined && l_eff == NIL)
        l_eff = CONS(EFFECT,n_eff, NIL);
      break;
    } /* case is_syntax_call */
    case is_syntax_cast :
    {
      /* Ignore the cast */
      cast c = syntax_cast(real_s);
      pips_user_warning("Cast effect is ignored\n");
      l_eff = c_summary_effect_to_proper_effects(eff, cast_expression(c));
      break;
    }
    case is_syntax_sizeofexpression :
    {
      pips_debug(5,"sizeof epxression -> NIL");
      break;
    }
    case is_syntax_va_arg :
    {
      pips_internal_error("va_arg() : should have been treated before");
      break;
    }
    case is_syntax_application :
    {
      pips_internal_error("Application not supported yet");
      break;
    }
    case is_syntax_range :
    {
      pips_user_error("Illegal effective parameter: range\n");
      break;
    }
    default:
      pips_internal_error("Illegal kind of syntax");
      break;
    } /* switch */

    free_type(real_arg_t);
  } /*else */

  ifdebug(8)
  {
    pips_debug(8, "end with effects :\n");
    print_effects(l_eff);
  }

  return(l_eff);

}

/* FI: I do not know how deep the copying should be and the
   preferences do not help for C because some references are not found
   in the program expression but built according to an abstraction. */
list c_summary_to_proper_effects(
    entity func,
    list /* of expression */ args,
    list /* of effect */ func_sdfi)
{
  list pel = NIL; /* proper effect list */
  list ce = list_undefined; /* current effect */
  bool param_varargs_p = false;
  type u_func_t = ultimate_type(entity_type(func));
  list params = functional_parameters(type_functional(u_func_t));

  ifdebug(2)
  {
    pips_debug(2, "begin for function %s\n", entity_local_name(func));
    pips_debug(2, "with actual arguments :\n");
    print_expressions(args);
    pips_debug(2, "and effects :\n");
    print_effects(func_sdfi);
  }

  /* first the case of va_args.
   * the approach is conservative : we generate may r/w effects
   * on all actual arguments. This could and should be refined...
   */

  pips_debug(8, "first check for varargs \n");
  MAP(PARAMETER, e_param,
      {
          type te = parameter_type(e_param);
          pips_debug(8, "parameter type : %s\n", type_to_string(te));
          if(type_varargs_p(te))
          {
            param_varargs_p = true;
          }
      },
      params);

  if (param_varargs_p)
  {
    pips_debug(5, "varargs parameters.\n");

    /* First, we keep those effects in the summary list that are not
     * effects on formal parameters, that is to say effects on global
     * variables.
     */

    MAP(EFFECT, eff,
        {
            reference r = effect_any_reference(eff);
            entity v = reference_variable(r);

            if(formal_parameter_p(v))
            {
              pips_debug(8, "effect on formal parameter skipped : %s\n",
                  entity_name(v));
            }
            else
            {
              bool force_may_p = true;

              pips_debug(8, "effect on global entity %s kept.\n",
                  entity_name(v));

              /* We keep a may effect on the global entity.*/
              // functions that can be pointed by effect_to_store_independent_effect_list_func:
              // effect_to_store_independent_sdfi_list
              // region_to_store_independent_region_list
              // functions that can be pointed by effect_dup_func:
              // simple_effect_dup
              // region_dup
              // copy_effect
              pel = gen_nconc
                  (pel,
                      (*effect_to_store_independent_effect_list_func)
                      ((*effect_dup_func)(eff), force_may_p));
            }
        },
        func_sdfi);

    ifdebug(8)
    {
      pips_debug(8, "effects on global variables :\n");
      (* effects_prettyprint_func)(pel);
    }


    /* Then, we add the read effects on actual parameters.
     * (I have to check if it is not done by in the callers of this function)
     */

    pel = gen_nconc(pel, generic_proper_effects_of_expressions(args));
    ifdebug(8)
    {
      pips_debug(8, "effects on actual parameters added :\n");
      (* effects_prettyprint_func)(pel);
    }


    /* Lastly, we add the read and write effects on the variables
     * pointed by the actual parameters if it makes sense.
     */

    pips_debug(5, "Generating r/w effects on variables pointed by actual parameters\n");

    MAP(EXPRESSION, arg,
        {
            pel = gen_nconc
            (pel, c_actual_argument_to_may_summary_effects(arg, 'x'));

        }, args);

  } /* if (param_varargs_p) */

  else
  {
    check_user_call_site(func, args);

    pips_debug(8, "no varargs \n");

    for(ce = func_sdfi; !ENDP(ce); POP(ce))
    {
      effect eff = EFFECT(CAR(ce));
      reference r = effect_any_reference(eff);
      entity v = reference_variable(r);

      if(formal_parameter_p(v))
      {
        storage s = entity_storage(v);
        formal fs = storage_formal(s);
        int rank = formal_offset(fs);
        expression ep = EXPRESSION(gen_nth(rank-1, args));
        list l_sum = NIL;

        l_sum = c_summary_effect_to_proper_effects(eff, ep);
        pel = gen_nconc(pel,l_sum);

        /* FI: I'm not too sure about this...
         * BC : It should be done later, because there can be several
         *       effects for one real arg entity.So it generates redundant
         *       effects that will need costly unioning.
         */
        pel = gen_nconc(pel, generic_proper_effects_of_expression(ep));

      } /* if(formal_parameter_p(v)) */
      else
      {
        /* This effect must be a global effect. It does not require
         * translation in C. However, it may not be in the scope of the caller. */
        // functions that can be pointed by effect_dup_func:
        // simple_effect_dup
        // region_dup
        // copy_effect
        pel = gen_nconc(pel, CONS(EFFECT, (*effect_dup_func)(eff), NIL));
      } /* else */
    } /* for */
  } /* else */

  ifdebug(5)
  {
    pips_debug(5, "resulting effects :\n");
    (*effects_prettyprint_func)(pel);
  }
  return pel;
}

list /* of effect */
summary_to_proper_effects(
    entity func,
    list /* of expression */ args,
    list /* of effect */ func_sdfi)
{
  list el = list_undefined;
  if(parameter_passing_by_reference_p(func))
    el = fortran_summary_to_proper_effects(func, args, func_sdfi);
  else
    el = c_summary_to_proper_effects(func, args, func_sdfi);

  return el;
}

/****************************************************** FORWARD TRANSLATION */

#define make_translated_effect(entity,action,approximation)\
    make_effect(make_cell(is_cell_reference, make_reference((entity), NIL)),\
		copy_action(action),\
		make_approximation(approximation, UU),\
		make_descriptor_none())

/* I'm responsible for this piece of code for out simple effects.
 * I have been inspired by the corresponding function in convex effects.
 *
 * FC. August 1997.
 */

/* actual/formal arguments forward translation:
 *
 * CALL FOO(A(*)), SUBROUTINE FOO(B(*)): effect on A => effect on B
 */
static list /* of effect */
real_simple_effects_forward_translation(
    entity func,
    list /* of expression */ real_args,
    list /* of effect */ l_eff)
{
    list /* of effect */ l_fwd_translated = NIL; /* what is being built */
    int arg_num = 1;

    MAP(EXPRESSION, e,
    {
	if (expression_reference_p(e))
	{
	    /* something to translate */
	    entity actu = reference_variable(expression_reference(e));
	    entity form = find_ith_formal_parameter(func, arg_num);

	    pips_assert("ith formal is defined", !entity_undefined_p(form));

	    /* look for an effect on actu or equivalenced
	     */
	    MAP(EFFECT, ef,
	    {
		/* ??? should be "may interfere" */
		if (same_entity_p(effect_variable(ef),actu))
		{
		    /* approximation:
		     * Actual is array => MAY(formal);
		     * Actual is scalar => actual(formal);
		     */
		    tag approx = entity_scalar_p(actu)?
		      is_approximation_may:
		      approximation_tag(effect_approximation(ef));

		    effect fef = make_translated_effect
			(form, effect_action(ef), approx);

		    /* better union not needed, all different ??? bof. */
		    l_fwd_translated = CONS(EFFECT, fef, l_fwd_translated);
		}
	    },
		l_eff);
	}
	/* else could checks sg */

	arg_num++;
    },
        real_args);

    return l_fwd_translated;
}

/* translation of global effects (i.e. variables in commons.
 */
static list /* of effect */
common_simple_effects_forward_translation(
    entity callee,
    list /* of effect */ l_eff)
{
    list l_fwd_translated = NIL;

    MAP(EFFECT, e,
    {
	entity a = effect_variable(e);
	storage s = entity_storage(a);
	if (storage_ram_p(s) && !dynamic_area_p(ram_section(storage_ram(s)))
	    && !heap_area_p(ram_section(storage_ram(s)))
	    && !stack_area_p(ram_section(storage_ram(s))))
	{
	    pips_debug(5, "considering common variable %s\n", entity_name(a));

	    /* some better union??? */
	    l_fwd_translated = gen_nconc(
		global_effect_translation
		(e, get_current_module_entity(), callee), l_fwd_translated);
	}
    },
        l_eff);

    return l_fwd_translated;
}

/* OUT effects are translated forward to a call.
 */
list /* of effect */
simple_effects_forward_translation(
    entity callee,
    list /* of expression */ real_args,
    list /* of effect */ l_eff,
    transformer context __attribute__ ((unused)))
{
    list /* of effect */ lr, lc;

    pips_debug(4, "forward translation of %s call to %s\n",
	       entity_name(get_current_module_entity()),
	       entity_name(callee));

    lr = real_simple_effects_forward_translation(callee, real_args, l_eff);
    lc = common_simple_effects_forward_translation(callee, l_eff);

    return gen_nconc(lr, lc);
}

list
c_simple_effects_on_actual_parameter_forward_translation (entity  callee,
    expression  real_exp,
    entity  formal_ent,
    list l_eff,
    transformer __attribute__ ((unused)) context)
{
  syntax real_s = expression_syntax(real_exp);
  list l_formal = NIL;

  pips_debug_effects(6,"initial effects :\n", l_eff);


  switch (syntax_tag(real_s))
  {
  case is_syntax_call:
  {
    call real_call = syntax_call(real_s);
    entity real_op = call_function(real_call);
    list args = call_arguments(real_call);
    type uet = ultimate_type(entity_type(real_op));
    value real_op_v = entity_initial(real_op);

    pips_debug(5, "call case, function %s \n", module_local_name(real_op));
    if(type_functional_p(uet))
    {
      if (value_code_p(real_op_v))
      {
        pips_debug(5, "external function\n");
        pips_user_warning("Nested function calls are ignored. Consider splitting the code before running PIPS\n");
        l_formal = NIL;
        break;
      }
      else /* it's an intrinsic */
      {
        pips_debug(5, "intrinsic function\n");

        if (ENTITY_ASSIGN_P(real_op))
        {
          pips_debug(5, "assignment case\n");
          l_formal = c_simple_effects_on_actual_parameter_forward_translation
              (callee, EXPRESSION(CAR(CDR(args))), formal_ent, l_eff, context);
          break;
        }
        else if(ENTITY_ADDRESS_OF_P(real_op))
        {
          expression arg1 = EXPRESSION(CAR(args));
          list l_real_arg = NIL;
          bool general_case = true;

          pips_debug(5, "address of case\n");

          /* first we compute an effect on the argument of the address_of operator.
           * This is to distinguish between the general case and the case where
           * the operand of the & operator is an array element.
           */
          list l_eff_real = NIL;
          l_real_arg = generic_proper_effects_of_complex_address_expression
              (arg1, &l_eff_real, true);
          gen_full_free_list(l_real_arg);

          effect eff_real = EFFECT(CAR(l_eff_real)); /* there should be a FOREACH here to scan the whole list */
          gen_free_list(l_eff_real);

          reference eff_real_ref = effect_any_reference(eff_real);
          list l_inds_real = reference_indices(eff_real_ref);
          int nb_phi_real = (int) gen_length(l_inds_real);


          /* there are indices but we don't know if they represent array dimensions,
           * struct/union/enum fields, or pointer dimensions.
           */
          if(nb_phi_real > 0)
          {
            type t = type_undefined;

            t = simple_effect_reference_type(eff_real_ref);

            if (type_undefined_p(t))
              pips_internal_error("undefined type not expected ");

            if(type_variable_p(t) && !ENDP(variable_dimensions(type_variable(t))))
            {
              pips_debug(5,"array element or sub-array case\n");
              /* array element operand : we replace the last index with
               * an unbounded dimension */
              simple_effect_change_ith_dimension_expression(eff_real,
                  make_unbounded_expression(),
                  nb_phi_real);
              general_case = false;
            }
            else
              pips_debug(5, "general case\n");
          }


          FOREACH(EFFECT, eff_orig, l_eff)
          {
            list l_inds_orig = reference_indices(effect_any_reference(eff_orig));

            /* First we have to test if the eff_real access path leads to the eff_orig access path */

            /* to do that, if the entities are the same (well in fact we should also
             * take care of aliasing), we add the constraints of eff_real to those of eff_orig,
             * and the system must be feasible.
             * We should also take care of linearization here.
             */
            bool exact_p;
            if(path_preceding_p(eff_real, eff_orig, transformer_undefined, false, &exact_p))
            {
              /* At least part of the original effect corresponds to the actual argument :
               * we need to translate it
               */
              pips_debug_effect(5, "matching access paths, considered effect is : \n", eff_orig);

              /* Then we skip the dimensions common to the two regions
               * except the last one if we are not in the general case */
              /* This is only valid when there is no linearization.
               */
              int i_max = general_case? nb_phi_real : nb_phi_real-1;
              for(int i = 1; i<= i_max; i++, POP(l_inds_orig));
              list l_new_inds = gen_full_copy_list(l_inds_orig);

              /* Then, we must add a first dimension with index 0 constraint
               * in the general case.
               * We must also change the resulting effect
               * entity for the formal entity in all cases.
               */

              if (general_case)
                l_new_inds = CONS(EXPRESSION, int_to_expression(0), l_new_inds);

              effect eff_formal = make_reference_simple_effect(make_reference(formal_ent,l_new_inds),
                  copy_action(effect_action(eff_orig)),
                  exact_p? copy_approximation(effect_approximation(eff_orig))
                      : make_approximation_may());

              pips_debug_effect(5, "resulting eff_formal\n", eff_formal);

              l_formal = EffectsMustUnion(l_formal, CONS(EFFECT, eff_formal, NIL),
                  effects_same_action_p);
              pips_debug_effects(6,"l_formal after adding new effect : \n", l_formal);


            } /* if(path_preceding_p...)*/

          } /* FOREACH */

          break;
        }
        else
        {
          pips_debug(5, "Other intrinsic case : entering general case \n");
        }
      }
    }
    else if(type_variable_p(uet))
    {
      pips_user_warning("Effects of call thru functional pointers are ignored\n");
      l_formal = NIL;
      break;
    }
    /* entering general case which includes general calls*/
  }
  case is_syntax_reference:
  case is_syntax_subscript:
  {
    effect eff_real = effect_undefined;

    pips_debug(5, "general case\n");

    /* first we compute an effect on the real_arg */
    if (syntax_reference_p(real_s))
      eff_real = reference_to_simple_effect(syntax_reference(real_s), make_action_write_memory(), true);
    else
    {
      list l_eff_real = NIL;
      list l_real_arg = generic_proper_effects_of_complex_address_expression
          (real_exp, &l_eff_real, true);
      gen_full_free_list(l_real_arg);
      if (!ENDP(l_eff_real))
        eff_real = EFFECT(CAR(l_eff_real)); /*there should be a foreach to scan all the elements */
      gen_free_list(l_eff_real);
    }

    if (!effect_undefined_p(eff_real))
    {
      FOREACH(EFFECT, eff_orig, l_eff)
	              {
        int nb_phi_orig = (int) gen_length(reference_indices(effect_any_reference(eff_orig)));
        int nb_phi_real = (int) gen_length(reference_indices(effect_any_reference(eff_real)));
        /* First we have to test if the eff_real access path leads to the eff_orig access path */

        bool exact_p;
        if(path_preceding_p(eff_real, eff_orig, transformer_undefined, true, &exact_p)
            &&  nb_phi_orig >= nb_phi_real)
        {
          /* At least part of the original effect corresponds to the actual argument :
           * we need to translate it
           */
          reference ref_formal = make_reference(formal_ent, NIL);
          effect eff_formal = reference_to_simple_effect(ref_formal, copy_action(effect_action(eff_orig)),
              false);

          pips_debug_effect(5, "matching access paths, considered effect is : \n", eff_orig);

          /* first we perform the path translation */
          reference n_eff_ref;
          descriptor n_eff_d;
          effect n_eff;
          bool exact_translation_p;
          simple_cell_reference_with_value_of_cell_reference_translation(effect_any_reference(eff_orig),
              effect_descriptor(eff_orig),
              ref_formal,
              effect_descriptor(eff_formal),
              nb_phi_real,
              &n_eff_ref, &n_eff_d,
              &exact_translation_p);
          n_eff = make_reference_simple_effect(n_eff_ref, copy_action(effect_action(eff_orig)),
              exact_translation_p? copy_approximation(effect_approximation(eff_orig)) : make_approximation_may());
          pips_debug_effect(5, "final eff_formal : \n", n_eff);

          pips_debug_effect(5, "eff_formal after context translation: \n", n_eff);

          l_formal = EffectsMustUnion(l_formal, CONS(EFFECT, n_eff, NIL),effects_same_action_p);
          pips_debug_effects(6, "l_formal after adding new effect : \n", l_formal);

        } /* if(effect_entity(eff_orig) == effect_entity(eff_real) ...)*/



        /* */

	              } /* FOREACH */
    }

    break;
  }
  case is_syntax_application:
  {
    pips_internal_error("Application not supported yet");
    break;
  }

  case is_syntax_cast:
  {
    pips_debug(6, "cast expression\n");
    type formal_ent_type = entity_basic_concrete_type(formal_ent);
    expression cast_exp = cast_expression(syntax_cast(real_s));
    type cast_exp_type = expression_to_type(cast_exp);
    if (basic_concrete_types_compatible_for_effects_interprocedural_translation_p(cast_exp_type, formal_ent_type))
    {
      l_formal =
          c_simple_effects_on_actual_parameter_forward_translation
          (callee, cast_exp,
              formal_ent, l_eff, context);
    }
    else
    {
      expression formal_exp = entity_to_expression(formal_ent);
      l_formal = c_actual_argument_to_may_summary_effects(formal_exp, 'w');
      free_expression(formal_exp);
    }
    free_type(cast_exp_type);
    break;
  }
  case is_syntax_range:
  {
    pips_user_error("Illegal effective parameter: range\n");
    break;
  }

  case is_syntax_sizeofexpression:
  {
    pips_debug(6, "sizeofexpression : -> NIL");
    l_formal = NIL;
    break;
  }
  case is_syntax_va_arg:
  {
    pips_internal_error("va_arg not supported yet");
    break;
  }
  default:
    pips_internal_error("Illegal kind of syntax");

  } /* switch */

  pips_debug_effects(6,"resulting effects :\n", l_formal);
  return(l_formal);
}
