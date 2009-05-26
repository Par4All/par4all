/* package simple effects :  Be'atrice Creusillet 5/97
 *
 * File: interprocedural.c
 * ~~~~~~~~~~~~~~~~~~~~~~~
 *
 * This File contains functions for the interprocedural computation of simple
 * effects.
 *
 * $Id$
 */

#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "text.h"

#include "misc.h"
#include "properties.h"
#include "text-util.h"
#include "ri-util.h"

#include "effects-generic.h"
#include "effects-simple.h"

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))

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
effect translate_effect_to_sdfi_effect(effect le)
{
  effect ge = copy_effect(le);
  reference ger = effect_any_reference(ge);
  entity v = reference_variable(ger);
  type ut = ultimate_type(entity_type(v));

  /* FI: I want to change this function and preserve indices when they
     are constant or unbounded. This is needed to extend the effect
     semantics and to analyze array elements and structure fields and
     pointer dereferencing. A new property would probably be needed to
     avoid a disaster, at least with the validation suite... althoug
     we might be safe with Fortran code...*/ 

  /*if(reference_indices(ger) != NIL) { */
  /* FI: could be rewritten like the other summarization routines,
     with a check for each subscript expression */
  if(!reference_with_constant_indices_p(ger) && !reference_with_unbounded_indices_p(ger)) {
    if(pointer_type_p(ut)) {
      /* A unique index is sufficient and necessary to distinguish
	 between the effect on the pointed area and the effect on the
	 pointer itself */
      gen_free_list(reference_indices(ger));
      reference_indices(ger) = CONS(EXPRESSION, make_unbounded_expression(), NIL);
  
      free_approximation(effect_approximation(ge));
      effect_approximation(ge) = make_approximation_may();
    }
    else {
      gen_free_list(reference_indices(ger));
      reference_indices(ger) = NIL;
  
      free_approximation(effect_approximation(ge));
      effect_approximation(ge) = make_approximation_may();
    }
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
  bool add_anywhere_write_effect_p = FALSE;
  bool add_anywhere_read_effect_p = FALSE;
  extern bool c_module_p(entity); /* From preprocessor.h */
  bool value_passing_p = c_module_p(get_current_module_entity());

  for(c_eff = l_eff; !ENDP(c_eff); POP(c_eff)) {
    effect eff = EFFECT(CAR(c_eff));

    entity eff_ent = effect_entity(eff);
    storage eff_s = entity_storage(eff_ent);
    boolean ignore_this_effect = FALSE;

    ifdebug(4) {
      pips_debug(4, "current effect \n%s for entity \"\%s\":\n%s\n",
		 list_to_string(words_effect(eff)), entity_name(eff_ent),
		 list_to_string(effect_words_reference_with_addressing_as_it_is
				(effect_any_reference(eff),
				 addressing_tag(effect_addressing(eff)))));
    }

    if(!anywhere_effect_p(eff)) {
      /* If the reference is a common variable (ie. with storage ram but
       * not dynamic) or a formal parameter, the effect is not ignored.
       */
      switch (storage_tag(eff_s)) {
      case is_storage_return:
	pips_debug(5, "return var ignored (%s)\n", entity_name(eff_ent));
	ignore_this_effect = TRUE;
	break;
      case is_storage_ram:
	{
	  ram r = storage_ram(eff_s);
	  /* FI: heap areas effects should be preserved... */
	  if (dynamic_area_p(ram_section(r)) || heap_area_p(ram_section(r))
	      || stack_area_p(ram_section(r))) {
	    type ut = ultimate_type(entity_type(eff_ent));
	    addressing ad = effect_addressing(eff);
	    list sl = reference_indices(effect_any_reference(eff));

	    if(pointer_type_p(ut))
	      if(!ENDP(sl) || !addressing_index_p(ad)) {
		/* Can we convert this effect using the pointer initial value? */
		/* FI: this should rather be done after a constant
		   pointer analysis but I'd like to improve quickly
		   results with Effects/fulguro01.c */
		value v = entity_initial(eff_ent);

		if(/*FALSE && */value_expression_p(v)) {
		  expression ae = value_expression(v);
		  /* Save the action before the effect may be changed */
		  action ac = effect_action(eff);

		  /* re-use an existing function... */
		  eff = c_summary_effect_to_proper_effect(eff, ae);
		  if(effect_undefined_p(eff)) {
		    pips_debug(5, "Local pointer \"%s\" initialization is not usable!\n", 
			       entity_name(eff_ent));
		    if(action_write_p(ac))
		      add_anywhere_write_effect_p = TRUE;
		    else
		      add_anywhere_read_effect_p = TRUE;
		    ignore_this_effect = TRUE;
		  }
		  else {
		    /* Should this effect be preserved? */
		    /* Let's hope we do not loop recursively... */
		    list nel = CONS(EFFECT, eff, NIL);
		    list fel = effects_dynamic_elim(nel);

		    if(ENDP(fel)) {
		      ignore_this_effect = TRUE;
		    }
		    gen_free_list(nel);
		    gen_free_list(fel);
		  }
		}
		else {
		  action ac = effect_action(eff);
		  pips_debug(5, "Local pointer \"%s\" is not initialized!\n", 
			     entity_name(eff_ent));
		  if(action_write_p(ac))
		    add_anywhere_write_effect_p = TRUE;
		  else
		    add_anywhere_read_effect_p = TRUE;
		  ignore_this_effect = TRUE;
		}
	      }
	      else {
		pips_debug(5, "Local pointer \"%s\" can be ignored\n", 
			   entity_name(eff_ent));
		ignore_this_effect = TRUE;
	      }
	    else { 
	      pips_debug(5, "dynamic or pointed var ignored (%s)\n", 
			 entity_name(eff_ent));
	      ignore_this_effect = TRUE;
	    }
	  }
	  break;
	}
      case is_storage_formal:
	if(value_passing_p) {
	  reference r = effect_any_reference(eff);
	  list inds = reference_indices(r);
	  addressing ad = effect_addressing(eff);
	  action ac = effect_action(eff);
	  if(action_write_p(ac) && addressing_index_p(ad) && ENDP(inds))
	    ignore_this_effect = TRUE;
	}
	break;
      case is_storage_rom:
	if(!area_entity_p(eff_ent) && !anywhere_effect_p(eff))
	  ignore_this_effect = TRUE;
	break;
	/*  pips_internal_error("bad tag for %s (rom)\n", 
	    entity_name(eff_ent));*/
      default:
	pips_internal_error("case default reached\n");
      }
    }
	
    if (! ignore_this_effect)  /* Eliminate dynamic variables. */ {
      /* FI: this macro is not flexible enough */
      /* effect eff_res = make_sdfi_effect(eff); */
      effect eff_res = translate_effect_to_sdfi_effect(eff);
      ifdebug(4) {
	pips_debug(4, "effect preserved for variable \"\%s\": \n\t %s\n\t%s\n", 
		   entity_name(effect_variable(eff_res)),
		   words_to_string(words_effect(eff_res)),
		   list_to_string(effect_words_reference_with_addressing_as_it_is
				  (effect_any_reference(eff_res),
				   addressing_tag(effect_addressing(eff_res)))));
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
    l_res = CONS(EFFECT, anywhere_effect(make_action_write()), l_res);
  if(add_anywhere_read_effect_p)
    l_res = CONS(EFFECT, anywhere_effect(make_action_read()), l_res);
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
        bool non_computable = FALSE;
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
                        non_computable = TRUE;
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
            non_computable = TRUE;          
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
    boolean found = FALSE;

    pips_debug(5, "target function: %s (local name: %s)\n",
	       entity_name(target_func), module_local_name(target_func));
    pips_debug(5,"input effect: \n%s\n", effect_to_string(ef));

    /* If the entity is a top-level entity, no translation;
     * It is the case for variables dexcribing I/O effects (LUNS).
     */
    if (top_level_entity_p(eff_ent))
      //return CONS(EFFECT, make_sdfi_effect(ef), NIL);
	return CONS(EFFECT, translate_effect_to_sdfi_effect(ef), NIL);

    if (io_entity_p(eff_ent))
      //return CONS(EFFECT, make_sdfi_effect(ef), NIL);
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
	    found = TRUE;
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
	    module_name(entity_name(ent)));
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
			 make_action(action_tag(effect_action(ef)), UU), 
			 make_approximation
			 (approximation_tag(effect_approximation(ef)), UU));
		}
		/* If they only have some elements in common */
		else
		{						
		    new_eff = 
			make_simple_effect
			(make_reference(new_ent, ind), /* ??? memory leak */
			 make_action(action_tag(effect_action(ef)), UU), 
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

    tag formal_tac = action_tag(effect_action(formal_effect));
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
    bad_reshaping = TRUE;
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

	bad_reshaping = FALSE;
	sc_creer_base(equations);
	if ((equations = sc_normalize(equations)) != NULL) {
	    debug(5, "translate_array_effect",
		  "Test feasability for normalized system\n");
	    bad_reshaping = sc_faisabilite(equations);
	    debug(5, "translate_array_effect",
		  "Test feasability for normalized system: %s\n",
		  bool_to_string(bad_reshaping));
	    sc_rm(equations);
	}
	else {
	    debug(5, "translate_array_effect",
		  "System could not be normalized\n");
	}
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
	    make_action(formal_tac, UU), 
	    make_approximation(formal_tap, UU));
	debug(5, "translate_array_effect",
	      "good reshaping between %s and %s\n",
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

    tag formal_tac = action_tag(effect_action(formal_effect));
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
		 make_action(formal_tac, UU), 
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
		make_action(formal_tac, UU), 
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
	     make_action(formal_tac, UU), 
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
      pips_internal_error("It is assumed that parameters are passed by reference\n");
    }

    if (storage_formal_p(st))
    {
	/* find the corresponding argument and returns the reference */
	effect res = (*effect_dup_func)(e);
	int n = formal_offset(storage_formal(st));
	expression nth = EXPRESSION(gen_nth(n-1, call_arguments(c)));

	pips_assert("expression is a reference or read effect", 
		    effect_read_p(e) || expression_reference_p(nth));
	/* FI: a preference is forced here */
	effect_reference(res) = expression_reference(nth);

	le = CONS(EFFECT, res, NIL);
    }
    else if (storage_ram_p(st))
    {
	list le;
	le = global_effect_translation
	    (e, call_function(c), get_current_module_entity());
	/* le = proper_effects_contract(le); */
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
	    MAP(EFFECT, formal_effect,
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
			    
			    if (effect_must_p(formal_effect))
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
		}, 
		    func_sdfi);
	    
	    /* if everything is fine, then the effects are the read effects
	     * of the expression */
	    le = gen_nconc(generic_proper_effects_of_expression(expr), le);
	}
    
    }
    pips_debug(3, "effects on statics and globals\n");
/* effets of func on static and global variables are translated */
    if (get_bool_property("GLOBAL_EFFECTS_TRANSLATION"))
    {
	MAP(EFFECT, ef,
	    {
		if (storage_ram_p(entity_storage(effect_entity(ef)))) 
		    le = gen_nconc(global_effect_translation
				   (ef, func, get_current_module_entity()),le);
	    },
		func_sdfi);
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

/* Checked with Effects/call02.c */
effect effect_scalar_substitution(effect eff, entity ev)
{
  addressing ad = effect_addressing(eff);
  action ac = effect_action(eff);
  reference r = effect_any_reference(eff);
  effect n_eff = effect_undefined;

  if(action_write_p(ac) && addressing_index_p(ad) && ENDP(reference_indices(r))) {
    pips_user_warning
      ("Ineffective write effect with value passing mode for formal parameter \"%s\"\n",
       entity_user_name(reference_variable(r)));
  }
  else {
    n_eff = copy_effect(eff);
    reference r = effect_any_reference(n_eff);

    reference_variable(r) = ev;
  }

  return n_eff;
}

effect effect_array_substitution(effect eff, reference er)
{
  effect n_eff = effect_undefined;
  entity av = reference_variable(er);
  list aind = reference_indices(er);
  int ad = type_depth(entity_type(av)); /* In general, do not use
					   ultimate_type() with
					   type_depth() */
  int asn = gen_length(aind);
  reference fr = effect_any_reference(eff);
  entity fv = reference_variable(fr);
  list find = reference_indices(fr);
  int fsn = gen_length(find);

  if(asn+fsn==ad) {
    reference n_ref = copy_reference(er);

    reference_indices(n_ref) = gen_nconc(reference_indices(n_ref),
					 gen_full_copy_list(find));
    n_eff = make_effect(make_cell_reference(n_ref),
			copy_action(effect_action(eff)),
			copy_addressing(effect_addressing(eff)),
			copy_approximation(effect_approximation(eff)),
			make_descriptor_none());
  }
  else if(asn+fsn==ad+1) {
    /* One dimension should be merged: which test case? */
    reference n_ref = copy_reference(er);
    expression e1 = EXPRESSION(CAR(gen_last(reference_indices(n_ref))));
    expression e2 = EXPRESSION(CAR(reference_indices(fr)));
    value sv = EvalExpression(e1);
    constant sc = value_constant(sv);
    /* FI: could/should be PLUS_C ? We could try to evaluate s+l...*/
    expression sl = (constant_int_p(sc) && constant_int(sc)==0)?
      copy_expression(e2) : MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME), 
					   copy_expression(e1),
					   copy_expression(e2));

    EXPRESSION_(CAR(gen_last(reference_indices(n_ref)))) = sl;
    n_eff = make_effect(make_cell_reference(n_ref),
			copy_action(effect_action(eff)),
			copy_addressing(effect_addressing(eff)),
			copy_approximation(effect_approximation(eff)),
			make_descriptor_none());
  }
  else if(ad==0 && fsn==1 && asn==1) {
    /* av must be a pointer */
    type avt = ultimate_type(entity_type(av));
    if(pointer_type_p(avt)) {
      type pavt = ultimate_type(basic_pointer(variable_basic(type_variable(avt))));
      if(pointer_type_p(pavt)) {
	/* The actual argument is a pointer to a pointer. It can be
	   subscripted twice. */
	reference n_ref = copy_reference(er);

	reference_indices(n_ref) = gen_nconc(reference_indices(n_ref),
					 gen_full_copy_list(find));
	n_eff = make_effect(make_cell_reference(n_ref),
			    copy_action(effect_action(eff)),
			    copy_addressing(effect_addressing(eff)),
			    copy_approximation(effect_approximation(eff)),
			    make_descriptor_none());
      }
      else {
	pips_internal_error("Conversion of formal effect on \"%s\" into effect on \"%s\": "
			    "Not implemented yet\n", entity_name(fv), entity_name(av));
      }
    }
    else {
      pips_internal_error("Conversion of formal effect on \"%s\" into effect on \"%s\": "
			  "Not implemented yet\n", entity_name(fv), entity_name(av));
    }
  }
  else
    pips_internal_error("Conversion of formal effect on \"%s\" into effect on \"%s\": "
			"Not implemented yet\n", entity_name(fv), entity_name(av));
  return n_eff;
}

/* Checked with Effects/call01.c */
effect effect_scalar_address_substitution(effect eff, entity ev)
{
  addressing ad = effect_addressing(eff);
  action ac = effect_action(eff);
  reference r = effect_any_reference(eff);
  effect n_eff = effect_undefined;

  if(addressing_index_p(ad) && ENDP(reference_indices(r))) {
    /* FI: I'm confused here. Why do we discard read effects? */
    pips_user_warning
      ("Ineffective %s effect with value passing mode for formal parameter \"%s\"\n",
       action_write_p(ac)? "write":"read", entity_user_name(reference_variable(r)));
  }
  else if(addressing_index_p(ad) || addressing_post_p(ad)){
    n_eff = copy_effect(eff);
    reference r = effect_any_reference(n_eff);

    addressing_tag(effect_addressing(n_eff)) = is_addressing_index;
    reference_variable(r) = ev;
  }
  else {
    /* Pre-indexing: *((&p)[i]) = *(*(p+i)) = *(p[i]) if p is a constant array address?*/
    pips_user_warning("To be improved if possible...\n");
    n_eff = anywhere_effect(copy_action(ac));
  }

  return n_eff;
}

/* The actual argument is the address of the reference, &(er) */
effect effect_array_address_substitution(effect eff,
					 reference er)
{
  effect n_eff = effect_undefined;
  reference eff_r = effect_any_reference(eff);
  int eff_d = gen_length(reference_indices(eff_r));
  int er_d = gen_length(reference_indices(er));
  entity er_v = reference_variable(er);
  type er_t = entity_type(er_v);
  int d = type_depth(er_t);

  /* One special case: a contiguous vector of the argument array is touched */
  /* FI: these are not the regions yet! We are going to bump into the unbounded expression...*/
  if(eff_d==1 && er_d==d) {
    reference nr = copy_reference(er);
    expression s = EXPRESSION(gen_nth(d-1, reference_indices(nr)));
    expression l = copy_expression(EXPRESSION(CAR(reference_indices(eff_r))));
    value sv = EvalExpression(s);
    constant sc = value_constant(sv);
    /* FI: could/should be PLUS_C ? We could try to evaluate s+l...*/
    expression sl = (value_constant_p(sv) && constant_int_p(sc) && constant_int(sc)==0)?
      l : MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME), s, l);

    CAR(gen_last(reference_indices(nr))).p = (void *) sl;

    /* FI: too bad for the memory leaks? */
    n_eff = make_effect(make_cell_reference(nr),
			copy_action(effect_action(eff)),
			copy_addressing(effect_addressing(eff)),
			copy_approximation(effect_approximation(eff)),
			copy_descriptor(effect_descriptor(eff)));
  }
  else if(er_d+eff_d==d+1) {
    /* Offset in an array at call site and sub-array touched by
       callee. The common dimension must be merged. The remaining sub
       array indices are concatenated to the array reference. */
    reference nr = copy_reference(er);
    list inds = reference_indices(nr);
    expression s = EXPRESSION(CAR(gen_last(inds)));
    expression l = copy_expression(EXPRESSION(CAR(reference_indices(eff_r))));
    value sv = EvalExpression(s);
    constant sc = value_constant(sv);
    /* FI: could/should be PLUS_C ? We could try to evaluate s+l...*/
    expression sl = (constant_int_p(sc) && constant_int(sc)==0)?
      l : MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME), s, l);

    CAR(gen_last(reference_indices(nr))).p = (void *) sl;
    reference_indices(nr) = gen_nconc(reference_indices(nr),
				      gen_full_copy_list(CDR(reference_indices(eff_r))));
    /* Let's hope the descriptor is none, or it's going to be wrong... */
    n_eff = make_effect(make_cell_reference(nr),
			copy_action(effect_action(eff)),
			copy_addressing(effect_addressing(eff)),
			copy_approximation(effect_approximation(eff)),
			copy_descriptor(effect_descriptor(eff)));
  }
  else if(d==0 && er_d==1 && eff_d==1) {
    /* We must assume that er_v is a pointer because d==0 and er_d==1.
     The two subscript expressions must be added */
    type er_tu = ultimate_type(er_t);
    if(pointer_type_p(er_tu)) {
      reference nr = copy_reference(er);
      expression exp_r = EXPRESSION(CAR(reference_indices(nr)));
      expression exp_e = EXPRESSION(CAR(reference_indices(eff_r)));
      expression n_exp = expression_undefined;

      if(unbounded_expression_p(exp_e))
	n_exp = make_unbounded_expression();
      else {
	expression s = copy_expression(exp_r);
	expression l = copy_expression(exp_e);
	n_exp = MakeBinaryCall(entity_intrinsic(PLUS_OPERATOR_NAME), s, l);
      }
      n_eff = make_effect(make_cell_reference(nr),
			  copy_action(effect_action(eff)),
			  copy_addressing(effect_addressing(eff)),
			  copy_approximation(effect_approximation(eff)),
			  make_descriptor_none());
      free_expression(EXPRESSION(CAR(reference_indices(nr))));
      EXPRESSION_(CAR(reference_indices(nr))) = n_exp;
    }
    else {
      pips_internal_error("Not implemented yet\n");
    }
  }
  else
    pips_internal_error("Not implemented yet\n");

  return n_eff;
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
 */
list c_summary_effect_to_proper_effects(effect eff, expression real_arg) 
{
  list l_eff = NIL; /* the result */


  reference eff_ref = effect_any_reference(eff);
  list eff_ind = reference_indices(eff_ref);

  ifdebug(8)
    {
      pips_debug(8, "begin for real arg %s, and effect :\n", 
		 words_to_string(words_expression(real_arg)));
      print_effect(eff);		 
    }

  /* Whatever the real_arg may be if there is an effect on the sole value of the
     formal arg, it generates no effect on the caller side.
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
		   to the real reference */
		pips_debug(8, "effect on the pointed area : \n");
		new_eff = (* reference_to_effect_func)
		  (new_ref,
		   copy_action(effect_action(eff)));
		FOREACH(EXPRESSION, eff_ind_exp, eff_ind)
		  {
		    (*effect_add_expression_dimension_func)
		      (new_eff, eff_ind_exp);
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
	    pips_internal_error("Subscript not supported yet\n");
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
		l_eff = c_summary_effect_to_proper_effects
		  (eff, EXPRESSION(CAR(CDR(args))));
	      }
	    else if(ENTITY_ADDRESS_OF_P(real_op)) 
	      {
		expression arg1 = EXPRESSION(CAR(args));
		syntax s1 = expression_syntax(arg1);
		reference r1 = syntax_reference(s1);
		entity ev1 = reference_variable(r1);
		list l_real_arg = NIL;
		effect eff1, eff2;

		/* first we compute an effect on the argument of the 
		 address_of operator (to treat cases like &(n->m))*/
		/* It's very costly because we do not re-use l_real_arg 
		   We should maybe scan the real args instead of scanning 
		   the effects : we would do this only once per
		   real argument */
		l_real_arg = generic_proper_effects_of_complex_lhs
		  (arg1, &eff1, &eff2, effect_write_p(eff));
		if (effect_undefined_p(eff1))
		  n_eff =  anywhere_effect
		    (copy_action(effect_action(eff)));
		else
		  {
		    n_eff = eff1;
		    effect_approximation(n_eff) = 
		      copy_approximation(effect_approximation(eff));
		  }
		if (!effect_undefined_p(eff2))
		  free_effect(eff2);
		gen_free_list(l_real_arg);
		
		/* BC : This must certainely be improved 
		   only simple cases are handled .*/
		if(ENDP(reference_indices(effect_any_reference(n_eff)))) 
		  {
		    expression first_ind = EXPRESSION(CAR(eff_ind));
		  
		    /* The operand of & is a scalar expression */
		    /* the first index of eff reference (which must
		       be equal to 0) must not be taken into account.
		       The other indices must be appended to n_eff indices
		    */
		    
		    pips_assert("scalar case : the first index of eff must be equal to [0]", expression_equal_integer_p(first_ind, 0));
		    FOREACH(EXPRESSION, eff_ind_exp, CDR(eff_ind))
		      {
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
			(*effect_add_expression_dimension_func)
			  (n_eff, eff_ind_exp);
		      }
		  }
	      }
	    else if(ENTITY_POINT_TO_P(real_op)|| ENTITY_FIELD_P(real_op))
	      {
		list l_real_arg = NIL;
		effect eff1, eff2;
		/* first we compute an effect on the real_arg */
		/* It's very costly because we do not re-use l_real_arg 
		   We should maybe scan the real args instead of scanning 
		   the effects. */
		l_real_arg = generic_proper_effects_of_complex_lhs
		  (real_arg, &eff1, &eff2, effect_write_p(eff));
		if (effect_undefined_p(eff1))
		  n_eff =  anywhere_effect
		    (copy_action(effect_action(eff)));
		else
		  {
		    n_eff = eff1;
		    effect_approximation(n_eff) = 
		      copy_approximation(effect_approximation(eff));
		  }
		if (!effect_undefined_p(eff2))
		  free_effect(eff2);
		gen_free_list(l_real_arg);
		
		/* Then we add the indices of the effect reference */
		/* Well this is valid only in the general case :
		 * we should verify that types are compatible.
		 */
		FOREACH(EXPRESSION, ind, eff_ind)
		  {
		    (*effect_add_expression_dimension_func)(n_eff, ind);
		  
		  }
	      }
	    else if(ENTITY_MALLOC_SYSTEM_P(real_op)) {
	      n_eff = heap_effect(get_current_module_entity(),
				  copy_action(effect_action(eff)));
	    }
	    else {
	      /* We do not know what to do with the initial value */
	      n_eff = anywhere_effect(copy_action(effect_action(eff)));
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
	    pips_internal_error("va_arg() : should have been treated before\n");
	    break;
	  }
	case is_syntax_application :
	  {
	    pips_internal_error("Application not supported yet\n");
	    break;
	  }
	case is_syntax_range :
	  {
	    pips_user_error("Illegal effective parameter: range\n");
	    break;
	  }
	default:
	  pips_internal_error("Illegal kind of syntax\n");
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
		pel = gen_nconc
		  (pel, 
		   (*effect_to_store_independent_effect_list_func)
		   (effect_dup(eff), force_may_p));
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
	    /*first, we compute the effects on the expression as if it were 
	     * a lhs. */
	    list arg_eff_tmp = NIL;
	    list arg_eff = NIL;

	    if (! expression_constant_p(arg))
	      {
		arg_eff_tmp =  generic_proper_effects_of_any_lhs(arg);
		
		ifdebug(8)
		  {
		    pips_debug
		      (8, 
		       "begin considering expression %s , effects are :\n",
		       words_to_string(words_expression(arg)));
		    (* effects_prettyprint_func)(arg_eff_tmp);
		  }
		
		/* Then, we replace write effects on references with write effects
		 * on the pointed variable if it makes sense.
		 * we skip read effects since they have already been computed 
		 * before. */
		MAP(EFFECT, eff,
		    {
		      if (effect_write_p(eff))
			{
			  reference ref = effect_any_reference(eff);
			  entity ent = reference_variable(ref);
			  type t = entity_type(ent);
			  
			  switch (type_tag(t))
			    {
			    case is_type_variable :
			      {
				variable var = type_variable(t);
				int nb_inds = gen_length(variable_dimensions(var)); 
				pips_debug(8, "It's a variable\n");
				
				if ( nb_inds > 0)
				  {
				    /* it's an array  : let us generate write and "
				       read effects on the whole array 
				    This really is not GENERIC at all 
				    */
				    effect eff_read; 
				    effect eff_write =effect_dup(eff);
				    reference eff_ref = copy_reference(ref);
				    int i;
				    list ind = NIL;
				    pips_debug(8, "It's an array \n");

				    /* add the indices */
				    for (i=0; i<nb_inds; i++)
				      {
					ind = gen_nconc
					  (ind, 
					   CONS(EXPRESSION,
						make_unbounded_expression(),
						NIL));
				      }
				    eff_ref = effect_any_reference(eff_write);
				    reference_indices(eff_ref) = ind;
				    
				    eff_read = effect_dup(eff_write);
				    action_tag(effect_action(eff_read)) =
				      is_action_read;
				    arg_eff = gen_nconc
				      (arg_eff, CONS(EFFECT, eff_write,
						     CONS(EFFECT,eff_read,NIL)));
				    ifdebug(8)
				      {
					pips_debug(8, "generated effects are :\n");
					(* effects_prettyprint_func)(arg_eff);
				      }
				  }
				else if (basic_pointer_p(variable_basic(var))) 
				  {
				    effect eff_write;
				    effect eff_read;
				    
				    pips_debug(8, "Its a pointer\n");
				    /* it's a pointer : let us generate write and
				       read effects on the pointed structure
				       I think it can be simplified with the next
				       representation of effects
				       THIS IS NOT GENERIC, but it should be
				       easily made so with the future
				       representation of effects
				    */
				    eff_write = effect_dup(eff);
				    addressing_tag(effect_addressing(eff_write)) =
				      is_addressing_pre;
				    approximation_tag
				      (effect_approximation(eff_write)) =
				      is_approximation_may;
				    eff_read = effect_dup(eff_write);
				    action_tag(effect_action(eff_read)) =
				      is_action_read;
				    arg_eff = gen_nconc
				      (arg_eff, CONS(EFFECT, eff_write,
						     CONS(EFFECT,eff_read,NIL)));
				    ifdebug(8)
				      {
					pips_debug(8, "generated effects are :\n");
					(* effects_prettyprint_func)(arg_eff);
				      }
				    
				  }
				else
				  {
				    pips_debug(8,"No write effect on passed by value actual parameter %s\n", entity_name(ent));
				  }
				break;
			      } /* case is_type_variable */
			    case is_type_struct:
			    case is_type_union:
			    case is_type_enum:
			      {
				pips_user_warning("Effect on struct, union or enum not handled yet\n");
				break;
			      }
			    default: 
			      {
				pips_internal_error
				  ("effect entity is a statement, an area, a functional, a vararg, unknown or void \n");
			      }
			    } /* end of switch */
			} /* end of if (effect_write_p)*/
		    }, arg_eff_tmp);
		
		/* The next statement was here to avoid memory leaks. 
		   But it aborts ! I don't know why, maybe due to preferences 
		   As it should maybe disappear with the change of effects, 
		   I leave it as it is.*/ 
		/* effects_free(arg_eff_tmp);*/ 
		pel = gen_nconc(pel, arg_eff);
	      } /* if (! expression_constant_p)*/
	    
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
	         BC : It should be done later, because there can be several 
		 effects for one real arg entity.So it generates redundant 
		 effects that will need costly unioning.
	      */
	      pel = gen_nconc(pel, generic_proper_effects_of_expression(ep));
	      
	    } /* if(formal_parameter_p(v)) */
	  else 
	    {
	      /* This effect must be a global effect. It does not require
		 translation in C. However, it may not be in the scope of the caller. */
	      pel = gen_nconc(pel, CONS(EFFECT, copy_effect(eff), NIL));
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
		make_action(action, UU),\
		make_addressing_index(),\
		make_approximation(approximation, UU),\
		make_descriptor(is_descriptor_none,UU))

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
			(form, action_tag(effect_action(ef)), approx);

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

/**************************************************************************/

/* kept because of a use in effects_dynamic_elim : to be removed later BC */
/* Substitute, it possible, the formal parameter in eff by its
   effective value. Since eff is modified by side effects, a copy of
   the summary effect must be passed. */
/* FI: Just starting... */
/* FI: it might be better to return a list of effects including the
   read implied by the evaluation of ep */
effect c_summary_effect_to_proper_effect(effect eff,
					 expression ep) /* effective parameter */
{
  effect n_eff = effect_undefined;
  type ept = expression_to_type(ep);
  reference r = effect_any_reference(eff);
  entity fp = reference_variable(r);
  int dim = 0;

  /* This is going to fail in general: temporary fix for simple cases
     only, with arrays, but not with structures I guess */
  if(expression_reference_p(ep)) {
    reference er = expression_reference(ep);
    entity erv = reference_variable(er);
    list erind = reference_indices(er);
    type ervt = entity_type(erv);

    /* Are we dealing with a pointer or with an array element? */
    dim = type_depth(ervt) - gen_length(erind);
  }

  /* FI: we probably need a more general definition of "pointer"
     type. A partialy indexed array is a pointer... */
  if(pointer_type_p(ept) || dim>0) {
    syntax eps = expression_syntax(ep);
    if(syntax_reference_p(eps)) {
      reference er = syntax_reference(eps);
      entity ev = reference_variable(er);
      list eind = reference_indices(er);
      if(ENDP(eind))
	n_eff = effect_scalar_substitution(eff, ev);
      else
	n_eff = effect_array_substitution(eff, er);
    }
    else if(syntax_range_p(eps)) {
      pips_user_error("Illegal effective parameter: range\n");
    }
    else if(syntax_call_p(eps)) {
      call ec = syntax_call(eps);
      entity eop = call_function(ec);
      list args = call_arguments(ec);

      if(ENTITY_ADDRESS_OF_P(eop)) {
	expression arg1 = EXPRESSION(CAR(args));
	syntax s1 = expression_syntax(arg1);
	reference r1 = syntax_reference(s1);
	entity ev1 = reference_variable(r1);

	pips_assert("Operator \"address of\" is applied to a reference\n",
		    syntax_reference_p(s1));

	if(ENDP(reference_indices(r1))) {
	  n_eff = effect_scalar_address_substitution(eff, ev1);
	}
	else {
	  n_eff = effect_array_address_substitution(eff, r1);
	}
      }
      else if(ENTITY_POINT_TO_P(eop)) {
	pips_internal_error("not implemented yet\n");
      }
      else if(ENTITY_FIELD_P(eop)) {
	pips_internal_error("not implemented yet\n");
      }
      else if(ENTITY_MALLOC_SYSTEM_P(eop)) {
	n_eff = heap_effect(get_current_module_entity(),
			    copy_action(effect_action(eff)));
      }
      else {
	/* We do not know what to do with the initial value */
	n_eff = anywhere_effect(copy_action(effect_action(eff)));
      }
    }
    else if(syntax_cast_p(eps)) {
      /* Ignore the cast */
      cast c = syntax_cast(eps);
      pips_user_warning("Cast effect is ignored\n");
      n_eff = c_summary_effect_to_proper_effect(eff, cast_expression(c));
    }
    else if(syntax_sizeofexpression_p(eps)) {
      /* No translation possible */
      n_eff = effect_undefined;
    }
    else if(syntax_subscript_p(eps)) {
      pips_internal_error("Subscript not supported yet\n");
    }
    else if(syntax_application_p(eps)) {
      pips_internal_error("Application not supported yet\n");
    }
    else if(syntax_va_arg_p(eps)) {
      pips_internal_error("va_arg() not supported yet\n");
    }
    else {
      pips_internal_error("Illegal kind of syntax\n");
    }
  }
  else { /* We are not dealing with a pointer */
    if(action_write_p(effect_action(eff))) {
      pips_user_warning("Write effect on parameter \"%s\" ignored\n",
			entity_user_name(fp));
    }
    else {
      /* No need to translate... We might distinguish between useful
	 and useless effects? Suppose there is no read of the formal
	 parameter. Do we request effects to compute the value of the
	 effective parameter? */
      n_eff = effect_undefined;
    }
  }

  /* Is ep an address expression? */

  free_type(ept);
  return n_eff;
}
