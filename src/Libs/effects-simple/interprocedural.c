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
    transformer context)
{
    return summary_to_proper_effects(func, real_args, l_eff);
}


/****************************************************************************/

/* list effects_dynamic_elim(list l_reg)
 * input    : a list of effects.
 * output   : a list of effects in which effects of dynamic variables are
 *            removed, and in which dynamic integer scalar variables are 
 *            eliminated from the predicate.
 * modifies : nothing; the effects l_reg initially contains are copied 
 *            if necessary.
 * comment  :	
 */
list 
effects_dynamic_elim(list l_eff)
{
    list l_res = NIL;

    MAP(EFFECT, eff, 
     {
        entity eff_ent = effect_entity(eff);
        storage eff_s = entity_storage(eff_ent);
        boolean ignore_this_effect = FALSE;

	ifdebug(4)
	{
	    pips_debug(4, "current effect: \n%s\n", effect_to_string(eff));
	}

	/* If the reference is a common variable (ie. with storage ram but
	 * not dynamic) or a formal parameter, the effect is not ignored.
	 */
        switch (storage_tag(eff_s))
	{
	case is_storage_return:
	    pips_debug(5, "return var ignored (%s)\n", entity_name(eff_ent));
	    ignore_this_effect = TRUE;
	    break;
	case is_storage_ram:
	{
	    ram r = storage_ram(eff_s);
	    if (dynamic_area_p(ram_section(r)) || heap_area_p(ram_section(r))
		|| stack_area_p(ram_section(r)))
	    {
		pips_debug(5, "dynamic or pointed var ignored (%s)\n", 
			   entity_name(eff_ent));
		ignore_this_effect = TRUE;
	    }
	    break;
	}
	case is_storage_formal:
	    break;
	case is_storage_rom:
	    pips_internal_error("bad tag for %s (rom)\n", 
				entity_name(eff_ent));
	default:
	    pips_internal_error("case default reached\n");
        }
	
        if (! ignore_this_effect)  /* Eliminate dynamic variables. */
	{
	    effect eff_res = make_sdfi_effect(eff);
	    ifdebug(4)
	    {
		pips_debug(4, "effect kept : \n\t %s\n", 
			   effect_to_string(eff_res));
	    }
            l_res = CONS(EFFECT, eff_res, l_res);
        }
	else 
	    ifdebug(4)
	    {
		pips_debug(4, "effect removed : \n\t %s\n", 
			   effect_to_string(eff));
	    }
    },
	l_eff);

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
    entity source_func,
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
	return CONS(EFFECT, make_sdfi_effect(ef), NIL);

    if (io_entity_p(eff_ent))
	return CONS(EFFECT, make_sdfi_effect(ef), NIL);

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
	target_func = local_name_to_top_level_entity(
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
		/* if the new entity is entirely contained in the original one
		 */
		if ((new_ent_begin_offset >= eff_ent_begin_offset) && 
		    (new_ent_end_offset <= eff_ent_end_offset)) 
		{
		    new_eff = 
			make_simple_effect
			(make_reference(new_ent, NIL), /* ??? memory leak */
			 make_action(action_tag(effect_action(ef)), UU), 
			 make_approximation
			 (approximation_tag(effect_approximation(ef)), UU));
		}
		/* If they only have some elements in common */
		else
		{						
		    new_eff = 
			make_simple_effect
			(make_reference(new_ent, NIL), /* ??? memory leak */
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

    entity formal_var = reference_variable(effect_reference(formal_effect));
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
	sc_fprint(stderr, equations, vect_debug_entity_name);
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
	vect_fprint(stderr, size_formal, vect_debug_entity_name);
	fprintf(stderr, "size of real: ");
	vect_fprint(stderr, size_real, vect_debug_entity_name);
	if(offset != (Pvecteur) -1) {
	    fprintf(stderr, "offset: ");
	    vect_fprint(stderr, offset, vect_debug_entity_name);
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
	    vect_fprint(stderr, ineq, vect_debug_entity_name);
	    fprintf(stderr, "\nsysteme a prouver: ");
	    sc_fprint(stderr, equations, vect_debug_entity_name);
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
    entity formal_var = reference_variable(effect_reference(formal_effect));
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
	pips_debug(8," translation failed\n");
        /* translation failed; returns the whole real arg */
        real_effect = make_simple_effect
	    (make_reference(real_var, NIL), /* ??? memory leak */
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



/* I developped this one afther noticing that the next one was useless. FC.
 */
list /* of effect */
summary_effect_to_proper_effect(
    call c,
    effect e)
{
    entity var = effect_variable(e);
    storage st = entity_storage(var);

    if (storage_formal_p(st))
    {
	/* find the corresponding argument and returns the reference */
	effect res = (*effect_dup_func)(e);
	int n = formal_offset(storage_formal(st));
	expression nth = EXPRESSION(gen_nth(n-1, call_arguments(c)));

	pips_assert("expression is a reference or read effect", 
		    effect_read_p(e) || expression_reference_p(nth));
	effect_reference(res) = expression_reference(nth);

	return CONS(EFFECT, res, NIL);
    }
    else if (storage_ram_p(st))
    {
	list le;
	le = global_effect_translation
	    (e, call_function(c), get_current_module_entity());
	/* le = proper_effects_contract(le); */
	return le;
    }
    return NIL;
}

/* argh! the translation is done the other way around...
 * from the call effects are derived and checked with summary ones,
 * while I expected the summary proper effects to be translated 
 * into proper effects considering the call site... 
 * the way it is done makes it useless to any other use
 * (for instance to translate reduction references)
 * Thus I have to develop my own function:-(
 */
list /* of effect */
summary_to_proper_effects(
    entity func,
    list /* of expression */ args,
    list /* of effect */ func_sdfi)
{
    list pc, le = NIL, l_formals;
    int ipc, n_formals;

    pips_debug(3, "effects on formals on call to %s\n", entity_name(func));

    /* check the number of parameters */
    l_formals = module_formal_parameters(func);
    n_formals = gen_length(l_formals);
   

    if (gen_length(args) < n_formals)
    {
	/* this is really a user error.
	 * if you move this as a user warning, the pips would drop
	 * effects about unbounded formals... why not? FC.
	 */
        fprintf(stderr,"%d formal arguments for module %s:\n",
		n_formals,module_local_name(func));
	dump_arguments(l_formals);
	fprintf(stderr,"%d actual arguments:\n",gen_length(args));
	print_expressions(args);
	pips_user_error("\nCall to module %s: "
			  "insufficient number of actual arguments.\n",
			  module_local_name(func));
    }
    else if (gen_length(args) > n_formals)
    {
	/* This can be survived... */        
      fprintf(stderr,"%d formal arguments for module%s:\n",
	      n_formals,module_local_name(func));
	dump_arguments(l_formals);
	fprintf(stderr,"%d actual arguments:\n",gen_length(args));
	print_expressions(args);

	pips_user_warning("\nCall to module %s: "
			  "too many actual arguments.\n",
			  module_local_name(func));
    }

    gen_free_list(l_formals), l_formals=NIL;
    /* effets of func on formal variables are translated */
    for (pc = args, ipc = 1; ! ENDP(pc) && ipc<=n_formals; pc = CDR(pc), ipc++)
    {
        expression expr = EXPRESSION(CAR(pc));
        syntax sexpr = expression_syntax(expr);
	boolean substringp = FALSE;
	boolean refp = FALSE;
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
				pips_user_error
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
    

/****************************************************** FORWARD TRANSLATION */

#define make_translated_effect(entity,action,approximation)\
    make_effect(make_cell(is_cell_reference, make_reference((entity), NIL)),\
		make_action(action, UU),\
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
    transformer context)
{
    list /* of effect */ lr, lc;

    pips_debug(4, "forward translation of %s call to %s\n",
	       entity_name(get_current_module_entity()),
	       entity_name(callee));

    lr = real_simple_effects_forward_translation(callee, real_args, l_eff);
    lc = common_simple_effects_forward_translation(callee, l_eff);

    return gen_nconc(lr, lc);
}
