/* package semantics
 * 
 * $Id$
 *
 * $Log: interprocedural.c,v $
 * Revision 1.32  2000/11/23 17:13:01  irigoin
 * New function for normalization, sc_safe_normalize(), to always retrieve a
 * consistent system of constraints
 *
 * Revision 1.31  1999/01/16 21:20:52  irigoin
 * Bug fix in intra_to_inter_precondition(): even non-feasible system must be
 * projected because of their possibly non-empty basis.
 *
 * Revision 1.30  1999/01/15 08:59:08  irigoin
 * Bug fix in precondition_intra_to_inter(): a non feasible transformer must
 * be preserved even though the intersection of the effects of the called
 * procedure and the basis of the transformer is empty (since the basis of
 * the transformer is empty).
 *
 * Revision 1.29  1999/01/14 17:27:43  irigoin
 * Kludge to avoid a useless costly projection in
 * precondition_intra_to_inter(). This was motivated by XERCTL in KIVA,
 * combustion code from Renault.
 *
 * Revision 1.28  1999/01/07 16:43:06  irigoin
 * Bug fix in translated_global_value() to resist aliasing between a global variable and a formal parameter. See spice01.f in Validation.
 *
 *
 */

#include <stdio.h>
#include <string.h>
/* #include <stdlib.h> */

#include "genC.h"


#include "linear.h"
#include "ri.h"
#include "ri-util.h"

#include "database.h"
#include "pipsdbm.h"
#include "resources.h"

#include "misc.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "transformer.h"

#include "semantics.h"

/* What follows has top be updated!!!
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
						TRUE);
    else
	p = transformer_undefined;

    /* FI: this does not work because the summary preconditions is reset
       each time it should be accumulated */
    /*
    if(check_resource_up_to_date(DBR_SUMMARY_PRECONDITION,
				 module_local_name(m))) {
	p = (transformer) db_get_memory_resource(DBR_SUMMARY_PRECONDITION,
						 module_local_name(m),
						 TRUE);
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
	debug(8,"add_module_call_site_precondition","begin\n");
	debug(8,"add_module_call_site_precondition","for module %s\n",
	      module_local_name(m));
	debug(8,"add_module_call_site_precondition",
	      "call site precondition %x:\n", p);
	/* p might not be printable; it may (should!) contain formal parameters
	   of module m */
	dump_transformer(p);
    }

    /* keep only the interprocedural part of p that can be easily used
       by m; this is non optimal because symbolic constants will be
       lost; this is due to value mappings; new and old values should be
       added to the mapping using the module precondition*/
    p = precondition_intra_to_inter(m, p, ef);

    ifdebug(8) {
	debug(8,"add_module_call_site_precondition",
	      "filtered call site precondition:\n");
	dump_transformer(p);
    }

    mp = get_module_precondition(m);

    ifdebug(8) {
	if (!transformer_undefined_p(mp)) {
	    debug(8,"add_module_call_site_precondition",
		  "old module precondition:\n");
	    dump_transformer(mp);
	}
	else
	   debug(8,"add_module_call_site_precondition",
		  "old module precondition undefined\n"); 
    }


    translate_global_values(get_current_module_entity(), p);
    ifdebug(8) {
	debug(8,"add_module_call_site_precondition",
	      "new module precondition in current frame:\n");
	dump_transformer(p);
    }
    
    if (!transformer_undefined_p(mp)) {

    /* convert global variables in the summary precondition in the local frame
     * as defined by value mappings (FI, 1 February 1994) */

    /* p is returned in the callee's frame; there is no need for a translation;
     * the caller's frame should always contain the callee's frame by definition
     * of effects;unfortunately, I do not remember *why* I added this translation;
     * it was linked to a problem encountered with transformer and "invisible"
     * variables, i.e. global variables which are indirectly changed by a procedure
     * which does not see them; such variables receive an arbitrary existing 
     * global name; they may receive different names in different context, because
     * there is no canonical name; each time, summary_precondition and summary_transformer
     * are used, they must be converted in a unique frame, which can only be the
     * frame of the current module.
     *
     * FI, 9 February 1994
     */
	translate_global_values(get_current_module_entity(), mp);
	ifdebug(8) {
	    debug(8,"add_module_call_site_precondition",
		  "old module precondition in current frame:\n");
	    dump_transformer(mp);
	}
	

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
	
    ifdebug(8) {
	debug(8,"add_module_call_site_precondition",
	      "new module precondition in current frame:\n");
	dump_transformer(new_mp);
    }

    DB_PUT_MEMORY_RESOURCE(DBR_SUMMARY_PRECONDITION, 
			   strdup(module_local_name(m)), 
			   (char*) new_mp );

    debug(8,"add_module_call_site_precondition","end\n");
}



/* returns a module's parameter's list */

list entity_to_formal_integer_parameters(f)
entity f;
{
    /* get unsorted list of formal integer parameters for f by declaration
       filtering; these parameters may not be used by the callee's
       semantics analysis, but we have no way to know it because
       value mappings are not available */

    list formals = NIL;
    list decl = list_undefined;

    pips_assert("entity_to_formal_parameters",entity_module_p(f));

    decl = code_declarations(entity_code(f));
    MAPL(ce, {entity e = ENTITY(CAR(ce));
	      if(storage_formal_p(entity_storage(e)) &&
		 entity_integer_scalar_p(e))
		  formals = CONS(ENTITY, e, formals);},
	 decl);

    return formals;
}

boolean 
integer_scalar_entity_list_p(l)
list l;
{
    boolean result= FALSE;
    if (!ENDP(l)) { 
	result= integer_scalar_entity_p(ENTITY(CAR(l)));
	MAP(ENTITY,el, { 
	    if (result) 
		result = result && integer_scalar_entity_p(el);
	},
	    CDR(l));
    }
    return result; 
}

/* add_formal_to_actual_bindings_to_precondition(call c, transformer pre):
 *
 * pre := pre  U  {f  = expr }
 *                  i       i
 * for all i such that formal fi is an integer scalar variable and 
 * expression expr-i is affine
 */
transformer add_formal_to_actual_bindings(c, pre)
call c;
transformer pre;
{
     entity f = call_function(c);
     list pc = call_arguments(c);
     list formals = entity_to_formal_integer_parameters(f);
     cons * ce;

     ifdebug(6) {
	 debug(6,"add_formal_to_actual_bindings",
	       "begin for call to %s pre=%x\n", module_local_name(f), pre);
	 dump_transformer(pre);
     }

     pips_assert("add_formal_to_actual_bindings", 
		 entity_module_p(f));
     pips_assert("add_formal_to_actual_bindings", 
		 pre != transformer_undefined);

     /* let's start a long, long, long MAPL, so long that MAPL is a pain */
     for( ce = formals; !ENDP(ce); POP(ce)) {
	 entity e = ENTITY(CAR(ce));
	 int r = formal_offset(storage_formal(entity_storage(e)));
	 expression expr;
	 normalized n;

	 if((expr = find_ith_argument(pc, r)) == expression_undefined)
	     user_error("add_formal_to_actual_bindings",
			"not enough args for formal parm. %d\n", r);

	 n = NORMALIZE_EXPRESSION(expr);
	 if(normalized_linear_p(n)) {
	     Pvecteur v = vect_dup((Pvecteur) normalized_linear(n));
	     entity e_new = external_entity_to_new_value(e);

	     vect_add_elem(&v, (Variable) e_new, -1);
	     pre = transformer_equality_add(pre, v);
	 }
     }

     free_arguments(formals);

     ifdebug(6) {
	 debug(6,"add_formal_to_actual_bindings",
	       "new pre=%x\n", pre);
	 dump_transformer(pre);
	 debug(6,"add_formal_to_actual_bindings","end for call to %s\n",
	       module_local_name(f));
     }

     return pre;
 }

transformer precondition_intra_to_inter(callee, pre, le)
entity callee;
transformer pre;
cons * le;
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
	    debug(DEBUG_PRECONDITION_INTRA_TO_INTER,"precondition_intra_to_inter",
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
	    debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
		  "precondition_intra_to_inter",
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
	    debug(DEBUG_PRECONDITION_INTRA_TO_INTER, "precondition_intra_to_inter",
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
    for(b = r->base; b != NULL; b = b->succ)
	values = arguments_add_entity(values, (entity) b->var);

    /* build a list of arguments to suppress; 
       get rid of variables that are not referenced, directly or indirectly,
       by the callee; translate what you can */
    for(ca = values; !ENDP(ca);  POP(ca))    {
	entity e = ENTITY(CAR(ca));
	list l_callee = (list) effects_conflict_with_entities(le, e);
	/* For clarity, all cases are presented */
	if (ENDP(l_callee)) {   /* no conflicts */
	    lost_values = arguments_add_entity(lost_values, e);
	    debug(DEBUG_PRECONDITION_INTRA_TO_INTER, 
		  "precondition_intra_to_inter",
		  "value %s lost according to effect list\n",
		  entity_name(e));
	}
	else {
	    /* list of conflicting entities */
	    entity e_callee = ENTITY(CAR(l_callee));
	    /* case 1: only one entity*/
	    if (gen_length(l_callee)==1) {
		/*  case 1.1: one conflicting integer entity */
		if (integer_scalar_entity_p(e_callee)) {
		    if(e_callee != e) {
			pre = transformer_value_substitute(pre, 
							   e, e_callee);
			ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER) {
			    debug(DEBUG_PRECONDITION_INTRA_TO_INTER, 
				  "precondition_intra_to_inter",
				  "value %s substituted by %s according to effect list le:\n", 
				  entity_name(e), entity_name(e_callee));
			    dump_arguments(lost_values);
			}
		    }
		}
		/* case 1.22: one conflicting non integer scalar entity*/
		else { 
		    lost_values = arguments_add_entity(lost_values, e);
		    debug(DEBUG_PRECONDITION_INTRA_TO_INTER, 
			  "precondition_intra_to_inter",
			  "value %s lost because non integer scalar entity\n",
			  entity_name(e));
		}
	    }   
	    else  { /* case 2: at least 2 conflicting entities */
		if (integer_scalar_entity_list_p(l_callee)) {
		    /* case 2.1: all entities have the same type, 
		       according to mapping_values the subtitution 
		       is made with the first list element e_callee*/
		    if(e_callee != e) {
			pre = transformer_value_substitute(pre, 
							   e, e_callee);
			ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER) {
			    debug(DEBUG_PRECONDITION_INTRA_TO_INTER, 
				  "precondition_intra_to_inter",
				  "value %s substituted by %s the first element list according to effect list le:\n", 
				  entity_name(e), entity_name(e_callee));
			    dump_arguments(lost_values);
			}
		    }
		}
		
		else { /* case 2.2:all entities do not have the same type*/ 
		    lost_values = arguments_add_entity(lost_values, e);
		    debug(DEBUG_PRECONDITION_INTRA_TO_INTER, 
			  "precondition_intra_to_inter",
			  "value %s lost - list of conflicting entities with different types\n",
			  entity_name(e));
		}
	    }
	}
	
    }

    preserved_values = arguments_difference(values, lost_values);
     
    ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER) {
	debug(DEBUG_PRECONDITION_INTRA_TO_INTER, "precondition_intra_to_inter",
	      "values lost because they do not appear in the effect list le:\n");
	dump_arguments(lost_values);
	debug(DEBUG_PRECONDITION_INTRA_TO_INTER, "precondition_intra_to_inter",
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

    ifdebug(DEBUG_PRECONDITION_INTRA_TO_INTER) 
	{
	    debug(DEBUG_PRECONDITION_INTRA_TO_INTER,"precondition_intra_to_inter",
		  "return pre=%x\n",pre);
	    (void) dump_transformer(pre);
	
	    debug(DEBUG_PRECONDITION_INTRA_TO_INTER,
		  "precondition_intra_to_inter","end\n");
	}
    

    return pre;
}

void translate_global_values(m, tf)
entity m;
transformer tf;
{
    Psysteme s = (Psysteme) predicate_system(transformer_relation(tf));
    /* a copy of sc_base(s) is needed because translate_global_value()
       modifies it at the same time */
    Pbase b = (Pbase) vect_dup(sc_base(s));
    Pbase bv;

    ifdebug(6) {
	debug(6, "translate_global_values", "Predicate for tf:\n");
	sc_fprint(stderr, s, dump_value_name);
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
	debug(7, "translate_global_value", "begin v = %s and tf = %x\n",
	      entity_name(v), tf);
    }


    if(v == NULL) {
	pips_error("translate_global_value", "Trying to translate TCST\n");
	return;
    }

    /* Filter out values *local* to the current module */
    if(value_entity_p(v)) {
	/* FI: to be modified to account for global values that have a name
	 * but that should nevertheless be translated on their canonical
	 * representant; this occurs for non-visible global variables
	 */
	/* FI: to be completed later... 3 December 1993
	entity var = value_to_variable(v);

	    debug(7, "translate_global_value", 
		  "%s is translated into %s\n",
		  entity_name(v), entity_name(e));
	    transformer_value_substitute(tf, v, e);
	    */

	debug(7, "translate_global_value", "end: No need to translate %s\n",
	      entity_name(v));
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
	debug(7, "translate_global_value", "end: No need to translate %s yet\n",
	      entity_name(v));
	return;
    }

    store = entity_storage(v);

    debug(7, "translate_global_value", "Trying to translate %s\n",
	  entity_name(v));

    if(!storage_ram_p(store)) {
	if(storage_rom_p(store)) {
	    debug(7, "translate_global_value", "%s is not translatable: store tag %d\n",
		  entity_name(v), storage_tag(store));
	    /* Should it be projected? No, this should occur later for xxxx#init
	     * variables when the xxxx is translated. Or before if xxxx has been
	     * translated 
	     */
	    return;
	}
	else 
	    if(storage_formal_p(store)) {
		debug(7, "translate_global_value", "formal %s is not translatable\n",
		      entity_name(v));
		return;
	    }
	    if(storage_return_p(store)) {
		debug(7, "translate_global_value", "return %s is not translatable\n",
		      entity_name(v));
		return;
	    }
	    else
		pips_error("translate_global_value", "%s is not translatable: store tag %d\n",
			   entity_name(v), storage_tag(store));
    }

    ifdebug(7) {
	debug(7, "translate_global_value", "let's do it for v = %s and tf =\n",
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

	/* try to find an equivalent entity by its name
	   (whereas we should use locations) */
	/*
	e = global_name_to_entity(module_local_name(m),
				  entity_local_name(v));
	e = value_alias(value_to_variable(v));
	*/
	e = value_alias(v);
	if(e == entity_undefined) {
	    /* no equivalent name found, get rid of v */
	    debug(7, "translate_global_value",
		  "No equivalent for %s in %s: project %s\n",
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
	    debug(7, "translate_global_value",
		  "No equivalent location for %s and %s: project %s\n",
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

	    debug(7, "translate_global_value", 
		  "%s has already been translated into %s\n",
		  entity_name(v), entity_name(e));

	    user_warning("translate_global_value",
			 "Variable %s is probably aliased with a formal parameter"
			 " by the current call to %s from %s.\n"
			 "This is forbidden by the Fortran 77 standard.\n",
			 entity_name(v), entity_module_name(v), module_local_name(m));

	    ifdebug(7) {
		debug(7, "translate_global_value", 
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
		debug(7, "translate_global_value", 
		  "After projection of %s\n",
		  entity_name(v));
		dump_transformer(tf);
	    }
	}
	else {
	    debug(7, "translate_global_value", 
		  "%s is translated into %s\n",
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
		    pips_error("translate_global_value",
			       "Cannot find value %s\n",
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
		    debug(7, "translate_global_value", 
		      "%s is not used in tf\n",
		      entity_name(v_init));
		}
	    }
	    else {
		debug(7, "translate_global_value", 
		      "%s is translated into %s\n",
		      entity_name(v), entity_name(e));
		transformer_value_substitute(tf, v_init, e_init);
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

/* FI: to be transferred into ri-util (should be used for effect translation
   as well) */
bool same_scalar_location_p(e1, e2)
entity e1;
entity e2;
{
    storage st1 = entity_storage(e1);
    storage st2 = entity_storage(e2);
    entity s1 = entity_undefined;
    entity s2 = entity_undefined;
    ram r1 = ram_undefined;
    ram r2 = ram_undefined;
    bool same = FALSE;

    /* e1 or e2 may be a formal parameter as shown by the benchmark m from CEA
     * and the call to SOURCE by the MAIN, parameter NPBF (FI, 13/1/93)
     *
     * I do not understand why I should return FALSE since they actually have
     * the same location for this call site. However, there is no need for
     * a translate_global_value() since the usual formal/actual binding
     * must be enough.
     */
    /*
     * pips_assert("same_scalar_location_p", storage_ram_p(st1) && storage_ram_p(st2));
     */
    if(!(storage_ram_p(st1) && storage_ram_p(st2)))
	return FALSE;

    r1 = storage_ram(entity_storage(e1));
    s1 = ram_section(r1);
    r2 = storage_ram(entity_storage(e2));
    s2 = ram_section(r2);

    if(s1 == s2) {
	if(ram_offset(r1) == ram_offset(r2))
	    same = TRUE;
	else {
	    debug(7, "same_scalar_location_p",
		  "Different offsets %d for %s in section %s and %d for %s in section %s\n",
		  ram_offset(r1), entity_name(e1), entity_name(s1),
		  ram_offset(r2), entity_name(e2), entity_name(s2));
	}
    }
    else {
	debug(7, "same_scalar_location_p",
	      "Disjoint entitites %s in section %s and %s in section %s\n",
	      entity_name(e1), entity_name(s1),
	      entity_name(e2), entity_name(s2));
    }

    return same;
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

    debug(8,"call_to_summary_precondition","begin\n");

    switch (tt = value_tag(entity_initial(e))) {

    case is_value_intrinsic:
	debug(5, "call_to_summary_precondition", "intrinsic function %s\n",
	      entity_name(e));
	/* propagate precondition pre as summary precondition 
	   of user functions */
 	expressions_to_summary_precondition(pre, args);
	break;

    case is_value_code:
	debug(5, "call_to_summary_precondition", "external function %s\n",
	      entity_name(e));
	pre_callee = transformer_dup(pre);
	pre_callee = 
	    add_formal_to_actual_bindings(c, pre_callee);
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
	pips_error("call_to_summary_precondition", "unknown function %s\n",
		   entity_name(e));
	break;

    default:
	pips_error("call_to_summary_precondition", "unknown tag %d\n", tt);
    }

    debug(8,"call_to_summary_precondition","end\n");

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
			      module_local_name(caller), TRUE));

    set_semantic_map((statement_mapping)
		     db_get_memory_resource
		     (DBR_PRECONDITIONS,
		      module_local_name(caller),
		      TRUE) );

    /* load caller preconditions */
    caller_prec = transformer_dup(load_statement_semantic(s));

    set_current_module_statement(s);

    /* first, we deal with the caller */
    set_current_module_entity(caller);
    /* create htable for old_values ... */
    module_to_value_mappings(caller);

    /* add to preconditions the links to the callee formal params */
    caller_prec = add_formal_to_actual_bindings (c, caller_prec);
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

    result = text_for_a_transformer(call_site_prec, FALSE);

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

static bool memorize_precondition(statement s)
{
  bool go_down = FALSE;

  if(statement_ordering(s)!=STATEMENT_ORDERING_UNDEFINED) {
    /* There may be or not a precondition associated to
     * this statement, depending on an intermediate disk
     * storage. However, if the ordering of statement s
     * is undefined, statement s is not reachable and the
     * corresponding call site should be ignored, as well
     * as call sites in statements controlled by s
     */
    current_precondition = load_statement_semantic(s);

    pips_assert("current precondition is defined",
		current_precondition!=transformer_undefined);

    go_down = TRUE;
  }
  else {
    current_precondition = transformer_undefined;
    go_down = FALSE;
  }

  return go_down;
}

/* Update the current_summary_precondition, if necessary */
static bool process_call(call c)
{
#define PROCESS_CALL_DEBUG_LEVEL 5

    transformer caller_prec = transformer_undefined;
    transformer call_site_prec = transformer_undefined;

    if(call_function(c) != current_callee) {
	return TRUE;
    }


    number_of_call_sites++;

    ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
	debug(PROCESS_CALL_DEBUG_LEVEL,"process_call","begin\n");
	debug(PROCESS_CALL_DEBUG_LEVEL,"Process_call","for module %s (call site %d)\n",
	      module_local_name(current_callee), number_of_call_sites);
	debug(PROCESS_CALL_DEBUG_LEVEL,"process_call",
	      "call site precondition %x:\n", current_precondition);
	/* p might not be printable; it may (should!) contain formal parameters
	   of module m */
	dump_transformer(current_precondition);
	debug(PROCESS_CALL_DEBUG_LEVEL,"process_call",
	      "current summary precondition %x:\n",
	      current_summary_precondition);
	/* p might not be printable; it may (should!) contain formal parameters
	   of module m */
	dump_transformer(current_summary_precondition);
    }

    /* add to call site preconditions the links to the callee formal params */
    caller_prec = add_formal_to_actual_bindings
	(c, transformer_dup(current_precondition));
    ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
	debug(PROCESS_CALL_DEBUG_LEVEL,"process_call",
	      "call site precondition with bindings %x:\n",
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
	debug(PROCESS_CALL_DEBUG_LEVEL,"process_call",
	      "call site precondition with filtered actual parameters:\n");
	dump_transformer(call_site_prec);
    }

    translate_global_values(current_caller, call_site_prec);

    ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
	debug(PROCESS_CALL_DEBUG_LEVEL,"process_call",
	      "new call site precondition in caller's frame:\n");
	dump_transformer(call_site_prec);
    }
    
    /* Provoque initialization with an undefined transformer... */
    /*pips_assert("process_call", !transformer_undefined_p(call_site_prec)); */

    if (!transformer_undefined_p(current_summary_precondition)) {

	/* convert global variables in the summary precondition in the
	 * caller's frame as defined by value mappings (FI, 1 February 1994)
	 */

	/* p is returned in the callee's frame; there is no need for 
	 * a translation; the caller's frame should always contain 
	 * the callee's frame by definition of effects;
	 *
	 * Unfortunately, I do not remember *why* I added this translation;
	 * It was linked to a problem encountered with transformer
	 * and "invisible" variables, i.e. global variables which
	 * are indirectly changed by a procedure which does not see them;
	 * such variables receive an arbitrary existing 
	 * global name; they may receive different names in different context,
	 * because there is no canonical name; each time, summary_precondition
	 * and summary_transformer are used, they must be converted in a 
	 * unique frame, which can only be the frame of the current module.
	 * In other words, you have to be in the same environment to
	 * be allowed to combine preconditions.
	 *
	 * FI, 9 February 1994
	 *
	 * This may be now useless...
	 */
	translate_global_values(current_caller, 
				current_summary_precondition);
	ifdebug(PROCESS_CALL_DEBUG_LEVEL) {
	    debug(PROCESS_CALL_DEBUG_LEVEL,"process_call",
		  "old module current summary precondition (%x) in current frame:\n",
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
	    pips_assert("process_call", 
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
	debug(PROCESS_CALL_DEBUG_LEVEL,"process_call: end\n",
	      "new module current summary precondition %x in current frame:\n",
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

    return TRUE;
}

transformer update_precondition_with_call_site_preconditions(transformer t, 
							     entity caller,
							     entity callee)
{
    statement caller_statement = (statement) db_get_memory_resource
	(DBR_CODE, module_local_name(caller), TRUE);
    /* summary effects for the callee */
    summary_effects_of_callee = load_summary_effects(callee);

    pips_assert("update_precondition_with_call_site_preconditions",
		get_current_module_entity() == callee);

    reset_current_module_entity();
    set_current_module_entity(caller);
    set_current_module_statement(caller_statement);
    current_summary_precondition = t;
    current_caller = caller;
    current_callee = callee;
	    
    set_cumulated_rw_effects((statement_effects)
			       db_get_memory_resource
			       (DBR_CUMULATED_EFFECTS,
				module_local_name(caller), TRUE));

    set_semantic_map((statement_mapping)
		     db_get_memory_resource
		     (DBR_PRECONDITIONS,
		      module_local_name(caller),
		      TRUE) );

    module_to_value_mappings(caller);

    gen_multi_recurse(caller_statement,
		      statement_domain, memorize_precondition, gen_null,
		      call_domain, process_call, gen_null,
		      NULL);

    free_value_mappings();
    reset_current_module_entity();
    reset_current_module_statement();
    reset_cumulated_rw_effects();
    reset_semantic_map();
    set_current_module_entity(callee);

    current_caller = entity_undefined;
    current_callee = entity_undefined;
    current_precondition = transformer_undefined;
    summary_effects_of_callee = list_undefined;
    /* FI: I believed that convex hull was performed by side effect only.
     * The two arguments are potentially modified, but a new transformer
     * is allocated
     */
    /*
    pips_assert("update_precondition_with_call_site_preconditions",
		t == transformer_undefined || 
		t == current_summary_precondition);
		*/
    /* FI: To be tried when I'm courageous!
    if(t!=transformer_undefined)
	free_transformer(t);
	*/
    t = current_summary_precondition;
    current_summary_precondition = transformer_undefined;
    return t;
}
