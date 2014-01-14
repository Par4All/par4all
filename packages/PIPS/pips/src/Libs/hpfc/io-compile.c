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
/* HPFC module by Fabien COELHO
 */

#include "defines-local.h"

#include "semantics.h"
#include "conversion.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

/************************************************* IO EFFICIENT COMPILATION */



/* OUT Regions(effects) are used to verify if the entity current_entity is 
read in another statement later
(note that we don't need to know which statement is reading current_entity)*/

static bool
current_entity_is_used_later_p(statement stat, entity current_entity)
{
    list list_out; 

    if (get_bool_property("HPFC_IGNORE_IN_OUT_REGIONS"))
	/*IN & OUT Regions are not used*/
	return true;


/* get OUT Regions(Effects) list for the io statement */
/*    list_out = load_statement_out_regions(stat);*/

    list_out = load_out_effects_list(stat);

    ifdebug(2)
    {
	pips_debug(3, "OUT regions: \n");
	print_regions(list_out);
    }

/* for all OUT Regions (Effects) for the statement stat, 
 test if reference in current_effect is egal to current_entity */

    MAP(EFFECT, current_effect,
	{
	    if (effect_variable(current_effect)==current_entity)
		return true;
	},
	list_out);    

    return false;
}

/* IN Regions(effects) are used to verify if the entity current_entity is 
initialized before it's used in the io statement*/

static bool
current_entity_is_updated_before_p(statement stat, entity current_entity)
{
    list list_in; 

    if (get_bool_property("HPFC_IGNORE_IN_OUT_REGIONS"))	/* & OUT Regions are not used*/
	return true;

/* get IN Regions(Effects) list for the io statement */
/*    list_in = load_statement_in_regions(stat);*/

    list_in = load_in_effects_list(stat);

    ifdebug(2)
    {
	pips_debug(3, "IN regions: \n");
	print_regions(list_in);
    }

/* for all IN Regions (Effects) for the statement stat, 
 test if reference in current_effect is egal to current_entity */

    MAP(EFFECT, current_effect,
	{
	    if (effect_variable(current_effect)==current_entity)
		return true;
	},
	list_in);

    return false;
}

static Psysteme 
statement_context(
    statement stat,
    tag move)
{
    return predicate_system(transformer_relation
			    ((movement_collect_p(move)) ? 
			     load_statement_precondition(stat) :
			     load_statement_postcondition(stat))); 
}

void
hpfc_algorithm_row_echelon(
    Psysteme syst,
    list scanners,
    Psysteme *pcond, 
    Psysteme *penum)
{
    Pbase base = entity_list_to_base(scanners);
    algorithm_row_echelon_generic(syst, base, pcond, penum,
	get_bool_property("HPFC_REDUNDANT_SYSTEMS_FOR_REMAPS"));
    /* this should improve conditions? */
    sc_find_equalities(pcond);
    base_rm(base);
}


list /* of entity */
make_list_of_dummy_variables(
    entity (*creation)(),
    int number)
{
    list result = NIL;

    for(;number>0;number--)
	result = CONS(ENTITY, creation(number), result);

    return result;
}

static Psysteme 
generate_shared_io_system(
    entity array,
    statement stat,
    tag move, 
    tag act)
{
    Psysteme
	result = SC_UNDEFINED, /* ??? bug post with region */
	region = effect_system(entity_to_region(stat, array, act)), 
	a_decl = entity_to_declaration_constraints(array, 0),
	stamme = hpfc_unstutter_dummies(array),
	contxt = statement_context(stat, move);

    pips_assert("distributed array", !array_distributed_p(array));

    result = sc_append(sc_rn(NULL), region);
    result = sc_append(result, a_decl);
    result = sc_append(result, stamme);
    result = sc_append(result, contxt);
    
    DEBUG_SYST(8, concatenate("whole: ", entity_name(array), NULL), result);
    
    /* the noisy system is cleaned
     * some variables are not used, they are removed here.
     */
    sc_nredund(&result);
    base_rm(sc_base(result));
    sc_base(result) = NULL;
    sc_creer_base(result);
    
    DEBUG_SYST(7, "region", region);
    DEBUG_SYST(7, "array declaration", a_decl);
    DEBUG_SYST(7, "unstammer", stamme);
    DEBUG_SYST(7, "context", contxt);
    
    DEBUG_SYST(6, concatenate("result: ", entity_name(array), NULL), result);
    
    return result;
}

static Psysteme 
generate_distributed_io_system(
    entity array,
    statement stat,
    tag move, 
    tag act)
{
    Psysteme result = SC_UNDEFINED,
	/* ??? bug: the preconditions may be in the regions.  To update, I
	 * should have the postconditions instead, that is the statement
	 * transformer should be applied to the system.
	 */
	region = effect_system(entity_to_region(stat, array, act)),
	dist_v = generate_system_for_distributed_variable(array),
	sother = hpfc_compute_unicity_constraints(array), /* ??? */
	stamme = hpfc_unstutter_dummies(array),
	contxt = statement_context(stat, move);
    
    /* ??? massive memory leak 
       polyhedron intersections
*/
    result = sc_append(sc_rn(NULL), region);
    result = sc_append(result, dist_v);
    result = sc_append(result, sother);
    result = sc_append(result, stamme);
    result = sc_append(result, contxt);
    
    DEBUG_SYST(8, concatenate("whole: ", entity_name(array), NULL), result);
    
    /* the noisy system is cleaned
     * some variables are not used, they are removed here.
     */
    build_sc_nredund_2pass(&result);
    base_rm(sc_base(result));
    sc_base(result) = NULL;
    sc_creer_base(result);
    
    DEBUG_SYST(9, "region", region);
    DEBUG_SYST(9, "array syst", dist_v);
    DEBUG_SYST(9, "hpf unicity", sother);
    DEBUG_SYST(9, "unstammer", stamme);
    DEBUG_SYST(9, "context", contxt);
    
    DEBUG_SYST(6, concatenate("result: ", entity_name(array), NULL), result);
    
    return(result);
}

void 
remove_variables_if_possible(
    Psysteme *psyst,
    list *plvars)
{
    Psysteme syst = *psyst;
    list kept = NIL;

    MAP(ENTITY, e, 
    {
	Variable var = (Variable) e;
	Value coeff = VALUE_MONE;
	
	(void) contrainte_var_min_coeff(sc_egalites(syst), var, &coeff, false);
	
	if (value_one_p(coeff))
	{
	    Pvecteur v = vect_new(var, VALUE_ONE);
	    bool exact = true;
	    
	    pips_debug(7, "removing variable %s\n", 
		       entity_local_name((entity) var));
	    
	    sc_projection_along_variables_with_test_ofl_ctrl
		(&syst, v, &exact, NO_OFL_CTRL);

	    pips_assert("exact projection", exact);
	    vect_rm(v);
	}
	else
	    kept = CONS(ENTITY, (entity) var, kept);
    }, 
	*plvars);
    
    base_rm(sc_base(syst)), sc_base(syst) = BASE_NULLE, sc_creer_base(syst);
    
    *psyst = syst;
    gen_free_list(*plvars), *plvars=kept;
}

static Psysteme
clean_shared_io_system(
    Psysteme syst,
    entity array,
    tag move)
{
    int	array_dim = NumberOfDimension(array);
    list keep = NIL, try_keep = NIL, remove = NIL,
         try_remove = base_to_list(sc_base(syst));

    pips_debug(5, "array %s, movement %s\n", entity_local_name(array), 
	       (movement_collect_p(move))?"collect":"update");

    /* ALPHA_i's
     * PHI_i's
     */
    add_to_list_of_vars(keep, get_ith_array_dummy, array_dim);
    add_to_list_of_vars(remove, get_ith_region_dummy, array_dim);
    
    /*   keep parameters !
     */
    MAP(ENTITY, e,
     {
	 const char* s = entity_module_name(e);

	 if (strcmp(s, HPFC_PACKAGE) && strcmp(s, REGIONS_MODULE_NAME))
	     keep = CONS(ENTITY, e, keep);
     },
	 try_remove);

    /*   others
     */
    gen_remove(&try_remove, (entity) TCST);
    MAP(ENTITY, e, gen_remove(&try_remove, e), keep);
    MAP(ENTITY, e, gen_remove(&try_remove, e), try_keep);
    MAP(ENTITY, e, gen_remove(&try_remove, e), remove);
    
    /*    remove variables that have to be removed
     */
    MAP(ENTITY, e, 
	sc_projection_along_variable_ofl_ctrl(&syst, (Variable) e, NO_OFL_CTRL),
	remove);
    
    /* Try to remove other unusefull variables
     */
    remove_variables_if_possible(&syst, &try_remove);

    /* the noisy system is cleaned
     * some variables are not used, they are removed here.
     */
    build_sc_nredund_2pass(&syst);
    base_rm(sc_base(syst));
    sc_base(syst) = BASE_NULLE;
    sc_creer_base(syst);

    DEBUG_SYST(6, entity_name(array), syst);
    
    return(syst);
}

void 
remove_variables_from_system(
    Psysteme *ps,
    list /* of entity (Variable) */ *plv)
{
    MAP(ENTITY, e,
    {
	pips_debug(8, "removing variable %s\n", entity_local_name(e));
	sc_projection_along_variable_ofl_ctrl(ps, (Variable) e, NO_OFL_CTRL);
    },
	*plv);

    gen_free_list(*plv), *plv=NIL;
}

void 
clean_the_system(
    Psysteme *ps,
    list /* of entities */ *plrm,
    list *pltry)
{
    remove_variables_from_system(ps, plrm);
    remove_variables_if_possible(ps, pltry);

    build_sc_nredund_2pass(ps);
    base_rm(sc_base(*ps)), sc_creer_base(*ps);
}

static Psysteme 
clean_distributed_io_system(
    Psysteme syst, /* system to be cleaned */
    entity array,  /* io on array */
    tag move)      /* collect or update */
{
    /* ??? what about the variables?
     * some are usefull, some are constants, and others should
     * be discarded. This selection and projection may be done here.
     */
    /* to be removed:
     * PHIi...
     * THETAi...
     * some GAMMAi...
     * some others coming from the *conditions
     *
     * loop generation on:
     * PSIi...
     * some GAMMAi...
     * some DELTAi...
     * complementary ALPHAi... (LALPHAi?)
     */
    entity
	template = array_to_template(array),
	processor = template_to_processors(template);
    int
	array_dim = NumberOfDimension(array),
	template_dim = NumberOfDimension(template),
	processor_dim = NumberOfDimension(processor);
    list
	keep = NIL, try_keep = NIL, remove = NIL,
	try_remove = base_to_list(sc_base(syst));

    pips_debug(5, "array %s, movement %s\n", entity_local_name(array), 
	       (movement_collect_p(move))?"collect":"update");

    pips_assert("distributed array", array_distributed_p(array));
    
    /* THETA_i's
     * PHI_i's
     * PSI_i's
     * ALPHA_i's
     * LALPHA_i's 
     */
    add_to_list_of_vars(remove, get_ith_template_dummy, template_dim);
    add_to_list_of_vars(remove, get_ith_region_dummy, array_dim);
    add_to_list_of_vars(keep, get_ith_processor_dummy, processor_dim);
    add_to_list_of_vars(keep, get_ith_array_dummy, array_dim);
    add_to_list_of_vars(try_keep, get_ith_local_dummy, array_dim);

    /*   Keep parameters !
     */
    MAP(ENTITY, e,
    {
	const char* s = entity_module_name(e);
	
	if (strcmp(s, HPFC_PACKAGE) && strcmp(s, REGIONS_MODULE_NAME))
	    keep = CONS(ENTITY, e, keep);
	/* should try to remove only variables that can be eliminated
	 * using the preconditions. Otherwise: ph1=old, 1<=old<=2,
	 * projecting old loses the right dimension...
	 */
	/* try_remove = CONS(ENTITY, e, try_remove); */
    },
	try_remove);

    /*   others
     */
    gen_remove(&try_remove, (entity) TCST);
    MAP(ENTITY, e, gen_remove(&try_remove, e), keep);
    MAP(ENTITY, e, gen_remove(&try_remove, e), try_keep);
    MAP(ENTITY, e, gen_remove(&try_remove, e), remove);

    DEBUG_ELST(7, "keep", keep);
    DEBUG_ELST(7, "try_keep", try_keep);
    DEBUG_ELST(7, "try_remove", try_remove);
    DEBUG_ELST(7, "remove", remove);

    clean_the_system(&syst, &remove, &try_remove);
    
    DEBUG_SYST(6, entity_name(array), syst);
    
    gen_free_list(keep);
    gen_free_list(try_keep);
    gen_free_list(try_remove);

    return(syst);
}

/* Variables of Psysteme syst are ordered and put in different
 * lists. Especially, deducable variables are listed, the equalities
 * that allow to rebuild them are also listed, and they are removed
 * from the original system by *exact* integer projection.
 *
 * Other variables are (should be) the parameters, the processors,
 * and the variables to be used to scan polyhedron.
 */
static void 
put_variables_in_ordered_lists(
    Psysteme *psyst,
    entity array,
    list *plparam,
    list *plproc,
    list *plscan,
    list *plrebuild)
{
    int processor_dim = (array_distributed_p(array) ?
	    NumberOfDimension(array_to_processors(array)) : 0 ),
	dim = -1;
    list
	all = base_to_list(sc_base(*psyst)),
	lparam = NIL, lproc = NIL, lscan = NIL, lrebuild = NIL;

    gen_remove(&all, (entity) TCST); /* just in case */

    pips_debug(5, "considering %zd variables\n", gen_length(all));

    /* parameters: those variables that are not dummies...
     */
    MAP(ENTITY, v,
	if (!entity_hpfc_dummy_p(v)) lparam = CONS(ENTITY, v, lparam),
	all);

    MAP(ENTITY, e, gen_remove(&all, e), lparam);
    
    /* processors
     */
    for(dim=processor_dim; dim>=1; dim--)
    {
	entity
	    dummy = get_ith_processor_dummy(dim);

	lproc = CONS(ENTITY, dummy, lproc);
	gen_remove(&all, dummy);
    }

    /* scanners and deducables
     */
    lrebuild = 
	simplify_deducable_variables(*psyst,
				     gen_nreverse(hpfc_order_variables(all,
								       true)),
				     &lscan);

    /* return results
     */
    *plparam 	= lparam, 
    *plproc 	= lproc, 
    *plscan 	= lscan, /* lscan is implicitely ordered */
    *plrebuild 	= lrebuild;

    gen_free_list(all);

    DEBUG_ELST(4, "params", lparam);
    DEBUG_ELST(4, "procs", lproc);
    DEBUG_ELST(4, "scanners", lscan);

    ifdebug(4)
    {
	Pcontrainte pc = contrainte_make(VECTEUR_NUL);

	fprintf(stderr, "deducables:\n   ");
	MAP(EXPRESSION, ex,
	{
	    pc->vecteur = normalized_linear(expression_normalized(ex));
	    
	    fprintf(stderr, "%s rebuilt with ", entity_local_name
		    (reference_variable(expression_reference(ex))));
	    egalite_fprint(stderr, pc, (string(*)(Variable))entity_local_name);
	},
	    lrebuild);
    }

    build_sc_nredund_2pass(psyst);
    sc_base(*psyst) = (base_rm(sc_base(*psyst)), BASE_NULLE);
    sc_creer_base(*psyst);

    DEBUG_SYST(4, entity_name(array), *psyst);
}

/* list simplify_deducable_variables(syst, vars, pleftvars)
 * Psysteme syst;
 * list vars, *pleftvars;
 *
 * variables from entity list vars that can be rebuilt by the Psysteme
 * syst are removed from it and stored as an expression list which is
 * returned. The variables that are not removed are returned as another
 * entity list, *pleftvars.
 */
list /* of expression */
simplify_deducable_variables(
    Psysteme syst,
    list vars,
    list *pleftvars)
{
    list result = NIL;
    *pleftvars = NIL;

    MAP(ENTITY, dummy,
    {
	Pcontrainte eq = CONTRAINTE_UNDEFINED;
	Value coeff = VALUE_ZERO;
	
	 if (eq = eq_v_min_coeff(sc_egalites(syst), (Variable) dummy, &coeff),
	     value_one_p(coeff))
	 {
	     result = 
		 CONS(EXPRESSION,
		      make_expression(make_syntax(is_syntax_reference,
						  make_reference(dummy, NIL)),
				      make_normalized(is_normalized_linear,
						      vect_dup(eq->vecteur))), 
		      result);

	     syst = sc_variable_substitution_with_eq_ofl_ctrl(syst, eq, 
							      (Variable) dummy,
							      FWD_OFL_CTRL);
	 }
	 else
	 {
	     *pleftvars = CONS(ENTITY, dummy, *pleftvars);
	 }
     },
	 vars);

    base_rm(sc_base(syst)), sc_base(syst) = BASE_NULLE, sc_creer_base(syst);

    return(result);
}

/* output 7 entities created by creation if in list le
 */
static list /* of entity */
hpfc_order_specific_variables(
    list le,
    entity (*creation)())
{
    list result = NIL;
    int i;

    for(i=7; i>=1; i--)
    {
	entity dummy = creation(i);

	if (gen_in_list_p(dummy, le))
	    result = CONS(ENTITY, dummy, result);
    }

    return(result);
}

/* list hpfc_order_variables(list)
 *
 * the input list of entities is ordered so that:
 * PSI_i's, GAMMA_i's, DELTA_i's, IOTA_i's, ALPHA_i's, LALPHA_i's...
 */
list 
hpfc_order_variables(
    list le,
    bool number_first)
{
    list result = NIL;

    result =
	gen_nconc(result,
		  hpfc_order_specific_variables(le, get_ith_processor_dummy));
    
    if (number_first)
    {
	int i;
	list l = NIL, lr = NIL;

	for (i=7; i>0; i--)
	    l = CONS(ENTITY, get_ith_array_dummy(i),
		CONS(ENTITY, get_ith_shift_dummy(i),
		CONS(ENTITY, get_ith_block_dummy(i),
		CONS(ENTITY, get_ith_cycle_dummy(i),
		       l))));

	MAP(ENTITY, e,
	{
	    if (gen_in_list_p(e, le)) lr = CONS(ENTITY, e, lr); /* reverse! */
	},
	    l);

	gen_free_list(l);
	result = gen_nconc(result, lr);
    }
    else
    {
	result = 
	    gen_nconc(result,
		      hpfc_order_specific_variables(le, get_ith_cycle_dummy));
	
	result = 
	    gen_nconc(result,
		      hpfc_order_specific_variables(le, get_ith_block_dummy));
	
	result = 
	    gen_nconc(result,
		      hpfc_order_specific_variables(le, get_ith_shift_dummy));
	
	result = 
	    gen_nconc(result,
		      hpfc_order_specific_variables(le, get_ith_array_dummy));
    }

    result = 
	gen_nconc(result,
		  hpfc_order_specific_variables(le, get_ith_local_dummy));

    pips_assert("same length", gen_length(result)==gen_length(le));
    
    return(result);
}

void 
hpfc_algorithm_tiling(
    Psysteme syst,
    list processors, 
    list scanners,
    Psysteme *pcondition,
    Psysteme *pproc_echelon,
    Psysteme *ptile_echelon)
{
    Pbase
	outer = entity_list_to_base(processors),
	inner = entity_list_to_base(scanners);

    DEBUG_SYST(8, "initial system", syst);

    algorithm_tiling(syst, outer, inner, 
		     pcondition, pproc_echelon, ptile_echelon);

    DEBUG_SYST(3, "condition", *pcondition);
    DEBUG_BASE(3, "proc vars", outer);
    DEBUG_SYST(3, "processors", *pproc_echelon);
    DEBUG_BASE(3, "tile vars", inner);
    DEBUG_SYST(3, "tiles", *ptile_echelon);

    base_rm(outer);
    base_rm(inner);
}


/* void hpfc_simplify_condition(psc, stat, move)
 *
 * remove conditions that are not usefull from *psc, i.e. that are
 * redundent with pre/post conditions depending on when the movement is
 * done
 */
void 
hpfc_simplify_condition(
    Psysteme *psc,
    statement stat,
    tag move)
{
    Psysteme
	pstrue = statement_context(stat, move),
	cleared = extract_nredund_subsystem(*psc, pstrue);

    *psc = (sc_rm(*psc), cleared);
}

/* generates the Psystem for IOs inside the statement stat,
 * that use entity ent which should be a variable.
 */
static Psysteme 
generate_io_system(
    entity array,
    statement stat,
    tag move,
    tag act)
{
    Psysteme result = SC_UNDEFINED;

    pips_assert("variable", entity_variable_p(array));

    if (array_distributed_p(array))
    {
	result = generate_distributed_io_system(array, stat, move, act);
	result = clean_distributed_io_system(result, array, move);
    }
    else
    {
	result = generate_shared_io_system(array, stat, move, act);
	result = clean_shared_io_system(result, array, move);
    }

    sc_vect_sort(result, compare_Pvecteur);

    DEBUG_SYST(2, concatenate("array ", entity_name(array), NULL), result);

    return result;
}

static void 
generate_io_collect_or_update(
    entity array,
    statement stat,
    tag move, 
    tag act,
    statement *psh,
    statement *psn)
{
    Psysteme syst = generate_io_system(array, stat, move, act);
    set_information_for_code_optimizations(syst);

    pips_assert("variable and syst",
		entity_variable_p(array) && syst!=SC_UNDEFINED);

    if (array_distributed_p(array))
    {
	/* SCANNING: Variables must be classified as:
	 *   - parameters 
	 *   - processors
	 *   - scanner
	 *   - deducable
	 */
	Psysteme proc_echelon, tile_echelon, condition;
	list parameters = NIL, processors = NIL, scanners = NIL, rebuild = NIL;

	/* Now we have a set of equations and inequations, and we are going
	 * to organise a scanning of the data and the communications that 
	 * are needed
	 */
	put_variables_in_ordered_lists
	    (&syst, array, &parameters, &processors, &scanners, &rebuild); 

	hpfc_algorithm_tiling(syst, processors, scanners, 
			      &condition, &proc_echelon, &tile_echelon);
	hpfc_simplify_condition(&condition, stat, move);

	/*  the sorting is done again at the code generation,
	 *  but this phase will ensure more determinism in the debug messages
	 */
	/* sc_vect_sort(condition, compare_Pvecteur); */
	sc_sort(condition, sc_base(condition), compare_Pvecteur);
	sc_vect_sort(proc_echelon, compare_Pvecteur);
	sc_vect_sort(tile_echelon, compare_Pvecteur);

	if (!sc_empty_p(proc_echelon) && !sc_empty_p(tile_echelon))
	{
	    generate_io_statements_for_distributed_arrays
		(array, move, 
		 condition, proc_echelon, tile_echelon,
		 parameters, processors, scanners, rebuild,
		 psh, psn);
	}
	else
	{
	    hpfc_warning("empty io for %s\n", entity_name(array));
	    *psh = make_continue_statement(entity_undefined);
	    *psn = make_continue_statement(entity_undefined);
	}
    }
    else  /* array not distributed */
    {
	Psysteme row_echelon = SC_UNDEFINED, condition = SC_UNDEFINED;
	list tmp = NIL, parameters = NIL, scanners = NIL, rebuild = NIL;

	pips_assert("update", movement_update_p(move));

	put_variables_in_ordered_lists
	    (&syst, array, &parameters, &tmp, &scanners, &rebuild);

	pips_assert("empty list", ENDP(tmp));

	hpfc_algorithm_row_echelon(syst, scanners, &condition, &row_echelon);
	hpfc_simplify_condition(&condition, stat, move);

	/*  the sorting is done again at the code generation,
	 *  but this phase will ensure more determinism in the debug messages
	 */
	/* sc_vect_sort(condition, compare_Pvecteur); */
	sc_sort(condition, sc_base(condition), compare_Pvecteur);
	sc_vect_sort(row_echelon, compare_Pvecteur);

	if (!sc_empty_p(row_echelon))
	{
	    generate_io_statements_for_shared_arrays
		(array, move,
		 condition, row_echelon,
		 parameters, scanners, rebuild,
		 psh, psn);
	}
	else
	{
	    hpfc_warning("empty io for %s\n", entity_name(array));
	    *psh = make_continue_statement(entity_undefined);
	    *psn = make_continue_statement(entity_undefined);
	}
    }

    reset_information_for_code_optimizations();

    DEBUG_STAT(8, "Host", *psh);
    DEBUG_STAT(8, "Node", *psn);
}

/* add a local declaration for entity array on the host.
 * if entity array is dynamic, then declaration based on the primary.
 */
static void 
add_declaration_to_host_and_link(
    entity array)
{
    entity primary, host_array;
    storage s;
    
    if (dynamic_entity_p(array))
    {
	primary = load_primary_entity(array);
	if (bound_new_host_p(primary)) /* already exists, just add the link */
	{
	    store_new_host_variable(load_new_host(primary), array);
	    return;		
	}
    }
    else 
	primary = array;
    
    /* create the host entity
     */
    host_array = AddEntityToModule(primary, host_module);
    store_new_host_variable(host_array, primary);
    
    if (primary!=array) /* second link if needed */
	store_new_host_variable(host_array, array);

    /* local, not passed as an argument
     */
    s = entity_storage(host_array);
    if (storage_formal_p(s))
	formal_offset(storage_formal(s)) = INT_MAX;
}

/*   compile an io statement
 */
void io_efficient_compile(
    statement stat, /* statement to compile */
    statement *hp,  /* returned Host code */
    statement *np)  /* returned Node code */
{
  list
    /* of effect */ entities = load_rw_effects_list(stat),
    lh_collect = NIL, lh_io = NIL, lh_update = NIL,
    ln_collect = NIL, ln_io = NIL, ln_update = NIL;
  statement sh, sn;

  debug_on("HPFC_IO_DEBUG_LEVEL");
  pips_debug(1, "compiling!\n");
  pips_debug(2, "statement %" _intFMT " (%p), %zd arrays\n",
	     statement_number(stat), stat, gen_length(entities));

  // quicker for continue and so...
  if (empty_code_p(stat))
  {
    pips_debug(3, "empty statement\n");
    *hp = copy_statement(stat);
    *np = copy_statement(stat);
    debug_off();
    return;
  }

  // for each effect e on that statement
  FOREACH(effect, e, entities)
  {
    entity array = reference_variable(effect_any_reference(e));
    pips_debug(3, "variable %s\n", entity_name(array));

    if (io_effect_entity_p(array)) // skip LUNS
      continue;

    action act = effect_action(e);
    approximation apr = effect_approximation(e);

    pips_assert("avoid replicated array I/O", /* not implemented */
		!(array_distributed_p(array) && replicated_p(array)));

    if ((!array_distributed_p(array)) && action_read_p(act))
    {
      pips_debug(7, "skipping array %s movements - none needed\n",
		 entity_name(array));
      continue;
    }

    // add array declaration on host if necessary
    if (array_distributed_p(array) && !bound_new_host_p(array))
      add_declaration_to_host_and_link(array);

    // collect data if necessary
    if (array_distributed_p(array) &&
	current_entity_is_updated_before_p(stat,array) &&
	(action_read_p(act) ||
	 (action_write_p(act) &&
	  approximation_may_p(apr) &&
	  !get_bool_property("HPFC_IGNORE_MAY_IN_IO"))))
    {
      generate_io_collect_or_update(array, stat, is_movement_collect,
				    action_tag(act), &sh, &sn);
      lh_collect = CONS(STATEMENT, sh, lh_collect);
      ln_collect = CONS(STATEMENT, sn, ln_collect);
    }

    // update data if necessary
    // action = write and data may be used later (out regions)
    if (action_write_p(act) && current_entity_is_used_later_p(stat,array))
    {
      generate_io_collect_or_update(array, stat, is_movement_update,
				    action_tag(act), &sh, &sn);
      lh_update = CONS(STATEMENT, sh, lh_update);
      ln_update = CONS(STATEMENT, sn, ln_update);
    }
  }

  lh_io =  CONS(STATEMENT, copy_statement(stat), NIL);

  if (get_bool_property("HPFC_SYNCHRONIZE_IO"))
  {
    // could do it only for write statements
    entity synchro = hpfc_name_to_entity(SYNCHRO);

    lh_io = CONS(STATEMENT, hpfc_make_call_statement(synchro, NIL), lh_io);
    ln_io = CONS(STATEMENT, hpfc_make_call_statement(synchro, NIL), ln_io);
  }

  *hp = make_block_statement(gen_nconc(lh_collect,
			     gen_nconc(lh_io,
				       lh_update)));
  *np = make_block_statement(gen_nconc(ln_collect,
			     gen_nconc(ln_io,
				       ln_update)));

  DEBUG_STAT(9, "Host", *hp);
  DEBUG_STAT(9, "Node", *np);

  debug_off();
}

/* that is all
 */
