/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: io-compile.c,v $ ($Date: 1995/03/22 10:57:02 $, ) version $Revision$,
 * got on %D%, %T%
 * $Id$
 */

/*
 * Standard includes
 */
 
#include <stdio.h>
#include <string.h> 
extern fprintf();

/*
 * Psystems stuff
 */

#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/*
 * Newgen stuff
 */

#include "genC.h"

#include "ri.h" 
#include "hpf.h" 
#include "hpf_private.h"

/*
 * PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "control.h"
#include "transformer.h"
#include "regions.h"
#include "semantics.h"
#include "effects.h"
#include "properties.h"

/* in paf-util.h: */
list base_to_list(Pbase base);
void fprint_entity_list(FILE *fp, list l);

 /* Yi-Qing stuff */
#include <values.h>

#include "arithmetique.h"
#include "text.h"
#include "graph.h"
#include "dg.h"
#include "database.h"
#include "rice.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "ricedg.h"      

/* 
 * my own local includes
 */

#include "hpfc.h"
#include "defines-local.h"
/* #include "compiler_parameters.h" */

/*----------------------------------------------------------------
 *
 * IO EFFICIENT COMPILATION
 */
Psysteme statement_context(stat, move)
statement stat;
tag move;
{
    return(predicate_system(transformer_relation
			    ((movement_collect_p(move)) ? 
			     load_statement_precondition(stat) :
			     load_statement_postcondition(stat)))); 
}

/*   compile an io statement
 */
void io_efficient_compile(stat, hp, np)
statement stat, *hp, *np;
{
    list
	entities = load_statement_local_regions(stat),
	lh_collect = NIL,
	lh_io = NIL,
	lh_update = NIL,
	ln_collect = NIL,
	ln_io = NIL,
	ln_update = NIL;
    statement
	sh = statement_undefined,
	sn = statement_undefined;

    debug_on("HPFC_IO_DEBUG_LEVEL");
    debug(1, "io_efficient_compile", "compiling!\n");
    debug(2, "io_efficient_compile", "statement 0x%x, %d arrays\n",
	  stat, gen_length(entities));

    MAPL(ce,
     {
	 effect
	     e = EFFECT(CAR(ce));
	 entity
	     array = reference_variable(effect_reference(e));
	 action
	     act = effect_action(e);
	 approximation
	     apr = effect_approximation(e);
	 
	 debug(3, "io_efficient_compile", 
	       "array %s\n", entity_name(array));

	 if ((!array_distributed_p(array)) && action_read_p(act)) 
	 {
	     debug(7, 
		   "io_efficient_compile", 
		   "skipping array %s movements - none needed\n", 
		   entity_name(array));
	     continue;
	 }

	 /*
	  * add array declaration on host if necessary
	  */
	 if (array_distributed_p(array) && 
	     (load_entity_host_new(array) == (entity) HASH_UNDEFINED_VALUE))
	     store_new_host_variable(AddEntityToModule(array, host_module), 
				     array);

	 /*
	  * collect data if necessary
	  */
	 if (array_distributed_p(array) && 
	     (action_read_p(act) || 
	      (action_write_p(act) && 
	       approximation_may_p(apr) && 
	       !get_bool_property("HPFC_IGNORE_MAY_IN_IO"))))
	 {
	     generate_io_collect_or_update(array, stat, 
					   is_movement_collect, 
					   action_tag(act), &sh, &sn);
	     lh_collect = CONS(STATEMENT, sh, lh_collect);
	     ln_collect = CONS(STATEMENT, sn, ln_collect);
	 }

	 /*
	  * update data if necessary
	  */
	 if (action_write_p(act))
	 {
	     generate_io_collect_or_update(array, stat, 
					   is_movement_update, 
					   action_tag(act), &sh, &sn);
	     lh_update = CONS(STATEMENT, sh, lh_update);
	     ln_update = CONS(STATEMENT, sn, ln_update);
	 }
     },
	 entities);

    lh_io =  CONS(STATEMENT, copy_statement(stat), NIL);

    if (get_bool_property("HPFC_SYNCHRONIZE_IO"))
    {
	/*
	 * could do it only for write statements
	 */
	entity
	    synchro = hpfc_name_to_entity(SYNCHRO);
	
	lh_io = CONS(STATEMENT, hpfc_make_call_statement(synchro, NIL), lh_io);
	ln_io = CONS(STATEMENT, hpfc_make_call_statement(synchro, NIL), ln_io);
    }

    *hp = make_block_statement(gen_nconc(lh_collect,
			       gen_nconc(lh_io,
			                 lh_update)));
    *np = make_block_statement(gen_nconc(ln_collect,
			       gen_nconc(ln_io,
					 ln_update)));

    ifdebug(9)
    {
	fprintf(stderr, "[io_efficient_compile] output:\n");
	fprintf(stderr, "Host:\n");
	print_statement(*hp);
	fprintf(stderr, "Node:\n");
	print_statement(*np);
    }

    debug_off();
}

void hpfc_algorithm_row_echelon(syst, scanners, pcond, penum)
Psysteme syst;
list scanners;
Psysteme *pcond, *penum;
{
    Pbase base = entity_list_to_base(scanners);

    algorithm_row_echelon(syst, base, pcond, penum);

    base_rm(base);
}

void generate_io_collect_or_update(array, stat, move, act, psh, psn)
entity array;
statement stat;
tag move, act;
statement *psh, *psn;
{
    Psysteme
	syst = generate_io_system(array, stat, move, act);

    assert(entity_variable_p(array) && syst!=SC_UNDEFINED);


    /* ifdebug(9)
     *	 fprintf(stderr, "[generate_io_collect_or_update] syst\n"),
     *	 sc_fprint(stderr, syst, entity_local_name);
     */

    if (array_distributed_p(array))
    {
	/*
	 * SCANNING: Variables must be classified as:
	 *   - parameters 
	 *   - processors
	 *   - scanner
	 *   - deducable
	 */
	Psysteme
	    proc_echelon = SC_UNDEFINED,
	    tile_echelon = SC_UNDEFINED,
	    condition = SC_UNDEFINED;
	list
	    parameters = NIL,
	    processors = NIL,
	    scanners = NIL,
	    rebuild = NIL;

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
	sc_vect_sort(condition, compare_Pvecteur);
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
	    hpfc_warning("generate_io_collect_or_update",
			 "empty operation for array %s\n", 
			 entity_name(array));
	    *psh = make_continue_statement(entity_undefined);
	    *psn = make_continue_statement(entity_undefined);
	}
    }
    else
    {
	Psysteme
	    row_echelon = SC_UNDEFINED,
	    condition = SC_UNDEFINED;
	list
	    tmp = NIL,
	    parameters = NIL,
	    scanners = NIL,
	    rebuild = NIL;

	assert(movement_update_p(move));

	put_variables_in_ordered_lists
	    (&syst, array, &parameters, &tmp, &scanners, &rebuild);

	assert(ENDP(tmp));

	hpfc_algorithm_row_echelon(syst, scanners, &condition, &row_echelon);
	hpfc_simplify_condition(&condition, stat, move);

	/*  the sorting is done again at the code generation,
	 *  but this phase will ensure more determinism in the debug messages
	 */
	sc_vect_sort(condition, compare_Pvecteur);
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
	    hpfc_warning("generate_io_collect_or_update",
			 "empty operation for array %s\n", 
			 entity_name(array));
	    *psh = make_continue_statement(entity_undefined);
	    *psn = make_continue_statement(entity_undefined);
	}
    }

    ifdebug(8)
    {
	fprintf(stderr, 
		"[generate_io_collect_or_update] output:\n");
	fprintf(stderr, "Host:\n");
	print_statement(*psh);
	fprintf(stderr, "Node:\n");
	print_statement(*psn);
    }
}

/*
 * generates the Psystem for IOs inside the statement stat,
 * that use entity ent which should be a variable.
 */
Psysteme generate_io_system(array, stat, move, act)
entity array;
statement stat;
tag move, act;
{
    Psysteme
	result = SC_UNDEFINED;

    assert(entity_variable_p(array));

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

    /*
     * Final DEBUG message
     */
    ifdebug(2)
    {
	fprintf(stderr, 
		"[generate_io_system] returning for array %s:\n",
		entity_local_name(array));
	fprintf(stderr, "Result:\n");
	sc_fprint(stderr, result, entity_local_name);
    }

    return(result);
}

list make_list_of_dummy_variables(creation, number)
entity (*creation)();
int number;
{
    list
	result = NIL;

    for(;number>0;number--)
	result = CONS(ENTITY, creation(number), result);

    return(result);
}

Psysteme generate_shared_io_system(array, stat, move, act)
entity array;
statement stat;
tag move, act;
{
    Psysteme
	result = SC_UNDEFINED, /* ??? bug post with region */
	region = effect_system(entity_to_region(stat, array, act)), 
	a_decl = entity_to_declaration_constraints(array),
	stamme = hpfc_unstutter_dummies(array),
	contxt = statement_context(stat, move); 
    
    assert(!array_distributed_p(array));

    result = sc_append(sc_rn(NULL), region);
    result = sc_append(result, a_decl);
    result = sc_append(result, stamme);
    result = sc_append(result, contxt);
    
    ifdebug(8)
    {
	fprintf(stderr, 
		"[generate_shared_io_system] whole system for array %s:\n",
		entity_local_name(array));
	fprintf(stderr, "Result:\n");
	sc_fprint(stderr, result, entity_local_name);
    }
    
    /*
     * the noisy system is cleaned
     * some variables are not used, they are removed here.
     */
    sc_nredund(&result);
    base_rm(sc_base(result));
    sc_base(result) = NULL;
    sc_creer_base(result);
    
    /*
     * DEBUG stuff: the systems are printed
     */
    ifdebug(7)
    {
	fprintf(stderr, 
		"[generate_shared_io_system] systems for array %s:\n",
		entity_local_name(array));
	fprintf(stderr, "Region:\n");
	sc_fprint(stderr, region, entity_local_name);
	fprintf(stderr, "Array declaration:\n");
	sc_fprint(stderr, a_decl, entity_local_name);
	fprintf(stderr, "Unstammer:\n");
	sc_fprint(stderr, stamme, entity_local_name);
	fprintf(stderr, "Context:\n");
	sc_fprint(stderr, contxt, entity_local_name);
    }
    
    ifdebug(6)
    {
	fprintf(stderr, 
	  "[generate_shared_io_system] resulting system for array %s:\n",
		entity_local_name(array));
	fprintf(stderr, "Result:\n");
	sc_fprint(stderr, result, entity_local_name);
    }
 
    return(result);

    

}

Psysteme generate_distributed_io_system(array, stat, move, act)
entity array;
statement stat;
tag move, act;
{
    entity
	template = array_to_template(array),
	processors = template_to_processors(template);
    Psysteme
	result = SC_UNDEFINED,
	/*
	 * ??? bug: the preconditions may be in the regions.  To update, I
	 * should have the postconditions instead, that is the statement
	 * transformer should be applied to the system.
	 */
	region = effect_system(entity_to_region(stat, array, act)),
	a_decl = entity_to_declaration_constraints(array),
	n_decl = entity_to_new_declaration(array),
	t_decl = entity_to_declaration_constraints(template),
	p_decl = entity_to_declaration_constraints(processors),
	salign = entity_to_hpf_constraints(array),
	sdistr = entity_to_hpf_constraints(template),
	sother = hpfc_compute_unicity_constraints(array), /* ??? */
	stamme = hpfc_unstutter_dummies(array),
	contxt = statement_context(stat, move);
    
    /* ??? massive memory leak 
     */
    result = sc_append(sc_rn(NULL), region);
    result = sc_append(result, a_decl);
    result = sc_append(result, n_decl);
    result = sc_append(result, t_decl);
    result = sc_append(result, p_decl);
    result = sc_append(result, salign);
    result = sc_append(result, sdistr);
    result = sc_append(result, sother);
    result = sc_append(result, stamme);
    result = sc_append(result, contxt);
    
    ifdebug(8)
    {
	fprintf(stderr, 
		"[generate_distributed_io_system] whole system for array %s:\n",
		entity_local_name(array));
	fprintf(stderr, "Result:\n");
	sc_fprint(stderr, result, entity_local_name);
    }
    
    /*
     * the noisy system is cleaned
     * some variables are not used, they are removed here.
     */
    build_sc_nredund_2pass(&result);
    base_rm(sc_base(result));
    sc_base(result) = NULL;
    sc_creer_base(result);
    
    /*
     * DEBUG stuff: the systems are printed
     */
    ifdebug(7)
    {
	fprintf(stderr, 
		"[generate_distributed_io_system] systems for array %s:\n",
		entity_local_name(array));
	fprintf(stderr, "Region:\n");
	sc_fprint(stderr, region, entity_local_name);
	fprintf(stderr, "Array declaration:\n");
	sc_fprint(stderr, a_decl, entity_local_name);
	fprintf(stderr, "Array new declaration:\n");
	sc_fprint(stderr, n_decl, entity_local_name);
	fprintf(stderr, "Template declaration:\n");
	sc_fprint(stderr, t_decl, entity_local_name);
	fprintf(stderr, "Processors declaration:\n");
	sc_fprint(stderr, p_decl, entity_local_name);
	fprintf(stderr, "Hpf align:\n");
	sc_fprint(stderr, salign, entity_local_name);
	fprintf(stderr, "Hpf distribute:\n");
	sc_fprint(stderr, sdistr, entity_local_name);
	fprintf(stderr, "Hpf unicity:\n");
	sc_fprint(stderr, sother, entity_local_name);
	fprintf(stderr, "Unstammer:\n");
	sc_fprint(stderr, stamme, entity_local_name);
	fprintf(stderr, "Context:\n");
	sc_fprint(stderr, contxt, entity_local_name);
    }
    
    ifdebug(6)
    {
	fprintf(stderr, 
	  "[generate_distributed_io_system] resulting system for array %s:\n",
		entity_local_name(array));
	fprintf(stderr, "Result:\n");
	sc_fprint(stderr, result, entity_local_name);
    }
 
    return(result);
}

void remove_variables_if_possible(psyst, lvars)
Psysteme *psyst;
list lvars;
{
    Psysteme 
	syst = *psyst;

    MAPL(ce, 
     {
	 Variable
	     var = (Variable) ENTITY(CAR(ce));
	 int
	     coeff = -1;

	 (void) contrainte_var_min_coeff(sc_egalites(syst), var, &coeff, FALSE);

	 if (coeff==1)
	 {
	     Pvecteur
		 v = vect_new(var, 1);
	     bool 
		 exact = TRUE;
	     
	     debug(7, "remove_variables_if_possible", 
		   "removing variable %s\n", 
		   entity_local_name((entity) var));
	     
	     sc_projection_along_variables_with_test_ofl_ctrl(&syst, v, 
							      &exact, 
							      NO_OFL_CTRL);
	     assert(exact);
	     vect_rm(v);
	 }
     }, 
	 lvars);

    *psyst = syst;
}

Psysteme clean_shared_io_system(syst, array, move)
Psysteme syst;
entity array;
tag move;
{
    int
	array_dim = NumberOfDimension(array);
    list
	keep = NIL,
	try_keep = NIL,
	remove = NIL,
	try_remove = base_to_list(sc_base(syst));

    debug(5, "clean_shared_io_system", "array %s, movement %s\n",
	  entity_local_name(array), 
	  (movement_collect_p(move))?"collect":"update");

    /* ALPHA_i's */
    keep =
	gen_nconc(make_list_of_dummy_variables
		  (get_ith_array_dummy, array_dim), 
		  keep);
    
    /* PHI_i's */
    remove = 
	gen_nconc(make_list_of_dummy_variables
		  (get_ith_region_dummy, array_dim), 
		  remove);

    /* Keep parameters ! */
    MAPL(ce,
     {
	 entity
	     e = ENTITY(CAR(ce));
	 string
	     s = entity_module_name(e);

	 if (strcmp(s, HPFC_PACKAGE) && strcmp(s, REGIONS_MODULE_NAME))
	     keep = CONS(ENTITY, e, keep);
     },
	 try_remove);

    /* others */
    gen_remove(&try_remove, (entity) TCST);
    MAPL(ce, {gen_remove(&try_remove, ENTITY(CAR(ce)));}, keep);
    MAPL(ce, {gen_remove(&try_remove, ENTITY(CAR(ce)));}, try_keep);
    MAPL(ce, {gen_remove(&try_remove, ENTITY(CAR(ce)));}, remove);
    
    /*
     * Remove variables that have to be removed
     */
    MAPL(ce, 
     {
	 sc_projection_along_variable_ofl_ctrl(&syst,
					       (Variable) ENTITY(CAR(ce)),
					       NO_OFL_CTRL);
     },
	 remove);
    
    /*
     * Try to remove other unusefull variables
     */
    remove_variables_if_possible(&syst, try_remove);

    /*
     * the noisy system is cleaned
     * some variables are not used, they are removed here.
     */
    build_sc_nredund_2pass(&syst);
    base_rm(sc_base(syst));
    sc_base(syst) = BASE_NULLE;
    sc_creer_base(syst);

    /*
     * DEBUG
     */
    ifdebug(6)
    {
	fprintf(stderr, 
	   "[clean_shared_io_system] resulting system for array %s:\n",
		entity_local_name(array));
	fprintf(stderr, "Result:\n");
	sc_fprint(stderr, syst, entity_local_name);
    }
    
    return(syst);
}

Psysteme clean_distributed_io_system(syst, array, move)
Psysteme syst;
entity array;
tag move;
{
    /*
     * ??? what about the variables?
     * some are usefull, some are constants, and others should
     * be discarded. This selection and projection may be done here.
     */
    /*
     * to be removed:
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
     * 
     */
    entity
	template = array_to_template(array),
	processor = template_to_processors(template);
    int
	array_dim = NumberOfDimension(array),
	template_dim = NumberOfDimension(template),
	processor_dim = NumberOfDimension(processor);
    list
	keep = NIL,
	try_keep = NIL,
	remove = NIL,
	try_remove = base_to_list(sc_base(syst));

    debug(5, "clean_distributed_io_system", "array %s, movement %s\n",
	  entity_local_name(array), 
	  (movement_collect_p(move))?"collect":"update");

    assert(array_distributed_p(array));
    
    /* THETA_i's */
    remove = 
	gen_nconc(make_list_of_dummy_variables
		  (get_ith_template_dummy, template_dim), 
		  remove);
    
    /* PHI_i's */
    remove = 
	gen_nconc(make_list_of_dummy_variables
		  (get_ith_region_dummy, array_dim), 
		  remove);
    
    /* PSI_i's */
    keep = 
	gen_nconc(make_list_of_dummy_variables
		  (get_ith_processor_dummy, processor_dim), 
		  keep);
    
    /* ALPHA_i's */
    keep =
	gen_nconc(make_list_of_dummy_variables
		  (get_ith_array_dummy, array_dim), 
		  keep);

    /* LALPHA_i's */
    try_keep =
	gen_nconc(make_list_of_dummy_variables
		  (get_ith_local_dummy, array_dim), 
		  try_keep);
    

    /* Keep parameters ! */
    MAPL(ce,
     {
	 entity
	     e = ENTITY(CAR(ce));
	 string
	     s = entity_module_name(e);

	 if (strcmp(s, HPFC_PACKAGE) && strcmp(s, REGIONS_MODULE_NAME))
	     keep = CONS(ENTITY, e, keep);
     },
	 try_remove);

    /* others */
    gen_remove(&try_remove, (entity) TCST);
    MAPL(ce, {gen_remove(&try_remove, ENTITY(CAR(ce)));}, keep);
    MAPL(ce, {gen_remove(&try_remove, ENTITY(CAR(ce)));}, try_keep);
    MAPL(ce, {gen_remove(&try_remove, ENTITY(CAR(ce)));}, remove);

    ifdebug(7)
    {
	fprintf(stderr, 
		"[clean_distributed_io_system] list of variables:\nkeep: ");
	fprint_entity_list(stderr, keep);
	fprintf(stderr, "\ntry_keep: ");
	fprint_entity_list(stderr, try_keep);
	fprintf(stderr, "\ntry_remove: ");
	fprint_entity_list(stderr, try_remove);
	fprintf(stderr, "\nremove: ");
	fprint_entity_list(stderr, remove);
	fprintf(stderr, "\n");
    }

    /*
     * Remove variables that have to be removed
     */
    MAPL(ce, 
     {
	 sc_projection_along_variable_ofl_ctrl(&syst,
					       (Variable) ENTITY(CAR(ce)),
					       NO_OFL_CTRL);
     },
	 remove);
    
    /*
     * Try to remove other unusefull variables
     */
    remove_variables_if_possible(&syst, try_remove);

    /*
     * the noisy system is cleaned
     * some variables are not used, they are removed here.
     */
    build_sc_nredund_2pass(&syst);
    base_rm(sc_base(syst));
    sc_base(syst) = BASE_NULLE;
    sc_creer_base(syst);

    /*
     * DEBUG
     */
    ifdebug(6)
    {
	fprintf(stderr, 
	   "[clean_distributed_io_system] resulting system for array %s:\n",
		entity_local_name(array));
	fprintf(stderr, "Result:\n");
	sc_fprint(stderr, syst, entity_local_name);
    }
    
    return(syst);
}

/* void put_variables_in_ordered_lists
 * Psysteme syst;
 * entity array;
 * entities* plparam, plproc, plscan;
 * expression* plrebuild;
 *
 * Variables of Psysteme syst are ordered and put in different
 * lists. Especially, deducable variables are listed, the equalities
 * that allow to rebuild them are also listed, and they are removed
 * from the original system by *exact* integer projection.
 *
 * Other variables are (should be) the parameters, the processors,
 * and the variables to be used to scan polyhedron.
 */
void put_variables_in_ordered_lists(psyst, array, 
				    plparam, plproc, plscan,
				    plrebuild)
Psysteme *psyst;
entity array;
list *plparam, *plproc, *plscan, *plrebuild;
{
    int
	processor_dim = (array_distributed_p(array) ?
	    NumberOfDimension(array_to_processors(array)) : 0 ),
	dim = -1;
    list
	all = base_to_list(sc_base(*psyst)),
	lparam = NIL,
	lproc = NIL,
	lscan = NIL,
	lrebuild = NIL;

    gen_remove(&all, (entity) TCST); /* just in case */

    debug(5, "put_variables_in_ordered_lists",
	  "considering %d variables\n", gen_length(all));

    /*
     * parameters: those variables that are not dummies...
     */
    MAPL(ce,
     {
	 entity
	     v = ENTITY(CAR(ce));

	 if (!entity_hpfc_dummy_p(v)) 
	     lparam = CONS(ENTITY, v, lparam);
     },
	 all);

    MAPL(ce, {gen_remove(&all, ENTITY(CAR(ce)));}, lparam);
    
    /*
     * processors
     */
    for(dim=processor_dim; dim>=1; dim--)
    {
	entity
	    dummy = get_ith_processor_dummy(dim);

	lproc = CONS(ENTITY, dummy, lproc);
	gen_remove(&all, dummy);
    }

    /*
     * scanners and deducables
     */
    lrebuild = 
	simplify_deducable_variables(*psyst,
				     gen_nreverse(hpfc_order_variables(all,
								       TRUE)),
				     &lscan);

    /*
     * return results
     */
    *plparam 	= lparam, 
    *plproc 	= lproc, 
    *plscan 	= lscan, /* lscan is implicitely ordered */
    *plrebuild 	= lrebuild;

    gen_free_list(all);

    ifdebug(4)
    {
	Pcontrainte
	    pc = contrainte_make(VECTEUR_NUL);

	fprintf(stderr, "[put_variables_in_ordered_lists] returning:\n");
	fprintf(stderr, " - params:\n   ");
	fprint_entity_list(stderr, lparam);
	fprintf(stderr, "\n - procs:\n   ");
	fprint_entity_list(stderr, lproc);
	fprintf(stderr, "\n - scanners:\n   ");
	fprint_entity_list(stderr, lscan);
	fprintf(stderr, "\n - deducables:\n   ");
	MAPL(ce,
	 {
	     expression
		 ex = EXPRESSION(CAR(ce));

	     pc->vecteur = 
		 normalized_linear(expression_normalized(ex));
	     
	     fprintf(stderr, "%s rebuilt with ", 
		     entity_local_name
		     (reference_variable(expression_reference(ex))));
	     egalite_fprint(stderr, pc, entity_local_name);
	 },
	     lrebuild);
    }

    /* syst = sc_elim_redund(syst); */

    build_sc_nredund_2pass(psyst);
    sc_base(*psyst) = (base_rm(sc_base(*psyst)), BASE_NULLE);
    sc_creer_base(*psyst);

    ifdebug(4)
    {
	fprintf(stderr, 
		"[put_variables_in_ordered_lists] system for %s:\n",
		entity_local_name(array));
	sc_fprint(stderr, *psyst, entity_local_name);
    }
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
list simplify_deducable_variables(syst, vars, pleftvars)
Psysteme syst;
list vars, *pleftvars;
{
    list
	result = NIL;

    MAPL(ce,
     {
	 entity
	     dummy = ENTITY(CAR(ce));
	 int 
	     coeff = 0;
	 Pcontrainte
	     eq = CONTRAINTE_UNDEFINED;

	 if (eq = eq_v_min_coeff(sc_egalites(syst), (Variable) dummy, &coeff),
	     (coeff == 1))
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

    return(result);
}

/* output 7 entities created by creation if in list le
 */
static list hpfc_order_specific_variables(le, creation)
list le;
entity (*creation)();
{
    list
	result = NIL;
    int i;

    for(i=7; i>=1; i--)
    {
	entity
	    dummy = creation(i);

	if (gen_find_eq(dummy, le)==dummy)
	    result = CONS(ENTITY, dummy, result);
    }

    return(result);
}

/*
 * list hpfc_order_variables(list)
 *
 * the input list of entities is ordered so that:
 * PSI_i's, GAMMA_i's, DELTA_i's, IOTA_i's, ALPHA_i's, LALPHA_i's...
 */
list hpfc_order_variables(le, number_first)
list le;
bool number_first;
{
    list
	result = NIL;

    result = 
	gen_nconc(result,
		  hpfc_order_specific_variables
		  (le, get_ith_processor_dummy));
    
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

	MAPL(ce,
	 {
	     entity e = ENTITY(CAR(ce));

	     if (gen_find_eq(e, le)==e) 
		 lr = CONS(ENTITY, e, lr); /* reverse! */
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

    assert(gen_length(result)==gen_length(le));
    
    return(result);
}

void hpfc_algorithm_tiling(syst, processors, scanners, 
			   pcondition, pproc_echelon, ptile_echelon)
Psysteme syst;
list processors, scanners;
Psysteme *pcondition, *pproc_echelon, *ptile_echelon;
{
    Pbase
	outer = entity_list_to_base(processors),
	inner = entity_list_to_base(scanners);

    ifdebug(8)
	fprintf(stderr, "[hpfc_algorithm_tiling] initial system:\n"),
	sc_fprint(stderr, syst, entity_local_name),
	fprintf(stderr, "\n");
	

    algorithm_tiling(syst, outer, inner, 
		     pcondition, pproc_echelon, ptile_echelon);

    ifdebug(3)
    {
	fprintf(stderr, "[hpfc_algorithm_tiling] results:\n - condition:\n");
	sc_fprint(stderr, *pcondition, entity_local_name);
	fprintf(stderr, "- processors: ");
	base_fprint(stderr, outer, entity_local_name);
	sc_fprint(stderr, *pproc_echelon, entity_local_name);
	fprintf(stderr, " - tiles: ");
	base_fprint(stderr, inner, entity_local_name);
	sc_fprint(stderr, *ptile_echelon, entity_local_name);
	fprintf(stderr, "\n");
    }

    base_rm(outer);
    base_rm(inner);
}

Pbase entity_list_to_base(l)
list l;
{
    list
	l2 = gen_nreverse(gen_copy_seq(l));
    Pbase
	result = BASE_NULLE;
	
    MAPL(ce,
     {
	 Pbase
	     new = (Pbase) vect_new((Variable) ENTITY(CAR(ce)), (Value) 1);

	 new->succ = result;
	 result = new;
     },
	 l2);

    gen_free_list(l2);
    return(result);
}

/* void hpfc_simplify_condition(psc, stat, move)
 *
 * remove conditions that are not usefull from *psc, i.e. that are
 * redundent with pre/post conditions depending on when the movement is
 * done
 */
void hpfc_simplify_condition(psc, stat, move)
Psysteme *psc;
statement stat;
tag move;
{
    Psysteme
	true = statement_context(stat, move),
	cleared = extract_nredund_subsystem(*psc, true);

    *psc = (sc_rm(*psc), cleared);
}

/*
 * that's all
 */
