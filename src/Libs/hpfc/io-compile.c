/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: io-compile.c,v $ ($Date: 1994/04/15 10:14:04 $, ) version $Revision$,
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

#include "types.h"
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

/*----------------------------------------------------------------
 *
 * IO EFFICIENT COMPILATION
 */

/*
 * bool io_efficient_compilable_p(stat)
 * statement stat;
 *
 * checks for the compilability of the statement.
 *
 */
bool io_efficient_compilable_p(stat)
statement stat;
{
    bool
	compilable = TRUE;

    debug(4, "io_efficient_compilable_p", "statement 0x%x\n", stat);

    pips_assert("io_efficient_compilable_p",
		load_statement_only_io(stat)==TRUE);

    /*
     * ??? should be MUST if write and MAY if read?
     * ??? and should consider any array, whether distributed or not?
     * ??? to be improved soon...
     */
    
    MAPL(ce,
     {
	 effect
	     e = EFFECT(CAR(ce));
	 entity
	     array = reference_variable(effect_reference(e));
	 
	 debug(5, "io_efficient_compilable_p", 
	       "considering array %s\n", entity_local_name(array));
	 
	 if ((array_distributed_p(array)) &&
	     (approximation_may_p(effect_approximation(e))))
	 {
	     debug(5, "io_efficient_compilable_p",
		   "false on reference to %s in statement %d\n",
		   entity_local_name(array), statement_number(stat));

	     compilable = FALSE;
	 }
     },
	 load_statement_local_regions(stat));

    user_warning("io_efficient_compilable_p", 
		 "only partially implemented, returning %d\n", compilable);

    ifdebug(5)
	if (compilable)
	{
	    fprintf(stderr, 
	   "[io_efficient_compilable_p] io efficient compilable statement:\n");
	    print_statement(stat);
	}

    return(compilable);
}

void io_efficient_compile(stat, hp, np)
statement stat, *hp, *np;
{
    list
	entities = load_statement_local_regions(stat);

    debug_on("HPFC_IO_DEBUG_LEVEL");
    debug(1, "io_efficient_compile", "compiling!\n");

    debug(2, "io_efficient_compile", 
	  "statement 0x%x, %d arrays\n",
	  stat, gen_length(entities));

    /*
     * no array to deal with: the old function should work!
     */
    if ENDP(entities)
    {
	hpfcompileIO(stat, hp, np);
	return;
    }

    MAPL(ce,
     {
	 effect
	     e = EFFECT(CAR(ce));
	 entity
	     array = reference_variable(effect_reference(e));
	 tag
	     act = action_tag(effect_action(e));
	 tag
	     apr = approximation_tag(effect_approximation(e));	     
	 bool
	     dist = array_distributed_p(array);
	 bool
	     write = (act==is_action_write);
	 bool
	     may = (apr==is_approximation_may);
	 statement
	     sh = statement_undefined;
	 statement
	     sn = statement_undefined;
	 
	 debug(3, "io_efficient_compile", 
	       "array %s\n", entity_local_name(array));

	 if ((!dist) && (!write)) continue;

	 if (dist && (!write || (write && may)))
	     generate_io_collect_or_update(array, stat, 
					   is_movement_collect, 
					   act, &sh, &sn);

	 if (write)
	     generate_io_collect_or_update(array, stat, 
					   is_movement_update, 
					   act, &sh, &sn);
     },
	 entities);

    pips_error("io_efficient_compile", "not implemented yet\n");

    debug_off();
}

void generate_io_collect_or_update(array, stat, move, act, psh, psn)
entity array;
statement stat;
tag move, act;
statement *psh, *psn;
{
    pips_assert("generate_io_collect_or_update", entity_variable_p(array));
    

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
	    syst = generate_io_system(array, stat, move, act),
	    proc_echelon = SC_UNDEFINED,
	    tile_echelon = SC_UNDEFINED,
	    condition = SC_UNDEFINED;
	list
	    parameters = NIL,
	    processors = NIL,
	    scanners = NIL,
	    rebuild = NIL;

	pips_assert("generate_io_collect_or_update", syst!=SC_UNDEFINED);

	/* Now we have a set of equations and inequations, and we are going
	 * to organise a scanning of the data and the communications that 
	 * are needed
	 */

	put_variables_in_ordered_lists
	    (syst, array, &parameters, &processors, &scanners, &rebuild); 
	hpfc_algorithm_tiling(syst, processors, scanners, 
			      &condition, &proc_echelon, &tile_echelon);

	
    }
    else
    {
	pips_assert("generate_io_collect_or_update", movement_update_p(move));

	user_warning("generate_io_collect_or_update", 
		     "shared arrays management not implemented yet\n");
    }
}

/*
 * Psysteme generate_io_system(ent, stat)
 * entity ent;
 * statement stat;
 *
 * generates the Psystem for IOs inside the statement stat,
 * that use entity ent which should be a variable.
 *
 *
 *
 */
Psysteme generate_io_system(array, stat, move, act)
entity array;
statement stat;
tag move, act;
{
    Psysteme
	result = SC_UNDEFINED;

    pips_assert("generate_io_system", entity_variable_p(array));

    if (array_distributed_p(array))
    {

	result = generate_distributed_io_system(array, stat, move, act);
	result = clean_distributed_io_system(result, array, move);
    }
    else
    {
	user_warning("generate_io_system",
		     "shared arrays management not implemented yet\n");
    }

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
	contxt = 
	    predicate_system(transformer_relation
			     ((movement_collect_p(move)) ? 
			      load_statement_precondition(stat) :
			      load_statement_postcondition(stat)));
    
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
    /* result = sc_elim_redond(result); */
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

    debug(5, "clean_distributed_io_system", "array %s, action %s\n",
	  entity_local_name(array), 
	  (movement_collect_p(move))?"collect":"update");

    pips_assert("clean_distributed_io_system", array_distributed_p(array));
    
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
    
    /* others */
    gen_remove(&try_remove, (chunk*) TCST);
    MAPL(ce, {gen_remove(&try_remove, ENTITY(CAR(ce)));}, keep);
    MAPL(ce, {gen_remove(&try_remove, ENTITY(CAR(ce)));}, try_keep);
    MAPL(ce, {gen_remove(&try_remove, ENTITY(CAR(ce)));}, remove);
    
    /*
     * Remove variables that have to be removed
     */
    MAPL(ce, {sc_projection(syst, (Variable) ENTITY(CAR(ce)));}, remove);
    
    /*
     * Try to remove other unusefull variables
     */
    MAPL(ce, 
     {
	 Variable
	     var = (Variable) ENTITY(CAR(ce));
	 int
	     coeff = 0;
	 
	 /*
	  * Yi-Qing Stuff in ricedg is used
	  */
	 if ((void) eq_v_min_coeff(sc_egalites(syst), var, &coeff), coeff==1)
	 {
	     Pvecteur
		 v = vect_new(var, 1);
	     
	     debug(7, "clean_distributed_io_system", "removing variable %s\n", 
		   entity_local_name((entity) var));
	     
	     syst = sc_projection_optim_along_vecteur(syst, v);
	     vect_rm(v);
	 }
     }, 
	 try_remove);

    /*
     * the noisy system is cleaned
     * some variables are not used, they are removed here.
     */
    /* syst = sc_elim_redond(syst); */
    sc_nredund(&syst);
    base_rm(sc_base(syst));
    sc_base(syst) = NULL;
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
void put_variables_in_ordered_lists(syst, array, 
				    plparam, plproc, plscan,
				    plrebuild)
Psysteme syst;
entity array;
list *plparam, *plproc, *plscan, *plrebuild;
{
    int
	processor_dim = 
	    NumberOfDimension
		(template_to_processors(array_to_template(array))),
	dim = -1;
    list
	all = base_to_list(sc_base(syst)),
	lparam = NIL,
	lproc = NIL,
	lscan = NIL,
	lrebuild = NIL;

    gen_remove(&all, (chunk*) TCST); /* just in case */

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
	simplify_deducable_variables(syst,
				     gen_nreverse(hpfc_order_variables(all)),
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

    /* syst = sc_elim_redond(syst); */
    sc_nredund(&syst);
    base_rm(sc_base(syst));
    sc_base(syst) = NULL;
    sc_creer_base(syst);

    ifdebug(4)
    {
	fprintf(stderr, 
		"[put_variables_in_ordered_lists] system for %s:\n",
		entity_local_name(array));
	sc_fprint(stderr, syst, entity_local_name);
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

	     syst = sc_projection_by_eq(syst, eq, (Variable) dummy);
	 }
	 else
	 {
	     *pleftvars = CONS(ENTITY, dummy, *pleftvars);
	 }
     },
	 vars);

    return(result);
}


/*
 * list hpfc_order_variables(list)
 *
 * the input list of entities is ordered so that:
 * PSI_i's, GAMMA_i's, DELTA_i's, IOTA_i's, ALPHA_i's, LALPHA_i's...
 */
list hpfc_order_variables(le)
list le;
{
    list
	result = NIL;

    result = 
	gen_nconc(result,
		  hpfc_order_specific_variables
		  (le, get_ith_processor_dummy));

    result = 
	gen_nconc(result,
		  hpfc_order_specific_variables
		  (le, get_ith_cycle_dummy));

    result = 
	gen_nconc(result,
		  hpfc_order_specific_variables
		  (le, get_ith_block_dummy));

    result = 
	gen_nconc(result,
		  hpfc_order_specific_variables
		  (le, get_ith_shift_dummy));

    result = 
	gen_nconc(result,
		  hpfc_order_specific_variables
		  (le, get_ith_array_dummy));

    result = 
	gen_nconc(result,
		  hpfc_order_specific_variables
		  (le, get_ith_local_dummy));

    pips_assert("hpfc_order_variables",
		gen_length(result)==gen_length(le));
    
    return(result);
}

list hpfc_order_specific_variables(le, creation)
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

void hpfc_algorithm_tiling(syst, processors, scanners, 
			   pcondition, pproc_echelon, ptile_echelon)
Psysteme syst;
list processors, scanners;
Psysteme *pcondition, *pproc_echelon, *ptile_echelon;
{
    Pbase
	outer = entity_list_to_base(processors),
	inner = entity_list_to_base(scanners);

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


/*
 * that's all
 */
