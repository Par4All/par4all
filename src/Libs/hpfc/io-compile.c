/*
 * HPFC module by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: io-compile.c,v $ ($Date: 1994/03/25 17:45:58 $, ) version $Revision$,
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
     * ??? should be MUST if write and MAY if read
     * ??? and should consider any array, whether distributed or not?
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
	    DebugPrintCode(5, get_current_module_entity(), stat);
	}

    return(compilable);
}

tag io_action_in_statement(stat)
statement stat;
{
    list
	le = load_statement_local_regions(stat);

    MAPL(ce,
     {
	 effect 
	     e = EFFECT(CAR(ce));

	 if (action_write_p(e)) return(is_action_write);
     },
	 le);

    return(is_action_read);
}

list io_entities_in_statement(stat, act)
statement stat;
tag act;
{
    list
	result = NIL,
	le = load_statement_local_regions(stat);

    MAPL(ce,
     {
	 effect
	     e = EFFECT(CAR(ce));
	 entity
	     array = reference_variable(effect_reference(e));

	 debug(8, "io_entities_in_statement",
	       "considering array %s\n", entity_local_name(array));

	 if ((action_tag(effect_action(e))==act) && 
	     (!((act==is_action_read) && (!array_distributed_p(array)))))
	     result = CONS(ENTITY, array, result);
     },
	 le);

    return(result);
}

void io_efficient_compile(stat, hp, np)
statement stat, *hp, *np;
{
    tag
	act = tag_undefined;
    list
	entities = NIL;

    debug_on("HPFC_IO_DEBUG_LEVEL");
    debug(1, "io_efficient_compile", "compiling!\n");

    act = io_action_in_statement(stat);
    entities = io_entities_in_statement(stat, act);

    debug(2, "io_efficient_compile", 
	  "statement 0x%x, action %s, %d arrays\n",
	  stat, (act==is_action_read)?"read":"write", gen_length(entities));

    /*
     * no array to deal with: the old function should work
     */
    if ENDP(entities)
    {
	hpfcompileIO(stat, hp, np);
	return;
    }

    MAPL(ce,
     {
	 entity
	     array = ENTITY(CAR(ce));
	 statement
	     sh = statement_undefined;
	 statement
	     sn = statement_undefined;
	 
	 debug(3, "io_efficient_compile", 
	       "array %s\n", entity_local_name(array));

	 generate_io_scatter_or_gather(array, stat, act, &sh, &sn);

     },
	 entities);

    pips_error("io_efficient_compile", "not implemented yet\n");

    debug_off();
}

void generate_io_scatter_or_gather(array, stat, act, psh, psn)
entity array;
statement stat;
tag act;
statement *psh, *psn;
{
    Psysteme
	syst = generate_io_system(array, stat, act);

    pips_assert("generate_io_scatter_or_gather", syst!=SC_UNDEFINED);
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
Psysteme generate_io_system(array, stat, act)
entity array;
statement stat;
tag act;
{
    Psysteme
	result = SC_UNDEFINED;

    pips_assert("generate_io_system", entity_variable_p(array));

    if (array_distributed_p(array))
    {
	entity
	    template = array_to_template(array),
	    processors = template_to_processors(template);
	Psysteme
	    region = effect_system(entity_to_region(stat, array, act)),
	    a_decl = entity_to_declaration_constraints(array),
	    n_decl = entity_to_new_declaration(array),
	    t_decl = entity_to_declaration_constraints(template),
	    p_decl = entity_to_declaration_constraints(processors),
	    salign = entity_to_hpf_constraints(array),
	    sdistr = entity_to_hpf_constraints(template),
	    sother = hpfc_compute_unicity_constraints(array),
	    contxt = 
		predicate_system(transformer_relation
				 ((act==is_action_write) ? 
				  load_statement_precondition(stat) :
				  load_statement_postcondition(stat)));

	result = sc_append(sc_rn(NULL), region);
	result = sc_append(result, a_decl);
	result = sc_append(result, n_decl);
	result = sc_append(result, t_decl);
	result = sc_append(result, p_decl);
	result = sc_append(result, salign);
	result = sc_append(result, sdistr);
	result = sc_append(result, sother);
	result = sc_append(result, contxt);

	/* sc_creer_base(result); */

	/*
	 * ??? what about the variables?
	 * some are usefull, some are constants, and others should
	 * be discarded. This selection and projection may be done here.
	 */

	

	ifdebug(6)
	{
	    fprintf(stderr, 
		    "[generate_io_system] systems for array %s:\n",
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
	    fprintf(stderr, "Context:\n");
	    sc_fprint(stderr, contxt, entity_local_name);
	    fprintf(stderr, "Result:\n");
	    sc_fprint(stderr, result, entity_local_name);
	}	    

    }
    else
    {
	pips_error("generate_io_system",
		   "shared arrays management not implemented yet\n");
    }

    return(result);
}

/*
 * that's all
 */
