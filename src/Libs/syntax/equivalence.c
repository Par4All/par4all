/* 	%A% ($Date: 1998/04/14 21:28:14 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.	 */

#ifndef lint
char vcid_syntax_equivalence[] = "%A% ($Date: 1998/04/14 21:28:14 $, ) version $Revision$, got on %D%, %T% [%P%].\n Copyright (c) École des Mines de Paris Proprietary.";
#endif /* lint */

/* equivalence.c: contains EQUIVALENCE related routines */

#include <stdio.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "parser_private.h"

#include "misc.h"

#include "syntax.h"

#define EQUIADD 0
#define EQUIMER 1

/* external variables used by functions from equivalence.c
 */
static equivalences TempoEquivSet = equivalences_undefined;
static equivalences FinalEquivSet = equivalences_undefined;

/* re-initialize chains between two successives calls to parser
 */
void 
ResetChains()
{
    TempoEquivSet = equivalences_undefined;
    FinalEquivSet = equivalences_undefined;
}


/* this function creates an atom of an equivalence chain. s is a
reference to a variable. */

atom 
MakeEquivAtom(s)
syntax s;
{
    reference r;
    entity e;
    int o;

    if (!syntax_reference_p(s)) {
	pips_assert("This is syntax is a call", syntax_call_p(s));
	pips_user_warning("A function call to %s has been identified by the parser "
			  "in an EQUIVALENCE statement. Maybe an array declaration "
			  "should be moved up ahead of the EQUIVALENCE\n",
			  module_local_name(call_function(syntax_call(s))));
	ParserError("MakeEquivAtom", "function call in equivalence chain\n");
    }

    r = syntax_reference(s);
    e = reference_variable(r);

    /* what is the offset of this reference ? */
    o = OffsetOfReference(r);

    debug(8, "MakeEquivAtom", "Offset %d for reference to %s\n",
	  o, entity_local_name(e));

    return(make_atom(e, o));
}

/* this function is called when an equivalence chain has been completely
parsed. it looks for the atom with the biggest offset, and then
substracts this maximum offset from all atoms. The result is that each
atom has its offset from the begining of the chain. */

void 
StoreEquivChain(c)
chain c;
{
    cons * pc;
    int maxoff;

    maxoff = 0;
    for (pc = chain_atoms(c); pc != NIL; pc = CDR(pc)) {
	int o = atom_equioff(ATOM(CAR(pc)));

	if (o > maxoff)
		maxoff = o;
    }

    debug(9, "StoreEquivChain", "maxoff %d\n", maxoff);

    if (maxoff > 0) {
	for (pc = chain_atoms(c); pc != NIL; pc = CDR(pc)) {
	    atom a = ATOM(CAR(pc));

	    atom_equioff(a) = abs(atom_equioff(a)-maxoff);
	}
    }

    if (TempoEquivSet == equivalences_undefined) {
	TempoEquivSet = make_equivalences(NIL);
    }

    equivalences_chains(TempoEquivSet) = 
	    CONS(CHAIN, c, equivalences_chains(TempoEquivSet));
}

/* this function merges all the equivalence chains to take into account
equivalences due to transitivity. it is called at the end of the parsing. */

void 
ComputeEquivalences()
{
    cons *pc;
    int again = TRUE;

    debug(8, "ComputeEquivalences", "Begin\n");

    if (TempoEquivSet == equivalences_undefined) {
	debug(8, "ComputeEquivalences", "Useless call, end\n");
	    return;
    }

    if (FinalEquivSet == equivalences_undefined) {
	FinalEquivSet = make_equivalences(NIL);
    }

    debug(8, "ComputeEquivalences", "Initial equivalence chains\n");

    PrintChains(TempoEquivSet);

    while (again) {
	for (pc = equivalences_chains(TempoEquivSet); pc != NIL; pc = CDR(pc))
		again = (AddOrMergeChain(CHAIN(CAR(pc))) == EQUIMER);

	gen_free(TempoEquivSet);

	if (again) {
	    TempoEquivSet = FinalEquivSet;
	    FinalEquivSet = make_equivalences(NIL);
	}
    }

    debug(8, "ComputeEquivalences", "Resulting equivalence chains\n");

    PrintChains(FinalEquivSet);

    debug(8, "ComputeEquivalences", "End\n");
}

/* this function adds a chain ct to the set of equivalences. if the
intersection with all other chains is empty, ct is just added to the
set.  Otherwise ct is merged with the chain that intersects ct. */

int 
AddOrMergeChain(ct)
chain ct;
{
    cons *pcl, *pcf, *pct;

    pct = chain_atoms(ct);
    chain_atoms(ct) = NIL;
    
    for (pcl = equivalences_chains(FinalEquivSet); pcl != NIL; pcl=CDR(pcl)) {
	chain cf;

	cf = CHAIN(CAR(pcl));
	pcf = chain_atoms(cf);

	if (ChainIntersection(pct, pcf)) {
	    chain_atoms(cf) = MergeTwoChains(pct, pcf);
	    return(EQUIMER);
	}
    }

    equivalences_chains(FinalEquivSet) =
	    CONS(CHAIN, make_chain(pct), 
		 equivalences_chains(FinalEquivSet));

    return(EQUIADD);
}



/* this function returns TRUE if the there is a variable that occurs in 
both atom lists. */

int 
ChainIntersection(opc1, opc2)
cons *opc1, *opc2;
{
    cons *pc1, *pc2;
    
    for (pc1 = opc1; pc1 != NIL; pc1 = CDR(pc1))
	    for (pc2 = opc2; pc2 != NIL; pc2 = CDR(pc2))
		    if (gen_eq((atom_equivar(ATOM(CAR(pc1)))),
			       (atom_equivar(ATOM(CAR(pc2))))))
			    return(TRUE);

    return(FALSE);
}



/* this function merges two equivalence chains whose intersection is not
empty, ie. one variable occurs in both chains. */

cons * 
MergeTwoChains(opc1, opc2)
cons *opc1, *opc2;
{
    int deltaoff;
    cons *pctemp, *pc1, *pc2=NIL;

    for (pc1 = opc1; pc1 != NIL; pc1 = CDR(pc1)) {
	for (pc2 = opc2; pc2 != NIL; pc2 = CDR(pc2)) {
	    if (gen_eq((atom_equivar(ATOM(CAR(pc1)))),
		       (atom_equivar(ATOM(CAR(pc2)))))) {
		break;
	    }
	}
	if (pc2 != NIL)
		break;
    }

    if (pc1 == NIL || pc2 == NIL)
	    FatalError("MergeTwoChains", "empty intersection\n");

    deltaoff = atom_equioff(ATOM(CAR(pc1)))-atom_equioff(ATOM(CAR(pc2)));

    if (deltaoff < 0) {
	pctemp = opc2; opc2 = opc1; opc1 = pctemp;
    }
    pc1 = opc1;
    pc2 = opc2;

    while (1) {
	atom_equioff(ATOM(CAR(pc2))) += abs(deltaoff);
	if (CDR(pc2) == NIL)
		break;
	pc2 = CDR(pc2);
    }

    CDR(pc2) = pc1;
    return(opc2);
}

/* two debugging functions, just in case ... */

void 
PrintChains(e)
equivalences e;
{
    cons *pcc;

    for (pcc = equivalences_chains(e); pcc != NIL; pcc = CDR(pcc)) {
	PrintChain(CHAIN(CAR(pcc)));
    }
}

void 
PrintChain(c)
chain c;
{
    cons *pca;
    atom a;

    ifdebug(9) {
	debug(9, "PrintChain", "Begin: ");

	for (pca = chain_atoms(c); pca != NIL; pca = CDR(pca)) {
	    a = ATOM(CAR(pca));

	    (void) fprintf(stderr, "(%s,%d) ; ",
			   entity_name(atom_equivar(a)), atom_equioff(a));
	}
	(void) fprintf(stderr, "\n");
	debug(9, "PrintChain", "\n");
    }
}

bool 
entity_in_equivalence_chains_p(entity e)
{
    equivalences equiv = TempoEquivSet;
    list pcc;
    bool is_in_p = FALSE;

    /* Apparently, TempoEquivSet stays undefined when there are no equivalences */
    if(!equivalences_undefined_p(equiv)) {
	for (pcc = equivalences_chains(equiv); !ENDP(pcc) && !is_in_p; POP(pcc)) {
	    is_in_p = entity_in_equivalence_chain_p(e, CHAIN(CAR(pcc)));
	}
    }

    return is_in_p;
}

bool
entity_in_equivalence_chain_p(entity e, chain c)
{
    cons *pca;
    atom a;
    bool is_in_p = FALSE;

    debug(9, "entity_in_equivalence_chain_p", "Begin for entity %s \n", entity_name(e));

    for (pca = chain_atoms(c); !ENDP(pca) && !is_in_p; POP(pca)) {
	a = ATOM(CAR(pca));

	is_in_p = (atom_equivar(a) == e);
    }
    debug(9, "PrintChain", "End\n");
    return is_in_p;
}

/* This function computes an address for each variable. All common
 * variables already have their own addresses. If such a variable occurs in
 * an equivalence chain, all variables of this chain will have an address
 * in this common. The exact address depends on the offset stored in the
 * atom. 
 * 
 * The same kind of processing is done for chains containing a static
 * variable. 
 * 
 * Otherwise, all variables of a chain have an address in the
 * dynamic area.
 */

void 
ComputeAddresses()
{
    cons *pcc, *pca, *pcv;
    entity sc;
    int lc, l, ac;
    list dynamic_aliases = NIL;

    if (FinalEquivSet != equivalences_undefined) {
	for (pcc = equivalences_chains(FinalEquivSet); pcc != NIL; 
	     pcc = CDR(pcc)) {
	    chain c = CHAIN(CAR(pcc));

	    /* the default section for variables with no address is the dynamic
	     * area.
	     */
	    sc = DynamicArea;
	    lc = 0;
	    ac = 0;

	    /* Try to locate the area for the current equivalence chain.
	     * Only one variable should have a well-defined storage.
	     * Or no variables have one because the equivalence chain 
	     * is located in the dynamic area
	     */
	    for (pca = chain_atoms(c); pca != NIL; pca = CDR(pca)) {
		entity e;
		int o;

		e = atom_equivar(ATOM(CAR(pca)));
		o = atom_equioff(ATOM(CAR(pca)));

		/* FI: I do not understand why this assignment is not better guarded.
		 * Maybe, because lc's later use *is* guarded.
		 */
		if ((l = SafeSizeOfArray(e)) > lc-o)
			lc = l+o;

		if (entity_storage(e) != storage_undefined) {
		    if (storage_ram_p(entity_storage(e))) {
			ram r = storage_ram(entity_storage(e));

			if (sc != ram_section(r)) {
			    if (sc == DynamicArea) {
				sc = ram_section(r);
				ac = ram_offset(r)-o;
			    } 
			    else if (sc == StaticArea) {
				/* A variable may be located in a static area because
				 * of a SAVE or a DATA statement and be equivalenced
				 * with a variable in a common.
				 */
				pips_assert("ComputeAddresses", ram_section(r) != DynamicArea);
				sc = ram_section(r);
				ac = ram_offset(r)-o;
			    } 
			    else {
				user_warning("ComputeAddresses",
					     "Incompatible default area %s and "
					     "area %s requested by equivalence for %s\n",
					     entity_name(sc), entity_name(ram_section(r)),
					     entity_local_name(e));
				    ParserError("ComputeAddresses",
					    "incompatible areas\n");
			    }
			}
		    }
		    else
			    FatalError("ComputeAddresses", "non ram storage\n");
		}
	    }

	    /* Compute the offset and set the storage information for each
	     * variable in the equivalence chain that has no storage yet.
	     */
	    for (pca = chain_atoms(c); pca != NIL; pca = CDR(pca)) {
		entity e;
		int o, adr;

		e = atom_equivar(ATOM(CAR(pca)));
		o = atom_equioff(ATOM(CAR(pca)));

		if (sc == DynamicArea) {
		    ac = area_size(type_area(entity_type(sc)));
		}

		/* check that the offset is positive */
		if ((adr = ac+o) < 0) {
		    user_warning("ComputeAddresses", "Offset %d for %s in common /%s/.\n",
				 ac+o, entity_local_name(e), entity_local_name(sc));
		    ParserError("ComputeAddresses", 
				"Attempt to extend common backwards. "
				"Have you checked the code with a Fortran compiler?\n");
		}

		if ((entity_storage(e)) != storage_undefined) {
		    ram r;
		    r = storage_ram(entity_storage(e));

		    if (adr != ram_offset(r)) {
			user_warning("ComputeAddresses",
				     "Two conflicting offsets for %s: %d and %d\n",
				     entity_local_name(e), adr, ram_offset(r));
			ParserError("ComputeAddresses", "incompatible addresses\n");
		    }
		}
		else {
		    area a = type_area(entity_type(sc));

		    entity_storage(e) = 
			    make_storage(is_storage_ram,
					 (make_ram(get_current_module_entity(), 
						   sc, adr, NIL)));
		    /* Add e in sc'layout and check that sc's size
		     * does not have to be increased as for:
		     * COMMON /FOO/X
		     * REAL Y(100)
		     * EQUIVALENCE (X,Y)
		     */
		    pips_assert("Entity e is not yet in sc's layout", 
				!entity_is_argument_p(e,area_layout(a)));
		    if(sc == DynamicArea && pca != chain_atoms(c)) {
			/* Dynamic aliases cannot be added right away because
			 * implicitly declared dynamic variables still have to
			 * be added whereas aliased variables are assumed to be
			 * put behind in the layout list. Except the first one.
			 */
			dynamic_aliases = arguments_add_entity(dynamic_aliases, e);
		    }
		    else {
			area_layout(a) = gen_nconc(area_layout(a),
						   CONS(ENTITY, e, NIL));
		    }

		    /* If sc really is a common, i.e. neither the *dynamic*
		     * nor the *static* area, check its size
		     */
		    if(top_level_entity_p(sc)) {
			int s = common_to_size(sc);
			int new_s = adr + SafeSizeOfArray(e);
			if(s < new_s) {
			    (void) update_common_to_size(sc, new_s);
			}
		    }
		}
	    }

	    if (sc == DynamicArea)
		area_size(type_area(entity_type(sc))) += lc;

	}
    }

    /* All declared variables are scanned and stored in the dynamic area if their
     * storage is still undefined.
     */

    for (pcv = code_declarations(EntityCode(get_current_module_entity())); pcv != NIL;
	 pcv = CDR(pcv)) {
	entity e = ENTITY(CAR(pcv));

	if (entity_storage(e) == storage_undefined) {
	    /* FI: I do not understand how this could work because
	     * DynamicArea is managed directly and not thru common_size_map
	     *
	     * This could be hidden in CurrentOffsetOfArea()...
	     *
	     * FI, 16 June 1997
	     */
	    entity_storage(e) = 
		    make_storage(is_storage_ram,
				 (make_ram(get_current_module_entity(), 
					   DynamicArea, 
					   CurrentOffsetOfArea(DynamicArea,
							       e), NIL)));
	}
    }

    /* Add aliased dynamic variables */
    if(!ENDP(dynamic_aliases)) {
	/* neither gen_concatenate() nor gen_append() are OK */
	list dynamic = area_layout(type_area(entity_type(DynamicArea)));

	pips_assert("aliased dynamic variables imply standard dynamic variables",
		    !ENDP(dynamic));
	/* side effect on area_layout */
	(void) gen_nconc(dynamic, dynamic_aliases);
    }
}

/* Initialize the shared fields of aliased variables */
void 
SaveChains()
{
    debug(8, "SaveChains", "Begin\n");

    if (FinalEquivSet == equivalences_undefined) {
    debug(8, "SaveChains", "No equivalence to process. End\n");
	return;
    }

    MAPL(pc, {
	cons *shared = NIL;
	chain c = CHAIN(CAR(pc));

	debug(8, "SaveChains", "Process an equivalence chain:\n");

	MAPL(pa, {
	    shared = CONS(ENTITY, atom_equivar(ATOM(CAR(pa))), shared);
	}, chain_atoms(c));
	
	pips_assert("SaveChains", !ENDP(shared));

	MAPL(pa, {
	    atom a = ATOM(CAR(pa));
	    entity e = atom_equivar(a);
	    storage se = entity_storage(e);
	    ram re = storage_ram(se);

	    debug(8, "SaveChains", "\talias %s\n", entity_name(e));

	    ram_shared(re) = gen_copy_seq(shared);

	}, chain_atoms(c));
	
    }, equivalences_chains(FinalEquivSet));

    debug(8, "SaveChains", "End\n");
}
