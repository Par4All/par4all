/*
 * by Fabien COELHO
 *
 * $RCSfile: postcondition.c,v $ ($Date: 1995/08/10 09:14:17 $, %U)
 * version $Revision$,
 */

/* Standard includes
 */
 
#include <stdio.h>
#include <string.h> 

/* Psystems stuff
 */

#include "types.h"
#include "boolean.h"
#include "vecteur.h"
#include "contrainte.h"
#include "sc.h"

/* Newgen stuff
 */

#include "genC.h"
#include "ri.h" 

/* PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "control.h"
#include "semantics.h"
#include "transformer.h"

/*-------------------------------------------------------------
 *
 * POSTCONDITIONS MAPPING
 *
 */

GENERIC_CURRENT_MAPPING(postcondition, transformer, statement);

static statement_mapping
  current_precondition_map = hash_table_undefined,
  current_postcondition_map = hash_table_undefined;

#define StorePost(stat, val) \
   (debug(9, "StorePost", "storing 0x%x, 0x%x\n", stat, val), \
    hash_put(current_postcondition_map, (char*) (stat), (char*) (val)))

#define StorePre(stat, val) \
   (debug(9, "StorePre", "storing 0x%x, 0x%x\n", stat, val), \
    hash_put(current_precondition_map, (char*) (stat), (char*) (val)))

#define LoadPost(stat) \
    (debug(9, "LoadPost", "loading 0x%x\n", stat), \
     (transformer) (hash_get(current_postcondition_map, (char*) stat)))

#define LoadPre(stat) \
    (debug(9, "LoadPre", "loading 0x%x\n", stat), \
     (transformer) (hash_get(current_precondition_map, (char*) stat)))

/*
 * filter used by gen_recurse:
 * Top-down computation of the postcondition mapping
 */
static bool postcondition_filter(stat)
statement stat;
{
    instruction
	inst = statement_instruction(stat);
    transformer
	post = LoadPost(stat);

    debug(5, "postcondition_filter", "statement 0x%x\n", stat);
    
    switch(instruction_tag(inst))
    {
    case is_instruction_block:
    {
	list
	    ls = gen_nreverse(gen_copy_seq(instruction_block(inst)));

	debug(6, "postcondition_filter", "in block\n");

	MAPL(cs,
	 {
	     statement
		 s = STATEMENT(CAR(cs));

	     StorePost(s, post);
	     post = LoadPre(s);
	 },
	     ls);

	gen_free_list(ls);
        break;
    }
    case is_instruction_test:
    {
	test
	    t = instruction_test(inst);

	debug(6, "postcondition_filter", "in test\n");

	StorePost(test_true(t), post);
	StorePost(test_false(t), post);
	
        break;
    }
    case is_instruction_loop:
	debug(6, "postcondition_filter", "in loop\n");
	StorePost(loop_body(instruction_loop(inst)), post);
        break;
    case is_instruction_goto:
	/* ??? may be false... */
	pips_error("postcondition_filter",
		   "unexpected goto encountered\n");
        break;
    case is_instruction_call:
        break;
    case is_instruction_unstructured:
	/* ??? may be false... */
    {
	control
	    c = unstructured_control(instruction_unstructured(inst));
	list
	    blocks = NIL;

	debug(6, "postcondition_filter", "in unstructured\n");

	CONTROL_MAP(ct,	{}, c, blocks);

	blocks = gen_nreverse(blocks);

	MAPL(cs,
	 {
	     statement
		 s = control_statement(CONTROL(CAR(cs)));

	     StorePost(s, post);
	     post = LoadPre(s);
	 },
	     blocks);

	gen_free_list(blocks);

        break;
    }
    default:
        pips_error("postcondition_filter",
		   "unexpected instruction tag\n");
        break;
    }

    return(TRUE); /* must go downward */
}

/*
 * Bottom-up pass: nothing to be down
 */
static void postcondition_rewrite(stat)
statement stat;
{
    return;
}

/*
 * statement_mapping compute_postcondition(stat, post_map, pre_map)
 * statement stat;
 * statement_mapping post_map, pre_map;
 *
 * computes the postcondition mapping post_map from the
 * precondition mapping pre_map and the related statement tree 
 * starting from stat. The rule applied is that the postcondition
 * of one statement is the precondition of the following one.
 * The last postcondition is arbitrary set to transformer_identity,
 * what is not enough. (??? should I take the stat transformer?)
 */
statement_mapping compute_postcondition(stat, post_map, pre_map)
statement stat;
statement_mapping post_map, pre_map;
{
    debug_on("SEMANTICS_POSTCONDITION_DEBUG_LEVEL");
    debug(1, "compute_postcondition", "computing!\n");

    current_postcondition_map = post_map;
    current_precondition_map = pre_map;

    /* the initial postcondition is empty ??? */
    StorePost(stat, transformer_identity());

    /* Top-down definition */
    gen_recurse(stat,
		statement_domain,
		postcondition_filter,
		postcondition_rewrite);

    current_precondition_map = hash_table_undefined;
    current_postcondition_map = hash_table_undefined;

    debug_off();

    return(post_map);
}

/*
 * that's all
 */
