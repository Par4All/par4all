/*
 * by Fabien COELHO
 *
 * SCCS stuff:
 * $RCSfile: postcondition.c,v $ ($Date: 1994/03/23 16:53:53 $, ) version $Revision$,
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

/*
 * PIPS stuff
 */

#include "ri-util.h" 
#include "misc.h" 
#include "control.h"
#include "semantics.h"

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
   (hash_put(current_postcondition_map, (char*) (stat), (char*) (val)))

#define StorePre(stat, val) \
   (hash_put(current_precondition_map, (char*) (stat), (char*) (val)))

#define LoadPost(stat) \
    ((transformer) (hash_get(current_postcondition_map, (char*) stat)))

#define LoadPre(stat) \
    ((transformer) (hash_get(current_precondition_map, (char*) stat)))

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
    
    switch(instruction_tag(inst))
    {
    case is_instruction_block:
    {
	list
	    ls = gen_nreverse(gen_copy_seq(instruction_block(inst)));

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

	StorePost(test_true(t), post);
	StorePost(test_false(t), post);
	
        break;
    }
    case is_instruction_loop:
	StorePost(loop_body(instruction_loop(inst)), post);
        break;
    case is_instruction_goto:
	pips_error("postcondition_filter",
		   "unexpected goto encountered\n");
        break;
    case is_instruction_call:
        break;
    case is_instruction_unstructured:
    {
	control
	    c = unstructured_control(instruction_unstructured(i));
	list
	    blocks = NIL;

	CONTROL_MAP(ct,	{}, c, blocks);

	blocks = gen_nreverse(blocks);

	MAPL(cs,
	 {
	     statement
		 s = STATEMENT(CAR(cs));

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
static statement postcondition_rewrite(stat)
statement stat;
{
    return(stat);
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

    return(post_map);
}

/*
 * that's all
 */
