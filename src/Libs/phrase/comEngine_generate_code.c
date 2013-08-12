/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
/*
This file contains functions used to generate the MMCDs generation code and
The HRE code if we want to have only one process on the HRE
 */

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "effects-util.h"

#include "text-util.h"
#include "properties.h"


#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "phrase_tools.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "phrase_distribution.h"
#include "comEngine.h"
#include "phrase.h"

hash_table gLoopToOuterInd;

// These variable is used to know if we are generating the MMCD code
// or the HRE code
bool gGenHRE = false;

static bool COM_OPT;

// This variable if the execution MMCD have been put for the current
// computation kernel
static bool gExecPut;

static list glCurRep = NIL;
static hash_table gHtCurNewInd;
static list glCurLoop = NIL;

// This is used to make sure that the loop indices out effects are correct
static list glOutEffStats = NIL;

// This stores the initialization for the toggle variables. These variables
// perform a switch mechanism to make sure that, because of the
// pipeline, one fifo will not be used to load data, compute data and store
// data at the same step.
list glToggleInitStats;

/*
This function create a new scalar variable
 */
entity comEngine_make_new_scalar_variable(const char* prefix,
					  basic bas)
{
  entity retEnt = entity_undefined;
  make_new_scalar_variable_with_prefix(prefix,
					     get_current_module_entity(),
					     bas);
  AddEntityToCurrentModule(retEnt);

  return retEnt;
}

/*
This function is called to prevent the optimization phase if a loop contains
some data dependence
 */
static void fill_gLoopToOpt(list lRef)
{
  if(lRef != NIL)
    {
      MAP(STATEMENT, loopStat,
      {
	hash_put(gLoopToOpt, loopStat, (void *)false);
      }, glCurLoop);
    }
}

/*
This function associate a toggle entities list to a loop statement 
from the list of reference lRef.
 */
static void fill_gLoopToToggleEnt(statement stat, list lRef)
{
  list lDone = NIL;

  // Go through the reference list attached to this loop
  FOREACH(REFERENCE, curRef, lRef)
  {
    // Get the fifo number for the current reference
    intptr_t fifoNum = (intptr_t)hash_get(gRefToFifo, curRef);

    pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
		fifoNum != (intptr_t)HASH_UNDEFINED_VALUE);

    // Get the fifo offset for the cureent reference
    intptr_t inc = (intptr_t)hash_get(gRefToFifoOff, (void *)fifoNum);

    pips_assert("inc != HASH_UNDEFINED_VALUE",
		inc != (intptr_t)HASH_UNDEFINED_VALUE);

    bool bDone = false;

    // If a toggle variable has already been created for this loop,
    // then continue
    MAP(INT, incDone,
    {
      if(inc == incDone)
	{
	  bDone = true;
	  break;
	}
    }, lDone);

    if(bDone)
      continue;

    lDone = CONS(INT, inc, lDone);

    // Create a new toggle variable
    entity toggleEnt = 
      comEngine_make_new_scalar_variable(strdup("toggle"),
					 make_basic(is_basic_int, (void *)4));

    list lToggleEnt = hash_get(gLoopToToggleEnt, stat);

    if(lToggleEnt == HASH_UNDEFINED_VALUE)
      {
	lToggleEnt = NIL;
      }

    // Store the new variable in the list of toggle variables
    lToggleEnt = CONS(ENTITY, toggleEnt, lToggleEnt);

    hash_put(gLoopToToggleEnt, stat, lToggleEnt);

    // Associate the value corresponding to the fifo offset to the toggle
    // variable
    hash_put(gToggleToInc, toggleEnt, (void *)inc);

  }
}

/*
This function makes a statement to initialize a toggle variable
 */
statement make_toggle_init_statement(entity toggleEnt)
{
  return make_assign_statement(entity_to_expression(toggleEnt),
			       int_to_expression(0));
}

/*
This function updates the toggle variables initialization statements list.
 */
static void update_toggle_init_stats_list(statement stat)
{
  list lToggleEnt = hash_get(gLoopToToggleEnt, stat);

  pips_assert("lToggleEnt != HASH_UNDEFINED_VALUE",
	      lToggleEnt != HASH_UNDEFINED_VALUE);

  list lStats = NIL;

  MAP(ENTITY, toggleEnt,
  {
    // Create an initialization statement for the curren toggle variable
    statement toggleStat = make_toggle_init_statement(toggleEnt);

    lStats = gen_nconc(lStats, CONS(STATEMENT, toggleStat, NIL));
  }, lToggleEnt);

  glToggleInitStats = gen_nconc(glToggleInitStats, lStats);
}

/*
This function makes a statement to increment the value of the toggle
variable toggleEnt.
 */
statement make_toggle_inc_statement(entity toggleEnt, int val)
{
  list modArg = gen_make_list(expression_domain,
			      entity_to_expression(toggleEnt),
			      int_to_expression(val),
			      NULL);

  expression modExp =
    call_to_expression(make_call(entity_intrinsic(C_MODULO_OPERATOR_NAME),
				 modArg));

  return make_assign_statement(entity_to_expression(toggleEnt),
			       modExp);
}

/*
This function makes statements to increment the value of the toggle
variables associated to the loop
 */
static void add_toggle_inc_statements(statement stat, list lInStats)
{
  list lToggleEnt = hash_get(gLoopToToggleEnt, stat);

  pips_assert("lToggleEnt != HASH_UNDEFINED_VALUE",
	       lToggleEnt!= HASH_UNDEFINED_VALUE);

  // The loop we consider is the first statement of the list lInStats
  statement newLoopStat = STATEMENT(CAR(lInStats));

  pips_assert("newLoopStat is a loop statement",
	      statement_loop_p(newLoopStat));

  statement body = loop_body(statement_loop(newLoopStat));

  list lStats = CONS(STATEMENT, body, NIL);

  MAP(STATEMENT, loopStat,
  {
    if(loopStat == STATEMENT(CAR(glCurLoop)))
      {
	continue;
      }

    entity inInd = hash_get(gHtCurNewInd, loopStat);

    statement indStat = make_init_newInd_stat(loopStat, inInd);

    if(indStat != statement_undefined)
      {
	lStats = gen_nconc(CONS(STATEMENT, indStat, NIL), lStats);
      }
  }, glCurLoop);

  // For each toggle variable, create a statement to increment it
  MAP(ENTITY, toggleEnt,
  {
    statement mmcdStat = make_toggle_mmcd(toggleEnt);

    lStats = gen_nconc(CONS(STATEMENT, mmcdStat, NIL), lStats);

    intptr_t inc = (intptr_t)hash_get(gToggleToInc, toggleEnt);

    statement toggleStat =
      make_toggle_inc_statement(toggleEnt, inc);

    lStats = gen_nconc(lStats, CONS(STATEMENT, toggleStat, NIL));
  }, lToggleEnt);

  loop_body(statement_loop(newLoopStat)) = make_block_statement(lStats);
}

/*
This function performs some transformation on the inner loop of the 
loop tiling operation
 */
static statement make_loopStat1(statement stat, entity transferSize,
				statement innerStat, entity newOuterInd,
				list lSupportedRef, hash_table htOffset, 
				expression bufferSizeExp, entity newInnerInd)
{
  loop curLoop = statement_loop(stat);

  if(innerStat != statement_undefined)
    {
      expression indExp = entity_to_expression(newInnerInd);

      // If we are generating the HRE code, then replace the references by
      // the private HRE references
      if(gGenHRE)
	{
	  if(!COM_OPT)
	    {
	      process_replacement_HRE(lSupportedRef, indExp, &loop_body(statement_loop(innerStat)));
	    }
	}

      free_expression(indExp);
    }

  // Add some statements to the innerStat
  list lStats = NIL;
  if(gGenHRE)
    {
      lStats = make_loop_lStats_HRE(stat, transferSize,
				    innerStat, newOuterInd,
				    lSupportedRef, htOffset,
				    bufferSizeExp);
    }
  else
    {
      lStats = make_loop_lStats_proc(stat, transferSize,
				     innerStat, newOuterInd,
				     lSupportedRef, htOffset,
				     bufferSizeExp);
    }

  innerStat = make_block_statement(lStats);

  // The following lines create the outer loop of the loop tiling
  // operation
  range outerRange1 = make_range(copy_expression(range_lower(loop_range(curLoop))),
				 copy_expression(range_upper(loop_range(curLoop))),
				 copy_expression(bufferSizeExp));

  loop outerLoop1 = make_loop(newOuterInd,
			      outerRange1,
			      innerStat,
			      loop_label(curLoop),
			      make_execution(is_execution_sequential, UU),
			      NIL);

  return make_statement(entity_empty_label(),
			STATEMENT_NUMBER_UNDEFINED,
			STATEMENT_ORDERING_UNDEFINED,
			empty_comments,
			make_instruction_loop(outerLoop1),NIL,NULL,
			empty_extensions (), make_synchronization_none());
}

#if 0
static statement make_init_prev_ind(statement oldBody)
{
  list lStats = NIL;

  MAP(STATEMENT, loopStat,
  {
    if(loopStat == STATEMENT(CAR(glCurLoop)))
      {
	continue;
      }

    entity inInd = hash_get(gHtCurNewInd, loopStat);

    statement indStat = make_init_newInd_stat(loopStat, inInd);

    if(indStat != statement_undefined)
      {
	lStats = gen_nconc(CONS(STATEMENT, indStat, NIL), lStats);
      }
  }, glCurLoop);

  lStats = gen_nconc(lStats, CONS(STATEMENT, oldBody, NIL));

  return make_block_statement(lStats);
}
#endif

/*
This function performs a loop tiling operation from the loop statement
stat.
 */
static statement usual_loop_tiling(statement stat, statement newBody,
				   list lSupportedRef, list lUnSupportedRef,
				   hash_table htOffset, expression bufferSizeExp,
				   entity newOuterInd, entity newInnerInd)
{
  loop curLoop = statement_loop(stat);

  statement innerStat1 = statement_undefined;

  statement loopStat1 = statement_undefined;

  entity transferSize =
    comEngine_make_new_scalar_variable(strdup("transSize"),
				       make_basic(is_basic_int, (void*)4));

  // If the new body is not undefined, then create the inner loop.
  if((newBody != statement_undefined) ||
     (lUnSupportedRef != NIL))
    {
      // If there are some unsupported references, then let us add some
      // statements to the code to support them
      if(lUnSupportedRef != NIL)
	{
	  // Let us do what needs to be done to handle the references list
	  if(gGenHRE)
	    {
	      if(!COM_OPT)
		{
		  process_replacement_HRE(lUnSupportedRef, expression_undefined, &newBody);
		}

	      newBody = generate_stat_from_ref_list_HRE(lUnSupportedRef, newBody);
	    }
	  else
	    {
	      list lToggleEnt = hash_get(gLoopToToggleEnt,
					 loop_body(statement_loop(stat)));

	      if(lToggleEnt == HASH_UNDEFINED_VALUE)
		{
		  lToggleEnt = NIL;
		}

	      newBody = generate_stat_from_ref_list_proc(lUnSupportedRef,
							 lToggleEnt,
							 newBody);

	      fill_gLoopToOpt(lUnSupportedRef);
	    }
	}
      else if(newBody == statement_undefined)
	{
	  newBody = make_block_statement(NIL);
	}

      //newBody = make_init_prev_ind(newBody);

      // The following lines create the inner loop of the loop tiling
      // operation
      range innerRange = make_range(int_to_expression(0),
				    entity_to_expression(transferSize),
				    int_to_expression(1));

      loop innerLoop = make_loop(newInnerInd,
				 innerRange,
				 newBody,
				 loop_label(curLoop),
				 make_execution(is_execution_sequential, UU),
				 NIL);

      innerStat1 = make_statement(entity_empty_label(),
				  STATEMENT_NUMBER_UNDEFINED,
				  STATEMENT_ORDERING_UNDEFINED,
				  empty_comments,
				  make_instruction_loop(innerLoop),NIL,NULL,
				  empty_extensions (), make_synchronization_none());

      if(!gGenHRE)
	{
	  process_innerStat1_proc(innerStat1, loop_index(curLoop),
				  newOuterInd, newInnerInd);
	}

      //printf("innerStat1\n");
      //print_statement(innerStat1);

      // Let us try and see if we can optimize the transfers
      if(COM_OPT)
	{
	  innerStat1 = comEngine_opt_loop_interchange(stat, innerStat1,
	  newOuterInd);
	}

    }

  // This function finishes the loop tiling operation
  loopStat1 = make_loopStat1(stat, transferSize,
			     innerStat1, newOuterInd,
			     lSupportedRef, htOffset,
			     bufferSizeExp, newInnerInd);

  return loopStat1;
}

/*
This function generates the MMCD code or the HRE code from a loop
statement.
 */
static statement generate_code_loop(statement stat)
{
  printf("generate_code_loop\n");
  print_statement(stat);

  statement newBody = statement_undefined;

  list lStats = NIL;

  hash_table htOffset = hash_get(gStatToHtOffset, stat);

  if(!gGenHRE)
    {
      pips_assert("htOffset != HASH_UNDEFINED_VALUE",
		  htOffset != HASH_UNDEFINED_VALUE);
    }

  loop curLoop = statement_loop(stat);

  list lRef = hash_get(gLoopToRef, stat);

  if(lRef == HASH_UNDEFINED_VALUE)
    {
      lRef = NIL;
    }

  list lSupportedRef = NIL;
  list lUnSupportedRef = NIL;

  // Store the list of supported and unsupported references for loop
  // statement stat.
  get_supportedRef_HRE(stat,
		       &lSupportedRef, &lUnSupportedRef);

  // Create a new index
  entity newOuterInd =
    comEngine_make_new_scalar_variable(entity_local_name(loop_index(curLoop)),
				       copy_basic(entity_basic(loop_index(curLoop))));

  // Memorize the index created in the hash_table gLoopToOuterInd
  hash_put(gLoopToOuterInd, stat, newOuterInd);

  // Create a new index
  entity newInnerInd = 
    comEngine_make_new_scalar_variable(entity_local_name(loop_index(curLoop)),
				       copy_basic(entity_basic(loop_index(curLoop))));

  // If we are generating the MMCD code, then create the fifos for the
  // supported and unsupported references
  if(!gGenHRE)
    {
      create_realFifo_proc(stat, lSupportedRef);
      create_realFifo_proc(stat, lUnSupportedRef);
    }

  // Create the toggle references and store them in gLoopToToggleEnt
  fill_gLoopToToggleEnt(stat, lSupportedRef);
  fill_gLoopToToggleEnt(loop_body(statement_loop(stat)), lUnSupportedRef);

  // Add the loop to the current loops list
  glCurLoop = CONS(STATEMENT, stat, glCurLoop);

  // Memorize the current newInnerInd in gHtCurNewInd
  hash_put(gHtCurNewInd, stat, newInnerInd);

  hash_put(gLoopToOpt, stat, (void *)true);

  gExecPut = false;

  // Get the new statements for the loop body
  newBody = generate_code_function(loop_body(curLoop), false);

  gExecPut = false;

  printf("newBody real\n");
  if(newBody == statement_undefined)
    {
      printf("undefined\n");
    }
  else
    {
      print_statement(newBody);
    }

  // If there are some supported references, the let us perform a loop
  // tiling operation
  if(lSupportedRef != NIL)
    {
      expression bufferSizeExp = expression_undefined;

      bufferSizeExp = copy_expression(gBufferSizeEnt);

      statement newStat = usual_loop_tiling(stat, newBody,
					    lSupportedRef, lUnSupportedRef,
					    htOffset, bufferSizeExp,
					    newOuterInd, newInnerInd);

      list lToggleEnt = hash_get(gLoopToToggleEnt, stat);

      if(lToggleEnt != HASH_UNDEFINED_VALUE)
	{
	  printf("put lToggleEnt\n");
	  print_statement(newStat);
	  hash_put(gLoopToToggleEnt, newStat, gen_copy_seq(lToggleEnt));
	}

      lStats = CONS(STATEMENT, newStat, NIL);

      if(!gGenHRE)
	{
	  glOutEffStats = add_index_out_effect_proc(curLoop, glOutEffStats);
	}
    }
  // else if the new body is not undefined, then just create a loop without
  // loop tiling
  else if(newBody != statement_undefined)
    {
      // If there are some unsupported references, then let us add some
      // statements to the code to support them
      if(lUnSupportedRef != NIL)
	{
	  // Let us do what needs to be done to handle the references list
	  if(gGenHRE)
	    {
	      if(!COM_OPT)
		{
		  process_replacement_HRE(lUnSupportedRef, expression_undefined, &newBody);
		}

	      newBody = generate_stat_from_ref_list_HRE(lUnSupportedRef, newBody);
	    }
	  else
	    {
	      list lToggleEnt = hash_get(gLoopToToggleEnt,
					 loop_body(statement_loop(stat)));

	      if(lToggleEnt == HASH_UNDEFINED_VALUE)
		{
		  lToggleEnt = NIL;
		}

	      newBody = generate_stat_from_ref_list_proc(lUnSupportedRef,
							 lToggleEnt,
							 newBody);

	      fill_gLoopToOpt(lUnSupportedRef);
	    }
	}

      // The following lines create the loop
      loop outerLoop = make_loop(newInnerInd,
				 copy_range(loop_range(curLoop)),
				 newBody,
				 loop_label(curLoop),
				 make_execution(is_execution_sequential, UU),
				 NIL);

      statement newStat = make_statement(entity_empty_label(),
					 STATEMENT_NUMBER_UNDEFINED,
					 STATEMENT_ORDERING_UNDEFINED,
					 empty_comments,
					 make_instruction_loop(outerLoop),NIL,NULL,
					 empty_extensions (), make_synchronization_none());

      lStats = CONS(STATEMENT, newStat, NIL);

      comEngine_replace_reference_in_stat(newBody,
					  make_reference(loop_index(curLoop), NIL),
					  entity_to_expression(newInnerInd));

      if(!gGenHRE)
	{
	  glOutEffStats = add_index_out_effect_proc(curLoop, glOutEffStats);
	}
    }
  else
    {
      return make_block_statement(NIL);
    }

  if(!gGenHRE)
    {
      add_toggle_inc_statements(stat, lStats);

      update_toggle_init_stats_list(stat);
    }

  //pips_assert("newStat != statement_undefined",
  //newStat != statement_undefined);

  // If some references attached to the loop statement but are not
  // some supported or unsupported references, then let us handle them
  list lOutRef = hash_get(gStatToRef, stat);

  if(lOutRef != HASH_UNDEFINED_VALUE)
    {
      // Let us do what needs to be done to handle the references list
      if(gGenHRE)
	{
	  if(!COM_OPT)
	    {
	      lStats = process_replacement_HRE_OutRef(lOutRef, lStats);
	    }

	  lStats = generate_stat_from_ref_list_HRE_list(lOutRef, lStats);
	}
      else
	{
	  lStats = generate_stat_from_ref_list_proc_list(lOutRef, lStats);

	  fill_gLoopToOpt(lOutRef);
	}
    }

  // Add some bubbles in the pipeline if needed
  if(gGenHRE)
    {
      lStats = process_gLoopToSync_HRE(stat, lStats);

      //fill_gLoopToOpt(!NIL);
    }
  else
    {
      lStats = process_gLoopToSync_proc(stat, lStats);

      //fill_gLoopToOpt(!NIL);
    }
  
  if(!gGenHRE)
    {
      hash_table_free(htOffset);
    }

  // Remove the loop statement to the current loops list
  gen_remove(&glCurLoop, stat);

  statement newStat = make_block_statement(lStats);

  printf("newStat real\n");
  if(newStat == statement_undefined)
    {
      printf("newStat == statement_undefined\n");
    }
  else
    {
      print_statement(newStat);
    }

  hash_put(gIsNewLoop, newStat, (void *)true);

  return newStat;
}

/*
This function add to references list lInRef the supported references of
the loop statement stat
 */
static void process_ref_lists(statement stat, list * lInRef)
{
  list lRef = NIL;

  lRef = hash_get(gLoopToSupRef, stat);

  if(lRef == HASH_UNDEFINED_VALUE)
    {
      lRef = NIL;
    }

  MAP(REFERENCE, curRef,
  {
    bool bInRef = false;

    MAP(REFERENCE, supRef,
    {
      if(reference_equal_p(curRef, supRef))
	{
	  bInRef = true;
	  break;
	}
    }, lRef);

    if(bInRef)
      {
	*lInRef = CONS(REFERENCE, curRef, *lInRef);
      }

  }, glCurRep);

  MAP(REFERENCE, curRef,
  {
    gen_remove(&glCurRep, curRef);
  }, *lInRef);
}

/*
This function replaces the old references by the HRE private references
for a call statement
 */
static void process_opt_replace(statement * newStat)
{
  MAP(STATEMENT, loopStat,
  {
    list lIn = NIL;

    process_ref_lists(loopStat, &lIn);

    entity curNewInd = hash_get(gHtCurNewInd, loopStat);

    pips_assert("curNewInd != HASH_UNDEFINED_VALUE",
		curNewInd != HASH_UNDEFINED_VALUE);

    expression indExp = entity_to_expression(curNewInd);

    process_replacement_HRE(lIn, indExp, newStat);

    free_expression(indExp);

    gen_free_list(lIn);

  }, glCurLoop);

  process_replacement_HRE(glCurRep, expression_undefined, newStat);

  fill_gLoopToOpt(glCurRep);

  gen_free_list(glCurRep);
  glCurRep = NIL;
}

/*
This function replaces the old references by the HRE private references
for block statement
 */
static list replace_glCurRep_in_seq(list lStats, list lPrevSeq, list lSeq)
{
  list newLStats = NIL;

  if(lStats == NIL)
    {
      return lSeq;
    }

  statement newStat = make_block_statement(lStats);

  process_opt_replace(&newStat);

  newLStats = CONS(STATEMENT, newStat, NIL);

  if(lPrevSeq != NIL)
    {
      CDR(lPrevSeq) = newLStats;
    }
  else
    {
      lSeq = newLStats;
    }

  return lSeq;
}

/*
This function generates the MMCD code or the HRE code from a block
statement.
 */
static statement generate_code_seq(statement stat)
{
  statement newStat = statement_undefined;

  instruction instr = statement_instruction(stat);

  list lSeq = NIL;
  list lTempSeq = NIL;
  list lPrevSeq = NIL;

  //printf("generate_code_seq\n");

  // Go through each statement of the sequence
  MAP(STATEMENT, curStat,
  {
    if((COM_OPT && gGenHRE)&& !statement_call_p(curStat) &&
       !statement_test_p(curStat))
      {
	lSeq = replace_glCurRep_in_seq(lTempSeq, lPrevSeq, lSeq);

	lTempSeq = NIL;
      }

    // Let us get the new statement for the current sequence statement
    statement seqNewStat = generate_code_function(curStat, true);

    if(seqNewStat != statement_undefined)
      {
	list newStatCons = CONS(STATEMENT, seqNewStat, NIL);

	lSeq = gen_nconc(lSeq, newStatCons);

	if((COM_OPT && gGenHRE)&&!statement_call_p(curStat) &&
	   !statement_test_p(curStat))
	  {
	    lPrevSeq = newStatCons;
	  }

	if((COM_OPT && gGenHRE)&&(lTempSeq == NIL) &&
	   (statement_call_p(curStat) ||
	    statement_test_p(curStat)))
	  {
	    lTempSeq = newStatCons;
	  }
      }

  }, sequence_statements(instruction_sequence(instr)));

  if(COM_OPT && gGenHRE)
    {
      lSeq = replace_glCurRep_in_seq(lTempSeq, lPrevSeq, lSeq);
    }

  if(lSeq == NIL)
    {
      newStat = statement_undefined;
    }
  else if(gen_length(lSeq) == 1)
    {
      newStat = STATEMENT(CAR(lSeq));
    }
  else
    {
      newStat = make_block_statement(lSeq);
    }

  return newStat;
}

/*
This function generate an execute mmcd statement and create a block statement
with this new statement and statement stat.
 */
static statement add_exec_mmcd(statement stat)
{
  statement mmcdStat = make_exec_mmcd();

  statement stepStat = make_step_inc_statement(1);

  mmcdStat = make_block_statement(gen_nconc(CONS(STATEMENT, stepStat, NIL),
					    CONS(STATEMENT, mmcdStat, NIL)));

  if(glCurLoop != NIL)
    {
      statement loopStat = STATEMENT(CAR(glCurLoop));

      list lUnSupportedRef = hash_get(gLoopToUnSupRef, loopStat);

      pips_assert("lUnSupportedRef != HASH_UNDEFINED_VALUE",
		  lUnSupportedRef != HASH_UNDEFINED_VALUE);

      if(lUnSupportedRef == NIL)
	{
	  loop curLoop = statement_loop(loopStat);

	  entity newInd = hash_get(gHtCurNewInd, loopStat);

	  list neArg = gen_make_list(expression_domain,
				     entity_to_expression(newInd),
				     copy_expression(range_lower(loop_range(curLoop))),
				     NULL);

	  expression neExp = 
	    call_to_expression(make_call(entity_intrinsic(C_EQUAL_OPERATOR_NAME),
					 neArg));

	  test t = make_test(neExp, mmcdStat, make_empty_statement());

	  mmcdStat = make_statement(entity_empty_label(), 
				    STATEMENT_NUMBER_UNDEFINED,
				    STATEMENT_ORDERING_UNDEFINED,
				    empty_comments,
				    make_instruction(is_instruction_test, t),
				    NIL,NULL,
				    empty_extensions (), make_synchronization_none());
	}
    }

  if(stat == statement_undefined)
    {
      return mmcdStat;
    }
  else
    {
      return make_block_statement(gen_nconc(CONS(STATEMENT, stat, NIL),
					    CONS(STATEMENT, mmcdStat, NIL)));
    }
}

/*
This function generates the MMCD code or the HRE code from a call
statement.
 */
static statement generate_code_call(statement stat, bool bCalledFromSeq)
{
  //printf("generate_code_call\n");
  //print_statement(stat);printf("\n");
  statement newStat = statement_undefined;

  if(gGenHRE)
    {
      newStat = get_call_stat_HRE(stat);
    }
  else
    {
      newStat = get_call_stat_proc(stat);
    }

  if(COM_OPT && gGenHRE)
    {
      list lCallRef = NIL;
      call curCall = instruction_call(statement_instruction(stat));

      MAP(EXPRESSION, exp,
      {
	list old = lCallRef;
	list new = NIL;
	new = comEngine_expression_to_reference_list(exp, new);

	lCallRef = gen_concatenate(old, new);

	gen_free_list(old);
	gen_free_list(new);

      }, call_arguments(curCall));

      glCurRep = gen_nconc(glCurRep, lCallRef);

      if(!bCalledFromSeq)
	{
	  process_opt_replace(&newStat);
	}
    }

  if(!get_bool_property("COMENGINE_CONTROL_IN_HRE")
     && !gGenHRE && !gExecPut)
    {
      newStat = add_exec_mmcd(newStat);

      gExecPut = true;
    }

  if(get_bool_property("COMENGINE_CONTROL_IN_HRE") &&
     !gExecPut)
    {
      statement stepStat;

      if(gGenHRE)
	{
	  stepStat = make_wait_step_statement();
	}
      else
	{
	  stepStat = make_step_inc_statement(1);
	}

      if(newStat == statement_undefined)
	{
	  newStat = stepStat;
	}
      else
	{
	  newStat =
	    make_block_statement(gen_nconc(CONS(STATEMENT, stepStat, NIL),
					   CONS(STATEMENT, newStat, NIL)));
	}

      gExecPut = true;
    }

  list lRef = hash_get(gStatToRef, stat);

  if(lRef == HASH_UNDEFINED_VALUE)
    {
      return newStat;
    }

  if(gGenHRE)
    {
      if(!COM_OPT)
	{
	  process_replacement_HRE(lRef, expression_undefined, &newStat);
	}

      newStat = generate_stat_from_ref_list_HRE(lRef, newStat);
    }
  else
    {
      newStat = generate_stat_from_ref_list_proc(lRef, NIL, newStat);

      fill_gLoopToOpt(lRef);
    }
      printf("generate_code_call 2\n");
  return newStat;
}

/*
This functions handles the references that have been attached to the 
g_externalized_code during the analysis phase
 */
static statement process_code_seq(statement newStat, statement stat)
{
  list lRef = hash_get(gStatToRef, stat);

  if(lRef == HASH_UNDEFINED_VALUE)
    {
      return newStat;
    }

  if(gGenHRE)
    {
      if(!COM_OPT)
	{
	  process_replacement_HRE(lRef, expression_undefined, &newStat);
	}

      newStat = generate_stat_from_ref_list_HRE(lRef, newStat);
    }
  else
    {
      newStat = generate_stat_from_ref_list_proc(lRef, NIL, newStat);

      fill_gLoopToOpt(lRef);
    }

  return newStat;
}

/*
This function generates the MMCD code or the HRE code from any
statements.
 */
statement generate_code_function(statement stat, bool bCalledFromSeq)
{
  statement newStat = statement_undefined;

  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_sequence:
      {
	newStat = generate_code_seq(stat);

	break;
      }
    case is_instruction_loop:
      {
	newStat = generate_code_loop(stat);
	break;
      }
    case is_instruction_call:
      {
	newStat = generate_code_call(stat, bCalledFromSeq);
	break;
      }
    case is_instruction_test:
      {
	if(gGenHRE)
	  {
	    newStat = generate_code_test_HRE(stat);
	  }
	else
	  {
	    newStat = generate_code_test_proc(stat);
	  }
	break;
      }
    default:
      {
	pips_assert("FALSE", false);
	break;
      }
    }

  return newStat;
}

/*
This function generates the MMCDs generation code and
the HRE code if we want to have only one process on the HRE
 */
statement comEngine_generate_code(statement stat)
{
  statement newStat = statement_undefined;

  // Initialize some global variables
  COM_OPT = true;
  gExecPut = false;
  gLoopToOuterInd = hash_table_make(hash_pointer, 0);
  gHtCurNewInd = hash_table_make(hash_pointer, 0);
  glCurLoop = NIL;
  glCurRep = NIL;
  gLoopToOpt = hash_table_make(hash_pointer, 0);
  glOutEffStats = NIL;
  glToggleInitStats = NIL;

  // Do the job
  newStat = generate_code_function(stat, false);

  if(statement_block_p(stat))
    {
      newStat = process_code_seq(newStat, stat);
    }

  // Add the loop indices statements
  if(glOutEffStats != NIL)
    {
      newStat =
	make_block_statement(gen_nconc(CONS(STATEMENT, newStat, NIL),
				       glOutEffStats));
    }

  // Add the toggle initialization statements
  if(glToggleInitStats != NIL)
    {
      newStat =
	make_block_statement(gen_nconc(glToggleInitStats,
				       CONS(STATEMENT, newStat, NIL)));
    }

  // Free some global variables
  hash_table_free(gHtCurNewInd);
  hash_table_free(gLoopToOpt);
  hash_table_free(gLoopToOuterInd);

  return newStat;
}

