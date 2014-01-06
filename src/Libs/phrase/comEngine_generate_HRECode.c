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
/*
This file contains functions to generate the HRE code if we want to have only 
one process on the HRE
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
#include "comEngine_generate_code.h"
#include "phrase.h"

// This hash_table associates an old reference to private
// HRE variable
hash_table gRefToHREVar;

/*
This function returns an constant expression whose value is the fifo
number associated to reference ref.
 */
expression get_fifo_from_ref(reference ref)
{
  expression retExp = expression_undefined;

  intptr_t fifoNum = (intptr_t)hash_get(gRefToHREFifo, ref);

  if(fifoNum != (intptr_t)HASH_UNDEFINED_VALUE)
    {
      retExp = int_to_expression(fifoNum);
    }

  return retExp;
}

/*
This function returns the private HRE entity associated to the reference ref.
 */
entity get_HRE_buff_ent_from_ref(reference ref)
{
  entity buffEnt = entity_undefined;

  HASH_MAP(curRef, hreBuff,
  {
    if(reference_equal_p(curRef, ref))
      {
	buffEnt = hreBuff;
	break;
      }

  }, gRefToHREVar);

  return buffEnt;
}

/*
This function creates a "wait for next step" statement that allows the HRE
to make sure that all the transfers that had to be done during the current 
phase are finished.
 */
statement make_wait_step_statement()
{
  statement newStat = statement_undefined;

  newStat = call_to_statement(
			      make_call(module_name_to_runtime_entity(WAIT_FOR_NEXT_STEP),
					NIL));

  return newStat;
}

/*
This function generate a statement to read or write a fifo
associated to curRef in the HRE
 */
static statement generate_fifo_stat(reference curRef, expression buffIndExp,
				    entity ind, bool bRead)
{
  statement newStat = statement_undefined;

  string name = NULL;
  list arg = NIL;

  expression buffExp = get_fifo_from_ref(curRef);

  if(buffExp == expression_undefined)
    {
      return statement_undefined;
    }

  if(ind != entity_undefined)
    {
      list addArg = gen_make_list(expression_domain,
				  buffExp,
				  entity_to_expression(ind),
				  NULL);

      buffExp = call_to_expression(make_call(entity_intrinsic(PLUS_OPERATOR_NAME),
					     addArg));
    }

  entity hreBuffEnt = get_HRE_buff_ent_from_ref(curRef);

  pips_assert("hreBuffEnt != entity_undefined",
	      hreBuffEnt != entity_undefined);

  expression realInd;

  if(bRead)
    {
      name = strdup(GEN_GET_BUFF);
    }
  else
    {
      name = strdup(GEN_WRITE_BUFF);
    }

  if(buffIndExp == expression_undefined)
    {
      realInd = int_to_expression(0);
    }
  else
    {
      realInd = copy_expression(buffIndExp);
    }

  arg = gen_make_list(expression_domain,
		      buffExp,
		      realInd,
		      entity_to_expression(hreBuffEnt),
		      NULL);

  newStat = call_to_statement(
			      make_call(module_name_to_runtime_entity(name), 
					arg));

  return newStat;
}

/*
This function generate statements to read or write a fifo
associated to the references of list lRef.
 */
static void generate_fifo_stats(list lRef, expression buffIndExp, entity ind,
				list * lReadStats, list * lWriteStats)
{
  list lReadDone = NIL;
  list lWriteDone = NIL;

  MAP(REFERENCE, curRef,
  {
    string effAction = hash_get(gRefToEff, curRef);

    pips_assert("effAction != HASH_UNDEFINED_VALUE",
		effAction != HASH_UNDEFINED_VALUE);

    //printf("fifo ref %d\n", (int)curRef);
    //print_reference(curRef);printf("\n");

    if(!strcmp(effAction, R_EFFECT))
      {
	bool alreadyDone = false;

	MAP(REFERENCE, doneRef,
	{
	  if(reference_equal_p(curRef, doneRef))
	    {
	      alreadyDone = true;
	      break;
	    }
	}, lReadDone);

	if(!alreadyDone)
	  {
	    //printf("read eff\n");

	    lReadDone = gen_nconc(lReadDone, CONS(REFERENCE, curRef, NIL));

	    statement readStat = generate_fifo_stat(curRef, buffIndExp,
						    ind, true);

	    if(readStat == statement_undefined)
	      {
		continue;
	      }

	    *lReadStats = gen_nconc(*lReadStats, CONS(STATEMENT, readStat, NIL));
	    //print_statement(readStat);
	  }
      }
    else
      {
	bool alreadyDone = false;

	MAP(REFERENCE, doneRef,
	{
	  if(reference_equal_p(curRef, doneRef))
	    {
	      alreadyDone = true;
	      break;
	    }
	}, lWriteDone);

	if(!alreadyDone)
	  {
	    //printf("write eff\n");

	    lWriteDone = gen_nconc(lWriteDone, CONS(REFERENCE, curRef, NIL));

	    statement writeStat = generate_fifo_stat(curRef, buffIndExp,
						     ind, false);

	    if(writeStat == statement_undefined)
	      {
		continue;
	      }

	    *lWriteStats = gen_nconc(*lWriteStats, CONS(STATEMENT, writeStat, NIL));

	    //print_statement(writeStat);
	  }
      }

  }, lRef);

  gen_free_list(lReadDone);
  gen_free_list(lWriteDone);
}

/*
This functions replaces the old references of lRef with the HRE private
variables
 */
void replace_array_ref_with_fifos(list lRef, expression buffIndExp,
				  entity ind, statement * newStat)
{
  list lStats = NIL;

  list lReadStats = NIL;
  list lWriteStats = NIL;

  // Generate the read or write fifo statements
  generate_fifo_stats(lRef, buffIndExp, ind, &lReadStats, &lWriteStats);

  // Replace the references by the private variables
  MAP(REFERENCE, curRef,
  {
    //printf("replace_array_ref %d\n", (int)curRef);
    //print_reference(curRef);printf("\n");

    entity hreBuffEnt = get_HRE_buff_ent_from_ref(curRef);

    pips_assert("hreBuffEnt != entity_undefined",
		hreBuffEnt != entity_undefined);

    reference hreBuffRef = make_reference(hreBuffEnt, NIL);

    comEngine_replace_reference_in_stat(*newStat, curRef, reference_to_expression(hreBuffRef));

  }, lRef);

  lStats = lReadStats;
  if(*newStat != statement_undefined)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, *newStat, NIL));
    }
  lStats = gen_nconc(lStats, lWriteStats);

  *newStat = make_block_statement(lStats);
}

/*
This function generates the scalar variables needed in the HRE
 */
static void generate_scalar_variables()
{
  list lDone = NIL;

  HASH_MAP(curRef, curBuff,
  {
    bool alreadyDone = false;
    reference refFound = reference_undefined;

    MAP(REFERENCE, ref1,
    {
      if(reference_equal_p(curRef, ref1))
	{
	  alreadyDone = true;
	  refFound = ref1;
	  break;
	}

    }, lDone);

    if(!alreadyDone)
      {
	//printf("it\n");
	//print_reference(curRef);printf("\n");

	string name = strdup(entity_local_name(reference_variable((reference)curRef)));

	basic bas = basic_of_reference((reference)curRef);

	pips_assert("bas != basic_undefined", bas != basic_undefined);

	entity new_ent = make_new_scalar_variable_with_prefix(name,
							      get_current_module_entity(),
							      bas);
    AddEntityToCurrentModule(new_ent);

	hash_put(gRefToHREVar, curRef, new_ent);

	lDone = CONS(REFERENCE, curRef, lDone);
      }
    else
      {
	entity scalEnt = hash_get(gRefToHREVar, refFound);

	hash_put(gRefToHREVar, curRef, scalEnt);
      }

  }, gRefToHREFifo);

  gen_free_list(lDone);
}

/*
This function add the wait statements to the statement stat
 */
statement generate_stat_from_ref_list_HRE(list lRef, statement stat)
{
  bool writeFound = false;

  MAP(REFERENCE, curRef,
  {
    string effAction = hash_get(gRefToEff, curRef);

    pips_assert("effAction != HASH_UNDEFINED_VALUE",
		effAction != HASH_UNDEFINED_VALUE);

    if(!strcmp(effAction, W_EFFECT))
      {
	writeFound = true;
	break;
      }

  }, lRef);

  statement stepStat = make_wait_step_statement();

  list lStats = NIL;

  lStats = gen_nconc(lStats, CONS(STATEMENT, copy_statement(stepStat), NIL));
  if(stat != statement_undefined)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, stat, NIL));
    }

  if(writeFound)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, copy_statement(stepStat), NIL));
      lStats = gen_nconc(lStats, CONS(STATEMENT, copy_statement(stepStat), NIL));
    }

  free_statement(stepStat);

  return make_block_statement(lStats);
}

/*
This function add the wait statements to the statement stat
 */
list generate_stat_from_ref_list_HRE_list(list lRef, list lInStats)
{
  bool writeFound = false;

  MAP(REFERENCE, curRef,
  {
    string effAction = hash_get(gRefToEff, curRef);

    pips_assert("effAction != HASH_UNDEFINED_VALUE",
		effAction != HASH_UNDEFINED_VALUE);

    if(!strcmp(effAction, W_EFFECT))
      {
	writeFound = true;
	break;
      }

  }, lRef);

  statement stepStat = make_wait_step_statement();

  list lStats = NIL;

  lStats = gen_nconc(lStats, CONS(STATEMENT, copy_statement(stepStat), NIL));

  lStats = gen_nconc(lStats, lInStats);

  if(writeFound)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, copy_statement(stepStat), NIL));
      lStats = gen_nconc(lStats, CONS(STATEMENT, copy_statement(stepStat), NIL));
    }

  free_statement(stepStat);

  return lStats;
}

/*
This function returns the supported and unsupported references associated to
statement stat
 */
void get_supportedRef_HRE(statement stat,
			  list * lSupportedRef, list * lUnSupportedRef)
{
  *lSupportedRef = hash_get(gLoopToSupRef, stat);
  *lUnSupportedRef = hash_get(gLoopToUnSupRef, stat);

  pips_assert("*lSupportedRef != HASH_UNDEFINED_VALUE",
	      *lSupportedRef != HASH_UNDEFINED_VALUE);

  pips_assert("*lUnSupportedRef != HASH_UNDEFINED_VALUE",
	      *lUnSupportedRef != HASH_UNDEFINED_VALUE);
}

/*
This function replaces the old references in lRef by the HRE private references
 */
void process_replacement_HRE(list lRef, expression buffIndExp,
			     statement * stat)
{
  replace_array_ref_with_fifos(lRef, buffIndExp, entity_undefined, stat);
}

/*
This function replaces the old references in lRef by the HRE private references
when lRef is lOutRef
 */
list process_replacement_HRE_OutRef(list lRef, list lStats)
{
  statement stat = STATEMENT(CAR(lStats));

  list savedList = CDR(lStats);

  CDR(lStats) = NIL;
  gen_free_list(lStats);

  replace_array_ref_with_fifos(lRef, expression_undefined,
			       entity_undefined, &stat);

  lStats = gen_nconc(CONS(STATEMENT, stat, NIL), savedList);

  return lStats;
}

/*
This function copies statement stat
 */
statement get_call_stat_HRE(statement stat)
{
  statement newStat = copy_statement(stat);

  statement_comments(newStat) = string_undefined;

  return newStat;
}

/*
This function adds some statements before innerStat
 */
list make_loop_lStats_HRE(statement stat, entity transferSize,
			  statement innerStat, entity newOuterInd,
			  list lSupportedRef, hash_table htOffset,
			  expression bufferSizeExp)
{
  lSupportedRef = lSupportedRef;
  htOffset = htOffset;

  list lStats = NIL;

  statement transStat = make_transStat(stat, newOuterInd,
				       transferSize, bufferSizeExp);

  lStats = gen_nconc(lStats, CONS(STATEMENT, transStat, NIL));

  lStats = gen_nconc(lStats, CONS(STATEMENT, innerStat, NIL));

  return lStats;
}

/*
This function generates the HRE code from a test
statement.
 */
statement generate_code_test_HRE(statement stat)
{
  //printf("generate_code_test\n");
  //print_statement(stat);printf("\n");
  statement newStat = statement_undefined;

  newStat = get_call_stat_HRE(stat);

  test newTest = statement_test(newStat);

  if(test_true(newTest) != statement_undefined)
    {
      free_statement(test_true(newTest));

      test_true(newTest) = make_empty_statement();
    }

  if(test_false(newTest) != statement_undefined)
    {
      free_statement(test_false(newTest));

      test_false(newTest) = make_empty_statement();
    }

  list lRef = hash_get(gStatToRef, stat);

  if(lRef != HASH_UNDEFINED_VALUE)
    {
      process_replacement_HRE(lRef, expression_undefined, &newStat);

      newStat = generate_stat_from_ref_list_HRE(lRef, newStat);
    }

  // Get the new statements for the true statement
  statement trueStat = generate_code_function(test_true(statement_test(stat)), false);

  // Get the new statements for the false statement
  statement falseStat = generate_code_function(test_false(statement_test(stat)), false);

  free_statement(test_true(newTest));
  free_statement(test_false(newTest));

  if(trueStat != statement_undefined)
    {
      test_true(newTest) = trueStat;
    }
  else
    {
      test_true(newTest) = make_empty_statement();
    }

  if(falseStat != statement_undefined)
    {
      test_false(newTest) = falseStat;
    }
  else
    {
      test_false(newTest) = make_empty_statement();
    }

  return newStat;
}

/*
This function add some bubbles in the pipeline if needed
 */
list process_gLoopToSync_HRE(statement stat, list lInStats)
{
  bool loopSync = (intptr_t)hash_get(gLoopToSync, stat);

  if(loopSync == (intptr_t)HASH_UNDEFINED_VALUE)
    {
      return lInStats;
    }

  statement stepStat = make_wait_step_statement();

  list lStats = NIL;

  lStats = gen_nconc(lStats, lInStats);
  lStats = gen_nconc(lStats, CONS(STATEMENT, stepStat, NIL));
  lStats = gen_nconc(lStats, CONS(STATEMENT, copy_statement(stepStat), NIL));

  return lStats;
}

/*
This function generates the HRE code
 */
statement comEngine_generate_HRECode(statement externalized_code,
				     string new_module_name,
				     list l_in, list l_out, list l_params, list l_priv,
				     const char* module_name, int hreMemSize)
{
  statement newStat;

  l_in = l_in;
  l_out = l_out;
  l_params = l_params;
  l_priv = l_priv;
  hreMemSize = hreMemSize;

  gRefToHREVar = hash_table_make(hash_pointer, 0);

  if(get_bool_property("COMENGINE_CONTROL_IN_HRE"))
    {
      entity new_fun = make_empty_subroutine(new_module_name,make_language_unknown());

      reset_current_module_entity();

      set_current_module_entity(new_fun);

      // Generate the HRE private variables
      generate_scalar_variables();

      // This means the HRE code is going to be generated
      gGenHRE = true;

      //This function generates the HRE code if we want to have only 
      //one process on the HRE
      newStat = comEngine_generate_code(externalized_code);

      // Create the new HRE module
      create_HRE_module(new_module_name,
			module_name, newStat, new_fun);
    }
  else
    {
      // Generate the HRE code if several HRE processes can be created
      newStat = HRE_distribute(externalized_code, new_module_name, module_name);
    }

  //printf("HRE\n");
  //print_statement(newStat);

  hash_table_free(gRefToHREVar);

  return newStat;
}
