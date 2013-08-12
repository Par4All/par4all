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
This file contains the functions to generate the HRE code using several
processes on the HRE
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

// gOldIndToNewInd associated a loop index of
// input code to a loop index of the new one
static hash_table gOldIndToNewInd;

// gIsIndex is used to store the fact that 
// an index has to be incremented
static hash_table gIsIndex;

static list gCurStats = NIL;
static list glCurRep = NIL;

// These are global variable to the names we want to
// give to the created modules
static const char* g_new_module_name = NULL;
static const char* g_module_name = NULL;

// This variable holds the number of
// if we have entered at a given point of
// the algorithm
static int gIfCount = 0;

// This list holds the enclosing loops at a given point of 
// the algorithm
static list glCurLoop = NIL;

static entity gNewInd = entity_undefined;

// These lists hold the read or write statements that 
// make to read values from the fifos
static list glReadStats = NIL;
static list glWriteStats = NIL;

static statement HRE_distribute_stat(statement stat, bool calledFromLoop);

/*
This function creates the private variables of the new module. Then,
it store the association between the created variable and the old reference
holded by lRef ingRefToHREVar
 */
static void generate_scalar_variables_from_list(list lRef)
{
  MAP(REFERENCE, curRef,
  {
    bool alreadyDone = false;
    reference refFound = reference_undefined;

    HASH_MAP(ref1, var1,
    {
      if(reference_equal_p(curRef, ref1))
	{
	  alreadyDone = true;
	  refFound = ref1;
	  break;
	}
    }, gRefToHREVar);

    if(!alreadyDone)
      {
	//printf("it\n");
	//print_reference(curRef);printf("\n");

	string name =
	  strdup(entity_local_name(reference_variable((reference)curRef)));

	basic bas = basic_of_reference((reference)curRef);

	pips_assert("bas != basic_undefined", bas != basic_undefined);

	//entity new_ent = make_new_scalar_entity(name, copy_basic(bas));

	entity new_ent = make_new_scalar_variable_with_prefix(name,
							      get_current_module_entity(),
							      bas);
    AddEntityToCurrentModule(new_ent);

	//printf("%s\n", entity_user_name(new_ent));

	if(hash_get(gRefToHREVar, curRef) == HASH_UNDEFINED_VALUE)
	  {
	    hash_put(gRefToHREVar, curRef, new_ent);
	  }
      }
    else
      {
	//printf("it found\n");
	//print_reference(curRef);printf("\n");

	entity scalEnt = hash_get(gRefToHREVar, refFound);

	if(hash_get(gRefToHREVar, curRef) == HASH_UNDEFINED_VALUE)
	  {
	    hash_put(gRefToHREVar, curRef, scalEnt);
	  }
      }

  }, lRef);

}

/*
This function creates a private entity whose prefix is the local name
of ind
 */
static entity find_or_create_newInd(entity ind, bool bIsInd)
{
  entity new_ent = entity_undefined;

  // If the entity was already created, then just return the existing
  // entity
  new_ent = hash_get(gOldIndToNewInd, ind);

  if(new_ent != HASH_UNDEFINED_VALUE)
    {
      return new_ent;
    }

  // Create a new entity
  new_ent =
    make_new_scalar_variable_with_prefix(entity_local_name(ind),
					 get_current_module_entity(),
					 copy_basic(entity_basic(ind)));
  AddEntityToCurrentModule(new_ent);

  statement loopStat = STATEMENT(CAR(glCurLoop));

  list lUnSupportedRef = hash_get(gLoopToUnSupRef, loopStat);

  pips_assert("lUnSupportedRef != HASH_UNDEFINED_VALUE",
	      lUnSupportedRef != HASH_UNDEFINED_VALUE);

  // If the entity created is actually the index of 
  // the new loop, then do gNewInd = new_ent
  if((loop_index(statement_loop(loopStat)) == ind) &&
     (lUnSupportedRef == NIL))
    {
      gNewInd = new_ent;
    }

  // Memorize the creation of the entity
  hash_put(gOldIndToNewInd, ind, new_ent);

  // If the entity is an index, then memorize it with gIsIndex
  if(bIsInd)
    {
      hash_put(gIsIndex, new_ent, (void *)true);
    }

  return new_ent;
}

/*
Get the expression that defines the index of the fifo that has to be
read or written
 */
static expression get_indExp_from_ref(reference curRef, hash_table ht,
				      bool * innerInd)
{
  entity ind = entity_undefined;

  HASH_MAP(ref1, ind1,
  {
    if(reference_equal_p(curRef, ref1))
      {
	ind = ind1;
	break;
      }
  }, ht);

  expression indExp;

  // If the index ind is undefined then let read from the 
  // beginning of the fifo
  if(ind == entity_undefined)
    {
      indExp = int_to_expression(0);
    }
  else
    {
      if(loop_index(statement_loop(STATEMENT(CAR(glCurLoop)))) == ind)
	{
	  *innerInd = true;
	}

      entity new_ent = find_or_create_newInd(ind, true);

      indExp = entity_to_expression(new_ent);
    }

  return indExp;
}

/*
Get the expression that defines the fifo that has to be
read or written
 */
expression get_fifoExp_from_ref(reference curRef, expression buffExp,
				hash_table ht)
{
  entity ind = entity_undefined;

  HASH_MAP(ref1, ind1,
  {
    if(reference_equal_p(curRef, ref1))
      {
	ind = ind1;
	break;
      }
  }, ht);

  expression fifoExp;

  if(ind == entity_undefined)
    {
      fifoExp = buffExp;
    }
  else
    {
      entity new_ent = find_or_create_newInd(ind, false);

      list addArg = gen_make_list(expression_domain,
				  buffExp,
				  entity_to_expression(new_ent),
				  NULL);

      fifoExp =
	call_to_expression(make_call(entity_intrinsic(PLUS_OPERATOR_NAME),
				     addArg));
    }

  return fifoExp;
}

static statement make_read_write_fifo_stat(string name,
					   expression fifoExp,
					   expression indExp,
					   expression hreBuffExp)
{
  statement newStat;

  if(!strcmp(name, READ_FIFO))
    {
      list arg = gen_make_list(expression_domain,
			       fifoExp,
			       indExp,
			       NULL);

      expression rExp = call_to_expression(make_call(module_name_to_runtime_entity(name), 
					    arg));

      newStat = make_assign_statement(hreBuffExp, rExp);
    }
  else
    {
      list arg = gen_make_list(expression_domain,
			       fifoExp,
			       indExp,
			       hreBuffExp,
			       NULL);

      newStat = call_to_statement(
				  make_call(module_name_to_runtime_entity(name), 
					    arg));
    }

  return newStat;
}

/*
This function generates the statement that read or write the value
(corresponding to the old reference curRef) in the fifo
*/
statement generate_fifo_stat2(reference curRef, bool bRead)
{
  statement newStat = statement_undefined;

  string name = NULL;

  expression buffExp = get_fifo_from_ref(curRef);

  //printf("generate_fifo_stat2\n");
  //print_reference(curRef);printf("\n");

  if(buffExp == expression_undefined)
    {
      return statement_undefined;
    }

  bool innerInd = false;

  expression indExp = get_indExp_from_ref(curRef, gRefToInd, &innerInd);

  expression fifoExp = get_fifoExp_from_ref(curRef, buffExp, gRefToToggle);

  entity hreBuffEnt = get_HRE_buff_ent_from_ref(curRef);

  pips_assert("hreBuffEnt != entity_undefined",
	      hreBuffEnt != entity_undefined);

  if(bRead)
    {
      name = strdup(READ_FIFO);
    }
  else
    {
      name = strdup(WRITE_FIFO);
    }

  newStat = make_read_write_fifo_stat(name, fifoExp, indExp,
				      entity_to_expression(hreBuffEnt));

  if(!innerInd)
    {
      if(bRead)
	{
	  glReadStats = CONS(STATEMENT, newStat, glReadStats);
	}
      else
	{
	  glWriteStats = CONS(STATEMENT, newStat, glWriteStats);
	}

      return statement_undefined;
    }

  return newStat;
}

/*
This function generates the statement that read or write the value
(corresponding to the old entity oldInd) in the fifo
*/
statement generate_ind_fifo_stat2(entity oldInd, entity newInd, bool bRead)
{
  statement newStat = statement_undefined;

  string name = NULL;

  intptr_t indNum = (intptr_t)hash_get(gIndToNum, oldInd);

  pips_assert("indNum != HASH_UNDEFINED_VALUE",
	      indNum != (intptr_t)HASH_UNDEFINED_VALUE);

  if(bRead)
    {
      name = strdup(READ_FIFO);
    }
  else
    {
      name = strdup(WRITE_FIFO);
    }

  intptr_t fifoNum = (intptr_t)hash_get(gEntToHREFifo, oldInd);

  pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
	      fifoNum != (intptr_t)HASH_UNDEFINED_VALUE);

  newStat =
    make_read_write_fifo_stat(name,
			      int_to_expression(fifoNum),
			      int_to_expression(indNum),
			      entity_to_expression(newInd));
  return newStat;
}

/*
This function generates the read or write statements
 */
void generate_fifo_stats2(list lRef,
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

	    statement readStat = generate_fifo_stat2(curRef, true);

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

	    statement writeStat = generate_fifo_stat2(curRef, false);

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
This function replaces the old references with new references
 */
static void replace_array_ref_with_fifos2(list lRef, statement * newStat)
{
  list lStats = NIL;

  list lReadStats = NIL;
  list lWriteStats = NIL;
printf("replace_array_ref_with_fifos2\n");
  generate_fifo_stats2(lRef, &lReadStats, &lWriteStats);
printf("replace_array_ref_with_fifos2 1\n");
  FOREACH(REFERENCE, curRef,lRef)
  {
    printf("replace_array_ref %p\n", curRef);
    print_reference(curRef);printf("\n");

    entity hreBuffEnt = get_HRE_buff_ent_from_ref(curRef);

    pips_assert("hreBuffEnt != entity_undefined",
		hreBuffEnt != entity_undefined);

    reference hreBuffRef = make_reference(hreBuffEnt, NIL);

    comEngine_replace_reference_in_stat(*newStat, curRef, reference_to_expression(hreBuffRef));

  }

  lStats = lReadStats;
  if(*newStat != statement_undefined)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, *newStat, NIL));
    }
  lStats = gen_nconc(lStats, lWriteStats);

  *newStat = make_block_statement(lStats);
}

/*
This function adds the read and write statements to the module statements
 */
static statement add_index_statements(statement stat)
{
  list lReadStats = NIL;
  //list lWriteStats = NIL;

  HASH_MAP(oldInd, newInd,
  {
    if(newInd == gNewInd)
      {
	continue;
      }

    statement readStat = generate_ind_fifo_stat2(oldInd, newInd, true);

    lReadStats = CONS(STATEMENT, readStat, lReadStats);

    //statement writeStat = generate_ind_fifo_stat2(oldInd, newInd, false);

    //lWriteStats = CONS(STATEMENT, writeStat, lWriteStats);

  }, gOldIndToNewInd);

  list lStats = NIL;

  lStats = gen_nconc(lStats, lReadStats);
  lStats = gen_nconc(lStats, glReadStats);
  lStats = gen_nconc(lStats, CONS(STATEMENT, stat, NIL));
  lStats = gen_nconc(lStats, glWriteStats);
  //lStats = gen_nconc(lStats, lWriteStats);

  return make_block_statement(lStats);
}

/*
This function generate the statement of a new module
 */
static statement generate_code()
{
  if(gCurStats == NIL)
    {
      return make_block_statement(NIL);
    }

  printf("generate_code\n");

  generate_scalar_variables_from_list(glCurRep);

  /*printf("glCurRep\n");
  MAP(REFERENCE, curRef,
  {
    printf("glCurRep it\n");
    print_reference(curRef);printf("\n");
  }, glCurRep);
*/
  statement newStat = make_block_statement(gCurStats);

  printf("newStat bef\n");
  print_statement(newStat);

  replace_array_ref_with_fifos2(glCurRep, &newStat);

  printf("newStat aft 1\n");
  print_statement(newStat);

  if(gNewInd != entity_undefined)
    {
      loop curLoop = statement_loop(STATEMENT(CAR(glCurLoop)));

      loop newLoop = make_loop(gNewInd,
			       make_range(int_to_expression(0),
					  copy_expression(gBufferSizeEnt),
					  int_to_expression(1)),
			       newStat,
			       loop_label(curLoop),
			       make_execution(is_execution_sequential, UU),
			       NIL);

      newStat = make_statement(entity_empty_label(),
			       STATEMENT_NUMBER_UNDEFINED,
			       STATEMENT_ORDERING_UNDEFINED,
			       empty_comments,
			       make_instruction_loop(newLoop),NIL,NULL,
			       empty_extensions (), make_synchronization_none());
    }

  newStat = add_index_statements(newStat);

  // Reinitialize global variables
  glReadStats = NIL;
  glWriteStats = NIL;

  gen_free_list(glCurRep);

  gCurStats = NIL;
  glCurRep = NIL;

  hash_table_free(gOldIndToNewInd);
  gOldIndToNewInd = hash_table_make(hash_pointer, 0);
  gNewInd = entity_undefined;

  hash_table_free(gRefToHREVar);
  gRefToHREVar = hash_table_make(hash_pointer, 0);

  return newStat;
}

/*
this function makes an empty module
 */
static void make_HRE_empty_module()
{
  static int number = 0;

  string num  = i2a(number++);
  string prefix = strdup(concatenate(g_new_module_name,"_",
				     num,
				     (char *) NULL));
  free(num);

  entity new_fun = make_empty_subroutine(prefix,make_language_unknown());

  reset_current_module_entity();

  set_current_module_entity(new_fun);
}

/*
This function assigns statement stat to the current empty module
 */
static void fill_HRE_module(statement stat)
{
  create_HRE_module(entity_local_name(get_current_module_entity()),
		    g_module_name, stat, get_current_module_entity());
}

/*
This function generates a new HRE module if (gCurStats != NIL)
 */
static void loop_enter()
{
  if(gCurStats == NIL)
    {
      return;
    }

  make_HRE_empty_module();

  statement newStat = generate_code();

  fill_HRE_module(newStat);
}

#if 0
static void create_loop_HRE_module()
{
  if(glCurLoop == NIL)
    {
      return;
    }

  make_HRE_empty_module();

  list lStats = NIL;
  list lReadStats = NIL;
  list lIncStats = NIL;
  list lWriteStats = NIL;

  MAP(STATEMENT, loopStat,
  {
    entity oldInd = loop_index(statement_loop(loopStat));

    entity newInd =
      make_new_scalar_variable_with_prefix(entity_local_name(oldInd),
					   get_current_module_entity(),
					   copy_basic(entity_basic(oldInd)));
      AddEntityToCurrentModule(newInd);

    statement readStat = generate_ind_fifo_stat2(oldInd, newInd, true);

    lReadStats = CONS(STATEMENT, readStat, lReadStats);

    list addArg = gen_make_list(expression_domain,
				entity_to_expression(newInd),
				int_to_expression(1),
				NULL);

    expression rExp =
      call_to_expression(make_call(get_function_entity(PLUS_OPERATOR_NAME),
				   addArg));

    statement incStat = make_assign_statement(entity_to_expression(newInd), rExp);

    lIncStats = CONS(STATEMENT, incStat, lIncStats);

    statement writeStat = generate_ind_fifo_stat2(oldInd, newInd, false);

    lWriteStats = CONS(STATEMENT, writeStat, lWriteStats);

  }, glCurLoop);

  lStats = gen_nconc(lStats, lReadStats);
  lStats = gen_nconc(lStats, lIncStats);
  lStats = gen_nconc(lStats, lWriteStats);

  statement newStat = make_block_statement(lStats);

  fill_HRE_module(newStat);
}
#endif

/*
This function processes the HRE code generation for a
loop statement
 */
static statement HRE_distribute_loop(statement stat)
{
  //printf("HRE_distribute_loop beg\n");

  loop_enter();

  glCurLoop = CONS(STATEMENT, stat, glCurLoop);

  HRE_distribute_stat(loop_body(statement_loop(stat)), true);

  loop_enter();

  gen_remove(&glCurLoop, stat);

  //create_loop_HRE_module();

  return statement_undefined;
}

/*
This function processes the HRE code generation for a
call statement
 */
static statement HRE_distribute_call(statement stat)
{
  statement newStat = statement_undefined;

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

  if(gIfCount == 0)
    {
      newStat = copy_statement(stat);

      statement_comments(newStat) = string_undefined;
    }

  return newStat;
}

/*
This function processes the HRE code generation for a
block statement
 */
static statement HRE_distribute_seq(statement stat)
{
  instruction instr = statement_instruction(stat);

  //printf("HRE_distribute_seq\n");
  MAP(STATEMENT, curStat,
  {
    statement seqStat = HRE_distribute_stat(curStat, false);

    //printf("seqStat\n");
    if(seqStat == statement_undefined)
      {
	//printf("undefined\n");
      }
    else
      {
	//print_statement(seqStat);
	if(gIfCount == 0)
	  {
	    gCurStats = gen_nconc(gCurStats, CONS(STATEMENT, seqStat, NIL));
	  }
      }

  }, sequence_statements(instruction_sequence(instr)));

  return statement_undefined;
}

/*
This function processes the HRE code generation for a
test statement
 */
static statement HRE_distribute_test(statement stat)
{
  statement newStat = statement_undefined;

  list lCond = NIL;
  lCond =
    comEngine_expression_to_reference_list(test_condition(statement_test(stat)),
					   lCond);

  glCurRep = gen_nconc(glCurRep, lCond);

  gIfCount++;

  // Generate the HRE code for the true statement
  HRE_distribute_stat(test_true(statement_test(stat)), true);

  // Generate the HRE code for the false statement
  HRE_distribute_stat(test_false(statement_test(stat)), true);

  gIfCount--;

  if(gIfCount == 0)
    {
      newStat = copy_statement(stat);

      statement_comments(newStat) = string_undefined;
    }

  return newStat;
}

/*
This function processes the HRE code generation for any
statements
 */
static statement HRE_distribute_stat(statement stat, bool calledFromLoop)
{
  statement newStat = statement_undefined;

  printf("HRE_distribute_stat\n");
  print_statement(stat);

  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_sequence:
      {
	newStat = HRE_distribute_seq(stat);

	break;
      }
    case is_instruction_loop:
      {
	newStat = HRE_distribute_loop(stat);
	break;
      }
    case is_instruction_call:
      {
	newStat = HRE_distribute_call(stat);

	if(calledFromLoop && (gIfCount == 0))
	  {
	    gCurStats = gen_nconc(gCurStats, CONS(STATEMENT, newStat, NIL));
	  }

	break;
      }
    case is_instruction_test:
      {
	newStat = HRE_distribute_test(stat);

	if(calledFromLoop && (gIfCount == 0))
	  {
	    gCurStats = gen_nconc(gCurStats, CONS(STATEMENT, newStat, NIL));
	  }

	break;
      }
    default:
      {
	pips_assert("FALSE", false);
	break;
      }
    }
  printf("HRE_distribute_stat end\n");
  return newStat;
}

/*
This function generates the HRE code using several
processes on the HRE
 */
statement HRE_distribute(statement stat, string new_module_name, const char* module_name)
{
  printf("stat bef HRE_distribute\n");
  print_statement(stat);

  // Global variables initialization
  g_new_module_name = new_module_name;
  g_module_name = module_name;
  gCurStats = NIL;
  glCurRep = NIL;
  gIfCount = 0;
  glCurLoop = NIL;
  gNewInd = entity_undefined;
  gOldIndToNewInd = hash_table_make(hash_pointer, 0);
  gIsIndex = hash_table_make(hash_pointer, 0);
  glReadStats = NIL;
  glWriteStats = NIL;

  HRE_distribute_stat(stat, true);

  loop_enter();

  // Free some global variables
  hash_table_free(gOldIndToNewInd);
  hash_table_free(gIsIndex);

  return make_block_statement(NIL);
}
