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
This file contains functions used to generate the MMCDs generation code
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

// This hash_table is used to store the fifo
// used at a given point of the algorithm
static hash_table gRealFifo;

// This is the variable that holds
// the current step value
static entity gStepEnt;

// This hash_table associates a reference to its
// old fifo value. The old fifo value was associated to the 
// reference during the analysis phase.
static hash_table gOldRefToHREFifo;

static list make_mmcd_stats_from_ref(statement stat, list lSupportedRef, hash_table htOffset,
				     entity transferSize, entity newOuterInd, statement innerStat);

/*
This function return true if the reference whose indices are lRefDim is
supported and, in this case, the value of the offset is put in retVal.
 */
static bool get_final_offset(list lRefDim, intptr_t offset, int rank, intptr_t * retVal)
{
  bool fort_org = get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION");

  int i = 0;
  int ind;

  intptr_t finalOffset = 1;

  for(i = 0; i < gen_length(lRefDim); i++)
    {
      if(fort_org)
	{
	  ind = i;
	}
      else
	{
	  ind = gen_length(lRefDim) - i - 1;
	}

      if(ind == rank)
	{
	  break;
	}

      dimension curDim = DIMENSION(gen_nth(ind, lRefDim));

      expression lowerExp = dimension_lower(curDim);
      expression upperExp = dimension_upper(curDim);

      if(!integer_constant_expression_p(lowerExp) ||
	 !integer_constant_expression_p(upperExp))
	{
	  return false;
	}

      int lower = integer_constant_expression_value(lowerExp);
      int upper = integer_constant_expression_value(upperExp);

      pips_assert("lower < upper", lower < upper);

      finalOffset = finalOffset * (upper - lower + 1);
    }

  *retVal = finalOffset * offset;

  return true;
}

/*
This function return true if reference ref is
supported and, in this case, it updates the hash_table htOffset.
 */
static bool supported_ref_p(reference ref, entity index, hash_table htOffset)
{
  bool success = true;
  int offset = 0;
  int rank = 0;

  //printf("supported_ref_p \n");
  //print_reference(ref);printf("\n");

  MAP(EXPRESSION, ind,
  {
    normalized norm = NORMALIZE_EXPRESSION(ind);

    list indRefList = NIL;

    if(normalized_linear_p(norm))
      {
	Pvecteur vect = normalized_linear(norm);

	int curOffset = vect_coeff((Variable) index, vect);

	if(curOffset != 0)
	  {
	    if(offset != 0)
	      {
		success = false;
	      }

	    offset = curOffset;
	  }
      }
    else
      {
	success = false;
      }

    if(offset == 0)
      {
	rank++;
      }

    gen_free_list(indRefList);

  }, reference_indices(ref));

  if(success)
    {
      type refType = entity_type(reference_variable(ref));

      pips_assert("ref type is not variable", type_variable_p(refType));

      list lRefDim = variable_dimensions(type_variable(refType));

      intptr_t finalOffset;

      if(!get_final_offset(lRefDim, offset, rank, &finalOffset))
	{
	  return false;
	}

      //printf("finalOffset %d\n", finalOffset);
      //printf("\n");

      pips_assert("finalOffset > 0", (finalOffset > 0));

      hash_put(htOffset, ref, (void*)finalOffset);
    }

  return success;
}

/*
This function is used by has_call_stat_inside
 */
static bool has_call_stat_inside_flt(statement stat, bool * bHasCallStat)
{
  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_loop:
    case is_instruction_forloop:
    case is_instruction_whileloop:
      {
	return false;
	break;
      }

    case is_instruction_sequence:
    case is_instruction_test:
      {
	return true;
	break;
      }
    case is_instruction_call:
      {
	*bHasCallStat = true;
	break;
      }
    default:
      {
	pips_internal_error("unsupported tag");
	break;
      }
    }

  return false;
}

/*
This function is used by has_call_stat_inside
 */
static void has_call_stat_inside_rwt(statement stat, bool * bHasCallStat)
{
  stat = stat;
  bHasCallStat = bHasCallStat;
}

/*
This function returns true if statement stat contains a call statement.
 */
bool has_call_stat_inside(statement stat)
{
  bool bHasCallStat = false;

  gen_context_recurse(stat, &bHasCallStat, statement_domain,
		      has_call_stat_inside_flt, has_call_stat_inside_rwt);

  return bHasCallStat;
}

/*
This function returns the real fifo number from the old fifo number
fifoNum.
 */
static intptr_t get_realFifoNum(intptr_t fifoNum)
{
  static intptr_t gCurFifoCounter = 1;

  //printf("get_realFifoNum %d\n", fifoNum);

  // fifoNum value can be -1, if we want to increment
  // gCurFifoCounter with 1
  if(fifoNum < 0)
    {
      intptr_t newFifoNum = gCurFifoCounter;

      gCurFifoCounter += 1;

      return newFifoNum;
    }

  intptr_t realFifoNum = (intptr_t)hash_get(gRealFifo, (void *)fifoNum);

  // If no realFifoNum was associated with fifoNum,
  // then create a new realFifoNum
  if(realFifoNum == (intptr_t)HASH_UNDEFINED_VALUE)
    {
      realFifoNum = gCurFifoCounter;

      hash_put(gRealFifo, (void *)fifoNum, (void *)realFifoNum);

      HASH_MAP(ref1, fifo1,
      {
	if((intptr_t)fifo1 == (intptr_t)fifoNum)
	  {
	    hash_put(gRefToHREFifo, ref1, (void *)realFifoNum);
	  }
      }, gOldRefToHREFifo);

      // Get the number of fifo that has to be allocated for this
      // realFifoNum
      intptr_t inc = (intptr_t)hash_get(gRefToFifoOff, (void *)fifoNum);

      if(inc == (intptr_t)HASH_UNDEFINED_VALUE)
	{
	  inc = 1;
	}

      gCurFifoCounter += inc;
    }

  return realFifoNum;
}

/*
This function get the toggle entity associated to reference curRef
 */
static entity get_toggleEnt_from_ref(reference curRef, list lToggleEnt)
{
  void* fifoNum = hash_get(gRefToFifo, curRef);

  pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
	      fifoNum != HASH_UNDEFINED_VALUE);

  intptr_t numOfFifo = (intptr_t)hash_get(gRefToFifoOff, fifoNum);

  pips_assert("(numOfFifo == 2) || (numOfFifo == 3)",
	      (numOfFifo == 2) || (numOfFifo == 3));

  entity toggleEnt = entity_undefined;

  MAP(ENTITY, curEnt,
  {
    intptr_t curInc = (intptr_t)hash_get(gToggleToInc, curEnt);

    if(curInc == numOfFifo)
      {
	toggleEnt = curEnt;
	break;
      }

  }, lToggleEnt);

  pips_assert("toggleEnt != entity_undefined",
	      toggleEnt != entity_undefined);

  return toggleEnt;
}

static statement make_mmcd_load_store_stat(string name,
					   expression hreBuff,
					   expression refExp,
					   expression offExp,
					   expression countExp)
{
  expression stepExp;

  if(!strcmp(name, GEN_STORE_MMCD))
    {
      list addArg = gen_make_list(expression_domain,
				  entity_to_expression(gStepEnt),
				  int_to_expression(1),
				  NULL);

      stepExp =
	call_to_expression(make_call(entity_intrinsic(PLUS_C_OPERATOR_NAME),
				     addArg));
    }
  else
    {
      stepExp = entity_to_expression(gStepEnt);
    }

  list arg = gen_make_list(expression_domain,
			   stepExp,
			   hreBuff,
			   refExp,
			   offExp,
			   countExp,
			   NULL);

  statement newStat =
    call_to_statement(make_call(module_name_to_runtime_entity(name),
				arg));

  return newStat;
}

/*
This function generates one Load an Store MMCD statement associated
to curRef
 */
static statement generate_mmcd_stat_from_ref(reference curRef, int offset,
					     expression count, bool bRead,
					     list lToggleEnt)
{
  statement newStat = statement_undefined;

  string name = NULL;

  intptr_t fifoNum = (intptr_t)hash_get(gRefToFifo, curRef);

  pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
	      fifoNum != (intptr_t)HASH_UNDEFINED_VALUE);

  // Get the fifo number
  int realFifoNum = get_realFifoNum(fifoNum);

  expression hreBuff = expression_undefined;

  // If the toggle entity list is NIL, then realFifoNum is 
  // the fifo Number to use
  if(lToggleEnt == NIL)
    {
      hreBuff = int_to_expression(realFifoNum);
    }
  // else, ...
  else
    {
      printf("generate_mmcd_stat_from_ref 1\n");
      print_reference(curRef);printf("\n");

      // Get the toggle entity
      entity toggleEnt = get_toggleEnt_from_ref(curRef, lToggleEnt);

      list addArg = gen_make_list(expression_domain,
				  int_to_expression(realFifoNum),
				  entity_to_expression(toggleEnt),
				  NULL);

      // Add the toggle entity and the fifo number
      hreBuff =
	call_to_expression(make_call(entity_intrinsic(PLUS_OPERATOR_NAME),
				     addArg));
    }

  if(bRead)
    {
      name = strdup(GEN_LOAD_MMCD);
    }
  else
    {
      name = strdup(GEN_STORE_MMCD);
    }

  // Generate the statement
  newStat =
    make_mmcd_load_store_stat(name, hreBuff,
			      reference_to_expression(copy_reference(curRef)),
			      int_to_expression(offset), count);

  return newStat;
}

/*
This function generates one Load an Store MMCD statement associated
to lRef
 */
static void generate_mmcd_stats_from_ref(list lRef, hash_table htOffset,
					 expression count, list lToggleEnt,
					 list * lReadStats, list * lWriteStats)
{
  list lReadDone = NIL;
  list lWriteDone = NIL;

  intptr_t offset = 0;

  MAP(REFERENCE, curRef,
  {
    if(htOffset != NULL)
      {
	offset = (intptr_t)hash_get(htOffset, curRef);

	pips_assert("ref offset undefined", offset != (intptr_t) HASH_UNDEFINED_VALUE);
      }
    else
      {
	offset = 1;
      }

    string effAction = hash_get(gRefToEff, curRef);

    pips_assert("effAction != HASH_UNDEFINED_VALUE",
		effAction != HASH_UNDEFINED_VALUE);

    //printf("mmcd ref\n");
    //print_reference(curRef);printf("\n");

    if(!strcmp(effAction, R_EFFECT))
      {
	//printf("read eff\n");

	statement readStat = generate_mmcd_stat_from_ref(curRef, offset,
							 count, true,
							 lToggleEnt);

	//print_statement(readStat);

	*lReadStats = gen_nconc(*lReadStats, CONS(STATEMENT, readStat, NIL));

	lReadDone = gen_nconc(lReadDone, CONS(REFERENCE, curRef, NIL));
      }
    else
      {
	//printf("write eff\n");

	statement writeStat = generate_mmcd_stat_from_ref(curRef, offset,
							  count, false,
							  lToggleEnt);

	//print_statement(writeStat);

	*lWriteStats = gen_nconc(*lWriteStats, CONS(STATEMENT, writeStat, NIL));

	lWriteDone = gen_nconc(lWriteDone, CONS(REFERENCE, curRef, NIL));
      }

  }, lRef);

  gen_free_list(lReadDone);
  gen_free_list(lWriteDone);
}

/*
This function creates a statement to increment the value of the step
variable
 */
statement make_step_inc_statement(int incNum)
{
  list addArg = gen_make_list(expression_domain,
			      entity_to_expression(gStepEnt),
			      int_to_expression(incNum),
			      NULL);

  expression rExp = call_to_expression(make_call(entity_intrinsic(PLUS_OPERATOR_NAME),
						 addArg));

  return make_assign_statement(entity_to_expression(gStepEnt), rExp);
}

/*
This function creates a statement that has to be put at the 
beginning of a generated loop statement to move forward in the pipeline
 */
statement make_loop_step_stat(statement stat, entity newOuterInd)
{
  loop curLoop = statement_loop(stat);

  statement stepStat;

  if(gGenHRE)
    {
      stepStat = make_wait_step_statement();
    }
  else
    {
      stepStat = make_step_inc_statement(1);
    }

  list neArg = gen_make_list(expression_domain,
			     entity_to_expression(newOuterInd),
			     copy_expression(range_lower(loop_range(curLoop))),
			     NULL);

  expression neExp = 
    call_to_expression(make_call(entity_intrinsic(C_GREATER_THAN_OPERATOR_NAME),
				 neArg));

  test t = make_test(neExp, stepStat, make_empty_statement());

  stepStat = make_statement(entity_empty_label(), 
			    STATEMENT_NUMBER_UNDEFINED,
			    STATEMENT_ORDERING_UNDEFINED,
			    empty_comments,
			    make_instruction(is_instruction_test, t),
			    NIL,NULL,
			    empty_extensions (), make_synchronization_none());

  return stepStat;
}

/*
This function is used by has_loop_inside
 */
static bool has_loop_inside_flt(statement stat, bool * bHasLoop)
{
  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_loop:
    case is_instruction_forloop:
    case is_instruction_whileloop:
      {
	*bHasLoop = true;
	return false;
	break;
      }

    case is_instruction_sequence:
    case is_instruction_test:
      {
	return true;
	break;
      }
    case is_instruction_call:
      {
	break;
      }
    default:
      {
	pips_internal_error("unsupported tag");
	break;
      }
    }

  return false;
}

/*
This function is used by has_loop_inside
 */
static void has_loop_inside_rwt(statement stat, bool * bHasLoop)
{
  stat = stat;
  bHasLoop = bHasLoop;
}

/*
This function returns true if statement stat contains a call statement.
 */
static bool has_loop_inside(statement stat)
{
  bool bHasLoop = false;

  gen_context_recurse(stat, &bHasLoop, statement_domain,
		      has_loop_inside_flt, has_loop_inside_rwt);

  return bHasLoop;
}

/*
This function creates a fifo associated to entity ent.
 */
static int alloc_new_slot(entity ent)
{
  static intptr_t curFifo = -1;
  static intptr_t curInd = 0;

  /*if((curFifo == -1) || (curInd == get_int_property("COMENGINE_SIZE_OF_FIFO")))
    {*/
      curFifo = get_realFifoNum(-1);

      curInd = 0;
      //}

  hash_put(gEntToHREFifo, ent, (void*)curFifo);

  hash_put(gIndToNum, ent, (void*)curInd++);

  return curFifo;
}

/*
This function creates a MMCD statement to update the toggle value in the 
HRE fifo.
 */
statement make_toggle_mmcd(entity ent)
{
  int fifoNum = alloc_new_slot(ent);

  statement newStat =
    make_mmcd_load_store_stat(strdup(GEN_LOAD_MMCD),
			      int_to_expression(fifoNum),
			      entity_to_expression(ent),
			      int_to_expression(1),
			      int_to_expression(1));

  return newStat;
}

/*
This function finds or creates a fifo associated to entity ent.
 */
static int find_or_create_slot(entity ent)
{
  intptr_t fifoNum = (intptr_t)hash_get(gEntToHREFifo, ent);

  if(fifoNum == (intptr_t)HASH_UNDEFINED_VALUE)
    {
      fifoNum = alloc_new_slot(ent);
    }

  return fifoNum;
}

/*
This function creates a MMCD statement to update the newInd value in the 
HRE fifo.
 */
statement make_init_newInd_stat(statement stat, entity newInd)
{
  //printf("make_init_newInd_stat beg\n");

  list lUnSupportedRef = hash_get(gLoopToUnSupRef, stat);

  pips_assert("lUnSupportedRef != HASH_UNDEFINED_VALUE",
	      lUnSupportedRef != HASH_UNDEFINED_VALUE);

  if((!has_loop_inside(loop_body(statement_loop(stat)))) &&
     (lUnSupportedRef == NIL))
    {
      return statement_undefined;
    }

  int fifoNum = find_or_create_slot(loop_index(statement_loop(stat)));

  statement newStat =
    make_mmcd_load_store_stat(strdup(GEN_LOAD_MMCD),
			      int_to_expression(fifoNum),
			      entity_to_expression(newInd),
			      int_to_expression(1),
			      int_to_expression(1));

  //printf("make_init_newInd_stat end\n");
  return newStat;
}

/*
This functions creates statements that are put at the beginning ot the body
of the loop tiling outer loop
 */
statement make_transStat(statement stat, entity newOuterInd,
			 entity transferSize, expression bufferSizeExp)
{
  statement newStat = statement_undefined;

  loop curLoop = statement_loop(stat);

  list rgSizeArg1 = gen_make_list(expression_domain,
				  copy_expression(range_upper(loop_range(curLoop))),
				  copy_expression(bufferSizeExp),
				  NULL);

  expression rgSizeExp1 = 
    call_to_expression(make_call(entity_intrinsic(DIVIDE_OPERATOR_NAME),
				 rgSizeArg1));


  list arg2Arg = gen_make_list(expression_domain,
			       copy_expression(bufferSizeExp),
			       rgSizeExp1,
			       NULL);

  expression arg2 = call_to_expression(make_call(entity_intrinsic(MULTIPLY_OPERATOR_NAME),
						 arg2Arg));

  list leArg = gen_make_list(expression_domain,
			     arg2,
			     entity_to_expression(newOuterInd),
			     NULL);

  expression leExp = 
    call_to_expression(make_call(entity_intrinsic(C_LESS_OR_EQUAL_OPERATOR_NAME),
				 leArg));

  list modArg = gen_make_list(expression_domain,
			      copy_expression(range_upper(loop_range(curLoop))),
			      copy_expression(bufferSizeExp),
			      NULL);

  expression modExp =
    call_to_expression(make_call(entity_intrinsic(C_MODULO_OPERATOR_NAME),
				 modArg));

  list addArg = gen_make_list(expression_domain,
			      modExp,
			      int_to_expression(1),
			      NULL);

  expression addExp =
    call_to_expression(make_call(entity_intrinsic(PLUS_C_OPERATOR_NAME),
				 addArg));

  statement trueStat = make_assign_statement(entity_to_expression(transferSize),
					     addExp);

  statement falseStat = make_assign_statement(entity_to_expression(transferSize),
					      copy_expression(bufferSizeExp));

  expression rgUpper = range_upper(loop_range(curLoop));
  intptr_t upVal = -1;
  intptr_t rate = -1;
  expression_integer_value(rgUpper, &upVal);
  expression_integer_value(bufferSizeExp, &rate);

  if((upVal != -1) &&
     (rate != -1) &&
     ((upVal+1)%rate == 0))
    {
      free_expression(leExp);
      free_statement(trueStat);
      newStat = falseStat;
    }
  else if((upVal+1) < rate)
    {
      free_expression(leExp);
      free_statement(falseStat);
      newStat = trueStat;
    }
  else
    {
      test t = make_test(leExp, trueStat, falseStat);

      newStat = make_statement(entity_empty_label(), 
			       STATEMENT_NUMBER_UNDEFINED,
			       STATEMENT_ORDERING_UNDEFINED,
			       empty_comments,
			       make_instruction(is_instruction_test, t),
			       NIL,NULL,
			       empty_extensions (), make_synchronization_none());
    }

  /*statement stepStat = make_loop_step_stat(stat, newOuterInd);

  newStat = make_block_statement(gen_nconc(CONS(STATEMENT, stepStat, NIL),
					   CONS(STATEMENT, newStat, NIL)));
  */
  /*if(!gGenHRE)
    {
      statement indStat = make_init_newInd_stat(stat, newOuterInd);

      if(indStat != statement_undefined)
	{
	  newStat =
	    make_block_statement(gen_nconc(CONS(STATEMENT, newStat, NIL),
					   CONS(STATEMENT, indStat, NIL)));
	}
    }
*/
  return newStat;
}

static reference gRefToReplace;

/*
This function is used by comEngine_replace_reference_in_stat
 */
static void comEngine_replace_reference_in_stat_rwt(expression exp, expression newExp)
{
  if(expression_reference_p(exp) &&
     reference_equal_p(gRefToReplace,
		       expression_reference(exp)))
    {
      free_normalized(expression_normalized(exp));

      expression_normalized(exp) = normalized_undefined;

      free_syntax(expression_syntax(exp));

      expression_syntax(exp) = expression_syntax(newExp);

      NORMALIZE_EXPRESSION(exp);
    }
}

/*
This function replace all the references equal to ref in statement stat
by the expression new.
 */
void comEngine_replace_reference_in_stat(statement stat,
					 reference ref, expression new)
{
  gRefToReplace = ref;

  gen_context_recurse(stat, new, expression_domain,
		      gen_true, comEngine_replace_reference_in_stat_rwt);
}

/*
This function creates the MMCD Load and Store statements generated from
lSupportedRef
 */
static list make_mmcd_stats_from_ref(statement stat, list lSupportedRef, hash_table htOffset,
				     entity transferSize, entity newOuterInd, statement innerStat)
{
  loop curLoop = statement_loop(stat);

  list lStats = NIL;

  list lSupReadStats = NIL;
  list lSupWriteStats = NIL;

  list lToggleEnt = hash_get(gLoopToToggleEnt, stat);

  pips_assert("lToggleEnt != HASH_UNDEFINED_VALUE",
	      lToggleEnt != HASH_UNDEFINED_VALUE);

  generate_mmcd_stats_from_ref(lSupportedRef, htOffset,
			       entity_to_expression(transferSize), lToggleEnt,
			       &lSupReadStats, &lSupWriteStats);

  MAP(REFERENCE, curRef,
  {
    hash_put(gRefToInd, curRef, loop_index(statement_loop(stat)));
  }, lSupportedRef);

  MAP(REFERENCE, curRef,
  {
    entity toggleEnt = get_toggleEnt_from_ref(curRef, lToggleEnt);

    hash_put(gRefToToggle, curRef, toggleEnt);

  }, lSupportedRef);

  MAP(STATEMENT, curStat,
  {
    comEngine_replace_reference_in_stat(curStat,
			      make_reference(loop_index(curLoop), NIL),
			      entity_to_expression(newOuterInd));
  }, lSupReadStats);

  MAP(STATEMENT, curStat,
  {
    comEngine_replace_reference_in_stat(curStat,
			      make_reference(loop_index(curLoop), NIL),
			      entity_to_expression(newOuterInd));
  }, lSupWriteStats);

  lStats = gen_nconc(lStats, lSupReadStats);

  if(innerStat != statement_undefined)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, innerStat, NIL));
    }

  lStats = gen_nconc(lStats, lSupWriteStats);

  return lStats;
}

static list make_lStats(statement stat, entity transferSize,
			statement innerStat, entity newOuterInd,
			list lSupportedRef, hash_table htOffset,
			expression bufferSizeExp)
{
  list lStats = NIL;

  statement transStat = make_transStat(stat, newOuterInd,
				       transferSize, bufferSizeExp);

  lStats = gen_nconc(lStats, CONS(STATEMENT, transStat, NIL));

  list lTemp = NIL;

  lTemp = make_mmcd_stats_from_ref(stat, lSupportedRef, htOffset,
				   transferSize, newOuterInd, innerStat);

  lStats = gen_nconc(lStats, lTemp);

  return lStats;
}

/*
This function creates an Execution MMCD
 */
statement make_exec_mmcd()
{
  static int number = 0;

  list arg = gen_make_list(expression_domain,
			   entity_to_expression(gStepEnt),
			   int_to_expression(number++),
			   NULL);

  statement mmcdStat =
    call_to_statement(
		      make_call(module_name_to_runtime_entity(strdup("GEN_EXEC_MMCD")),
				arg));

  return mmcdStat;
}

/*
This function generates the Load and Store MMCD statements associated
to lRef. Then, it updates the step value and concatenate the created 
statements to stat
 */
statement generate_stat_from_ref_list_proc(list lRef, list lToggleEnt,
					   statement stat)
{
  list lStats = NIL;

  list lReadStats = NIL;
  list lWriteStats = NIL;

  generate_mmcd_stats_from_ref(lRef, NULL,
			       int_to_expression(1),
			       lToggleEnt,
			       &lReadStats, &lWriteStats);

  statement stepStat1 = make_step_inc_statement(1);
  statement stepStat2 = make_step_inc_statement(2);

  lStats = gen_nconc(CONS(STATEMENT, stepStat1, NIL), lReadStats);
  if(stat != statement_undefined)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, stat, NIL));
    }
  lStats = gen_nconc(lStats, lWriteStats);

  if(lWriteStats != NIL)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, stepStat2, NIL));
    }

  return make_block_statement(lStats);
}

/*
This function generates the Load and Store MMCD statements associated
to lRef. Then, it updates the step value and concatenate the created 
statements to stat
 */
list generate_stat_from_ref_list_proc_list(list lRef, list lInStats)
{
  list lStats = NIL;

  list lReadStats = NIL;
  list lWriteStats = NIL;

  generate_mmcd_stats_from_ref(lRef, NULL,
			       int_to_expression(1),
			       NIL,
			       &lReadStats, &lWriteStats);

  statement stepStat1 = make_step_inc_statement(1);
  statement stepStat2 = make_step_inc_statement(2);

  lStats = gen_nconc(CONS(STATEMENT, stepStat1, NIL), lReadStats);

  lStats = gen_nconc(lStats, lInStats);

  lStats = gen_nconc(lStats, lWriteStats);

  if(lWriteStats != NIL)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, stepStat2, NIL));
    }

  return lStats;
}

/*
This function replace in stat the old index by the new ones
 */
void process_innerStat1_proc(statement stat, entity oldInd,
			     entity newOuterInd, entity newInnerInd)
{
  list addArg = gen_make_list(expression_domain,
			      entity_to_expression(newOuterInd),
			      entity_to_expression(newInnerInd),
			      NULL);

  expression arg2 = call_to_expression(make_call(entity_intrinsic(PLUS_OPERATOR_NAME),
						 addArg));

  comEngine_replace_reference_in_stat(stat,
				      make_reference(oldInd, NIL),
				      arg2);
}

/*
This function computes the supported and unsupported references and 
memorize them in gLoopToSupRef and gLoopToUnSupRef
 */
void get_supportedRef_proc(statement stat, hash_table htOffset,
			   list * lSupportedRef, list * lUnSupportedRef)
{
  loop curLoop = statement_loop(stat);

  list lRef = hash_get(gLoopToRef, stat);

  if(lRef == HASH_UNDEFINED_VALUE)
    {
      lRef = NIL;
    }

  MAP(REFERENCE, curRef,
  {
    //printf("loop ref\n");
    //print_reference(curRef);printf("\n");

    if(supported_ref_p(curRef, loop_index(curLoop), htOffset))
      {
	*lSupportedRef = gen_nconc(*lSupportedRef, CONS(REFERENCE, curRef, NIL));
      }
    else
      {
	*lUnSupportedRef = gen_nconc(*lUnSupportedRef, CONS(REFERENCE, curRef, NIL));
      }

  }, lRef);

  hash_put(gLoopToSupRef, stat, *lSupportedRef);
  hash_put(gLoopToUnSupRef, stat, *lUnSupportedRef);
}

statement get_call_stat_proc(statement stat)
{
  stat = stat;

  return statement_undefined;
}

list make_loop_lStats_proc(statement stat, entity transferSize,
			   statement innerStat, entity newOuterInd,
			   list lSupportedRef, hash_table htOffset,
			   expression bufferSizeExp)
{
  return make_lStats(stat, transferSize,
		     innerStat, newOuterInd,
		     lSupportedRef, htOffset,
		     bufferSizeExp);
}

/*
This function creates a statement that make sure that the out effect of
the old index of curLoop is respected
 */
list add_index_out_effect_proc(loop curLoop, list lStats)
{
  statement assignStat =
    make_assign_statement(entity_to_expression(loop_index(curLoop)),
			  copy_expression(range_upper(loop_range(curLoop))));

  return gen_nconc(lStats, CONS(STATEMENT, assignStat, NIL));
}

/*
This function generates the MMCD code from a test
statement.
 */
statement generate_code_test_proc(statement stat)
{
  printf("generate_code_test\n");
  print_statement(stat);printf("\n");
  statement newStat = statement_undefined;

  newStat = get_call_stat_proc(stat);

  list lRef = hash_get(gStatToRef, stat);

  if(lRef != HASH_UNDEFINED_VALUE)
    {
      newStat = generate_stat_from_ref_list_proc(lRef, NIL, newStat);
    }

  // Get the new statements for the true statement
  statement trueStat = generate_code_function(test_true(statement_test(stat)), false);

  // Get the new statements for the false statement
  statement falseStat = generate_code_function(test_false(statement_test(stat)), false);

  list lStats = NIL;

  if(newStat != statement_undefined)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, newStat, NIL));
    }

  if(trueStat != statement_undefined)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, trueStat, NIL));
    }

  if(falseStat != statement_undefined)
    {
      lStats = gen_nconc(lStats, CONS(STATEMENT, falseStat, NIL));
    }

  newStat = make_block_statement(lStats);

  return newStat;
}

/*
This function adds some bubbles in the pipeline if needed
 */
list process_gLoopToSync_proc(statement stat, list lInStats)
{
  bool loopSync = (intptr_t)hash_get(gLoopToSync, stat);

  if(loopSync == (intptr_t)HASH_UNDEFINED_VALUE)
    {
      return lInStats;
    }

  statement stepStat = make_step_inc_statement(2);

  list lStats = NIL;

  lStats = gen_nconc(lStats, lInStats);
  lStats = gen_nconc(lStats, CONS(STATEMENT, stepStat, NIL));

  return lInStats;
}

/*
This function creates the real fifos associated to lRef references
 */
void create_realFifo_proc(statement stat, list lRef)
{
  list lDone = NIL;

  //bool readAndWrite = false;

  MAP(REFERENCE, curRef1,
  {
    bool bDone = false;

    MAP(REFERENCE, refDone,
    {
      if(reference_equal_p(curRef1, refDone))
	{
	  hash_put(gRefToFifoOff, hash_get(gRefToFifo, refDone), (void *)3);
	  bDone = true;
	  break;
	}
    }, lDone);

    if(bDone)
      continue;

    void* fifoNum = hash_get(gRefToFifo, curRef1);

    pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
		fifoNum != HASH_UNDEFINED_VALUE);

    hash_put(gRefToFifoOff, fifoNum, (void *)2);

    MAP(REFERENCE, curRef2,
    {
      if(reference_equal_p(curRef1, curRef2))
	{
	  string effAction1 = hash_get(gRefToEff, curRef1);

	  pips_assert("effAction1 != HASH_UNDEFINED_VALUE",
		      effAction1 != HASH_UNDEFINED_VALUE);

	  string effAction2 = hash_get(gRefToEff, curRef2);

	  pips_assert("effAction2 != HASH_UNDEFINED_VALUE",
		      effAction2 != HASH_UNDEFINED_VALUE);

	  if(strcmp(effAction1, effAction2))
	    {
	      hash_put(gRefToFifoOff, fifoNum, (void *)3);

	      lDone = CONS(REFERENCE, curRef1, lDone);

	      break;
	    }
	}
    }, lRef);
  }, lRef);

  gen_free_list(lDone);

  printf("create_realFifo_proc\n");
  MAP(REFERENCE, curRef,
  {
    printf("create_realFifo_proc it\n");
    print_reference(curRef);printf("\n");
    intptr_t fifoNum = (intptr_t)hash_get(gRefToFifo, curRef);

    pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
		fifoNum != (intptr_t)HASH_UNDEFINED_VALUE);

    get_realFifoNum(fifoNum);
  }, lRef);

  printf("create_realFifo_proc end\n");
}

/*
This function generates the MMCDs generation code
 */
statement comEngine_generate_procCode(statement externalized_code,
				      list l_in, list l_out)
{
  l_in = l_in;
  l_out = l_out;

  statement newStat;

  // Initialize some global variables
  gOldRefToHREFifo = gRefToHREFifo;
  gRefToHREFifo = hash_table_make(hash_pointer, 0);
  gRealFifo = hash_table_make(hash_pointer, 0);
  gStepEnt =comEngine_make_new_scalar_variable
    (strdup("step"),
					   make_basic(is_basic_int, (void*)4));
  gGenHRE = false;

  // Do the job
  newStat = comEngine_generate_code(externalized_code);

  // Add the step variable initialization
  statement stepStat = 
    make_assign_statement(entity_to_expression(gStepEnt),
			  int_to_expression(1));
  
  // Add the start HRE statement
  statement startStat = 
    call_to_statement(make_call(module_name_to_runtime_entity(strdup(START_HRE)),
				NIL));

  // Add the wait HRE statement
  statement waitStat = 
    call_to_statement(make_call(module_name_to_runtime_entity(strdup(WAIT_FOR_HRE)),
				NIL));

  list lStats = gen_make_list(statement_domain,
			      stepStat,
			      newStat,
			      startStat,
			      waitStat,
			      NULL);

  newStat = make_block_statement(lStats);

  // Fre some global variables
  hash_table_free(gRealFifo);
  hash_table_free(gOldRefToHREFifo);

  //printf("final newStat\n");
  //print_statement(newStat);

  return newStat;
}
