/*
This file contains functions used to generate the MMCDs generation code
 */

#include <stdio.h>
#include <ctype.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "text-util.h"
#include "properties.h"
#include "prettyprint.h"

#include "dg.h"
#include "transformations.h"
#include "transformer.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "ricedg.h"
#include "semantics.h"
#include "control.h"

#include "phrase_tools.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "phrase_distribution.h"
#include "comEngine.h"
#include "comEngine_generate_code.h"

// See the file comEngine_distribute.c to know
// what this hash_table's mean
extern hash_table gLoopToRef;
extern hash_table gStatToRef;
extern hash_table gRefToEff;
extern hash_table gLoopToSync;
extern hash_table gLoopToSupRef;
extern hash_table gLoopToUnSupRef;
extern expression gBufferSizeEnt;
extern hash_table gRefToFifo;
extern hash_table gRefToFifoOff;
extern hash_table gRefToHREFifo;
extern hash_table gLoopToToggleEnt;
extern hash_table gEntToHREFifo;
extern hash_table gIndToNum;
extern hash_table gRefToInd;
extern hash_table gRefToToggle;
extern hash_table gToggleToInc;

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
This function return TRUE if the reference whose indices are lRefDim is
supported and, in this case, the value of the offset is put in retVal.
 */
static bool get_final_offset(list lRefDim, int offset, int rank, int * retVal)
{
  bool fort_org = get_bool_property("SIMD_FORTRAN_MEM_ORGANISATION");

  int i = 0;
  int ind;

  int finalOffset = 1;

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
	  return FALSE;
	}

      int lower = integer_constant_expression_value(lowerExp);
      int upper = integer_constant_expression_value(upperExp);

      pips_assert("lower < upper", lower < upper);

      finalOffset = finalOffset * (upper - lower + 1);
    }

  *retVal = finalOffset * offset;

  return TRUE;
}

/*
This function return TRUE if reference ref is
supported and, in this case, it updates the hash_table htOffset.
 */
static bool supported_ref_p(reference ref, entity index, hash_table htOffset)
{
  bool success = TRUE;
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
		success = FALSE;
	      }

	    offset = curOffset;
	  }
      }
    else
      {
	success = FALSE;
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

      int finalOffset;

      if(!get_final_offset(lRefDim, offset, rank, &finalOffset))
	{
	  return FALSE;
	}

      //printf("finalOffset %d\n", finalOffset);
      //printf("\n");

      pips_assert("finalOffset > 0", (finalOffset > 0));

      hash_put(htOffset, ref, (void*)finalOffset);
    }

  return success;
}

/*
This function moves the entity declaration from the module declaration
list to the module statement declaration list
 */
static void procCode_move_declarations(entity ent, entity new_fun,
				       statement stat)
{
  gen_remove(&code_declarations(value_code(entity_initial(new_fun))), ent);

  statement_declarations(stat) = gen_nconc(statement_declarations(stat),
					   CONS(ENTITY, ent, NIL));
}

/*
This function creates a new entity and stores its declaration
in the module statement declaration list
 */
entity make_new_C_scalar_variable_with_prefix(string prefix,
					      entity module,
					      statement stat,
					      basic b)
{
  entity retEnt = 
    make_new_scalar_variable_with_prefix(prefix,
					 module,
					 b);

  procCode_move_declarations(retEnt, module, stat);

  return retEnt;
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
	return FALSE;
	break;
      }

    case is_instruction_sequence:
    case is_instruction_test:
      {
	return TRUE;
	break;
      }
    case is_instruction_call:
      {
	*bHasCallStat = TRUE;
	break;
      }
    default:
      {
	pips_error("has_call_stat_inside_flt", "unsupported tag");
	break;
      }
    }

  return FALSE;
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
This function returns TRUE if statement stat contains a call statement.
 */
bool has_call_stat_inside(statement stat)
{
  bool bHasCallStat = FALSE;

  gen_context_recurse(stat, &bHasCallStat, statement_domain,
		      has_call_stat_inside_flt, has_call_stat_inside_rwt);

  return bHasCallStat;
}

/*
This function returns the real fifo number from the old fifo number
fifoNum.
 */
static int get_realFifoNum(int fifoNum)
{
  static int gCurFifoCounter = 1;

  //printf("get_realFifoNum %d\n", fifoNum);

  // fifoNum value can be -1, if we want to increment
  // gCurFifoCounter with 1
  if(fifoNum < 0)
    {
      int newFifoNum = gCurFifoCounter;

      gCurFifoCounter += 1;

      return newFifoNum;
    }

  int realFifoNum = (int)hash_get(gRealFifo, (void *)fifoNum);

  // If no realFifoNum was associated with fifoNum,
  // then create a new realFifoNum
  if(realFifoNum == (int)HASH_UNDEFINED_VALUE)
    {
      realFifoNum = gCurFifoCounter;

      hash_put(gRealFifo, (void *)fifoNum, (void *)realFifoNum);

      HASH_MAP(ref1, fifo1,
      {
	if((int)fifo1 == (int)fifoNum)
	  {
	    hash_put(gRefToHREFifo, ref1, (void *)realFifoNum);
	  }
      }, gOldRefToHREFifo);

      // Get the number of fifo that has to be allocated for this
      // realFifoNum
      int inc = (int)hash_get(gRefToFifoOff, (void *)fifoNum);

      if(inc == (int)HASH_UNDEFINED_VALUE)
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
  int fifoNum = (int)hash_get(gRefToFifo, curRef);

  pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
	      fifoNum != (int)HASH_UNDEFINED_VALUE);

  int numOfFifo = (int)hash_get(gRefToFifoOff, fifoNum);

  pips_assert("(numOfFifo == 2) || (numOfFifo == 3)",
	      (numOfFifo == 2) || (numOfFifo == 3));

  entity toggleEnt = entity_undefined;

  MAP(ENTITY, curEnt,
  {
    int curInc = (int)hash_get(gToggleToInc, curEnt);

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
				  make_integer_constant_expression(1),
				  NULL);

      stepExp =
	call_to_expression(make_call(get_function_entity(PLUS_C_OPERATOR_NAME),
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
    call_to_statement(make_call(get_function_entity(name),
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

  int fifoNum = (int)hash_get(gRefToFifo, curRef);

  pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
	      fifoNum != (int)HASH_UNDEFINED_VALUE);

  // Get the fifo number
  int realFifoNum = get_realFifoNum(fifoNum);

  expression hreBuff = expression_undefined;

  // If the toggle entity list is NIL, then realFifoNum is 
  // the fifo Number to use
  if(lToggleEnt == NIL)
    {
      hreBuff = make_integer_constant_expression(realFifoNum);
    }
  // else, ...
  else
    {
      printf("generate_mmcd_stat_from_ref 1\n");
      print_reference(curRef);printf("\n");

      // Get the toggle entity
      entity toggleEnt = get_toggleEnt_from_ref(curRef, lToggleEnt);

      list addArg = gen_make_list(expression_domain,
				  make_integer_constant_expression(realFifoNum),
				  entity_to_expression(toggleEnt),
				  NULL);

      // Add the toggle entity and the fifo number
      hreBuff =
	call_to_expression(make_call(get_function_entity(PLUS_OPERATOR_NAME),
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
			      make_integer_constant_expression(offset), count);

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

  int offset = 0;

  MAP(REFERENCE, curRef,
  {
    if(htOffset != NULL)
      {
	offset = (int)hash_get(htOffset, curRef);

	pips_assert("ref offset undefined", offset != (int) HASH_UNDEFINED_VALUE);
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
							 count, TRUE,
							 lToggleEnt);

	//print_statement(readStat);

	*lReadStats = gen_nconc(*lReadStats, CONS(STATEMENT, readStat, NIL));

	lReadDone = gen_nconc(lReadDone, CONS(REFERENCE, curRef, NIL));
      }
    else
      {
	//printf("write eff\n");

	statement writeStat = generate_mmcd_stat_from_ref(curRef, offset,
							  count, FALSE,
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
			      make_integer_constant_expression(incNum),
			      NULL);

  expression rExp = call_to_expression(make_call(get_function_entity(PLUS_OPERATOR_NAME),
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
    call_to_expression(make_call(get_function_entity(C_GREATER_THAN_OPERATOR_NAME),
				 neArg));

  test t = make_test(neExp, stepStat, make_empty_statement());

  stepStat = make_statement(entity_empty_label(), 
			    STATEMENT_NUMBER_UNDEFINED,
			    STATEMENT_ORDERING_UNDEFINED,
			    empty_comments,
			    make_instruction(is_instruction_test, t),
			    NIL,NULL,
			    extensions_undefined);

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
	*bHasLoop = TRUE;
	return FALSE;
	break;
      }

    case is_instruction_sequence:
    case is_instruction_test:
      {
	return TRUE;
	break;
      }
    case is_instruction_call:
      {
	break;
      }
    default:
      {
	pips_error("has_call_stat_inside_flt", "unsupported tag");
	break;
      }
    }

  return FALSE;
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
This function returns TRUE if statement stat contains a call statement.
 */
static bool has_loop_inside(statement stat)
{
  bool bHasLoop = FALSE;

  gen_context_recurse(stat, &bHasLoop, statement_domain,
		      has_loop_inside_flt, has_loop_inside_rwt);

  return bHasLoop;
}

/*
This function creates a fifo associated to entity ent.
 */
static int alloc_new_slot(entity ent)
{
  static int curFifo = -1;
  static int curInd = 0;

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
			      make_integer_constant_expression(fifoNum),
			      entity_to_expression(ent),
			      make_integer_constant_expression(1),
			      make_integer_constant_expression(1));

  return newStat;
}

/*
This function finds or creates a fifo associated to entity ent.
 */
static int find_or_create_slot(entity ent)
{
  int fifoNum = (int)hash_get(gEntToHREFifo, ent);

  if(fifoNum == (int)HASH_UNDEFINED_VALUE)
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
			      make_integer_constant_expression(fifoNum),
			      entity_to_expression(newInd),
			      make_integer_constant_expression(1),
			      make_integer_constant_expression(1));

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
    call_to_expression(make_call(get_function_entity(DIVIDE_OPERATOR_NAME),
				 rgSizeArg1));


  list arg2Arg = gen_make_list(expression_domain,
			       copy_expression(bufferSizeExp),
			       rgSizeExp1,
			       NULL);

  expression arg2 = call_to_expression(make_call(get_function_entity(MULTIPLY_OPERATOR_NAME),
						 arg2Arg));

  list leArg = gen_make_list(expression_domain,
			     arg2,
			     entity_to_expression(newOuterInd),
			     NULL);

  expression leExp = 
    call_to_expression(make_call(get_function_entity(C_LESS_OR_EQUAL_OPERATOR_NAME),
				 leArg));

  list modArg = gen_make_list(expression_domain,
			      copy_expression(range_upper(loop_range(curLoop))),
			      copy_expression(bufferSizeExp),
			      NULL);

  expression modExp =
    call_to_expression(make_call(get_function_entity(C_MODULO_OPERATOR_NAME),
				 modArg));

  list addArg = gen_make_list(expression_domain,
			      modExp,
			      make_integer_constant_expression(1),
			      NULL);

  expression addExp =
    call_to_expression(make_call(get_function_entity(PLUS_C_OPERATOR_NAME),
				 addArg));

  statement trueStat = make_assign_statement(entity_to_expression(transferSize),
					     addExp);

  statement falseStat = make_assign_statement(entity_to_expression(transferSize),
					      copy_expression(bufferSizeExp));

  expression rgUpper = range_upper(loop_range(curLoop));
  int upVal = -1;
  int rate = -1;
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
			       extensions_undefined);
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
			   make_integer_constant_expression(number++),
			   NULL);

  statement mmcdStat =
    call_to_statement(
		      make_call(get_function_entity(strdup("GEN_EXEC_MMCD")),
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
			       make_integer_constant_expression(1),
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
			       make_integer_constant_expression(1),
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

  expression arg2 = call_to_expression(make_call(get_function_entity(PLUS_OPERATOR_NAME),
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
  statement trueStat = generate_code_function(test_true(statement_test(stat)), FALSE);

  // Get the new statements for the false statement
  statement falseStat = generate_code_function(test_false(statement_test(stat)), FALSE);

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
  bool loopSync = (bool)hash_get(gLoopToSync, stat);

  if(loopSync == (bool)HASH_UNDEFINED_VALUE)
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

  bool readAndWrite = FALSE;

  MAP(REFERENCE, curRef1,
  {
    bool bDone = FALSE;

    MAP(REFERENCE, refDone,
    {
      if(reference_equal_p(curRef1, refDone))
	{
	  hash_put(gRefToFifoOff, hash_get(gRefToFifo, refDone), (void *)3);
	  bDone = TRUE;
	  break;
	}
    }, lDone);

    if(bDone)
      continue;

    int fifoNum = (int)hash_get(gRefToFifo, curRef1);

    pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
		fifoNum != (int)HASH_UNDEFINED_VALUE);

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
    int fifoNum = (int)hash_get(gRefToFifo, curRef);

    pips_assert("fifoNum != HASH_UNDEFINED_VALUE",
		fifoNum != (int)HASH_UNDEFINED_VALUE);

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
  gStepEnt =
    make_new_C_scalar_variable_with_prefix(strdup("step"),
					   get_current_module_entity(),
					   get_current_module_statement(),
					   make_basic(is_basic_int, (void*)4));
  gGenHRE = FALSE;

  // Do the job
  newStat = comEngine_generate_code(externalized_code);

  // Add the step variable initialization
  statement stepStat = 
    make_assign_statement(entity_to_expression(gStepEnt),
			  make_integer_constant_expression(1));
  
  // Add the start HRE statement
  statement startStat = 
    call_to_statement(make_call(get_function_entity(strdup(START_HRE)),
				NIL));

  // Add the wait HRE statement
  statement waitStat = 
    call_to_statement(make_call(get_function_entity(strdup(WAIT_FOR_HRE)),
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
