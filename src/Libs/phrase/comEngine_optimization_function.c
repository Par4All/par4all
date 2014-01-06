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
#include "phrase.h"

hash_table gLoopToOpt;

static list glSeqs = NIL;
static list glStats = NIL;

static void opt_loop_interchange_fill_lists_stat(statement stat)
{
  glStats = gen_nconc(glStats, CONS(STATEMENT, stat, NIL));
}

static void opt_loop_interchange_fill_lists(statement stat)
{
  void* bIsNewLoop = hash_get(gIsNewLoop, stat);

  if(bIsNewLoop != HASH_UNDEFINED_VALUE)
    {
      if(glStats != NIL)
	{
	  printf("put glSeqs 1\n");
	  MAP(STATEMENT, curStat,
	  {
	    print_statement(curStat);
	  }, glStats);

	  glSeqs = gen_nconc(glSeqs, CONS(LIST, glStats, NIL));

	  glStats = NIL;
	}

      printf("put glSeqs 2\n");
      print_statement(stat);

      glSeqs = gen_nconc(glSeqs, CONS(LIST, CONS(STATEMENT, stat, NIL), NIL));

      return;
    }

  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_sequence:
      {
        MAP(STATEMENT, curStat,
	{
	  opt_loop_interchange_fill_lists(curStat);

	}, sequence_statements(instruction_sequence(instr)));
	break;
      }
      /*case is_instruction_loop:
      {
        //opt_loop_interchange_fill_lists_loop(stat);
	break;
	}*/
    case is_instruction_call:
    case is_instruction_test:
      {
	opt_loop_interchange_fill_lists_stat(stat);
	break;
      }
    default:
      {
	pips_internal_error("impossible");
	break;
      }
    }
}

static statement make_loopPattern(statement innerStat)
{
  loop curLoop = statement_loop(innerStat);

  range newRange = copy_range(loop_range(curLoop));

  loop newLoop = make_loop(loop_index(curLoop),
			   newRange,
			   statement_undefined,
			   loop_label(curLoop),
			   make_execution(is_execution_sequential, UU),
			   NIL);

  return make_statement(entity_empty_label(),
			STATEMENT_NUMBER_UNDEFINED,
			STATEMENT_ORDERING_UNDEFINED,
			empty_comments,
			make_instruction_loop(newLoop),NIL,NULL,
			empty_extensions (), make_synchronization_none());
}

static entity gIndEnt = entity_undefined;

static void entity_in_stat_rwt(reference curRef, bool * entInStat)
{
  if(reference_variable(curRef) == gIndEnt)
    {
      *entInStat = true;
    }
}

static bool entity_in_stat(entity ind, statement stat)
{
  bool entInStat = false;
  gIndEnt = ind;

  gen_context_recurse(stat, &entInStat, reference_domain, gen_true,
		      entity_in_stat_rwt);

  return entInStat;
}

static void move_statements_stat(statement stat, entity ind,
				 list * lInBody, list * lOutBody)
{
  if(entity_in_stat(ind, stat) ||
     statement_loop_p(stat))
    {
      *lInBody = gen_nconc(*lInBody, 
			   CONS(STATEMENT, copy_statement(stat), NIL));
    }
  else
    {
      *lOutBody = gen_nconc(*lOutBody,
			    CONS(STATEMENT, copy_statement(stat), NIL));
    }
}

static void move_statements(statement stat, entity ind,
			    list * lInBody, list * lOutBody)
{
  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_sequence:
      {
        MAP(STATEMENT, curStat,
	{
	  move_statements(curStat, ind, lInBody, lOutBody);

	}, sequence_statements(instruction_sequence(instr)));
	break;
      }
    case is_instruction_loop:
    case is_instruction_call:
    case is_instruction_test:
      {
	move_statements_stat(stat, ind, lInBody, lOutBody);
	break;
      }
    default:
      {
	pips_internal_error("impossible");
	break;
      }
    }
}

static entity gSearchedEnt = entity_undefined;
static bool gEntFound = false;

static void entity_in_ref_rwt(reference curRef)
{
  printf("entity_in_ref_rwt\n");
  print_reference(curRef);printf("\n");
  if(reference_variable(curRef) == gSearchedEnt)
    {
      gEntFound = true;
    }
}

static bool entity_in_ref(entity ent, reference ref)
{
  gSearchedEnt = ent;
  gEntFound = false;

  printf("entity_in_ref\n");
  print_entity_variable(ent);
  print_reference(ref);printf("\n");

  gen_recurse(ref, reference_domain, gen_true, entity_in_ref_rwt);

  if(gEntFound)
    {
      printf("found\n");
    }
  else
    {
      printf("not found\n");
    }

  return gEntFound;
}

static void regenerate_toggles(statement stat, statement newStat,
			       list * lInBody)
{
  //printf("regenerate_toggles\n");
  //print_statement(stat);

  list lToggleEnt = hash_get(gLoopToToggleEnt, newStat);

  if(lToggleEnt == HASH_UNDEFINED_VALUE)
    {
      return;
    }

  statement tempStat = make_block_statement(*lInBody);

  list lIncStats = NIL;
  list lMmcdStats = NIL;

  FOREACH(ENTITY, oldTog,lToggleEnt)
  {
    printf("oldTog\n");
    print_entity_variable(oldTog);

    entity newTog =
      comEngine_make_new_scalar_variable(strdup("toggle"),
					 make_basic(is_basic_int, (void *)4));
      AddEntityToCurrentModule(newTog);

    comEngine_replace_reference_in_stat(tempStat,
					make_reference(oldTog, NIL),
					entity_to_expression(newTog));

    intptr_t inc = (intptr_t)hash_get(gToggleToInc, oldTog);

    pips_assert("inc != HASH_UNDEFINED_VALUE",
		inc != (intptr_t)HASH_UNDEFINED_VALUE);

    statement incStat = make_toggle_inc_statement(newTog, inc);

    lIncStats = gen_nconc(lIncStats, CONS(STATEMENT, incStat, NIL));

    list lRef = NIL;
    HASH_MAP(ref1, tog1,
    {
      if((entity)tog1 == oldTog)
	{
	  if(entity_in_ref(loop_index(statement_loop(stat)),
			   ref1))
	    {
	      lRef = CONS(REFERENCE, ref1, lRef);
	    }
	}
    }, gRefToToggle);

    MAP(REFERENCE, ref1,
    {
      hash_put(gRefToToggle, ref1, newTog);
    }, lRef);
    gen_free_list(lRef);

    statement mmcdStat = make_toggle_mmcd(newTog);

    lMmcdStats = gen_nconc(lMmcdStats, CONS(STATEMENT, mmcdStat, NIL));

    statement initStat = make_toggle_init_statement(newTog);

    glToggleInitStats = gen_nconc(glToggleInitStats, CONS(STATEMENT, initStat, NIL));

  } 

  instruction_block(statement_instruction(tempStat)) = NIL;

  free_statement(tempStat);

  *lInBody = gen_nconc(lMmcdStats, *lInBody);
  *lInBody = gen_nconc(*lInBody, lIncStats);
}

static statement make_seqStat(statement stat, statement loopPattern,
			      list oldCurList, entity newInd)
{
  statement tempStat = STATEMENT(CAR(oldCurList));

  void* bIsNewLoop = hash_get(gIsNewLoop, tempStat);

  if(bIsNewLoop == HASH_UNDEFINED_VALUE)
    {
      statement seqStat = copy_statement(loopPattern);

      loop_body(statement_loop(seqStat)) = make_block_statement(oldCurList);

      return seqStat;
    }

  statement saveStat = statement_undefined;

  MAP(STATEMENT, curStat,
  {
    if(statement_loop_p(curStat))
      {
	saveStat = curStat;
      }
  }, statement_block(tempStat));

  pips_assert("saveStat != statement_undefined",
	      saveStat != statement_undefined);

  list curList = gen_full_copy_list(oldCurList);

  tempStat = STATEMENT(CAR(curList));

  statement seqStat = statement_undefined;// = STATEMENT(CAR(curList));

  pips_assert("statement_block_p(tempStat)", statement_block_p(tempStat));

  MAP(STATEMENT, curStat,
  {
    if(statement_loop_p(curStat))
      {
	seqStat = curStat;
      }
  }, statement_block(tempStat));

  pips_assert("seqStat != statement_undefined",
	      seqStat != statement_undefined);

  gen_free_list(curList);

  statement inLoop = copy_statement(loopPattern);

  list lOutBody = NIL;
  list lInBody = NIL;

  move_statements(loop_body(statement_loop(seqStat)),
		  loop_index(statement_loop(loopPattern)),
		  &lInBody, &lOutBody);

  free_statement(loop_body(statement_loop(seqStat)));

  /*statement stepStat =
    make_loop_step_stat(inLoop,
			loop_index(statement_loop(loopPattern)));

  lInBody = gen_nconc(CONS(STATEMENT, stepStat, NIL), lInBody);
  */

  regenerate_toggles(stat, saveStat, &lInBody);

  loop_body(statement_loop(inLoop)) = make_block_statement(lInBody);

  lOutBody = gen_nconc(lOutBody, CONS(STATEMENT, inLoop, NIL));

  loop_body(statement_loop(seqStat)) = make_block_statement(lOutBody);

  return tempStat;
}

statement comEngine_opt_loop_interchange(statement stat, statement innerStat,
					 entity newInd)
{
  printf("comEngine_opt_loop_interchange\n");
  print_statement(innerStat);

  pips_assert("innerStat is a loop statement", statement_loop_p(innerStat));

  statement body = loop_body(statement_loop(innerStat));

  if(!statement_block_p(body))
    {
      return innerStat;
    }

  printf("stat\n");
  print_statement(stat);

  bool allowedOpt = (intptr_t)hash_get(gLoopToOpt, stat);

  pips_assert("allowedOpt != HASH_UNDEFINED_VALUE",
	      allowedOpt != (intptr_t)HASH_UNDEFINED_VALUE);

  if(!allowedOpt)
    {
      printf("opt not all\n");
      return innerStat;
    }
  else
    {
      printf("opt all\n");
    }

  glSeqs = NIL;
  glStats = NIL;

  opt_loop_interchange_fill_lists(body);

  if(glStats != NIL)
    {
	  printf("put glSeqs 3\n");
	  MAP(STATEMENT, curStat,
	  {
	    print_statement(curStat);
	  }, glStats);

      glSeqs = gen_nconc(glSeqs, CONS(LIST, glStats, NIL));

      glStats = NIL;
    }

  list lStats = NIL;
  statement loopPattern = make_loopPattern(innerStat);

  //printf("glSeqs\n");
  MAP(LIST, curList,
  {
  /*
    //printf("glSeqs it 1\n");
    MAP(STATEMENT, curStat,
    {
      //printf("glSeqs it 2\n");
      //print_statement(curStat);
      //printf("glSeqs it 2 end\n");

    }, curList);
    */

    statement seqStat = make_seqStat(stat, loopPattern, curList, newInd);

    lStats = gen_nconc(lStats, CONS(STATEMENT, seqStat, NIL));

  }, glSeqs);

  gen_free_list(glSeqs);

  free_statement(loopPattern);

  printf("bef 2\n");
  loop_body(statement_loop(innerStat)) = statement_undefined;
  printf("aft 2\n");

  free_statement(innerStat);

  statement newStat = make_block_statement(lStats);

  printf("comEngine_opt_loop_interchange newStat\n");
  print_statement(newStat);

  return newStat;
}
