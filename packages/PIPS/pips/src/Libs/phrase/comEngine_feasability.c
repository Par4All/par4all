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
This file is used to analyze the code that we want to export to the
HRE.
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
#include "phrase.h"

static bool check_distribution_feasability(statement stat);

static bool check_for_conflict(reference ref);

// This variable holds the number of
// if we have entered at a given point of
// the algorithm
static int gIfCount = 0;

// This hash_table associate a reference to the
// its list of enclosing loops
static hash_table gRefToEncLoop;

// This list holds the enclosing loops at a given point of 
// the algorithm
static list lLoop;

// This list holds the loop index used in the programm
static list glIndUsed;

// This is a global pointer to the dependence graph
static graph gDg;

// This is a global pointer to the statement of the code
// to export to the HRE
static statement g_externalized_code;

static list local_syntax_to_reference_list(syntax s, list lr);

/* conversion of an expression into a list of references; references are
   appended to list lr as they are encountered; array references are
   added before their index expressions are scanned;

   references to functions and constants (which are encoded as null-ary
   functions) are not recorded 
*/
list comEngine_expression_to_reference_list(e, lr)
expression e;
list lr;
{
    syntax s = expression_syntax(e);
    lr = local_syntax_to_reference_list(s, lr);
    return lr;
}

static list local_syntax_to_reference_list(s, lr)
syntax s;
list lr;
{
    switch(syntax_tag(s)) {
    case is_syntax_reference:
	lr = gen_nconc(lr, CONS(REFERENCE, syntax_reference(s), NIL));
	break;
    case is_syntax_range:
	lr = comEngine_expression_to_reference_list(range_lower(syntax_range(s)), lr);
	lr = comEngine_expression_to_reference_list(range_upper(syntax_range(s)), lr);
	lr = comEngine_expression_to_reference_list(range_increment(syntax_range(s)),
					  lr);
	break;
    case is_syntax_call:
	MAPL(ce, {
	    expression e = EXPRESSION(CAR(ce));
	    lr = comEngine_expression_to_reference_list(e, lr);
	    },
	     call_arguments(syntax_call(s)));
	break;
    default:
	pips_internal_error("illegal tag %d", 
		   syntax_tag(s));

    }
    return lr;
}

/*
This function is used by has_loop_inside().
 */
static void has_loop_inside_rwt(statement stat, bool * bHasLoopInside)
{
  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_loop:
    case is_instruction_forloop:
    case is_instruction_whileloop:
      *bHasLoopInside = true;
      break;
    default:
      break;
    }
}

/*
This function returns true if the statement stat contains at least
one loop.
 */
static bool has_loop_inside(statement stat)
{
  bool bHasLoopInside = false;

  gen_context_recurse(stat, &bHasLoopInside, statement_domain,
		      gen_true, has_loop_inside_rwt);

  return bHasLoopInside;
}

static void fill_gRefToEncLoop(statement stat);

/*
This function is used by fill_gRefToEncLoop().
 */
static void fill_gRefToEncLoop_loop(statement stat)
{
  loop curLoop = statement_loop(stat);

  statement body = loop_body(curLoop);

  lLoop = gen_nconc(lLoop, CONS(STATEMENT, stat, NIL));

  fill_gRefToEncLoop(body);

  gen_remove(&lLoop, stat);
}

/*
This function is used by fill_gRefToEncLoop().
 */
static void fill_gRefToEncLoop_call(statement stat)
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

  MAP(REFERENCE, callRef,
  {
    hash_put(gRefToEncLoop, callRef, gen_copy_seq(lLoop));
  }, lCallRef);

  gen_free_list(lCallRef);
}

/*
This function is used by fill_gRefToEncLoop().
 */
static void fill_gRefToEncLoop_test(statement stat)
{
  test curTest = statement_test(stat);

  list lCondRef = NIL;
  lCondRef = comEngine_expression_to_reference_list(test_condition(curTest), lCondRef);

  MAP(REFERENCE, condRef,
  {
    if(gen_length(reference_indices(condRef)) != 0)
      {
	hash_put(gRefToEncLoop, condRef, gen_copy_seq(lLoop));
      }
  }, lCondRef);

  if(test_true(curTest) != statement_undefined)
    {
      fill_gRefToEncLoop(test_true(curTest));
    }

  if(test_false(curTest) != statement_undefined)
    {
      fill_gRefToEncLoop(test_false(curTest));
    }

  gen_free_list(lCondRef);
}

/*
This function goes through the statement stat and, for each
reference of the statement, it stores in the hash_table 
gRefToEncLoop the list of the enclosing loops.
 */
static void fill_gRefToEncLoop(statement stat)
{
  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_sequence:
      {
        MAP(STATEMENT, curStat,
	{
	  fill_gRefToEncLoop(curStat);

	}, sequence_statements(instruction_sequence(instr)));
	break;
      }
    case is_instruction_loop:
      {
	fill_gRefToEncLoop_loop(stat);
	break;
      }
    case is_instruction_call:
      {
	fill_gRefToEncLoop_call(stat);
	break;
      }
    case is_instruction_test:
      {
	fill_gRefToEncLoop_test(stat);
	break;
      }
    default:
      {
	break;
      }
    }
}

/*
This function returns true if the entity searchedEnt is in the
list of references lRef.
 */
static bool entity_in_ref_list(entity searchedEnt, list lRef)
{
  MAP(REFERENCE, ref,
  {
    entity ent = reference_variable(ref);

    if(same_entity_p(ent, searchedEnt))
      {
	return true;
      }
  }, lRef);

  return false;
}

/*
This function returns looks for a fifo associated to the reference
ref. If it does not exist, a new fifo is created and the association
of the fifo and the reference is stored in the hash_table refToFifo.
 */
static int find_or_create_fifo_from_ref(reference ref, bool bReadAct,
					hash_table refToFifo, int inc)
{
  intptr_t fifoNum = -1;

  // Goes through the hash_table refToFifo to see if
  // a fifo is already associated with the reference ref.
  HASH_MAP(curRef,
	   buff,
  {
    if(reference_equal_p(curRef, ref))
      {
	fifoNum = (intptr_t)buff;
	break;
      }
  }, refToFifo);

  // If no fifo is associated to ref, then create one
  if(fifoNum == -1)
    {
      //printf("create %d\n", (int) fifoNum);
      //print_reference(ref);printf("\n");
      static int numFifo = 1;

      fifoNum = numFifo;
      numFifo += inc;

      hash_put(refToFifo, ref, (void *)fifoNum);
    }
  // else a fifo already exists, so just store the value
  // in refToFifo
  else
    {
      //printf("put %d\n", (int) fifoNum);
      //print_reference(ref);printf("\n");

      hash_put(refToFifo, ref, (void *)fifoNum);
    }

  //printf("real fifoNum %d\n", fifoNum);

  return fifoNum;
}

/*
This function calls find_or_create_fifo_from_ref() to find or create
a fifo to associate to the reference ref. Then, the function stores the 
association between the found or created fifo and the reference ref in
gRefToHREFifo.
 */
static void compute_fifo_from_ref(reference ref, bool bReadAct, int inc)
{
  intptr_t fifoNum = 0;

  fifoNum = find_or_create_fifo_from_ref(ref, bReadAct, gRefToFifo, inc);

  hash_put(gRefToHREFifo, ref, (void *)fifoNum);
}

/*
This function associates the reference callRef with the statement
stat by storing an association in pTable. pTable can be equal to
gStatToRef or to gLoopToRef.
 */
static void attach_ref_to_stat(reference callRef, statement stat,
			       hash_table pTable,
			       bool firstRef, int inc)
{
  list lRef = hash_get(pTable, stat);

  // If no association is in the hash_table pTable, then
  // initialize the list that will contains the associations
  if(lRef == HASH_UNDEFINED_VALUE)
    {
      lRef = NIL;
    }

  // Go through the list of association for this statement
  // to see if the reference is already associated to this statement or not
  bool alreadyAttached = false;
  MAP(REFERENCE, curRef,
  {
    if(!reference_equal_p(callRef, curRef))
      continue;

    string effAction1 = hash_get(gRefToEff, (reference)curRef);

    pips_assert("effAction1 != HASH_UNDEFINED_VALUE",
		effAction1 != HASH_UNDEFINED_VALUE);

    if((!strcmp(effAction1, R_EFFECT) && !firstRef) ||
       (!strcmp(effAction1, W_EFFECT) && firstRef))
      {
	alreadyAttached = true;
      }
  }, lRef);

  //printf("put %d\n", inc);
  //print_reference(callRef);
  //printf("\n");

  // If the reference is not yet associated to the statement,
  // then do it.
  if(!alreadyAttached)
    {
      lRef = gen_nconc(lRef, CONS(REFERENCE, callRef, NIL));
      hash_put(pTable, stat, lRef);
    }

  // Store the effect of the reference in gRefToEff
  if(firstRef)
    {
      hash_put(gRefToEff, callRef, W_EFFECT);
    }
  else
    {
      hash_put(gRefToEff, callRef, R_EFFECT);
    }

  // Create or find the fifo associated to the reference callRef
  compute_fifo_from_ref(callRef, !firstRef, inc);
}

/*
This function performs the analysis needed for a loop statement
 */
static bool check_loop_distribution_feasability(statement stat)
{
  bool success = true;

  loop curLoop = statement_loop(stat);

  entity index = loop_index(curLoop);
  range lpRange = loop_range(curLoop);
  statement body = loop_body(curLoop);

  expression rgLower = range_lower(lpRange);
  expression rgUpper = range_upper(lpRange);
  expression rgIncr = range_increment(lpRange);

  list lLowerRef = NIL;
  list lUpperRef = NIL;
  list lIncrRef = NIL;

  lLowerRef = expression_to_reference_list(rgLower, lLowerRef);
  lUpperRef = expression_to_reference_list(rgUpper, lUpperRef);
  lIncrRef = expression_to_reference_list(rgIncr, lIncrRef);

  list bodyEff = load_cumulated_rw_effects_list(body);

  // The same index cannot be used in several loops
  if(gen_in_list_p(index, glIndUsed))
    {
      pips_internal_error("loop index used in several loops: %s",
		 entity_user_name(index));
      return false;
    }
  glIndUsed = CONS(ENTITY, index, glIndUsed);

  // A reference that is used in range_lower, range_upper or 
  // range_increment of a loop cannot be written in the body of the loop
  MAP(EFFECT, eff,
  {
    if(action_write_p(effect_action(eff)))
      {
	if(same_entity_p(index, effect_entity(eff)))
	  {
	    success = false;
	    break;
	  }

	if(entity_in_ref_list(effect_entity(eff), lLowerRef) ||
	   entity_in_ref_list(effect_entity(eff), lUpperRef) ||
	   entity_in_ref_list(effect_entity(eff), lIncrRef))
	  {
	    success = false;
	    break;
	  }
      }
  }, bodyEff);

  if(!same_expression_p(rgIncr, int_to_expression(1)) ||
     !same_expression_p(rgLower, int_to_expression(0)))
    {
      success = false;

      gen_free_list(lLowerRef);
      gen_free_list(lUpperRef);
      gen_free_list(lIncrRef);

      return success;
    }

  MAP(REFERENCE, curRef,
  {
    // If a reference, used in the range_lower of a loop,
    // is an array or has a written effect in the code to export,
    // then the exportation is not feasible
    if((gen_length(reference_indices(curRef)) != 0) ||
       code_has_write_eff_ref_p(curRef, g_externalized_code))
      {
	success = false;
	break;
      }

    attach_ref_to_stat(curRef, g_externalized_code, gStatToRef,
		       false, 1);

  }, lLowerRef);

  MAP(REFERENCE, curRef,
  {
    // If a reference, used in the range_upper of a loop,
    // is an array or has a written effect in the code to export,
    // then the exportation is not feasible
    if((gen_length(reference_indices(curRef)) != 0) ||
       code_has_write_eff_ref_p(curRef, g_externalized_code))
      {
	success = false;
	break;
      }

    attach_ref_to_stat(curRef, g_externalized_code, gStatToRef,
		       false, 1);

  }, lUpperRef);

  gen_free_list(lLowerRef);
  gen_free_list(lUpperRef);
  gen_free_list(lIncrRef);

  // Add the current loop to the loop list
  lLoop = CONS(STATEMENT, stat, lLoop);

  success = check_distribution_feasability(body);

  // Remove the current loop from the loop list
  gen_remove(&lLoop, stat);

  return success;
}

/*
This function returns true if the reference has a write effect in
the statement stat
 */
bool code_has_write_eff_ref_p(reference ref, statement stat)
{
   bool actionWrite = false;

   MAP(EFFECT, f, 
   {
      bool effEntIsIndex = false;
      entity effEnt = effect_entity(f);

      MAP(STATEMENT,loopStat,
      {
	entity index = loop_index(statement_loop(loopStat));

	if(same_entity_p(index, effEnt))
	  {
	    effEntIsIndex = true;
	  }

      }, lLoop);

      if(effEntIsIndex)
	{
	  continue;
	}

      if(action_write_p(effect_action(f)) && 
	 same_entity_p(reference_variable(ref), effEnt))
	{
	  actionWrite = true;
	}

   }, load_cumulated_rw_effects_list(stat));

   return actionWrite;
}

/*
This function returns a list containing the reference used in
the indices of the reference ref.
 */
list reference_indices_entity_list(reference ref)
{
  list lIndRef = NIL;

  MAP(EXPRESSION, index,
  {
    list old = lIndRef;
    list new = NIL;
    new = comEngine_expression_to_reference_list(index, new);

    lIndRef = gen_concatenate(old, new);

    gen_free_list(old);
    gen_free_list(new);

  }, reference_indices(ref));

  return lIndRef;
}

#define LESSER_DIRECTION 4
#define GREATER_DIRECTION 1
#define ZERO_DIRECTION 2
#define ANY_DIRECTION 7
#define NO_DIRECTION 0

static int vertex_to_direction[3] = {LESSER_DIRECTION, ZERO_DIRECTION, GREATER_DIRECTION};

static int ray_to_direction[3] = {LESSER_DIRECTION, NO_DIRECTION, GREATER_DIRECTION};

static int line_to_direction[3] = {ANY_DIRECTION, NO_DIRECTION, ANY_DIRECTION};

static char * direction_to_representation[8] = {"?", "<", "=", "<=", ">", "*", ">=", "*"};

/*
This function returns true if the conflict whose sg is the system is not a
real dependence
 */
static bool is_good_direction_p(Ptsg sg, int loopLev, bool bReadIsSource)
{
    Pbase c = BASE_NULLE;
    int ddv = 0;

    int counter = 1;
    string dir = "";

    /* For each coordinate */
    for(c=sg->base; !BASE_NULLE_P(c); c = c->succ) {
	Psommet v;
	Pray_dte r;
	Pray_dte l;

	/* For all vertices */
	for(v=sg_sommets(sg); v!= NULL; v = v->succ) {
	    ddv |= vertex_to_direction[1+value_sign(vect_coeff(vecteur_var(c), v->vecteur))];
	}

	/* For all rays */
	for(r=sg_rayons(sg); r!= NULL; r = r->succ) {
	    ddv |= ray_to_direction[1+value_sign(vect_coeff(vecteur_var(c), r->vecteur))];
	}

	/* For all lines */
	for(l=sg_droites(sg); l!= NULL; l = l->succ) {
	    ddv |= line_to_direction[1+value_sign(vect_coeff(vecteur_var(c), l->vecteur))];
	}

	if(counter == loopLev)
	  {
	    dir = direction_to_representation[ddv];
	    break;
	  }

	counter++;
    }

    if(((!strcmp(dir, "<")) ||
	(!strcmp(dir, "<="))) &&
       bReadIsSource)
      {
	return true;
      }
    else if(((!strcmp(dir, ">")) ||
	     (!strcmp(dir, ">="))) &&
	    !bReadIsSource)
      {
	return true;
      }
    else if(!strcmp(dir, "="))
      {
	return false;
      }

    return false;
}

/*
This function finds the first common loop between the loop lists
n1 and n2.
 */
static statement find_common_loop(list n1, list n2, list * lWLoops)
{
  statement comLoop = statement_undefined;

  statement lastLoop = statement_undefined;

  while (n1 != NIL && n2 != NIL) {
    if (STATEMENT(CAR(n1)) != STATEMENT(CAR(n2)))
      {
	break;
      }
    lastLoop = STATEMENT(CAR(n1));
    n1 = CDR(n1); 
    n2 = CDR(n2);
  }

  while(n2 != NIL)
    {
      *lWLoops = gen_nconc(*lWLoops, CONS(STATEMENT, STATEMENT(CAR(n2)), NIL));
      n2 = CDR(n2);
    }


  comLoop = lastLoop;

  return comLoop;
}

/*
This function returns true if there is a dependence that prevents the pipeline
 */
static bool check_for_conflict(reference ref)
{
  //printf("check_for_conflict:\n");
  //print_reference(ref);printf("\n");
  MAP(VERTEX, a_vertex, 
  {
    MAP(SUCCESSOR, suc,
    {
      MAP(CONFLICT, c, 
      {
	effect sourceEff = conflict_source(c);
	effect sinkEff = conflict_sink(c);

	// If this is a read-read or a write-write dependence,
	// then there is no problem
	if((effect_read_p(sourceEff) &&
	    effect_read_p(sinkEff)) ||
	   (effect_write_p(sourceEff) &&
	    effect_write_p(sinkEff)))
	  continue;

	reference readRef = effect_any_reference(conflict_source(c));
	reference writeRef = effect_any_reference(conflict_sink(c));

	if(effect_read_p(sourceEff))
	  {
	    readRef = effect_any_reference(sourceEff);
	    writeRef = effect_any_reference(sinkEff);
	  }
	else
	  {
	    writeRef = effect_any_reference(sourceEff);
	    readRef = effect_any_reference(sinkEff);
	  }

	// If this conflict does not involve ref,
	// let us not consider this conflict
	if((readRef != ref) &&
	   (writeRef != ref))
	  continue;

	// If there is a conflict but that the references are
	// equal then the conflict does not prevent the pipeline
	if(reference_equal_p(readRef, writeRef))
	  continue;

	list lReadLoop = hash_get(gRefToEncLoop, readRef);
	list lWriteLoop = hash_get(gRefToEncLoop, writeRef);

	pips_assert("lReadLoop != HASH_UNDEFINED_VALUE",
		    lReadLoop != HASH_UNDEFINED_VALUE);

	pips_assert("lWriteLoop != HASH_UNDEFINED_VALUE",
		    lWriteLoop != HASH_UNDEFINED_VALUE);

	/*printf("conf\n");
	printf("read\n");
	print_reference(readRef);printf("\n");
	printf("write\n");
	print_reference(writeRef);printf("\n");
	*/
	//print_reference(effect_any_reference(sourceEff));printf("\n");
	//print_reference(effect_any_reference(sinkEff));printf("\n");

	// If the conflict cone is not undefined, let us study it.
	if(conflict_cone(c) != cone_undefined)
	  {
	    int maxLevel = -1;

	    MAPL(pl,
            {
		if(maxLevel == -1)
		  {
		    maxLevel = INT(CAR(pl));
		  }
		else if(INT(CAR(pl)) > maxLevel)
		  {
		    maxLevel = INT(CAR(pl));
		  }

	    }, cone_levels(conflict_cone(c)));

	    pips_assert("maxLevel != -1", maxLevel != -1);

	    Ptsg gs = (Ptsg)cone_generating_system(conflict_cone(c));
	    if (!SG_UNDEFINED_P(gs))
	      {
		// If the conflict does not prevent the pipeline then continue
		if(is_good_direction_p(gs, maxLevel,
				       (readRef == effect_any_reference(sourceEff))))
		  {
		    sg_fprint_as_ddv(stdout, gs);
		    continue;
		  }
	      }
	  }

	statement lastRLoop = statement_undefined;
	statement lastWLoop = statement_undefined;

	if(lReadLoop != NIL)
	  {
	    lastRLoop = STATEMENT(CAR(gen_last(lReadLoop)));
	  }

	if(lWriteLoop != NIL)
	  {
	    lastWLoop = STATEMENT(CAR(gen_last(lWriteLoop)));
	  }

	// If there is a conflict but that the references invloved are in to 
	// different nested loops, then let us consider that there is no conflicts
	// but let us put a bubble in the pipeline.
	if(lastRLoop != lastWLoop)
	  {
	    list lWLoops = NIL;

	    statement comLoop = find_common_loop(lReadLoop, lWriteLoop, &lWLoops);

	    if((comLoop != lastRLoop) &&
	       (comLoop != lastWLoop))
	      {
		pips_assert("lWLoops != NIL", lWLoops != NIL);

		// Let us put a bubble in the pipeline
		hash_put(gLoopToSync, STATEMENT(CAR(lWLoops)), (void *)true);

		gen_free_list(lWLoops);
		continue;
	      }

	    gen_free_list(lWLoops);
	  }

	// If we got here, it means there is a true conflict.
	return true;

      }, dg_arc_label_conflicts(successor_arc_label(suc)));

    }, vertex_successors(a_vertex));

  },graph_vertices(gDg));

  return false;
}

/*
This function tries to attach the reference callRef to the most outer
loop possible.
 */
static bool attach_ref_to_loop(reference callRef, statement inStat,
			       bool firstRef, list lIndRef)
{
  //printf("callRef\n");
  //print_reference(callRef);printf("\n");

  MAPL(lLoopStat,
  {
    statement loopStat = STATEMENT(CAR(lLoopStat));

    bool attachToThisLoop = false;
    entity loopIndex = loop_index(statement_loop(loopStat));

    // If the indices of the reference callRef contains the index
    // of the loop loopStat, then let us try to attach the reference
    // to this loop.
    MAP(REFERENCE, indRef,
    {
      if(same_entity_p(loopIndex, reference_variable(indRef)))
	{
	  attachToThisLoop = true;
	  break;
	}

    }, lIndRef);

    // If callRef is involved in a data dependency that prevents
    // the pipeline, then let us attach the reference to the
    // stat where it is used and not to a loop
    if(check_for_conflict(callRef))
      {
	attach_ref_to_stat(callRef, inStat, gStatToRef,
			   firstRef, 1);

	if(gIfCount != 0)
	  {
	    pips_internal_error("conflict with reference: %s",
		       words_to_string(words_reference(callRef, NIL)));

	    return false;
	  }
	break;
      }

    // If the reference must be attached to this loop, let us do this.
    if(attachToThisLoop)
      {
	attach_ref_to_stat(callRef, loopStat, gLoopToRef,
			   firstRef, 1);

	break;
      }

    if(loopStat == STATEMENT(CAR(gen_last(lLoop))))
      {
	attach_ref_to_stat(callRef, STATEMENT(CAR(gen_last(lLoop))), gStatToRef,
			   firstRef, 1);
	break;
      }

  }, lLoop);

  return true;
}

/*
This function associates the references holded by lCallRef to the suitable
statements.
 */
static bool process_ref_list(list lCallRef, bool firstRef, statement inStat)
{
  list lIndRef = NIL;

  MAP(REFERENCE, callRef,
  {

    if(gen_length(reference_indices(callRef)) == 0)
      {
	// If a loop index is used in some call or test statement,
	// then the compilation is impossible
	MAP(STATEMENT,loopStat,
	{
	  if(same_entity_p(loop_index(statement_loop(loopStat)),
			   reference_variable(callRef)))
	    {
	      pips_internal_error("loop index must not vary in the code: %s",
			 words_to_string(words_reference(callRef, NIL)));

	      return false;
	    }

	}, lLoop);

	lIndRef = NIL;

	// Attach callRef to the suitable statement
	if(lLoop == NIL)
	  {
	    attach_ref_to_stat(callRef, g_externalized_code, gStatToRef,
			       firstRef, 1);
	  }
	else
	  {
	    if(!attach_ref_to_loop(callRef, inStat, firstRef, lIndRef))
	      {
		gen_free_list(lIndRef);
		return false;
	      }
	  }

	firstRef = false;
	continue;
      }

    /*printf("call ref\n");
    print_reference(callRef);
    printf("\n");
    */
    lIndRef = reference_indices_entity_list(callRef);

    MAP(REFERENCE, indRef,
    {
      // If one reference contained in callRef indices is an
      // array reference or that the reference is written in
      // the code to compile, then the compilation is impossible
      if((gen_length(reference_indices(indRef)) != 0) ||
	 code_has_write_eff_ref_p(indRef, g_externalized_code))
	{
	  gen_free_list(lIndRef);

	  pips_internal_error("%s is not a valid reference",
		     words_to_string(words_reference(callRef, NIL)));

	  return false;
	}

    }, lIndRef);

    // Attach callRef to the suitable statement
    if(lLoop == NIL)
      {
	attach_ref_to_stat(callRef, g_externalized_code, gStatToRef,
			   firstRef, 1);
      }
    else
      {
	if(!attach_ref_to_loop(callRef, inStat, firstRef, lIndRef))
	  {
	    gen_free_list(lIndRef);
	    return false;
	  }
      }

    gen_free_list(lIndRef);

    firstRef = false;

  }, lCallRef);

  return true;
}

/*
This function performs the analysis needed for a call statement
 */
static bool check_call_distribution_feasability(statement inStat)
{
  list lCallRef = NIL;

  call curCall = instruction_call(statement_instruction(inStat));

  bool firstRef = true;

  MAP(EXPRESSION, exp,
  {
    list old = lCallRef;
    list new = NIL;
    new = comEngine_expression_to_reference_list(exp, new);

    lCallRef = gen_concatenate(old, new);

    gen_free_list(old);
    gen_free_list(new);

  }, call_arguments(curCall));

  bool success = process_ref_list(lCallRef, firstRef, inStat);

  gen_free_list(lCallRef);

  return success;
}

/*
This function performs the analysis needed for a test statement
 */
static bool check_test_distribution_feasability(statement inStat)
{
  test curTest = statement_test(inStat);

  // If the test statement has at least one loop inside, then
  // the compilation is impossbile.
  if(has_loop_inside(inStat))
    {
      return false;
    }

  list lCondRef = NIL;
  lCondRef = comEngine_expression_to_reference_list(test_condition(curTest), lCondRef);

  bool firstRef = false;

  // Process the condition
  if(!process_ref_list(lCondRef, firstRef, inStat))
    {
      return false;
    }

  gen_free_list(lCondRef);

  gIfCount++;

  // Process the true statement 
  if(test_true(curTest) != statement_undefined)
    {
      if(!check_distribution_feasability(test_true(curTest)))
	{
	  return false;
	}
    }

  // Process the false statement 
  if(test_false(curTest) != statement_undefined)
    {
      if(!check_distribution_feasability(test_false(curTest)))
	{
	  return false;
	}
    }

  gIfCount--;

  return true;
}

/*
This function analyzes the statement stat.
 */
static bool check_distribution_feasability(statement stat)
{
  bool success = true;
  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_sequence:
      {
        MAP(STATEMENT, curStat,
	{
	  success &= check_distribution_feasability(curStat);

	}, sequence_statements(instruction_sequence(instr)));
	break;
      }
    case is_instruction_loop:
      {
	success = check_loop_distribution_feasability(stat);
	break;
      }
    case is_instruction_call:
      {
	success = check_call_distribution_feasability(stat);
	break;
      }
    case is_instruction_test:
      {
	success = check_test_distribution_feasability(stat);
	break;
      }
    default:
      {
	success = false;
	break;
      }
    }

  return success;
}

/*
This function performes the analyzes needed to export the code externalized_code
to the HRE
 */
bool comEngine_feasability(statement externalized_code, graph dg)
{
  bool success = false;

  // Initialize some global variables
  gDg = dg;
  g_externalized_code = externalized_code;
  gIfCount = 0;
  gRefToEncLoop = hash_table_make(hash_pointer, 0);
  lLoop = NIL;
  glIndUsed = NIL;

  hash_dont_warn_on_redefinition();

  // Fill the hash_table gRefToEncLoop
  fill_gRefToEncLoop(externalized_code);

  pips_assert("lLoop == NIL", lLoop == NIL);

  // Perform the actual analysis
  success = check_distribution_feasability(externalized_code);

  hash_warn_on_redefinition();

  // Free some global variables
  HASH_MAP(curRef, curLoopList,
  {
    gen_free_list(curLoopList);
  }, gRefToEncLoop);
  hash_table_free(gRefToEncLoop);
  gen_free_list(glIndUsed);

  return success;
}
