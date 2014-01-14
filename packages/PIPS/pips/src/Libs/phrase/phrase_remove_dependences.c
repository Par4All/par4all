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
#include "pipsdbm.h"

#include "text-util.h"


#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "control.h"

#include "phrase_tools.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "phrase_distribution.h"
#include "comEngine.h"
#include "phrase.h"

static hash_table gRefTolRef;
static entity gIndex;
static statement gBody;

static list gLConflicts;

static bool vect_same_variables_p(Pvecteur v1, Pvecteur v2)
{
    Pvecteur ev;

    bool sameVars = true;

    for(ev = v1; !VECTEUR_NUL_P(ev); ev = ev->succ)
      {
	if(vecteur_var(ev) == TCST)
	  continue;

	if(!vect_contains_variable_p(v2, vecteur_var(ev)))
	  sameVars = false;
      }

    for(ev = v2; !VECTEUR_NUL_P(ev); ev = ev->succ)
      {
	if(vecteur_var(ev) == TCST)
	  continue;

	if(!vect_contains_variable_p(v1, vecteur_var(ev)))
	  sameVars = false;
      }

    return sameVars;
}

static bool loop_index_in_several_indices(entity index, reference ref)
{
  bool loopIndFound = false;

  MAP(EXPRESSION, ind,
  {
    normalized norm = NORMALIZE_EXPRESSION(ind);

    if(normalized_linear_p(norm))
      {
	Pvecteur vect = normalized_linear(norm);

	int coeff = vect_coeff((Variable) index, vect);

	if((coeff != 0) &&
	   (!loopIndFound))
	  {
	    loopIndFound = true;
	  }
	else if((coeff != 0) &&
		loopIndFound)
	  {
	    return true;
	  }

      }
    else
      {
	return true;
      }

  }, reference_indices(ref));

  return false;
}

static int get_const_off(entity index, reference ref)
{
  MAP(EXPRESSION, ind,
  {
    normalized norm = NORMALIZE_EXPRESSION(ind);

    if(normalized_linear_p(norm))
      {
	Pvecteur vect = normalized_linear(norm);

	int coeff = vect_coeff((Variable) index, vect);

	if(coeff != 0)
	  {
	    return vect_coeff((Variable) TCST, vect);
	  }

      }

  }, reference_indices(ref));

  return 0;
}

static int get_const_diff(Pvecteur vect1, Pvecteur vect2)
{
  int coeff1 = vect_coeff((Variable) TCST, vect1);
  int coeff2 = vect_coeff((Variable) TCST, vect2);

  return (coeff2 - coeff1);
}

static bool write_conf_on_ref(reference ref)
{
	printf("ref\n");
	print_reference(ref);printf("\n");
  MAP(CONFLICT, conf,
  {
    effect sourceEff = conflict_source(conf);
    effect sinkEff = conflict_sink(conf);

    reference sourceRef = effect_any_reference(sourceEff);
    reference sinkRef = effect_any_reference(sinkEff);

    if((effect_write_p(sourceEff) &&
	sourceRef == ref) ||
       (effect_write_p(sinkEff) &&
	sinkRef == ref))
      {
	printf("write ref\n");
	return true;
      }
        
  }, gLConflicts);

  printf("not write ref\n");
  return false;
}

static void phrase_check_reference(reference ref)
{
  if(gen_length(reference_indices(ref)) == 0)
    {
      return;
    }

  bool bRefPut = false;

  list lIndRef = reference_indices_entity_list(ref);

  MAP(REFERENCE, indRef,
  {
    if(code_has_write_eff_ref_p(indRef, gBody))
      {
	gen_free_list(lIndRef);
	return;
      }

  }, lIndRef);

  gen_free_list(lIndRef);

  if(loop_index_in_several_indices(gIndex, ref))
    {
      return;
    }

  bool bReplaceKeyRef = false;

  HASH_MAP(par1, par2,
  {
    reference keyRef = (reference) par1;
    list lRef = (list) par2;

    if(!same_entity_p(reference_variable(keyRef),
		      reference_variable(ref)))
      {
	continue;
      }

    bool success = true;

    list pInd2 = reference_indices(ref);
    expression ind2;

    MAP(EXPRESSION, ind1,
    {
      ind2 = EXPRESSION(CAR(pInd2));

      normalized norm1 = NORMALIZE_EXPRESSION(ind1);
      normalized norm2 = NORMALIZE_EXPRESSION(ind2);

      if(normalized_linear_p(norm1) &&
	 normalized_linear_p(norm2))
	{
	  Pvecteur vect1 = normalized_linear(norm1);
	  Pvecteur vect2 = normalized_linear(norm2);

	  if(vect_same_variables_p(vect1, vect2))
	    {
	      int diff = get_const_diff(vect1, vect2);

	      if(diff > 0)
		{
		  bReplaceKeyRef = true;
		}

	      if((diff >= 10) ||
		 (diff <= -10))
		{
		  success = false;
		  break;
		}
	    }
	  else
	    {
	      success = false;
	      break;
	    }
	}
      else
	{
	  success = false;
	  break;
	}

      pInd2 = CDR(pInd2);

    }, reference_indices(keyRef));

    if(success)
      {
	int refOff = get_const_off(gIndex, ref);

	bool refPut = false;

	MAP(REFERENCE, curRef,
	{
	  int curOff = get_const_off(gIndex, curRef);

	  if(refOff > curOff)
	    {
	      lRef = gen_insert_before(ref, curRef, lRef);
	      refPut = true;
	      break;
	    }

	}, lRef);

	if(!refPut)
	  {
	    lRef = gen_nconc(lRef, CONS(REFERENCE, ref, NIL));
	  }

	if(!bReplaceKeyRef)
	  {
	    hash_put(gRefTolRef, keyRef, lRef);
	  }
	else
	  {
	    hash_del(gRefTolRef, keyRef);

	    hash_put(gRefTolRef, ref, lRef);
	  }

	bRefPut = true;

	break;
      }

  }, gRefTolRef);

  if(!bRefPut)
    {
      hash_put(gRefTolRef, ref, CONS(REFERENCE, ref, NIL));
    }
}

static list create_new_ent_list(int minOff, int maxOff,
				entity oldEnt)
{
  list lNewEnt = NIL;
  int i;

  for(i = minOff; i <= maxOff; i++)
    {
      entity keyEnt = oldEnt;

      entity newEnt =
	make_new_scalar_variable_with_prefix(entity_local_name(keyEnt),
					     get_current_module_entity(),
					     entity_basic(keyEnt));
      AddEntityToCurrentModule(newEnt);

      lNewEnt = gen_nconc(lNewEnt, CONS(ENTITY, newEnt, NIL));
    }

  return lNewEnt;
}

static void replace_ref_by_ent(list lRef, list lNewEnt,
			       entity index, int max)
{
  list pCurRef = lRef;
  int count = max;

  MAP(ENTITY, newEnt,
  {
    reference curRef = REFERENCE(CAR(pCurRef));

    int curOff = get_const_off(index, curRef);

    if(count == curOff)
      {
	MAPL(pSameRef,
	{
	  reference sameRef = REFERENCE(CAR(pSameRef));

	  int sameOff = get_const_off(index, sameRef);

	  if(sameOff != curOff)
	    {
	      pCurRef = pSameRef;
	      break;
	    }

	  gen_free_list(reference_indices(sameRef));

	  reference_indices(sameRef) = NIL;

	  reference_variable(sameRef) = newEnt;

	}, pCurRef);

      }

    count--;
  }, lNewEnt);
}

static list make_lInitStats(list lInitStats, reference maxOffRef, list lNewEnt,
			    entity loopInd, expression indInit)
{
  int count = 0;

  printf("maxOffRef\n");
  print_reference(maxOffRef); printf("\n");

  MAP(ENTITY, newEnt,
  {
    statement newStat = make_assign_statement(entity_to_expression(newEnt),
					      reference_to_expression(copy_reference(maxOffRef)));

    list addArg = gen_make_list(expression_domain,
				int_to_expression(count),
				copy_expression(indInit),
				NULL);

    expression addExp = call_to_expression(make_call(entity_intrinsic(PLUS_OPERATOR_NAME),
						     addArg));

    comEngine_replace_reference_in_stat(newStat,
					make_reference(loopInd, NIL),
					addExp);

    print_statement(newStat);

    lInitStats = gen_nconc(lInitStats, CONS(STATEMENT, newStat, NIL));

    count--;
  }, lNewEnt);

  return lInitStats;
}

static list make_lSwitchStats(list lSwitchStats, reference maxOffRef, list lNewEnt,
			      int diff, entity loopInd)
{
  int i;

  printf("maxOffRef\n");
  print_reference(maxOffRef); printf("\n");

  for(i = diff; i > 0; i--)
    {
      entity newEnt1 = ENTITY(gen_nth(i, lNewEnt));
      entity newEnt2 = ENTITY(gen_nth(i - 1, lNewEnt));

      statement newStat = make_assign_statement(entity_to_expression(newEnt1),
						entity_to_expression(newEnt2));

      print_statement(newStat);

      lSwitchStats = gen_nconc(lSwitchStats, CONS(STATEMENT, newStat, NIL));

    }

  statement newStat = make_assign_statement(entity_to_expression(ENTITY(gen_nth(0, lNewEnt))),
					    reference_to_expression(copy_reference(maxOffRef)));

  list addArg = gen_make_list(expression_domain,
			      entity_to_expression(loopInd),
			      int_to_expression(1),
			      NULL);

  expression addExp = call_to_expression(make_call(entity_intrinsic(PLUS_OPERATOR_NAME),
						   addArg));

  comEngine_replace_reference_in_stat(newStat,
				      make_reference(loopInd, NIL),
				      addExp);

  print_statement(newStat);

  lSwitchStats = gen_nconc(lSwitchStats, CONS(STATEMENT, newStat, NIL));

  return lSwitchStats;
}

static list remove_write_ref(list lRef)
{
  list lOut = NIL;

  MAP(REFERENCE, curRef,
  {
    if(write_conf_on_ref(curRef))
      {
	continue;
      }
    printf("adddddddddddddddddddddddddddddd\n");
    print_reference(curRef);printf("\n");
    lOut = gen_nconc(lOut, CONS(REFERENCE, curRef, NIL));

  }, lRef);
  printf("remove_write_ref done\n");
  return lOut;
}

static void phrase_remove_dependences_rwt(statement stat)
{
  print_statement(stat);
  if((statement_comments(stat) != string_undefined) &&
     (statement_comments(stat) != NULL))
    {
      printf("%s", statement_comments(stat));
    }
  if(!statement_loop_p(stat))
    {
      return;
    }

  printf("loop\n");
  print_statement(stat);

  gRefTolRef = hash_table_make(hash_pointer, 0);

  loop curLoop = statement_loop(stat);

  entity loopInd = loop_index(curLoop);
  statement body = loop_body(curLoop);
  expression indInit = range_lower(loop_range(curLoop));

  gIndex = loopInd;
  gBody = body;

  gen_recurse(body, reference_domain, gen_true, phrase_check_reference);

  list lInitStats = NIL;
  list lSwitchStats = NIL;

  HASH_MAP(par1, par2,
  {
    reference keyRef = (reference) par1;
    list lRef = (list) par2;

    list lNewEnt = NIL;

    int maxBefOff = get_const_off(loopInd, REFERENCE(CAR(lRef)));

    lRef = remove_write_ref(lRef);

    if(lRef == NIL)
      {
	continue;
      }

    int maxOff = get_const_off(loopInd, REFERENCE(CAR(lRef)));

    int minOff = get_const_off(loopInd,
			       REFERENCE(CAR(gen_last(lRef))));

    if(minOff == maxBefOff)
      {
	gen_free_list(lRef);
	continue;
      }

    reference saveRef = copy_reference(REFERENCE(CAR(lRef)));

    printf("maxOff: %d\n", maxOff);
    printf("minOff: %d\n", minOff);

    MAP(REFERENCE, curRef,
    {
      print_reference(curRef);printf("\n");
    }, lRef);
    printf("end\n");
    lNewEnt = create_new_ent_list(minOff, maxOff,
				  reference_variable(keyRef));

    replace_ref_by_ent(lRef, lNewEnt, loopInd, maxOff);

    lInitStats = make_lInitStats(lInitStats, saveRef, lNewEnt,
				 loopInd, indInit);

    lSwitchStats = make_lSwitchStats(lSwitchStats, saveRef, lNewEnt,
				     (maxOff - minOff), loopInd);

    gen_free_list(lRef);
    
  }, gRefTolRef);

  HASH_MAP(par1, par2,
  {
    list lRef = (list) par2;

    gen_free_list(lRef);

  }, gRefTolRef);

  hash_table_free(gRefTolRef);

  statement newBody = make_block_statement(gen_nconc(CONS(STATEMENT, body, NIL),
						     lSwitchStats));

  loop_body(curLoop) = newBody;

  list lStats = gen_nconc(lInitStats, CONS(STATEMENT, copy_statement(stat), NIL));

  instruction newInstr = make_instruction_block(lStats);

  free_instruction(statement_instruction(stat));

  if(statement_comments(stat) != string_undefined)
    {
      free(statement_comments(stat));
      statement_comments(stat) = string_undefined;
    }

  statement_instruction(stat) = newInstr;
}

bool phrase_remove_dependences(const char* module_name)
{
  statement module_stat;

  /* Get the code of the module. */
  set_current_module_entity(module_name_to_entity(module_name));
  set_current_module_statement( (statement)
				db_get_memory_resource(DBR_CODE, module_name, true) );
  module_stat = get_current_module_statement();
  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));

  debug_on("SIMD_LOOP_CONST_ELIM_SCALAR_EXPANSION_DEBUG_LEVEL");

  graph dg = (graph) db_get_memory_resource(DBR_DG, module_name, true);

  gLConflicts = NIL;

  MAP(VERTEX, a_vertex, 
  {
    MAP(SUCCESSOR, suc,
    {
      MAP(CONFLICT, c, 
      {

	gLConflicts = CONS(CONFLICT, c, gLConflicts);

      }, dg_arc_label_conflicts(successor_arc_label(suc)));

    }, vertex_successors(a_vertex));

  },graph_vertices(dg));

  hash_dont_warn_on_redefinition();

  // Go through all the statements
  gen_recurse(module_stat, statement_domain,
	      gen_true, phrase_remove_dependences_rwt);
  
  hash_warn_on_redefinition();

  pips_assert("Statement is consistent after SIMD_SCALAR_EXPANSION", 
	      statement_consistent_p(module_stat));

  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
 
  debug_off();
    
  reset_current_module_entity();
  reset_current_module_statement();
  reset_cumulated_rw_effects();

  return true;
}
