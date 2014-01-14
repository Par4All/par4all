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
#include "properties.h"
#include "prettyprint.h"

#include "transformer.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
#include "semantics.h"
#include "control.h"
#include "callgraph.h"

#include "phrase_tools.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "effects-convex.h"

#include "phrase_distribution.h"
#include "comEngine.h"
#include "phrase.h"

hash_table gLoopToRef;
hash_table gStatToRef;
hash_table gRefToEff;
hash_table gLoopToSync;
hash_table gLoopToSupRef;
hash_table gLoopToUnSupRef;
entity gHREMemory;
expression gBufferSizeEnt;
int gBufferSize;
hash_table gRefToFifo;
hash_table gRefToFifoOff;
hash_table gRefToHREFifo;
hash_table gStatToHtOffset;
hash_table gRefToBuffSize;
hash_table gIndToNum;
hash_table gRefToInd;
hash_table gLoopToToggleEnt;
hash_table gEntToHREFifo;
hash_table gRefToToggle;
hash_table gToggleToInc;
hash_table gIsNewLoop;

static entity  make_new_symbolic_entity(string inName)
{
  string name = strdup(concatenate(TOP_LEVEL_MODULE_NAME,
				   MODULE_SEP_STRING, inName, NULL));

  entity newEnt = make_entity(name,
			      make_type(is_type_variable,
					make_variable(make_basic(is_basic_int, (void *)4), 
						      NIL,
						      NIL)),
			      make_storage(is_storage_rom, UU),
			      make_value(is_value_unknown, UU));
  
  return newEnt;
}

static void give_value_to_symbolic_entity(entity ent, intptr_t val)
{
  constant constVal = make_constant(is_constant_int, (void *)val);

  entity_initial(ent) = make_value(is_value_constant, constVal);
}

static int gCurVar = 0;
static int gCurBuff = 0;
static int gMaxVar = 0;
static int gMaxBuff = 0;

static void update_max()
{
  if(gCurBuff > gMaxBuff)
    {
      gMaxBuff = gCurBuff;
    }

  if(gCurVar > gMaxVar)
    {
      gMaxVar = gCurVar;
    }
}

static void update_HRE_mapping_from_list(list lRef, bool supRef, int inc)
{
  //printf("update_HRE_mapping_from_list\n");
  MAP(REFERENCE, curRef,
  {
    //printf("curRef\n");
    //print_reference(curRef);printf("\n");

    if(supRef)
      {
	gCurBuff += inc;
	hash_put(gRefToBuffSize, curRef, (void *)true);
      }
    else
      {
	gCurVar += inc;
      }
  }, lRef);
}

static void do_HRE_memory_mapping(statement stat);

static void do_HRE_memory_mapping_loop(statement stat)
{
  list lSupportedRef = NIL;
  list lUnSupportedRef = NIL;

  hash_table htOffset = hash_table_make(hash_pointer, 0);

  get_supportedRef_proc(stat, htOffset,
			&lSupportedRef, &lUnSupportedRef);

  hash_put(gStatToHtOffset, stat, htOffset);

  update_HRE_mapping_from_list(lSupportedRef, true, 1);
  update_HRE_mapping_from_list(lUnSupportedRef, false, 1);

  update_max();

  do_HRE_memory_mapping(loop_body(statement_loop(stat)));

  update_HRE_mapping_from_list(lSupportedRef, true, -1);
  update_HRE_mapping_from_list(lUnSupportedRef, false, -1);
}

static void do_HRE_memory_mapping_stat(statement stat)
{
  list lRef = hash_get(gStatToRef, stat);

  if(lRef != HASH_UNDEFINED_VALUE)
    {
      update_HRE_mapping_from_list(lRef, false, 1);

      update_max();

      update_HRE_mapping_from_list(lRef, false, -1);
    }
}

static void do_HRE_memory_mapping(statement stat)
{
  //printf("do_HRE_memory_mapping\n");

  instruction instr = statement_instruction(stat);

  switch(instruction_tag(instr))
    {
    case is_instruction_sequence:
      {
	do_HRE_memory_mapping_stat(stat);

        MAP(STATEMENT, curStat,
	{
	  do_HRE_memory_mapping(curStat);

	}, sequence_statements(instruction_sequence(instr)));
	break;
      }
    case is_instruction_loop:
      {
        do_HRE_memory_mapping_loop(stat);
	break;
      }
    case is_instruction_call:
    case is_instruction_test:
      {
	do_HRE_memory_mapping_stat(stat);
	break;
      }
    default:
      {
	pips_internal_error("impossible");
	break;
      }
    }
}

static bool compute_HRE_memory_mapping(statement stat, int hreMemSize)
{
  gCurVar = 0;
  gCurBuff = 0;
  gMaxVar = 0;
  gMaxBuff = 0;

  entity newEnt = make_new_symbolic_entity("COMENGINE_BUFFER_SIZE");

  gBufferSizeEnt = entity_to_expression(newEnt);

  do_HRE_memory_mapping(stat);

  //printf("gMaxVar %d\n", gMaxVar);
  //printf("gMaxBuff %d\n", gMaxBuff);

  if(gMaxBuff != 0)
    {
      gBufferSize = (hreMemSize - gMaxVar) / gMaxBuff;
    }
  else
    {
      gBufferSize = (hreMemSize - gMaxVar);
    }

  give_value_to_symbolic_entity(newEnt, gBufferSize);

  //printf("gBufferSize %d\n", gBufferSize);

  return true;
}

static void move_declarations(entity new_fun, statement stat)
{
  list lEnt = NIL;

  MAP(ENTITY, curEnt,
  {
    if(!type_varargs_p(entity_type(curEnt)) &&
       !type_statement_p(entity_type(curEnt)) &&
       !type_area_p(entity_type(curEnt)))
      {
	lEnt = gen_nconc(lEnt, CONS(ENTITY, curEnt, NIL));
      }

  }, code_declarations(value_code(entity_initial(new_fun))));

  statement_declarations(stat) = lEnt;

  gen_free_list(code_declarations(value_code(entity_initial(new_fun))));

  code_declarations(value_code(entity_initial(new_fun))) = NIL;
}

#define ALL_DECLS  "PRETTYPRINT_ALL_DECLARATIONS"
#define STAT_ORDER "PRETTYPRINT_STATEMENT_NUMBER"

void create_HRE_module(const char* new_module_name,
		       const char* module_name,
		       statement stat, entity new_fun)
{
  //printf("new HRE module\n");
  //print_statement(stat);

  reset_current_module_entity();

  set_current_module_entity(module_name_to_entity(module_name));

  entity cloned = get_current_module_entity();
  const char* name = entity_local_name(cloned);
  const char* new_name;
  string comments;
  //entity new_fun;
  //statement stat;
  bool saved_b1, saved_b2;
  text t;
    
  /* builds some kind of module / statement for the clone.
   */
  new_name = new_module_name;
  //new_fun = make_empty_subroutine(new_name);

  saved_b1 = get_bool_property(ALL_DECLS);
  saved_b2 = get_bool_property(STAT_ORDER);
  set_bool_property(ALL_DECLS, true);
  set_bool_property(STAT_ORDER, false);

  //stat = comEngine_generate_HRECode(externalized_code, new_fun, l_in, l_out);

  //add_variables_to_module(l_params, l_priv,
  //		  new_fun, stat);

  move_declarations(new_fun, stat);

  reset_current_module_entity();

  set_current_module_entity(module_name_to_entity(module_name));

  t = text_named_module(new_fun, cloned, stat);

  set_bool_property(ALL_DECLS, saved_b1);
  set_bool_property(STAT_ORDER, saved_b2);

  /* add some comments before the code.
   */
  comments = strdup(concatenate(
				"//\n"
				"// PIPS: please caution!\n"
				"//\n"
				"// this routine has been generated as a clone of ", name, "\n"
				"// the code may change significantly with respect to the original\n"
				"// version, especially after program transformations such as dead\n"
				"// code elimination and partial evaluation, hence the function may\n"
				"// not have the initial behavior, if called under some other context.\n"
				"//\n", 0));
  text_sentences(t) = 
    CONS(SENTENCE, make_sentence(is_sentence_formatted, comments),
	 text_sentences(t));
  
  make_text_resource_and_free(new_name, DBR_C_SOURCE_FILE, ".c", t);
  free_statement(stat);

  /* give the clonee a user file.
   */
  DB_PUT_MEMORY_RESOURCE(DBR_USER_FILE, new_name, 
			 strdup(db_get_memory_resource(DBR_USER_FILE, name, true)));

}

/*static void
create_HRE_modules(statement externalized_code,
		  string new_module_name,
		  list l_in, list l_out, list l_params, list l_priv,
		  const char* module_name, int hreMemSize)
{
}
*/
static void comEngine_distribute_code (const char* module_name,
				       string function_name,
			               statement externalized_code, 
			               list l_in,
				       list l_out,
				       list l_params,
			               list l_priv,
				       graph dg) 
{
  bool success = false;

  gLoopToRef = hash_table_make(hash_pointer, 0);
  gStatToRef = hash_table_make(hash_pointer, 0);
  gRefToEff = hash_table_make(hash_pointer, 0);
  gLoopToSync = hash_table_make(hash_pointer, 0);
  gLoopToSupRef = hash_table_make(hash_pointer, 0);
  gLoopToUnSupRef = hash_table_make(hash_pointer, 0);
  gRefToFifo = hash_table_make(hash_pointer, 0);
  gRefToFifoOff = hash_table_make(hash_pointer, 0);
  gRefToHREFifo = hash_table_make(hash_pointer, 0);
  gStatToHtOffset = hash_table_make(hash_pointer, 0);
  gRefToBuffSize = hash_table_make(hash_pointer, 0);
  gIndToNum = hash_table_make(hash_pointer, 0);
  gRefToInd = hash_table_make(hash_pointer, 0);
  gLoopToToggleEnt = hash_table_make(hash_pointer, 0);
  gEntToHREFifo = hash_table_make(hash_pointer, 0);
  gRefToToggle = hash_table_make(hash_pointer, 0);
  gToggleToInc = hash_table_make(hash_pointer, 0);
  gIsNewLoop = hash_table_make(hash_pointer, 0);

  statement procCode;

  success = comEngine_feasability(externalized_code, dg);

  if(success)
    {
      printf("distribution can be done\n");
    }
  else
    {
      printf("distribution cannot be done\n");
      return;
    }

  int hreMemSize = 512;

  success = compute_HRE_memory_mapping(externalized_code, hreMemSize);

  if(success)
    {
      printf("mapping is successful\n");
    }
  else
    {
      printf("mapping is not successful\n");
      return;
    }

  procCode = comEngine_generate_procCode(externalized_code, l_in, l_out);

  comEngine_generate_HRECode(externalized_code,
			     function_name,
			     l_in, l_out, l_params, l_priv,
			     module_name, hreMemSize);

printf("mapping is not successful 1\n");
  free_instruction(statement_instruction(externalized_code));
printf("comEngine_distribute_code 2\n");
  statement_instruction(externalized_code) = 
    statement_instruction(procCode);
printf("comEngine_distribute_code 3\n");
printf("comEngine_distribute_code 4\n");
  statement_comments(externalized_code) = string_undefined;
  statement_number(externalized_code) = STATEMENT_NUMBER_UNDEFINED;
  statement_label(externalized_code) = entity_empty_label();
printf("comEngine_distribute_code 5\n");

 HASH_MAP(loop, lEnt,
 {
   gen_free_list(lEnt);
   lEnt = NIL;
 }, gLoopToToggleEnt);

  printf("comEngine_distribute_code 8\n");
  hash_table_free(gLoopToRef);
  hash_table_free(gStatToRef);
  hash_table_free(gRefToEff);
  hash_table_free(gLoopToSync);
  hash_table_free(gLoopToSupRef);
  hash_table_free(gLoopToUnSupRef);
  free_expression(gBufferSizeEnt);
  hash_table_free(gRefToFifo);
  hash_table_free(gRefToFifoOff);
  hash_table_free(gRefToHREFifo);
  hash_table_free(gStatToHtOffset);
  hash_table_free(gRefToBuffSize);
  hash_table_free(gIndToNum);
  hash_table_free(gRefToInd);
  hash_table_free(gLoopToToggleEnt);
  hash_table_free(gEntToHREFifo);
  hash_table_free(gRefToToggle);
  hash_table_free(gToggleToInc);
  hash_table_free(gIsNewLoop);

}

void comEngine_distribute (const char* module_name, 
			   statement module_stat, 
		           entity module) 
{
  list l_stats;
  hash_table ht_stats;
  hash_table ht_params;
  hash_table ht_private;
  hash_table ht_in_regions;
  hash_table ht_out_regions;

  graph dg = (graph) db_get_memory_resource(DBR_DG, module_name, true);

  l_stats = identify_analyzed_statements_to_distribute (module_stat);

  compute_distribution_context (l_stats, 
				module_stat,
				module,
				&ht_stats,
				&ht_params,
				&ht_private,
				&ht_in_regions,
				&ht_out_regions);
  
  HASH_MAP (function_name, stat, {
      comEngine_distribute_code (module_name,
				 function_name,
				 stat, 
				 hash_get(ht_in_regions,function_name),
				 hash_get(ht_out_regions,function_name),
				 hash_get(ht_params,function_name),
				 hash_get(ht_private,function_name),
				 dg);
  },ht_stats);
  
  hash_table_free(ht_stats);
  hash_table_free(ht_params);
  hash_table_free(ht_private);
  hash_table_free(ht_in_regions);
  hash_table_free(ht_out_regions);
}

bool phrase_comEngine_distributor(const char* module_name)
{
  statement module_stat;
  entity module;

  entity dynamic_area = entity_undefined;

  // set and get the current properties concerning regions 
  set_bool_property("MUST_REGIONS", true);
  set_bool_property("EXACT_REGIONS", true);
  get_regions_properties();
  
  // get the resources 
  module_stat = (statement) db_get_memory_resource(DBR_CODE, 
						   module_name, 
						   true);
  
  module = module_name_to_entity(module_name);
  
  set_current_module_statement(module_stat);
  set_current_module_entity(module_name_to_entity(module_name)); // FI: a bit redundant since module is already available
  
  set_cumulated_rw_effects((statement_effects)
			   db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, true));
  module_to_value_mappings(module);
  
  // sets dynamic_area 
  if (entity_undefined_p(dynamic_area)) {   	
    dynamic_area = FindOrCreateEntity(module_local_name(module),
				      DYNAMIC_AREA_LOCAL_NAME); 
  }

  debug_on("PHRASE_COMENGINE_DISTRIBUTOR_DEBUG_LEVEL");

    // Get the READ, WRITE, IN and OUT regions of the module
    set_rw_effects((statement_effects) 
	db_get_memory_resource(DBR_REGIONS, module_name, true));
    set_in_effects((statement_effects) 
	db_get_memory_resource(DBR_IN_REGIONS, module_name, true));
    set_out_effects((statement_effects) 
	db_get_memory_resource(DBR_OUT_REGIONS, module_name, true));

  // Now do the job

  pips_debug(2, "BEGIN of PHRASE_DISTRIBUTOR\n");
  comEngine_distribute(module_name, module_stat, module);
  pips_debug(2, "END of PHRASE_DISTRIBUTOR\n");

  //print_statement(module_stat);

  pips_assert("Statement structure is consistent after PHRASE_DISTRIBUTOR", 
	      gen_consistent_p((gen_chunk*)module_stat));
	      
  pips_assert("Statement is consistent after PHRASE_DISTRIBUTOR", 
	      statement_consistent_p(module_stat));
  
  // Reorder the module, because new statements have been added  
  module_reorder(module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, 
			 compute_callees(module_stat));
  
  // update/release resources 
  reset_current_module_statement();
  reset_current_module_entity();
  dynamic_area = entity_undefined;
  reset_cumulated_rw_effects();
  reset_rw_effects();
  reset_in_effects();
  reset_out_effects();
  free_value_mappings();
  
  debug_off();
  
  return true;
}

