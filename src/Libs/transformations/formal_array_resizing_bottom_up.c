/******************************************************************
 *
 *		     BOTTOM UP ARRAY RESIZING
 *
 *
*******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "makefile.h"
#include "misc.h"
#include "control.h"
#include "properties.h"
#include "semantics.h"
#include "transformer.h"
#include "pipsmake.h"
#include "abc_private.h"
#include "effects-generic.h"
#include "effects-convex.h"
#include "effects-simple.h"
#include "conversion.h"
#include "text-util.h" /* for words_to_string*/
#include "instrumentation.h"
#include "transformations.h"

static int number_of_right_array_declarations = 0;
static string current_mod ="";

#define PREFIX_DEC  "$DEC"


/* This phase do array resizing for unnormalized and formal arguments only. 
   So we only need SUMMARY_REGIONS */
bool formal_array_resizing_bottom_up(char* mod_name)
{
  entity mod_ent = local_name_to_top_level_entity(mod_name);
  list l_decl = code_declarations(entity_code(mod_ent)), l_regions = NIL; 
  statement mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, TRUE);
  transformer mod_pre;
  Psysteme pre;
  current_mod = mod_name;  
  set_precondition_map((statement_mapping)
		       db_get_memory_resource(DBR_PRECONDITIONS,mod_name,TRUE));
  set_rw_effects((statement_effects) 
		 db_get_memory_resource(DBR_SUMMARY_REGIONS, mod_name, TRUE));
  regions_init(); 
  debug_on("FORMAL_ARRAY_RESIZING_BOTTOM_UP_DEBUG_LEVEL");
  debug(1," Begin bottom up formal array resizing for %s\n", mod_name);
  l_regions = load_rw_effects_list(mod_stmt);  
  mod_pre = load_statement_precondition(mod_stmt);
  pre = predicate_system(transformer_relation(mod_pre));
  user_log("\n-------------------------------------------------------------------------------------\n");
  user_log("Prefix \tFile \tModule \tArray \tNdim \tNew declaration\tOld declaration\n");
  user_log("---------------------------------------------------------------------------------------\n");
 
  while (!ENDP(l_decl))
    {
      entity e = ENTITY(CAR(l_decl));
      if (unnormalized_array_p(e))
	{
	  storage s = entity_storage(e);
	  if (storage_formal_p(s))
	    { 
	      region reg = find_union_regions(l_regions,e);
	      new_array_declaration_from_region(reg,e,pre);
	    }
	}
      l_decl = CDR(l_decl);
    }
  user_log(" \n The total number of right array declarations : %d \n"
	   ,number_of_right_array_declarations );
  
  debug(1,"End bottom up formal array resizing for %s\n", mod_name);
  debug_off();  
  regions_end();
  reset_precondition_map();
  reset_rw_effects();
  current_mod = "";
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
  return TRUE;
}

