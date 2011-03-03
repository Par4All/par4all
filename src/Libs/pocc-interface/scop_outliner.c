#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif


#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "boolean.h"


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "conversion.h"
#include "properties.h"
#include "transformations.h"

/*static_control*/
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes	*/
#include "ri.h"
/* Types arc_label and vertex_label must be defined although they are
   not used */
typedef void * arc_label;
typedef void * vertex_label;
#include "graph.h"
#include "paf_ri.h"
#include "database.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "paf-util.h"
#include "accel-util.h"
#include "callgraph.h"

#define ITEM_NOT_IN_ARRAY -1


static statement stmt_sauv ;
static string pragma_begin;
static string pragma_end;
static list lists_to_outline ;
static list save_seq;
static bool begin;

/*
 * compute iteratively SCoPs to outline --> lists_to_outline and then
 * call the outliner
 */

static bool outlining_scop(sequence s)
{
  list stmts = sequence_statements(s);
  FOREACH(statement, stmt, stmts) {
    list l_exts = extensions_extension (statement_extensions (stmt));
    FOREACH (EXTENSION, ext, l_exts) {
      pragma pr = extension_pragma (ext);
      if(strcmp(pragma_string(pr),pragma_begin)==0 )
	begin = TRUE;
      if(strcmp(pragma_string(pr),pragma_end)==0)
	{
	  begin = FALSE;
	  lists_to_outline  = CONS(STATEMENT, stmt, lists_to_outline );
	  lists_to_outline = gen_nreverse(lists_to_outline);
	  /*Outline SCoPs
	   */ 
	  (void )outliner(build_new_top_level_module_name(get_string_property("Function_Prefix"), false), lists_to_outline);
	  return TRUE;
	}
    }
    /*
     *We handle stmts if we have already pragma scop,  
     *that test avoids redundancy
     */
    if(begin)
      {
	if(statement_loop_p(stmt))
	  { 
	    save_seq=NIL;
	    instruction  inst = statement_instruction(stmt);
	    loop l = instruction_loop (inst);
	    statement body = loop_body(l);
	    if (statement_sequence_p(body))
	      save_seq = sequence_statements(statement_sequence(body));
	    else 
	      save_seq = CONS(statement, body, NIL );
	  }
	bool find_p = FALSE;
	FOREACH(statement, st, save_seq) {
	  if(stmt != st)
	    find_p = TRUE;
	}
	if( (!find_p ||statement_loop_p(stmt) ))
	  lists_to_outline  = CONS(STATEMENT, stmt, lists_to_outline );      
      }
  }
  return TRUE;
}
bool scop_outliner(char * module_name)
{ 
  entity	module;
  statement	module_stat;
  lists_to_outline = NIL;
  module = local_name_to_top_level_entity(module_name);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
 
  set_current_module_statement((statement)
			       db_get_memory_resource(DBR_CODE, module_name, TRUE));
  set_prettyprint_language_from_property(is_language_c);
  module_stat = get_current_module_statement();
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS,module_name,TRUE)); 

  pragma_begin = get_string_property("Pragma_Begin");
  pragma_end = get_string_property("Pragma_End");
   
  begin= FALSE;
  gen_recurse(module_stat, sequence_domain/*pragma_domain*/, outlining_scop , gen_null);
  lists_to_outline = gen_nreverse(lists_to_outline);
     
  gen_free_list(lists_to_outline);
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name, compute_callees(module_stat));

  DB_PUT_MEMORY_RESOURCE(DBR_CODE,module_name,module_stat);

  reset_cumulated_rw_effects();
  reset_current_module_entity();
  reset_current_module_statement();
  return TRUE;  
} 
 
