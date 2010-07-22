/* A simple phase that outline task with pragma for SCMP

   clement.marguet@hpc-project.com
*/
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "misc.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "control.h"
#include "callgraph.h"
#include "pipsdbm.h"
#include "transformations.h"
#include "resources.h"
#include "accel-util.h"
#include "properties.h"

list statement_to_outline;


static bool pragma_scmp_task_p(pragma p){
  string s =  pragma_to_string(p);
  if(strstr(s,"scmp task")!=NULL)
    return TRUE;
  else
    return FALSE;
}


static bool find_pragma(const statement s){
  /* print_statement(s); */
  pips_debug(1,"______\n");
  if(extensions_defined_p(statement_extensions(s))){
    list l = extensions_extension(statement_extensions(s));
    FOREACH(EXTENSION, ext, l){
      pragma pra = extension_pragma(ext);
      if(pragma_scmp_task_p(pra)){
	pips_debug(1,"SCMP_task\n");
	statement stmt = (statement) gen_get_ancestor(statement_domain,s);
	if(statement_loop_p(s))
	  statement_to_outline = CONS(STATEMENT, s , statement_to_outline);
	
	else if(!statement_undefined_p(stmt) && statement_block_p(stmt))
	  statement_to_outline = CONS(STATEMENT, stmt , statement_to_outline);
	return TRUE;
      }
    }
  }
  return TRUE;
}

bool scalopragma (char* module_name) {
  
  // Use this module name and this environment variable to set
  statement module_statement = PIPS_PHASE_PRELUDE(module_name,
						  "SCALOPRAGMA_DEBUG_LEVEL");
  
  set_cumulated_rw_effects((statement_effects)db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));
  
  statement_to_outline=NIL;
  
  /*look for statement with SCMP pragma*/
  gen_recurse(module_statement,statement_domain,find_pragma,gen_identity);

  statement_to_outline = gen_nreverse(statement_to_outline);

  
  FOREACH(STATEMENT, stmt, statement_to_outline){

    /*delete statement pragma*/
    extensions_extension(statement_extensions(stmt))=NIL;

    outliner(build_new_top_level_module_name(get_string_property("GPU_LAUNCHER_PREFIX")), CONS(STATEMENT,stmt,NIL));
  }





  gen_free_list(statement_to_outline);
  
  // We may have outline some code, so recompute the callees:
  DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, module_name,
			 compute_callees(get_current_module_statement()));
  
  reset_cumulated_rw_effects();
  // Put back the new statement module
  PIPS_PHASE_POSTLUDE(module_statement);
  
  return TRUE;
}
