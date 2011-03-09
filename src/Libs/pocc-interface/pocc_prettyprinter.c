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
#include "static_controlize.h"

#define ITEM_NOT_IN_ARRAY -1

/* Global variables */
static statement_mapping	Gsc_map;
static statement save_stmt;
static bool in_loop_p = FALSE, pragma_added_p = false, wehaveloop = FALSE;
static gen_array_t outer_loops_array;


/**
 *search for an element in gen_array_t, if not found return -1
*/
static int gen_array_index(gen_array_t ar, intptr_t item){
  size_t i;
  for(i = 0; i< gen_array_nitems(ar); i++){
    if(gen_array_item(ar, i) != NULL){
      if(item == (intptr_t)gen_array_item(ar, i)){
	return i;
      }
    }
  }
  return ITEM_NOT_IN_ARRAY;
}


/**
 * use the result of control static to add pragmas for pocc
 * compiler , that pragmas delimit  control static parts (static
 * control region which can contains many loop nests 
 */

static void add_pragma_endscop(statement s)
{
 /**
  *insert the pragma (endscop) as a string at the end of the precedent statement(SCoP)
  */
  string str = strdup("endscop");
  add_pragma_str_to_statement (s, str, FALSE);
  pragma_added_p = false; 
  wehaveloop = false;
}


static bool pragma_scop(statement s)
{
  instruction  inst = statement_instruction(s);
  static_control sc;
  if ( instruction_loop_p(inst)  )
    {
      loop l = instruction_loop (inst);
      sc = (static_control) GET_STATEMENT_MAPPING(Gsc_map, loop_body(l));
      if (static_control_yes( sc ) == TRUE && !wehaveloop )
	{
	  string str = strdup("scop");	   
	  /**
	   *insert the pragma (scop) as a string to the current statement
	   */
	  if (gen_array_index(outer_loops_array, statement_ordering(s)) != ITEM_NOT_IN_ARRAY)
	    {
	      add_pragma_str_to_statement (s, str, FALSE);
	      pragma_added_p = true;
	    }
	}
    } 
  else
    {
      if ( instruction_tag(inst) == is_instruction_call)
	{
	  if (continue_statement_p(s) &&  !declaration_statement_p(s))
	    {
	      if ( pragma_added_p )
		{
		  /**
		   * we save the end of the SCoP to mark it after with
		   * pragma end scop if it is closed
		   */
		  save_stmt = s;
		  wehaveloop = true;
		}
	    }
	  else
	    {
	      sc = (static_control) GET_STATEMENT_MAPPING(Gsc_map, s);
	      if ( pragma_added_p && static_control_yes( sc ) == FALSE)
		add_pragma_endscop(save_stmt);
	    }
	}
    }
  pips_assert("statement s is consistent", statement_consistent_p(s));
  return TRUE;
}


/**
 * compute outer_loops
 */
static bool outer_loop(statement s)
{ 
  pips_debug(1,"statement_ordering = %"PRIdPTR", stmt = %s\n",statement_ordering(s), text_to_string(statement_to_text(s)));
  instruction  inst = statement_instruction(s);
  if ( instruction_tag(inst) ==  is_instruction_loop )
    {
      if (in_loop_p == FALSE)
	{
	  gen_array_append (outer_loops_array,(void*)statement_ordering(s));
	  in_loop_p = TRUE;
	  statement end_outermost_loop = make_continue_statement(entity_empty_label());
	  insert_statement(s, end_outermost_loop, false);
	}
    }
   else
    {
      if ( instruction_tag(inst) == is_instruction_call)
	{
	  if (continue_statement_p(s) &&  !declaration_statement_p(s))
	    in_loop_p = FALSE;
	}
    }
     
  return TRUE;
}

#define  SCOPPRETTY ".scop.c"
bool pocc_prettyprinter(char * module_name)
{ 
  entity	module;
  statement	module_stat;
     
  module = local_name_to_top_level_entity(module_name);
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  set_current_module_statement((statement)
	 db_get_memory_resource(DBR_CODE, module_name, TRUE));


 
  module_stat = get_current_module_statement();
 

 
  Gsc_map = (statement_mapping)
    db_get_memory_resource( DBR_STATIC_CONTROL, module_name, TRUE);
  
  /*
   *save the outer_loops in outer_loops_array
   */
  outer_loops_array = gen_array_make(0);


  gen_recurse(module_stat, statement_domain, outer_loop, gen_null);
  gen_recurse(module_stat, statement_domain, pragma_scop, gen_null);
 
  /* case of, a function which doesn't contain a statement after the
   *SCoP, we have to test if we have added pragma scop without its endscop
   */
  if ( pragma_added_p )
    {
      add_pragma_endscop(save_stmt);
    }
  
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,module_stat);
  
  reset_current_module_entity();
  reset_current_module_statement();
  gen_array_free(outer_loops_array);
 
  return TRUE;  
}
