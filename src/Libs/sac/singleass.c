
#include "genC.h"
#include "linear.h"
#include "ri.h"

#include "resources.h"

#include "misc.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "semantics.h"
#include "effects-generic.h"
#include "transformations.h"

#include "dg.h"
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
#include "graph.h"

#include "sac-local.h" /* needed because sac.h may not exist when 
                        * simdizer.c is compiled */
#include "sac.h"
#include "ricedg.h"
#include "control.h"


//Creates a new entity to replace the given one
static entity make_replacement_entity(entity e)
{
   return make_new_scalar_variable_with_prefix(entity_local_name(e),
					       get_current_module_entity(),
					       entity_basic(e));
}

static void single_assign_statement(graph dg)
{
   hash_table nbPred = hash_table_make(hash_pointer, 0);

   //First, compute the number of incoming DU arcs for each reference
   MAP(VERTEX,
       a_vertex, 
   {
      MAP(SUCCESSOR,
	  suc,
      {
	 MAP(CONFLICT, 
	     c, 
	 {
	    reference r = effect_reference(conflict_sink(c));
	    int nbRef;

	    //Consider only potential DU arcs (may or must does not matter),
	    if (//!effect_scalar_p(conflict_source(c)) ||
	        !effect_write_p(conflict_source(c)) ||
		!effect_read_p(conflict_sink(c)))
	       continue;

	    nbRef = (int)hash_get(nbPred, r);
	    if (nbRef == (int)HASH_UNDEFINED_VALUE)
	       nbRef = 0;
	    nbRef++;
	    hash_put(nbPred, r, (void*)nbRef);
	 },
	     dg_arc_label_conflicts(successor_arc_label(suc)));
      },
	  vertex_successors(a_vertex));
   },
       graph_vertices(dg));

   //Then, for each reference which does never stem from more than one Def,
   //change the variable name
   MAP(VERTEX,
       a_vertex,
   {
      hash_table toBeDone = hash_table_make(hash_pointer, 0);

      MAP(SUCCESSOR,
	  suc,
      {
	 MAP(CONFLICT,
	     c,
	 {
	    list l;

	    //do something only if we are sure to write
	    if (//!effect_scalar_p(conflict_source(c)) ||
	        !effect_write_p(conflict_source(c)) ||
		!effect_must_p(conflict_source(c)) ||
		!effect_read_p(conflict_sink(c)))
	       continue;

	    //if the module has an OUT effect on the variable, do not replace
	    if (0)
	       continue;

	    l = hash_get(toBeDone, effect_reference(conflict_source(c)));

	    //If the sink reference has more than one incoming arc, do not change 
	    //the variable name.
	    //In this case, previous conflicts related to this reference are removed
	    //from the work list, and the list is set to NIL in the work list: this way
	    //it can be seen in later conflicts also.
	    if ((int)hash_get(nbPred, effect_reference(conflict_sink(c))) > 1)
	    {
	       if (l != HASH_UNDEFINED_VALUE)
		  gen_free_list(l);
	       l = NIL;
	    }
	    else if (l != NIL)
	    {
	       if (l == HASH_UNDEFINED_VALUE)
		  l = NIL;
	       l = CONS(CONFLICT, c, l);
	    }

	    hash_put(toBeDone, effect_reference(conflict_source(c)), l);
	 },
	     dg_arc_label_conflicts(successor_arc_label(suc)));
      },
	  vertex_successors(a_vertex));

	    
      HASH_MAP(r,
	       l,
      {
	 MAP(CONFLICT, 
	     c,
	 {
	    entity ne;
	    entity e;

	    e = effect_entity(conflict_source(c));
	    ne = make_replacement_entity(effect_entity(conflict_source(c)));
	    
	    reference_variable(effect_reference(conflict_source(c))) = ne;
	    reference_variable(effect_reference(conflict_sink(c))) = ne;
	 },
	     (list)l);

	 gen_free_list(l);
      },
	       toBeDone);

      hash_table_free(toBeDone);
   },
       graph_vertices(dg));

   hash_table_free(nbPred);
}

bool single_assignment(char * mod_name)
{
   /* get the resources */
   statement mod_stmt = (statement)
      db_get_memory_resource(DBR_CODE, mod_name, TRUE);
   graph dg = (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);

   set_current_module_statement(mod_stmt);
   set_current_module_entity(local_name_to_top_level_entity(mod_name));

   debug_on("SINGLE_ASSIGNMENT_DEBUG_LEVEL");

   /* Now do the job */
   module_reorder(mod_stmt);  
   single_assign_statement(dg);
   
   pips_assert("Statement is consistent after SINGLE_ASSIGNMENT", 
	       statement_consistent_p(mod_stmt));

   /* Reorder the module, because new statements have been added */  
   module_reorder(mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CALLEES, mod_name, 
			  compute_callees(mod_stmt));
 
   /* update/release resources */
   reset_current_module_statement();
   reset_current_module_entity();

   debug_off();

   return TRUE;
}
