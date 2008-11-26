
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

bool simd_supported_stat(statement stat);

//Creates a new entity to replace the given one
static entity make_replacement_entity(entity e)
{
   return make_new_scalar_variable_with_prefix(entity_local_name(e),
					       get_current_module_entity(),
					       entity_basic(e));
}

void saCallReplace(call c, reference ref, entity next);

void saReplaceReference(expression e, reference ref, entity next)
{
    syntax s = expression_syntax(e);

    switch(syntax_tag(s)) {
      case is_syntax_reference : {
	  reference r = syntax_reference(s);
	  /* replace if equal to ref */

	  if ( reference_equal_p(syntax_reference(s), ref)) {
	    reference_variable(syntax_reference(s)) = next;
	    gen_free_list(reference_indices(syntax_reference(s)));
	    reference_indices(syntax_reference(s)) = NIL;

	    if(expression_normalized(e) != normalized_undefined)
	    {
	      free_normalized(expression_normalized(e));
	      expression_normalized(e) = normalized_undefined;
	    }

	    NORMALIZE_EXPRESSION(e);
	  }
	  else {
	      MAPL(lexpr, {
		  expression indice = EXPRESSION(CAR(lexpr));
		  saReplaceReference(indice, ref, next);
	      }, reference_indices(r));
	  }
      }
	break;
      case is_syntax_range :
	saReplaceReference(range_lower(syntax_range(s)), ref, next);
	saReplaceReference(range_upper(syntax_range(s)), ref, next);
	saReplaceReference(range_increment(syntax_range(s)), ref, next);
	break;
      case is_syntax_call :
	saCallReplace(syntax_call(s), ref, next);
	break;
      default : 
	pips_error("checkReplaceReference", "unknown tag: %d\n", 
		   (int) syntax_tag(expression_syntax(e)));
    }
}

void saCallReplace(call c, reference ref, entity next)
{
    value vin;
    entity f;

    f = call_function(c);
    vin = entity_initial(f);
	
    switch (value_tag(vin)) {
      case is_value_constant:
	/* nothing to replace */
	break;
      case is_value_symbolic:
	/* 
	pips_error("CallReplaceReference", 
		   "case is_value_symbolic: replacement not implemented\n");
		   */
	/* FI: I'd rather assume, nothing to replace for symbolic constants */
	break;
      case is_value_intrinsic:
      case is_value_code:
      case is_value_unknown:
	/* We assume that it is legal to replace arguments (because it should
	   have been verified with the effects that the index is not WRITTEN).
	   */
	MAPL(a, {
	    saReplaceReference(EXPRESSION(CAR(a)), ref, next);
	}, call_arguments(c));
	break;
      default:
	pips_error("CallReplaceReference", "unknown tag: %d\n", 
		   (int) value_tag(vin));

	abort();
    }
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
	    reference r = effect_any_reference(conflict_sink(c));
	    int nbRef;

	    //Consider only potential DU arcs (may or must does not matter) and do not consider
	    // arrays
	    if ((gen_length(reference_indices(effect_any_reference(conflict_source(c)))) != 0) ||
                (gen_length(reference_indices(effect_any_reference(conflict_sink(c)))) != 0) ||
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
      hash_table hashSuc = hash_table_make(hash_pointer, 0);
      bool var_created = FALSE;
      entity se = entity_undefined;

      MAP(SUCCESSOR,
	  suc,
      {
	 MAP(CONFLICT,
	     c,
	 {
	    list l;
	    list lSuc;

	    //do something only if we are sure to write
	    if ((gen_length(reference_indices(effect_any_reference(conflict_source(c)))) != 0) ||
                (gen_length(reference_indices(effect_any_reference(conflict_sink(c)))) != 0) ||
	        !effect_write_p(conflict_source(c)) ||
		!effect_must_p(conflict_source(c)) ||
		!effect_read_p(conflict_sink(c)))
	       continue;

	    //if the module has an OUT effect on the variable, do not replace
	    if (0)
	       continue;

	    l = hash_get(toBeDone, effect_any_reference(conflict_source(c)));
	    lSuc = hash_get(hashSuc, effect_any_reference(conflict_source(c)));

	    //If the sink reference has more than one incoming arc, do not change 
	    //the variable name.
	    //In this caeffect_entity(conflict_source(c))se, previous conflicts related to this reference are removed
	    //from the work list, and the list is set to NIL in the work list: this way
	    //it can be seen in later conflicts also.
	    if ((int)hash_get(nbPred, effect_any_reference(conflict_sink(c))) > 1)
	    {
	       if (l != HASH_UNDEFINED_VALUE)
		  gen_free_list(l);
	       l = NIL;

	       if (lSuc != HASH_UNDEFINED_VALUE)
		  gen_free_list(lSuc);
	       lSuc = NIL;
	    }
	    else if (l != NIL)
	    {
	       if(simd_supported_stat(vertex_to_statement(a_vertex)) &&
		  simd_supported_stat(vertex_to_statement(successor_vertex(suc))))
	       {
	          if (l == HASH_UNDEFINED_VALUE)
		     l = NIL;
	          l = CONS(CONFLICT, c, l);

	          if (lSuc == HASH_UNDEFINED_VALUE)
		     lSuc = NIL;
	          lSuc = CONS(SUCCESSOR, suc, lSuc);
	       }
	       else
	       {
	          if (l != HASH_UNDEFINED_VALUE)
		     gen_free_list(l);
	          l = NIL;

	          if (lSuc != HASH_UNDEFINED_VALUE)
		     gen_free_list(lSuc);
	          lSuc = NIL;
	       }
	    }

	    hash_put(toBeDone, effect_any_reference(conflict_source(c)), l);
            hash_put(hashSuc, effect_any_reference(conflict_source(c)), lSuc);
	 },
	     dg_arc_label_conflicts(successor_arc_label(suc)));
      },
	  vertex_successors(a_vertex));

      HASH_MAP(r,
	       l,
      {
         list lSuc = hash_get(hashSuc, r);
	 list lCurSuc = lSuc;
	 MAP(CONFLICT, 
	     c,
	 {
	    entity ne;
	    reference rSource;
	    reference rSink;
	    entity eSource;
	    entity eSink;

            // Get the entity corresponding to the source and to the sink
	    eSource = effect_entity(conflict_source(c));
	    eSink = effect_entity(conflict_sink(c));

            rSource = effect_any_reference(conflict_source(c));
            rSink = effect_any_reference(conflict_sink(c));

	    // Get the successor
            successor suc = SUCCESSOR(CAR(lCurSuc));

            statement stat2 = vertex_to_statement(successor_vertex(suc));

            // If the source variable hasn't be replaced yet for the source 
            if(var_created == FALSE)
            {
               // Create a new variable
	       ne = make_replacement_entity(eSource);

               // Replace the source by the created variable
	       reference_variable(rSource) = ne;

               pips_debug(1, "ref created %s\n", entity_local_name(effect_entity(conflict_source(c))));

               // Save the entity corresponding to the created variable
               se = ne;
               var_created = TRUE;
            }

	    bool actionWrite = FALSE;
            MAP(EFFECT, f, 
            {
	       entity effEnt = effect_entity(f) ;

	       if(action_write_p(effect_action(f)) && same_entity_p(eSink, effEnt))
               {
		  actionWrite = TRUE;
	       }
	    }, load_proper_rw_effects_list(stat2)) ;

            expression exp2 = EXPRESSION(CAR(call_arguments(
                                 instruction_call(statement_instruction(stat2)))));

	    if(!actionWrite)
	    {
              saReplaceReference(exp2, rSink, se);
	    }

  	    exp2 = EXPRESSION(CAR(CDR(call_arguments(
                      instruction_call(statement_instruction(stat2))))));

            saReplaceReference(exp2, rSink, se);

	    lCurSuc = CDR(lCurSuc);
	 },
	     (list)l);
         
         var_created = FALSE;

	 gen_free_list(l);
	 gen_free_list(lSuc);
      },
	       toBeDone);

      hash_table_free(toBeDone);
      hash_table_free(hashSuc);
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
   set_current_module_entity(module_name_to_entity(mod_name));

   set_proper_rw_effects((statement_effects) 
      db_get_memory_resource(DBR_PROPER_EFFECTS, mod_name, TRUE));

   debug_on("SINGLE_ASSIGNMENT_DEBUG_LEVEL");

   /* Now do the job */
   module_reorder(mod_stmt); 

   // To prevent some warnings
   hash_dont_warn_on_redefinition();

   single_assign_statement(dg);

   // Restore the warning
   hash_warn_on_redefinition();
   
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
   reset_proper_rw_effects();

   debug_off();

   return TRUE;
}
