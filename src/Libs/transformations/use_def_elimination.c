/* $Id$
 */

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "misc.h"
#include "control.h"
#include "effects-generic.h"
#include "effects-simple.h"
#include "database.h"
#include "pipsdbm.h"
#include "resources.h"
#include "graph.h"
#include "dg.h"
/* Instantiation of the dependence graph: */
typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;
/* Just to be able to use ricedg.h: */
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"
/* */
#include "ricedg.h"
#include "semantics.h"
#include "transformations.h"

static graph dependence_graph;


static hash_table ordering_to_dg_mapping;

static set the_useful_statements;

/* Define the mapping to store the statements generating something
   useful for a given statement and the functions used to deal
   with. It is a mapping from a statement to a *set* of statements,
   those that generate interesting use-def values.*/
static hash_table statements_use_def_dependence;


static bool
use_def_true_filter(statement s)
{
   /* Go down: */
   return TRUE;
}


/* Define a static stack and related functions to remember the current
   statement for build_statement_to_statement_father_mapping(): */
DEFINE_LOCAL_STACK(current_statement, statement)


/* Define the mapping to store the statement owning the statements and the
   functions used to deal with: */
GENERIC_LOCAL_FUNCTION(statement_father, persistant_statement_to_statement)


static void
add_statement_to_the_statement_to_statement_father_mapping(statement s)
{
   /* Pop the current_statement_stack: */
   current_statement_rewrite(s);

   ifdebug(4)
      fprintf(stderr, "add_statement_to_the_statement_to_statement_father_mapping statement %p (%#x), father %p\n",
              s, statement_ordering(s), current_statement_head());
   
   /* First add the current father for this statement: */
   /* Since statement_undefined == hash_undefined_value, we cannot put
      a statement_undefined in the hash_table... */
   if (current_statement_head() != statement_undefined)
      store_statement_father(s, current_statement_head());
}


/* Define the mapping to store the control fathers of the statements
   and the functions used to deal with: */
GENERIC_LOCAL_FUNCTION(control_father, persistant_statement_to_control)


/* Build a mapping from a statement to its eventual control father. */
void
set_control_statement_father(control c)
{
   store_control_father(control_statement(c), c);
}


static void
build_statement_to_statement_father_mapping(statement s)
{
   ifdebug(4)
      fprintf(stderr, "build_statement_to_statement_father_mapping statement %p (%#x)\n",
              s, statement_ordering(s));

   make_current_statement_stack();
   /* The first statement has no father: */
   current_statement_push(statement_undefined);
   
   gen_multi_recurse(s, statement_domain,
                     /* Just push the current statement on the
                        current_statement_stack: */
                     current_statement_filter,
                     add_statement_to_the_statement_to_statement_father_mapping,
                     /* Build a mapping from a statement to its
                        eventual control father. */
                     control_domain, gen_true, set_control_statement_father,
                     NULL);

   free_current_statement_stack();
}


/* Build the mapping from each statement to the statements generating
   something useful for it: */
static void
build_statement_to_statement_dependence_mapping(graph dependence_graph)
{
   statements_use_def_dependence = hash_table_make(hash_pointer, 0);
   
   MAP(VERTEX,
       a_vertex,
       {
          statement s1 = vertex_to_statement(a_vertex);

          debug(7, "build_statement_to_statement_dependence_mapping",
                "\tSuccessor list: %p for statement ordering %p\n", 
                vertex_successors(a_vertex),
                dg_vertex_label_statement(vertex_vertex_label(a_vertex)));
          MAP(SUCCESSOR, a_successor,
              {
                 vertex v2 = successor_vertex(a_successor);
                 statement s2 = vertex_to_statement(v2);
                 dg_arc_label an_arc_label = successor_arc_label(a_successor);
                 ifdebug(7)
                    fprintf(stderr, "\t%p --> %p with conflicts\n", s1, s2);
                 /* Try to find at least one of the use-def chains between
                    s and a successor: */
                 MAP(CONFLICT, a_conflict,
                     {
                        statement use;
                        statement def;
                        
                        ifdebug(7) 
                           {
                              fprintf(stderr, "\t\tfrom ");
                              print_words(stderr, words_effect(conflict_source(a_conflict)));
                              fprintf(stderr, " to ");
                              print_words(stderr, words_effect(conflict_sink(a_conflict)));
                              fprintf(stderr, "\n");
                           }
                    
                        /* Something is useful for the current statement if
                           it writes something that is used in the current
                           statement: */
                        if (action_read_p(effect_action(conflict_source(a_conflict)))
                            && action_write_p(effect_action(conflict_sink(a_conflict)))) {
                           use = s1;
                           def = s2;
                        }
                        else if (action_write_p(effect_action(conflict_source(a_conflict)))
                                 && action_read_p(effect_action(conflict_sink(a_conflict)))) {
                           def = s1;
                           use = s2;
                        }
                        else
                           /* The dependance is not a use-def one,
                              look forward... */
                           continue;
                        
                        {
                           /* Mark that we will visit the node that defined a
                              source for this statement, if not already
                              visited: */
                           set statements_set =
                              (set) hash_get(statements_use_def_dependence,
                                             (char *) use);
                                       
                           if (statements_set == (set) HASH_UNDEFINED_VALUE) {
                              /* It is the first dependence we found
                                 for s1. Create the set: */
                              statements_set = set_make(set_pointer);
                              hash_put(statements_use_def_dependence,
                                       (char *) use,
                                       (char *) statements_set);
                           }

                           /* Mark the fact that s2 create something
                              useful for s1: */
                           set_add_element(statements_set,
                                           statements_set,
                                           (char *) def);

                           ifdebug(6)
                              fprintf(stderr, "\tUse: statement %p (%#x). Def: statement %p (%#x).\n",
                                      use, statement_ordering(use),
                                      def, statement_ordering(def));
                        }
                        
                        /* One use-def is enough for this variable
                           couple: */
                        break;
                     },
                        dg_arc_label_conflicts(an_arc_label));
              },
                 vertex_successors(a_vertex));

       },
          graph_vertices(dependence_graph));
}


void static
free_statement_to_statement_dependence_mapping()
{
   HASH_MAP(key, value,
            {
               set_free((set) value);
            },
            statements_use_def_dependence);
}


void static
mark_this_node_and_its_predecessors_in_the_dg_as_useful(set s,
                                                        vertex v)
{
   if (set_belong_p(s, (char *) v))
      /* We have already seen this node: */
      return;

   /* Mark the current vertex as useful: */
   set_add_element(s, s, (char *) v);

   if (get_debug_level() >= 6)
      fprintf(stderr, "mark_this_node_and_its_predecessors_in_the_dg_as_useful: vertex %p marked, statement ordering (%#x).\n",
              v,      
              dg_vertex_label_statement(vertex_vertex_label(v)));
  
   MAP(SUCCESSOR, a_successor,
       {
          dg_arc_label label = successor_arc_label(a_successor);
          /* Try to find at least one use-def chain: */
          MAP(CONFLICT, a_conflict,
              {
                 /* Something is useful for the current statement if
                    it writes something that is used in the current
                    statement: */
                 if (action_read_p(conflict_source(a_conflict))
                     && action_write_p(conflict_sink(a_conflict))) {
                    /* Mark the node that generate something useful
                       for the current statement as useful: */
                    mark_this_node_and_its_predecessors_in_the_dg_as_useful(s,
                                                                            successor_vertex(a_successor));
                    /* Only needed to mark once: */
                    break;
                 }
              },
                 dg_arc_label_conflicts(label));
       },
          vertex_successors(v));
}


static void
iterate_through_the_predecessor_graph(statement s,
                                      set elements_to_visit)
{
   ifdebug(6)
      fprintf(stderr, "iterate_through_the_predecessor_graph, statement %p (%#x).\n",
              s, statement_ordering(s));

   /* Mark the current statement as useful: */
   set_add_element(the_useful_statements, the_useful_statements, (char *) s);
  
   /* First mark the dependence graph predecessors: */
   {
      set statements_set = (set) hash_get(statements_use_def_dependence,
                                          (char *) s);
      if (statements_set != (set) HASH_UNDEFINED_VALUE) {
         /* There is at least one statement that generates something
            useful for s: */
         SET_MAP(element,
                 {
                    statement s2 = (statement) element;
                    
                    /* Mark that we will visit the node that defined a
                       source for this statement, if not already
                       visited: */
                    set_add_element(elements_to_visit,
                                    elements_to_visit,
                                    (char *) s2);
                       ifdebug(6)
                          fprintf(stderr, "\tstatement %p (%#x) useful by use-def.\n",
                                  s2, statement_ordering(s2));
                 },
                    statements_set);
      }
   }
   
   /* Mark the father too for control dependences: */
   if (bound_statement_father_p(s)) {
      statement father = load_statement_father(s);
      set_add_element(elements_to_visit, elements_to_visit, (char *) father);
      ifdebug(6)
         fprintf(stderr, "\tstatement %p (%#x) useful as the statement owner.\n",
                 father, statement_ordering(father));
   }

   {
      /* And if the statement is in an unstructured, mark all the
         controlling unstructured nodes predecessors as useful, that
         is all the unstructured IF back-reachable. */
      if (bound_control_father_p(s)) {
         list blocks = NIL;       
         control control_father = load_control_father(s);
         BACKWARD_CONTROL_MAP(pred, {
            if (gen_length(control_successors(pred)) == 2) {
               /* pred is an unstructured IF that control control_father: */
               set_add_element(elements_to_visit,
                               elements_to_visit,
                               (char *) control_statement(pred));
               ifdebug(6)
                  fprintf(stderr, "\tstatement unstructed IF %p (%#x) useful by control dependence.\n",
                          control_statement(pred), statement_ordering(control_statement(pred)));
            }           
         }, control_father, blocks);
         gen_free_list(blocks);
      }
   }            
}


static void
propagate_the_usefulness_through_the_predecessor_graph()
{
   ifdebug(5)
      fprintf(stderr, "Entering propagate_the_usefulness_through_the_predecessor_graph\n");
   
   gen_set_closure((void (*)(char *, set)) iterate_through_the_predecessor_graph,
                   the_useful_statements);

   ifdebug(5)
      fprintf(stderr, "Exiting propagate_the_usefulness_through_the_predecessor_graph\n");
}


static void
use_def_deal_if_useful(statement s)
{
   bool this_statement_has_an_io_effect;
   bool this_statement_writes_a_procedure_argument;
   bool this_statement_is_a_format;
   bool this_statement_is_an_unstructured_test = FALSE;

   if (get_debug_level() >= 5) {
      fprintf(stderr, "use_def_deal_if_useful: statement %p (%#x)\n",
              s, statement_ordering(s));
      print_text(stderr, text_statement(get_current_module_entity(), 0, s));
   }

   if (statement_ordering(s) == STATEMENT_ORDERING_UNDEFINED) {
      user_warning("use_def_deal_if_useful", "exited since it found a statement without ordering: statement %p (%#x)\n", s, statement_ordering(s));
      return;
   }
   
   /* The possible reasons to have useful code: */
   /* - the statement does an I/O: */
   this_statement_has_an_io_effect = statement_io_effect_p(s);
   /* - the statement writes a procedure argument or the return
      variable of the function, so the value may be used by another
      procedure: */
   /* Regions out should be more precise: */
   this_statement_writes_a_procedure_argument =
       statement_has_a_formal_argument_write_effect_p(s);
   
   /* Avoid to remove formats in a first approach: */
   this_statement_is_a_format = instruction_format_p(statement_instruction(s));

   /* Unstructured tests are very hard to deal with since they can
      have major control effects, such as leading to an infinite loop,
      etc. and it is very hard to cope with... Thus, keep all
      unstructured tests in this approach since I cannot prove the
      termination of the program and so on.  */
   if (bound_control_father_p(s)) {
       control control_father = load_control_father(s);
       if (gen_length(control_successors(control_father)) == 2)
	   /* It is an unstructured test: keep it: */
	   this_statement_is_an_unstructured_test = TRUE;
   }

   if (get_debug_level() >= 6) {
      if (this_statement_has_an_io_effect)
         fprintf(stderr, "Statement %p has an io effect.\n", s);
      if (this_statement_writes_a_procedure_argument)
         fprintf(stderr,
                 "Statement %p writes an argument of its procedure.\n", s);
      if (this_statement_is_a_format)
         fprintf(stderr, "Statement %p is a FORMAT.\n", s);
      if (this_statement_is_an_unstructured_test)
         fprintf(stderr, "Statement %p is an unstructured test.\n", s);
   }
   
   if (this_statement_has_an_io_effect
       || this_statement_writes_a_procedure_argument
       || this_statement_is_a_format
       || this_statement_is_an_unstructured_test)
      /* Mark this statement as useful: */
      set_add_element(the_useful_statements, the_useful_statements, (char *) s);

   if (get_debug_level() >= 5)
      fprintf(stderr, "end use_def_deal_if_useful\n");
}


void static
remove_this_statement_if_useless(statement s)
{
   if (! set_belong_p(the_useful_statements, (char *) s)) {
      gen_free(statement_instruction(s));
      statement_instruction(s) = make_instruction_block(NIL);
      /* Since the RI need to have no label on instruction block: */
      fix_sequence_statement_attributes(s);
      if (get_debug_level() >= 6)
         fprintf(stderr, "remove_this_statement_if_useless removes statement %p (%#x).\n", s, statement_ordering(s));
   }
}


void static
remove_all_the_non_marked_statements(statement s)
{
   ifdebug(5)
      fprintf(stderr, "Entering remove_all_the_non_marked_statements\n");
   
   gen_recurse(s, statement_domain,
               /* Since statements can be nested, only remove in a
                  bottom-up way: */
               use_def_true_filter,
               remove_this_statement_if_useless);

   ifdebug(5)
      fprintf(stderr, "Exiting remove_all_the_non_marked_statements\n");
}


void
use_def_elimination_on_a_statement(statement s)
{
   the_useful_statements = set_make(set_pointer);
   init_control_father();
   init_statement_father();
   
   /* pips_assert("use_def_elimination_on_a_statement", */
   ordering_to_dg_mapping = compute_ordering_to_dg_mapping(dependence_graph);

   build_statement_to_statement_father_mapping(s);
   build_statement_to_statement_dependence_mapping(dependence_graph);

   /* Mark as useful the seed statements: */
   gen_recurse(s, statement_domain,
               use_def_true_filter,
               use_def_deal_if_useful);

   /* Propagate the usefulness through all the predecessor graph: */
   propagate_the_usefulness_through_the_predecessor_graph();
   
   remove_all_the_non_marked_statements(s);

   hash_table_free(ordering_to_dg_mapping);
   free_statement_to_statement_dependence_mapping();
   close_statement_father();
   close_control_father();
   set_free(the_useful_statements);
}


bool
use_def_elimination(char * module_name)
{
   statement module_statement;

   /* Get the true ressource, not a copy. */
   module_statement =
      (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

   /* Get the data dependence graph: */
   /* The dg is more precise than the chains, so I (RK) guess I should
      remove more code with the dg, specially with array sections and
      so on. */
   /* FI: it's much too expensive; and how can you gain something
    * with scalar variables?
    */
   /*
   dependence_graph =
      (graph) db_get_memory_resource(DBR_DG, module_name, TRUE);
      */

   dependence_graph =
      (graph) db_get_memory_resource(DBR_CHAINS, module_name, TRUE);

   /* The proper effect to detect the I/O operations: */
   set_proper_rw_effects((statement_effects)
			 db_get_memory_resource(DBR_PROPER_EFFECTS,
						module_name,
						TRUE)); 

   set_current_module_statement(module_statement);
   set_current_module_entity(local_name_to_top_level_entity(module_name));

   initialize_ordering_to_statement(module_statement);

   debug_on("USE_DEF_ELIMINATION_DEBUG_LEVEL");

   use_def_elimination_on_a_statement(module_statement);

   /* Reorder the module, because some statements have been deleted.
      Well, the order on the remaining statements should be the same,
      but by reordering the statements, the number are consecutive. Just
      for pretty print... :-) */
   module_reorder(module_statement);

   debug(2, "use_def_elimination", "done for %s\n", module_name);

   debug_off();

   DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_statement);

   reset_proper_rw_effects();
   reset_current_module_statement();
   reset_current_module_entity();

   /* Should have worked: */
   return TRUE;
}
