#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "misc.h"
#include "control.h"
#include "effects.h"
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

static statement_mapping the_proper_effects;

static hash_table ordering_to_dg_mapping;

static set the_useful_statements;

static hash_table statement_to_statement_father_mapping;

static bool
use_def_true_filter(statement s)
{
   /* Go down: */
   return TRUE;
}


static void
use_def_rewrite_nothing(statement s)
{
   /* Just do nothing... */
}


/* Define a static stack and related functions to remember the current
   statement for build_statement_to_statement_father_mapping(): */
DEFINE_LOCAL_STACK(current_statement, statement)


static void
add_statement_to_the_statement_to_statement_father_mapping(statement s)
{
   /* Pop the current_statement_stack: */
   current_statement_rewrite(s);

   ifdebug(4)
      fprintf(stderr, "add_statement_to_the_statement_to_statement_father_mapping statement %#x (%#x), father %#x\n",
              (int) s, statement_ordering(s), (int) current_statement_head());
   
   /* First add the current father for this statement: */
   /* Since statement_undefined == hash_undefined_value, we cannot put
      a statement_undefined in the hash_table... */
   if (current_statement_head() != statement_undefined)
      hash_put(statement_to_statement_father_mapping,
               (char *) s,
               (char *) current_statement_head());
}


GENERIC_LOCAL_FUNCTION(control_father, controlmap)


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
      fprintf(stderr, "build_statement_to_statement_father_mapping statement %#x (%#x)\n",
              (int) s, statement_ordering(s));

   statement_to_statement_father_mapping = hash_table_make(hash_pointer, 0);

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
      fprintf(stderr, "mark_this_node_and_its_predecessors_in_the_dg_as_useful: vertex %#x marked, statement ordering (%#x).\n",
              (int) v,      
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
   /* First mark the dependence graph predecessors: */
   /* Get the dependence list for this statement: */
   vertex the_statement_vertex = (vertex) hash_get(ordering_to_dg_mapping,
                                                   (char *) statement_ordering(s));

   ifdebug(6)
      fprintf(stderr, "iterate_through_the_predecessor_graph, statement %#x (%#x).\n",
              (int) s, statement_ordering(s));
  
   if (the_statement_vertex != (vertex) HASH_UNDEFINED_VALUE)
      MAP(SUCCESSOR, a_successor,
          {
             dg_arc_label label = successor_arc_label(a_successor);
             /* Try to find at least one of the use-def chains between
                s and a successor: */
             MAP(CONFLICT, a_conflict,
                 {
                    /* Something is useful for the current statement if
                       it writes something that is used in the current
                       statement: */
                    if (action_read_p(conflict_source(a_conflict))
                        && action_write_p(conflict_sink(a_conflict))) {
                       /* Mark that we will visit the node that defined a
                          source for this statement, if not already
                          visited: */
                       set_add_element(elements_to_visit,
                                       elements_to_visit,
                                       (char *) ordering_to_statement(dg_vertex_label_statement(a_successor)));
                       /* One use-def is enough: */
                       break;
                    }
                 },
                    dg_arc_label_conflicts(label));
          },
             vertex_successors(the_statement_vertex));

   {
      /* Mark the father too for control dependences: */
      statement father =
         (statement) hash_get(statement_to_statement_father_mapping, (char *) s);
      if (father != (statement) HASH_UNDEFINED_VALUE)
         set_add_element(elements_to_visit, elements_to_visit, (char *) father);
   }
   {
      /* And if the statement is in an unstructured, mark all the
         unstructured nodes predecessors as useful. It is quite
         conservative to deal with control dependences... */
      if (bound_control_father_p(s)) {
         control control_father = load_control_father(s);
         MAP(CONTROL, a_control,
             {
                set_add_element(elements_to_visit,
                                elements_to_visit,
                                (char *) control_statement(a_control));
             }, control_predecessors(control_father));
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
   
   if (get_debug_level() >= 5) {
      fprintf(stderr, "use_def_deal_if_useful: statement %#x (%#x)\n",
              (int) s, statement_ordering(s));
      print_text(stderr, text_statement(get_current_module_entity(), 0, s));
   }

   /* The possible reasons to have useful code: */
   /* - the statement does an I/O: */
   this_statement_has_an_io_effect = statement_io_effect_p(the_proper_effects, s);
   /* - the statement writes a procedure argument, so the value may be
      used by another procedure: */
   this_statement_writes_a_procedure_argument =
      statement_write_argument_of_module_effect_p(s,
                                                  get_current_module_entity(),
                                                  the_proper_effects);

   if (get_debug_level() >= 6) {
      if (this_statement_has_an_io_effect)
         fprintf(stderr, "Statement %#x has an io effect.\n", (int) s);
      if (this_statement_writes_a_procedure_argument)
         fprintf(stderr,
                 "Statement %#x writes an argument of its procedure.\n",
                 (int) s);
   }
   
   if (this_statement_has_an_io_effect
      || this_statement_writes_a_procedure_argument)
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
      if (get_debug_level() >= 6)
         fprintf(stderr, "remove_this_statement_if_useless removes statement %#x (%#x).\n", (int) s, statement_ordering(s));
   }
}


void static
remove_all_the_non_marked_statements(statement s)
{
   gen_recurse(s, statement_domain,
               /* Since statements can be nested, only remove in a
                  bottom-up way: */
               use_def_true_filter,
               remove_this_statement_if_useless);
}


void
use_def_elimination_on_a_statement(statement s)
{
   the_useful_statements = set_make(set_pointer);
   init_control_father();
   
   /* pips_assert("use_def_elimination_on_a_statement", */
   ordering_to_dg_mapping = compute_ordering_to_dg_mapping(dependence_graph);

   build_statement_to_statement_father_mapping(s);

   /* Mark as useful the seed statements: */
   gen_recurse(s, statement_domain,
               use_def_true_filter,
               use_def_deal_if_useful);

   /* Propagate the usefulness through all the predecessor graph: */
   propagate_the_usefulness_through_the_predecessor_graph();
   
   remove_all_the_non_marked_statements(s);

   hash_table_free(ordering_to_dg_mapping);
   hash_table_free(statement_to_statement_father_mapping);
   close_control_father();
   set_free(the_useful_statements);
}


bool
use_def_elimination(char * module_name)
{
   statement module_statement;

   debug_on("USE_DEF_ELIMINATION_DEBUG_LEVEL");

   /* Get the true ressource, not a copy. */
   module_statement =
      (statement) db_get_memory_resource(DBR_CODE, module_name, TRUE);

   /* Get the data dependence graph: */
   dependence_graph =
      (graph) db_get_memory_resource(DBR_DG, module_name, TRUE);

   /* The proper effect to detect the I/O operations: */
   the_proper_effects = effectsmap_to_listmap((statement_mapping)
                                              db_get_memory_resource(
                                                 DBR_PROPER_EFFECTS,
                                                 module_name,
                                                 TRUE)); 

   set_current_module_statement(module_statement);
   set_current_module_entity(local_name_to_top_level_entity(module_name));

   use_def_elimination_on_a_statement(module_statement);

   /* Reorder the module, because some statements have been deleted.
      Well, the order on the remaining statements should be the same,
      but by reordering the statements, the number are consecutive. Just
      for pretty print... :-) */
   module_reorder(module_statement);

   DB_PUT_MEMORY_RESOURCE(DBR_CODE, strdup(module_name), module_statement);

   reset_current_module_statement();
   reset_current_module_entity();

   debug(2, "use_def_elimination", "done for %s\n", module_name);
   debug_off();

   /* Should have worked: */
   return TRUE;
}
