/* Some utilities to deal with the control graph.
   It is mainly used by my unspaghettify.

   Ronan Keryell.
   */

#include <stdlib.h> 
#include <stdio.h> 
#include <string.h> 

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "text-util.h"
#include "misc.h"


/* Remove all the control nodes (with its statement) from c in the
   successor tree of c up to the nodes with more than 1 predecessor: */
void
remove_all_the_controls_in_a_control_successor_tree_till_some_predecessors(control c)
{
   /* For each successor of c: */
   MAP(CONTROL, a_successor,
       {
          if (gen_length(control_predecessors(a_successor)) <= 1)
             /* If there is no more than 1 predecessor, we can remove
                down this way: */
             remove_all_the_controls_in_a_control_successor_tree_till_some_predecessors(a_successor);
          else
             /* Just remove any predecessor reference of itself in
                this successor: */
             gen_remove(&control_predecessors(a_successor), c);
       },
          control_successors(c));

   gen_free_list(control_predecessors(c));
   control_predecessors(c) = NIL;
   gen_free_list(control_successors(c));
   control_successors(c) = NIL;
   gen_free(c);
}


/* Remove all the control sequences that are unreachable: */
void
remove_the_unreachable_controls_of_an_unstructured(unstructured u)
{
   list blocs = NIL;
   list control_remove_list = NIL;
   
   /* The entry point of the unstructured: */
   control entry_node = unstructured_control(u);

   CONTROL_MAP(c,
               {
                  if (c != entry_node)
                     /* Well, the entry node is guessed as
                        reachable... :-) */
                     if (control_predecessors(c) == NIL) {
                        /* A control without predecessor is
                           unreachable, so it is dead code: */
                        control_remove_list = CONS(CONTROL,
                                                   c,
                                                   control_remove_list);
                     }
               },
                  entry_node,
                  blocs);
   gen_free_list(blocs);

   /* Now remove all the marqued sequences: */
   MAP(CONTROL, c,
       {
          remove_all_the_controls_in_a_control_successor_tree_till_some_predecessors(c);
       },
          control_remove_list);
   gen_free_list(control_remove_list);
}


/* Replace each occurence of c in a_source_control_list_of_c with a
   a_dest_control_list_of_c: */
void
remove_a_control_from_a_list_and_relink(control c,
                                        list a_source_control_list_of_c,
                                        list a_dest_control_list_of_c,
                                        remove_a_control_from_a_list_and_relink_direction which_way)
{
   MAPL(a_control_list,
        {
           list *the_dest_of_a_source_list = NULL;
           list the_position_of_c = NIL;
           list the_position_before_c = NIL;
           list l = NIL;
           
           control a_dest_of_a_source = CONTROL(CAR(a_control_list));
           /* Now, find the corresponding dest in the source list
              with the same value as c: */
           switch(which_way) {
             case source_is_predecessor_and_dest_is_successor:
               the_dest_of_a_source_list = &control_successors(a_dest_of_a_source);
               break;
             case source_is_successor_and_dest_is_predecessor:
               the_dest_of_a_source_list = &control_predecessors(a_dest_of_a_source);
               break;
             default:
               pips_assert("remove_a_control_from_a_list_and_relink with not a good \"which_way\".\n", FALSE);
           }

           /* Find the reference to c in the the_dest_of_a_source_list: */
           the_position_before_c = NIL;
           MAPL(a_list,
                {
                   /* l is local to the MAPL... */
                   if (CONTROL(CAR(the_position_of_c = a_list)) == c)
                      break;
                   the_position_before_c = the_position_of_c;
                },
                   *the_dest_of_a_source_list);
           
           /* And add the a_dest_control_list_of_c instead of c: */
           /* First concatenate (with copy) a_dest_control_list_of_c
              before what follow c in the list: */
           l = gen_append(a_dest_control_list_of_c, CDR(the_position_of_c));

           /* Then place this list instead of c in
              *the_dest_of_a_source_list: */
           if (the_position_before_c == NIL)
              /* Modify the begin of the list: */
              *the_dest_of_a_source_list = l;
           else
              CDR(the_position_before_c) = l;

           /* Deallocate the cons that point to c: */
           CDR(the_position_of_c) = NIL;
           gen_free_list(the_position_of_c);
        },
           a_source_control_list_of_c);

   /* c is now in the memory nowhere: */
   control_predecessors(c) = NIL;
   control_successors(c) = NIL;
}


/* It removes a control node from its successor and predecessor list
   and relink the successor and the predecessor.   
   */
void
remove_a_control_from_an_unstructured(control c)
{
   list the_predecessors = control_predecessors(c);
   list the_successors = control_successors(c);

   int number_of_successors = gen_length(the_successors);
   
   /* Unlink from the predecessor. Note that a node may have more than
      one predecessor. Since we cannot discard an IF this way, we have
      at most 1 successor: */
   pips_assert("remove_a_control_from_an_unstructured: more than one successor", number_of_successors <= 1);
   remove_a_control_from_a_list_and_relink(c,
                                           the_predecessors,
                                           the_successors,
                                           source_is_predecessor_and_dest_is_successor);

   /* Unlink from the successor: */
   remove_a_control_from_a_list_and_relink(c,
                                           the_successors,
                                           the_predecessors,
                                           source_is_successor_and_dest_is_predecessor);
   
   /* Remove the control node: */
   gen_free(c);
}


/* Used to discard an unstructured without touching its
   statements. The statements are assumed to be referenced in another
   way: */
void
discard_an_unstructured_without_its_statements(unstructured u)
{
   list blocs = NIL;

   /* Protect the statements by unlinking them: */
   CONTROL_MAP(c,
               {
                  control_statement(c) = statement_undefined;
               },
               unstructured_control(u),
               blocs);
   gen_free_list(blocs);

   /* And then free the discard the unstructured: */
   gen_free(u);
}


/* Used to discard a control sequence without touching its statements.
 It also removes the reference to the sequence from the predecessors
 or the successors. */
void
discard_a_control_sequence_without_its_statements(control begin,
                                                  control end)
{
   control c;
   list successor_list;

   /* Unlink any extern reference to the control sequence: */
   MAP(CONTROL, a_predecessor,
       {
          gen_remove(&control_successors(a_predecessor),
                     begin);
       },
       control_predecessors(begin));
   
   MAP(CONTROL, a_successor,
       {
          gen_remove(&control_predecessors(a_successor) ,
                     end);
       },
       control_successors(end));
   
   for(c = begin; ; c = CONTROL(CAR(successor_list))) {
      /* To pass through the free: */
      successor_list = control_successors(c);
      
      pips_assert("discard_a_control_sequence_without_its_statements: not a sequence.", gen_length(successor_list) <= 1);
      
      /* Protect the statement: */
      control_statement(c) = statement_undefined;
      control_successors(c) = NIL;
      control_predecessors(c) = NIL;

      gen_free(c);

      if (c == end)
         break;
   }
}


/* Take a control sequence and return a list of all the statements in
   the sequence (in the same order... :-) ). */
list
generate_a_statement_list_from_a_control_sequence(control begin,
                                                  control end)
{
   control c;
   list the_statements_of_the_sequence = NIL;

   /* Because of the way CONS is working, reversed the walk through
      the sequence. */
   
   for(c = end; ; c = CONTROL(CAR(control_predecessors(c)))) {
      int number_of_predecessor = gen_length(control_predecessors(c));
      pips_assert("discard_a_control_sequence_without_its_statements: not a sequence.", number_of_predecessor <= 1);

      /* Add the statement to the list: */
      the_statements_of_the_sequence = CONS(STATEMENT,
                                            control_statement(c),
                                            the_statements_of_the_sequence);
      if (c == begin)
         break;
   }

   return the_statements_of_the_sequence;
}


