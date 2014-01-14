/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

/*
 * This file contains functions used to deal with unstructured statements
 *
 * The similar functions in Amira's implementation are located in
 * points_to_analysis_general_algorithm.c.
 */

#include <stdlib.h>
#include <stdio.h>
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "ri-util.h"
#include "effects-util.h"
#include "constants.h"
#include "misc.h"
#include "properties.h"
#include "effects-generic.h"
#include "effects-simple.h"
//#include "effects-convex.h"
#include "newgen_set.h"
#include "points_to_private.h"
#include "alias-classes.h"

// FI: I have not cleaned up these functions yet

// FI: I avoid name conflicts by declaring them static

static bool statement_equal_p(statement s1, statement s2)
{
    return (statement_ordering(s1) == statement_ordering(s2));
}

/* test if a control belong to a set */
bool control_in_set_p(control c, set s)
{
  bool  in_p = false;
  SET_FOREACH(control, n, s) {
    if(statement_equal_p(control_statement(c), control_statement(n)))
      in_p = true;
  }
  return in_p;
}
 
bool control_equal_p(const void* vc1, const void* vc2)
{
  control c1 = (control)vc1;
  control c2 = (control)vc2;
  statement s1 = control_statement(c1);
  statement s2 = control_statement(c2);
  
  return statement_ordering(s1) == statement_ordering(s2);

}

/* create a key which is the statement number */
_uint control_rank( const void *  vc, size_t size)
{
  control c = (control)vc;
  statement s = control_statement(c);
  // FI: use asprintf?
  extern char * i2a(int);
  string key = strdup(i2a(statement_ordering(s)));
  return hash_string_rank(key,size);
}


/* A node is ready to be processed if its predecessors are not
 * reachable or processed.
 */
bool Ready_p(control c, set Processed, set Reachable)
{ 
  set Pred =  set_make(set_pointer);
  list Pred_l = control_predecessors(c);
  FOREACH(control, ctr, Pred_l) {
    Pred = set_add_element(Pred, Pred, (void*)ctr);
  }
  /* Pred = set_assign_list(Pred,gen_full_copy_list( Pred_l)); */
  bool ready_p = false;
  SET_FOREACH(control, p , Pred) {
    ready_p = set_belong_p(Processed,(void*) p) || !set_belong_p(Reachable,(void*) p); 
  }
  return ready_p;
}


/* A set containing all the successors of n that are ready to be processed */
set ready_to_be_processed_set(control n, set Processed, set Reachable)
{
   set Succ =  set_make(set_pointer);
   set rtbp =  set_make(set_pointer);
   list Succ_l = control_successors(n);
   FOREACH(control, ctr, Succ_l) {
    Succ = set_add_element(Succ, Succ, (void*)ctr);
  }
   /* Succ = set_assign_list(Succ, gen_full_copy_list(Succ_l)); */
   SET_FOREACH(control, p , Succ){
     if(Ready_p(p, Processed, Reachable))
       set_add_element(rtbp, rtbp, (void*)p);
   }
   return rtbp;
}


static pt_map control_to_points_to(control c, pt_map in, __attribute__ ((__unused__))bool store)
{
  pt_map out = new_pt_map();

  out = points_to_graph_assign(out, in);
  statement s = control_statement(c);
  out =  statement_to_points_to(s, in); // FI: seems to anihilate
  // previous assignment to out

  list Succ_l = gen_full_copy_list(control_successors(c));
  FOREACH(control, ctr, Succ_l) {
    out =  statement_to_points_to(control_statement(ctr), out);
    }
  out = merge_points_to_graphs(in, out);
 
  return out;
}


//
// FI: a little bit of clean-up from here down, in fact, renaming
// rather than cleaning
//


/*
  in: unreachable controls 
  out: points_to computed in a flow-insensitive way
*/
static pt_map cyclic_graph_to_points_to(set ctrls,
					pt_map in,
					bool store __attribute__ ((unused)))
{
  pt_map out = new_pt_map();
  set Pred = set_make(set_pointer);
  set Succ = set_make(set_pointer);
  /* sequence seq = sequence_undefined; */
  list /* seq_l = NIL, */ succ = NIL, pred = NIL;
  out = points_to_graph_assign(out, in);
  SET_FOREACH(control, c, ctrls) {
    out = statement_to_points_to(control_statement(c), out);
    pred = control_predecessors(c);
    /* Pred = set_assign_list(Pred, gen_full_copy_list(pred)); */
  }
  FOREACH(control, ctr, pred) {
    Pred = set_add_element(Pred, Pred, (void*)ctr);
  }
  SET_FOREACH(control, p, Pred) {
    out = statement_to_points_to(control_statement(p), out);
    succ = gen_copy_seq(control_successors(p));
  }
  
  
  Succ = set_assign_list(Succ, succ);
  SET_FOREACH(control, s, Succ) {
    out = statement_to_points_to(control_statement(s), out);
  }
  return out;
}


/*
   For unstructured code, we ignore the statements order and we construct a
   sequence of statments from the entry and the exit control. This sequence 
   will be analyzed in an flow-insensitive way.

   After the analysis of the unstructured we have to change the exact points-to
   relations into may points-to.

   FI: is the may flag enough? Are the comments above still consistent
with the code? Is this correct and cover all execution orders? How is a fix point reached?

 */
pt_map new_points_to_unstructured(unstructured uns,
				  pt_map pt_in_g,
				  bool __attribute__ ((__unused__))store)
{
  set pt_in = points_to_graph_set(pt_in_g);
  set pt_in_n = set_generic_make(set_private, points_to_equal_p,
                                 points_to_rank);
  set out = set_generic_make(set_private, points_to_equal_p,
                             points_to_rank);
  pt_map out_g = new_pt_map();
  list Pred_l = NIL, nodes_l = NIL,  nodes_l_exit = NIL ;
  list blocs = NIL ;
  set Pred =  set_generic_make(set_private, control_equal_p,
                                 control_rank);
  set Nodes =  set_generic_make(set_private, control_equal_p,
                                 control_rank);
  set Reachable =  set_generic_make(set_private, control_equal_p,
                                 control_rank);
  set Processed =  set_generic_make(set_private, control_equal_p,
                                 control_rank);
  set UnProcessed = set_generic_make(set_private, control_equal_p,
                                 control_rank);
  set rtbp =  set_generic_make(set_private, control_equal_p,
                                 control_rank);
  set rtbp_tmp = set_generic_make(set_private, control_equal_p,
                                 control_rank);
  set inter =  set_generic_make(set_private, control_equal_p,
                                 control_rank);
  set tmp =  set_generic_make(set_private, control_equal_p,
                                 control_rank);
  control entry = unstructured_control(uns) ;
  control exit = unstructured_exit(uns) ;
  Pred_l = control_predecessors(entry);
  Pred = set_assign_list(Pred, gen_full_copy_list(Pred_l));
  list trail = unstructured_to_trail(uns);

  /* Get all the nodes of the unstructured*/
#if 0
  CONTROL_MAP(c, {
      ;
    }, entry, nodes_l) ;
#endif
  FOREACH(control, ctrl, trail) {
    Nodes = set_add_element(Nodes, Nodes, (void*)ctrl);
  }
#if 0
  CONTROL_MAP(c, {
      ;
    }, exit, nodes_l_exit) ;
#endif
  nodes_l = gen_nconc(nodes_l, nodes_l_exit);
 
  /* Gel all the reachable nodes of the unstructured */
/* #if 0 */
  FORWARD_CONTROL_MAP(c, {
      Reachable = set_add_element(Reachable, Reachable, (void*)c);
    }, entry, blocs) ;
/* #endif */
 
  bool inter_p = set_intersection_p(Reachable, Pred);
  if(!inter_p)
    set_add_element(rtbp, rtbp, (void*)entry);
  else {
    /* The entry node is part of a cycle, rtbp is empty and the while
       loop below is soing to be skipped. Relation "out_g" must be
       properly initialized. */
    out_g = full_copy_pt_map(pt_in_g);
  }

  while(!set_empty_p(rtbp)) {
    rtbp_tmp = set_assign(rtbp_tmp,rtbp);
    SET_FOREACH(control, n , rtbp) {
      // to test the control's equality, I test if their statements are equals
      if ( statement_ordering(control_statement(n)) == statement_ordering(control_statement(entry)) ) {
        pt_in_n = set_assign(pt_in_n, pt_in);
        out = set_assign(out, pt_in_n);
      }
      Pred_l = NIL;
      Pred_l = control_predecessors(n);
      FOREACH(control, ctr, Pred_l) {
        Pred = set_add_element(Pred, Pred, (void*)ctr);
      }
      set_clear(Pred);
      /* Pred = set_assign_list(Pred, gen_full_copy_list(Pred_l)); */
      inter = set_intersection(inter,Reachable, Pred);
      out_g = make_points_to_graph(false, out);
      SET_FOREACH(control, p , inter) {
        out_g = control_to_points_to(p, out_g, true);
      }
      out_g = control_to_points_to(n, out_g, true);
      if(!control_in_set_p(n, Processed))
        Processed = set_add_element(Processed,Processed, (void*)n );
      rtbp_tmp = set_del_element(rtbp_tmp,rtbp_tmp,(void*)n);
      rtbp_tmp = set_union(rtbp_tmp,rtbp_tmp,
			   ready_to_be_processed_set(n, Processed, Reachable));
      set_clear(rtbp);
      rtbp = set_assign(rtbp,rtbp_tmp);
    }
  }

  UnProcessed = set_difference(UnProcessed, Reachable, Processed);
  out_g = cyclic_graph_to_points_to(UnProcessed, out_g, store);
  tmp = set_difference(tmp, Nodes, Reachable);
  SET_FOREACH(control, cc, tmp) {
    pt_in_n = set_clear(pt_in_n);
    out_g = statement_to_points_to(control_statement(cc), out_g);
  }

  free(blocs);
  free(nodes_l);
  statement exit_s = control_statement(exit);
  out_g = merge_points_to_graphs(out_g, pt_in_g);
  out_g = statement_to_points_to(exit_s, out_g); 
  return out_g;
}
