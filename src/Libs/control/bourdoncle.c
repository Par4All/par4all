/* bourdoncle.c

   Decomposition of a CFG into SCC's and sub-SCC's. Algorithm 3.9, Page
   43, Francois Bourdoncle's PhD. Define heuristically SCC and sub-SCC
   heads.

   Build a set nof new data structures to represent any CFG as a recursive
   stucture of CFG's and DAG's.

 $Id$

 $Log: bourdoncle.c,v $
 Revision 1.1  2002/05/28 15:04:03  irigoin
 Initial revision
 */

#include <stdio.h>
#include <strings.h>

#include "linear.h"

#include "genC.h"
#include "ri.h"
#include "ri-util.h"
#include "control.h"
/* #include "properties.h" */

#include "misc.h"

/* #include "constants.h" */


/* Data structures for Bourdoncle's heuristics:
 *
 * dfn = depth first number
 *
 * num = current vertex number
 *
 * vertex_stack = stack of visited nodes
 *
 */

static hash_table dfn = hash_table_undefined;

static void reset_dfn(control c)
{
  hash_put(dfn, (void *) c, (void *) 0);
}

static int get_dfn(control c)
{
  int d = 0;
  
  if((d = (int) hash_get(dfn, (void *) c)) == (int) (HASH_UNDEFINED_VALUE))
    pips_internal_error("No dfn value for control %p\n", c);

  return d;
}

static void update_dfn(control c, int d)
{
  hash_update(dfn, (void *) c, (void *) d);
}


static int num = 0;

DEFINE_LOCAL_STACK(vertex, control)



     /* Code replicated from semantics/unstructured.c to be able to taylor
        it to the exact needs. */
static void print_control_node(control c)
{
  fprintf(stderr,
	  "ctr %p, %d preds, %d succs: %s", 
          c,
	  gen_length(control_predecessors(c)),
	  gen_length(control_successors(c)),
	  statement_identification(control_statement(c)));
  fprintf(stderr,"\tsuccessors:\n");
  MAP(CONTROL, s, {
    fprintf(stderr, "\t\t%p %s", s,
	    statement_identification(control_statement(s)));
  }, control_successors(c));
  fprintf(stderr,"\tpredecessors:\n");
  MAP(CONTROL, p, {
    fprintf(stderr, "\t\t%p %s", p,
	    statement_identification(control_statement(p)));
  }, control_predecessors(c));
  fprintf(stderr, "\n");
}

static void print_control_nodes(list l)
{
  MAP(CONTROL, c, {
    fprintf(stderr, "%x, %s", (unsigned int) c,
	    statement_identification(control_statement(c)));
  }, l);
  fprintf(stderr, "\n");
}

/* Replication of unstructured (i.e. CFG) and control nodes
   (i.e. vertices) */

static hash_table replicate_map = hash_table_undefined;

control control_shallow_copy(control c)
{
  control new_c = control_undefined;

  new_c = make_control(control_statement(c),
		       gen_copy_seq(control_successors(c)),
		       gen_copy_seq(control_predecessors(c)));
  hash_put(replicate_map, (void *) c, (void *) new_c);
  
  return new_c;
}

static void control_translate_arcs(control c)
{
  MAPL(c_c,
  {
    control old_c = CONTROL(CAR(c_c));
    control new_c = hash_get(replicate_map, (void *) old_c);
    pips_assert("new_c is defined", new_c!=(control) HASH_UNDEFINED_VALUE);
    CONTROL(CAR(c_c)) = new_c;
  }
      , control_predecessors(c));
  MAPL(c_c,
  {
    control old_c = CONTROL(CAR(c_c));
    control new_c = hash_get(replicate_map, (void *) old_c);
    pips_assert("new_c is defined", new_c!=(control) HASH_UNDEFINED_VALUE);
    CONTROL(CAR(c_c)) = new_c;
  }
      , control_successors(c));
}

static control control_to_replicate(control old_c)
{
  control new_c =  hash_get(replicate_map, (void *) old_c);
  pips_assert("new_c is defined", new_c!=(control) HASH_UNDEFINED_VALUE);
  return new_c;
}

unstructured unstructured_shallow_copy(unstructured u, hash_table ancestor_map)
{
  unstructured new_u = unstructured_undefined;
  
  replicate_map = hash_table_make(hash_pointer, 0);

  /* Do not go down into nested unstructured */
  gen_multi_recurse(u, statement_domain, gen_false, gen_null,
		    control_domain, gen_true, control_shallow_copy, NULL);

  /* Update arcs */
  gen_multi_recurse(u, statement_domain, gen_false, gen_null,
		    control_domain, gen_true, control_translate_arcs, NULL);

  /* Generate new unstructured with relevant entry and exit nodes */
  new_u = make_unstructured(control_to_replicate(unstructured_control(u)),
			    control_to_replicate(unstructured_exit(u)));
  /* Generate ancestor_map as the inverse function of replicate_map */
  HASH_MAP(old_n, new_n,{
    void * really_old_n;
    
    if((really_old_n = hash_get(ancestor_map, old_n))==HASH_UNDEFINED_VALUE)
      really_old_n = old_n;
    hash_put(ancestor_map, new_n, really_old_n);
    
  }, replicate_map);
  
  hash_table_free(replicate_map);
  replicate_map = hash_table_undefined;

  return new_u;
}

/* Build a new unstructured (CFG) for partition. vertex is the entry and
   exit point. New nodes must be allocated because the parent graph is
   untouched. vertex is supposed to be included into partition. */
unstructured partition_to_unstructured(control vertex, list partition)
{
  unstructured new_u = unstructured_undefined;
  
  replicate_map = hash_table_make(hash_pointer, 0);

  /* Create the translation table replicate_map */
  MAP(CONTROL, c,
  {
    (void) control_shallow_copy(c);
  }
      , partition);

  /* Generate new unstructured with relevant entry and exit node vertex */
  new_u = make_unstructured(control_to_replicate(vertex),
			    control_to_replicate(vertex));

  /* Update arcs */
  MAP(CONTROL, c,
  {
    /* control c_new = control_to_replicate(c); */
    MAPL(c_c,
    {
      control old_c = CONTROL(CAR(c_c));
      control new_c = control_undefined;

      if(gen_in_list_p(old_c, partition)) {
	new_c = hash_get(replicate_map, (void *) old_c);
	pips_assert("new_c is defined", new_c!=(control) HASH_UNDEFINED_VALUE);
      }
      else {
	/* This predecessor is irrelevant */
	new_c = make_control(statement_undefined, NIL, NIL);
      }
	
      CONTROL(CAR(c_c)) = new_c;
    }
	 , control_predecessors(c));
    MAPL(c_c,
    {
      control old_c = CONTROL(CAR(c_c));
      control new_c = control_undefined;

      if(gen_in_list_p(old_c, partition)) {
	new_c = hash_get(replicate_map, (void *) old_c);
	pips_assert("new_c is defined", new_c!=(control) HASH_UNDEFINED_VALUE);
      }
      else {
	/* This successor is irrelevant */
	new_c = make_control(statement_undefined, NIL, NIL);
      }
	
      CONTROL(CAR(c_c)) = new_c;
    }
	 , control_successors(c));
  }
      , partition);

  hash_table_free(replicate_map);
  replicate_map = hash_table_undefined;

  return new_u;
}


/* Decomposition of control flow graph u into a DAG new_u and two
   mappings. Mapping scc_map maps nodes of u used to break cycles to the
   unstructured representing these cycles. Mapping ancestor_map maps nodes
   used in DAG new_u or in unstructured refered to by scc_mapp to nodes in
   u. */
unstructured bourdoncle_partition(unstructured u,
				  hash_table *p_ancestor_map,
				  hash_table * p_scc_map)
{
  list partition = NIL;
  control root = control_undefined;
  /* DAG derived from u by elimination all cycles */
  unstructured new_u = unstructured_undefined;
  /* mapping from nodes in the new unstructured to the node in the input unstructured */
  hash_table ancestor_map = hash_table_make(hash_pointer, 0);
  /* mapping from nodes in u, used as cycle breakers, to the corresponding
     scc represented as an unstructured */
  hash_table scc_map = hash_table_make(hash_pointer, 0);
  
  ifdebug(2) {
    pips_debug(2, "Begin for nodes:\n");
    /* Do not go down into nested unstructured */
    gen_multi_recurse(u, statement_domain, gen_false, gen_null,
		      control_domain, gen_true, print_control_node, NULL);
    pips_debug(2, "With entry nodes\n");
    print_control_node(unstructured_control(u));
    pips_debug(2, "And exit node\n");
    print_control_node(unstructured_exit(u));
  }

  make_vertex_stack();
  num = 0;
  dfn = hash_table_make(hash_pointer, 0);

  new_u = unstructured_shallow_copy(u, ancestor_map);

  /* Initialize dfn to 0 */
  gen_multi_recurse(new_u, statement_domain, gen_false, gen_null,
		    control_domain, gen_true, reset_dfn, NULL);

  /* Start from the entry point */
  root = unstructured_control(new_u);  
  (void) bourdoncle_visit(root, &partition, ancestor_map, scc_map);
    
  free_vertex_stack();
  hash_table_free(dfn);

  ifdebug(2) {
    pips_debug(2, "End with partition:");
    print_control_nodes(partition);
  }

  *p_ancestor_map = ancestor_map;
  *p_scc_map = scc_map;

  return new_u;
}

list bourdoncle_component(control vertex)
{
  list partition = NIL;
  
  ifdebug(2) {
    pips_debug(2, "Begin for node: \n");
    print_control_node(vertex);
  }

  MAP(CONTROL, c, 
  {
    if(get_dfn(c)==0) {
      (void) bourdoncle_visit(c, &partition);
    }
  }
      , control_successors(vertex));
  
  partition = CONS(CONTROL, vertex, partition);
  
  /* Build sub-unstructured associated to vertex and partition */

  ifdebug(2) {
    pips_debug(2, "End with partition: ");
    print_control_nodes(partition);
  }

  return partition;
}


int bourdoncle_visit(control vertex,
		     list * ppartition,
		     hash_table ancestor_map,
		     hash_table scc_map)
{
  int min = 0;
  int head = 0;
  bool loop = FALSE;
  
  vertex_push(vertex);
  num = num+1;
  head = num;
  update_dfn(vertex, num);
  
  MAP(CONTROL, succ, 
  {
    if(get_dfn(succ)==0) {
      min = bourdoncle_visit(succ, ppartition, ancestor_map, scc_map);
    }
    else {
      min = get_dfn(succ);
    }
    if(min<=head) {
      head = min;
      loop = TRUE;
    }
  }
      , control_successors(vertex));

  if (head==get_dfn(vertex)) {
    control e = vertex_pop();

    update_dfn(vertex, LONG_MAX);

    if(loop) {
      while(e!=vertex) {
	update_dfn(e, 0);
	e = vertex_pop();
      }
      *ppartition = gen_nconc(bourdoncle_component(vertex, ancestor_map, scc_map),
			      *ppartition);
    }
    else {
      *ppartition = CONS(CONTROL, vertex, *ppartition);
    }

  }
  
  return head;
}
