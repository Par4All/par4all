
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

#include "sac-local.h" /* needed because sac.h may not exist when 
                        * simdizer.c is compiled */
#include "sac.h"

#include "dg.h"

typedef dg_arc_label arc_label;
typedef dg_vertex_label vertex_label;

#include "graph.h"

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "ricedg.h"
#include "semantics.h"
#include "transformations.h"

static graph dependence_graph;

static hash_table matches = NULL;

void init_statement_matches_map(list l)
{
   init_operator_id_mappings();
   init_tree_patterns();

   matches = hash_table_make(hash_pointer, 0);
   MAP(STATEMENT,
       s,
   {
      list match = match_statement(s);

      if (match != NIL)
	 hash_put(matches, (void *)s, (void *)match);
   },
       l);
}

void free_statement_matches_map()
{
   hash_table_free(matches);
}

list get_statement_matches(statement s)
{
   list m;

   m = (list)hash_get(matches, (void*)s);

   return (m == HASH_UNDEFINED_VALUE) ? NIL : m; 
}

match get_statement_match_of_kind(statement s, int kind)
{
   list l = get_statement_matches(s);

   for( ; l!=NIL; l=CDR(l))
   {
      match m = MATCH(CAR(l));
      if (m->type == kind)
	 return m;
   }

   return NULL;
}

list get_statement_matching_types(statement s)
{
   list m = get_statement_matches(s);
   list k = NIL;

   for( ; m != NIL; m = CDR(m))
      k = CONS(INT, MATCH(CAR(m))->type, k);

   return k;
}

static hash_table successors;

void init_statement_successors_map(list l)
{
   successors = hash_table_make(hash_pointer, 0);

   MAP(VERTEX,
       a_vertex,
   {
      list succ;
      statement s = vertex_to_statement(a_vertex);

      if (gen_find_eq(s, l) == gen_chunk_undefined)
	 continue;

      succ = vertex_successors(a_vertex);
      if (succ != NIL)
	 hash_put(successors, (void *)s, (void *)succ);
   },
       graph_vertices(dependence_graph));
}

void free_statement_successors_map()
{
   hash_table_free(successors);
}

/* is s2 a successor of s1 ? */
bool successor_p(statement s1, statement s2)
{
   list succ;

   succ = (list)hash_get(successors, (void*)s1);

   if (succ != HASH_UNDEFINED_VALUE)
   {
      MAP(SUCCESSOR, 
	  s,
      {
	 if (vertex_to_statement(successor_vertex(s)) == s2)
	    return TRUE;
      }, 
	  succ);
   }
   return FALSE;
}

static bool move_allowed_p(list group, statement s)
{ 
   cons * i;

   for(i = group; (i != NIL) && (STATEMENT(CAR(i)) != s); i = CDR(i))
   {
      if (successor_p(STATEMENT(CAR(i)), s))
	 return FALSE;
   }

   return TRUE;
}

/* Transform the code to use SIMD instructions. The input statement
 * should be a sequence of simple statements.
 * If the block contains loops, ifs, or any other control flow 
 * instruction, the result may not be correct.
 * Non-simple calls should be handled properly, but WILL NOT be 
 * optimized to use SIMD instructions.
 */
static void simdize_simple_statements(statement s)
{
   cons * i;
   list seq;
   list sinfo; /* <statement_info> */
   list sinfo_begin;
   list newseq;

   if (!instruction_sequence_p(statement_instruction(s)))
      /* not much we can do with a single statement, or with
       * "complex" statements (ie, with tests/loops/...)
       */
      return;

   seq = sequence_statements(instruction_sequence(statement_instruction(s)));
   init_statement_matches_map(seq);
   init_statement_successors_map(seq);
   reset_argument_info();

   sinfo = sinfo_begin = CONS(NULL, NULL, NIL);

   /* Traverse to list to group isomorphic statements */
   for( i = seq;
	i != NIL;
	i = CDR(i) )
   {
      cons * j, * p;
      cons * group_first, * group_last;
      statement si = STATEMENT(CAR(i));
      list group_matches;

      /* Initialize current group */
      group_first = i;
      group_last = i;

      /* if this is not a recognized statement (ie, no match), skip it */
      group_matches = get_statement_matching_types(si);
      if (group_matches == NIL)
	 continue;

      /* try to find all the compatible isomorphic statements after the
       * current statement
       */
      for( j = CDR(group_last), p = NIL;
	   j != NIL;
	   p = j, j = CDR(j) )
      {
	 statement sj = STATEMENT(CAR(j));
	 list m_sj;

	 /* if this is not a recognized statement (ie, no match), skip it */
	 m_sj = get_statement_matching_types(sj);
	 if (m_sj == NIL)
	    continue;

	 /* if the matches for statement sj and for the group have a non-empty
	  * intersection, and the move is legal (ie, does not break dependency
	  * chain) then we can add the statement sj to the group.
	  */
	 gen_list_and(&m_sj, group_matches);
	 if ( (m_sj!=NIL) &&
	      move_allowed_p(group_first, sj) )
	 {
	    if (j != CDR(group_last))
	    {
	       /* do the move */
	       CDR(p) = CDR(j);
	       CDR(group_last) = CONS(STATEMENT, sj, CDR(group_last));
	       
	       /* keep searching from the next one */
	       j = p;
	    }

	    group_last = CDR(group_last);
	    gen_free_list(group_matches);
	    group_matches = m_sj;
	 }
      }

      /* the previous group of isomorphic statements is complete. 
       * we can now generate SIMD statement info.
       * group is delimited by group_first and group_last (included)
       */
      CDR(sinfo) = make_simd_statements(group_matches, 
					group_first, 
					group_last);
      while(CDR(sinfo) != NIL)
	 sinfo = CDR(sinfo);

      /* skip what has already been matched */
      i = group_last;
   }

   /* Now, based on the statement information gathered, 
    * generate the actual code (new sequence of statements)
    */
   newseq = generate_simd_code(CDR(sinfo_begin));

   /* Free the list of statements info */
   gen_free_list(sinfo_begin);

   /* Set the new list as the statements' instructions */
   sequence_statements(instruction_sequence(statement_instruction(s))) = newseq;
   gen_free_list(seq);

   free_statement_matches_map();
   free_statement_successors_map();
}

static bool simd_simple_sequence_filter(statement s)
{
   instruction i;
   cons * j;

   /* Make sure this is a sequence */
   i = statement_instruction(s);
   if (!instruction_sequence_p(i))
      return TRUE; /* keep searching recursively */
   
   /* Make sure all statements are calls */
   for( j = sequence_statements(instruction_sequence(i));
	j != NIL;
	j = CDR(j) )
   {
      if (!instruction_call_p(statement_instruction(STATEMENT(CAR(j)))))
	 return TRUE; /* keep searching recursively */
   }
   
   /* Try to parallelize this */
   simdize_simple_statements(s);

   return FALSE;
}

static void simd_simple_sequence_rewrite(statement s)
{
   return;
}

bool simdizer(char * mod_name)
{
   /* get the resources */
   statement mod_stmt = (statement)
      db_get_memory_resource(DBR_CODE, mod_name, TRUE);

   set_current_module_statement(mod_stmt);
   set_current_module_entity(local_name_to_top_level_entity(mod_name));
   dependence_graph = 
      (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);

   debug_on("SIMDIZER_DEBUG_LEVEL");
   /* Now do the job */
  
   gen_recurse(mod_stmt, statement_domain,
	       simd_simple_sequence_filter, simd_simple_sequence_rewrite);

   pips_assert("Statement is consistent after SIMDIZER", 
	       statement_consistent_p(mod_stmt));

   /* Reorder the module, because new statements have been added */  
   module_reorder(mod_stmt);
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name, mod_stmt);
 
   /* update/release resources */
   reset_current_module_statement();
   reset_current_module_entity();

   debug_off();

   return TRUE;
}
