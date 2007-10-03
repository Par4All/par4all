
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

#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "polyedre.h"

#include "ricedg.h"
#include "semantics.h"
#include "transformations.h"
#include "control.h"

#include "effects-convex.h"

static graph dependence_graph;

static hash_table matches = NULL;

/*
This function stores in the matches hash table the
simd pattern matches for each statement
 */
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

/*
This function frees the matches hash table
 */
void free_statement_matches_map()
{
   hash_table_free(matches);
}

/*
This function gets the simd pattern matches
for the statement s
 */
list get_statement_matches(statement s)
{
   list m;

   m = (list)hash_get(matches, (void*)s);

   return (m == HASH_UNDEFINED_VALUE) ? NIL : m; 
}

/*
This function gets the simd pattern matches
for the statement s checking that the opcodeClass
of the match is kind
 */
match get_statement_match_of_kind(statement s, opcodeClass kind)
{
   list l = get_statement_matches(s);

   for( ; l!=NIL; l=CDR(l))
   {
      match m = MATCH(CAR(l));
      if (match_type(m) == kind)
	 return m;
   }

   return match_undefined;
}

/*
This function gets the simd pattern matches opcodeClass's
list for the statement s
 */
list get_statement_matching_types(statement s)
{
   list m = get_statement_matches(s);
   list k = NIL;

   for( ; m != NIL; m = CDR(m))
      k = CONS(OPCODECLASS, match_type(MATCH(CAR(m))), k);

   return k;
}

static hash_table successors;

/*
This function stores in the hash_table successors
the successors of each statement in the list l
 */
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

/*
This function frees the successors hash table
 */
void free_statement_successors_map()
{
   hash_table_free(successors);
}

/*
This function returns TRUE if there is a conflict between s1 and s2
that prevents the simdization of s2
 */
static bool successor_p(statement s1, statement s2, bool nonGroupStat)
{
   list succ;

   if(!instruction_call_p(statement_instruction(s1)))
   {
      return TRUE;
   }

   succ = (list)hash_get(successors, (void*)s1);

   if (succ != HASH_UNDEFINED_VALUE)
   {
      MAP(SUCCESSOR, 
	  s,
      {
	 if (vertex_to_statement(successor_vertex(s)) == s2)
	 {
	    MAP(CONFLICT, c,
	    {
	       // If there is a write-read conflict between
	       // s1-s2, then s2 can't be simdized
	       if((effect_write_p(conflict_source(c)) && 
		   effect_read_p(conflict_sink(c))))
	       {
		  return TRUE;
	       }

	       // If there is a read-write conflict or 
	       // a write-write conflict between s1-s2
	       // then s2 can't be simdized. Because, otherwise,
	       // since s1 doesn't belong to the same simd
	       // group as s2, then s2 would be executed before s1.
	       if(((effect_read_p(conflict_source(c)) && 
		   effect_write_p(conflict_sink(c)))) ||
                  ((effect_write_p(conflict_source(c)) && 
		   effect_write_p(conflict_sink(c)))))
	       {
		  if(nonGroupStat)
		  {
		     return TRUE;
		  }
	       }

	    }, dg_arc_label_conflicts(successor_arc_label(s)));
	 }
      }, 
	  succ);
   }
   return FALSE;
}

#define SIMD_COMMENT "SIMD_COMMENT_"

/*
This function return TRUE if SIMD_COMMENT_ has been
found in the statement comments. And, if so, num holds
the simd number of the statement.
 */
static bool getSimdCommentNum(statement stat, int * num) 
{
  string comments;
  char*  next_line;

  bool res = FALSE;

  if (!statement_with_empty_comment_p(stat)) 
  {
    comments = strdup(statement_comments(stat));
    next_line = strtok (comments, "\n");
    if (next_line != NULL) {
      do {
	sscanf(next_line, "c SIMD_COMMENT_%d", num);
	res = TRUE;

	next_line = strtok(NULL, "\n");
      }
      while (next_line != NULL);
    }
  }

  return res;
}

/*
This function returns TRUE if s can be added to the simd group
whose STATEMENT(CAR(group_first)) is the first element
 */
static bool move_allowed_p(list group_first, list group_last, statement s)
{ 
   cons * i;
   bool nonGroupStat = FALSE;

   for(i = group_first; (i != NIL) && (STATEMENT(CAR(i)) != s); i = CDR(i))
   {
      if (successor_p(STATEMENT(CAR(i)), s, nonGroupStat))
      {
	 return FALSE;
      }

      if((STATEMENT(CAR(group_last)) == STATEMENT(CAR(i))) && (!nonGroupStat))
      {
	 nonGroupStat = TRUE;
      }
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
static list simdize_simple_statements_pass1(list seq, float * simdCost)
{
   cons * i;
   list sinfo; /* <statement_info> */
   list sinfo_begin;
   list newseq;

   //argument info dependencies are local to each sequence -> RESET
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

      group_matches = get_statement_matching_types(si);

      //printf("si\n");print_statement(si);
      /* try to find all the compatible isomorphic statements after the
       * current statement
       */
      for( j = CDR(group_last), p = NIL;
	   j != NIL;
	   p = j, j = CDR(j) )
      {
	 statement sj = STATEMENT(CAR(j));
	 int num1, num2;
	 list m_sj;

	 //printf("sj\n");print_statement(sj);
	 /* if the two statements came from the same original statement
	  */
         if(getSimdCommentNum(si, &num1) &&
	    getSimdCommentNum(sj, &num2))
         {
            if(num1 != num2)
	    {
	        continue;
	    }
         }
         else
         {
            continue;
         }

	 m_sj = get_statement_matching_types(sj);

	 gen_list_and(&group_matches, m_sj);
	 if (move_allowed_p(group_first, group_last, sj))
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
	 }
      }

      /* if this is not a recognized statement (ie, no match), skip it */
      if (group_matches == NIL)
      {
	 list ind = group_first;
         /* No optimized opcode found... */
	 for( ;
	      (ind != CDR(group_last));
	      ind = CDR(ind) )
	 {
	    CDR(sinfo) = CONS(STATEMENT_INFO, 
			      make_nonsimd_statement_info(STATEMENT(CAR(ind))),
			      NIL);
	    sinfo = CDR(sinfo);
	 }
      }
      else
      {
         /* the previous group of isomorphic statements is complete. 
          * we can now generate SIMD statement info.
          * group is delimited by group_first and group_last (included)
          */
         CDR(sinfo) = make_simd_statements(group_matches, 
					   group_first, 
					   group_last);
      }

      while(CDR(sinfo) != NIL)
	 sinfo = CDR(sinfo);
      
      /* skip what has already been matched */
      i = group_last;
   }

   /* Now, based on the statement information gathered, 
    * generate the actual code (new sequence of statements)
    */
   newseq = generate_simd_code(CDR(sinfo_begin), simdCost);

   /* Free the list of statements info */
   gen_free_list(sinfo_begin);

   /* Set the new list as the statements' instructions */
   return newseq;
}

static list simdize_simple_statements_pass2(list seq, float * simdCost)
{
   cons * i;
   list sinfo; /* <statement_info> */
   list sinfo_begin;
   list newseq;

   //argument info dependencies are local to each sequence -> RESET
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
      {
	 sinfo = CDR(sinfo) = CONS(STATEMENT_INFO,
				   make_nonsimd_statement_info(STATEMENT(CAR(i))),
				   NIL);
	 continue;
      }
      //printf("si\n");print_statement(si);

      simd_fill_finalArgType(si);

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
	 //printf("sj\n");print_statement(sj);
	 /* if the matches for statement sj and for the group have a non-empty
	  * intersection, and the move is legal (ie, does not break dependency
	  * chain) then we can add the statement sj to the group.
	  */
	 gen_list_and(&m_sj, group_matches);

	 simd_fill_curArgType(sj);

	 if ( (m_sj!=NIL) &&
	      simd_check_argType() &&
	      move_allowed_p(group_first, group_last, sj) )
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

      simd_reset_finalArgType();
   }

   /* Now, based on the statement information gathered, 
    * generate the actual code (new sequence of statements)
    */
   newseq = generate_simd_code(CDR(sinfo_begin), simdCost);

   /* Free the list of statements info */
   gen_free_list(sinfo_begin);

   /* Set the new list as the statements' instructions */
   return newseq;
}

/*
This function tries to simdize with two algorithms.

simdize_simple_statements_pass1() attempts to simdize by grouping
the statements that have the same simd number.

simdize_simple_statements_pass2() attempts to simdize by grouping
as many statements as possible together.
 */
static void simdize_simple_statements(statement s)
{
   list seq = NIL;
   list copyseq = NIL;

   list newseq = NIL;
   list saveseq = NIL;

   float newSimdCost = 0;
   float saveSimdCost = 0;

   if (!instruction_sequence_p(statement_instruction(s)))
      /* not much we can do with a single statement, or with
       * "complex" statements (ie, with tests/loops/...)
       */
      return;

   seq = sequence_statements(instruction_sequence(statement_instruction(s)));
   copyseq = gen_copy_seq(seq);
   init_statement_matches_map(seq);
   init_statement_successors_map(seq);

   saveseq = simdize_simple_statements_pass1(seq, &saveSimdCost);

   newseq = simdize_simple_statements_pass2(copyseq, &newSimdCost);

   //printf("opcode cost1 %f\n", saveSimdCost);
   //printf("opcode cost2 %f\n", newSimdCost);

   if((saveSimdCost >= 0.0001) && (newSimdCost >= 0.0001))
   {
      gen_free_list(newseq);
      gen_free_list(saveseq);
   }
   else if(saveSimdCost <= newSimdCost)
   {
      sequence_statements(instruction_sequence(statement_instruction(s))) = saveseq;
      gen_free_list(seq);
      gen_free_list(newseq);
   }
   else
   {
      sequence_statements(instruction_sequence(statement_instruction(s))) = newseq;
      gen_free_list(seq);
      gen_free_list(saveseq);
    }

   gen_free_list(copyseq);

   free_statement_matches_map();
   free_statement_successors_map();
}

static bool simd_simple_sequence_filter(statement s)
{
   instruction i;

   /* Do not recurse through simple calls, for better performance */ 
   i = statement_instruction(s);
   if (instruction_call_p(i))
      return FALSE;
   else
      return TRUE;
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
	       simd_simple_sequence_filter, simdize_simple_statements);

   pips_assert("Statement is consistent after SIMDIZER", 
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
