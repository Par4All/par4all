
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

/* Tell if the statement is a "simple statement", ie a statement of the form:
 * X = Y oper Z, X = Y or X = -Y, where arguments are all references or 
 * ranges and oper is an operator (+, *, / or -)
 */
static bool simple_statement_p(statement s)
{
   cons * arg;
   syntax syn;
   instruction i = statement_instruction(s);
   entity function;
   
   /* Check if the statement is a call */
   if (!instruction_call_p(i))
      return FALSE;

   /* Check if the statement is a function call */
   function = call_function(instruction_call(i));
   if ( !type_functional_p(entity_type(function)) ||
        !ENTITY_ASSIGN_P(function) )
      return FALSE;

   /* First argument must be a reference */
   arg = call_arguments(instruction_call(i));
   if ( (arg == NIL) ||
	!syntax_reference_p(expression_syntax(EXPRESSION(CAR(arg)))))
      return FALSE;
   
   /* Second argument can be a reference, a constant, or a call to
    * a supported operator. */
   arg = CDR(arg);
   if (arg == NIL)
      return FALSE;
   syn = expression_syntax(EXPRESSION(CAR(arg)));
   if (syntax_call_p(syn))
   {
      cons * k;
      call c;

      /* Only supported operators and constants*/
      c = syntax_call(syn);
      if (!ENTITY_FOUR_OPERATION_P(call_function(c)) &&
	  !call_constant_p(c) )
	 return FALSE;

      /* All arguments MUST be references or constants (which are calls) */
      for( k = call_arguments(c);
	   k != NIL;
	   k = CDR(k) )
      {
	 syntax syn = expression_syntax(EXPRESSION(CAR(k)));

	 if ( syntax_call_p(syn) &&
	      !call_constant_p(syntax_call(syn)) )
	    return FALSE;
      }
   }

   return TRUE;
}

static bool same_function_call_p(call c1, call c2)
{
   return same_entity_lname_p(call_function(c1), call_function(c2));
}

/* Tell if the two statements are isomrophic, ie they perform
 * of the same "operations" in the same order.
 *
 * WARNING: Works only for simple statements.
 */
static bool isomorphic_p(statement s1, statement s2)
{
   call c1, c2;
   cons * arg1, * arg2;

   c1 = instruction_call(statement_instruction(s1));
   c2 = instruction_call(statement_instruction(s2));
   if (!same_function_call_p(c1, c2))
      return FALSE;

   for( arg1 = call_arguments(c1), arg2 = call_arguments(c2);
	(arg1 != NIL) && (arg2 != NIL);
	arg1 = CDR(arg1), arg2 = CDR(arg2) )
   {
      syntax syn1;
      syntax syn2;

      syn1 = expression_syntax(EXPRESSION(CAR(arg1)));
      syn2 = expression_syntax(EXPRESSION(CAR(arg2)));

      /* Must be the same kind of argument */
      if (syntax_tag(syn1) != syntax_tag(syn2))
	 return FALSE;

      /* If they are function calls, both arguments must be:
       *      - constants
       *  or 
       *      - functions with the same name
       */
      if ( syntax_call_p(syn1) &&
	   !( call_constant_p(syntax_call(syn1)) && 
	      call_constant_p(syntax_call(syn2)) ) &&
	   !same_function_call_p(syntax_call(syn1), syntax_call(syn2)) )
	 return FALSE;
   }

   return TRUE;
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

   if (!instruction_sequence_p(statement_instruction(s)))
      /* not much we can do with a single statement, or with
       * "complex" statements (ie, with tests/loops/...)
       */
      return;

   seq = sequence_statements(instruction_sequence(statement_instruction(s)));
   init_statement_successors_map(seq);

   /* Traverse to list to group isomorphic statements */
   for( i = seq;
	i != NIL;
	i = CDR(i) )
   {
      cons * j, * p;
      cons * group_first, * group_last;
      statement si = STATEMENT(CAR(i));

      /* Initialize current group */
      group_first = i;
      group_last = i;

      /* if this is not a "simple" statement, skip it */
      if (!simple_statement_p(si))
	 continue;

      /* try to find an isomorphic statement after the current
       * statement
       */
      for( j = CDR(group_last), p = NIL;
	   j != NIL;
	   p = j, j = CDR(j) )
      {
	 statement sj = STATEMENT(CAR(j));

	 /* if this is not a "simple" statement, skip it */	       
	 if (!simple_statement_p(sj))
	    continue;

	 /* if the statement are isomorphic and independant, then we
	  * can group them. So move it right after the other one, and
	  * keep searching.
	  */
	 if ( isomorphic_p(si,sj) &&
	      move_allowed_p(group_first, sj) )
	    /* would also need to check if the move is legal */
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

      /* the previous group of isomorphic statements is complete. 
       * we can now generate SIMD code.
       * group is delimited by group_first and group_last (included)
       */
      if (group_first != group_last)
      {
	 cons * tmp;
	 cons * statements = make_simd_statements(group_first, group_last);

	 /* replace the first element */
	 free_statement(STATEMENT(CAR(group_first)));
	 tmp = CDR(group_first);
	 CAR(group_first) = CAR(statements);
	 CDR(group_first) = CDR(statements);
	 free(statements); /* free the first node of the list, as we do not 
			    * use it anymore */
	 statements = group_first;
	 
	 /* bind the end of the list to the remaining of the statements */
	 while(CDR(statements) != NIL)
	    statements = CDR(statements);
	 CDR(statements) = CDR(group_last);

	 /* free the original statements */
	 group_first = tmp;
	 CDR(group_last) = NIL;
	 while(group_first != NIL)
	 {
	    cons * next = CDR(group_first);
	    free_statement(STATEMENT(CAR(group_first)));
	    free(group_first);
	    group_first = next;
	 }

	 /* Maybe this is useful */
	 fix_sequence_statement_attributes(s);

	 group_first = NIL;
	 group_last = statements;
      }

      /* skip what has already been matched */
      i = group_last;
   }
   
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
   string resp;

   /* get the resources */
   statement mod_stmt = (statement)
      db_get_memory_resource(DBR_CODE, mod_name, TRUE);

   set_current_module_statement(mod_stmt);
   set_current_module_entity(local_name_to_top_level_entity(mod_name));
   dependence_graph = 
      (graph) db_get_memory_resource(DBR_DG, mod_name, TRUE);

   debug_on("SIMDIZER_DEBUG_LEVEL");
   /* Now do the job */
  
   resp = user_request("The Great Wizard knows how to:\n"
		       "  1- unroll loops as needed\n"
		       "  2- SIMDize simple statements\n"
		       "Your choice ?\n");
   
   switch(resp[0])
   {
      case '1':
      {
	 printf("unrolling loops...\n");
	 simd_unroll_as_needed(mod_stmt);
      }
      break;
      
      case '2':
      {
	 printf("SIMDizing simple statements...\n");
	 gen_recurse(mod_stmt, statement_domain,
		     simd_simple_sequence_filter, simd_simple_sequence_rewrite);
      }
      break;

      default:
	 printf("The Great Wizard can go to bed and do nothing, too...\n");
   }

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

