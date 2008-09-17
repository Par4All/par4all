
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

#include "sac-local.h" 

#include "sac.h"

#include "control.h"

#include "ricedg.h"

static graph dependence_graph;

static entity gSimdCurVector;
static entity gSimdVector;

/*
This function returns true if the statement is a simd loadsave
statement
 */
static bool simd_loadsave_stat_p(statement stat)
{
   if(!statement_call_p(stat))
   {
      return FALSE;
   }

   string funcName = entity_local_name(
      call_function(statement_call(stat)));

   if((!strncmp(funcName, SIMD_LOAD_NAME, SIMD_LOAD_SIZE)) ||
      (!strncmp(funcName, SIMD_GEN_LOAD_NAME, SIMD_GEN_LOAD_SIZE)) ||
      (!strncmp(funcName, SIMD_CONS_LOAD_NAME, SIMD_CONS_LOAD_SIZE)) ||
      (!strncmp(funcName, SIMD_SAVE_NAME, SIMD_SAVE_SIZE)) ||
      (!strncmp(funcName, SIMD_GEN_SAVE_NAME, SIMD_GEN_SAVE_SIZE)) ||
      (!strncmp(funcName, SIMD_CONS_SAVE_NAME, SIMD_CONS_SAVE_SIZE)))
      return TRUE;

   return FALSE;
}

/*
This function returns true if the statement is a simd
statement
 */
static bool simd_stat_p(statement stat)
{
   if(!statement_call_p(stat))
   {
      return FALSE;
   }

   string funcName = entity_local_name(
      call_function(statement_call(stat)));

   if(!strncmp(funcName, SIMD_NAME, SIMD_SIZE))
      return TRUE;

   return FALSE;
}

static bool listArgEq(list args1, list args2)
{
   if(gen_length(args1) != gen_length(args2))
      return FALSE;

   list pArgs2 = args2;

   MAP(EXPRESSION, exp1,
   {
      expression exp2 = EXPRESSION(CAR(pArgs2));

      if(!same_expression_p(exp1, exp2))
	 return FALSE;

      pArgs2 = CDR(pArgs2);
   }, args1);

   return TRUE;
}

static bool index_argument_conflict(list args, list l_reg)
{
   MAP(EXPRESSION, arg,
   {
      if(!expression_reference_p(arg))
         continue;

      MAP(EXPRESSION, ind,
      {

	 list ef = expression_to_proper_effects(ind);

	 MAP(EFFECT, indEff,
	 {
	    MAP(EFFECT, loopEff,
	    {
	       if(action_write_p(effect_action(loopEff)) &&
		  same_entity_p(effect_entity(indEff), effect_entity(loopEff)))
	       {
		  gen_free_list(ef);
		  return TRUE;
	       }
	    }, l_reg);
	 }, ef);

	 gen_free_list(ef);

      }, reference_indices(expression_reference(arg)));

   }, args);

   return FALSE;
}

/*
This function returns true if the arguments of the simd statement lsStat do 
not depend on the loop iteration
 */
static bool constant_argument_list_p(list args, statement lsStat, list stats, list l_reg)
{
   bool bConstArg = TRUE;

   MAP(VERTEX,
       a_vertex, 
   {
      statement stat1 = vertex_to_statement(a_vertex);

      if (stat1 != lsStat)
	 continue;

      MAP(SUCCESSOR,
	  suc,
      {
	 statement stat2 = vertex_to_statement(successor_vertex(suc));

         if ((gen_find_eq(stat2, stats) == gen_chunk_undefined) || 
	     (stat1 == stat2))
	    continue;

	 MAP(CONFLICT, 
	     c, 
	 {
	    // If stat2 is not a simd statement, then return FALSE
	    if(!simd_stat_p(stat2))
	    {
	       bConstArg = FALSE;
	    }
	    else if(!simd_loadsave_stat_p(stat2))
	    {
	    }
	    // If stat2 is a loadsave statement and that there is a conflict
	    // between the arguments of 
	    else if(!listArgEq(args, CDR(call_arguments(statement_call(stat2)))))
	    {
	       bConstArg = FALSE;
	    }
	 },
	     dg_arc_label_conflicts(successor_arc_label(suc)));
      },
	  vertex_successors(a_vertex));
   },
       graph_vertices(dependence_graph));

   if(index_argument_conflict(args, l_reg))
   {
      bConstArg = FALSE;
   }

   return bConstArg;
}

/* 
This function searches for simd load or save statements that can be
put out of the loop body. It stores these statements in
constArgs hash table.
 */
static bool searchForConstArgs(statement body, hash_table constArgs, list l_reg)
{
   if(!statement_block_p(body))
      return FALSE;

   MAP(STATEMENT, curStat,
   {
     // If if it is a simd load or save statement, ...
     if(simd_loadsave_stat_p(curStat))
      {
	list args = CDR(call_arguments(statement_call(curStat)));

	// If the arguments of the statement do not depend on the iteration,
	// then store the statement in constArgs
	if(constant_argument_list_p(args, curStat, statement_block(body), l_reg))
	{
	   hash_put(constArgs, curStat, args);
	}
      }
   }, statement_block(body));

   return TRUE;
}

static entity args_to_vector(list args, hash_table hash)
{
   entity vector = entity_undefined;

   HASH_MAP(cVect, cArgs,
   {
      if(listArgEq(args, cArgs))
      {
         vector = cVect;
	 break;
      }
   }, hash);

   return vector;
}

static void replace_simd_vector_rwt(expression e)
{
   syntax s = expression_syntax(e);

   if (!syntax_reference_p(s) ||
       !same_entity_p(reference_variable(expression_reference(e)),
		      gSimdCurVector))
      return;

   free_reference(syntax_reference(s));

   syntax_reference(s) = make_reference(gSimdVector, NIL);

   free_normalized(expression_normalized(e));
   expression_normalized(e) = normalized_undefined;
}

/*
This function replace the vector curVector by simdVector in the statement body
 */
static void replace_simd_vector(statement body, entity curVector, entity simdVector)
{
   gSimdCurVector = curVector;
   gSimdVector = simdVector;

   gen_recurse(body, expression_domain, gen_true, replace_simd_vector_rwt);
}

/*
This function replaces the index 5, 6, 7, 8 of string tFuncName
by 'L', 'O', 'A', 'D'.
 */
static string simd_load_function_name(string tFuncName)
{
   char * temp = tFuncName;
   temp[5] = 'L';
   temp[6] = 'O';
   temp[7] = 'A';
   temp[8] = 'D';

   return tFuncName;
}

/*
This function replaces the index 5, 6, 7, 8 of string tFuncName
by 'S', 'A', 'V', 'E'.
 */
static string simd_save_function_name(string tFuncName)
{
   char * temp = tFuncName;
   temp[5] = 'S';
   temp[6] = 'A';
   temp[7] = 'V';
   temp[8] = 'E';

   return tFuncName;
}

/*
This function inserts before and after the loop s the statement corresponding to the vector and arguments in argsToVect.
 */
static void insert_prelude_postlude(statement s, hash_table argsToVect, hash_table argsToFunc)
{
   list loadStats = NIL;
   list saveStats = NIL;

   HASH_MAP(vector,
	    args,
   {
      string tFuncName = (string) hash_get(argsToFunc, vector);

      string loadFuncName = simd_load_function_name(strdup(tFuncName));

      list callArgs = gen_concatenate(
         CONS(EXPRESSION, entity_to_expression(vector),NIL), args);

      statement loadStat = call_to_statement(make_call(
         get_function_entity(loadFuncName), callArgs));

      loadStats = CONS(STATEMENT, loadStat, loadStats);

      if(strncmp(loadFuncName, SIMD_CONS_LOAD_NAME, SIMD_CONS_LOAD_SIZE))
      {
         string saveFuncName = simd_save_function_name(strdup(tFuncName));

         callArgs = gen_concatenate(
            CONS(EXPRESSION, entity_to_expression(vector),NIL), args);

         statement saveStat = call_to_statement(make_call(
            get_function_entity(saveFuncName), callArgs));

         saveStats = CONS(STATEMENT, saveStat, saveStats);

	 free(saveFuncName);
      }

      free(loadFuncName);

   }, argsToVect);

   list oldStatDecls = statement_declarations(s);
   statement_declarations(s) = NIL;

   list newseq = gen_concatenate(loadStats, CONS(STATEMENT, copy_statement(s), NIL));

   list old = newseq;
   newseq = gen_concatenate(newseq, saveStats);
   gen_free_list(old);

   free_instruction(statement_instruction(s));

   // Replace the old statement instruction by the new one
   statement_instruction(s) = make_instruction_sequence(
      make_sequence(newseq));

   statement_label(s) = entity_empty_label();
   statement_number(s) = STATEMENT_NUMBER_UNDEFINED;
   statement_ordering(s) = STATEMENT_ORDERING_UNDEFINED;
   statement_comments(s) = empty_comments;
   statement_declarations(s) = oldStatDecls;
   statement_decls_text(s) = string_undefined;
}

/*
This function moves the statements in constArgs out of the loop body
 */
static void moveConstArgsStatements(statement s, statement body, hash_table constArgs)
{
   hash_table argsToVect = hash_table_make(hash_pointer, 0);
   hash_table argsToFunc = hash_table_make(hash_pointer, 0);

   list newSeq = NIL;
   list pNewSeq = NIL;

   // Go through the statements in the loop body to fill argsToVect
   // and argsToFunc and to replace simdVector, if necessary
   MAP(STATEMENT, curStat,
   {
      list args = (list) hash_get(constArgs, curStat);

      if(args == HASH_UNDEFINED_VALUE)
	 continue;

      entity curVector = reference_variable(expression_reference(
	 EXPRESSION(CAR(call_arguments(statement_call(curStat))))));

      entity simdVector = args_to_vector(args, argsToVect);

      if(simdVector == entity_undefined)
      {
         string funcName = entity_local_name(
            call_function(statement_call(curStat)));

	 hash_put(argsToVect, curVector, args);
	 hash_put(argsToFunc, curVector, funcName);
      }
      else
      {
	 replace_simd_vector(body, curVector, simdVector);
      }

   }, statement_block(body));

   // Create the new sequence and the statements in constArgs are
   // removed from the original sequence
   MAP(STATEMENT, curStat,
   {
      list args = (list) hash_get(constArgs, curStat);

      if(args == HASH_UNDEFINED_VALUE)
      {
         if(newSeq == NIL)
         {
	    newSeq = pNewSeq = CONS(STATEMENT, copy_statement(curStat), NIL);
         }
         else
         {
	    CDR(pNewSeq) = CONS(STATEMENT, copy_statement(curStat), NIL);
	    pNewSeq = CDR(pNewSeq);
         }
      }
   }, statement_block(body));

   gen_free_list(instruction_block(statement_instruction(body)));
   instruction_block(statement_instruction(body)) = newSeq;

   // Insert the statements remove from the sequence before 
   // and after the loop
   insert_prelude_postlude(s, argsToVect, argsToFunc);

   hash_table_free(argsToVect);
   hash_table_free(argsToFunc);
}

/*
This function is called for each statement and performs the 
simd_loop_const_elim on loop
 */
static void simd_loop_const_elim_rwt(statement s)
{
   instruction i = statement_instruction(s);
   statement body;

   hash_table constArgs = hash_table_make(hash_pointer, 0);

   bool success = FALSE;

   //We are only interested in loops
   switch(instruction_tag(i))
   {
      case is_instruction_loop:
	 body = loop_body(instruction_loop(i));
	 break;

      case is_instruction_whileloop:
	 body = whileloop_body(instruction_whileloop(i));
	 break;

      case is_instruction_forloop:
	 body = forloop_body(instruction_forloop(i));
	 break;

      default:
	 return;
   }

   // Load the read write effects of the loop in l_reg
   list l_reg = load_cumulated_rw_effects_list(s);

   // Search for simd load or save statements that can be
   // put out of the loop body. It stores these statements in
   // constArgs hash table
   success = searchForConstArgs(body, constArgs, l_reg);

   if(!success)
      return;

   // Move the statements in constArgs out of the loop body
   moveConstArgsStatements(s, body, constArgs);

   hash_table_free(constArgs);
}

static void move_declarations(entity new_fun, statement stat)
{
  list lEnt = NIL;
  list lModDecls = gen_copy_seq(code_declarations(value_code(entity_initial(new_fun))));

  type te = entity_type(new_fun);

  pips_assert("is functionnal", type_functional_p(te));

  functional fe = type_functional(te);
  int nparams = gen_length(functional_parameters(fe));

  MAP(ENTITY, curEnt,
  {
    int i = 0;
    bool move = TRUE;

    for(i = 1; i <= nparams; i++)
      {
	entity curPar = find_ith_parameter(new_fun, i);

	if(curPar == curEnt)
	  {
	    move = FALSE;
	    break;
	  }
      }

    if(!move)
      {
	continue;
      }

    if(!type_varargs_p(entity_type(curEnt)) &&
       !type_statement_p(entity_type(curEnt)) &&
       !type_area_p(entity_type(curEnt)))
      {
	lEnt = gen_nconc(lEnt, CONS(ENTITY, curEnt, NIL));

	gen_remove(&lModDecls, curEnt);
      }

  }, code_declarations(value_code(entity_initial(new_fun))));

  statement_declarations(stat) = gen_nconc(statement_declarations(stat),
					   lEnt);

  gen_free_list(code_declarations(value_code(entity_initial(new_fun))));

  code_declarations(value_code(entity_initial(new_fun))) = lModDecls;
}

/*
This phase looks for load or save statements that can be
put out of the loop body and move these statements, if possible.
 */
bool simd_loop_const_elim(char * module_name)
{
   statement module_stat;
   entity module;

   /* Get the code of the module. */
   set_current_module_entity(module_name_to_entity(module_name));
   module = get_current_module_entity();
   set_current_module_statement( (statement)
       db_get_memory_resource(DBR_CODE, module_name, TRUE) );
   module_stat = get_current_module_statement();
   set_cumulated_rw_effects((statement_effects)
	  db_get_memory_resource(DBR_CUMULATED_EFFECTS, module_name, TRUE));

   dependence_graph = (graph) db_get_memory_resource(DBR_DG, module_name, TRUE);

   debug_on("SIMD_LOOP_CONST_ELIM_SCALAR_EXPANSION_DEBUG_LEVEL");

   // Go through all the statements
   gen_recurse(module_stat, statement_domain,
	       gen_true, simd_loop_const_elim_rwt);

   move_declarations(get_current_module_entity(), module_stat);

   pips_assert("Statement is consistent after SIMD_SCALAR_EXPANSION", 
	       statement_consistent_p(module_stat));

   module_reorder(module_stat);
   DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, module_stat);
 
   debug_off();
    
   reset_current_module_entity();
   reset_current_module_statement();
   reset_cumulated_rw_effects();

   return TRUE;
}
