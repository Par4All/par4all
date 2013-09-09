#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif


#include <stdio.h>
#include <ctype.h>
#include <string.h>
#include "boolean.h"


#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "database.h"
#include "misc.h"
#include "text.h"
#include "text-util.h"
#include "ri-util.h"
#include "effects-util.h"

#include "effects-generic.h"
#include "effects-simple.h"

#include "pipsdbm.h"
#include "resources.h"
#include "control.h"
#include "conversion.h"
#include "properties.h"
#include "transformations.h"

/*static_control*/
#include "arithmetique.h"
#include "vecteur.h"
#include "contrainte.h"
#include "ray_dte.h"
#include "sommet.h"
#include "sg.h"
#include "sc.h"
#include "polyedre.h"
#include "matrix.h"

/* Pips includes	*/
#include "ri.h"
/* Types arc_label and vertex_label must be defined although they are
   not used */
typedef void * arc_label;
typedef void * vertex_label;
#include "graph.h"
#include "paf_ri.h"
#include "database.h"
#include "ri-util.h"
#include "constants.h"
#include "misc.h"
#include "control.h"
#include "text-util.h"
#include "pipsdbm.h"
#include "resources.h"
#include "paf-util.h"
#include "static_controlize.h"
#include "pocc-interface.h"

// Global variable
static statement_mapping Gsc_map;

static bool are_stmts_eq (statement stmt1, statement stmt2) {
  return statement_number(stmt1) == statement_number(stmt2);
}

/* Look for a loop in a statement */
bool condition_body_contains_loop_p(statement s) {
  if(statement_sequence_p(s)) {
    instruction inst = statement_instruction(s);
    sequence seq = instruction_sequence(inst);
    list stmts = sequence_statements(seq);
    FOREACH(statement, stmt, stmts) {
      if (statement_loop_p(stmt) || statement_forloop_p(stmt) || (statement_test_p(stmt) && condition_contains_loop_p(stmt)))
	  return true;
    }
  }
  else {
    if (statement_loop_p(s) || statement_forloop_p(s) || (statement_test_p(s) && condition_contains_loop_p(s)))
      return true;
  }
  return false;
}

/* Checks if a test contains a loop */
bool condition_contains_loop_p(statement s) {
  if (!statement_test_p(s))
    return false;
  instruction inst = statement_instruction(s);
  test t = instruction_test(inst);
  statement true_body = test_true(t);
  statement false_body = test_false(t);
  if (condition_body_contains_loop_p(true_body) || condition_body_contains_loop_p(false_body))
    return true;
  else
    return false;
}

/* Checks if there is at least one loop in the sequence.
   If not it removes the previously added pragma on the last_added_pragma statement.
*/
bool is_SCOP_rich (sequence seq, statement last_added_pragma, statement curStmt, bool* pragma_added_p) {
  // Should not occur but just in case
  if(!statement_with_pragma_p(last_added_pragma)) {
    return false;
  }
  bool startSearch = false;
  list stmts = sequence_statements(seq);
  FOREACH(statement, stmt, stmts) {
    // We start searching in sequence from the current statement
    if (are_stmts_eq(stmt, last_added_pragma)) {
      startSearch = true;
    }
    // We stop searching when we are at the current statement in the sequence
    else if (are_stmts_eq(stmt, curStmt)) {
      startSearch = false;
    }
    if (startSearch) {
      // If the statement is a loop or if it is a test containing a loop 
      if (statement_loop_p(stmt) || statement_forloop_p(stmt) || (statement_test_p(stmt) && condition_contains_loop_p(stmt))) {
	return true;
      }
    }
  }
  // Clearing pragma
  clear_pragma_on_statement(last_added_pragma);
  *pragma_added_p = false;
  return false;
}

/* Check if the SCoP is rich, add pragma "endscop" before stmt and change the value of pragma_added_p */
static void insert_endscop_before_stmt(statement stmt, bool* pragma_added_p, sequence seqInst, list stmts, statement last_added_pragma) {
  if (!is_SCOP_rich(seqInst, last_added_pragma, stmt, pragma_added_p))
    return;
  statement endscop = make_continue_statement(entity_empty_label());
  add_pragma_str_to_statement(endscop, "endscop", true);
  *pragma_added_p = false;
  sequence_statements(seqInst) = gen_insert_before(endscop, stmt, stmts);
}

/* Check if the SCoP is rich, add pragma "endscop" after stmt and change the value of pragma_added_p */
static void insert_endscop_after_stmt(statement stmt, bool* pragma_added_p, sequence seqInst, statement last_added_pragma) {
  if (!is_SCOP_rich(seqInst, last_added_pragma, stmt, pragma_added_p))
    return;
  statement endscop = make_continue_statement(entity_empty_label());
  add_pragma_str_to_statement(endscop, "endscop", true);
  *pragma_added_p = false;
  gen_insert_after(endscop, stmt, sequence_statements(seqInst));
}

/*Insert a pragma "endscop" before/after the statement stmt
  Insert a ";" statement in sequence with stmt thereby converting stmt to a sequence */
static void insert_endscop_in_sequence(statement stmt, bool* pragma_added_p, bool insertBefore) {
  statement endscop = make_continue_statement(entity_empty_label());
  add_pragma_str_to_statement(endscop, "endscop", true);
  *pragma_added_p = false;
  // Insert the continue statement after the statement parameter, thereby creating a sequence
  insert_statement(stmt, endscop, insertBefore);
}

/* Add a "scop" pragma to the statement stmt. Set pragma_added_p to true */
static void add_to_stmt(statement stmt, bool* pragma_added_p) {
  add_pragma_str_to_statement(stmt, "scop", true);
  *pragma_added_p = true;	 
}

/* Returns true if the instruction is a subroutine */
bool is_subroutine(instruction inst) {
  // Is the call's entity a functional type ?
  bool testFunction = false;
  // If it is a functional type, is the functional's result type void ? 
  bool isVoid = false;
  if (instruction_call_p(inst)) {
    testFunction = type_functional_p(entity_type(call_function(instruction_call(inst))));
    if(testFunction) {
      isVoid = type_void_p(functional_result(type_functional(entity_type(call_function(instruction_call(inst))))));
    }
  }
  return isVoid;
}

/* Called recursively, place pragmas on statements according to SCoP conditions */
static void pragma_scop(statement s, bool pragma_added_p, bool in_loop_p) {
  // The purpose of this variable is to keep track of the last stmt checked in the FOREACH loop. 
  // In case we go through the entire part of the code without finding any non-SCoP part, we go 
  // out of the FOREACH loop and add the end pragma to the last iterated statement
  // The second variable is used to keep the place where we add the last pragma in order to be able
  // to remove it if necessary
  statement save_stmt, last_added_pragma;

  instruction instTop = statement_instruction(s);

  switch (instruction_tag(instTop)) {
    // If that instruction is a sequence, we go through the statements of the sequence and look for SCoP part
  case is_instruction_sequence :
    {
      sequence seqInst = instruction_sequence(instTop);
      list stmts = sequence_statements(seqInst);
      FOREACH(statement, stmt, stmts) {
	// The current statement is saved
	save_stmt = stmt;
	instruction  inst = statement_instruction(stmt);
	static_control sc;
	switch (instruction_tag(inst)) {
	case is_instruction_loop :
	  {
	    loop l = instruction_loop(inst);
	    // We test if the body of the loop is a SCoP
	    sc = (static_control)GET_STATEMENT_MAPPING(Gsc_map, loop_body(l));
	    // If it is and we don't already have a pragma in the current "layer"
	    if (static_control_yes(sc) && !pragma_added_p) {
	      add_to_stmt(stmt, &pragma_added_p);
	      last_added_pragma = stmt;
	    }
	    // If it is not a SCoP and we already have a start pragma, that means we have reached
	    // the limit of our SCoP and therefore we put an end pragma
	    else if (!static_control_yes(sc) && pragma_added_p) {
	      // As adding a pragma to a statement is putting it on top of that statement, to place a 
	      // pragma at the end of a statement means we have to put it on the next statement
	      // Here we will create an continue_statement (a ";"), add an end pragma to it and add it to
	      // the list representing the current sequence of the statement. Please note that the continue
	      // statement is added before the current statement (which is not SCoP) 
	      insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	      // As the non-SCoP part is actually a loop, the body of that loop may contains SCoP, therefore we 
	      // call the function on that subset of the code
	      pragma_scop(stmt, pragma_added_p, true);
	    }
	    // If the loop is not a SCoP and we do not already have a start pragma, we can call the function again on
	    // that subset of the code without having to take care of the current SCoP zone (as there is none)
	    else if (!static_control_yes(sc) && !pragma_added_p) {
	      pragma_scop(stmt, pragma_added_p, true);	    
	    }
	  }
	  break;
	case is_instruction_test : 
	  {
	    static_control scCond = (static_control)GET_STATEMENT_MAPPING(Gsc_map, stmt);
	    bool testStatic = static_control_yes(scCond);
	    // If the test is static control : the condition and both true and false body have to fit the conditions
	    if (testStatic && !pragma_added_p) {
	      add_to_stmt(stmt, &pragma_added_p);
	      last_added_pragma = stmt;
	    }
	    // If the test is not static, we close the current SCoP and go explore the true and false body
	    else if (!testStatic && pragma_added_p) {
	      insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	      pragma_scop(test_true(instruction_test(inst)), pragma_added_p, in_loop_p);
	      pragma_scop(test_false(instruction_test(inst)), pragma_added_p, in_loop_p);
	    }
	    else if (!testStatic && !pragma_added_p) {
	      pragma_scop(test_true(instruction_test(inst)), pragma_added_p, in_loop_p);
	      pragma_scop(test_false(instruction_test(inst)), pragma_added_p, in_loop_p);
	    }
	  }
	  break;
	case is_instruction_call :
	  {
	    // Depending on this option we will consider function calls part of SCoP or not
	    bool across_call = get_bool_property("STATIC_CONTROLIZE_ACROSS_USER_CALLS");
	    if (return_statement_p(stmt) && pragma_added_p) {
	      insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	    }
	    else if ((!across_call && is_subroutine(inst)) || (across_call && !user_call_p(instruction_call(inst)))) {
	      if (pragma_added_p)
		insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	    }
	    else if (!declaration_statement_p(stmt)) {
	       sc = (static_control)GET_STATEMENT_MAPPING(Gsc_map, stmt);
	       if (static_control_yes(sc) && !pragma_added_p) {
		 add_to_stmt(stmt, &pragma_added_p);
		 last_added_pragma = stmt;
	       }
	       else if (!static_control_yes(sc) && pragma_added_p) {
		 insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	       }		 
	    }
	  }
	  break;
	default : 
	  {
	    sc = (static_control)GET_STATEMENT_MAPPING(Gsc_map, stmt);
	    // If we already have a start pragma and the current statement is not SCoP or if it is a function call,
	    // that puts an end to our current SCoP and as we did above, we add the end pragma
	    bool isRoutine = is_subroutine(inst);
	    bool across_call = get_bool_property("STATIC_CONTROLIZE_ACROSS_USER_CALLS");
	    if (pragma_added_p && (!static_control_yes(sc) || (isRoutine && !across_call))) {
	      insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	    }
	    // We go explore this statement
	    pragma_scop(stmt, pragma_added_p, in_loop_p);
	  }
	  break;
	}
      }
      // If we go into that block, it means that we had a start pragma, went through every statement in the list and everything is static control.
      // Therefore we just use the same technique, we add a continue statement AFTER the last statement encountered (save_stmt)
      if (pragma_added_p) {
	insert_endscop_after_stmt(save_stmt, &pragma_added_p, seqInst, last_added_pragma);
      }
    }
    break;
    // If it the loop is static control, we add a set of pragma to it else we call the function on the body
    // of the loop
  case is_instruction_loop :
    {
      loop loopAlone = instruction_loop(instTop);
      static_control scAlone = (static_control)GET_STATEMENT_MAPPING(Gsc_map, loop_body(loopAlone));
      if (static_control_yes(scAlone) && !pragma_added_p) {
	add_to_stmt(s, &pragma_added_p);
	last_added_pragma = s;
	insert_endscop_in_sequence(s, &pragma_added_p, false);	
      }
      else {
	pragma_scop(loop_body(instruction_loop(instTop)), pragma_added_p, true);
      }
    }
    break;
    // A while loop is never considered SCoP, so we just explore the body
  case is_instruction_whileloop :
    pragma_scop(whileloop_body(instruction_whileloop(instTop)), false, true);
    break;
    // Same principle as the loop
  case is_instruction_forloop : 
    {
      forloop forLoopAlone = instruction_forloop(instTop);
      static_control scAlone = (static_control)GET_STATEMENT_MAPPING(Gsc_map, forloop_body(forLoopAlone));
      if (static_control_yes(scAlone) && !pragma_added_p) {
	add_to_stmt(s, &pragma_added_p);
	last_added_pragma = s;
	insert_endscop_in_sequence(s, &pragma_added_p, false);
      }
      else {
	pragma_scop(forloop_body(instruction_forloop(instTop)), false, true);
      }
    }
    break;
    // If test is a SCoP (both conditions and true/false body fit the conditions) we place
    // pragma around the condition thereby creating a sequence with a ";"
    // else we call the function on both statement : true and false 
  case is_instruction_test :
    {
      static_control scCond = (static_control)GET_STATEMENT_MAPPING(Gsc_map, s);
      bool testStatic = static_control_yes(scCond);
      if (testStatic && !pragma_added_p) {
	add_to_stmt(s, &pragma_added_p);
	last_added_pragma = s;
	insert_endscop_in_sequence(s, &pragma_added_p, false);
      }
      else if (!testStatic && !pragma_added_p) {
	pragma_scop(test_true(instruction_test(instTop)), pragma_added_p, in_loop_p);
	pragma_scop(test_false(instruction_test(instTop)), pragma_added_p, in_loop_p);
      }
    }
    break;
  default :
    break;
  }
}

/**
 * use the result of control static to add pragmas for pocc
 * compiler , that pragmas delimit  control static parts (static
 * control region which can contains many loop nests 
 */

bool pocc_prettyprinter(char * module_name) {
  // Standard setup
  statement module_stat;
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, module_name, true));
  module_stat = get_current_module_statement();

  Gsc_map = (statement_mapping)db_get_memory_resource(DBR_STATIC_CONTROL, module_name, true);
  
  // First call of the function on the global statement
  pragma_scop(module_stat, false, false);
  
  // Standard procedure
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,module_stat);
  reset_current_module_entity();
  reset_current_module_statement();
   
  return true;
}
