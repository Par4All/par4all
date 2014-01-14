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
#include "preprocessor.h"
#include "callgraph.h" // For compute_callees()
#include "effects-generic.h" // For compute_callees()
#include "accel-util.h" // For outliner()

#define PRAGMA_TEMP_BEGIN "nonscop"
#define PRAGMA_TEMP_END "endnonscop"
#define PREFIX_TEMP "nonscopf"

#define statement_has_this_pragma_string_p(stmt,str) \
(get_extension_from_statement_with_pragma(stmt,str)!=NULL)

static statement_mapping Gsc_map;
static int hasPragma = 0;

static bool statement_with_pragma_p_inv(statement s) {
  if (statement_with_pragma_p(s)) {
    hasPragma++;
  }
  return true;
}

static bool searchPragma (statement s) {
  gen_recurse(s, statement_domain, statement_with_pragma_p_inv, gen_null);
  if (hasPragma != 0) {
    hasPragma = 0;
    return true;
  }
  else {
    hasPragma = 0;
    return false;
  }
}

static void insert_endscop_before_stmt(statement stmt, bool* pragma_added_p, sequence seqInst, list stmts, statement last_added_pragma) {
  if (!is_SCOP_rich(seqInst, last_added_pragma, stmt, pragma_added_p))
    return;
  statement endscop = make_continue_statement(entity_empty_label());
  add_pragma_str_to_statement(endscop, PRAGMA_TEMP_END, true);
  *pragma_added_p = false;
  sequence_statements(seqInst) = gen_insert_before(endscop, stmt, stmts);
}

static void insert_endscop_after_stmt(statement stmt, bool* pragma_added_p, sequence seqInst, statement last_added_pragma) {
  if (!is_SCOP_rich(seqInst, last_added_pragma, stmt, pragma_added_p))
    return;
  statement endscop = make_continue_statement(entity_empty_label());
  add_pragma_str_to_statement(endscop, PRAGMA_TEMP_END, true);
  *pragma_added_p = false;
  gen_insert_after(endscop, stmt, sequence_statements(seqInst));
}

static void insert_endscop_in_sequence(statement stmt, bool* pragma_added_p, bool insertBefore) {
  statement endscop = make_continue_statement(entity_empty_label());
  add_pragma_str_to_statement(endscop, PRAGMA_TEMP_END, true);
  *pragma_added_p = false;
  // Insert the continue statement after the statement parameter, thereby creating a sequence
  insert_statement(stmt, endscop, insertBefore);
}

static void add_to_stmt(statement stmt, bool* pragma_added_p) {
  add_pragma_str_to_statement(stmt, PRAGMA_TEMP_BEGIN, true);
  *pragma_added_p = true;	 
}


void pragma_nonscop (statement s, bool pragma_added_p, bool in_loop_p) {
  
  statement save_stmt, last_added_pragma;
  instruction instTop = statement_instruction(s);
  
  switch (instruction_tag(instTop)) {

  case is_instruction_sequence :
    {
      sequence seqInst = instruction_sequence(instTop);
      list stmts = sequence_statements(seqInst);
      FOREACH(statement, stmt, stmts) {
	save_stmt = stmt;
	instruction inst = statement_instruction(stmt);
	static_control sc;
	switch (instruction_tag(inst)) {
	case is_instruction_loop : 
	  {
	    bool insideScop = searchPragma(stmt);
	    bool currentScop = statement_with_pragma_p(stmt);
	    loop l = instruction_loop(inst);
	    sc = (static_control)GET_STATEMENT_MAPPING(Gsc_map, loop_body(l));
	    if (!static_control_yes(sc) && !pragma_added_p && !insideScop) {
	      add_to_stmt(stmt, &pragma_added_p);
	      last_added_pragma = stmt;
	    }
	    else if (!static_control_yes(sc) && !pragma_added_p && insideScop && !currentScop) {
	      pragma_nonscop(stmt, pragma_added_p, true);
	    }
	    else if (static_control_yes(sc) && pragma_added_p) {
	      insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	    }
	  }
	  break;
	case is_instruction_test :
	  {
	    bool insideScop = searchPragma(stmt);
	    sc = (static_control)GET_STATEMENT_MAPPING(Gsc_map, stmt);
	    if (!static_control_yes(sc) && !pragma_added_p && !insideScop) {
	      add_to_stmt(stmt, &pragma_added_p);
	      last_added_pragma = stmt;
	    }
	    else if (!in_loop_p && !pragma_added_p && insideScop) {
	      pragma_nonscop(stmt, pragma_added_p, in_loop_p);
	    }
	    else if (static_control_yes(sc) && pragma_added_p) {
	      insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	    }
	  }
	  break;
	case is_instruction_whileloop :
	  {
	    bool insideScop = searchPragma(stmt);
	    if (!insideScop && !pragma_added_p) {
	      add_to_stmt(stmt, &pragma_added_p);
	      last_added_pragma = stmt;
	    }
	    else if (insideScop) {
	      pragma_nonscop(whileloop_body(instruction_whileloop(inst)), pragma_added_p, in_loop_p);
	    }
	  }
	  break;
	case is_instruction_call :
	  {
	    sc = (static_control)GET_STATEMENT_MAPPING(Gsc_map, stmt);
	    bool isDeclaration = declaration_statement_p(stmt);
	    if (return_statement_p(stmt) && pragma_added_p) {
	      insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	    }
	    else if (!static_control_yes(sc) && !pragma_added_p && !isDeclaration) {
	      add_to_stmt(stmt, &pragma_added_p);
	      last_added_pragma = stmt;
	    }
	    else if (static_control_yes(sc) && pragma_added_p) {
	      insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	    }
	  }
	  break;
	default :
	  {
	    sc = (static_control)GET_STATEMENT_MAPPING(Gsc_map, stmt);
	    bool currentScop = statement_with_pragma_p(stmt);
	    bool insideScop = searchPragma(stmt);
	    if (pragma_added_p && static_control_yes(sc)) {
	      insert_endscop_before_stmt(stmt, &pragma_added_p, seqInst, stmts, last_added_pragma);
	    }
	    else if (!pragma_added_p && !static_control_yes(sc) && !currentScop && !insideScop) {
	      add_to_stmt(stmt, &pragma_added_p);
	      last_added_pragma = stmt;
	    }
	  }
	  break;
	}
      }
      if (pragma_added_p) {
	insert_endscop_after_stmt(save_stmt, &pragma_added_p, seqInst, last_added_pragma);
      }
    }
    break;
  case is_instruction_loop :
    {
      loop loopAlone = instruction_loop(instTop);
      static_control scAlone = (static_control)GET_STATEMENT_MAPPING(Gsc_map, loop_body(loopAlone));
      bool insideScop = searchPragma(s);
      bool currentScop = statement_with_pragma_p(s);
      if (!static_control_yes(scAlone) && !pragma_added_p && !insideScop && !currentScop) {
	add_to_stmt(s, &pragma_added_p);
	last_added_pragma = s;
	insert_endscop_in_sequence(s, &pragma_added_p, false);
      }
      else if (!static_control_yes(scAlone) && insideScop && !currentScop) {
	pragma_nonscop(loop_body(loopAlone), pragma_added_p, in_loop_p);
      }
    }
    break;

  case is_instruction_forloop :
    {
    }
    break;

  case is_instruction_whileloop :
    {
      whileloop whileAlone = instruction_whileloop(instTop);
      bool currentScop = statement_with_pragma_p(s);
      if (!currentScop)      
	pragma_nonscop(whileloop_body(whileAlone), pragma_added_p, in_loop_p);
    }
    break;
    
  case is_instruction_test :
    {
      test testAlone = instruction_test(instTop);
      bool currentScop = statement_with_pragma_p(s);
      if (!currentScop) {
	pragma_nonscop(test_true(testAlone), pragma_added_p, in_loop_p);
	pragma_nonscop(test_false(testAlone), pragma_added_p, in_loop_p);
      }
    }
    break;

  default :
    break;
  }
  return;
}

/* Phase in charge of putting pragmas around the non-SCoP of the code
   This phase only works if there has been a previous phase applied putting
   pragmas around SCoP of the code.
*/
bool rstream_interface (char *module_name) {
  statement module_stat;
  set_current_module_entity(local_name_to_top_level_entity(module_name));
  set_current_module_statement((statement)db_get_memory_resource(DBR_CODE, module_name, true));

  module_stat = get_current_module_statement();
  
  Gsc_map = (statement_mapping)db_get_memory_resource(DBR_STATIC_CONTROL, module_name, true);

  pragma_nonscop(module_stat, false, false);

  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name,module_stat);
  
  reset_current_module_entity();
  reset_current_module_statement();
  
  return true;
}
