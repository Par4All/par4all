/*

  $Id$

  Copyright 1989-2014 MINES ParisTech

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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <stdio.h> 
#include <stdlib.h> 
#include <string.h> 

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "effects.h"
#include "ri-util.h"
#include "effects-util.h"
#include "text-util.h"
#include "database.h"
#include "misc.h"
#include "pipsdbm.h"
#include "resources.h"

#include "effects-generic.h"
#include "effects-simple.h"
#include "control.h"

#include "transformations.h"


typedef struct { list old, new; } entity_lists;

static bool  rw_effect_on_variable_p(list efs, entity var, bool b)
{
  bool readeff = false;
  list le = NIL;
			    
  for( le =efs ; !ENDP(le ) && !readeff ; POP(le) ) {
    effect e= EFFECT(CAR(le));
    reference r = effect_any_reference(e);
    entity v = reference_variable(r); 
    if ((v==var) &&  
	((b && action_read_p(effect_action(e))) || 
	 (!b && action_write_p(effect_action(e)))))
	readeff= true;
  }
  return readeff;
}  

static bool  entity_in_call_arguments_p(entity ent,  call c)
{
  list le=call_arguments(c);
  bool seen=false;
  for( ; !ENDP(le ) && !seen ; POP(le) ) {
    expression e= EXPRESSION(CAR(le));
    if (expression_reference_p(e) 
	&& (reference_variable(expression_reference(e)) == ent))
      seen = true;
  }
  return seen; 
}


static void substitute_entity_in_call_arguments(entity old, entity new, call c)
{
        
      list args = call_arguments(c);
      list tempargs = NIL;
      while (!ENDP(args))
	{
	  expression exp = EXPRESSION(CAR(args));
	  expression temp = substitute_entity_in_expression(old,new,exp);
	  tempargs = gen_nconc(tempargs,CONS(EXPRESSION,temp,NIL));
	  args = CDR(args);
	}
      call_arguments(c) = tempargs;
}

static void freeze_variables_in_statement(statement s, entity_lists * el)
{
  list efs = load_proper_rw_effects_list(s);
  bool read_eff_on_var_already_seen = false;
  bool READ_EFF = true;
  list new_entl = el->new;
  list old_entl = el->old;
  list first_st = NIL;
  list  last_st = NIL;
  bool st_to_insert_before= false;
  bool one_entity_in_args = false;
  if (statement_call_p(s))
  {  
    entity lb = statement_label(s); 

    // scan the lists of entities to freeze
    // some memory leaks should be fixed... FC.
    for (; !ENDP(old_entl) && !ENDP(new_entl);POP(old_entl), POP(new_entl)) 
    { 
      entity ent = ENTITY(CAR(old_entl)); 
      entity new_ent  = ENTITY(CAR(new_entl));
      bool this_entity_in_args = false;
      string strg = 
	strdup(concatenate("'Frozen variable ", 
			   entity_local_name(ent), 
			   " in ", 
			   get_current_module_name(),
			   " changed'",(char *) NULL));
      one_entity_in_args = one_entity_in_args ||
	(this_entity_in_args =
	 entity_in_call_arguments_p(ent, 
		     instruction_call(statement_instruction(s))));
      read_eff_on_var_already_seen=false;
      if (this_entity_in_args) {
	MAP(EFFECT, eff, {
	  reference r = effect_any_reference(eff);
	  entity v = reference_variable(r);  
	  if (v == ent) {
	    if (action_write_p(effect_action(eff))) {
	      call c1=call_undefined;
	      test t1=test_undefined; 
	      expression e1= expression_undefined;
	      expression e2= expression_undefined;
	      expression e3= expression_undefined;
	      instruction i1 = instruction_undefined;
	      statement s1=statement_undefined;
	      if (read_eff_on_var_already_seen 
		  || rw_effect_on_variable_p(efs,ent,READ_EFF)) {
		/* if call  and READ effect on ent, initialize new_ent=ent*/   
		e1 = reference_to_expression(make_reference(new_ent, NIL));
		e2 = reference_to_expression(copy_reference(r));
		i1 = make_assign_instruction(e1,e2);
		s1=instruction_to_statement(i1);
		statement_label(s)= entity_empty_label();
		first_st=CONS(STATEMENT,s1, first_st);
		st_to_insert_before = true;
	      }
	      /* subtitute ent with new_ent in assignment or call */
	      c1 = instruction_call(statement_instruction(s));
	      substitute_entity_in_call_arguments(ent, new_ent,c1);
	      
	      /* insert after: if (new_ent.NE.ent) STOP */
	      e1 = reference_to_expression(make_reference(new_ent, NIL));
	      e2 = reference_to_expression(copy_reference(r));
	      c1 = make_call(entity_intrinsic(NON_EQUAL_OPERATOR_NAME),
                  make_expression_list(e1,e2));
	      
	      e3 = make_expression(make_syntax(is_syntax_call, c1), 
				   normalized_undefined);
	      t1 =  make_test(e3, 
			      make_stop_statement(strg),
			      make_block_statement(NIL));
	      s1 =test_to_statement(t1);
	      last_st=CONS(STATEMENT,s1, last_st);	    
	    }
	    else  read_eff_on_var_already_seen = true;
	  } 
	}, efs);
      }
    }
    if (one_entity_in_args){
      if (st_to_insert_before) {
	MAPL(st, {
	  insert_statement(s, STATEMENT(CAR(st)),true);
	  if (ENDP(CDR(st))) statement_label(STATEMENT(CAR(st)))=lb;
	}, first_st);
      }
      MAPL(st, {
	insert_statement(s, STATEMENT(CAR(st)),false);
      }, last_st);
    }
    
  }
}

static bool initialized_variable_p(entity e)
{
  return variable_static_p(e);
}

bool freeze_variables(char *mod_name) 
{
  entity module;
  list frozen_entities = NIL;
  list cumu_eff=NIL;
  string names="";
  string rep="";
  bool WRITE_EFF = false;
  entity_lists new_le = {NIL,NIL};
  statement mod_stmt = (statement) db_get_memory_resource(DBR_CODE, mod_name, true);
  /* INITIALISATION */
  if (!statement_undefined_p(mod_stmt)) {
    set_current_module_statement(mod_stmt);
    module =module_name_to_entity(mod_name);
    set_current_module_entity(module);
    set_proper_rw_effects((statement_effects)
			  db_get_memory_resource(DBR_PROPER_EFFECTS,
						 mod_name,
						 true));
    set_cumulated_rw_effects((statement_effects)
			     db_get_memory_resource(DBR_CUMULATED_EFFECTS,
					       mod_name,
						    true));
    
    cumu_eff = load_cumulated_rw_effects_list(mod_stmt);
    /* USER REQUEST */
    rep = user_request("Which variables do you want to freeze?\n",mod_name);
    if (rep[0] == '\0') {
      user_log("No variable to freeze\n");
      return false;
    }
    else names=rep;
    frozen_entities = string_to_entity_list(mod_name,names);
    
    /*  build the list of scalar entities to freeze in the module*/
    MAP(ENTITY,ent, {  
            if (rw_effect_on_variable_p(cumu_eff,ent,WRITE_EFF)) {
	      basic b=basic_undefined;
	      pips_assert("entity to freeze is a scalar variable",
			  (type_variable_p(entity_type(ent)) || 
			   variable_dimensions(type_variable(entity_type(ent))) == NIL)); 
	      if (!variable_in_common_p(ent))
		user_log("Variable %s to freeze is not in a COMMON \n",
			 entity_local_name(ent)); 
	          if (!initialized_variable_p(ent))
		user_log("Variable %s to freeze is not  present in a DATA \n",
			 entity_local_name(ent));
	      b = variable_basic(type_variable(entity_type(ent)));
	      new_le.old= CONS(ENTITY, ent,  new_le.old);    
	      new_le.new = CONS(ENTITY, 
				make_new_scalar_variable(module,b), 
				new_le.new);
	    }
    },
	frozen_entities);
    
    /* recurse on  module statements */
    if (!ENDP(new_le.old))
      gen_context_recurse(mod_stmt, 
			  &new_le, 
			  statement_domain,
			  gen_true,
			  freeze_variables_in_statement);
    
    gen_free_list(new_le.new);
    pips_assert("Statement is consistent after FREEZING VARIABLES", 
		statement_consistent_p(mod_stmt));
    
  /* Reorder the module, because new statements have been added */  
    module_reorder(mod_stmt);
    DB_PUT_MEMORY_RESOURCE(DBR_CODE, mod_name,mod_stmt);
    reset_proper_rw_effects();
    reset_cumulated_rw_effects();
    reset_current_module_statement();
    reset_current_module_entity();
  }
  return (true);
  
}
