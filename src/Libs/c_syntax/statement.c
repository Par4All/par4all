
/*******************  STATEMENTS *******************/

#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "parser_private.h"

#include "c_syntax.h"

#include "resources.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "pipsdbm.h"

/* BlockStack is used to handle block scope.
   LoopSwitchStack is used to find the most enclosing 
   loop or switch of a break or continue statement. */
stack BlockStack = make_stack(statement_domain,0,0);
stack LoopSwitchStack = make_stack(statement_domain,0,0); 

extern list LabelledStatements;

statement MakeNullStatement()
{
  return make_statement(entity_empty_label(), 
			STATEMENT_NUMBER_UNDEFINED, 
			STATEMENT_ORDERING_UNDEFINED, 
			string_undefined,
			make_instruction(is_instruction_call, 
					 make_call(CreateIntrinsic(";"),NIL)),
			NIL,NULL);
}

statement FindStatementFromLabel(entity l)
{
  MAP(STATEMENT,smt,
  {
    if (statement_label(smt) == l)
      return smt;
  },LabelledStatements);
  return statement_undefined;
}

statement MakeLabelledStatement(string label, statement s)
{
  entity l = MakeCLabel(label);
  statement smt = FindStatementFromLabel(l);
  statement st;
  if (smt == statement_undefined) 
    {
      st = make_statement(l, STATEMENT_NUMBER_UNDEFINED, 
			 STATEMENT_ORDERING_UNDEFINED, 
			 string_undefined, 
			 statement_instruction(s),
			 NIL,string_undefined);
      LabelledStatements = gen_nconc(LabelledStatements,CONS(STATEMENT,st,NIL));
    }
  else 
    /* The statement is already created pseudoly, replace it by the real one*/
    {
      statement_instruction(smt) =  statement_instruction(s);
      st = smt;
    }
  
  printf(" Labelled statement: \n");
  print_statement(st);
  return st;
}

statement MakeGotoStatement(string label)
{
  entity l = MakeCLabel(label);
  
  /* Find the corresponding statement from its label, 
     if not found, create a pseudo one, which will be replaced lately when
     we see the statement (label: statement) */
 
  statement smt = FindStatementFromLabel(l);
  if (smt == statement_undefined) 
    {
      smt = make_statement(l,STATEMENT_NUMBER_UNDEFINED,
			   STATEMENT_ORDERING_UNDEFINED,
			   empty_comments, 
			   instruction_undefined,NIL,NULL);
      LabelledStatements = gen_nconc(LabelledStatements,CONS(STATEMENT,smt,NIL));
    }
  return instruction_to_statement(make_instruction(is_instruction_goto,smt));
}

/* The labels in C have function scope. */

entity MakeCLabel(string s)
{
  entity l = FindOrCreateEntity(get_current_module_name(),strdup(concatenate(LABEL_PREFIX,s,NULL)));
  if (entity_type(l) == type_undefined) 
    {
      pips_debug(5,"Label %s\n", s);
      entity_type(l) = MakeTypeStatement();
      entity_storage(l) = MakeStorageRom();
      entity_initial(l) = make_value(is_value_constant,
				     MakeConstantLitteral());
    }
  else 
    pips_debug(5, "Label %s already exists\n", s);
  return(l);
}
