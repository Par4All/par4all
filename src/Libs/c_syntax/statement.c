
/*******************  STATEMENTS ******************
 $Id
 $Log */

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
#include "text-util.h"
#include "instrumentation.h"

/* The following global variables will be replaced by functions*/
string CurrentSourceFile = ""; 
entity CurrentSourceFileEntity = entity_undefined;
entity CurrentSourceFileStaticArea;
int CurrentSourceFileStaticAreaOffset = 0;

int CurrentStaticAreaOffset = 0;
int CurrentDynamicAreaOffset = 0;

entity TopLevelEntity = entity_undefined;
entity TopLevelStaticArea = entity_undefined; 
int TopLevelStaticAreaOffset = 0;

/* BlockStack is used to handle block scope */
stack BlockStack;

int loop_counter; 
bool is_loop;
list CurrentSwitchGotoStatements = NIL;
expression CurrentSwitchController = expression_undefined; 

list LabelledStatements = NIL; /* list of labelled statements of the current module*/

int derived_counter = 0; /* to generate unique counter for unnamed struct/union/enum and their members*/

extern statement ModuleStatement;

extern entity DynamicArea;
extern entity StaticArea;

void init_c_areas()
{
    DynamicArea = FindOrCreateEntity(get_current_module_name(), DYNAMIC_AREA_LOCAL_NAME);
    entity_type(DynamicArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(DynamicArea) = MakeStorageRom();
    entity_initial(DynamicArea) = MakeValueUnknown();

    StaticArea = FindOrCreateEntity(get_current_module_name(), STATIC_AREA_LOCAL_NAME);
    entity_type(StaticArea) = make_type(is_type_area, make_area(0, NIL));
    entity_storage(StaticArea) = MakeStorageRom();
    entity_initial(StaticArea) = MakeValueUnknown();
}

void CreateCurrentModule(entity e)
{
  entity_initial(e) = make_value(is_value_code, make_code(NIL, NULL, make_sequence(NIL)));
  set_current_module_entity(e);
  init_c_areas(); 
  LabelledStatements = NIL;
  derived_counter = 0;
}

void ResetCurrentModule()
{
  reset_current_module_entity();
}

void InitializeBlock(statement s)
{
  ModuleStatement = s;
  BlockStack = stack_make(statement_domain,0,0);
  loop_counter = 0;
}

statement MakeBlock(list decls, list stms)
{
  statement s = make_statement(entity_empty_label(), 
			       STATEMENT_NUMBER_UNDEFINED, 
			       STATEMENT_ORDERING_UNDEFINED, 
			       string_undefined,
			       make_instruction_sequence(make_sequence(stms)),
			       decls,NULL);
  return s;
}

statement MakeNullStatement()
{
  return make_statement(entity_empty_label(), 
			STATEMENT_NUMBER_UNDEFINED, 
			STATEMENT_ORDERING_UNDEFINED, 
			string_undefined,
			MakeNullInstruction(),
			NIL,NULL);
}

instruction MakeNullInstruction()
{
  return make_instruction(is_instruction_call, 
			  make_call(CreateIntrinsic(";"),NIL));
}

statement FindStatementFromLabel(entity l)
{
  MAP(STATEMENT,s,
  {
    if (statement_label(s) == l)
      return s;
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
  return st;
}

statement MakeGotoStatement(string label)
{
  entity l = MakeCLabel(label);
  
  /* Find the corresponding statement from its label, 
     if not found, create a pseudo one, which will be replaced lately when
     we see the statement (label: statement) */
 
  statement s = FindStatementFromLabel(l);
  if (s == statement_undefined) 
    {
      s = make_statement(l,STATEMENT_NUMBER_UNDEFINED,
			   STATEMENT_ORDERING_UNDEFINED,
			   empty_comments, 
			   MakeNullInstruction(),NIL,NULL);
      LabelledStatements = gen_nconc(LabelledStatements,CONS(STATEMENT,s,NIL));
    }
  return instruction_to_statement(make_instruction(is_instruction_goto,s));
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


statement MakeWhileLoop(list lexp, statement s, bool before)
{  
  whileloop w; 
  statement smt;
  string lab1 = strdup(concatenate("loop_end_",int_to_string(loop_counter),NULL));
  statement s1 = FindStatementFromLabel(MakeCLabel(lab1));
  string lab2 = strdup(concatenate("loop_exit_",int_to_string(loop_counter),NULL));
  statement s2 = FindStatementFromLabel(MakeCLabel(lab2));
  if (!statement_undefined_p(s1))
    {
      /* This loop has a continue statement which has been transformed to goto 
	 Add the labelled statement at the end of loop body*/
      insert_statement(s,s1,FALSE);
    }
  w  = make_whileloop(MakeCommaExpression(lexp),
		      s,entity_empty_label(),
		      before ? make_evaluation_before(): make_evaluation_after());
  smt = instruction_to_statement(make_instruction_whileloop(w)); 
  if (!statement_undefined_p(s2))
    {
      /* This loop has a break statement which has been transformed to goto 
	 Add the labelled statement after the loop */
      insert_statement(smt,s2,FALSE);
    }
  pips_assert("While loop is consistent",statement_consistent_p(smt));
  ifdebug(2) 
    {
      printf("While loop statement: \n");
      print_statement(smt);
    }
  return smt;
}

statement MakeForloop(expression e1, expression e2, expression e3, statement s)
{								 
  forloop f;
  statement smt;
  string lab1 = strdup(concatenate("loop_end_",int_to_string(loop_counter),NULL));
  statement s1 = FindStatementFromLabel(MakeCLabel(lab1));
  string lab2 = strdup(concatenate("loop_exit_",int_to_string(loop_counter),NULL));
  statement s2 = FindStatementFromLabel(MakeCLabel(lab2));
  if (!statement_undefined_p(s1))
    {
      /* This loop has a continue statement which has been transformed to goto 
	 Add the labelled statement at the end of loop body*/
      insert_statement(s,s1,FALSE);
    }
  f = make_forloop(e1,e2,e3,s);
  smt = instruction_to_statement(make_instruction_forloop(f));
  if (!statement_undefined_p(s2))
    {
      /* This loop has a break statement which has been transformed to goto 
	 Add the labelled statement after the loop */
      insert_statement(smt,s2,FALSE);
    }
  pips_assert("For loop is consistent",statement_consistent_p(smt));
  ifdebug(2) 
    {
      printf("For loop statement: \n");
      print_statement(smt);
    }
  return smt;
}

statement MakeSwitchStatement(statement s)
{
  /* Transform a switch statement to if - else - goto. Example:

     switch (c) {
     case 1:
     s1;
     case 2: 
     s2;  
     default: 
     sd;
     }

     if (c==1) goto switch_xxx_case_1;
     if (c==2) goto switch_xxx_case_2;
     goto switch_xxx_default;
     switch_xxx_case_1: ;
     s1;
     switch_xxx_case_2: ;
     s2;
     switch_xxx_default: ;
     sd;

     In si, we can have goto switch_exit_xxx; (which was a break) 

     and switch_exit_xxx: ; is inserted at the end of the switch statement

     The statement s corresponds to the body 

     switch_xxx_case_1: ;
     s1;
     switch_xxx_case_2: ;
     s2;
     switch_xxx_default: ;
     sd;

     we have to insert 

     if (c==1) goto switch_xxx_case_1;
     if (c==2) goto switch_xxx_case_2;
     goto switch_xxx_default;
     

     before s and return the inserted statement.  */

  string lab = strdup(concatenate("switch_exit_",int_to_string(loop_counter),NULL));
  statement smt = FindStatementFromLabel(MakeCLabel(lab));
  statement seq = instruction_to_statement(make_instruction_sequence(make_sequence(CurrentSwitchGotoStatements)));
  insert_statement(s,seq,TRUE);

  if (!statement_undefined_p(smt))
    {
      /* This switch has a break statement which has been tranformed to goto 
	 Add the labelled statement after the switch */
      insert_statement(s,smt,FALSE);
    }  
  pips_assert("Switch is consistent",statement_consistent_p(s));
  ifdebug(2) 
    {
      printf("Switch statement: \n");
      print_statement(s);
    }
  return s;
}

statement MakeCaseStatement(expression e)
{
  /* Tranform 
         case e: 
     to
         switch_xxx_e: ;
     and generate 	 
        if (c == e) goto switch_xxx_e 
     where c is retrieved from CurrentSwitchController
           xxx is unique from loop_counter */
  string lab = strdup(concatenate("switch_",int_to_string(loop_counter),
				  "_case_",words_to_string(words_expression(e)),NULL));
  statement s = MakeLabelledStatement(lab,MakeNullStatement());
  expression cond = eq_expression(CurrentSwitchController,e);
  test t = make_test(cond,MakeGotoStatement(lab),make_continue_statement(entity_undefined));
  CurrentSwitchGotoStatements = gen_nconc(CurrentSwitchGotoStatements,CONS(STATEMENT,test_to_statement(t),NULL));
  return s;
}

statement MakeDefaultStatement()
{
  /* Return the labelled statement 
       switch_xxx_default: ; 
     and add 
       goto switch_xxx_default;
     to the switch header */
  string lab = strdup(concatenate("switch_",int_to_string(loop_counter),"_default",NULL));
  statement s = MakeLabelledStatement(lab,MakeNullStatement());
  CurrentSwitchGotoStatements = gen_nconc(CurrentSwitchGotoStatements,CONS(STATEMENT,MakeGotoStatement(lab),NULL));
  return s;
}

statement MakeBreakStatement()
{
  /* Loop or switch statement ?*/
  string lab;
  if (is_loop)
    lab = strdup(concatenate("loop_exit_",int_to_string(loop_counter),NULL));
  else
    lab = strdup(concatenate("switch_exit_",int_to_string(loop_counter),NULL));
  return MakeGotoStatement(lab);
}

statement MakeContinueStatement()
{
  /* Unique label with the loop_counter */
  string lab = strdup(concatenate("loop_end_",int_to_string(loop_counter),NULL));
  return MakeGotoStatement(lab);
}








