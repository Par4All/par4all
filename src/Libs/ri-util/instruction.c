/*

  $Id: instruction.c 14258 2009-06-09 15:21:42Z guelton $

  Copyright 1989-2009 MINES ParisTech

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
/** @file

   Methods dealing with instructions.
*/

#include <stdlib.h>
#include <stdio.h>
#include "linear.h"
#include "genC.h"
#include "misc.h"
#include "ri.h"
#include "ri-util.h"


/** @defgroup instruction_constructors Instruction constructors

    @{
 */

/* Build an instruction that call a function entity with an argument list.

   @param e is the function entity to call
   @param l is the list of argument expressions given to the function to call
 */
instruction make_call_instruction(entity e, list l) {
  return make_instruction(is_instruction_call, make_call(e, l));
}


/* Creates a call instruction to a function with no argument.

   @param f is the function entity to call
 */
instruction
MakeNullaryCallInst(entity f) {
  return make_call_instruction(f, NIL);
}


/* Creates a call instruction to a function with one argument.

   @param f is the function entity to call
   @param e is the argument expression given to the function to call
 */
instruction
MakeUnaryCallInst(entity f,
		  expression e) {
    return make_call_instruction(f, CONS(EXPRESSION, e, NIL));
}


/* Creates a CONTINUE instruction, that is the FORTRAN nop, the ";" in C
   or the "pass" in Python for example.
*/
instruction
make_continue_instruction() {
  entity called_function;
  called_function = entity_intrinsic(CONTINUE_FUNCTION_NAME);
  return MakeNullaryCallInst(called_function);
}

/** @} */


/** @defgroup instructions_p Predicates on instructions

    @{
*/


/* Test if an instruction is the native instruction of the language

   @param i is the instruction to investigate
   @param s is the name of the native instruction to investigate
   @return true if the instruction is a native instruction of the language
 */
bool
native_instruction_p(instruction i,
		     string s) {
  bool call_s_p = FALSE;

  if (instruction_call_p(i)) {
    call c = instruction_call(i);
    entity f = call_function(c);

    if (strcmp(entity_user_name(f), s) == 0)
      call_s_p = TRUE;
  }

  return call_s_p;
}

/* Test if an instruction is an assignment. */
bool
instruction_assign_p(instruction i) {
    return native_instruction_p(i, ASSIGN_OPERATOR_NAME);
}


/* Test if an instruction is a CONTINUE, that is the FORTRAN nop, the ";" in C
   or the "pass" in Python... according to the language.
*/
bool
instruction_continue_p(instruction i) {
  return native_instruction_p(i, CONTINUE_FUNCTION_NAME);
}


/* Test if an instruction is a "return"
   Note that this function is not named "instruction_return_p" since
   it would mean return is a field of instruction ... which used to be the case :)
*/
bool
return_instruction_p(instruction i) {
  return native_instruction_p(i, RETURN_FUNCTION_NAME)
    || native_instruction_p(i, C_RETURN_FUNCTION_NAME);
}


/* Test if an instruction is a Fortran STOP.
*/
bool
instruction_stop_p(instruction i) {
  return native_instruction_p(i, STOP_FUNCTION_NAME);
}


/* Test if an instruction is a Fortran FORMAT.
*/
bool
instruction_format_p(instruction i) {
  return native_instruction_p(i, FORMAT_FUNCTION_NAME);
}

/** @} */


/* Flatten an instruction block if necessary.

   Detects sequences of sequences and reorder as one sequence.

   Some memory leaks. Should not be used. Use the functions from control
   instead.
*/
void flatten_block_if_necessary(instruction i)
{
  if (instruction_block_p(i))
  {
    list ls = NIL;
    MAP(STATEMENT, s, {
      instruction ib = statement_instruction(s);
      if (instruction_block_p(ib))
	ls = gen_nconc(ls, instruction_block(ib));
      else
	ls = gen_nconc(ls, CONS(STATEMENT, s, NIL));
    },
      instruction_block(i));
    gen_free_list(instruction_block(i));
    instruction_block(i) = ls;
  }
}


/* Checks if an instruction block is a list of assignments, possibly
   followed by a continue.
*/
bool
assignment_block_p(i)
instruction i;
{
    MAPL(cs,
     {
	 statement s = STATEMENT(CAR(cs));

	 if(!assignment_statement_p(s))
	     if(!(continue_statement_p(s) && ENDP(CDR(cs)) ))
		 return FALSE;
     },
	 instruction_block(i));
    return TRUE;
}


/* Return a constant string representing symbolically the instruction type.

   Does not work for undefined instructions.

   @return a constant string such as "WHILE LOOP" for a "while()" or "do
   while()" loop and so on.
*/
string
instruction_identification(instruction i)
{
    string instrstring = NULL;

    switch (instruction_tag(i))
    {
    case is_instruction_loop:
	instrstring="DO LOOP";
	break;
    case is_instruction_whileloop:
	instrstring="WHILE LOOP";
	break;
    case is_instruction_test:
	instrstring="TEST";
	break;
    case is_instruction_goto:
	instrstring="GOTO";
	break;
    case is_instruction_call:
      {if (instruction_continue_p(i))
	instrstring="CONTINUE";
    else if (return_instruction_p(i))
	instrstring="RETURN";
    else if (instruction_stop_p(i))
	instrstring="STOP";
    else if (instruction_format_p(i))
	instrstring="FORMAT";
    else if (instruction_assign_p(i))
	instrstring="ASSIGN";
    else {
	instrstring="CALL";
    }
    break;
    }
    case is_instruction_block:
	instrstring="BLOCK";
	break;
    case is_instruction_unstructured:
	instrstring="UNSTRUCTURED";
	break;
    case is_instruction_forloop:
      instrstring="FOR LOOP";
      break;
    case is_instruction_expression:
      instrstring="EXPRESSION";
      break;
    default: pips_error("instruction_identification",
			"ill. instruction tag %d\n",
			instruction_tag(i));
    }

    return instrstring;
}
