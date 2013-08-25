/*

  $Id$

  Copyright 1989-2010 MINES ParisTech

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
instruction MakeNullaryCallInst(entity f) {
  return make_call_instruction(f, NIL);
}


/* Creates a call instruction to a function with one argument.

   @param f is the function entity to call
   @param e is the argument expression given to the function to call
 */
instruction MakeUnaryCallInst(entity f,
			      expression e) {
    return make_call_instruction(f, CONS(EXPRESSION, e, NIL));
}


/* Creates a CONTINUE instruction, that is the FORTRAN nop, the ";" in C
   or the "pass" in Python for example.
*/
instruction make_continue_instruction() {
  entity called_function;
  called_function = entity_intrinsic(CONTINUE_FUNCTION_NAME);
  return MakeNullaryCallInst(called_function);
}


instruction
make_assign_instruction(expression l,
                        expression r)
{
   call c = call_undefined;
   instruction i = instruction_undefined;

/*   SG: not true in C
 *   pips_assert("make_assign_statement",
               syntax_reference_p(expression_syntax(l)));*/
   c = make_call(entity_intrinsic(ASSIGN_OPERATOR_NAME),
                 CONS(EXPRESSION, l, CONS(EXPRESSION, r, NIL)));
   i = make_instruction(is_instruction_call, c);

   return i;
}


/** Build an instruction block from a list of statements
 */
instruction make_instruction_block(list statements) {
  return make_instruction_sequence(make_sequence(statements));
}
/** @} */


/** @defgroup instructions_p Predicates on instructions

    @{
*/


/* Test if a call is a native instruction of the language

   @param c is the call to investigate
   @param s is the name of the native instruction to investigate
   @return true if the instruction is a native instruction of the language
 */
bool native_call_p(call c,
        string op_name) {
  bool call_s_p = false;

  // The called function
  entity f = call_function(c);

  if (strcmp(entity_user_name(f), op_name) == 0)
    call_s_p = true;

  return call_s_p;
}


/* Test if an instruction is a native instruction of the language

   @param i is the instruction to investigate
   @param s is the name of the native instruction to investigate
   @return true if the instruction is a native instruction of the language
 */
bool native_instruction_p(instruction i,
			  string op_name)
{
  bool call_s_p = false;

  // Call can be directly inside the instruction,
  // or wrapped inside an expression
  if (instruction_call_p(i)) {
    call_s_p = native_call_p(instruction_call(i), op_name);
  } else if(instruction_expression_p(i)) {
    syntax s = expression_syntax(instruction_expression(i));
    if(syntax_call_p(s)) {
      call_s_p = native_call_p(syntax_call(s), op_name);
    }
  }

  return call_s_p;
}

/* Test if an instruction is an assignment. */
bool instruction_assign_p(instruction i)
{
    return native_instruction_p(i, ASSIGN_OPERATOR_NAME);
}


/* Test if an instruction is a CONTINUE, that is the FORTRAN nop, the ";" in C
   or the "pass" in Python... according to the language.
*/
bool instruction_continue_p(instruction i)
{
    return native_instruction_p(i, CONTINUE_FUNCTION_NAME);
}


/* Test if an instruction is a C or Fortran "return"

   Note that this function is not named "instruction_return_p" since
   it would mean return is a field of instruction ... which used to be
   the case :)
*/
bool return_instruction_p(instruction i)
{
  return native_instruction_p(i, RETURN_FUNCTION_NAME)
    || native_instruction_p(i, C_RETURN_FUNCTION_NAME);
}

bool fortran_return_instruction_p(instruction i)
{
  return native_instruction_p(i, RETURN_FUNCTION_NAME);
}

bool C_return_instruction_p(instruction i)
{
  return native_instruction_p(i, C_RETURN_FUNCTION_NAME);
}

bool exit_instruction_p(instruction i)
{
  return native_instruction_p(i, EXIT_FUNCTION_NAME);
}
bool abort_instruction_p(instruction i)
{
  return native_instruction_p(i, ABORT_FUNCTION_NAME);
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


/* Checks if an instruction block is a list of assignments, possibly
   followed by a continue.
*/
bool assignment_block_p(instruction i)
{
  /* FOREACH cannot be used in this case */
  MAPL(cs,
       {
	 statement s = STATEMENT(CAR(cs));

	 if(!assignment_statement_p(s))
	   if(!(continue_statement_p(s) && ENDP(CDR(cs)) ))
	     return false;
       },
       instruction_block(i));
  return true;
}


/** @} */


/* Flatten an instruction block if necessary.

   Detects sequences of sequences and reorder as one sequence.

   Some memory leaks. Should not be used. Use the functions from control
   instead.

   This function cannot be used for C code as local declarations are
   discarded without warnings.
*/
void flatten_block_if_necessary(instruction i)
{
  if (instruction_block_p(i))
  {
    list ls = NIL;
    FOREACH(STATEMENT, s, instruction_block(i)) {
      instruction ib = statement_instruction(s);
      if (instruction_block_p(ib))
	ls = gen_nconc(ls, instruction_block(ib));
      else
	ls = gen_nconc(ls, CONS(STATEMENT, s, NIL));
    }
    gen_free_list(instruction_block(i));
    instruction_block(i) = ls;
  }
}


/* Return a constant string representing symbolically the instruction type.

   Does not work for undefined instructions: core dump.

   @return a constant string such as "WHILE LOOP" for a "while()" or "do
   while()" loop and so on.
*/
string instruction_identification(instruction i)
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
    default: pips_internal_error("ill. instruction tag %d",
			instruction_tag(i));
    }

  return instrstring;
}

string safe_instruction_identification(instruction i)
{
  string instrstring = string_undefined;
  if(instruction_undefined_p(i))
    instrstring = "UNDEFINED INSTRUCTION";
  else
    instrstring = instruction_identification(i);
  return instrstring;
}


