/* $Id$ 
   $Log: util.c,v $
   Revision 1.1  2003/06/24 09:00:48  nguyen
   Initial revision

   Revision 1.1  2003/06/24 07:27:28  nguyen
   Initial revision
 
*/

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

expression MakeSizeofExpression(expression e)
{
  
  syntax s = make_syntax_sizeofexpression(make_sizeofexpression_expression(e));
  expression exp =  make_expression(s,normalized_undefined); 
  return exp; /* exp = sizeof(e)*/
}


expression MakeSizeofType(type t)
{
  syntax s = make_syntax_sizeofexpression(make_sizeofexpression_type(t));
  expression exp =  make_expression(s,normalized_undefined); 
  return exp;  /* exp = sizeof(t) */
}

expression MakeCastExpression(type t, expression e)
{
  syntax s = make_syntax_cast(make_cast(t,e));
  expression exp = make_expression(s,normalized_undefined); 
  return exp; /* exp = (t) e */
}

expression MakeCommaExpression(list l)
{
  return make_call_expression(CreateIntrinsic(","),l);
}

string list_to_string(list l)
{
  string result = strdup("");
  MAP(STRING,s, 
  {
    string old = result;
    result = strdup(concatenate(old,s,NULL)); 
    free (old);
  }, l);
  return result;
}

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







