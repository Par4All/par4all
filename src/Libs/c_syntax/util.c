/* $Id$ 
   $Log: util.c,v $
   Revision 1.11  2004/02/20 13:57:47  nguyen
   Treat EXTERN entities

   Revision 1.10  2004/02/19 15:18:24  nguyen
   Bug related to "const void", qualifier for not variable type

   Revision 1.9  2004/02/19 14:07:59  nguyen
   Correct things about qualifiers

   Revision 1.8  2004/02/18 10:33:07  nguyen
   Rewrite declarators (parenthese, array and function)

   Revision 1.7  2003/12/18 22:41:55  nguyen
   Change FILE_SEP_STRING from % to !

   Revision 1.6  2003/12/05 17:19:04  nguyen
   Add more syntax types : subscript and application.
   Handle entity creation : array, function, ..

   Revision 1.5  2003/09/05 14:18:58  nguyen
   Improved version of CreateCurrentEntity

   Revision 1.4  2003/08/13 08:00:00  nguyen
   Modify MakeCurrentEntity by taking into account the function case

   Revision 1.3  2003/08/06 14:00:08  nguyen
   Replace global variables such as CurrentCompilationUnit, static and dynamic
   ares by function calls

   Revision 1.2  2003/08/04 14:19:41  nguyen
   Preliminary version of the C parser

   Revision 1.1  2003/06/24 09:00:48  nguyen
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
#include "c_parser_private.h"

#include "c_syntax.h"
 
#include "cyacc.h"

#include "resources.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "pipsdbm.h"
#include "transformations.h"

/* To avoid warnings */
extern char *strdup(const char *s1);

extern string compilation_unit_name;
extern hash_table keyword_typedef_table;

extern list CalledModules;
extern entity StaticArea;
extern entity DynamicArea;

/******************* TOP LEVEL ENTITY  **********************/

entity get_top_level_entity()
{
  return FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,TOP_LEVEL_MODULE_NAME);
}

void MakeTopLevelEntity()
{
  /* To be economic, group this top level entity to it areas*/
  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,TOP_LEVEL_MODULE_NAME);
  entity_storage(e) = make_storage_rom();
  entity_type(e) = make_type(is_type_area, make_area(0, NIL));
  entity_initial(e) = make_value_unknown();
}


/******************* CURRENT MODULE AREAS **********************/

void init_c_areas()
{
  DynamicArea = FindOrCreateEntity(get_current_module_name(), DYNAMIC_AREA_LOCAL_NAME);
  entity_type(DynamicArea) = make_type(is_type_area, make_area(0, NIL));
  entity_storage(DynamicArea) = make_storage_rom();
  entity_initial(DynamicArea) = make_value_unknown();
  
  StaticArea = FindOrCreateEntity(get_current_module_name(), STATIC_AREA_LOCAL_NAME);
  entity_type(StaticArea) = make_type(is_type_area, make_area(0, NIL));
  entity_storage(StaticArea) = make_storage_rom();
  entity_initial(StaticArea) = make_value_unknown();
}


/******************* COMPILATION UNIT **********************/

entity get_current_compilation_unit_entity()
{
  return FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,compilation_unit_name);
}


/* A compilation unit is also considered as a module*/

void MakeCurrentCompilationUnitEntity(string name)
{
  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,name);
  /* Normally, the storage must be rom but in order to store the list of entities
     declared with extern, we use the ram storage to put this list in ram_shared*/
  entity_storage(e) = make_storage_ram(make_ram(entity_undefined,entity_undefined,0,NIL));
  entity_type(e) = make_type_functional(make_functional(NIL,make_type_unknown()));
  entity_initial(e) = make_value(is_value_code, make_code(NIL,strdup(""), make_sequence(NIL)));
  pips_debug(4,"Set current module entity for compilation unit %s\n",name);
  set_current_module_entity(e);
  init_c_areas(); 
}

void ResetCurrentCompilationUnitEntity()
{
  pips_debug(4,"Reset current module entity for compilation unit %s\n",get_current_module_name());
  reset_current_module_entity();
}


/******************* EXPRESSIONS **********************/

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
  if (ENDP(l))
    return expression_undefined;
  if (gen_length(l)==1)
    return EXPRESSION(CAR(l));
  return make_call_expression(CreateIntrinsic(","),l);
}

expression MakeBraceExpression(list l)
{
  return make_call_expression(CreateIntrinsic(BRACE_INTRINSIC),l);
}

expression MakeFunctionExpression(expression e, list le)
{
  /* There are 2 cases :
     1. The first argument corresponds to a function name (an entity). 
        In this case, we create a normal call expression and the corresponding
	entity is added to the list of callees.
     2. The first argument can be any expression denoting a called function (a pointer 
        to a function,... such as (*ctx->Driver.RenderString)() in the benchmark mesa in SPEC2000). 
	In this case, we create a function application expression.  */
  expression exp = expression_undefined;
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_reference:
      {
	entity ent = reference_variable(syntax_reference(s));
	AddToCalledModules(ent);
	pips_debug(6,"Normal function call\n");
	exp = make_call_expression(ent,le);
	break;
      }
    case is_syntax_call:
      {
	application a = make_application(e,le);
	pips_debug(6,"Application function call\n");
	exp = make_expression(make_syntax_application(a),normalized_undefined);
	break;
      }
    case is_syntax_range:
    case is_syntax_cast: 
    case is_syntax_sizeofexpression:
    case is_syntax_subscript:
    case is_syntax_application:
      CParserError("This is not a functional expression\n");
      break;
    default: 
      {
	pips_internal_error("unexpected syntax tag: %d\n", syntax_tag(s));
      }
    }
  return exp;
}

expression MemberDerivedIdentifierToExpression(type t,string m)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      pips_debug(6,"Basic tag is %d",basic_tag(b));
      switch (basic_tag(b)) {
      case is_basic_pointer:
	{
	  type tp = basic_pointer(b);
	  return MemberDerivedIdentifierToExpression(tp,m);
	}
      case is_basic_typedef:
	{
	  entity te = basic_typedef(b);
	  type tp = entity_type(te);
	  return MemberDerivedIdentifierToExpression(tp,m);
	}
      case is_basic_derived:
	{
	  entity de = basic_derived(b);
	  string name = entity_user_name(de);
	  return IdentifierToExpression(strdup(concatenate(name,MEMBER_SEP_STRING,m,NULL))); 
	}
      default:
	break;
      }
    }
  CParserError("Cannot find the field identifier from current type\n");
  return expression_undefined;
}

expression MemberIdentifierToExpression(expression e, string m)
{
  /* Find the name of struct/union of m from the type of expression e*/
  
  syntax s = expression_syntax(e);
  ifdebug(6)
    {
      pips_debug(6,"Find the struct/union of %s from expression:\n",m);
      print_expression(e);
    }
  switch (syntax_tag(s))
    {
    case is_syntax_call:
      {
	call c = syntax_call(s);
	entity f = call_function(c);
	pips_debug(6,"Call expression %s\n",entity_name(f));
	if ((strcmp(entity_name(f),"TOP-LEVEL:->")==0) ||
	    (strcmp(entity_name(f),"TOP-LEVEL:.")==0))
	  {
	    expression exp = EXPRESSION(CAR(CDR(call_arguments(c))));
	    return MemberIdentifierToExpression(exp,m);
	  }
	if (strcmp(entity_name(f),"TOP-LEVEL:*indirection")==0)
	  {
	    expression exp = EXPRESSION(CAR(call_arguments(c)));
	    return MemberIdentifierToExpression(exp,m);
	  }
	/* More types of call must be added */
	if (type_functional_p(entity_type(f)))
	  {
	    /* User defined call */
	    type t = functional_result(type_functional(entity_type(f)));
	    return MemberDerivedIdentifierToExpression(t,m);
	  }
	break;
      }
    case is_syntax_reference:
      {
	entity ent = reference_variable(syntax_reference(s));
	type t = entity_type(ent);
	pips_debug(6,"Reference expression\n");
	return MemberDerivedIdentifierToExpression(t,m);
      }
    case is_syntax_cast: 
      {
	type t = cast_type(syntax_cast(s));
	pips_debug(6,"Cast expression\n");
	return MemberDerivedIdentifierToExpression(t,m);
      }
    case is_syntax_range:
      break;
    case is_syntax_sizeofexpression:
      break;
    case is_syntax_subscript:
      {
	expression exp = subscript_array(syntax_subscript(s));
	pips_debug(6,"Subscripting array expression\n");
	return MemberIdentifierToExpression(exp,m);
      }
      
    default: 
      {
	pips_internal_error("unexpected syntax tag: %d\n", syntax_tag(s));
      }
    } 
  CParserError("Cannot find the field identifier from current expression\n");
  return expression_undefined;
}

expression IdentifierToExpression(string s)
{
  entity ent = FindEntityFromLocalName(s);
  pips_debug(5,"Identifier is: %s\n",s);
  if (entity_undefined_p(ent))
    {
      /* This identifier has not been passed by the parser. 
         It is probably a function call => try this case now and complete others later.
         The scope of this function is global */
      pips_debug(5,"Create unparsed global function: %s\n",s);
      ent = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,s);
      entity_storage(ent) = make_storage_return(ent);
      entity_type(ent) = make_type_functional(make_functional(NIL,make_type_unknown()));
      entity_initial(ent) = make_value(is_value_code,make_code(NIL,strdup(""),make_sequence(NIL)));
      return make_expression(make_syntax_reference(make_reference(ent,NIL)),normalized_undefined);
      /*return MakeNullaryCall(ent);*/
    }
  switch (type_tag(entity_type(ent))) {
  case is_type_variable: 
  case is_type_functional:
    return make_expression(make_syntax_reference(make_reference(ent,NIL)),normalized_undefined);
  default:
    {
      CParserError("Which kind of expression?\n");
      return expression_undefined;
    }
  }
}

expression MakeArrayExpression(expression exp, list lexp)
{
  /* There are two cases: 
   1. Simple array reference, where the first argument is a simple array or 
   pointer name. We create a reference expression (syntax = reference). 
   2. Complicated subscripting array, where the first argument can be a function 
   call (foo()[]), a structure or union member (str[5].field[7], ... We create a 
   subscripting expression (syntax = subscript). */

  expression e = expression_undefined;
  syntax s = expression_syntax(exp);
  switch(syntax_tag(s)) {
  case is_syntax_reference:
    {
      reference r = syntax_reference(s);
      entity ent = reference_variable(r);
      list l = reference_indices(r);
      pips_debug(6,"Normal reference expression\n");
      e = reference_to_expression(make_reference(ent,gen_nconc(l,lexp)));
      break;
    }
  case is_syntax_call:
  case is_syntax_range:
  case is_syntax_cast: 
  case is_syntax_sizeofexpression:
  case is_syntax_subscript:
  case is_syntax_application:
    {
      subscript a = make_subscript(exp,lexp);
      syntax s = make_syntax_subscript(a);
      pips_debug(6,"Subscripting array expression\n");
      e = make_expression(s,normalized_undefined);
      break;
    }
  default: 
    {
      pips_internal_error("unexpected syntax tag: %d\n", syntax_tag(s));
    }
  } 
  return e;
}

/*******************  TYPES *******************/

bool signed_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_int_p(b))
	if (basic_int(b)/10 == DEFAULT_SIGNED_TYPE_SIZE)
	  return TRUE;
    }
  return FALSE;
}

bool unsigned_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_int_p(b))
	if (basic_int(b)/10 == DEFAULT_UNSIGNED_TYPE_SIZE)
	  return TRUE;
    }
  return FALSE;
}

bool long_type_p(type t)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_int_p(b))
	if (basic_int(b) == DEFAULT_LONG_INTEGER_TYPE_SIZE)
	  return TRUE;
    }
  return FALSE;
}

bool bit_type_p(type t)
{
  if (!type_undefined_p(t) && type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (!basic_undefined_p(b) && basic_bit_p(b))
	return TRUE;
    }
  return FALSE;
}


type make_standard_integer_type(type t, int size)
{
  if (t == type_undefined)
    {
      variable v = make_variable(make_basic_int(size),NIL,NIL);
      return make_type_variable(v);
    }
  else
    {
      if (signed_type_p(t) || unsigned_type_p(t))
	{
	  basic b = variable_basic(type_variable(t));
	  int i = basic_int(b);
	  variable v = make_variable(make_basic_int(10*(i/10)+size),NIL,NIL);
	  pips_debug(8,"Old basic size: %d, new size : %d\n",i,10*(i/10)+size);
	  return make_type_variable(v);
	}
      else 
	{
	  if (bit_type_p(t))
	    /* If it is int i:5, keep the bit basic type*/
	    return t; 
	  else
	    user_warning("Parse error", "Standard integer types\n");
	  return type_undefined;
	}
    }
}

type make_standard_long_integer_type(type t)
{
  if (t == type_undefined)
    {
      variable v = make_variable(make_basic_int(DEFAULT_LONG_INTEGER_TYPE_SIZE),NIL,NIL);
      return make_type_variable(v); 
    } 
  else
    {
      if (signed_type_p(t) || unsigned_type_p(t) || long_type_p(t))
	{
	  basic b = variable_basic(type_variable(t));
	  int i = basic_int(b);
	  variable v; 
	  if (i%10 == DEFAULT_INTEGER_TYPE_SIZE)
	    {
	      /* long */
	      v = make_variable(make_basic_int(10*(i/10)+DEFAULT_LONG_INTEGER_TYPE_SIZE),NIL,NIL);
	      pips_debug(8,"Old basic size: %d, new size : %d\n",i,10*(i/10)+DEFAULT_LONG_INTEGER_TYPE_SIZE);
	    }
	  else 
	    {
	      /* long long */
	      v = make_variable(make_basic_int(10*(i/10)+DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE),NIL,NIL);
	      pips_debug(8,"Old basic size: %d, new size : %d\n",i,10*(i/10)+DEFAULT_LONG_LONG_INTEGER_TYPE_SIZE);
	    }
	  return make_type_variable(v);
	}
      else 
	{
	  if (bit_type_p(t))
	    /* If it is long int i:5, keep the bit basic type*/
	    return t; 
	  else
	    user_warning("Parse error", "Standard long integer types\n");
	  return type_undefined;
	}
    }
}

/******************* ENTITIES *******************/


bool static_module_p(entity e)
{
  return static_module_name_p(entity_name(e));
}

bool compilation_unit_p(string module_name)
{
  /* A module name is a compilation unit if and only if its last character is FILE_SEP */
  if (module_name[strlen(module_name)-1]==FILE_SEP)
    return TRUE;
  return FALSE;
}

entity FindEntityFromLocalName(string name)
{
  /* Find an entity from its local name.
     We have to look for all possible prefixes, which are:
     blank, STRUCT_PREFIX, UNION_PREFIX, ENUM_PREFIX, TYPEDEF_PREFIX
     
     How about multiple results ? The order of prefixes ?  */

  entity ent;
  string prefixes[] = {"",STRUCT_PREFIX, UNION_PREFIX, ENUM_PREFIX, TYPEDEF_PREFIX,NULL};
  int i;
  for (i=0; prefixes[i]!=NULL; i++)
    {
      if ((ent = FindEntityFromLocalNameAndPrefix(name,prefixes[i])) != entity_undefined) 
	return ent;
    }
  pips_user_warning("Cannot find entity %s\n",name);
  return entity_undefined;
}

entity FindOrCreateEntityFromLocalNameAndPrefix(string name,string prefix, bool is_external)
{
  entity e; 
  if ((e = FindEntityFromLocalNameAndPrefix(name,prefix)) != entity_undefined) 
    return e;
  return CreateEntityFromLocalNameAndPrefix(name,prefix,is_external);
}

entity FindEntityFromLocalNameAndPrefix(string name,string prefix)
{
  /* Find an entity from its local name and prefix.
     We have to look from the most enclosing scope.
 
     Possible name combinations and the looking order:
       
	1. FILE!MODULE:BLOCK`PREFIXname or MODULE:BLOCK`PREFIXname
        2. FILE!MODULE:PREFIXname or MODULE:PREFIXname
        3. FILE!PREFIXname
	4. TOP-LEVEL:PREFIXname
			      
     with 5 possible prefixes: blank, STRUCT_PREFIX, UNION_PREFIX, ENUM_PREFIX, TYPEDEF_PREFIX */

  entity ent;
  string global_name; 

  pips_debug(5,"Entity local name is %s with prefix %s\n",name,prefix);

  /* Add block scope case here */

  if (!entity_undefined_p(get_current_module_entity()))
    {
      if (static_module_p(get_current_module_entity()))
	global_name = strdup(concatenate(compilation_unit_name,
					 get_current_module_name(),MODULE_SEP_STRING,
					 prefix,name,NULL));
      else
	global_name = strdup(concatenate(get_current_module_name(),MODULE_SEP_STRING,
					 prefix,name,NULL));
      if ((ent = gen_find_tabulated(global_name,entity_domain)) != entity_undefined) 
	{
	  pips_debug(5,"Entity global name is %s\n",global_name);
	  return ent;
	}
    }
  global_name = strdup(concatenate(compilation_unit_name,
				   prefix,name,NULL));
  if ((ent = gen_find_tabulated(global_name,entity_domain)) != entity_undefined) 
    {
      pips_debug(5,"Entity global name is %s\n",global_name);
      return ent;
    }
   
  global_name = strdup(concatenate(TOP_LEVEL_MODULE_NAME,MODULE_SEP_STRING,
				   prefix,name,NULL));
  if ((ent = gen_find_tabulated(global_name,entity_domain)) != entity_undefined) 
    {
      pips_debug(5,"Entity global name is %s\n",global_name);
      return ent;
    }
  pips_user_warning("Cannot find entity %s\n",name);
  return entity_undefined;
}

entity CreateEntityFromLocalNameAndPrefix(string name, string prefix, bool is_external)
{
  /* We have to know the context : 
     - if the entity is declared outside any function, their scope is the CurrentCompilationUnit
     - if the entity is declared inside a function, we have to know the CurrentBlock, 
     which is omitted for the moment 
        - if the function is static, their scope is CurrentCompilationUnit#CurrentModule
        - if the function is global, their scope is CurrentModule */
  entity ent;
  pips_debug(5,"Entity local name is %s with prefix %s\n",name,prefix);
  
  if (is_external)
    ent = find_or_create_entity(strdup(concatenate(compilation_unit_name,
					    prefix,name,NULL)));	
  else
    {
      /* Add block scope here */
      if (static_module_p(get_current_module_entity()))
	ent = find_or_create_entity(strdup(concatenate(compilation_unit_name,
						get_current_module_name(),MODULE_SEP_STRING,
						prefix,name,NULL)));	
      else 
	ent = find_or_create_entity(strdup(concatenate(get_current_module_name(),MODULE_SEP_STRING,
						prefix,name,NULL)));	
    }
  pips_debug(5,"Entity global name is %s\n",entity_name(ent));
  return ent;
}

/* This function finds or creates the current entity. Only entity full name is created, 
   other fields such as type, storage and initial value are undefined.  */

entity FindOrCreateCurrentEntity(string name,stack ContextStack,stack FormalStack,
				 stack FunctionStack, bool is_external)
{
  entity ent; 
  c_parser_context context = stack_head(ContextStack);
  string scope = c_parser_context_scope(context);
  bool is_typedef = c_parser_context_typedef(context);
  bool is_static = c_parser_context_static(context);
  entity function = entity_undefined;
  bool is_formal;
  if (stack_undefined_p(FormalStack) || stack_empty_p(FormalStack))
    is_formal = FALSE;
  else
    {
      is_formal= TRUE; 
      function = stack_head(FunctionStack);
    }
  ifdebug(5)
    {
      pips_debug(5,"Entity local name %s\n",name);
      if (scope != NULL)
	pips_debug(5,"Current scope: %s\n",scope);
      pips_debug(5,"is_typedef: %d\n",is_typedef);
      pips_debug(5,"is_static: %d\n",is_static);
      pips_debug(5,"is_external: %d\n",is_external);
      pips_debug(5,"is_formal: %d\n",is_formal);
      if (is_formal)
	pips_debug(5,"of current function %s\n",entity_user_name(function));
      /* function is only used for formal variables*/
    }
  
  if (is_typedef)
    {
      /* Tell the lexer about the new type names : add to keyword_typedef_table */
      hash_put(keyword_typedef_table,strdup(name),(void *) TK_NAMED_TYPE);
      pips_debug(5,"Add typedef name %s to hash table\n",name);
      ent = CreateEntityFromLocalNameAndPrefix(name,TYPEDEF_PREFIX,is_external);
    }
  else
    {
      if (scope != NULL)
	{
	  ent = find_or_create_entity(strdup(concatenate(scope,name,NULL)));
	  if (is_external && strstr(scope,TOP_LEVEL_MODULE_NAME) != NULL)
	    {
	      /* This entity is declared in a compilation unit with keyword EXTERN.
		 Add it to the storage of the compilation unit to help code generation*/
	      entity com_unit = get_current_compilation_unit_entity();
	      ram_shared(storage_ram(entity_storage(com_unit))) = 
		gen_nconc(ram_shared(storage_ram(entity_storage(com_unit))), CONS(ENTITY,ent,NIL));
	    }
	}
      else 
	{
	  if (is_formal)
	    /* Formal parameter */
	    ent = find_or_create_entity(strdup(concatenate(top_level_entity_p(function)?
							   entity_user_name(function):entity_name(function),
							   MODULE_SEP_STRING,name,NULL)));
	  else 
	    {
	      /* scope = NULL, not extern/typedef/struct/union/enum  */
	      if (is_external)
		{
		  /* This is a variable/function declared outside any module's body*/
		  if (is_static) 
		    ent = find_or_create_entity(strdup(concatenate(compilation_unit_name,
								   name,NULL)));
		  else 
		    ent = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,name);
		}
	      else 
		{
		  /* This is a variable/function declared inside a module's body: add block scope here 
		     Attention, the scope of a function declared in module is the module, not global.*/
		  if (static_module_p(get_current_module_entity()))
		    ent = find_or_create_entity(strdup(concatenate(compilation_unit_name,
								   get_current_module_name(),MODULE_SEP_STRING,
								   name,NULL)));      
		  else
		    ent = find_or_create_entity(strdup(concatenate(get_current_module_name(),MODULE_SEP_STRING,
								   name,NULL)));
		}
	    }
	}
    }
  pips_debug(5,"Entity global name %s\n",entity_name(ent));
  return ent;
}


void UpdateParenEntity(entity e, list lq)
{
  type t = entity_type(e);
  pips_debug(3,"Update entity in parentheses %s\n",entity_name(e));
  if (lq != NIL)
    {
      if (type_undefined_p(t))
	t = make_type_variable(make_variable(basic_undefined,list_undefined,lq));
      else 
	{
	  if (type_variable_p(t))
	    {
	      variable v = type_variable(t);
	      variable_qualifiers(v) = gen_nconc(variable_qualifiers(v),lq);
	    }
	  else 
	    {
	      CParserError("Attributes for not variable type\n");
	    }
	}
    }
}


dimension MakeDimension(list le)
{
  dimension d;
  if (le == NIL)
    {
      d = make_dimension(int_to_expression(0),make_unbounded_expression());
      pips_debug(5,"Unbounded dimension\n");
    }
  else 
    {
      /* Take only the first expression of le, do not know why it can be a list ?*/
      expression e = EXPRESSION(CAR(le));
      int up;
      if (expression_integer_value(e,&up))
	d = make_dimension(int_to_expression(0),int_to_expression(up-1));
      else 
	d = make_dimension(int_to_expression(0),MakeBinaryCall(CreateIntrinsic("-C"),e,
							       int_to_expression(1)));
      ifdebug(5) 
	{
	  pips_debug(5,"Array dimension:\n");
	  print_expression(e);
	}
    }
  return d;
}

type UpdateFinalPointer(type pt, type t)
{
  /* This function replaces the type pointed by the pointer pt
     (this can be a pointer of pointer,... so we have to go until the last one)
     by the type t*/
  pips_debug(3,"Update final pointer type %d and %d\n",type_tag(pt),type_tag(t));
  if (type_variable_p(pt) && basic_pointer_p(variable_basic(type_variable(pt))))
    {
      type ppt = basic_pointer(variable_basic(type_variable(pt)));
      if (type_undefined_p(ppt))
	return make_type_variable(make_variable(make_basic_pointer(t),NIL,variable_qualifiers(type_variable(pt))));
      return UpdateFinalPointer(ppt,t);
    }
  CParserError("pt is not a pointer\n");
  return type_undefined;
}

void UpdatePointerEntity(entity e, type pt, list lq)
{
  type t = entity_type(e);
  pips_debug(3,"Update pointer entity %s\n",entity_name(e));
  
  if (type_undefined_p(t))
    {
      pips_debug(3,"Undefined type entity\n");
      entity_type(e) = pt;
      variable_qualifiers(type_variable(entity_type(e))) = gen_nconc(variable_qualifiers(type_variable(entity_type(e))),lq);
    }
  else
    {
      switch (type_tag(t)) {
      case is_type_variable:
	{
	  /* Make e an array of pointers whose type is this of pt */
	  variable v = type_variable(t);
	  pips_debug(3,"Array of pointers\n");
	  entity_type(e) = make_type_variable(make_variable(variable_basic(type_variable(pt)),
							    variable_dimensions(v),
							    gen_nconc(variable_qualifiers(v),lq)));
	  break;
	}
      case is_type_functional:
	{
	  /* Make e a function returns a pointer */
	  functional f = type_functional(t);
	  pips_debug(3,"Function returns a pointer \n");
	  entity_type(e) = make_type_functional(make_functional(functional_parameters(f),pt));
	  break;
	}
      default:
	{
	  CParserError("Entity is neither an array of pointers nor a pointer to a function?\n");
	}
      }
    }
}

void UpdateArrayEntity(entity e, list lq, list le)
{
  type t = entity_type(e);
  pips_debug(3,"Update array entity %s\n",entity_name(e));

  /* lq is for what ? e or le ????*/
  if (type_undefined_p(t))
    {
      pips_debug(3,"First array dimension\n");
      entity_type(e) = make_type_variable(make_variable(basic_undefined,CONS(DIMENSION,MakeDimension(le),NIL),lq));
    }
  else 
    {
      pips_debug(3,"Next array dimension\n");
      if (type_variable_p(t))
	{
	  variable v = type_variable(t);
	  variable_qualifiers(v) = gen_nconc(variable_qualifiers(v),lq);
	  variable_dimensions(v) = gen_nconc(variable_dimensions(v),CONS(DIMENSION,MakeDimension(le),NIL));
	}
      else 
	{
	  CParserError("Dimension for not variable type\n");
	} 
    }
}

void UpdateFunctionEntity(entity e, list la)
{
  type t = entity_type(e);
  pips_debug(3,"Update function entity %s\n",entity_name(e));
  if (type_undefined_p(t))
    entity_type(e) = make_type_functional(make_functional(la,type_undefined));
  else
    CParserError("This entity must have undefined type\n");
}

/* This function replaces the undefined field in t1 by t2. 
   If t1 is an array type and the basic of t1 is undefined, it is replaced by the basic of t2.
   If t1 is a pointer type, if the pointed type is undefined it is replaced by t2. 
   If t1 is a functional type, if the result type of t1 is undefined, it is replaced by t2.
   The function is recursive */

type UpdateType(type t1, type t2)
{
  if (type_undefined_p(t1))
    return t2;
  switch (type_tag(t1)) 
    {
    case is_type_variable: 
      {
	variable v = type_variable(t1);
	if (basic_undefined_p(variable_basic(v)))
	  {
	    if (type_variable_p(t2))
	      return make_type_variable(make_variable(variable_basic(type_variable(t2)),
						      variable_dimensions(v),
						      gen_nconc(variable_qualifiers(v),
								variable_qualifiers(type_variable(t2)))));
	    CParserError("t1 is a variable type but not t2\n");
	  }
	else 
	  {
	    /* Basic pointer */
	    if (basic_pointer_p(variable_basic(v)))
	      {
		type pt = basic_pointer(variable_basic(v));
		return make_type_variable(make_variable(make_basic_pointer(UpdateType(pt,t2)),
							variable_dimensions(v),
							variable_qualifiers(v)));
	      }
	    CParserError("This basic has which field undefined ?\n");
	  }
	break;
      }
    case is_type_functional:
      {
	functional f = type_functional(t1);
	if (type_undefined_p(functional_result(f)))
	  {
	    if (type_undefined_p(t2))
	      t2 = make_type_unknown();
	    return make_type_functional(make_functional(functional_parameters(f),t2));
	  }
	return make_type_functional(make_functional(functional_parameters(f),UpdateType(functional_result(f),t2)));
      }
    default:
      {
	CParserError("t1 has which kind of type?\n");
      }
    } 
  return type_undefined;
}

void UpdateEntity(entity e, stack ContextStack, stack FormalStack, stack FunctionStack,
		  stack OffsetStack, bool is_external)
{
  /* Update the entity with final type, storage, initial value */
  
  stack s = (stack) entity_storage(e); 
  type t = entity_type(e);
  c_parser_context context = stack_head(ContextStack);
  type tc = c_parser_context_type(context);
  type t1,t2;
  list lq = c_parser_context_qualifiers(context);

  pips_debug(3,"Update entity %s\n",entity_name(e));

  if (lq != NIL)
    {
      /* tc must have variable type, add lq to its qualifiers */
      if (!type_undefined_p(tc) && type_variable_p(tc))
	variable_qualifiers(type_variable(tc)) = lq;
      /*else
	const void, void is not of type variable, store const where ?????????  
	CParserError("Entity has qualifier but no type or is not variable type in the context?\n");*/
    }
  
  /************************* TYPE PART *******************************************/
 
  /* Use the type stack in entity_storage to create the final type for the entity*/
  t2 = UpdateType(t,tc);

  while (stack_size(s) > 1)
    {
      t1 = stack_pop(s);
      t2 = UpdateType(t1,t2);
    }
  entity_type(e) = t2;
   
  /************************* STORAGE PART *******************************************/

  if (!storage_undefined_p(c_parser_context_storage(context)))
    {
      pips_debug(3,"Current storage context is %d\n",
		 storage_tag(c_parser_context_storage(context)));
      entity_storage(e) = c_parser_context_storage(context);
    }
  else
    {
      if (!stack_undefined_p(FormalStack) && (FormalStack != NULL) && !stack_empty_p(FormalStack))
	{
	  entity function = stack_head(FunctionStack);
	  int offset = basic_int((basic) stack_head(OffsetStack));
	  pips_debug(3,"Create formal variable %s for function %s with offset %d\n",
		     entity_name(e),entity_name(function),offset);
	  AddToDeclarations(e,function);
	  entity_storage(e) = make_storage_formal(make_formal(function,offset));
	}
      else
	{
	  entity_storage(e) = MakeStorageRam(e,is_external,c_parser_context_static(context));
	}
    }
  
  pips_assert("Current entity is consistent",entity_consistent_p(e));
}


void UpdateEntities(list le, stack ContextStack, stack FormalStack, stack FunctionStack,
		    stack OffsetStack, bool is_external)
{
  MAP(ENTITY, e,
  {
    UpdateEntity(e,ContextStack,FormalStack,FunctionStack,OffsetStack,is_external);
  },le);
}


/******************* ABSTRACT TYPE DECLARATION ***************************/

void UpdateAbstractEntity(entity e, stack ContextStack)
{
  /* Update the entity with final type, storage */
 
  stack s = (stack) entity_storage(e); 
  type t = entity_type(e);
  c_parser_context context = stack_head(ContextStack);
  type tc = c_parser_context_type(context);
  type t1,t2;
  list lq = c_parser_context_qualifiers(context);

  pips_debug(3,"Update abstract entity %s\n",entity_name(e));
  
  if (lq != NIL)
    {
      /* tc must have variable type, add lq to its qualifiers */
      if (!type_undefined_p(tc) && type_variable_p(tc))
	variable_qualifiers(type_variable(tc)) = lq;
      /*else
	const void, void is not of type variable, store const where ?????????  
	CParserError("Entity has qualifier but no type or is not variable type in the context?\n");*/
    }
  
  /************************* TYPE PART *******************************************/
 
  /* Use the type stack in entity_storage to create the final type for the entity*/
  t2 = UpdateType(t,tc);

  while (stack_size(s) > 1)
    {
      t1 = stack_pop(s);
      t2 = UpdateType(t1,t2);
    }
  entity_type(e) = t2;
   
  /************************* STORAGE PART *******************************************/
  
  entity_storage(e) = storage_undefined;
}


bool entity_in_list_p(entity e, list le)
{
  MAP(ENTITY, f, if (e==f) return TRUE, le);
  return FALSE;
}

void AddToDeclarations(entity e, entity mod)
{
  if (!gen_in_list_p(e,code_declarations(value_code(entity_initial(mod)))))
    {
      pips_debug(5,"Add entity %s to module %s\n",entity_user_name(e),entity_user_name(mod));
      code_declarations(value_code(entity_initial(mod))) = gen_nconc(code_declarations(value_code(entity_initial(mod))),
								     CONS(ENTITY,e,NIL));
    }
}
/************************* STRUCT/UNION ENTITY*********************/

void UpdateDerivedEntity(list ld, entity e, stack ContextStack)
{
  /* Update the derived entity with final type and rom storage. 
     If the entity has bit type, do not need to update its type*/
  type t = entity_type(e);
  pips_debug(3,"Update derived entity %s\n",entity_name(e));
  if (!bit_type_p(t))
    {
      stack s = (stack) entity_storage(e); 
      c_parser_context context = stack_head(ContextStack);
      type tc = c_parser_context_type(context);
      type t1,t2;
      
      /* what about context qualifiers ? */
      t2 = UpdateType(t,tc);
      
      while (stack_size(s) > 1)
	{
	  t1 = stack_pop(s);
	  t2 = UpdateType(t1,t2);
	}
      entity_type(e) = t2;
    }
  entity_storage(e) = make_storage_rom(); 
  
  /* Temporally put the list of struct/union entities defined in decl_psec_list to 
     initial value of ent*/
  entity_initial(e) = (value) ld;
  
}

list TakeDeriveEntities(list le)
{
  list lres = NIL;
  MAP(ENTITY, e, 
  {
    list ltmp = (list) entity_initial(e);
    if (ltmp != NIL)
      lres = gen_nconc(lres,ltmp);
    entity_initial(e) = value_undefined;
  },le);
  return lres;
}

void UpdateDerivedEntities(list ld, list le, stack ContextStack)
{
  MAP(ENTITY, e,
  {
    UpdateDerivedEntity(ld,e,ContextStack);
  },le);

} 
entity MakeDerivedEntity(string name, list members, bool is_external, int i)
{
  entity ent;  
  switch (i) {
  case is_type_struct: 
    {
      ent = CreateEntityFromLocalNameAndPrefix(name,STRUCT_PREFIX,is_external);	
      entity_type(ent) = make_type_struct(members);
      break;
    }
  case is_type_union: 
    {
      ent = CreateEntityFromLocalNameAndPrefix(name,UNION_PREFIX,is_external);	
      entity_type(ent) = make_type_union(members);
      break;
    }
  case is_type_enum:
    {
      ent = CreateEntityFromLocalNameAndPrefix(name,ENUM_PREFIX,is_external);	
      entity_type(ent) = make_type_enum(members);
      break;
    }
  }
  entity_storage(ent) = make_storage_rom();

  return ent;
}

/*******************  MISC *******************/


storage MakeStorageRam(entity e, bool is_external, bool is_static)
{
  ram r; 
  if (is_external)
    {
      if (is_static)
	{
	  r = make_ram(get_current_compilation_unit_entity(),
		       StaticArea,
		       ComputeAreaOffset(StaticArea,e),
		       NIL);
	  /*the offset must be recomputed lately, when we know the size of the variable */
	}
      else 
	{
	  /* This must be a variable, not a function/typedef/struct/union/enum. 
	     The variable is declared outside any function, and hence is global*/
	  r = make_ram(get_top_level_entity(),
		       get_top_level_entity(), 
		       ComputeAreaOffset(get_top_level_entity(),e),
		       NIL);
	  /* the offset must be recomputed lately, when we know the size of the variable */
	}
    }
  else
    { 
      /* ADD BLOCK SCOPE */
      if (is_static)
	{
	  r = make_ram(get_current_module_entity(),
		       StaticArea,
		       ComputeAreaOffset(StaticArea,e),
		       NIL);
	  /*the offset must be recomputed lately, when we know the size of the variable */
	}
      else
	{
	  r = make_ram(get_current_module_entity(),
		       DynamicArea,
		       ComputeAreaOffset(DynamicArea,e),
		       NIL);
	  /* the offset must be recomputed lately, when we know the size of the variable */
	}
    }
  return make_storage_ram(r);
}

string CreateMemberScope(string derived, bool is_external)
{
  /* We have to know the context : 
     - if the struct/union is declared outside any function, its scope is the CurrentCompilationUnit
     - if the struct/union is declared inside a function, we have to know the CurrentBlock, 
     which is omitted for the moment 
        - if the function is static, its scope is CurrentCompilationUnit!CurrentModule
        - if the function is global, its scope is CurrentModule

  The name of the struct/union is then added to the field entity name, with 
  the MEMBER_SEP_STRING */

  string s;
  pips_debug(3,"Struc/union name is %s\n",derived);
  if (is_external)
    s = strdup(concatenate(compilation_unit_name,derived,MEMBER_SEP_STRING,NULL));	
  else
    {
      if (static_module_p(get_current_module_entity()))
	s = strdup(concatenate(compilation_unit_name,
			get_current_module_name(),MODULE_SEP_STRING,
			derived,MEMBER_SEP_STRING,NULL));	
      else 
	s = strdup(concatenate(get_current_module_name(),MODULE_SEP_STRING,
			derived,MEMBER_SEP_STRING,NULL));
    } 
  pips_debug(3,"The struct/union member's scope is %s\n",s);
  return s;
}

/* Move this to ri-util !!!*/

string list_to_string(list l)
{
  string result = NULL;
  if (l==NIL) return "";
  MAP(STRING,s, 
  {
    if (result==NULL)
      result = strdup((const char *)s);
    else 
      result = strdup(concatenate(result,s,NULL)); 
  }, l);
  return result;
}

value MakeEnumeratorInitialValue(list enum_list, int counter)
{
  /* The initial value = 0 if this is the first member in the enumerator list
     else, it is equal to : intial_value(predecessor) + 1 */
  value v = value_undefined; 
  if (counter == 1)
    v = make_value_constant(make_constant_int(0));
  else 
    {
      /* Find the predecessor of the counter-th member */
      entity pre = ENTITY(gen_nth(counter-1, enum_list));
      value vp = entity_initial(pre);
      if (value_constant_p(vp))
	{
	  constant c = value_constant(vp);
	  if (constant_int_p(c))
	    {
	      int i = constant_int(c);
	      v = make_value_constant(make_constant_int(i+1));
	    }
	}
    }
  return v;
}


int ComputeAreaOffset(entity a, entity v)
{
  type ta = entity_type(a);
  area aa = type_area(ta);
  int offset = area_size(aa);
  
  /* Update the size and layout of the area aa. 
     This function is called too earlier, we may not have the size of v.
     To be changed !!!*/

  area_size(aa) = offset + 0;
  area_layout(aa) = gen_nconc(area_layout(aa), CONS(ENTITY, v, NIL));
  return offset;
}

list MakeParameterList(list l1, list l2, stack FunctionStack)
{
  /* l1 is a list of parameter names and it represents the exact order in the parameter list
     l2 is a list of entities with their type, storage,... and the order can be different from l1
     In addition, l2 can be incomplete wrt l1, so other entities must be created from l1, with 
     default type : scalar integer variable. 
     
     We create the list of parameters with the order of l1, and the parameter type and mode 
     are retrieved from l2. 

     Since the offset of formal argument in l2 can be false, we have to update it here by using l1 */
  list l = NIL;
  int offset = 1;
  entity function = stack_head(FunctionStack);
  MAP(STRING, s,
  {
    parameter p = FindParameterEntity(s,offset,l2);
    if (parameter_undefined_p(p))
      {
	/* s is not declared in l2, create the corresponding entity/ formal variable
	 and add it to the declaration list, because it cannot be added with par_def in l2*/
	entity ent = FindOrCreateEntity(entity_user_name(function),s);
	variable v = make_variable(make_basic_int(DEFAULT_INTEGER_TYPE_SIZE),NIL,NIL);
	entity_type(ent) = make_type_variable(v);
	entity_storage(ent) = make_storage_formal(make_formal(function,offset));
	AddToDeclarations(ent,function);
	p = make_parameter(entity_type(ent),make_mode_reference());
      }
    l = gen_nconc(l,CONS(PARAMETER,p,NIL));
    offset++;
  },l1);
  return l;
}

parameter FindParameterEntity(string s, int offset, list l)
{
  MAP(ENTITY,e,
  {
    string name = entity_user_name(e);
    if (strcmp(name,s)==0)
      {
	type t = entity_type(e);
	mode m = make_mode_reference(); /* to be verified in C, when by reference, when by value*/
	/*
	  What about the storage of 

	  void AMMPmonitor( vfs,ffs,nfs,op )
	  int  (*vfs[])(),(*ffs[])();
	  int nfs;
	  FILE *op;*/

	storage st = entity_storage(e);
	if (storage_formal_p(st))
	  {
	    formal f = storage_formal(st);
	    formal_offset(f) = offset;
	  }
	return make_parameter(t,m);
      }
  },l);
  return parameter_undefined;
}

void AddToCalledModules(entity e)
{
  if (!intrinsic_entity_p(e))
    {
      bool already_here = FALSE;
      string n = top_level_entity_p(e)?entity_local_name(e):entity_name(e);
      MAP(STRING,s, 
      {
	if (strcmp(n, s) == 0)
	  {
	    already_here = TRUE;
	    break;
	  }
      }, CalledModules);
      
      if (! already_here)
	{
	  pips_debug(2, "Adding %s to list of called modules\n", n);
	  CalledModules = CONS(STRING, strdup(n), CalledModules);
	}
    }
}

/******************** STACK *******************************/

/* Pop n times the stack s*/
void NStackPop(stack s, int n)
{
  while (n-->0)
    stack_pop(s);
}

/* The OffsetStack is poped n times, where n is the number of formal arguments
   of the actual function */
void StackPop(stack OffsetStack)
{
  int n = basic_int((basic) stack_head(OffsetStack));
  NStackPop(OffsetStack,n);
}

/* The OffsetStack is pushed incrementally */
void StackPush(stack OffsetStack)
{
  int i = basic_int((basic) stack_head(OffsetStack));
  stack_push((char *) make_basic_int(i+1),OffsetStack); 
}
