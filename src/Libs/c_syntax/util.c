/* $Id$ 
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
#include "parser_private.h" /* FI: for syntax.h */

#include "c_syntax.h"
#include "syntax.h" /* FI: To dump the symbol table. move in ri-util? */

#include "cyacc.h"

#include "resources.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "pipsdbm.h"
#include "transformations.h"
#include "properties.h"

extern string compilation_unit_name;
extern hash_table keyword_typedef_table;

extern list CalledModules;
extern entity StaticArea;
extern entity DynamicArea;

/* The data structure to tackle the memory allocation problem due to reparsing of compilatio unit 
static int previoussizeofGlobalArea;
entity previouscompunit;

*/

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
/* In C we have 4 areas 
   1. globalStaticArea: For the Global Variables, these variables are added into StaticArea.
   2. moduleStaticArea: For the Static module variables
   3. localStaticArea(SticArea): General Static variables
   4. DynamicArea
   The GlobalStaticArea and ModuleStaticArea are basically static areas but they are not defined globally.
   It is to be added to the Declarations for C
*/

void init_c_areas()
{
  DynamicArea = FindOrCreateEntity(get_current_module_name(), DYNAMIC_AREA_LOCAL_NAME);
  entity_type(DynamicArea) = make_type(is_type_area, make_area(0, NIL));
  entity_storage(DynamicArea) = make_storage_rom();
  entity_initial(DynamicArea) = make_value_unknown();
  AddEntityToDeclarations(DynamicArea, get_current_module_entity());

  StaticArea = FindOrCreateEntity(get_current_module_name(), STATIC_AREA_LOCAL_NAME);
  entity_type(StaticArea) = make_type(is_type_area, make_area(0, NIL));
  entity_storage(StaticArea) = make_storage_rom();
  entity_initial(StaticArea) = make_value_unknown();
  AddEntityToDeclarations(StaticArea, get_current_module_entity());

  HeapArea = FindOrCreateEntity(compilation_unit_name, HEAP_AREA_LOCAL_NAME);
  entity_type(HeapArea) = make_type(is_type_area, make_area(0, NIL));
  entity_storage(HeapArea) = MakeStorageRom();
  entity_initial(HeapArea) = MakeValueUnknown();
  AddEntityToDeclarations(HeapArea, get_current_module_entity());

  // Dynamic variables whose size are not known are stored in Stack area 
  StackArea = FindOrCreateEntity(get_current_module_name(), STACK_AREA_LOCAL_NAME);
  entity_type(StackArea) = make_type(is_type_area, make_area(0, NIL));
  entity_storage(StackArea) = MakeStorageRom();
  entity_initial(StackArea) = MakeValueUnknown();
  AddEntityToDeclarations(StackArea, get_current_module_entity());

  entity msae = FindOrCreateEntity(compilation_unit_name,  STATIC_AREA_LOCAL_NAME);
  AddEntityToDeclarations(msae, get_current_module_entity());
  
  entity gsae = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,  STATIC_AREA_LOCAL_NAME);
  AddEntityToDeclarations(gsae, get_current_module_entity());

  /* This is because of the reparsing of the compilation unit 
     The area is set to zero and all the declarations are overrided and the memory is reallocated
     The area is reset to only when it is called by same compilation unit twice. 
     The code is dangerous hence it is commented. Please have a look

     if( get_current_compilation_unit_entity() == get_current_module_entity() &&
     (get_current_compilation_unit_entity() == previouscompilationunit))
     area_size(type_area(entity_type(msae))) = 0;
  
     if( get_current_compilation_unit_entity() == get_current_module_entity() && 
     (get_current_compilation_unit_entity() == previouscompilationunit))
     area_size(type_area(entity_type(gsae))) = previoussizeofGlobalArea ;
  */
}


/******************* COMPILATION UNIT **********************/

entity get_current_compilation_unit_entity()
{
  return FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,compilation_unit_name);
}


/* A compilation unit is also considered as a module*/

void MakeCurrentCompilationUnitEntity(string name)
{
  entity e = MakeCompilationUnitEntity(name);

  pips_debug(4,"Set current module entity for compilation unit %s\n",name);
  set_current_module_entity(e);
  //init_stack_storage_table();
  init_c_areas(); 
}

void ResetCurrentCompilationUnitEntity(bool is_compilation_unit_parser)
{
  if(is_compilation_unit_parser)
    CCompilationUnitMemoryAllocation(get_current_module_entity());
  reset_entity_type_stack_table();
  if (get_bool_property("PARSER_DUMP_SYMBOL_TABLE"))
    fprint_C_environment(stderr, get_current_module_entity());
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
  return make_call_expression(CreateIntrinsic(COMMA_OPERATOR_NAME),l);
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
      pips_debug(6,"Basic tag is %td",basic_tag(b));
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
      /* Could this be non declared variables ?*/
      /* This identifier has not been passed by the parser. 
         It is probably a function call => try this case now and complete others later.
         The scope of this function is global */
      pips_debug(5,"Create unparsed global function: %s\n",s);
      ent = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,s);
      entity_storage(ent) = make_storage_return(ent);
      entity_type(ent) = make_type_functional(make_functional(NIL,make_type_unknown()));
      entity_initial(ent) = make_value(is_value_code,make_code(NIL,strdup(""),make_sequence(NIL),NIL));
      return make_expression(make_syntax_reference(make_reference(ent,NIL)),normalized_undefined);
      /*return MakeNullaryCall(ent);*/
    }
  switch (type_tag(entity_type(ent))) {
  case is_type_variable: 
  case is_type_functional:
    {
      expression e = expression_undefined;
      value iv = entity_initial(ent);
      if(!value_undefined_p(iv) && value_symbolic_p(iv))
	/* Generate a call to an enum member */
	e = make_expression(make_syntax_call(make_call(ent, NIL)), normalized_undefined);
      else
	e = make_expression(make_syntax_reference(make_reference(ent,NIL)), normalized_undefined);
      return e;
    }
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

// Moved to ri-util/type.c

/******************* ENTITIES *******************/


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

  entity ent = entity_undefined;
  string global_name = string_undefined; 

  pips_debug(5,"Entity local name is \"%s\" with prefix \"%s\"\n",name,prefix);

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
  global_name = strdup(concatenate(compilation_unit_name,MODULE_SEP_STRING,
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
  pips_user_warning("Cannot find entity %s with prefix \"%s\" at line %d\n",
		    name, prefix, get_current_C_line_number());
  /* It may be a parser error or a normal behavior when an entity is
     used before it is defined as, for example, a struct in a typedef:
     typedef struct foo foo; */
  /* CParserError("Variable appears to be undefined\n"); */
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
    ent = find_or_create_entity(strdup(concatenate(compilation_unit_name,MODULE_SEP_STRING,
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
  string full_scope = c_parser_context_scope(context);
  string scope = strrchr(full_scope, BLOCK_SEP_CHAR);
  string block_scope = gen_strndup0(full_scope, 
				    (scope==NULL)? strlen(full_scope) : (unsigned) (scope-full_scope+1));
  type ct = c_parser_context_type(context);
  bool is_typedef = c_parser_context_typedef(context);
  bool is_static = c_parser_context_static(context);
  entity function = entity_undefined;
  bool is_formal;

  if(scope!=NULL)
    scope++;
  else {
    scope = full_scope;
  }

  if (stack_undefined_p(FormalStack) || stack_empty_p(FormalStack))
    is_formal = FALSE;
  else
    {
      is_formal= TRUE; 
      function = stack_head(FunctionStack);
    }

  ifdebug(5)
    {
      pips_debug(5,"Entity local name \"%s\"\n",name);
      pips_debug(5,"Context %p\n",context);
      if (full_scope != NULL) {
	pips_debug(5,"Current scope: \"%s\"\n",full_scope);
	pips_debug(5,"Local declaration scope: \"%s\"\n",scope);
	pips_debug(5,"Block scope: \"%s\"\n",block_scope);
      }
      pips_debug(5,"type %p: %s\n", ct, list_to_string(safe_c_words_entity(ct, NIL)));
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
      if (strcmp(scope,"") != 0)
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
	  if (is_formal) {
	    /* Formal parameter */
	    if(top_level_entity_p(function))
	      ent = find_or_create_entity(strdup(concatenate(entity_user_name(function),
							     MODULE_SEP_STRING,name,NULL)));
	    else {
	      // The function is local to a compilation unit
	      // Was this the best possible design?
	      string mn = entity_module_name(function);
	      string ln = entity_local_name(function);
	      ent = find_or_create_entity(strdup(concatenate(mn,ln,
							     MODULE_SEP_STRING,name,NULL)));
	    }
	  }
	  else 
	    {
	      /* scope = NULL, not extern/typedef/struct/union/enum  */
	      if (is_external)
		{
		  /* This is a variable/function declared outside any module's body*/
		  if (is_static)
		    /* Depending on the type, we should or not
		       introduce a MODULE_SEP_STRING, but the type is
		       still not fully known. Wait for UpdateFunctionEntity(). */
		    ent = find_or_create_entity(strdup(concatenate(compilation_unit_name,
								   MODULE_SEP_STRING,
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
  pips_debug(3,"Update final pointer type %td and %td\n",type_tag(pt),type_tag(t));
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
  pips_debug(3,"Update pointer entity %s with type pt=\"%s\"\n",
	     entity_name(e), list_to_string(c_words_entity(pt, NIL)));
  
  if (type_undefined_p(t))
    {
      pips_debug(3,"Undefined entity type\n");
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
  pips_debug(3,"Ends with type \"%s\" for entity %s\n",
	     list_to_string(c_words_entity(entity_type(e), NIL)), entity_name(e));
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

entity RenameFunctionEntity(entity oe)
{
  entity ne = oe;
  string s = entity_name(oe);

  /* The function name may be wrong because not enough information was
     available when it was created by FindOrCreateCurrentEntity(). */
  if(strchr(s, MODULE_SEP)!=NULL) {
    string mn = entity_module_name(ne);
    string ln = entity_local_name(ne);
    value voe = entity_initial(oe);
    stack s = get_from_entity_type_stack_table(oe);
    stack ns = stack_copy(s);

    ne = find_or_create_entity(strdup(concatenate(mn, ln, NULL)));
    entity_type(ne) = copy_type(entity_type(oe));
    entity_storage(ne) = copy_storage(entity_storage(oe));
    /* FI I do not understand how formal parameters could be declared before */
    if(value_undefined_p(voe) || value_unknown_p(voe))
      entity_initial(ne) = make_value(is_value_code,
				      make_code(NIL,strdup(""), make_sequence(NIL),NIL));
    else
      entity_initial(ne) = copy_value(entity_initial(oe));
    put_to_entity_type_stack_table(ne, ns);
   pips_debug(1, "entity %s renamed %s\n", entity_name(oe), entity_name(ne));
  }
  return ne;
}

void UpdateFunctionEntity(entity oe, list la)
{
  type t = entity_type(oe);
  //string s = entity_name(oe);
  //entity ne = oe;

  pips_debug(3,"Update function entity %s\n",entity_name(oe));

  ifdebug(8) {
    pips_debug(8, "with type list la: ");
    if(ENDP(la)) {
      (void) fprintf(stderr, "empty list");
    }
    MAP(PARAMETER, at, {
      (void) fprintf(stderr, "%s, ", list_to_string(safe_c_words_entity(parameter_type(at), NIL)));
    }, la);
      (void) fprintf(stderr, "\n");
  }

  if (type_undefined_p(t))
    entity_type(oe) = make_type_functional(make_functional(la,type_undefined));
  else
    CParserError("This entity must have undefined type\n");

  pips_debug(3,"Update function entity \"%s\" with type \"\%s\"\n",
	     entity_name(oe), list_to_string(safe_c_words_entity(entity_type(oe), NIL)));
}

/* This function replaces the undefined field in t1 by t2. 
   If t1 is an array type and the basic of t1 is undefined, it is replaced by the basic of t2.
   If t1 is a pointer type, if the pointed type is undefined it is replaced by t2. 
   If t1 is a functional type, if the result type of t1 is undefined, it is replaced by t2.
   The function is recursive.

   FI: This function used to create sharing between t1 and t2, which
   creates problems when t2 is later freed. t1 may be updated and
   returned or a new type may be created. */

type UpdateType(type t1, type t2)
{
  if (type_undefined_p(t1)) {
    if(!type_undefined_p(t2))
      return copy_type(t2);
    else
      return t2;
  }
  switch (type_tag(t1)) 
    {
    case is_type_variable: 
      {
	variable v = type_variable(t1);
	if (basic_undefined_p(variable_basic(v)))
	  {
	    pips_assert("type t2 is defined", !type_undefined_p(t2));
	    if (type_variable_p(t2))
	      return make_type_variable(make_variable(copy_basic(variable_basic(type_variable(t2))),
						      variable_dimensions(v),
						      gen_nconc(variable_qualifiers(v),
								gen_full_copy_list(variable_qualifiers(type_variable(t2))))));
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
	      type nt = type_undefined;
	    if (type_undefined_p(t2))
	      nt = make_type_unknown();
	    else
	      nt = copy_type(t2);
	    return make_type_functional(make_functional(functional_parameters(f),nt));
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

/* This function allocates the memory to the Current Compilation Unit */

void CCompilationUnitMemoryAllocation(entity module)
{
  /* Should be followed by preconditions */
  //entity msae = FindOrCreateEntity(compilation_unit_name,  STATIC_AREA_LOCAL_NAME);
  //area msa = type_area(entity_type(msae));
  //entity gsae = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,  STATIC_AREA_LOCAL_NAME);
  //area gsa = type_area(entity_type(gsae));
  //entity fdae = FindOrCreateEntity(get_current_module_name(), DYNAMIC_AREA_LOCAL_NAME);
  //area fda = type_area(entity_type(fdae));
  //entity fsae = FindOrCreateEntity(get_current_module_name(), STATIC_AREA_LOCAL_NAME);
  //area fsa = type_area(entity_type(fsae));
  
  /* Code for reallocation of memory problem due to reparsing of compilation unit
     previouscompunit = get_current_compilation_unit_entity();
     previoussizeofGlobalArea = area_size(gsa);
  */
  
  nodecl_p(module,ModuleStatement);
  list ld = entity_declarations(module);
  entity var = entity_undefined;

  //printf("\n\nArea Layout: %s\n\n", area_layout(fsa));
  
  pips_debug(8,"MEMORY ALLOCATION BEGINS\n");
    
  for(; !ENDP(ld); ld = CDR(ld))
    {
      var = ENTITY(CAR(ld));
      if(type_variable_p(entity_type(var)))
	{
	  // Add some preconditions here
	  storage s = entity_storage(var);
	  if(storage_ram_p(s))
	    {
	      ram r = storage_ram(s);
	      entity a = ram_section(r);
	      /* check the type of variable here to avoid conflict declarations */
	      if(!gen_in_list_p(var, code_externs(value_code(entity_initial(module))))) {
		  if(ram_offset(r) != UNDEFINED_RAM_OFFSET 
			    && ram_offset(r) != UNKNOWN_RAM_OFFSET 
		     && ram_offset(r) != DYNAMIC_RAM_OFFSET ) {
		    pips_user_warning
		      ("Multiple declarations of variable %s in different files\n"
		       ,entity_local_name(var));
		    CParserError("Fix your source code!\n");
		  }

		ram_offset(r) = area_size(type_area(entity_type(a)));
		add_C_variable_to_area(a,var);
	      }
	      else
		{
		  /* Donot allocate the memory for external variables:
		     Set the offset of ram -2 which signifies UNKNOWN offset
		  */

		  // Check type here to avoid conflict declarations
		  if(ram_offset(r) == UNKNOWN_RAM_OFFSET)
		    ram_offset(r) = UNDEFINED_RAM_OFFSET;
		}
	    }
	}
    }
}

/* This function is for MemoryAllocation for Module of C programs*/

void CModuleMemoryAllocation(entity module)
{
  list ld = entity_declarations(module);
  entity var = entity_undefined;
  
  pips_debug(8,"MEMORY ALLOCATION BEGINS\n");
  nodecl_p(module,ModuleStatement);
  //print_entities(ld);
  for(; !ENDP(ld); ld = CDR(ld))
    {
      var = ENTITY(CAR(ld));
     
      if(type_variable_p(entity_type(var)))
	{
	  storage s = entity_storage(var);
	  if(storage_ram_p(s))
	    {
	      ram r = storage_ram(s);
	      entity a = ram_section(r);
	      if(!gen_in_list_p(var, code_externs(value_code(entity_initial(module))))) {
		ram_offset(r) = area_size(type_area(entity_type(a)));
		if(a == StackArea)
		  ram_offset(r) = DYNAMIC_RAM_OFFSET;
		else
		  add_C_variable_to_area(a,var);
	      }
	      else{
		if(ram_offset(r) == UNKNOWN_RAM_OFFSET)
		ram_offset(r) = UNDEFINED_RAM_OFFSET;
	      }
	    }
	  if(storage_formal_p(s))
	    {
	      //DO NOTHING
	    }
	}
    }
}

void UpdateEntity(entity e, stack ContextStack, stack FormalStack, stack FunctionStack,
		  stack OffsetStack, bool is_external)
{
  /* Update the entity with final type, storage, initial value */
  
  //stack s = (stack) entity_storage(e); 
  stack s = get_from_entity_type_stack_table(e);
  type t = entity_type(e);
  c_parser_context context = stack_head(ContextStack);
  type tc = c_parser_context_type(context);
  type t1,t2;
  list lq = c_parser_context_qualifiers(context);

  pips_debug(3,"Update entity begins for \"%s\" with context %p\n", entity_name(e), context);

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
  //pips_assert("context type tc is defined", !type_undefined_p(tc));
  t2 = UpdateType(t,tc);

  while (stack_size(s) > 1)
    {
      t1 = stack_pop(s);
      t2 = UpdateType(t1,t2);
    }
  entity_type(e) = t2;
  
   
    
      
  /************************* STORAGE PART *******************************************/

  /* this field is always pre-defined. It is temporarilly used to store a type. See cyacc.y rule direct-decl: */


  if (!storage_undefined_p(c_parser_context_storage(context)))
    {
      pips_debug(3,"Current storage context is %td\n",
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
	  if(!value_intrinsic_p(entity_initial(function))) {
	    /* FI: Intrinsic do not have formal named parameters in PIPS
	       RI, however such parameters can be named in intrinsic
	       declarations. Problem with Validation/C_syntax/memcof.c */
	    AddToDeclarations(e,function);
	  }
	  entity_storage(e) = make_storage_formal(make_formal(function,offset));
	}
      else
	{
	  if(type_variable_p(entity_type(e))) {
	    /* The entities for the type_variable is added to the current module and the declarations*/  
	    entity function = get_current_module_entity();
	    
	    if(extern_entity_p(function, e))
	      AddToExterns(e,function);
	    
	    /* To avoid multiple declarations */
	    if(!gen_in_list_p(e, code_externs(value_code(entity_initial(function)))) &&
	       gen_in_list_p(e, code_declarations(value_code(entity_initial(function)))))
	      {
		user_log("Multiple declarations of variable %s in file\n",
			 entity_local_name(e));
		CParserError("Illegal Input");
		
	      }

	    AddToDeclarations(e,function);

	    // Check here if already stored the value
	    if(storage_undefined_p(entity_storage(e)))
	      entity_storage(e) = 
		MakeStorageRam(e,is_external,c_parser_context_static(context));
	  }
	  else if (type_functional_p(entity_type(e))){
	    /* The function should also added to the declarations */
	    if(!entity_undefined_p(get_current_module_entity()))
	      AddToDeclarations(e,get_current_module_entity());
	    entity_storage(e) = MakeStorageRom();
	  }
	  else
	    pips_assert("not implemented yet", FALSE);
	}
    }
  
   
  /************************* INITIAL VALUE PART ****************************************/
  if(value_undefined_p(entity_initial(e))) {
    entity_initial(e) = make_value_unknown();
  }

  pips_debug(3,"Update entity ends for \"%s\" with type \"%s\"\n",
	     entity_name(e),list_to_string(safe_c_words_entity(entity_type(e), NIL)));
  
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
 
  //stack s = (stack) entity_storage(e); 
  stack s = get_from_entity_type_stack_table(e);
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

void AddToExterns(entity e, entity mod)
{
  // the entity e can be extern variable and extern functions
  if(!gen_in_list_p(e, code_externs(value_code(entity_initial(mod)))))
  {
    pips_debug(5,"Add entity %s to extern declaration %s \n", entity_user_name(e), entity_user_name(mod));
    code_externs(value_code(entity_initial(mod)))
      = gen_nconc(code_externs(value_code(entity_initial(mod))),
		  CONS(ENTITY,e,NIL));
  }
}
void AddToDeclarations(entity e, entity mod)
{
  if (!gen_in_list_p(e,code_declarations(value_code(entity_initial(mod)))))
    {
      pips_debug(5,"Add entity %s to module %s\n",entity_user_name(e),entity_user_name(mod));
      code_declarations(value_code(entity_initial(mod)))
	= gen_nconc(code_declarations(value_code(entity_initial(mod))),
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
      //stack s = (stack) entity_storage(e); 
      stack s = get_from_entity_type_stack_table(e);
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

void InitializeEnumMemberValues(list lem)
{
  // enum member with implicit values are not yet fully instantiated
  list cem = list_undefined;
  int cv = 0;

  for(cem = lem; !ENDP(cem); POP(cem)) {
    entity em = ENTITY(CAR(cem));
    value emv = entity_initial(em);

    if(value_undefined_p(emv)) {
      entity_initial(em) = 
	make_value_symbolic(make_symbolic(int_to_expression(cv),
					  make_constant(is_constant_int, (void *) cv)));
    }
    else {
      cv = constant_int(symbolic_constant(value_symbolic(emv)));
    }
    cv++;
  }
}

entity MakeDerivedEntity(string name, list members, bool is_external, int i)
{
  entity ent = entity_undefined;  
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

  /* The storage part should not be called twice when reparsing compilation unit.
     We assume that double declarations are dealt with someone else */

storage MakeStorageRam(entity v, bool is_external, bool is_static)
{
  ram r = ram_undefined; 
  area lsa = type_area(entity_type(StaticArea));
  area dsa = type_area(entity_type(DynamicArea));
  entity moduleStaticArea = FindOrCreateEntity(compilation_unit_name,  STATIC_AREA_LOCAL_NAME);
  area msa = type_area(entity_type(moduleStaticArea));
  entity globalStaticArea = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,  STATIC_AREA_LOCAL_NAME);
  area gsa = type_area(entity_type(globalStaticArea));
  area stack = type_area(entity_type(StackArea));
  //area heap = type_area (entity_type(HeapArea));
  //entity m = get_current_module_entity();
  
  pips_assert("RAM Storage is used only for variables", type_variable_p(entity_type(v)));

  if (is_external)
    {
      if (is_static)
	{
	  r = make_ram(get_current_compilation_unit_entity(),
		       moduleStaticArea,
		       UNKNOWN_RAM_OFFSET /* ComputeAreaOffset(StaticArea,e) */,
		       NIL);
	    
	  /*the offset must be recomputed lately, when we know for
	    sure the size of the variables */
	  if(get_current_compilation_unit_entity() != get_current_module_entity() 
	     || !gen_in_list_p(v,area_layout(msa)))  
	    area_layout(msa) = gen_nconc(area_layout(msa), CONS(ENTITY, v, NIL));
	}
      else 
	{
	  /* This must be a variable, not a function/typedef/struct/union/enum. 
	     The variable is declared outside any function, and hence is global*/
	  
	  r = make_ram(get_top_level_entity(),
		       globalStaticArea, 
		       UNKNOWN_RAM_OFFSET /* ComputeAreaOffset(get_top_level_entity(),e) */,
		       NIL);
	  /* the offset must be recomputed lately, when we know for
	     sure the size of the variable */
	  /* Global variable can be declared in many different file */
	  if(!gen_in_list_p(v,area_layout(gsa)))
	    area_layout(gsa) = gen_nconc(area_layout(gsa), CONS(ENTITY, v, NIL));
	}
    }
  else
    { 
      /* ADD BLOCK SCOPE */
      if (is_static)
	{
	  r = make_ram(get_current_module_entity(),
		       StaticArea,
		       UNKNOWN_RAM_OFFSET /* ComputeAreaOffset(StaticArea,e) */,
		       NIL);
	  /*the offset must be recomputed lately, when we know for
	    sure the size of the variable */
	  area_layout(lsa) = gen_nconc(area_layout(lsa), CONS(ENTITY, v, NIL));
	}
      else
	{
	  int s = 0;
	  if(!SizeOfArray(v, &s))
	    {
	      r = make_ram(get_current_module_entity(),
			   StackArea,
			   DYNAMIC_RAM_OFFSET,
			   NIL);
	      area_layout(stack) = gen_nconc(area_layout(stack), CONS(ENTITY, v, NIL));
	    }
	  else {
	    r = make_ram(get_current_module_entity(),
			 DynamicArea,
			 UNKNOWN_RAM_OFFSET /* ComputeAreaOffset(DynamicArea,e) */,
			 NIL);
	    /* the offset must be recomputed lately, when we know for
	       sure the size of the variable */
	    area_layout(dsa) = gen_nconc(area_layout(dsa), CONS(ENTITY, v, NIL));
	  }
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

  pips_assert("Not used", FALSE);
  
  /* Update the size and layout of the area aa. 
     This function is called too earlier, we may not have the size of v.
     To be changed !!!

     FI: who wrote this? when should the offsets be computed? how do we deal with
     scoping?
  */

  pips_assert("Type is correctly defined", CSafeSizeOfArray(v)!=0);

  /* area_size(aa) = offset + 0; */
  area_size(aa) = offset + CSafeSizeOfArray(v);
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


// No Declaration Check for variables and functions
static bool declarationerror_p;

static bool referencenodeclfilter(reference r)
{
  entity e = reference_variable(r);

  if(variable_entity_p(e)){
  if(!(gen_in_list_p(e, entity_declarations(get_current_module_entity()))
       || gen_in_list_p(e, entity_declarations(get_current_compilation_unit_entity()))))
    {
      declarationerror_p = TRUE;
      user_log("\n\nNo declaration of variable: %s in module: %s \n",
	       entity_local_name(e),get_current_module_name());
    }
  }
  // There cannot a reference to variable of storage return
  if(storage_return_p(entity_storage(e))){
    declarationerror_p = TRUE;
     user_log("\n\nNo declaration of variable: %s in module: %s \n",
	      entity_local_name(e),get_current_module_name());
  }

  return TRUE;
}


static bool callnodeclfilter(call c)
{
  entity e = call_function(c);
  if(value_code_p(entity_initial(e))) {
  if(!(gen_in_list_p(e, entity_declarations(get_current_module_entity()))
       || gen_in_list_p(e, entity_declarations(get_current_compilation_unit_entity()))))
    {
      // Implicit declaration of an external function: returns an int
      // Compute arguments type from call c
      type ot = entity_type(e);
      type rt = make_type(is_type_variable, 
			  make_variable(make_basic(is_basic_int, (void *) DEFAULT_INTEGER_TYPE_SIZE),
					NIL,
					NIL));
      list ptl = NIL;
      list args = call_arguments(c);
      list carg = list_undefined;
      type ft = type_undefined;

      for(carg=args; !ENDP(carg); POP (carg)) {
	expression ce = EXPRESSION(CAR(carg));
	type ct = expression_to_type(ce);
	parameter cp = make_parameter(ct, make_mode(is_mode_value, UU));

	ptl = gen_nconc(ptl, CONS(PARAMETER, cp, NIL));
      }
      ft = make_type(is_type_functional, make_functional(ptl, rt));

      free_type(ot);
      entity_type(e) = ft;

      pips_user_warning("\n\nNo declaration of function %s in module %s\n"
			"Implicit declaration added\n",
	       entity_local_name(e), get_current_module_name());
    }
  }

  return TRUE;
}


void
nodecl_p(entity __attribute__ ((unused)) module, statement stat)
{
  declarationerror_p = FALSE;
  gen_multi_recurse(stat, reference_domain,referencenodeclfilter,gen_null,
		    call_domain,callnodeclfilter,gen_null, NULL);

  if(declarationerror_p)
    CParserError("Illegal Input\n");
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
