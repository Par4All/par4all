/* $Id$ 
   $Log: util.c,v $
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

/* To avoid warnings */
extern char *strdup(const char *s1);

extern string compilation_unit_name;

extern hash_table keyword_typedef_table;

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

/******************* COMPILATION UNIT **********************/

entity get_current_compilation_unit_entity()
{
  return FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,compilation_unit_name);
}


/* A compilation unit is also considered as a module*/

void MakeCurrentCompilationUnitEntity(string name)
{
  entity e = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,name);
  entity_storage(e) = make_storage_rom();
  entity_type(e) = make_type_functional(make_functional(NIL,make_type_unknown()));
  entity_initial(e) = make_value(is_value_code, make_code(NIL,strdup(""), make_sequence(NIL)));
  set_current_module_entity(e);
  init_c_areas(); 
}

void ResetCurrentCompilationUnitEntity()
{
  reset_current_module_entity();
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


entity ExpressionToEntity(expression e)
{
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_call:
      return call_function(syntax_call(s));
    case is_syntax_reference:
      return reference_variable(syntax_reference(s));
    case is_syntax_range:
    case is_syntax_cast: 
    case is_syntax_sizeofexpression:
      break;
    default: 
      {
	pips_internal_error("unexpected syntax tag: %d\n", syntax_tag(s));
      }
    }
  return entity_undefined; 
}

expression MemberDerivedIdentifierToExpression(type t,string m)
{
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_pointer_p(b))
	{
	  type tp = basic_pointer(b);
	  return MemberDerivedIdentifierToExpression(tp,m);
	}
      if (basic_typedef_p(b))
	{
	  entity te = basic_typedef(b);
	  type tp = entity_type(te);
	  return MemberDerivedIdentifierToExpression(tp,m);
	}
      if (basic_derived_p(b))
	{
	  entity de = basic_derived(b);
	  string name = entity_user_name(de);
	  return IdentifierToExpression(strdup(concatenate(name,MEMBER_SEP_STRING,m,NULL))); 
	}
    }
  return expression_undefined;
}

expression MemberIdentifierToExpression(expression e, string m)
{
  /* Find the name of struct/union of m from the type of expression e*/
  syntax s = expression_syntax(e);
  switch (syntax_tag(s))
    {
    case is_syntax_call:
      {
	break;
      }
    case is_syntax_reference:
      {
	entity ent = reference_variable(syntax_reference(s));
	type t = entity_type(ent);
	return MemberDerivedIdentifierToExpression(t,m);
      }
    case is_syntax_range:
    case is_syntax_cast: 
    case is_syntax_sizeofexpression:
      break;
    default: 
      {
	pips_internal_error("unexpected syntax tag: %d\n", syntax_tag(s));
      }
    } 
  return expression_undefined;
}

expression IdentifierToExpression(string s)
{
  entity ent = FindEntityFromLocalName(s);
  type t = entity_type(ent);
  pips_debug(5,"Identifier is :%s\n",s);
  switch (type_tag(t)) {
  case is_type_variable: 
    return reference_to_expression(make_reference(ent,NIL));
  case is_type_functional:
    return MakeNullaryCall(ent);
  default:
    {
      pips_user_error("Which kind of expression?\n");
      return expression_undefined;
    }
  }
}

expression MakeReferenceExpression(expression exp, list lexp)
{
  syntax s = expression_syntax(exp);
  switch(syntax_tag(s)) {
  case is_syntax_reference:
    {
      reference r = syntax_reference(s);
      entity ent = reference_variable(r);
      list l = reference_indices(r);
      return reference_to_expression(make_reference(ent,gen_nconc(l,lexp)));
    }
  case is_syntax_call:
    {
      call c = syntax_call(s);
      list args = call_arguments(c);
      expression e = EXPRESSION(CAR(CDR(args)));
      reference r = expression_reference(e);
      entity ent = reference_variable(r);
      list l = reference_indices(r);
      return reference_to_expression(make_reference(ent,gen_nconc(l,lexp)));
    }
  default: 
    {
      pips_internal_error("expression bracket_comma_expression default: not treated yet\n");
      return expression_undefined;
    }
  }
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
  if (type_variable_p(t))
    {
      basic b = variable_basic(type_variable(t));
      if (basic_bit_p(b))
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
  return (strstr(entity_name(e), FILE_SEP_STRING) != NULL);
}

bool compilation_unit_p(string module_name)
{
  /* To be modified, the static function also has FILE_SEP_STRING */
  if (strstr(module_name,FILE_SEP_STRING) != NULL)
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
       
	1. FILE%MODULE:BLOCK~PREFIXname or MODULE:BLOCK~PREFIXname
        2. FILE%MODULE:PREFIXname or MODULE:PREFIXname
        3. FILE%PREFIXname
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

entity FindOrCreateCurrentEntity(string name, 
				 c_parser_context context,
				 bool is_external,
				 bool is_formal, 
				 int offset, 
				 entity CurrentFunction)
{
  entity ent; 

  /*************************** Scope part ***************************************/

  ifdebug(5)
    {
      pips_debug(5,"Entity local name %s\n",name);
      if (c_parser_context_scope(context) != NULL)
	pips_debug(5,"Current scope: %s\n",c_parser_context_scope(context));
      if (c_parser_context_type(context) != type_undefined)
	pips_debug(5,"Current type tag: %d\n",type_tag(c_parser_context_type(context)));
      if (c_parser_context_storage(context) != storage_undefined)
	pips_debug(5,"Current storage tag: %d\n",storage_tag(c_parser_context_storage(context)));
      pips_debug(5,"is_typedef: %d\n",c_parser_context_typedef(context));
      pips_debug(5,"is_static: %d\n",c_parser_context_static(context));
      pips_debug(5,"is_external: %d\n",is_external);
      pips_debug(5,"is_formal: %d\n",is_formal);
      if (is_formal)
	pips_debug(5,"Current function %s\n",entity_local_name(CurrentFunction));
      /* CurrentFunction is only used for formal variable cases */
    }

  if (c_parser_context_typedef(context))
    {
      /* Tell the lexer about the new type names : add to keyword_typedef_table */
      hash_put(keyword_typedef_table,strdup(name),(void *) TK_NAMED_TYPE);
      pips_debug(5,"Add typedef name %s to hash table\n",name);
      ent = CreateEntityFromLocalNameAndPrefix(name,TYPEDEF_PREFIX,is_external);
    }
  else
    {
      if (c_parser_context_scope(context)!=NULL)
	ent = find_or_create_entity(strdup(concatenate(c_parser_context_scope(context),name,NULL)));
      else 
	{
	  /* CurrentScope = NULL, not extern/typedef/struct/union/enum  */
	  if (is_external)
	    {
	      if (c_parser_context_static(context)) 
		ent = find_or_create_entity(strdup(concatenate(compilation_unit_name,
							       name,NULL)));
	      else 
		ent = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,name);
	    }
	  else 
	    {
	      /* This can be a function entity declaration 
		 or the module's variable : add block scope here */
	      if (!type_undefined_p(c_parser_context_type(context)) && type_functional_p(c_parser_context_type(context)))
		{
		  if (c_parser_context_static(context))
		    ent = find_or_create_entity(strdup(concatenate(compilation_unit_name,
								   FILE_SEP_STRING,name,NULL)));
		  else 
		    ent = FindOrCreateEntity(TOP_LEVEL_MODULE_NAME,name);
		}
	      else
		{
		  /* There are 2 cases :
		     1. We are in the compilation unit, this must be a formal variable
		     2. We are in a normal module, and we have to see if the module is static or not */
		
		  /* Attention: the compilation unit satisfies the static_module_p test :-)*/

		  if (is_formal)
		    {
		      if (top_level_entity_p(CurrentFunction))
			ent = find_or_create_entity(strdup(concatenate(entity_local_name(CurrentFunction),
								       MODULE_SEP_STRING,name,NULL)));
		      else
			ent = find_or_create_entity(strdup(concatenate(entity_name(CurrentFunction),
								       MODULE_SEP_STRING,name,NULL)));
		    }
		  else
		    {
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
    }
  pips_debug(5,"Entity global name: %s\n",entity_name(ent));
  

  if (!entity_intrinsic_p(ent))
    {

      /*************************** Type part ***************************************/
      
      if (!type_undefined_p(c_parser_context_type(context)))
	{
	  if (type_variable_p(c_parser_context_type(context)))
	    {
	      variable v = type_variable(c_parser_context_type(context)); 
	      variable_qualifiers(v) = c_parser_context_qualifiers(context);
	    }
	  entity_type(ent) = c_parser_context_type(context);
	}
      
      /*************************** Storage part ***************************************/
      
      if (!storage_undefined_p(c_parser_context_storage(context)))
	entity_storage(ent) = c_parser_context_storage(context);
      else
	{
	  if (is_external)
	    {
	      ram r; 
	      if (c_parser_context_static(context))
		{
		  r = make_ram(get_current_compilation_unit_entity(),
			       StaticArea,
			       ComputeAreaOffset(StaticArea,ent),
			       NIL);
		  /*the offset must be recomputed lately, when we know the size of the variable */
		}
	      else 
		{
		  /* This must be a variable, not a function/typedef/struct/union/enum. 
		     The variable is declared outside any function, and hence is global
		     
		     THIS CAN BE FUNTION ? */
		  r = make_ram(get_top_level_entity(),
			       get_top_level_entity(), 
			       ComputeAreaOffset(get_top_level_entity(),ent),
			       NIL);
		  /* the offset must be recomputed lately, when we know the size of the variable */
		}
	      entity_storage(ent) = make_storage_ram(r);
	    }
	  else 
	    {
	      /* This can be a function whose storage will be updated lately ? No, because CurrentStorage != undefined
		 or a static / dynamic / formal variable 
		 not a typedef/struct/union/enum */
	      if (is_formal)
		{
		  formal f = make_formal(CurrentFunction,offset);
		  entity_storage(ent) = make_storage_formal(f);
		  AddToDeclarations(ent,CurrentFunction);
		}
	      else
		{ 
		  /* ADD BLOCK SCOPE */
		  if (c_parser_context_static(context))
		    {
		      ram r = make_ram(get_current_module_entity(),
				       StaticArea,
				       ComputeAreaOffset(StaticArea,ent),
				       NIL);
		      /*the offset must be recomputed lately, when we know the size of the variable */
		      entity_storage(ent) = make_storage_ram(r);
		    }
		  else
		    {
		      ram r = make_ram(get_current_module_entity(),
				       DynamicArea,
				       ComputeAreaOffset(DynamicArea,ent),
				       NIL);
		      /* the offset must be recomputed lately, when we know the size of the variable */
		      entity_storage(ent) = make_storage_ram(r);
		    }
		}
	    }
	}
    }
  return ent;
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
      pips_debug(5,"Add entity %s to module %s\n",entity_local_name(e),entity_local_name(mod));
      code_declarations(value_code(entity_initial(mod))) = gen_nconc(code_declarations(value_code(entity_initial(mod))),
								     CONS(ENTITY,e,NIL));
    }
}

entity MakeDerivedEntity(string name, list members, bool is_external, int i)
{
  entity ent;  
  switch (i) {
  case 1: 
    {
      ent = CreateEntityFromLocalNameAndPrefix(name,STRUCT_PREFIX,is_external);	
      entity_type(ent) = make_type_struct(members);
      break;
    }
  case 2: 
    {
      ent = CreateEntityFromLocalNameAndPrefix(name,UNION_PREFIX,is_external);	
      entity_type(ent) = make_type_union(members);
      break;
    }
  case 3:
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

string CreateMemberScope(string derived, bool is_external)
{
  /* We have to know the context : 
     - if the struct/union is declared outside any function, its scope is the CurrentCompilationUnit
     - if the struct/union is declared inside a function, we have to know the CurrentBlock, 
     which is omitted for the moment 
        - if the function is static, its scope is CurrentCompilationUnit#CurrentModule
        - if the function is global, its scope is CurrentModule

  The name of the struct/union is then added to the field entity name, with 
  the MEMBER_SEP_STRING */

  string s;
  pips_debug(3,"Struc/union name is %s\n",derived);
  if (is_external)
    s = concatenate(compilation_unit_name,derived,MEMBER_SEP_STRING,NULL);	
  else
    {
      if (static_module_p(get_current_module_entity()))
	s = concatenate(compilation_unit_name,
			get_current_module_name(),MODULE_SEP_STRING,
			derived,MEMBER_SEP_STRING,NULL);	
      else 
	s = concatenate(get_current_module_name(),MODULE_SEP_STRING,
			derived,MEMBER_SEP_STRING,NULL);
    } 
  pips_debug(3,"The struct/union member's scope is %s\n",s);
  return strdup(s);
}

string list_to_string(list l)
{
  string result = NULL;
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

list MakeParameterList(list l1, list l2)
{
  /* l1 is a list of parameter names and it represents the exact order in the parameter list
     l2 is a list of entities with their type, storage,... and the order can be different from l1
     
     We create the list of parameters with the order of l1, and the parameter type and mode 
     are retrived from l2. 

     Since the offset of formal argument in l2 can be false, we have to update it here by using l1 */
  list l = NIL;
  int offset = 1;
  MAP(STRING, s,
  {
    parameter p = FindParameterEntity(s,offset,l2);
    l = CONS(PARAMETER,p,l);
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

/* To be moved to ri-util.c*/
bool entity_intrinsic_p(entity e)
{
  return (!value_undefined_p(entity_initial(e)) && value_intrinsic_p(entity_initial(e)));
}






