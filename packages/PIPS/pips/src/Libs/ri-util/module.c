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
/* To have asprintf(): */
#include <stdlib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "text.h"
#include "text-util.h"
#include "ri.h"
#include "ri-util.h"

#include "misc.h"

static bool module_coherent_p=true;
static entity checked_module = entity_undefined;


/* Check if the given name is a plain module (and not a compilation unit)
   internal name
 */
bool module_name_p(string name) {
  return (!compilation_unit_p(name) && strstr(name, MODULE_SEP_STRING) == NULL);
}


/* Check if the given name is a static module name.
 */
bool static_module_name_p(const char* name)
{
  /* An entity is a static module if its name contains the FILE_SEP_STRING
     but the last one is not the last character of the name string */
  /* FI: I doubt this is true. Maybe if you're sure name is the name of a module? */
  return (!compilation_unit_p(name) && strstr(name, FILE_SEP_STRING) != NULL);
}

/* Get a new name for a module built from a prefix.

   @param prefix is the prefix string

   @param prevent_suffix is a flag that prevent from adding a "_0" suffix if possible

   @return the first module name (malloc'd string) of the form
   "<prefix>_<integer>" that do not correspond to an existing module with
   integer starting at 0
 */
string
build_new_top_level_module_name(const char* prefix, bool prevent_suffix) {
  string name;
  int version = 0;

  if(prevent_suffix) {
    name = strdup(prefix);
  }

  for(;;) {
    if(!prevent_suffix || version!=0)
      asprintf(&name, "%s_%d", prefix, version);

    if (module_name_to_entity(name) == entity_undefined)
      break;

    free(name);
    version++;
  }

  return name;
}


/* Check if the given module entity is a static module.
 */
bool static_module_p(entity e) {
  return static_module_name_p(entity_name(e));
}


/* Check if the given name is a compilation unit internal name.
 */
bool compilation_unit_p(const char* module_name) {
  /* A module name is a compilation unit if and only if its last character is
     FILE_SEP */
  if (module_name[strlen(module_name)-1]==FILE_SEP)
    return true;
  return false;
}


/* Check if the given module entity is a compilation unit.
 */
bool compilation_unit_entity_p(entity e) {
  if (e == entity_undefined) return false;
  return compilation_unit_p(entity_name(e));
}


bool
variable_in_module_p2(v,m)
entity v;
entity m;
{
    bool in_module_1 = 
	same_string_p(module_local_name(m), entity_module_name(v));
    list decl =  code_declarations (value_code(entity_initial (m)));

    bool in_module_2 = entity_is_argument_p(v,decl);

    ifdebug(8) {
	if (!in_module_1 || !in_module_2) {
	    pips_debug(8, "variable %s not in declarations of %s\n", 
		       entity_name(v),entity_name(m));
	    dump_arguments(decl);
	}
    }
    return (in_module_1 && in_module_2);
}


void 
variable_declaration_verify(reference ref)
{

    if (!variable_in_module_p2(reference_variable(ref),checked_module))
    { 	module_coherent_p = false;
	fprintf(stderr,
			   "variable non declaree %s\n",
			   entity_name(reference_variable(ref)));
    }
}

void 
symbolic_constant_declaration_verify(call c)
{
    
    if (value_symbolic_p(entity_initial(call_function(c)))) {
	if (!variable_in_module_p2(call_function(c),checked_module))
	{    module_coherent_p = false;
	     ifdebug(4) fprintf(stderr,
				"variable non declaree %s\n",
				entity_name(call_function(c)));
	 }
    }
}

void 
add_non_declared_reference_to_declaration(reference ref)
{
    entity var = reference_variable(ref);
    /* type t = entity_type(var); */

    if (!variable_in_module_p2(var,checked_module))
    { 
	value val = entity_initial(checked_module);
	code ce = value_code(val);
	list decl =  code_declarations (ce);
	entity ent= entity_undefined;
	string name=strdup(concatenate(
	    module_local_name(checked_module), 
	    MODULE_SEP_STRING,
	    entity_local_name(var), 
	    NULL));
	if ((ent = gen_find_tabulated(name,entity_domain)) ==entity_undefined)
	{
	    pips_debug(8, "ajout de la variable symbolique %s au module %s\n",
		       entity_name(var), entity_name(checked_module)); 

	    ent =  make_entity(name,
			       copy_type(entity_type(var)),
			       copy_storage(entity_storage(var)),
			       copy_value(entity_initial(var)));

	    code_declarations(ce) = gen_nconc(decl,
					      CONS(ENTITY, ent, NIL));
	}
    }
}

void 
add_symbolic_constant_to_declaration(call c)
{    
    if (value_symbolic_p(entity_initial(call_function(c)))) {
	value val = entity_initial(checked_module);
	code ce = value_code(val);
	list decl =  code_declarations (ce);
	entity ent= entity_undefined;
	string name=strdup(concatenate(module_local_name(checked_module), 
				       MODULE_SEP_STRING,
				       entity_local_name(call_function(c)), NULL));
	if ((ent = gen_find_tabulated(name,entity_domain)) == entity_undefined) {
	    ifdebug(8) 
		(void) fprintf(stderr,"ajout de la variable symbolique %s au module %s\n",
			       entity_name(call_function(c)), 
			       entity_name(checked_module)); 

	    ent =  make_entity( name,
				copy_type(entity_type(call_function(c))),
			       copy_storage(entity_storage(call_function(c))),
			       copy_value(entity_initial(call_function(c))));

	    code_declarations(ce) = gen_nconc(decl,
					      CONS(ENTITY, ent, NIL));
	}
    }
}

bool 
variable_declaration_coherency_p(entity module, statement st)
{
    module_coherent_p = true;
    checked_module = module;
    gen_multi_recurse(st,
		reference_domain,
		gen_true,
		add_non_declared_reference_to_declaration, NULL);
    module_coherent_p = true;
    gen_multi_recurse(st,
		call_domain,
		gen_true,
		add_symbolic_constant_to_declaration, NULL);
    checked_module = entity_undefined;
    return module_coherent_p;
}

/****************************************************** DECLARATION COMMENTS */

/*  Look for the end of header comments: */
static string
get_end_of_header_comments(string the_comments)
{
    string end_of_comment = the_comments;
    
    for(;;) {
	string next_line;
	/* If we are at the end of string or the line is not a comment
           (it must be a PROGRAM, FUNCTION,... line), it is the end of
           the header comment: */
	if (!comment_string_p(end_of_comment))
	    break;

	if ((next_line = strchr(end_of_comment, '\n')) != NULL
	    || *end_of_comment !=  '\0')
	    /* Look for next line */
	    end_of_comment = next_line + 1;
	else
	    /* No return char: it is the last line and it is a
               comment... Should never happend in a real program
               without any useful header! */
	    pips_assert("get_end_of_header_comments found only comments!",
			false);
    }
    return end_of_comment;
}


/* Get the header comments (before PROGRAM, FUNCTION,...) from the
   text declaration: */
sentence
get_header_comments(entity module)
{
    int length;
    string end_of_comment;
    string extracted_comment;
    /* Get the textual header: */
    string the_comments = code_decls_text(entity_code(module));

    end_of_comment = get_end_of_header_comments(the_comments);
    /* Keep room for the trailing '\0': */
    length = end_of_comment - the_comments;
    extracted_comment = (string) malloc(length + 1);
    (void) strncpy(extracted_comment, the_comments, length);
    extracted_comment[length] = '\0';
    return make_sentence(is_sentence_formatted, extracted_comment);  
}


/* Get all the declaration comments, that are comments from the
   PROGRAM, FUNCTION,... stuff up to the end of the declarations: */
sentence
get_declaration_comments(entity module)
{
    string comment;
    string old_extracted_comments;
    string extracted_comments = strdup("");
    /* Get the textual header: */
    string the_comments = code_decls_text(entity_code(module));
    
    comment = get_end_of_header_comments(the_comments);
    /* Now, gather all the comments: */
    for(;;) {
	string next_line;
	if (*comment == '\0')
	    break;
	next_line = strchr(comment, '\n');
	if (comment_string_p(comment)) {
	    string the_comment;
	    /* Add it to the extracted_comments: */
	    if (next_line == NULL)
		/* There is no newline, so this is the last
		   comment line: */
		the_comment = strdup(comment);
	    else {
		/* Build a string that holds the comment: */
		the_comment = (string) malloc(next_line - comment + 2);
		(void) strncpy(the_comment, comment, next_line - comment + 1);
		the_comment[next_line - comment + 1] = '\0';
	    }
	    old_extracted_comments = extracted_comments;
	    extracted_comments = strdup(concatenate(old_extracted_comments,
						    the_comment,
						    NULL));
	    free(old_extracted_comments);
	    free(the_comment);
	}
	/* Have a look to next line if any... */
	if (next_line == NULL)
	    break;
	comment = next_line + 1;
    }
    
    return make_sentence(is_sentence_formatted, extracted_comments);
}

/* list module_formal_parameters(entity func)
 * input    : an entity representing a function.
 * output   : the ordered list (of entities) of parameters of the function "func".
 * modifies : nothing.
 * comment  : Made from "entity_to_formal_integer_parameters()" that considers 
 *            only integer variables.
 */
list 
module_formal_parameters(entity func)
{
    list formals = NIL;
    list decl = list_undefined;

    pips_assert("func must be a module",entity_module_p(func));

    decl = code_declarations(entity_code(func));
    FOREACH(ENTITY, e, decl)
     {
       /* Dummy parameters should have been filtered out of the
	  declarations by the parser, but let's be careful here. As a
	  consequence, dummy parameters cannot be retrieved, except
	  thru the type. */
       if(!dummy_parameter_entity_p(e) && storage_formal_p(entity_storage(e)))
	     formals = CONS(ENTITY, e, formals);
     }
    
    return gen_nreverse(formals);
}

/* Number of user declaration lines for a module */
int
module_to_declaration_length(entity func)
{
    value v = entity_initial(func);
    code c = code_undefined;
    int length = 0;

    if(!value_undefined_p(v)) {
	if(value_code_p(v)) {
	    string s = string_undefined;

	    c = value_code(v);
	    s = code_decls_text(c);
	    if(string_undefined_p(s)) {
		/* Is it allowed? The declaration text may be destroyed or
		 * may not exist when a module is synthesized or heavily
		 * transformed.
		 */
		length = 0;
	    }
	    else {
		char l;
		while((l=*s++)!= '\0')
		    if(l=='\n')
			length++;
	    }
	}
	else {
	    pips_internal_error("Entity %s is not a module",
				entity_name(func));
	}
    }
    else {
	/* Entity func has not been parsed yet */
	length = -1;
    }

    return length;
}

/* Find all references in the declaration list */
list declaration_supporting_references(list dl)
{
  list srl = NIL;

  /* FI: for efficiency, the type cache used in
     type_supporting_references() should be moved up here */
  FOREACH(ENTITY, v,dl)
  {
    if( ! entity_special_area_p(v))
    {
      type t = entity_type(v);
      /* FI: we should also look up the initial values */
      srl = type_supporting_references(srl, t);
    }
  }

  return srl;
}



/* The function itself is not in its declarations. Maybe, it should be changed in the parser?

   A new list is allocated to avoid sharing with code_declarations.
*/

list module_all_declarations(entity m)
{
  list dl = CONS(ENTITY, m, gen_copy_seq(entity_declarations(m)));
  return dl;
}

/*
   For C, the declaration in the module statements are added.

   Because this function relies on pipsdbm, it should be relocated
   into another library. Prime candidate is preprocessor : - (
 */
#include "pipsdbm.h"
#include "resources.h"
list module_to_all_declarations(entity m)
{
  list dl = CONS(ENTITY, m, gen_copy_seq(code_declarations(value_code(entity_initial(m)))));
  bool c_module_p(entity);

  if(c_module_p(m)) {
    //const char* module_name = entity_user_name(m);
    const char* module_name = entity_local_name(m);
    statement s = (statement) db_get_memory_resource(DBR_PARSED_CODE, module_name, true);
    list sdl = statement_to_declarations(s);

    FOREACH(ENTITY, e, sdl) {
      if(!entity_is_argument_p(e, dl))
	dl = gen_nconc(dl, CONS(ENTITY, e, NIL));
    }
    gen_free_list(sdl);
  }

  return dl;
}

/* Simplified versions of c_module_p() and fortran_module_p().
 *
 * They cannot be used before the parsing is advanced unlike the other
 * above two functions located in preprocessor, but they only require
 * stuff from ri-util.
*/
static bool x_language_module_p(entity m, unsigned int x)
{
  bool is_x = true;
  value mv = entity_initial(m);

  if(value_code_p(mv)) {
    code c = value_code(mv);
    is_x = language_tag(code_language(c))==x;
  }
  else
    is_x = false;

  return is_x;
}

bool c_language_module_p(entity m)
{
  return x_language_module_p(m, is_language_c);
}

bool fortran_language_module_p(entity m)
{
  return x_language_module_p(m, is_language_fortran);
}

bool fortran95_language_module_p(entity m)
{
  return x_language_module_p(m, is_language_fortran95);
}

/* Returns the entity rv that carries the value returned by module m,
 * when m is not a C void function or a Fortran subroutine. If safe_p,
 * abort if the function is void. Otherwise, return an
 * undefined_entity.
 *
 * rv is supposed to be allocated early by the parser.
 */
entity generic_function_to_return_value(entity m, bool safe_p)
{
  type ft = entity_basic_concrete_type(m);
  entity rv = entity_undefined;
  if(compilation_unit_entity_p(m)) {
    if(safe_p)
      pips_user_error("A return value is requested for compilation unit"
		      " \"%s\".\n", entity_user_name(m));
    else
      rv = entity_undefined;
  }
  else if(type_functional_p(ft)) {
    functional f = type_functional(ft);
    type r = compute_basic_concrete_type(functional_result(f));
    if(type_void_p(r)) {
      if(safe_p)
	pips_user_error("A return value is used for void function \"%s\".\n",
			entity_user_name(m));
      else
	rv = entity_undefined;
    }
    else {
      const char* mn = entity_local_name(m);
      rv = FindEntity(mn, mn);

      pips_assert("rv is defined", entity_defined_p(rv));
    }
  }
  else
    pips_internal_error("Return value requested for non-functional object"
			" \"%s\".\n", entity_user_name(m));

  return rv;
}

/* Returns the entity rv that carries the value returned by module m,
 * when m is not a C void function or a Fortran subroutine.
 *
 * rv is supposed to be allocated early by the parser.
 */
entity function_to_return_value(entity m)
{
  return generic_function_to_return_value(m, true);
}

/* Same as function_to_return_value(), but returns value_undefined
   when m is a C void function or a Fortran subroutine. */
entity any_function_to_return_value(entity m)
{
  return generic_function_to_return_value(m, false);
}

/* Check if m is a C void function or a Fortran subroutine. No
   information about m is available when false is returned. */
bool void_function_p(entity m)
{
  type ft = ultimate_type(entity_type(m));
  bool void_p = false;

  if(type_functional_p(ft)) {
    functional f = type_functional(ft);
    type r = functional_result(f);
    if(type_void_p(r)) {
      void_p = true;
    }
  }

  return void_p;
}

/* Build a list of functions from a string s containing SPACE
 * separated function names.
 *
 * Beware of ambiguities that static function names may cause.
 *
 * See also string_to_entities() and a similar piece of code in
 * inlining.c that was used to code this function.
 */
list string_to_user_modules(const char* s)
{
  list ml = NIL; // module list
  string ds = strdup(s);

  string c_name= NULL;
  for(c_name = strtok(ds," ") ;
      c_name ;
      c_name=strtok(NULL," ") ) {
    entity m = local_name_to_top_level_entity(c_name);
    if(entity_undefined_p(m))
      /* FI: warning at a very low level, with no information about
	 the context. Might be better to return list_undefined on a
	 conversion failure? */
      pips_user_warning("No function found for \"%s\".\n", c_name);
    else
      ml = CONS(ENTITY, m, ml);
  }
  free(ds);
  return ml;
}
/*
 *  that is all
 */
