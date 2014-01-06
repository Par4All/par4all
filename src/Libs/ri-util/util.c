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
/* Pot-pourri of utilities for the internal representation.
 * Some functions could be moved to non-generic files such as entity.c.
 */
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"

#include "top-level.h"

#include "ri-util.h"
//#include "preprocessor.h"

/* functions on strings for entity names */

/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */

/* Does take care of block scopes */
const char* global_name_to_user_name(const char* global_name)
{
  const char* user_name = string_undefined;
  char lc = global_name[strlen(global_name)-1];
  const char* p;

  /* First, take care of constrant strings and characters, wich may
     contain any of the PIPS special characters and strings.
     And do not forget Fortran format */

  if(lc=='"' || lc=='\'') {
    user_name = strchr(global_name, lc);
  }
  else if(lc==')') {
    user_name = strchr(global_name, '(');
  }
  else {

    /* Then all possible prefixes first */

    if (strstr(global_name,STRUCT_PREFIX) != NULL)
      user_name = strstr(global_name,STRUCT_PREFIX) + 1;
    else if (strstr(global_name,UNION_PREFIX) != NULL) {
      user_name = strstr(global_name,UNION_PREFIX) + 1;
    }
    else if (strstr(global_name,ENUM_PREFIX) != NULL)
      user_name = strstr(global_name,ENUM_PREFIX) + 1;
    else if (strstr(global_name,TYPEDEF_PREFIX) != NULL)
      user_name = strstr(global_name,TYPEDEF_PREFIX) + 1;

    else if (strstr(global_name,MEMBER_SEP_STRING) != NULL)
      user_name = strstr(global_name,MEMBER_SEP_STRING) + 1;

    else if (strstr(global_name,LABEL_PREFIX) != NULL)
      user_name = strstr(global_name,LABEL_PREFIX) + 1;
    else if (strstr(global_name,COMMON_PREFIX) != NULL)
      user_name = strstr(global_name,COMMON_PREFIX) + 1;
    else if (strstr(global_name,BLOCKDATA_PREFIX) != NULL) {
      /* Clash with the address-of C operator */
      user_name = strstr(global_name,BLOCKDATA_PREFIX)+1;
      //string s = strstr(global_name,BLOCKDATA_PREFIX);
      //
      //if(strlen(s)>1)
      //	user_name = s + 1);
      //else
      //user_name = s;
    }
    else if (strstr(global_name,MAIN_PREFIX) != NULL) {
      string s = strstr(global_name,MAIN_PREFIX) + 1;
      pips_debug(8, "name = %s \n", s);
      user_name = s;
    }
    /* Then F95module seperator */
    else if (strstr(global_name,F95MODULE_PREFIX) != NULL)
      user_name = strstr(global_name,F95MODULE_PREFIX) + 1;

    /* Then block seperators */
    else if (strstr(global_name,BLOCK_SEP_STRING) != NULL)
      user_name = strrchr(global_name,BLOCK_SEP_CHAR) + 1;

    /* Then module seperator */
    else if (strstr(global_name,MODULE_SEP_STRING) != NULL)
      user_name = strstr(global_name,MODULE_SEP_STRING) + 1;


    /* Then file seperator */
    else if (strstr(global_name,FILE_SEP_STRING) != NULL)
      user_name = strstr(global_name,FILE_SEP_STRING) + 1;
    else {
      pips_internal_error("no seperator ?");
      user_name = NULL;
    }

    /* Take care of the special case of static functions, leaving
       compilation unit names untouched */
    if ((p=strstr(user_name,FILE_SEP_STRING)) != NULL && *(p+1)!='\0')
      user_name = p + 1;
  }

  pips_debug(9, "global name = \"%s\", user_name = \"%s\"\n",
	     global_name, user_name);
  return user_name;
}

/* Does not take care of block scopes and returns a pointer */
const char* local_name(const char * s)
{
  char *start_ptr = strchr(s, MODULE_SEP);
  pips_assert("some separator", start_ptr != NULL);
  return start_ptr+1;
}

/* END_EOLE */

string make_entity_fullname(const char* module_name, const char* local_name)
{
    return(concatenate(module_name,
		       MODULE_SEP_STRING,
		       local_name,
		       (char *) 0));
}

//empty_local_label_name_p(s)
bool empty_string_p(const char* s)
{
    return(strcmp(s, "") == 0);
}


bool blank_string_p(const char*s ){
    extern int isspace(int);
    while(*s&&isspace(*s++));
    return !*s;
}

bool return_local_label_name_p(const char* s)
{
    return(strcmp(s, RETURN_LABEL_NAME) == 0);
}

bool empty_label_p(const char* s)
{
  // s must be a local label name
  pips_assert("no separator", strchr(s, MODULE_SEP) == NULL);
  // return(empty_local_label_name_p(local_name(s)+sizeof(LABEL_PREFIX)-1)) ;
  return (strcmp(s, EMPTY_LABEL_NAME) == 0);
}

bool empty_global_label_p(const char* gln)
{
  // gln must be a global label name
  const char* lln = local_name(gln);

  return empty_label_p(lln);
}

bool return_label_p(const char* s)
{
    return(return_local_label_name_p(local_name(s)+sizeof(LABEL_PREFIX)-1)) ;
}

entity find_label_entity(const char* module_name, const char* label_local_name)
{
    string full = concatenate(module_name, MODULE_SEP_STRING,
			      LABEL_PREFIX, label_local_name, NULL);

    pips_debug(5, "searched entity: %s\n", full);
    void * found = gen_find_tabulated(full, entity_domain);
    return (entity) (gen_chunk_undefined_p(found) ? entity_undefined : found);
}

/* Return the module part of an entity name.
 *
 * OK, this function name is pretty misleading: entity_name_to_entity_module_name().
 *
 * Maybe, it should be wrapped up in a higher-level function such as
 * entity_to_module_name().
 *
 * It uses a static buffer and should create trouble with long
 * function names.
 *
 * No memory allocation is performed, but it's result is transient. It
 * must be strdup'ed by the caller if it is to be preserved.
 *
 * To eliminate the static buffer and to allocate the returned string
 * would require lots of changes or add lots of memory leaks in PIPS
 * since the callers do not know they must free the result. Maybe we
 * should stack allocate a buffer of size strlen(s), but we would end
 * up returning a pointer to a popped area of the stack...
 */
const char* module_name(const char * s)
{
  /* FI: shouldn't we allocate dynamically "local" since its size is
     smaller than the size of "s"? */
    static char local[MAXIMAL_MODULE_NAME_SIZE + 1];

    char * local_iter=&local[0];
    const char *iter,*end;
    for(iter=s, end=s+MAXIMAL_MODULE_NAME_SIZE; *iter != MODULE_SEP && iter != end ; ++iter) {
        *local_iter++=*iter;
    }
    pips_assert("module name too long, or illegal", *iter == MODULE_SEP);
    *local_iter=0;
    return(local);
}

string string_codefilename(const char *s)
{
    return(concatenate(TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING,
		       s, SEQUENTIAL_CODE_EXT, NULL));
}

/* generation des noms de fichiers */
string module_codefilename(e)
entity e;
{
    return(string_codefilename(entity_local_name(e)));
}

string string_par_codefilename(const char *s)
{
    return(concatenate(TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING,
		       s, PARALLEL_CODE_EXT, NULL));
}

string module_par_codefilename(entity e)
{
    return(string_par_codefilename(entity_local_name(e)));
}

string string_fortranfilename(const char* s)
{
    return(concatenate(TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING,
		       s, SEQUENTIAL_FORTRAN_EXT, NULL));
}

bool string_fortran_filename_p(const char* s)
{
  int fnl = strlen(s);
  int sl = sizeof(SEQUENTIAL_FORTRAN_EXT)-1;
  bool is_fortran = false;

  if(fnl<=sl)
    is_fortran = false;
  else
    is_fortran = strcmp(SEQUENTIAL_FORTRAN_EXT, s+(fnl-sl))==0;

  return is_fortran;
}

string module_fortranfilename(entity e)
{
    return(string_fortranfilename(entity_local_name(e)));
}

string string_par_fortranfilename(const char* s)
{
    return(concatenate(TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING,
		       s, PARALLEL_FORTRAN_EXT, NULL));
}

string module_par_fortranfilename(entity e)
{
    return(string_par_fortranfilename(entity_local_name(e)));
}

string string_pp_fortranfilename(const char* s)
{
    return(concatenate(TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING,
		       s, PRETTYPRINT_FORTRAN_EXT, NULL));
}

string module_pp_fortranfilename(entity e)
{
    return(string_pp_fortranfilename(entity_local_name(e)));
}

string string_predicat_fortranfilename(const char* s)
{
    return(concatenate(TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING,
		       s, PREDICAT_FORTRAN_EXT, NULL));
}

string module_predicat_fortranfilename(entity e)
{
    return(string_predicat_fortranfilename(entity_local_name(e)));
}

string string_entitiesfilename(const char* s)
{
    return(concatenate(TOP_LEVEL_MODULE_NAME MODULE_SEP_STRING,
		       s, ENTITIES_EXT, NULL));
}

string module_entitiesfilename(entity e)
{
    return(string_entitiesfilename(entity_local_name(e)));
}




entity find_ith_parameter(entity e, int i)
{
  cons *pv = code_declarations(value_code(entity_initial(e)));

  if (! entity_module_p(e)) {
    pips_internal_error("entity %s is not a module",
			 entity_name(e));
  }
  while (pv != NIL) {
    entity v = ENTITY(CAR(pv));
    type tv = entity_type(v);
    storage sv = entity_storage(v);

    if (type_variable_p(tv) && storage_formal_p(sv)) {
      if (formal_offset(storage_formal(sv)) == i) {
        return(v);
      }
    }

    pv = CDR(pv);
  }

  return(entity_undefined);
}

/* returns true if v is the ith formal parameter of function f */
bool ith_parameter_p(entity f, entity v, int i)
{
  type tv = entity_type(v);
  storage sv = entity_storage(v);

  if (! entity_module_p(f)) {
    pips_internal_error("[ith_parameter_p] %s is not a module\n", entity_name(f));
  }

  if (type_variable_p(tv) && storage_formal_p(sv)) {
    formal fv = storage_formal(sv);
    return(formal_function(fv) == f && formal_offset(fv) == i);
  }

  return(false);
}

/* functions for references */

/* returns the ith index of an array reference */
expression reference_ith_index(reference ref, int i)
{
  int count = i;
  cons *pi = reference_indices(ref);

  while (pi != NIL && --count > 0)
    pi = CDR(pi);

  pips_assert("reference_ith_index", pi != NIL);

  return(EXPRESSION(CAR(pi)));
}

/* functions for areas */
bool allocatable_area_p(entity aire) {
  return same_string_p(module_local_name(aire), ALLOCATABLE_AREA_LOCAL_NAME);
}

bool dynamic_area_p(entity aire)
{
#ifndef NDEBUG
    bool result = same_string_p(module_local_name(aire), DYNAMIC_AREA_LOCAL_NAME);
    pips_assert("entity_kind is consistent", result == ( (entity_kind(aire) & ENTITY_DYNAMIC_AREA) == ENTITY_DYNAMIC_AREA));
#endif
  return entity_kind(aire) & ENTITY_DYNAMIC_AREA;
}

bool static_area_p(entity aire)
{
#ifndef NDEBUG
    bool result = same_string_p(module_local_name(aire), STATIC_AREA_LOCAL_NAME);
    pips_assert("entity_kind is consistent", result == ( (entity_kind(aire) & ENTITY_STATIC_AREA) == ENTITY_STATIC_AREA));
#endif
  return entity_kind(aire) & ENTITY_STATIC_AREA;
}

bool heap_area_p(entity aire)
{
#ifndef NDEBUG
    bool result = same_string_p(module_local_name(aire), HEAP_AREA_LOCAL_NAME);
    pips_assert("entity_kind is consistent", result == ( (entity_kind(aire) & ENTITY_HEAP_AREA) == ENTITY_HEAP_AREA));
#endif
  return entity_kind(aire) & ENTITY_HEAP_AREA;
}

bool formal_area_p(entity aire)
{
#ifndef NDEBUG
    bool result = same_string_p(module_local_name(aire), FORMAL_AREA_LOCAL_NAME);
    pips_assert("entity_kind is consistent", result == ( (entity_kind(aire) & ENTITY_FORMAL_AREA) == ENTITY_FORMAL_AREA));
#endif
  return entity_kind(aire) & ENTITY_FORMAL_AREA;
}

bool stack_area_p(entity aire)
{
#ifndef NDEBUG
    bool result = same_string_p(module_local_name(aire), STACK_AREA_LOCAL_NAME);
    pips_assert("entity_kind is consistent", result == ( (entity_kind(aire) & ENTITY_STACK_AREA) == ENTITY_STACK_AREA));
#endif
  return entity_kind(aire) & ENTITY_STACK_AREA;
}

bool pointer_dummy_targets_area_p(entity aire)
{
#ifndef NDEBUG
    bool result = same_string_p(module_local_name(aire), POINTER_DUMMY_TARGETS_AREA_LOCAL_NAME);
    pips_assert("entity_kind is consistent", result == ( (entity_kind(aire) & ENTITY_POINTER_DUMMY_TARGETS_AREA) == ENTITY_POINTER_DUMMY_TARGETS_AREA));
#endif
  return entity_kind(aire) & ENTITY_POINTER_DUMMY_TARGETS_AREA;
}



/* Returns the heap area "a" associated to module "f". Area "a" is always a
   defined entity.

   Assumes that f is defined, as well as its heap area.
*/
entity module_to_heap_area(entity f)
{
  entity a = FindEntity(entity_local_name(f), HEAP_AREA_LOCAL_NAME);

  pips_assert("The heap area is defined for module f.\n",
	      !entity_undefined_p(a));

  return a;
}

bool entity_area_p(entity e)
{
  return !type_undefined_p(entity_type(e)) && type_area_p(entity_type(e));//static_area_p(e) || dynamic_area_p(e) || heap_area_p(e) || stack_area_p(e);
}

bool entity_special_area_p(entity e)
{
  return entity_area_p(e) &&
    ( static_area_p(e) || dynamic_area_p(e) || heap_area_p(e) || stack_area_p(e) || pointer_dummy_targets_area_p(e));
}

/* Test if a string can be a Fortran 77 comment: */
bool comment_string_p(const string comment)
{
  char c = *comment;
  /* If a line begins with a non-space character, claims it may be a
     Fortran comment. Assume empty line are comments. */
  return c != '\0' && c != ' ' && c != '\t';
}


/* Remove trailing line feeds */
string string_remove_trailing_line_feeds(string s)
{
  int sl = strlen(s);
  if(sl>0) {
    string ntl = s+sl-1;
    while(sl>0 && *ntl=='\n') {
      *ntl='\000';
      ntl--;
      sl--;
    }
  }
  return s;
}


/* Get rid of linefeed/newline at the end of a string.
 *
 * This is sometimes useful to cleanup comments messed up by the
 * lexical analyzer.
 *
 * Warning: the argument s is updated if it ends up with LF
 */
string string_strip_final_linefeeds(string s)
{
  int l = strlen(s)-1;

  while(l>=0 && *(s+l)=='\n') {
    *(s+l) = '\000';
    l--;
  }

  return s;
}

/* Get rid of linefeed/newline at the end of a string.
 *
 * This is sometimes useful to cleanup comments messed up by the
 * lexical analyzer.
 *
 * Warning: the argument s is updated if it ends up with LF
 */
string string_fuse_final_linefeeds(string s)
{
  int l = strlen(s)-1;

  while(l>=1 && *(s+l)=='\n' && *(s+l-1)=='\n') {
    *(s+l) = '\000';
    l--;
  }

  return s;
}

/**
 * Test if a call is a user call
 */
bool user_call_p(call c) {
  entity f = call_function(c);
  value v = entity_initial(f);
  return value_code_p(v);
}

