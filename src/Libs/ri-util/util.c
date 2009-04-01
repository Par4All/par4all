/* Pot-pourri of utilities for the internal representation. 
 * Some functions could be moved to non-generic files such as entity.c.
 *
 * $Id$
 */
#include <stdio.h>
#include <string.h>

#include "linear.h"

#include "genC.h"
#include "misc.h"
#include "ri.h"

#include "top-level.h"

#include "ri-util.h"

/* functions on strings for entity names */

/* BEGIN_EOLE */ /* - please do not remove this line */
/* Lines between BEGIN_EOLE and END_EOLE tags are automatically included
   in the EOLE project (JZ - 11/98) */

/* Does take care of block scopes */
string global_name_to_user_name(string global_name)
{
  string user_name = string_undefined;
  char lc = global_name[strlen(global_name)-1];
  string p;

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
      pips_internal_error("no seperator ?\n");
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
string 
local_name(s)
string s;
{
    pips_assert("some separator", strchr(s, MODULE_SEP) != NULL);
    return strchr(s, MODULE_SEP)+1;
}

/* END_EOLE */

string 
make_entity_fullname(module_name, local_name)
string module_name, local_name;
{
    return(concatenate(module_name, 
		       MODULE_SEP_STRING, 
		       local_name, 
		       (char *) 0));
}

bool 
//empty_local_label_name_p(s)
empty_string_p(s)
string s;
{
    return(strcmp(s, "") == 0);
}

bool 
return_local_label_name_p(s)
string s;
{
    return(strcmp(s, RETURN_LABEL_NAME) == 0);
}

bool
empty_label_p(s)
string s;
{
  // s must be a local label name
  pips_assert("no separator", strchr(s, MODULE_SEP) == NULL);
  // return(empty_local_label_name_p(local_name(s)+strlen(LABEL_PREFIX))) ;
  return (strcmp(s, EMPTY_LABEL_NAME) == 0);
}

bool
empty_global_label_p(gln)
string gln;
{
  // gln must be a global label name
  string lln = local_name(gln); 

  return empty_label_p(lln);
}

bool 
return_label_p(s)
string s;
{
    return(return_local_label_name_p(local_name(s)+strlen(LABEL_PREFIX))) ;
}

entity 
find_label_entity(module_name, label_local_name)
string module_name, label_local_name;
{
    string full = concatenate(module_name, MODULE_SEP_STRING, 
			      LABEL_PREFIX, label_local_name, NULL);

    debug(5, "find_label_entity", "searched entity: %s\n", full);
    return (entity) gen_find_tabulated(full, entity_domain);
}

string 
module_name(s)
string s;
{
  /* FI: shouldnt'we allocate dynamically "local" since its size is
     smaller than the size of "s"? */
    static char local[MAXIMAL_MODULE_NAME_SIZE + 1];
    string p_sep = NULL;

    /* We have problem with C, an entity name does not always need MODULE_SEP*/
    /* FI: that is a problem for PIPS. Which entity name do not have a MODULE_SEP? */

    strncpy(local, s, MAXIMAL_MODULE_NAME_SIZE);
    local[MAXIMAL_MODULE_NAME_SIZE] = 0;
    if (((p_sep = strchr(local, MODULE_SEP)) == NULL) /* && ((p_sep = strstr(local, FILE_SEP_STRING)) == NULL ) */ ) 
      pips_error("module_name", 
		   "module name too long, or illegal: \"%s\"\n", s);
    else
	*p_sep = '\0';
    return(local);
}

string 
string_codefilename(s)
char *s;
{
    return(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, 
		       s, SEQUENTIAL_CODE_EXT, NULL));
}

/* generation des noms de fichiers */
string 
module_codefilename(e)
entity e;
{
    return(string_codefilename(entity_local_name(e)));
}

string 
string_par_codefilename(s)
char *s;
{
    return(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, 
		       s, PARALLEL_CODE_EXT, NULL));
}

string 
module_par_codefilename(e)
entity e;
{
    return(string_par_codefilename(entity_local_name(e)));
}

string 
string_fortranfilename(s)
char *s;
{
    return(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, 
		       s, SEQUENTIAL_FORTRAN_EXT, NULL));
}

bool string_fortran_filename_p(string s)
{
  int fnl = strlen(s);
  int sl = strlen(SEQUENTIAL_FORTRAN_EXT);
  bool is_fortran = FALSE;

  if(fnl<=sl)
    is_fortran = FALSE;
  else
    is_fortran = strcmp(SEQUENTIAL_FORTRAN_EXT, s+(fnl-sl))==0;

  return is_fortran; 
}

string 
module_fortranfilename(e)
entity e;
{
    return(string_fortranfilename(entity_local_name(e)));
}

string 
string_par_fortranfilename(s)
char *s;
{
    return(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, 
		       s, PARALLEL_FORTRAN_EXT, NULL));
}

string 
module_par_fortranfilename(e)
entity e;
{
    return(string_par_fortranfilename(entity_local_name(e)));
}

string 
string_pp_fortranfilename(s)
char *s;
{
    return(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, 
		       s, PRETTYPRINT_FORTRAN_EXT, NULL));
}

string 
module_pp_fortranfilename(e)
entity e;
{
    return(string_pp_fortranfilename(entity_local_name(e)));
}

string 
string_predicat_fortranfilename(s)
char *s;
{
    return(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, 
		       s, PREDICAT_FORTRAN_EXT, NULL));
}

string 
module_predicat_fortranfilename(e)
entity e;
{
    return(string_predicat_fortranfilename(entity_local_name(e)));
}

string 
string_entitiesfilename(s)
char *s;
{
    return(concatenate(TOP_LEVEL_MODULE_NAME, MODULE_SEP_STRING, 
		       s, ENTITIES_EXT, NULL));
}

string 
module_entitiesfilename(entity e)
{
    return(string_entitiesfilename(entity_local_name(e)));
}

/* functions for expressions */

expression 
make_entity_expression(entity e, cons *inds)
{
    syntax s = syntax_undefined;
    if( entity_constant_p(e) )
    {
        s = make_syntax_call(make_call(e,NIL));
    }
    else
    {
        reference r = make_reference(e, inds);
        s = make_syntax_reference(r);			    
    }
    return make_expression(s, normalized_undefined);
}

static int init = 100000;

void reset_label_counter()
{
  init = 100000;
}

string 
new_label_name(entity module)
{
    static char name[ 64 ];
    char *module_name ;

    pips_assert( "new_label_name", module != 0 ) ;

    if( module == entity_undefined ) {
	module_name = "__GENSYM" ;
    }
    else {
	module_name = module_local_name(module) ;
    }
    for(sprintf(name, "%s%s%s%d", module_name, MODULE_SEP_STRING, LABEL_PREFIX,
		--init);
	 init >= 0 && 
	    !entity_undefined_p(gen_find_tabulated(name, entity_domain)) ;
	sprintf(name, "%s%s%s%d", module_name, MODULE_SEP_STRING, LABEL_PREFIX,
		--init)) {
    /* loop */ 
    }
    if(init == 0) {
	pips_error("new_label_name", "no more available labels");
    }
    return(name);
}
	 
entity 
find_ith_parameter(e, i)
entity e;
int i;
{
    cons *pv = code_declarations(value_code(entity_initial(e)));

    if (! entity_module_p(e)) {
	pips_internal_error( "entity %s is not a module\n", 
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

/* returns TRUE if v is the ith formal parameter of function f */
bool 
ith_parameter_p(f, v, i)
entity f, v;
int i;
{
    type tv = entity_type(v);
    storage sv = entity_storage(v);

    if (! entity_module_p(f)) {
	fprintf(stderr, "[ith_parameter_p] %s is not a module\n",
		entity_name(f));
	exit(1);
    }

    if (type_variable_p(tv) && storage_formal_p(sv)) {
	formal fv = storage_formal(sv);
	return(formal_function(fv) == f && formal_offset(fv) == i);
    }

    return(FALSE);
}

/* functions for effects */
entity effect_entity(effect e)
{
  return(reference_variable(effect_reference(e)));
}

/* functions for references */

/* returns the ith index of an array reference */
expression 
reference_ith_index(ref, i)
reference ref;
int i;
{
  int count = i;
    cons *pi = reference_indices(ref);

    while (pi != NIL && --count > 0)
	pi = CDR(pi);
    
    pips_assert("reference_ith_index", pi != NIL);

    return(EXPRESSION(CAR(pi)));
}

/* functions for areas */

bool dynamic_area_p(entity aire)
{
  return(strcmp(entity_local_name(aire), DYNAMIC_AREA_LOCAL_NAME) == 0);
}

bool static_area_p(entity aire)
{
  return(strcmp(entity_local_name(aire), STATIC_AREA_LOCAL_NAME) == 0);
}

bool heap_area_p(entity aire)
{
  return(strcmp(entity_local_name(aire), HEAP_AREA_LOCAL_NAME) == 0);
}

bool stack_area_p(entity aire)
{
  return(strcmp(entity_local_name(aire), STACK_AREA_LOCAL_NAME) == 0);
}

bool area_entity_p(entity a)
{
  return static_area_p(a) || dynamic_area_p(a) || heap_area_p(a) || stack_area_p(a);
}


/* Test if a string can be a Fortran 77 comment: */
bool comment_string_p(const string comment)
{
  char c = *comment;
  /* If a line begins with a non-space character, claims it may be a
     Fortran comment. Assume empty line are comments. */
  return c != '\0' && c != ' ' && c != '\t';
}
