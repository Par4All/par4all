/* Pot-pourri of utilities for the internal representation. Some functions could be moved
 * to non-generic files such as entity.c.
 *
 * $Id$
 *
 * $Log: util.c,v $
 * Revision 1.13  1998/11/05 09:27:50  zory
 * EOLE tags for automatic function extraction added
 *
 * Revision 1.12  1998/10/09 11:35:15  irigoin
 * predicate heap_area_p() added
 *
 *
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
empty_local_label_name_p(s)
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
    return(empty_local_label_name_p(local_name(s)+strlen(LABEL_PREFIX))) ;
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
    static char local[MAXIMAL_MODULE_NAME_SIZE + 1];
    string p_sep = NULL;

    strncpy(local, s, MAXIMAL_MODULE_NAME_SIZE);
    local[MAXIMAL_MODULE_NAME_SIZE] = 0;
    if ((p_sep = strchr(local, MODULE_SEP)) == NULL) 
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
module_entitiesfilename(e)
entity e;
{
    return(string_entitiesfilename(entity_local_name(e)));
}

/* functions for expressions */

expression 
make_entity_expression(e, inds)
entity e;
cons *inds;
{
    reference E = make_reference(e, inds);

    return(make_expression(make_syntax(is_syntax_reference,E),
			   normalized_undefined));			    
}

string 
new_label_name(module)
entity module;
{
    static char name[ 64 ];
    static int init = 99999;
    char *module_name ;

    pips_assert( "new_label_name", module != 0 ) ;

    if( module == entity_undefined ) {
	module_name = "__GENSYM" ;
    }
    else {
	module_name = entity_local_name(module) ;
    }
    for(sprintf(name, "%s%s%s%d", module_name, MODULE_SEP_STRING, LABEL_PREFIX,
		--init);
	 init >= 0 && 
	    (entity)gen_find_tabulated(name, entity_domain) != entity_undefined ;
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
	fprintf(stderr, "[find_ith_parameter] entity %s is not a module\n", 
		entity_name(e));
	exit(1);
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
entity 
effect_entity(e)
effect e;
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
    cons *pi = reference_indices(ref);

    while (pi != NIL && --i > 0)
	pi = CDR(pi);
    
    pips_assert("reference_ith_index", pi != NIL);

    return(EXPRESSION(CAR(pi)));
}

/* functions for area */

bool 
dynamic_area_p(aire)
entity aire;
{
    return(strcmp(entity_local_name(aire), DYNAMIC_AREA_LOCAL_NAME) == 0);
}

bool 
static_area_p(aire)
entity aire;
{
    return(strcmp(entity_local_name(aire), STATIC_AREA_LOCAL_NAME) == 0);
}

bool 
heap_area_p(aire)
entity aire;
{
    return(strcmp(entity_local_name(aire), HEAP_AREA_LOCAL_NAME) == 0);
}


/* Test if a string can be a Fortran 77 comment: */
bool
comment_string_p(const string comment)
{
    char c = *comment;
    /* If a line begins with a non-space character, claims it may be a
       Fortran comment. Assume empty line are comments. */
    return c != '\0' && c != ' ' && c != '\t';
}
