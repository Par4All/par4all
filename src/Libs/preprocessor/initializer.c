/* 
 * $Id$
 */
#include <stdio.h>
#include <string.h>

#include "genC.h"
#include "text.h"

#include "text-util.h"
#include "misc.h"
#include "properties.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "pipsdbm.h"

#include "resources.h"
#include "phases.h"

#define FILE_WARNING 							\
 "!\n"									\
 "!     This module was automatically generated by PIPS and should\n"	\
 "!     be updated by the user with READ and WRITE effects on\n"	\
 "!     formal parameters to be useful...\n"				\
 "!\n"

#define C_FILE_WARNING 							\
 "//\n"									\
 "//     This module was automatically generated by PIPS and should\n"	\
 "//     be updated by the user with READ and WRITE effects on\n"	\
 "//     formal parameters to be useful...\n"				\
 "//\n"

static string 
nth_formal_name(int n)
{
    char buf[20]; /* hummm... */
    sprintf(buf, "f%d", n);
    return strdup(buf);
}

static sentence
stub_var_decl(parameter p, int n, bool is_fortran)
{
    sentence result;
    string name = nth_formal_name(n);
    type t = parameter_type(p);
    pips_assert("is a variable", type_variable_p(t));

    if(basic_overloaded_p(variable_basic(type_variable(t))))
    {
	string comment = strdup(concatenate(
        "!     Unable to determine the type of parameter number ", name, "\n",
	"!     ", basic_to_string(variable_basic(type_variable(t))),
	" ", name, "\n", NULL));
	free(name);
	result = make_sentence(is_sentence_formatted, comment);
    }
    else
    {
	result = make_sentence(is_sentence_unformatted,
	  make_unformatted(string_undefined, 0, 0,
	     gen_make_list(string_domain,
		 strdup(basic_to_string(variable_basic(type_variable(t)))),
		   strdup(" "), name, strdup(is_fortran? "" : ";"), NULL)));
    }
    return result;
}

static sentence
stub_head(entity f, bool is_fortran)
{
  list ls = NIL;
  type t = entity_type(f);
  functional fu;
  int number, n;

  pips_assert("type is functional", type_functional_p(t));

  fu = type_functional(t);
  n = gen_length(functional_parameters(fu));
    
  /* is it a subroutine or a function? */
  if(!type_void_p(functional_result(fu)))
    {
      type tf = functional_result(fu);
      pips_assert("result is a variable", type_variable_p(tf));
      ls = CONS(STRING, strdup(is_fortran? " FUNCTION " : " "), 
		CONS(STRING,
		     strdup(basic_to_string(variable_basic(type_variable(tf)))),
		     NIL));
    }
  else 
    ls = CONS(STRING, strdup(is_fortran? "SUBROUTINE ":"void "), NIL);

  ls = CONS(STRING, strdup(module_local_name(f)), ls);
    
  if(is_fortran) {
    /* generate the formal parameter list. */
    for(number=1; number<=n; number++) 
      ls = CONS(STRING, nth_formal_name(number),
		CONS(STRING, strdup(number==1? "(": ", "), ls));

    /* close if necessary. */
    if (number>1) ls = CONS(STRING, strdup(")"), ls);
  }
  else {
    // Assume C and generate the formal parameter list with their types
    for(number=1; number<=n; number++) {
      ls = CONS(STRING, nth_formal_name(number),
		CONS(STRING, strdup(number==1? "(": ", "), ls));
    }
    /* close */
    if (number>1) ls = CONS(STRING, strdup(")"), ls);
    else  ls = CONS(STRING, strdup("()"), ls);
  }

  return make_sentence(is_sentence_unformatted, 
		       make_unformatted(string_undefined, 0, 0, gen_nreverse(ls)));
}

/* generates the text for a missing module.
 */
static text 
stub_text(entity module, bool is_fortran)
{
    sentence warning, head;
    type t = entity_type(module);
    int n=1;
    list /* of sentence */ ls = NIL;

    if (type_undefined_p(t))
	pips_user_error("undefined type for %s\n", entity_name(module));

    if (!type_functional_p(t))
	pips_user_error("non functional type for %s\n", entity_name(module));

    warning = make_sentence(is_sentence_formatted, strdup(is_fortran? FILE_WARNING:C_FILE_WARNING));
    head = stub_head(module, is_fortran);
    
    MAP(PARAMETER, p, 
	ls = CONS(SENTENCE, stub_var_decl(p, n++, is_fortran), ls),
	functional_parameters(type_functional(t)));

    ls = CONS(SENTENCE, make_sentence(is_sentence_unformatted,
	     make_unformatted(string_undefined, 0, 0, 
			      CONS(STRING, strdup(is_fortran?"END":"{}"), NIL))), ls);
    
    ls = CONS(SENTENCE, warning, CONS(SENTENCE, head, gen_nreverse(ls)));

    return make_text(ls);
}

/* generates a source file for some module, if none available.
 */
static bool 
missing_file_initializer(string module_name, bool is_fortran)
{
    boolean success_p = TRUE;
    entity m = local_name_to_top_level_entity(module_name);
    string file_name, dir_name, src_name, full_name, init_name, finit_name; 
    /* relative to the current directory */
    FILE * f;
    text stub;
    string res = is_fortran? DBR_INITIAL_FILE : DBR_C_SOURCE_FILE;
    /* For C only: compilation unit cu and compilation unit name cun */
    string cun = string_undefined;
    entity cu = entity_undefined;
 
    pips_user_warning("no source file for %s: synthetic code is generated\n",
		      module_name);

    if(entity_undefined_p(m))
    {
      pips_user_error(
	"No entity defined for module %s although it might"
	" have been encountered at a call site\n", module_name);
      return FALSE;
    }

    // Build the coresponding compilation unit for C code
    if(!is_fortran) {
      // Function defined in pipsmake
      extern string compilation_unit_of_module(string);
      cun = compilation_unit_of_module(module_name);

      if(string_undefined_p(cun)) {
	cun = strdup(concatenate(module_name, FILE_SEP_STRING, NULL));
	cu = MakeCompilationUnitEntity(cun);
      }
    }
    
    /* pips' current directory is just above the workspace
     */
    file_name = strdup(concatenate(module_name, is_fortran? ".f" : ".cpp_processed.c", NULL));
    file_name = strlower(file_name, file_name);
    dir_name = db_get_current_workspace_directory();
    src_name = strdup(concatenate(WORKSPACE_TMP_SPACE, "/", file_name, NULL));
    full_name = strdup(concatenate(dir_name, "/", src_name, NULL));
    init_name = 
      db_build_file_resource_name(res, module_name, is_fortran? ".f_initial" : ".c");
    finit_name = strdup(concatenate(dir_name, "/", init_name, NULL));

    /* builds the stub.
     */
    stub = stub_text(m, is_fortran);

    /* put it in the source file and link the initial file.
     */
    db_make_subdirectory(WORKSPACE_TMP_SPACE);
    f = safe_fopen(full_name, "w");
    print_text(f, stub);
    safe_fclose(f, full_name);
    /* A PIPS database may be partly incoherent after a core dump but
       still usable (Cathare 2, FI)*/
    if(file_exists_p(finit_name))
      safe_unlink(finit_name);
    safe_link(finit_name, full_name);

    /* Add the new file as a file resource...
     * should only put a new user file, I guess?
     */
    user_log("Registering synthesized file %s\n", file_name);
    DB_PUT_FILE_RESOURCE(res, module_name, init_name);
    DB_PUT_FILE_RESOURCE(DBR_USER_FILE, module_name, src_name);

    if(!is_fortran) { // is C assumed
      /* Add the compilation unit files */
      pips_assert("The compilation unit name is defined", !string_undefined_p(cun));
      user_log("Registering synthesized compilation unit %s\n", file_name);

      /*
	file_name = strdup(concatenate(module_name, is_fortran? ".f" : ".cpp_processed.c", NULL));
	file_name = strlower(file_name, file_name);
	dir_name = db_get_current_workspace_directory();
	src_name = strdup(concatenate(WORKSPACE_TMP_SPACE, "/", file_name, NULL));
	full_name = strdup(concatenate(dir_name, "/", src_name, NULL));
      */

    init_name = 
      db_build_file_resource_name(res, cun, is_fortran? ".f_initial" : ".c");
    finit_name = strdup(concatenate(dir_name, "/", init_name, NULL));

    /* builds the compilation unit stub: it can be empty or include
       module_name declaration as an extern function.
     */
    stub = c_text_entity(cu, m, 0);

    /* put it in the source file and link the initial file.
     */
    db_make_subdirectory(WORKSPACE_TMP_SPACE);
    f = safe_fopen(finit_name, "w");
    print_text(f, stub);
    safe_fclose(f, finit_name);
      DB_PUT_FILE_RESOURCE(res, cun, init_name);
      DB_PUT_FILE_RESOURCE(DBR_USER_FILE, cun, strdup(src_name));
  }

    free(file_name), free(dir_name), free(full_name), free(finit_name);
    return success_p;
}

extern bool process_user_file(string); /* ??? in top-level */

static bool
ask_a_missing_file(string module, bool is_fortran)
{
    string file;
    bool ok, cont;
    string res= is_fortran? DBR_INITIAL_FILE : DBR_C_SOURCE_FILE;
    
    do {
	file = user_request("Please enter a file for module %s\n", module);
	if(file && strcmp(file, "quit")==0)
	  break;
	if (file)
	  {
	    if (same_string_p(file, "generate"))
		ok = missing_file_initializer(module, string_fortran_filename_p(module));
	    else
		ok = process_user_file(file);
	  }
	cont = file && !same_string_p(file, "quit") &&
	    !db_resource_p(res, module);
	if(cont)
	  pips_user_warning("Module \"%s\" not found in \"%s\".\n"
			    "Please type \"quit\" or another file name.\n", module, file);
	if (file) free(file);
    } while (cont);
    return db_resource_p(res, module);
}

/* there is no real rule to produce source or user files; it was introduced
 * by Remi to deal with source and user files as with any other kind
 * of resources
 */
bool generic_initializer(string module_name, bool is_fortran)
{
    bool success_p = FALSE;
    string missing = get_string_property("PREPROCESSOR_MISSING_FILE_HANDLING");

    if (same_string_p(missing, "error"))
	pips_user_error("no source file for %s (might be an ENTRY point)\n"
			"set PREPROCESSOR_MISSING_FILE_HANDLING"
			" to \"query\" or \"generate\"...\n", module_name);
    else if (same_string_p(missing, "generate")) 
	success_p = missing_file_initializer(module_name, is_fortran);
    else if (same_string_p(missing, "query"))
	success_p = ask_a_missing_file(module_name, is_fortran);
    else 
	pips_user_error("invalid value of property "
			" PREPROCESSOR_MISSING_FILE_HANDLING = \"%s\"",
			missing);

    return success_p;
}

bool fortran_initializer(string module_name)
{
  return generic_initializer(module_name, TRUE);
}

bool initializer(string module_name)
{
  return generic_initializer(module_name, TRUE);
}
bool c_initializer(string module_name)
{
  return generic_initializer(module_name, FALSE);
}
