/*

  $Id$

  Copyright 1989-2010 MINES ParisTech
  Copyright 2009-2010 TÉLÉCOM Bretagne
  Copyright 2009-2010 HPC Project

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

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

#include "linear.h"
#include "genC.h"

#include "ri.h"
#include "database.h"

#include "misc.h"

#include "ri-util.h" /* ri needed for statement_mapping in pipsdbm... */
#include "pipsdbm.h"
#include "resources.h"
#include "phases.h"
#include "properties.h"
#include "pipsmake.h"
#include "text-util.h" // for words_to_string

#include "top-level.h"

static FILE * logstream = NULL;
static list log_list = list_undefined;

static void pyps_log_handler(const char *fmt, va_list args)
{
	FILE * log_file = get_log_file();

	/* It goes to stderr to have only displayed files on stdout.
	 */

	/* To be C99 compliant, a va_list can be used only once...
		 Also to avoid exploding on x86_64: */
	va_list args_copy,args_copy2;
	va_copy (args_copy, args);
	va_copy (args_copy2, args);

	vfprintf(logstream, fmt, args_copy);
	fflush(logstream);
	if (!list_undefined_p(log_list))
	{
		char* tmp;
		vasprintf(&tmp, fmt, args_copy2);
		log_list = CONS(STRING,tmp,log_list);
	}

	if (!log_file || !get_bool_property("USER_LOG_P"))
		return;

	if (vfprintf(log_file, fmt, args) <= 0) {
		perror("user_log");
		abort();
	}
	else fflush(log_file);
}
char * pyps_last_error = NULL;
static void pyps_error_handler(const char * calling_function_name,
                   const char * a_message_format,
                   va_list *some_arguments)
{
	// Save pre-existing error message
	char *old_error = pyps_last_error;

    char * tmp;
    vasprintf(&tmp,a_message_format,*some_arguments);
    asprintf(&pyps_last_error,"in %s: %s",calling_function_name,tmp);
    free(tmp);

    // If we already had a message before, we stack it over the new one
    if(old_error) {
    	char *tmp = pyps_last_error;
        asprintf(&pyps_last_error,"%s%s",old_error,tmp);
    	free(old_error);
    	free(tmp);
    }

   /* terminate PIPS request */
   /* here is an issue: what if the user error was raised from properties */
   if (get_bool_property("ABORT_ON_USER_ERROR")) 
       abort();
   else {
      static int user_error_called = 0;

      if (user_error_called > get_int_property("MAXIMUM_USER_ERROR")) {
         (void) fprintf(stderr, "This user_error is too much! Exiting.\n");
         exit(1);
      }
      else {
         user_error_called++;
      }

      /* throw according to linear exception stack! */
      THROW(user_exception_error);
   }
}

void open_log_buffer()
{
	if (!list_undefined_p(log_list))
		close_log_buffer();
	log_list = NIL;

}

char* get_log_buffer()
{
	assert(!list_undefined_p(log_list));
	list log_list_tmp=gen_copy_seq(log_list);
	log_list_tmp = gen_nreverse(log_list_tmp);
	char* ret = words_to_string(log_list_tmp);
	gen_free_list(log_list_tmp);
	return ret;
}

void close_log_buffer()
{
	if (!list_undefined_p(log_list))
	{
		gen_free_string_list(log_list);
		log_list = list_undefined;
	}
}

DEFINE_LOCAL_STACK(properties,property);
void atinit()
{
    /* init various composants */
    initialize_newgen();
    initialize_sc((char*(*)(Variable))entity_local_name);
    pips_log_handler = pyps_log_handler;
    pips_error_handler = pyps_error_handler;
    set_exception_callbacks(push_pips_context, pop_pips_context);
    make_properties_stack();
}

void verbose(bool on) {
    if(on) logstream=stderr;
    else logstream=fopen("/dev/null","w");
}


void create(char* workspace_name, char ** filenames)
{
    if (workspace_exists_p(workspace_name))
        pips_user_error
            ("Workspace %s already exists. Delete it!\n", workspace_name);
    else if (db_get_current_workspace_name()) {
        pips_user_error("Close current workspace %s before "
                "creating another one!\n",
                db_get_current_workspace_name());
    }
    else
    {
        if (db_create_workspace(workspace_name))
        {
            /* create the array of arguments */
            gen_array_t filename_list = gen_array_make(0);
            while(*filenames)
            {
                gen_array_append(filename_list,*filenames);
                filenames++;
            }

            bool success = create_workspace(filename_list);

            gen_array_free(filename_list);

            if (!success)
            {
                db_close_workspace(false);
                pips_user_error("Could not create workspace %s\n",
				workspace_name);
            }
        }
        else {
            pips_user_error("Cannot create directory for workspace, "
			    "check rights!\n");
        }
    }
}

void set_property(const char* propname, const char* value)
{
    // thank's to rk, this hack is no longer needed
#if 0
    /* nice hack to temporarly redirect stderr */
    int saved_stderr = dup(STDERR_FILENO);
    char *buf;
    freopen("/dev/null","w",stderr);
    asprintf(&buf, "/dev/fd/%d", saved_stderr);
#endif
    if (!safe_set_property(propname, value)) {
#if 0
        freopen(buf,"w",stderr);
        free(buf);
#endif
        pips_user_error("error in setting property %s to %s\n",
			propname, value);
    }
#if 0
    else {
        freopen(buf,"w",stderr);
        free(buf);
    }
#endif
}

char* info(char * about)
{
    string sinfo = NULL;
    if (same_string_p(about, "workspace"))
    {
        sinfo = db_get_current_workspace_name();
        if(sinfo) sinfo=strdup(sinfo);
    }
    else if (same_string_p(about, "module"))
    {
        sinfo = db_get_current_module_name();
        if(sinfo) sinfo=strdup(sinfo);
    }
    else if (same_string_p(about, "modules") && db_get_current_workspace_name())
    {
        gen_array_t modules = db_get_module_list();
        int n = gen_array_nitems(modules), i;

        size_t sinfo_size=0;
        for(i=0; i<n; i++)
        {
            string m = gen_array_item(modules, i);
            sinfo_size+=strlen(m)+1;
        }
        sinfo=strdup(string_array_join(modules, " "));
        if(!sinfo) fprintf(stderr,"not enough memory to hold all module names\n");
        gen_array_full_free(modules);
    }
    else if (same_string_p(about, "directory"))
    {
        char pathname[MAXPATHLEN];
        sinfo=getcwd(pathname, MAXPATHLEN);
        if(sinfo)
            sinfo=strdup(sinfo);
        else
            fprintf(stderr,"failer to retreive current working directory\n");
    }

    if(!sinfo)
        sinfo=strdup("");
    return sinfo;
}

void apply(char * phasename, char * target)
{
    if (!safe_apply(phasename,target)) {
        if(!pyps_last_error) asprintf(&pyps_last_error,"phase %s on module %s failed" , phasename, target);
        THROW(user_exception_error);
    }
}

void capply(char * phasename, char ** targets)
{
    /* create the array of arguments */
    gen_array_t target_list = gen_array_make(0);
    while(*targets)
    {
        gen_array_append(target_list,*targets);
        targets++;
    }
    bool ok = safe_concurrent_apply(phasename,target_list);
    gen_array_free(target_list);
    if(!ok) {
      if(!pyps_last_error) asprintf(&pyps_last_error,"capply phase %s failed without setting error message" , phasename);
      THROW(user_exception_error);
    }
}

void display(char *rname, char *mname)
{
    string old_current_module_name = db_get_current_module_name();
    if(old_current_module_name) {
      old_current_module_name = strdup(old_current_module_name);
      db_reset_current_module_name();
    }

    db_set_current_module_name(mname);
    string fname = build_view_file(rname);
    db_reset_current_module_name();
    if(old_current_module_name) {
      db_set_current_module_name(old_current_module_name);
      free(old_current_module_name);
    }

    if (!fname)
    {
        pips_user_error("Cannot build view file %s\n", rname);
        return;
    }

    safe_display(fname);
    free(fname);
    return;
}

char* show(char * rname, char *mname)
{
    if (!db_resource_p(rname, mname)) {
        pips_user_warning("no resource %s[%s].\n", rname, mname);
        return  strdup("");
    }

    if (!displayable_file_p(rname)) {
        pips_user_warning("resource %s cannot be displayed.\n", rname);
        return strdup("");
    }

    /* now returns the name of the file.
    */
    return strdup(db_get_memory_resource(rname, mname, true));
}

/* Returns the list of the modules that call that specific module,
   separated by ' '. */
char * get_callers_of(char * module_name)
{
    safe_make("CALLERS",module_name);
    gen_array_t callers = get_callers(module_name);

    char * callers_string = strdup(string_array_join(callers, " "));

    gen_array_free(callers);

    return callers_string;
}

/* Returns the list of the modules called by that specific module,
   separated by ' '. */
char * get_callees_of(char * module_name)
{
    safe_make("CALLERS",module_name);
    gen_array_t callees = get_callees(module_name);

    char * callees_string = strdup(string_array_join(callees, " "));

    gen_array_free(callees);

    return callees_string;
}

/* Returns the list of the modules called by that specific module,
   separated by ' '. */
char * pyps_get_stubs()
{
    gen_array_t stubs = get_stubs();

    char * stubs_string = strdup(string_array_join(stubs, " "));

    gen_array_free(stubs);

    return stubs_string;
}

void setenviron(char *name, char *value)
{
    setenv(name, value, 1);
}

char* getenviron(char *name)
{
    return getenv(name);
}


void push_property(const char* name, const char * value) {
    property p = copy_property(get_property(name,false));
    set_property(strdup(name),value);
    properties_push(p);
}

void pop_property(const char* name) {
    property p = properties_pop();
    if(property_bool_p(p))
        set_bool_property(name,property_bool(p));
    else if(property_string_p(p))
        set_string_property(name,property_string(p));
    else
        set_int_property(name,property_int(p));
}



/* Add a source file to the workspace
 * We wrap process_user_file() here with a hack
 * to define the workspace language so that some
 * pipsmake activate specific to the language
 * will be defined.
 * This function will be removed when pipsmake
 * will be improved to handle multilanguage workspace !
 */
bool add_a_file( string file ) {
  gen_array_t filename_list = gen_array_make(0);
  gen_array_append(filename_list,file);
  language workspace_language(gen_array_t files);
  language l = workspace_language(filename_list);
  activate_language(l);
  free_language(l);
  gen_array_free(filename_list);
  bool process_user_file(string file);
  if(process_user_file(file)==false) {
    pips_user_error("Error adding new file.");
    return false;
  }
  return true;
}


/*
 * Retrieve the language (string form) for a module
 */
char *get_module_language( string mod_name ) {
  language l = module_language(module_name_to_entity(mod_name));

  switch(language_tag(l)) {
    case is_language_fortran: return "fortran";
    case is_language_fortran95: return "fortran95";
    case is_language_c: return "c";
    default: return "unknown";
  }
}


// Broker registering python callback to C

#include <Python.h>


// Store the python object that will provide stub for missing module on the fly
static PyObject *stub_broker = NULL;

/*
 * This is the callback interface for PIPS missing module to python
 * It'll be called by pips to retrieve a file name when a module wasn't found
 * by PIPS.
 */
static string get_stub_from_broker( string str )
{
   string result = "";

   // Assert that a broker is defined !
   if(stub_broker) {

      // Get the stub file
      PyObject *pyStubFile =  PyEval_CallMethod(stub_broker, "stub_file_for_module", "(s)", str);

      // Sanity check
      if(pyStubFile) {
        // Convert to a regular C string
        result = PyString_AsString(pyStubFile);
        Py_XDECREF(pyStubFile);
      } else {
        fprintf(stderr,"Callback failed !\n");
        PyErr_Print();
      }
   }

   return result;
}


/*
 *  Register the workspace (or any other object) as a resolver for missing
 *  modules. Method "stub_file_for_module" will be called and have to be defined
 */
void set_python_missing_module_resolver_handler(PyObject *PyObj)
{
    Py_XDECREF(stub_broker);          /* Dispose of previous callback */
    stub_broker = PyObj;         /* Remember new callback */
    Py_XINCREF(stub_broker);          /* Record of new callback */

    void set_internal_missing_module_resolver_handler(string (*)(string));
set_internal_missing_module_resolver_handler(get_stub_from_broker);
}

/* Read execution mode for a loop */
bool get_loop_execution_parallel(const char* module_name,
                  const char* loop_label) {

  /* prelude */
  set_current_module_entity(module_name_to_entity( module_name ));
  set_current_module_statement
    ((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

  entity label = find_label_entity(module_name,loop_label);
  if(entity_undefined_p(label))
    pips_user_error("label '%s' does not exist\n",loop_label);
  statement stmt = find_loop_from_label(get_current_module_statement(),label);
  if(statement_undefined_p(stmt))
    pips_user_error("label '%s' is not on a loop\n",loop_label);
  bool is_exec_parallel_p = execution_parallel_p(loop_execution(statement_loop(stmt)));

  /* reset current state */
  reset_current_module_entity();
  reset_current_module_statement();
  return is_exec_parallel_p;
}

/* Change execution mode for a loop */
void set_loop_execution_parallel(const char* module_name,
                  const char* loop_label,
                  bool exec_parallel_p) {

  /* prelude */
  set_current_module_entity(module_name_to_entity( module_name ));
  set_current_module_statement
    ((statement) db_get_memory_resource(DBR_CODE, module_name, true) );

  entity label = find_label_entity(module_name,loop_label);
  if(entity_undefined_p(label))
    pips_user_error("label '%s' does not exist\n",loop_label);
  statement stmt = find_loop_from_label(get_current_module_statement(),label);
  if(statement_undefined_p(stmt))
    pips_user_error("label '%s' is not on a loop\n",loop_label);
  execution_tag(loop_execution(statement_loop(stmt)))
      = (exec_parallel_p ? is_execution_parallel : is_execution_sequential);

  /* Store the new code */
  DB_PUT_MEMORY_RESOURCE(DBR_CODE, module_name, get_current_module_statement());

  /* reset current state */
  reset_current_module_entity();
  reset_current_module_statement();
}

