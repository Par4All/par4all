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
/*
 * Load, Save, Free a resource, with available methods.
 * DBLL = Data Base Low Level
 */

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "private.h"

/**************************************************** METHODS FOR RESOURCES */

/* the current module is expected by some load/save functions...
 */
const char* dbll_current_module = (const char*) NULL;

typedef char * (* READER)(FILE *);
typedef void  (* WRITER)(FILE *, void *);
typedef void  (* FREER) (void *);
typedef bool  (* CHECKER)(void *);

typedef struct {
    string  name;
    READER  read_function;
    WRITER  write_function;
    FREER   free_function;
    CHECKER check_function;
} methods;

// default do-nothing methods.
#define no_read (READER) abort
#define no_write (WRITER) abort
static void no_free(void * p) { pips_debug(2, "memory leak (%p)\n", p); }
static void writeln_string(FILE * f, void * p) {fprintf(f, "%s\n", (char*)p);}
static void unexpected(void)
{ pips_internal_error("unexpected pipsdbm method");}

// from external.c
extern void pipsdbm_write_statement_mapping(FILE *, statement_mapping);
extern hash_table pipsdbm_read_statement_mapping(FILE *);
extern bool pipsdbm_check_statement_mapping(statement_mapping);
extern void pipsdbm_free_statement_mapping(statement_mapping);
extern void free_static_control_mapping(statement_mapping);

// all methods are stored in this separate file. as an array.
static methods all_methods[] = {
#include "methods.h"
};

/* return the methods for resource name
 */
static methods * get_methods(const char* name)
{
    /* we use a local cache for fast retrieval.
     */
    static hash_table cache = hash_table_undefined;
    methods * m;

    if (hash_table_undefined_p(cache)) { /* initialize at first call. */
	cache = hash_table_make(hash_string, 2*dbll_number_of_resources());
	for (m = all_methods; m->name; m++)
	    hash_put(cache, m->name, (void *) m);
    }

    /* get the methods! */
    m = (methods*) hash_get(cache, name);
    if (m==(methods*)HASH_UNDEFINED_VALUE)
	m = &all_methods[dbll_number_of_resources()]; /* last is unexpected */
    return m;
}

int dbll_number_of_resources(void)
{   /* I'm not sure sizeof(all_methods) is ANSI C. FC */
    return sizeof(all_methods)/sizeof(methods) - 1;
}

string dbll_get_ith_resource_name(int i)
{
    pips_assert("valid resource number", i>=0 && i<dbll_number_of_resources());
    return all_methods[i].name;
}

bool dbll_very_special_resource_p(const char* rname, const char* oname)
{
  return same_string_p(rname, DBR_ENTITIES) && same_string_p(oname, "");
}

/********************************************************************** FILE */

static char* current_builder = NULL;

void db_set_current_builder_name(const char* name)
{
    pips_assert("no current builder", !current_builder);
    current_builder = strdup(name);
}

void db_reset_current_builder_name(void)
{
    pips_assert("some current builder", current_builder);
    free(current_builder), current_builder = NULL;
}

string db_get_current_builder_name(void)
{
    pips_assert("some current builder", current_builder);
    return current_builder;
}

#define DEFAULT_OWNER_NAME WORKSPACE_PROGRAM_SPACE

/* returns the allocated and mkdir'ed directory for module name
 */
string db_get_directory_name_for_module(const char* name)
{
    string dir_name, ws_dir_name;
    pips_assert("some valid name", name && !same_string_p(name, ""));
    ws_dir_name = db_get_current_workspace_directory();
    dir_name = strdup(concatenate(ws_dir_name, "/", name, NULL));
    free(ws_dir_name);
    if (!directory_exists_p(dir_name))
	if (!create_directory(dir_name)) /* MKDIR */
	    pips_user_irrecoverable_error
		("cannot create directory %s\n", dir_name);
    return dir_name;
}

/* returns an allocated file name for a file resource.
 * may depend on the current builder, someday.
 * this function is to be used by all phases that generate files.
 * it does not include the directory for movability
 */
string db_build_file_resource_name(const char* rname, const char* oname, const char* suffix)
{
  string result;
  if (same_string_p(oname, "")) oname = DEFAULT_OWNER_NAME;
  free(db_get_directory_name_for_module(oname));/* mkdir as a side effect. */
  /* the next name must be compatible with the Display script...
   * it may depend on the builder function maybe (if pipsmake tells)
   * may include the resource name? as lower letters?
   */
  result = strdup(concatenate(oname, "/", oname, suffix, NULL));

  pips_debug(8, "file name for %s[%s] with suffix '%s' is '%s'\n",
	     rname, oname, suffix, result);
  return result;
}

/* allocate a full file name for the given resource.
 */
string get_resource_file_name(const char* rname, const char* oname)
{
    string dir_name, file_name;
    if (same_string_p(oname, "")) oname = DEFAULT_OWNER_NAME;
    dir_name = db_get_directory_name_for_module(oname);
    file_name = strdup(concatenate(dir_name, "/", rname, NULL));
    free(dir_name);
    return file_name;
}

static FILE * open_resource_file(const char* rname, const char* oname, const char* what)
{
    FILE * file;
    string file_name = get_resource_file_name(rname, oname);
    file = safe_fopen(file_name, what);
    free(file_name);
    return file;
}

static void close_resource_file(FILE * file, const char* rname, const char* oname)
{
    string file_name = get_resource_file_name(rname, oname);
    safe_fclose(file, file_name);
    free(file_name);
}


/*********************************************************** BASIC INTERFACE */

#include <sys/stat.h>
#include <unistd.h>

void dbll_unlink_resource_file(const char* rname, const char* oname, bool erroriffailed)
{
    string full_name = get_resource_file_name(rname, oname);
    if (unlink(full_name) && erroriffailed) {
	perror(full_name);
	pips_internal_error("cannot unlink resource %s of %s", rname, oname);
    }
    free(full_name);
}

/* returns 0 on errors (say no file).
 * otherwise returns the modification time.
 */
static int dbll_stat_file(const char* file_name, bool okifnotthere)
{
    struct stat buf;
    int time = 0, error = stat(file_name, &buf);
    if (error<0) { /* some error */
	if (!okifnotthere || get_bool_property("WARNING_ON_STAT_ERROR")) {
	    perror(file_name);
	    pips_user_warning("error in stat for %s\n", file_name);
	}
	if (!okifnotthere) {
	    pips_internal_error("stat error not permitted here");
	}
    } else time = (int) buf.st_mtime; /* humm... unsigned... */
    return time;
}

/* It is impportant that the workspace directory does not appear in the
 * file name so as to allow workspaces to be moveable.
 */
int dbll_stat_local_file(const char* file_name, bool okifnotthere)
{
    string full_name;
    int time;
    if (file_name[0]!='/' && file_name[0]!='.') {
	string dir_name = db_get_current_workspace_directory();
	full_name = strdup(concatenate(dir_name, "/", file_name, NULL));
	free(dir_name);
    } else full_name = strdup(file_name);
    time = dbll_stat_file(full_name, okifnotthere);
    free(full_name);
    return time;
}

int dbll_stat_resource_file(const char* rname, const char* oname, bool okifnotthere)
{
    string file_name = get_resource_file_name(rname, oname);
    int time = dbll_stat_file(file_name, okifnotthere);
    free(file_name);
    return time;
}

/* save rname of oname p. get the method, then apply it.
 */
void dbll_save_resource(const char* rname, const char* oname, void * p)
{
    methods * m;
    FILE * f;
    pips_debug(7, "saving resource %s[%s] (0x%p)\n", rname, oname, p);

    dbll_current_module = oname;
    m = get_methods(rname);
    if (m->write_function==no_write) {
	pips_user_warning("resource %s of %s lost, no unload function\n",
			  rname, oname);
    } else {
	f = open_resource_file(rname, oname, "w");
	m->write_function(f, p);
	close_resource_file(f, rname, oname);
    }
    dbll_current_module = (string) NULL;
}

void * dbll_load_resource(const char* rname, const char* oname)
{
    methods * m;
    FILE * f;
    void * p = NULL;
    pips_debug(7, "loading resource %s[%s]\n", rname, oname);

    dbll_current_module = oname;
    m = get_methods(rname);
    if (m->read_function==no_read)
	pips_internal_error("cannot load %s of %s, no load function",
			    rname, oname);
    f = open_resource_file(rname, oname, "r");
    p = m->read_function(f);
    close_resource_file(f, rname, oname);
    dbll_current_module = (string) NULL;
    return p;
}

void dbll_free_resource(const char* rname, const char* oname, void * p)
{
    methods * m;
    pips_debug(7, "freeing resource %s[%s] (%p)\n", rname, oname, p);
    m = get_methods(rname);
    m->free_function(p);
}

/* Internal consistency of the resource, not the global consistency
   wrt other resources and pipsmake rules which is managed by the
   pipsmake library at a higher level. */
bool dbll_check_resource(const char* rname, const char* oname, void * p)
{
    methods * m;
    /* Know right away which resource is Newgen inconsistent */
    pips_debug(9, "checking Newgen resource consistency %s[%s] (0x%p)\n",
	       rname, oname, p);
    m = get_methods(rname);
    bool b = m->check_function(p);
    /* FI: I find this message I have added myself pretty useless in
       general as a core dump occurs in case some inconsistency is
       found. */
    pips_debug(9, "checking Newgen resource consistency %s[%s] (0x%p): %s\n",
	       rname, oname, p,
	       bool_to_string(b));
    return b;
}

bool dbll_storable_p(const char* rname)
{
    methods * m = get_methods(rname);
    return m->write_function!=no_write;
}


/****************************************************** LESS BASIC INTERFACE */

void dbll_save_and_free_resource(const char* rname, const char* oname,
				 void * p, bool do_free)
{
    dbll_save_resource(rname, oname, p);
    if (do_free) dbll_free_resource(rname, oname, p);
}

/* rather approximated. */
bool displayable_file_p(const char* name)
{
    methods * m = get_methods(name);
    return m->write_function==writeln_string && strstr(name, "_FILE");
}

/* returns whether the file is managed within the database.
 * that is it is not a SOURCE_FILE.
 * basically SOURCE_FILEs are given relative names with leading . or /,
 * while within the database a leading "*.database/" is always appended.
 */
bool dbll_database_managed_file_p(const char* name)
{
    return name[0]!='.' && name[0]!='/';
}
