/*
 * $Id$
 *
 * Load, Save, Free a resource, with available methods.
 * DBLL = Data Base Low Level
 */

#include "private.h"

/**************************************************** METHODS FOR RESOURCES */

typedef char * (* READER)(FILE *);
typedef void  (* WRITER)(FILE *, char *);
typedef void  (* FREER) (char *);
typedef bool  (* CHECKER)(char *);

typedef struct {
    string  name;
    READER  read_function;
    WRITER  write_function;
    FREER   free_function;
    CHECKER check_function;
} methods;

/* default do-nothing methods.
 */
#define no_read (READER) abort
#define no_write (WRITER) abort
static void no_free(char * p) { pips_debug(2, "memory leak\n"); }
static void writeln_string(FILE * f, char * p) { fprintf(f, "%s\n", p); }
static void unexpected(void)
{ pips_internal_error("unexpected pipsdbm method\n");}

/* all methods are stored in this separate file.
 */
static methods all_methods[] = {
#include "methods.h"
};

/* return the methods for resource name
 */
static methods *
get_methods(string name)
{
    methods * m = all_methods;
    while (m->name && !same_string_p(name, m->name)) m++;
    return m;
}

int 
dbll_number_of_resources(void)
{   /* I'm not sure sizeof(all_methods) is ANSI C. FC */
    return sizeof(all_methods)/sizeof(methods) - 1;
}

string
dbll_get_ith_resource_name(int i)
{
    pips_assert("valid resource number", i>=0 && i<dbll_number_of_resources());
    return all_methods[i].name;
}

/********************************************************************** FILE */

static string current_builder = NULL;

void
db_set_current_builder_name(string name)
{
    pips_assert("no current builder", !current_builder);
    current_builder = strdup(name);
}

void 
db_reset_current_builder_name(void)
{
    pips_assert("some current builder", current_builder);
    free(current_builder), current_builder = NULL;
}

string
db_get_current_builder_name(void)
{
    pips_assert("some current builder", current_builder);
    return current_builder;
}

#define DEFAULT_OWNER_NAME "Program"

/* returns the allocated and mkdir'ed directory for module name
 */
string 
db_get_directory_name_for_module(string name)
{
    string dir_name;
    pips_assert("some valid name", name && !same_string_p(name, ""));
    dir_name = strdup(concatenate
		      (db_get_current_workspace_directory(), "/", name, 0));
    if (!directory_exists_p(dir_name))
	if (!create_directory(dir_name)) /* MKDIR */
	    pips_internal_error("cannot create directory %d\n", dir_name);
    return dir_name;
}    

/* returns an allocated file name for a file resource.
 * may depend on the current builder, someday.
 * this function is to be used by all phases that generate files.
 */
string
db_build_file_resource_name(string rname, string oname, string suffix)
{
    string dir_name, file_name;
    if (same_string_p(oname, "")) oname = DEFAULT_OWNER_NAME;
    dir_name = db_get_directory_name_for_module(oname);
    file_name = strdup(concatenate(dir_name, "/", oname, ".", suffix, 0));
    free(dir_name);
    return file_name;
}

/* allocate a full file name for the given resource.
 * could be switched to a directory maybe. 
 */
static string
get_resource_file_name(string rname, string oname)
{
    return db_build_file_resource_name(rname, oname, rname);
}

static FILE *
open_resource_file(string rname, string oname, string what)
{
    FILE * file;
    string file_name = get_resource_file_name(rname, oname);
    file = safe_fopen(file_name, what);
    free(file_name);
    return file;
}

static void
close_resource_file(FILE * file, string rname, string oname)
{
    string file_name = get_resource_file_name(rname, oname);
    safe_fclose(file, file_name);
    free(file_name);
}


/*********************************************************** BASIC INTERFACE */

#include <sys/stat.h>
#include <unistd.h>

/* returns 0 on errors (say no file).
 * otherwise returns the modification time.
 */
int 
dbll_stat_resource_file(string rname, string oname, bool okifnotthere)
{
    string full_name = get_resource_file_name(rname, oname);
    struct stat buf;
    int time = 0, code = stat(full_name, &buf);

    if (code<0) /* some error */
	if (!okifnotthere || get_bool_property("WARNING_ON_STAT_ERROR")) {
	    perror(full_name);
	    pips_internal_error("error in stat for %s\n", full_name);
	}	    
    else time = (int) buf.st_mtime; /* humm... unsigned... */

    free(full_name);
    return time;
}

/* save rname of oname p. get the method, then apply it.
 */
void
dbll_save_resource(string rname, string oname, char * p)
{
    methods * m;
    FILE * f;
    pips_debug(7, "saving resource %s of %s\n", rname, oname);

    m = get_methods(rname);
    if (m->write_function==no_write) {
	pips_user_warning("resource %s of %s lost, no unload function\n", 
			  rname, oname);
    } else {
	f = open_resource_file(rname, oname, "w");
	m->write_function(f, p);
	close_resource_file(f, rname, oname);
    }
}

char *
dbll_load_resource(string rname, string oname)
{
    methods * m;
    FILE * f;
    char * p = NULL;
    pips_debug(7, "loading resource %s of %s\n", rname, oname);

    m = get_methods(rname);
    if (m->read_function==no_read) 
	pips_internal_error("cannot load %s of %s, no load function\n",
			    rname, oname);
    f = open_resource_file(rname, oname, "r");
    p = m->read_function(f);
    close_resource_file(f, rname, oname);
    return p;
}

void
dbll_free_resource(string rname, string oname, char * p)
{
    methods * m; 
    pips_debug(7, "freeing resource %s of %s\n", rname, oname);
    m = get_methods(rname);
    m->free_function(p);
}

bool 
dbll_check_resource(string rname, string oname, char * p)
{
    methods * m;
    pips_debug(7, "checking resource %s of %s\n", rname, oname);
    m = get_methods(rname);
    return m->check_function(p);
}

bool 
dbll_storable_p(string rname)
{
    methods * m = get_methods(rname);
    return m->write_function!=no_write;
}


/****************************************************** LESS BASIC INTERFACE */

void
dbll_save_and_free_resource(string rname, string oname, char * p)
{
    dbll_save_resource(rname, oname, p);
    dbll_free_resource(rname, oname, p);
}

/* rather approximated. */
bool 
displayable_file_p(string name)
{
    methods * m = get_methods(name);
    return m->write_function==writeln_string;
}
