/*
 * $Id$
 *
 * Workspace management.
 *
 * Basically, a workspace is the object that manages a database.
 * The database will be stored into that workspace.
 * Just to say it is a directory, with a name.
 */

#include "private.h"
#include "pipsdbm_private.h"


/******************************************************************** UTILS */

/* the workspace name must be composed of the following characters.
 */
#define WORKSPACE_NAME_CHARS \
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"

bool workspace_name_p(string name)
{
    return strlen(name)==strspn(name, WORKSPACE_NAME_CHARS);
}


/************************************************************ THE WORKSPACE */

/* the workspace has a name.
 *
 * {set,reset,get}_workspace
 */
static string current_workspace_name = NULL;

static void db_set_current_workspace_name(string name)
{
    if (current_workspace_name)
	pips_internal_error("current workspace %s not closed\n", 
			    current_workspace_name);
    pips_assert("valid workspace name", workspace_name_p(name));
    current_workspace_name = strdup(name);
}

static void db_reset_current_workspace_name(void)
{
    pips_assert("some current workspace", current_workspace_name);
    free(current_workspace_name), current_workspace_name = NULL;
}

static void db_reset_current_workspace_name_if_necessary(void)
{
    if (current_workspace_name) db_reset_current_workspace_name();
}

/* the function is used to check that there is some current workspace...
 */
string db_get_current_workspace_name(void)
{
    return current_workspace_name;
}

/* returns an allocated string. */
string db_get_workspace_directory_name(string name)
{
    return strdup(concatenate("./", name, ".database", 0)); 
}

string db_get_current_workspace_directory(void)
{
    string ws_name = db_get_current_workspace_name();
    pips_assert("some current workspace", ws_name);
    return db_get_workspace_directory_name(ws_name);
}

/************************************************************* LOGICAL TIME */

static int logical_time = 1; /* 0 means not set... */

int db_inc_logical_time(void)
{
    return logical_time++; 
}

int db_get_logical_time(void) 
{
    return logical_time; 
}

static void db_set_logical_time(int time)
{
    pips_assert("positive time set", time>=1);
    logical_time = time;
}

static void db_reset_logical_time(void)
{
    logical_time = 0;
}

/***************************************************************** META DATA */

#define METADATA		"Metadata"

#define MD_DATABASE		"database"
#define DATABASE_STATUS		"STATUS"
#define DATABASE_SYMBOLS	"SYMBOLS"
#define DATABASE_MISC		"MISC"
#define DATABASE_NEWGEN		"NEWGEN" /* newgen type translation data... */

string db_get_meta_data_directory()
{
    return db_get_directory_name_for_module(METADATA);
}

static string meta_data_db_file_name(string data)
{
    string dir_name = db_get_meta_data_directory(),
	res = strdup(concatenate(dir_name, "/" MD_DATABASE ".", data, 0));
    free(dir_name); return res;
}

/* reset all meta data on failures.
 */
static void reset_meta_data(void)
{
    db_reset_current_workspace_name_if_necessary();
    db_reset_logical_time();
    /* what about db_symbols? */
    db_reset_pips_database_if_necessary();
}

/* save (and maybe free) metadata.
 * @return whether sucessful.
 */
static bool save_meta_data(bool do_free)
{
    string file_name;
    FILE * file;

    pips_debug(2, "saving database status\n");
    file_name = meta_data_db_file_name(DATABASE_STATUS);
    file = safe_fopen(file_name, "w");
    db_save_pips_database(file);
    safe_fclose(file, file_name);
    free(file_name);

    if (do_free) db_close_pips_database();

    pips_debug(2, "saving database symbols\n");
    file_name = meta_data_db_file_name(DATABASE_SYMBOLS);
    file = safe_fopen(file_name, "w");
    write_tabulated_db_symbol(file);
    safe_fclose(file, file_name);
    free(file_name);

    if (do_free) gen_free_tabulated(db_symbol_domain);

    pips_debug(2, "saving database misc data\n");
    file_name = meta_data_db_file_name(DATABASE_MISC);
    file = safe_fopen(file_name, "w");
    fprintf(file, "%d\n%s\n", 
	    db_get_logical_time(), db_get_current_workspace_name());
    safe_fclose(file, file_name);
    free(file_name);

    pips_debug(2, "saving newgen type translation data\n");
    file_name = meta_data_db_file_name(DATABASE_NEWGEN);
    gen_type_translation_write(file_name);
    free(file_name);
    
    pips_debug(2, "done\n");

    return TRUE;
}

#define ONERROR(cond, what) \
  if (cond) { what; return FALSE; }

/* load metadata from workspace.
 * @return whether successful.
 */
static bool load_meta_data(void)
{
    string file_name, ws_name;
    FILE * file;
    int time;
    bool ok;

    pips_debug(2, "loading newgen type translation data\n");
    file_name = meta_data_db_file_name(DATABASE_NEWGEN);
    if (file_exists_p(file_name))
      gen_type_translation_read(file_name);
    else
      pips_user_warning("no newgen type translation file...");
    free(file_name);
    
    pips_debug(2, "loading database misc data\n");
    file_name = meta_data_db_file_name(DATABASE_MISC);
    file = check_fopen(file_name, "r");
    ONERROR(!file,/* nope */)
    ONERROR(fscanf(file, "%d\n", &time)!=1,
	    pips_internal_error("fscanf failed\n"))
    db_set_logical_time(time);
    ws_name = safe_readline(file);
    if (!same_string_p(ws_name, db_get_current_workspace_name()))
	pips_user_warning("Workspace %s has been moved to %s\n", 
			  ws_name, db_get_current_workspace_name());
    free(ws_name);
    safe_fclose(file, file_name);
    free(file_name);

    pips_debug(2, "loading database symbols\n");
    file_name = meta_data_db_file_name(DATABASE_SYMBOLS);
    file = check_fopen(file_name, "r");
    ONERROR(!file,/* nope */)
    read_tabulated_db_symbol(file);
	    
    safe_fclose(file, file_name);
    free(file_name);

    pips_debug(2, "loading database status\n");
    file_name = meta_data_db_file_name(DATABASE_STATUS);
    file = check_fopen(file_name, "r");
    ONERROR(!file,/* nope */)
    ok = db_open_pips_database(file);
    ONERROR(!ok, pips_user_error("Could not read database content!"))
    safe_fclose(file, file_name);
    free(file_name);
    
    pips_debug(2, "done\n");

    return TRUE;
}

/**************************************************************** MANAGEMENT */

bool workspace_exists_p(string name)
{
    string full_name = db_get_workspace_directory_name(name);
    bool result = directory_exists_p(full_name);
    free(full_name); 
    return result;
}

bool workspace_ok_p(string name)
{
    string full_name = db_get_workspace_directory_name(name);
    bool result = file_readable_p(full_name);
    free(full_name);
    return result;
}

bool db_create_workspace(string name)
{
    bool ok;
    string dir_name;
    debug_on(PIPSDBM_DEBUG_LEVEL);

    dir_name = db_get_workspace_directory_name(name);

    pips_debug(1, "workspace %s in directory %s\n", name, dir_name);

    if ((ok = purge_directory(dir_name)))
    {
	if ((ok = create_directory(dir_name)))
	{
	    db_set_current_workspace_name(name);
	    db_create_pips_database();
	}
	else
	{
	    pips_user_warning("could not create directory %s\n", dir_name);
	}
    }
    else
    {
	pips_user_warning("could not remove old directory %s\n", dir_name);
    }

    debug_off();
    free(dir_name);
    pips_debug(1, "done\n");
    return ok;
}

/* stores all resources of module oname.
 */
static void db_close_module(string what, string oname)
{
    /* the download order is retrieved from the methods... */
    int nr = dbll_number_of_resources(), i;
    if (!same_string_p(oname, "")) /* log if necessary. */
	user_log("  %s module %s.\n", what, oname);
    for (i=0; i<nr; i++)
	db_save_and_free_memory_resource_if_any
	    (dbll_get_ith_resource_name(i), oname);
}

static void db_save_workspace(string what, bool do_free)
{
    gen_array_t a;

    user_log("%s all modules.\n", what);
    a = db_get_module_list();
    GEN_ARRAY_MAP(module, db_close_module(what, module), a);
    gen_array_full_free(a);
    
    user_log("%s program.\n", what);
    db_close_module(what, ""); /* ENTITIES are saved here... */

    user_log("%s workspace.\n", what);
    save_meta_data(do_free);
}

void db_checkpoint_workspace(void)
{
    debug_on(PIPSDBM_DEBUG_LEVEL);
    pips_debug(1, "Checkpointing workspace %s\n", 
	       db_get_current_workspace_name());

    db_save_workspace("Saving", FALSE); 

    /* load ENTITIES (since no one ask for them as they should...)
     */
    if (db_resource_p(DBR_ENTITIES, "")) 
	(void) db_get_memory_resource(DBR_ENTITIES, "", TRUE);
    debug_off();
}

bool db_close_workspace(void)
{
    debug_on(PIPSDBM_DEBUG_LEVEL);
    pips_debug(1, "Closing workspace %s\n", db_get_current_workspace_name());

    db_save_workspace("Closing", TRUE);
    db_reset_current_workspace_name();
    
    pips_debug(1, "done\n");
    debug_off();
    return TRUE;
}

bool db_open_workspace(string name)
{
    bool ok = TRUE;
    string dir_name;
    debug_on(PIPSDBM_DEBUG_LEVEL);
    pips_debug(1, "Opening workspace %s\n", name);

    dir_name = db_get_workspace_directory_name(name);
    if (directory_exists_p(dir_name))
    {
	db_set_current_workspace_name(name);
	ok = load_meta_data();
	
	if (!ok)  /* failure! */
	{
	    reset_meta_data();
	    pips_user_warning("Cannot load workspace metadata for %s.\n",
			      name);
	}

	/* load ENTITIES (since no one ask for them as they should;-) */
	if (ok && db_resource_p(DBR_ENTITIES, "")) {
	    (void) db_get_memory_resource(DBR_ENTITIES, "", TRUE);
	    /* should touch them somehow to force latter saving? */
	}
    }
    else ok = FALSE;

    pips_debug(1, "done (%d)\n", ok);
    debug_off();
    free(dir_name);
    return ok;
}
