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
#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
/*
 * Here is the database!
 *
 * Here are just the internals.
 * They rely on pipsdbm private data structures, which are not exported!
 * all exported functions should check that DB_OK.
 */

#include "private.h"
#include "pipsdbm_private.h"

/******************************************************************** UTILS */

/* shorthands
 */
#define db_resource_stored_p(r) db_status_stored_p(db_resource_db_status(r))
#define db_resource_loaded_p(r) db_status_loaded_p(db_resource_db_status(r))
#define db_resource_required_p(r) \
        db_status_required_p(db_resource_db_status(r))
#define db_resource_loaded_and_stored_p(r) \
        db_status_loaded_and_stored_p(db_resource_db_status(r))

/* Module names must use some characters.
 * Upper case letters and underscore for Fortran,
 * but also lower case letters and the FILE_SEP_STRING
 * "#" added for C compilation unit FC 12/08/2003
 *
 * FILE_SEP_STRING added for compilation units (FI)
 * MODULE_SEP_STRING added for static C functions (FI)
 */
#ifndef FILE_SEP_STRING /* in ri-util */
#define FILE_SEP_STRING "!"
#endif /* FILE_SEP_STRING */
#ifndef MODULE_SEP_STRING /* in ri-util */
#define MODULE_SEP_STRING ":"
#endif /* MODULE_SEP_STRING */

#define MODULE_NAME_CHARS \
  ( "ABCDEFGHIJKLMNOPQRSTUVWXYZ" \
    "0123456789" \
    "abcdefghijklmnopqrstuvwxyz" \
    FILE_SEP_STRING MODULE_SEP_STRING "|_#-." )

static bool simple_name_p(const char* name)
{
    return strlen(name)==strspn(name, MODULE_NAME_CHARS);
}

static db_symbol find_or_create_db_symbol(const char* name)
{
    db_symbol s = gen_find_tabulated(name, db_symbol_domain);
    if (!simple_name_p(name))
      pips_user_warning("strange name \"%s\"\n", name);
    return db_symbol_undefined_p(s)? make_db_symbol(strdup(name)): s;
}


/************************************************************** THE DATABASE */

/* the pips_database stores pips resources.
 *
 * {init,close,set_reset,get}_pips_database()
 * {store,load,delete,update,store_or_update}_pips_database()
 * bound_pips_database_p()
 */
GENERIC_LOCAL_FUNCTION(pips_database, db_resources)

#define DB_OK    pips_assert("defined database", !pips_database_undefined_p())
#define DB_UNDEF pips_assert("undefined database", pips_database_undefined_p())

/* exported interface is minimal.
 */
void db_create_pips_database(void)
{
    DB_UNDEF; init_pips_database(); DB_OK;
}

/* latter */
static void db_clean_db_resources();

/* @return whether okay.
 */
bool db_open_pips_database(FILE * fd)
{
    db_resources rs;
    DB_UNDEF;
    rs = read_db_resources(fd);
    if (db_resources_undefined_p(rs)) return false;
    set_pips_database(rs);

    /* coredump in copy if done on save in next function ???. */
    db_clean_db_resources();

    ifdebug(1)
      dump_all_db_resource_status(stderr, "db_open_pips_database");

    DB_OK;
    return true;
}

void db_save_pips_database(FILE * fd)
{
  /* db_resources dbres; */
  DB_OK;
  /* ??? check for required resources left over? */

  /* save a cleaned COPY with status artificially set as STORED... */
  /* dbres = copy_db_resources(get_pips_database());
     db_clean_db_resources(dbres);*/
  write_db_resources(fd, get_pips_database());
  /* free_db_resources(dbres); */
}

void db_close_pips_database(void)
{
    DB_OK; close_pips_database(); DB_UNDEF;
}

void db_reset_pips_database_if_necessary(void)
{
    /* rough! memory leak... */
    if (!pips_database_undefined_p()) reset_pips_database();
}

/******************************************************** LOAD/SAVE RESOURCE */

static string db_status_string(db_status s)
{
  if (db_status_undefined_p(s))
    return "undefined";
  switch (db_status_tag(s))
  {
  case is_db_status_stored:
    return "stored";
  case is_db_status_loaded:
    return "loaded";
  case is_db_status_required:
    return "required";
  case is_db_status_loaded_and_stored:
    return "loaded&stored";
  default:
    pips_internal_error("unexpected db_status tag %d\n", db_status_tag(s));
    return NULL;
  }
}

static void dump_db_resource(const char* rname, const char* oname, db_resource r)
{
  ifdebug(1)
  {
    pips_debug(1, "rname=%s, oname=%s, r=%p\n", rname, oname, r);
    if (!db_resource_undefined_p(r)) {
      db_status s = db_resource_db_status(r);
      pips_debug(1, "pointer=%p, status=%s, time=%td, file_time=%td\n",
		 db_resource_pointer(r), db_status_string(s),
		 db_resource_time(r), db_resource_file_time(r));
    }
  }
}

void dump_all_db_resource_status(FILE * file, string where)
{
  pips_debug(1, "doing at '%s'\n", where);

  DB_RESOURCES_MAP(os, or,
  {
    DB_OWNED_RESOURCES_MAP(rs, r,
    {
      string rn = db_symbol_name(rs);
      string on = db_symbol_name(os);
      fprintf(file, "resource %s[%s] status '%s' since %td (%td) 0x%p\n",
	      rn, on,
	      db_status_string(db_resource_db_status(r)),
	      db_resource_time(r),
	      db_resource_file_time(r),
	      db_resource_pointer(r));
    },
			   or);
  },
		   get_pips_database());

}

#define debug_db_resource(l, r, o, p) ifdebug(l) { dump_db_resource(r, o, p);}

static void init_owned_resources_if_necessary(const char* name)
{
    db_symbol s = find_or_create_db_symbol(name);
    /* set the owner_resources in the pips_database. */
    if (!bound_pips_database_p(s))
	store_pips_database(s, make_db_owned_resources());;
}

static db_owned_resources get_db_owned_resources(const char * oname)
{
    db_symbol o = find_or_create_db_symbol(oname);
    return bound_pips_database_p(o)?
	load_pips_database(o): db_owned_resources_undefined;
}

static db_resource get_resource(const char * rname, db_owned_resources or)
{
    db_symbol rs = find_or_create_db_symbol(rname);
    db_resource r = bound_db_owned_resources_p(or, rs)?
	apply_db_owned_resources(or, rs): db_resource_undefined;
    debug_db_resource(9, rname, "?", r);
    return r;
}

static db_resource get_db_resource(const char * rname, const char * oname)
{
  db_owned_resources or;
  or = get_db_owned_resources(oname);
  if (db_owned_resources_undefined_p(or)) { /* lazy... */
    pips_debug(1, "creating or for %s...\n", oname);
    init_owned_resources_if_necessary(oname);
    or = get_db_owned_resources(oname);
  } /* pips_internal_error("no owned resources for %s", oname, rname);*/
  return get_resource(rname, or);
}

static db_resource get_real_db_resource(const char* rname, const char* oname)
{
  db_resource r = get_db_resource(rname, oname);
  if (db_resource_undefined_p(r) || db_resource_required_p(r))
    pips_internal_error("no resource %s of %s\n", rname, oname);
  return r;
}

static db_resource find_or_create_db_resource(const char* rname, const char* oname)
{
    db_resource r;
    db_owned_resources or;
    or = get_db_owned_resources(oname);
    if (db_owned_resources_undefined_p(or))
    { /* still lazy... */
	pips_debug(1, "module %s should have been registered\n", oname);
	init_owned_resources_if_necessary(oname);
	or = get_db_owned_resources(oname);
    }
    r = get_resource(rname, or);
    if (db_resource_undefined_p(r))
    { /* create it */
	db_symbol rs = find_or_create_db_symbol(rname);
	r = make_db_resource(NULL, db_status_undefined, 0, 0);
	extend_db_owned_resources(or, rs, r);
    }
    return r;
}

/* on checkpoints... there are some incoherencies!
 * - status may be 'loaded', but it is not true!
 * - status may be 'required', but it is not true either.
 * - also, as obsolete resources are not cleaned...
 *
 * maybe the required stuff should be handled by pipsmake on its own?
 */
static void db_clean_db_resources()
{
  list lr = NIL, lo = NIL, lo_init = NIL, lr_init = NIL;

  /* scan */
  DB_RESOURCES_MAP(os, or,
  {
    DB_OWNED_RESOURCES_MAP(rs, r,
    {
      string rn = db_symbol_name(rs);
      string on = db_symbol_name(os);
      pips_debug(8, "considering %s[%s] (0x%p)\n", rn, on, (void*) r);

      if (db_resource_required_p(r))
      {
	pips_debug(1, "resource %s[%s] in state required...\n", rn, on);
	/* to be deleted later on */
	lr = CONS(STRING, rn, lr);
	lo = CONS(STRING, on, lo);
      }
      /* also if the file vanished...
       * maybe on checkpoints where obsolete resources are not removed...
       */
      else if (dbll_stat_resource_file(rn, on, true)==0)
      {
	pips_debug(1, "resource %s[%s] file vanished...\n", rn, on);
	lr = CONS(STRING, rn, lr);
	lo = CONS(STRING, on, lo);
      }
      else if (db_resource_loaded_p(r) || db_resource_loaded_and_stored_p(r))
      {
	pips_debug(1, "resource %s[%s] set as stored\n", rn, on);
	db_status_tag(db_resource_db_status(r)) = is_db_status_stored;
	db_resource_pointer(r) = NULL;
      }
    },
			   or)
      },
		   get_pips_database());

  /* delete */
  for (; lr && lo; lr = CDR(lr), lo = CDR(lo))
  {
    string rn = STRING(CAR(lr)), on = STRING(CAR(lo));
    db_resource r = get_db_resource(rn, on);
    pips_debug(1, "deleting required %s[%s]\n", rn, on);
    dump_db_resource(rn, on, r);
    db_delete_resource(rn, on);
  }

  gen_free_list(lr_init);
  gen_free_list(lo_init);

  ifdebug(9) dump_all_db_resource_status(stderr, "db_clean_db_resources");
}

/** Delete a resource
 */
void db_delete_resource(const char* rname, const char* oname)
{
    db_resource r;
    db_owned_resources or;
    DB_OK;

    or = get_db_owned_resources(oname);
    pips_assert("valid owned resources", !db_owned_resources_undefined_p(or));
    r = get_resource(rname, or);
    if (!db_resource_undefined_p(r))
    {
      /* let us do it! */
      db_symbol rs = find_or_create_db_symbol(rname);
      if ((db_resource_loaded_p(r) || db_resource_loaded_and_stored_p(r)) &&
	  db_resource_pointer(r))
      {
	dbll_free_resource(rname, oname, db_resource_pointer(r));
	/* ??? I should unlink the file */
      }
      db_resource_pointer(r) = NULL;
      free_db_resource(r);
      delete_db_owned_resources(or, rs);
    }
}

/* this should really be a put. Just there for upward compatibility.
 */
bool db_update_time(const char* rname, const char* oname)
{
  db_resource r;
  DB_OK;
  pips_assert("displayable resource", displayable_file_p(rname));
  r = get_real_db_resource(rname, oname);
  db_resource_time(r) = db_get_logical_time();
  db_resource_file_time(r) =
    dbll_stat_local_file((char*) db_resource_pointer(r), false);
  /*dbll_stat_resource_file(rname, oname, true); */
  return true;
}

/* FI wants a sort... so here it is, FC. */
typedef struct
{
  int time;
  const char* owner_name;
  const char* res_name;
} t_tmp_result, * p_tmp_result;

#define gen_DB_VOID_cons(i,l) gen_cons(i,l)

static p_tmp_result make_tmp_result(int t, const char* on, const char* rn)
{
  p_tmp_result res = (p_tmp_result) malloc(sizeof(t_tmp_result));
  res->time = t;
  res->owner_name = on;
  res->res_name = rn;
  return res;
}

static int tmp_result_cmp(const p_tmp_result * p1, const p_tmp_result * p2)
{
  if ((*p1)->time != (*p2)->time)
    return (*p1)->time - (*p2)->time;
  if ((*p1)->owner_name != (*p2)->owner_name)
    return strcmp((*p1)->owner_name, (*p2)->owner_name);
  return strcmp((*p1)->res_name, (*p2)->res_name);
}

void db_print_all_required_resources(FILE * file)
{
  list lres = NIL;

  /* first collect result */
  DB_RESOURCES_MAP(os, or,
  {
    DB_OWNED_RESOURCES_MAP(rs, r,
    {
      string rn = db_symbol_name(rs);
      string on = db_symbol_name(os);
      pips_debug(8, "resource %s[%s] is %s\n",
		 rn, on, db_status_string(db_resource_db_status(r)));

      if (db_resource_required_p(r)) {
	lres = CONS(DB_VOID,
		    make_tmp_result(db_resource_time(r), on, rn),
		    lres);
      }
    },
      or);
  },
		   get_pips_database());

  /* then sort, dump and free */
  gen_sort_list(lres, (gen_cmp_func_t) tmp_result_cmp);

  MAPL(l,
  {
    p_tmp_result p = (p_tmp_result) CAR(l).e;
    fprintf(file,
	    "resource %s[%s] is in 'required' status since %d\n",
	    p->res_name, p->owner_name, p->time);
    free(p);
  },
       lres);
  gen_free_list(lres);
}

void db_clean_all_required_resources(void)
{
  list /* of db_symbols */ owners_to_delete = NIL;
  db_resources db = get_pips_database();

  /* Owner Symbol, Owned Resources */
  DB_RESOURCES_MAP(os, or,
  {
    /* Resource Symbol, DB Resource */
    DB_OWNED_RESOURCES_MAP(rs, r,
    {
      string rn = db_symbol_name(rs);
      string on = db_symbol_name(os);
      pips_debug(8, "considering %s[%s] (%p)\n", rn, on, (void*) r);

      if (db_resource_required_p(r))
      {
	pips_debug(1, "deleting %s[%s]\n", rn, on);
	dump_db_resource(rn, on, r); /* DEBUG? */
	db_delete_resource(rn, on);
      }
    },
      or);

    /* Mark owner symbol os as to be deleted if set of owned resouces is
       empty. */
    if (hash_table_entry_count(db_owned_resources_hash_table(or))==0)
    {
      pips_user_warning("module '%s' to be deleted, no more resources owned.\n",
			/* maybe temporarily required by pipsmake but some error occured */
			db_symbol_name(os));

      owners_to_delete = CONS(DB_SYMBOL, os, owners_to_delete);
    }
  },
		   db);

  MAP(DB_SYMBOL, os, delete_db_resources(db, os), owners_to_delete);
}

/******************************************************* RESOURCE MANAGEMENT */

/* from now on we must not know about the database internals? */

/* true if exists and in *ANY* state. */
bool db_resource_required_or_available_p(const char* rname, const char* oname)
{
  DB_OK;
  return !db_resource_undefined_p(get_db_resource(rname, oname));
}

/* true if exists and in required state. */
bool db_resource_is_required_p(const char* rname, const char* oname)
{
  db_resource r;
  DB_OK;
  r = get_db_resource(rname, oname);
  if (db_resource_undefined_p(r))
    return false;
  else
    return db_resource_required_p(r);
}

/* true if exists and in loaded or stored state. */
bool db_resource_p(const char* rname, const char* oname)
{
  db_resource r;
  DB_OK;
  r = get_db_resource(rname, oname);
  if (db_resource_undefined_p(r))
    return false;
  else
    return db_resource_loaded_p(r) || db_resource_stored_p(r) ||
      db_resource_loaded_and_stored_p(r);
}

/* touch logical time for resource[owner], possibly behind the back of pipsdbm.
 */
bool db_touch_resource(const char* rname, const char* oname)
{
  db_resource r;
  DB_OK;
  r = get_real_db_resource(rname, oname);
  db_resource_time(r) = db_get_logical_time();
  return true;
}

static void db_check_time(const char* rname, const char* oname, db_resource r)
{
  pips_assert("resource is loaded",
	      db_resource_loaded_and_stored_p(r) || db_resource_loaded_p(r));

  /* just check for updates */
  if (displayable_file_p(rname))
  {
    int its_time = dbll_stat_local_file(db_resource_pointer(r), false);
    if (its_time > db_resource_file_time(r))
    {
      pips_user_warning("file resource %s[%s] updated!\n", rname, oname);

      /* update time of actual resource if appropriate
       */
      if ((db_resource_loaded_p(r) || db_resource_loaded_and_stored_p(r)) &&
	  dbll_database_managed_file_p(db_resource_pointer(r)))
      {
	pips_user_warning("file '%s' for %s[%s] edited (%d -> %d)\n",
			  db_resource_pointer(r), rname, oname,
			  db_resource_file_time(r), its_time);
	db_resource_file_time(r) = its_time;

	db_inc_logical_time();
	db_resource_time(r) = db_get_logical_time();
	db_inc_logical_time();
      }
    }
  }
  else
  {
    int its_time = dbll_stat_resource_file(rname, oname, true);
    if (its_time > db_resource_file_time(r))
    {
      /* ??? just warn... may be a user error? */
      pips_user_warning("internal resource %s[%s] updated!\n", rname, oname);
    }
  }
}

static void db_load_resource(const char* rname, const char* oname, db_resource r)
{
  pips_debug(7, "loading %s[%s]\n", rname, oname);
  pips_assert("resource is stored", db_resource_stored_p(r));

  db_resource_pointer(r) = dbll_load_resource(rname, oname);

  if (dbll_very_special_resource_p(rname, oname))
    db_status_tag(db_resource_db_status(r)) = is_db_status_loaded;
  else
    db_status_tag(db_resource_db_status(r)) = is_db_status_loaded_and_stored;

  /* should it be checked elsewhere? */
  db_check_time(rname, oname, r);
}

int db_time_of_resource(const char* rname, const char* oname)
{
  db_resource r;
  DB_OK;

  r = get_db_resource(rname, oname);

  if (db_resource_undefined_p(r) || db_resource_required_p(r))
    return -1;

  /* we load the resource if it is a simple file name...
   * the file time stamps are checked here anyway.
   */
  if (displayable_file_p(rname))
  {
    if (db_resource_stored_p(r))
      db_load_resource(rname, oname, r); /* will update time if needed. */
    else
      db_check_time(rname, oname, r); /* may update time... */
  }

  return db_resource_time(r);
}

static void db_save_resource(const char* rname, const char* oname, db_resource r)
{
    pips_debug(7, "saving %s[%s]\n", rname, oname);
    pips_assert("resource loaded",
		db_resource_loaded_p(r) || db_resource_loaded_and_stored_p(r));

    if (!dbll_storable_p(rname))
      pips_internal_error("cannot store %s\n",rname);

    /* already saved */
    if (db_resource_loaded_and_stored_p(r))
      return;

    dbll_save_resource(rname, oname, db_resource_pointer(r));
    db_status_tag(db_resource_db_status(r)) = is_db_status_stored;

    /* let us set the file time if appropriate... */
    if (!displayable_file_p(rname))
      db_resource_file_time(r) = dbll_stat_resource_file(rname, oname, true);
}

static void db_save_and_free_resource(
    const char* rname, const char* oname, db_resource r, bool do_free)
{
    pips_debug(7, "saving%s... %s[%s]\n",
	       do_free? " and freeing": "", rname, oname);

    pips_assert("resource is loaded",
		db_resource_loaded_p(r) || db_resource_loaded_and_stored_p(r));

    if (db_resource_loaded_and_stored_p(r))
    {
      if (do_free) {
	dbll_free_resource(rname, oname, db_resource_pointer(r));
	db_status_tag(db_resource_db_status(r)) = is_db_status_stored;
	db_resource_pointer(r) = NULL;
      }
      return;
    }
    else if (dbll_storable_p(rname))
    {
      dbll_save_and_free_resource(rname, oname,
				  db_resource_pointer(r), do_free);
      if (do_free)
      {
	db_resource_pointer(r) = NULL;
	db_status_tag(db_resource_db_status(r)) = is_db_status_stored;
      }
      else
      {
	/* ??? manual fix for entities... which are not well integrated. */
	if (!dbll_very_special_resource_p(rname, oname))
	{
	  db_status_tag(db_resource_db_status(r)) =
	    is_db_status_loaded_and_stored;
	}
	else /* is loaded */
	{
	  db_status_tag(db_resource_db_status(r)) = is_db_status_loaded;
	}
      }

      db_resource_file_time(r) = dbll_stat_resource_file(rname, oname, true);
    }
    else
    { /* lost.. just delete the resource. */
      if (do_free)
      {
	dbll_free_resource(rname, oname, db_resource_pointer(r));
	db_resource_pointer(r) = NULL;
	db_delete_resource(rname, oname);
      }
    }
}

/* some way to identify a resource... could be an id...
 */
string db_get_resource_id(const char* rname, const char* oname)
{
    return (char*) get_real_db_resource(rname, oname);
}


/** Return the pointer to the resource, whatever it is.

    @ingroup pipsdbm

    Assume that the resource is available.

    @param rname is a resource name, such as DBR_CODE for the code of a
    module. The construction of these aliases are DBB_ + the uppercased
    name of a resource defined in pipsmake-rc.tex. They are defined
    automatically in include/resources.h

    @param oname is the resource owner name, typically a module name.

    @param pure is used to declare the programmer intentions about the
    future of the resource.

    - If pure is true, the real resource is given as usual. If the
      programmer use it in a read-only way, this is fine and the good way
      to go. If it is modified by the programmer in a phase, all
      subsequent phases that use this resource will access a modified
      resource in an unnoticed way and use other resources that will be
      out-of-sync because unnoticed by pipsmake. So in this case, the
      programmer is assumed to put back the modified in the database later
      to notify pipsmake.

    - If pure is false, then the resource is saved on disk and/or marked
      as outdated before being returned. The idea is that we can modify
      and use this resource as we want in a phase, it is a throwable
      resource. Next time someone will want to use this resource, it will
      be read back from disk or recomputed and nobody will seen it has
      been changed by the first phase. The idea here is to behave as if
      db_get_memory_resource() return a copy, so you are responsible of
      its future (garbage collecting for example if you do not want it any
      longer).

      This feature has been introduced in a time when gen_copy() did not
      exist to duplicate resources and using persistence as a way to get a
      copy was a sensible trick. But now we have gen_copy(), it is far
      more efficient to use it instead of writing it to disk and parsing
      it again or recomputing it. So in all new phases developed in PIPS,
      pure should be always true and gen_copy() should be used when
      necessary.

    @return an opaque pointer to the resource in memory.
 */
string db_get_memory_resource(const char* rname, const char* oname, bool pure)
{
    db_resource r;
    void * result;
    DB_OK;

    debug_on("PIPSDBM_DEBUG_LEVEL");
    pips_debug(2, "get %s[%s] (%s)\n", rname, oname, pure? "pure": "not pure");

    r = get_db_resource(rname, oname);
    debug_db_resource(9, rname, oname, r);
    if (db_resource_undefined_p(r) || db_resource_required_p(r))
      pips_internal_error("requested resource \"%s\" for module \"%s\" not available\n",
			  rname, oname);
    /* else we have something. */

    if (db_resource_stored_p(r))
	db_load_resource(rname, oname, r); /* does it unlink the file? */

    result = db_resource_pointer(r);

    if (!pure)
    {
      /* Save if possible to hide side effects. */
      if (dbll_storable_p(rname)) {
	db_save_resource(rname, oname, r); /* the pointer is there... */
	/* make as stored now... */
	db_resource_pointer(r) = NULL;
	db_status_tag(db_resource_db_status(r)) = is_db_status_stored;
      } else
	/* Mark the resource as lost so next time it will be required,
	   pipsmake will recompute it for example: */
	db_delete_resource(rname, oname);
    }

    ifdebug(7)
      pips_assert("resource is consistent",
		  dbll_check_resource(rname, oname, result));
    debug_off();
    return result;
}

void db_set_resource_as_required(const char* rname, const char* oname)
{
  db_resource r;
  db_status s;
  DB_OK;

  pips_debug(5, "set %s[%s] as required at %d\n", rname, oname,
	     db_get_logical_time());

  r = find_or_create_db_resource(rname, oname);
  s = db_resource_db_status(r);
  if (db_status_undefined_p(s)) {
    /* newly created db_resource... */
    db_resource_db_status(r) = make_db_status(is_db_status_required, UU);
  }
  else
  {
    pips_debug(1, "set %s[%s] as 'required' from '%s' at %d\n",
	       rname, oname,
	       db_status_string(db_resource_db_status(r)),
	       db_get_logical_time());

    if ((db_status_loaded_p(s) || db_status_loaded_and_stored_p(s)) &&
	db_resource_pointer(r))
    {
      dbll_free_resource(rname, oname, db_resource_pointer(r));
      db_resource_pointer(r) = NULL;
    }
  }

  db_status_tag(db_resource_db_status(r)) = is_db_status_required;
  db_resource_time(r) = db_get_logical_time();
}


/** Put a resource into the current workspace database

    @ingroup pipsdbm

    @param rname is a resource name, such as DBR_CODE for the code of a
    module. The construction of these aliases are DBB_ + the uppercased
    name of a resource defined in pipsmake-rc.tex. They are defined
    automatically in include/resources.h

    @param oname is the resource owner name, typically a module name.

    @param p is an opaque pointer to the resource to be stored. Methods
    defined in methods.h will know how to deal with.

    @param update_is_ok is a parameter to allow updating a resource.

    - If false and a resource is valid and not marked as required, this
      function will fail

    - If true, even if a resource is valid and not marked as required,
      this function will succeed to update it.
*/
void db_put_or_update_memory_resource(const char* rname, const char* oname,
				      void * p, bool update_is_ok) {
  db_resource r;
  /* Check the database coherency: */
  DB_OK;

  debug_on("PIPSDBM_DEBUG_LEVEL");
  pips_debug(2, "putting or updating %s[%s]\n", rname, oname);
  ifdebug(7) pips_assert("resource is consistent",
			 dbll_check_resource(rname, oname, p));

  /* Get the database resource associated to the given resource: */
  r = find_or_create_db_resource(rname, oname);
  if (db_status_undefined_p(db_resource_db_status(r)))
    /* The resource does not exist: it was just created, so mark it as
       loaded into memory: */
    db_resource_db_status(r) = make_db_status_loaded();
  else
    /* The resource already exists... */
    if (!update_is_ok && !db_resource_required_p(r))
      /* If the resource is not required and we do not want to update it: */
      pips_internal_error("resource %s[%s] already there\n",
			  rname, oname);

  /* Store data */
  db_resource_pointer(r) = p; /**< ??? memory leak? depends? */
  /* Mark the resource as loaded into memory: */
  db_status_tag(db_resource_db_status(r)) = is_db_status_loaded;
  /* Timestamp the resource with the current logical time: */
  db_resource_time(r) = db_get_logical_time();

  if (displayable_file_p(rname))
    /* If there is a text file associated to the resource, get its
       modification time: */
    db_resource_file_time(r) =
      dbll_stat_local_file(db_resource_pointer(r), false);
  else
    db_resource_file_time(r) = 0; /**< Or what else? */

  debug_db_resource(9, rname, oname, r);
  debug_off();
}

void db_invalidate_memory_resource(const char * rname, const char *oname)
{
    db_resource r;
    /* Check the database coherency: */
    DB_OK;

    /* Get the database resource associated to the given resource: */
    r = find_or_create_db_resource(rname, oname);
    if (!db_status_undefined_p(db_resource_db_status(r)))
        db_resource_file_time(r)=0;
}

/** Delete all the resources of a given type "rname"
 *
 * Return the number of deleted resources
 *
 * Each owner map has to be checked because of two-level mapping used.
 */
int db_unput_resources(const char* rname)
{
    db_symbol r;
    int count = 0;
    DB_OK;
    r = find_or_create_db_symbol(rname);
    DB_RESOURCES_MAP(s, or,
    {
	pips_debug(7, "deleting %s[%s] if any\n", rname, db_symbol_name(s));
	if (bound_db_owned_resources_p(or, r)) {
	    db_delete_resource(rname, db_symbol_name(s));
	    dbll_unlink_resource_file(rname, db_symbol_name(s), false);
	    count ++;
	}
    },
		     get_pips_database());
    return count;
}

/* Retrieve all the db resources of a given resource type, "rname".
 *
 * Scan all module hash tables to find all resources of kind "rname",
 * no matter what the owner is.
 *
 * Used only to clean up the make cache in pipsmake.c.
 *
 * Derived from db_unput_resources()
 */
list db_retrieve_resources(const char* rname)
{
  list rl = NIL;
  db_symbol r;
  DB_OK;
  r = find_or_create_db_symbol(rname);
  /* Scan all resource maps or of owners */
  DB_RESOURCES_MAP(s, or,
		   {
		     pips_debug(9, "Resources for owner \"%s\"\n",  (string) s);
		     /* See if it contains a resource of kind rname,
			normalized to r */
		     if (bound_db_owned_resources_p(or, r)) {
		       /* */
		       db_resource dbr = get_resource(rname, or);
		       rl = CONS(DB_RESOURCE, dbr, rl);
		     }
		   },
		   get_pips_database());
  return rl;
}

/* Retrieve the resource name, a.k.a. kind or nature, or the resource
 * owner name of db_resource "dbr" according to "owner_p".
 *
 * The two-level mapping must be inverted.
 *
 * string_undefined is returned if "dbr" is not found in the current
 * resource database.
 *
 * This is used for interactive debugging and for debug messages.
 */
static string db_resource_name_or_owner_name(db_resource dbr, bool owner_p)
{
  string name = string_undefined;
    DB_OK;
    DB_RESOURCES_MAP(s1, or,
    {
      pips_debug(9, "Resources for module \"%s\":\n", db_symbol_name(s1));
      DB_OWNED_RESOURCES_MAP(s2, dbr21,
			     {
			       pips_debug(9, "\t\"%s\":\n", db_symbol_name(s2));
			       if(dbr==dbr21) {
				 if(owner_p)
				   name =  db_symbol_name(s1);
				 else
				   name =  db_symbol_name(s2);
				 break;
			       }
			     },
			     or);
    },
		      get_pips_database());
     return name;
}

/* To be used for debugging */
string db_resource_name(void * dbr)
{
  return db_resource_name_or_owner_name((db_resource) dbr, false);
}

/* To be used for debugging */
string db_resource_owner_name(void * dbr)
{
  return db_resource_name_or_owner_name((db_resource) dbr, true);
}


void db_save_and_free_memory_resource_if_any
  (const char* rname, const char* oname, bool do_free)
{
    db_resource r;
    DB_OK;

    pips_debug(8, "maybe saving%s... %s[%s]\n",
	       do_free? " and freeing":"", rname, oname);

    r = get_db_resource(rname, oname);
    if (!db_resource_undefined_p(r) &&
	(db_resource_loaded_p(r) || db_resource_loaded_and_stored_p(r)))
	db_save_and_free_resource(rname, oname, r, do_free);
}

/* FC: I added this function to clean all resources, hence avoiding
 * to save them. This speed up hpfc at low cost;-).
 */
void db_delete_all_resources(void)
{
    int nr = dbll_number_of_resources(), i;
    DB_OK;
    for (i=0; i<nr; i++)
      (void) db_unput_resources(dbll_get_ith_resource_name(i));
}

/******************************************************************* MODULES */

/* when telling the database about a module name, the module is
 * registered as a db_symbol, and it is added to the database.
 */
static char* current_module_name = NULL;

bool db_set_current_module_name(const char* name)
{
    bool ok = false;
    DB_OK; pips_assert("no current module", !current_module_name);
    if (simple_name_p(name)) {
	current_module_name = strdup(name);
	init_owned_resources_if_necessary(name);
	ok = true;
    } else /* can be rejected softly */
	pips_user_warning("invalid module name \"%s\"\n", name);
    return ok;
}

/* Also used to check whether set... so no asserts, even DB_OK. */
string db_get_current_module_name(void)
{
    return current_module_name;
}

void db_reset_current_module_name(void)
{
    DB_OK; pips_assert("some current module name", current_module_name);
    free(current_module_name), current_module_name = NULL;
}

/***************************************************************** CLEANING */


/* delete all obsolete resources before a close.
 * return the number of resources destroyed.
 */
int db_delete_obsolete_resources(bool (*keep_p)(const char*, const char*))
{
    int ndeleted = 0;
    list /* of string */ lr = NIL, lo = NIL, lrp, lop;
    DB_OK;
    debug_on("PIPSDBM_DEBUG_LEVEL");

    /* builds the lists to delete. */
    DB_RESOURCES_MAP(os, or,
    {
      DB_OWNED_RESOURCES_MAP(rs, r,
      {
	string rn = db_symbol_name(rs);
	string on = db_symbol_name(os);
	pips_debug(8, "considering %s of %s (%p)\n", rn, on, (void *) r);
	if (!db_resource_required_p(r) && !keep_p(rn, on)) {
	  pips_debug(8, "to be destroyed: %s of %s\n", rn, on);
	  ndeleted++;
	  lr = CONS(STRING, rn, lr);
	  lo = CONS(STRING, on, lo);
	}
      },
			     or);
    },
		     get_pips_database());

    /* delete the resources. */
    for(lrp=lr, lop=lo; !ENDP(lrp); POP(lrp), POP(lop)) {
	string rname = STRING(CAR(lrp)), oname = STRING(CAR(lop));
	db_delete_resource(rname, oname);
	dbll_unlink_resource_file(rname, oname, false);
    }

    gen_free_list(lr);
    gen_free_list(lo);
    debug_off();
    return ndeleted;
}



/** Return whether name is a "valid" module.
 *
 * @ingroup pipsdbm
 *
 * As FI points out to me (FC), it just means that the
 * name has been used by some-one, some-where, some-time...
 *
 * It just checks that an non empty resource table is associated to
 * this name. The table may be created when resources are marked as
 * required by pipsmake, and is never destroyed?
 */
bool db_module_exists_p(const char* name)
{
  bool ok = false;

  if (!pips_database_undefined_p()) /* some database? */
  {
    db_symbol s = gen_find_tabulated(name, db_symbol_domain);
    if (!db_symbol_undefined_p(s)) /* some symbol? */
    {
      db_resources dbr = get_pips_database();
      if (bound_db_resources_p(dbr, s)) /* some resource table? */
      {
	db_owned_resources or = apply_db_resources(dbr, s);

	/* some actual resource? */
	ok = hash_table_entry_count(db_owned_resources_hash_table(or))>0;
      }
    }
  }

  return ok;
}

gen_array_t db_get_module_list_initial_order(void)
{
  list /* of db_symbol */ ls;
  db_resources dbr;
  gen_array_t a;

  DB_OK;

  /* the list returned is reversed... */
  ls = gen_filter_tabulated(gen_true, db_symbol_domain);
  a = gen_array_make(0);
  dbr = get_pips_database();


  FOREACH(DB_SYMBOL, symbol, ls)
  {
      string name = db_symbol_name(symbol);
      /* if it is a module, append... */
      if (!string_undefined_p(name) &&
              !same_string_p(name, "") &&
              /* I should check that some actual resources is stored? */
              bound_db_resources_p(dbr, symbol) &&
              compilation_unit_p(name)
      )
          gen_array_dupappend(a, name);
  }
  FOREACH(DB_SYMBOL, symbol, ls)
  {
      string name = db_symbol_name(symbol);
      /* if it is a module, append... */
      if (!string_undefined_p(name) &&
              !same_string_p(name, "") &&
              /* I should check that some actual resources is stored? */
              bound_db_resources_p(dbr, symbol) &&
              !compilation_unit_p(name)
      )
          gen_array_dupappend(a, name);
  }


  gen_free_list(ls);
  return a;
}


/** @addtogroup pipsdbm */

/** @{ */


/* Returns an allocated array a with the sorted list of modules.
 *
 * strings are duplicated.
 *
 * Compilation units were not added because Fabien Coelho wanted to
 * avoid them in validation files: they do depend on include files
 * varying from machine to machine. Another reason to avoid them could
 * be that they are not real module with a signature and
 * code. However, the semantics of tpips %ALL does include them. It's
 * up to the validation designer to avoid including varying stuff in
 * test files. Another possibility would be to regenerate include
 * statements...
 *
 * @param module_p is used to select or not compilation units too. If
 * true, compilation units are also included.
 */
gen_array_t db_get_module_or_function_list(bool module_p)
{
    gen_array_t a = gen_array_make(0);
    DB_OK;

    DB_RESOURCES_MAP(os, or,
    {
	string on = db_symbol_name(os);
	pips_assert("some symbol name", on);
	pips_debug(9, "considering %s -> %p\n", on, or);
	if (!same_string_p(on, "") && (module_p || !compilation_unit_p(on)))
	    gen_array_dupappend(a, on);
    },
		     get_pips_database());

    gen_array_sort(a);
    return a;
}


/** Get an array of all the modules (functions, procedures and compilation
    units) of a workspace.

    @return an array of sorted strdup()ed strings.
 */
gen_array_t db_get_module_list(void)
{
  return db_get_module_or_function_list(true);
}


/** Get an array of all the functions and procedures (not compilation
    units) of a workspace.

    @return an array of sorted strdup()ed strings.
 */
gen_array_t db_get_function_list(void)
{
  return db_get_module_or_function_list(false);
}

/** @} */
