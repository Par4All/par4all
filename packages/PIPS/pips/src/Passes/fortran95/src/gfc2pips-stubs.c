/*

 $Id$

 Copyright 1989-2014 MINES ParisTech
 Copyright 2009-2010 HPC-Project

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

#include "gfc2pips-private.h"

#include "c_parser_private.h"
#include "misc.h"
#include "text-util.h"
#include <stdio.h>



/************************************
 * STUB FUNCTIONS
 * They are there to get rid of the whole spaghetti
 * that is PIPS libraries linking :-(
 */

#define _STUB_MSG_(severity,msg) \
  pips_debug(0,"@%s(%d) !%s! : %s \n",__FILE__,__LINE__,severity,msg);
#define STUB_ERROR_MSG(msg) { _STUB_MSG_("ERROR",msg); exit(1); }
#define STUB_WARNING_MSG(msg) _STUB_MSG_("WARNING",msg)
#define STUB_ERROR() STUB_ERROR_MSG("")
#define STUB_WARNING() STUB_WARNING_MSG("")

// Preprocessor
#include "preprocessor.h"

const char* entity_minimal_user_name(entity e) {
  STUB_ERROR();
}


const char* entity_minimal_name(entity e) {
  STUB_ERROR();
}

string compilation_unit_of_module(string module_name) {
  STUB_WARNING();
  return "";
}


/*************************************************
 * PIPSDBM
 */

string db_get_memory_resource( const string rname,
                               const string oname,
                               bool pure ) {
  STUB_ERROR();
}

string db_get_meta_data_directory() {
  STUB_ERROR();
}

string db_get_directory_name_for_module( string name ) {
  STUB_WARNING();
  return string_undefined;
}

gen_array_t db_get_module_list( void ) {
  STUB_WARNING();
  gen_array_t a = gen_array_make( 0 );
  return a;
}

string db_get_current_workspace_directory( void ) {
  STUB_WARNING();
  return string_undefined;
}

bool db_resource_p( string rname, string oname ) {
  STUB_WARNING();
  return FALSE;
}
void db_put_or_update_memory_resource( string rname,
                                       string oname,
                                       void * p,
                                       bool update_is_ok ) {
  STUB_ERROR();
}

/************************************************
 * C_Syntax
 */

void CParserError( char *msg ) {
  STUB_ERROR();
}

/***********************************************
 * Effects
 */
list expression_to_proper_effects( expression e ) {
  STUB_ERROR();
}

/***********************************************
 * Control
 */
void clean_up_sequences( statement s ) {
  STUB_ERROR();
} // FIXME Really unused ??
unstructured control_graph( statement st ) {
  STUB_ERROR();
}
void unspaghettify_statement( statement mod_stmt ) {
  STUB_ERROR();
}

statement hcfg(statement st){
  STUB_ERROR();
}
/***********************************************
 * STEP
 */
void load_global_directives( entity k ) {
  STUB_ERROR();
}
string directive_to_string( void *d, bool close ) {
  STUB_ERROR();
}

/***********************************************
 * PRETTYPRINT
 */
bool make_text_resource_and_free() {
  STUB_ERROR();
}
void dump_functional( functional f, string_buffer result ) {
  STUB_ERROR();
}

/**********************************************
 * FLINT
 */
bool check_loop_range( range r, hash_table h ) {
  STUB_ERROR();
}

/**********************************************
 * PROPERTIES
 */
bool get_bool_property( const string name ) {
  if ( strcmp( "PRETTYPRINT_LISTS_WITH_SPACES", name ) == 0
      || strcmp( "PRETTYPRINT_REGENERATE_ALTERNATE_RETURNS", name ) == 0
      || strcmp( "PRETTYPRINT_C_CODE", name ) == 0
      || strcmp( "ABORT_ON_USER_ERROR", name ) == 0 ) {
    return true;
  }
  if ( strcmp( "NO_USER_WARNING", name ) == 0 ) {
    return false;
  }
  fprintf( stderr, "***** Property requested : %s ***** ", name );
  STUB_WARNING();
  return 0;
}
string get_string_property( const string name ) {
  fprintf( stderr, "***** Property requested : %s ***** ", name );
  STUB_ERROR();
}
int get_int_property( const string name ) {
  fprintf( stderr, "***** Property requested : %s ***** ", name );
  STUB_ERROR();
}
void set_bool_property( const string name, bool b ) {
  fprintf( stderr, "***** Property requested : %s ***** ", name );
  STUB_ERROR();
}
void set_string_property( const string name, string s ) {
  fprintf( stderr, "***** Property requested : %s ***** ", name );
  STUB_ERROR();
}
bool too_many_property_errors_pending_p() {
  STUB_WARNING_MSG("unimplemented in gfc2pips !");
}


/*************************************************
 * PIPSMAKE
 */
bool active_phase_p(const char* phase)
{
  STUB_ERROR();
}

