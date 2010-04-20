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
/* Test if a name ends with .F */
bool dot_F_file_p( string name ) {
  STUB_WARNING();
  return !!find_suffix( name, RATFOR_FILE_SUFFIX );
}

/* Test if a name ends with .f */
bool dot_f_file_p( string name ) {
  STUB_WARNING();
  return !!find_suffix( name, FORTRAN_FILE_SUFFIX );
}

/* Test if a name ends with .c */
bool dot_c_file_p( string name ) {
  STUB_WARNING();
  return !!find_suffix( name, C_FILE_SUFFIX );
}

/** Test if a module is in C */
bool c_module_p( entity m ) {
  STUB_WARNING();
  return FALSE;
}

/** Test if a module is in Fortran */
bool fortran_module_p( entity m ) {
  STUB_WARNING();
  return TRUE;
}

entity find_enum_of_member( entity m ) {
  STUB_ERROR();
}

string entity_minimal_name( entity e ) {
  STUB_ERROR();
}

string compilation_unit_of_module(string module_name) {
  STUB_ERROR();
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

string empty_scope() {
  STUB_WARNING();
  return strdup( "" );
}
bool empty_scope_p( string s ) {
  STUB_WARNING();
  return strcmp( s, "" ) == 0;
}
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
