/*

 $Id$

 Copyright 1989-2014 MINES ParisTech

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
/*
 * procedures used in both PIPS top-level, wpips and tpips.
 *
 * problems to use those procedures with wpips: show_message() and
 * update_props() .
 */
#ifdef HAVE_CONFIG_H
#include "pips_config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "ri-util.h"
#include "parser_private.h"

#include "resources.h"
#include "database.h"

#include "misc.h"
#include "pipsdbm.h"

char * f95split( char * dir_name, char * file_name, FILE ** out ) {

  FILE *fd;


  debug_on( "SYNTAX_DEBUG_LEVEL" );

  /*fprintf(stderr,
   "\n-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n"
   "+ Starting gfc parser in PIPS.                 -\n"
   "-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+\n"
   );*/

  //  string callees_filename = get_resource_file_name( DBR_CALLEES, module );
  // Create if it doesn't exist
  //  close( open( callees_filename, O_CREAT, S_IRWXU ) );


  /*
   *
   */

  // Create xxx.database/Program if it doesn't exist Yet
  string program_dirname = get_resource_file_name( "", "" );
  mkdir( program_dirname, S_IRWXU );
  free( program_dirname );

  /*
   * Dump entities
   */
  string entities_filename = get_resource_file_name( DBR_ENTITIES, "" );
  fd = (FILE *) safe_fopen( (char *) entities_filename, "w" );
  gen_write_tabulated( fd, entity_domain );
  safe_fclose( fd, (char *) entities_filename );

  /*
   * directory where module are precompiled
   */
  char *compiled_dir_name = get_resource_file_name( "", "/Precompiled" );

  // Check if source file exist
  /*  string source_filename =
   strdup( concatenate( dir, "/", db_get_file_resource( DBR_SOURCE_FILE,
   module,
   true ), NULL ) );
   */
  // "char **argv" for gfc2pips :-)
  char* gfc2pips_args[] = { "gfc2pips", "-Wall",// "-Werror",
                            // We give it where to output...
                            "-pips-entities",
                            entities_filename,
                            "-o","/dev/null",
                            //parsedcode_filename,
                            file_name,
                            // ... and where to read inputs
                            "-auxbase",
                            //source_filename,
                            dir_name,
                            /* dump_parse_tree is the gfc pass that have been
                             * hacked to call the gfc2pips stuff
                             */
                            "-fdump-parse-tree",
                            "-quiet",
                            "-I",compiled_dir_name,
                            /* we may have non-standard file extensions
                             * (e.g. .f_initial) and gfortran will not be able
                             * to know what it is so we force the language input
                             */
                            // "-x",
                            // "f95",
                            "-cpp",
                            // "-quiet",// "-Werror",
                            /* I don't know what the following stuff is ... */
                            /*"-fcray-pointer",*/
                            "-ffree-form",
                            //"-fdefault-double-8",
                            //"-fdefault-integer-8",
                            //"-fdefault-real-8",

                            // Argv must be null terminated
                            NULL };

  // we will fork now in order to call GFC
  int statut;
  pid_t child_pid = fork( );

  // Now we have two process runing the same code
  // We differentiate them with fork() return value

  if ( child_pid == -1 ) {
    // Error
    perror( "Fork" );
    return false;
  } else if ( child_pid == 0 ) {
    // in the child


    pips_debug( 2, "build module %s\n", file_name );

    // MAIN CALL TO GFC
    char** arg = gfc2pips_args;
    ifdebug(1) {
      fprintf( stderr, "execvp : " );
      while ( *arg ) {
        fprintf( stderr, " %s", *arg );
        arg++;
      }
      fprintf( stderr, "\n" );
    }
    execvp( "gfc2pips", gfc2pips_args );
    // No return from exec
    pips_user_error("gfc2pips is not installed, did you compile PIPS with"
        " Fortran95 support ?\n");
    exit( -1 );
  } else {
    // in the Father

    // Wait that gfc has done the job
    if ( waitpid( child_pid, &statut, 0 ) == -1 ) {
      // Error in waitpid
      perror( "waitpid" );
      return "Erreur in wait pid";
    } else {
      // Child has correctly finished

      // Check the gfc2pips return code
      if ( statut != EXIT_SUCCESS ) {
        fprintf(stderr,"error code %d\n",statut);
        return "gfc2pips return an error";
      }
    }
  }

  // If we are, everything was done properly :-)


  // Reload entities

  // We have to close it and re-open since gfc2pips has written into it.
  fclose( *out );

  //out will be close in caller
  *out = safe_fopen( entities_filename, "r" );

  gen_read_tabulated( *out, 0 );
//  safe_fclose( fp, entities_filename );



  /* This debug_off() occurs too late since pipsdbm has been called
   * before. Initially, the parser was designed to parse more than
   * one subroutine/function/program at a time.  */
  debug_off( );



  return NULL;

}

