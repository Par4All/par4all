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
#include "linear.h"
#include "resources.h"

#define PIPSDBM_DEBUG_LEVEL "PIPSDBM_DEBUG_LEVEL"

#define PROGRAM_RESOURCE_OWNER ""

#define WORKSPACE_TMP_SPACE 		"Tmp"
#define WORKSPACE_SRC_SPACE 		"Src"
#define WORKSPACE_PROGRAM_SPACE 	"Program"
#define WORKSPACE_METADATA_SPACE	"Metadata"

/* symbols exported by parser / lexer */
extern FILE *genread_in;
extern int genread_input();

/* conform to old interface.
 */

/** Put a memory resource into the current workspace database

    @ingroup pipsdbm
    
    This function allows to update a memory resource already available.

    @param rname is a resource name, such as DBR_CODE for the code of a
    module. The construction of these aliases are DBB_ + the uppercased
    name of a resource defined in pipsmake-rc.tex. They are defined
    automatically in include/resources.h

    @param oname is the resource owner name, typically a module name.
    
    @param res_val is an opaque pointer to the resource to be
    stored. Methods defined in methods.h will know how to deal with.
*/
#define DB_PUT_MEMORY_RESOURCE(res_name, own_name, res_val) \
  db_put_or_update_memory_resource(res_name, own_name, (void*) res_val, true)

/** Put a file resource into the current workspace database

    @ingroup pipsdbm
    
    This function allows to update a file resource already available.

    @param rname is a resource name, such as DBR_CODE for the code of a
    module. The construction of these aliases are DBB_ + the uppercased
    name of a resource defined in pipsmake-rc.tex. They are defined
    automatically in include/resources.h

    @param oname is the resource owner name, typically a module name.
    
    @param res_val is an opaque pointer to the resource to be
    stored. Methods defined in methods.h will know how to deal with.
*/
#define DB_PUT_FILE_RESOURCE DB_PUT_MEMORY_RESOURCE

/** Put a new file resource into the current workspace database

    @ingroup pipsdbm
    
    This function disallows to update a resource already available.

    @param rname is a resource name, such as DBR_CODE for the code of a
    module. The construction of these aliases are DBB_ + the uppercased
    name of a resource defined in pipsmake-rc.tex. They are defined
    automatically in include/resources.h

    @param oname is the resource owner name, typically a module name.
    
    @param res_val is an opaque pointer to the resource to be
    stored. Methods defined in methods.h will know how to deal with.
*/
#define DB_PUT_NEW_FILE_RESOURCE(res_name, own_name, res_val) \
  db_put_or_update_memory_resource(res_name, own_name, (void*) res_val, false)

#define db_get_file_resource db_get_memory_resource
#define db_unput_a_resource(r,o) (db_delete_resource(r,o), true)
#define build_pgmwd db_get_workspace_directory_name

#define db_make_subdirectory(n) free(db_get_directory_name_for_module(n))

#include "newgen.h" /* ??? statement_mapping */
