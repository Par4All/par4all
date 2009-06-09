/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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

#define PIPSDBM_DEBUG_LEVEL "PIPSDBM_DEBUG_LEVEL"

#define PROGRAM_RESOURCE_OWNER ""

#define WORKSPACE_TMP_SPACE 		"Tmp"
#define WORKSPACE_SRC_SPACE 		"Src"
#define WORKSPACE_PROGRAM_SPACE 	"Program"
#define WORKSPACE_METADATA_SPACE	"Metadata"

/* conform to old interface.
 */
#define DB_PUT_MEMORY_RESOURCE(res_name, own_name, res_val) \
  db_put_or_update_memory_resource(res_name, own_name, (void*) res_val, TRUE)

/* put a resource which is a file. it is just a resource as any other.
 */
#define DB_PUT_FILE_RESOURCE DB_PUT_MEMORY_RESOURCE
#define DB_PUT_NEW_FILE_RESOURCE(res_name, own_name, res_val) \
  db_put_or_update_memory_resource(res_name, own_name, (void*) res_val, FALSE)

#define db_get_file_resource db_get_memory_resource
#define db_unput_a_resource(r,o) (db_delete_resource(r,o), TRUE)
#define build_pgmwd db_get_workspace_directory_name

#define db_make_subdirectory(n) free(db_get_directory_name_for_module(n))

#include "newgen.h" /* ??? statement_mapping */
