/*
 * $Id$
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
