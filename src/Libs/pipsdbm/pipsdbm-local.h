/*
 * $Id$
 */

#define PROGRAM_RESOURCE_OWNER ""
#define WORKSPACE_TMP_SPACE "Tmp"

/* conform to old interface.
 */
#define DB_PUT_MEMORY_RESOURCE(res_name, own_name, res_val) \
  db_put_or_update_memory_resource(res_name, own_name, (char*) res_val, TRUE)

/* put a resource which is a file. it is just a resource as any other.
 */
#define DB_PUT_FILE_RESOURCE DB_PUT_MEMORY_RESOURCE
#define DB_PUT_NEW_FILE_RESOURCE(res_name, own_name, res_val) \
  db_put_or_update_memory_resource(res_name, own_name, (char*) res_val, FALSE)

#define db_get_file_resource db_get_memory_resource
#define db_unput_a_resource(r,o) (db_delete_resource(r,o), TRUE)
#define build_pgmwd db_get_workspace_directory_name
