#include <stdio.h>

#include "genC.h"

#include "ri.h"
#include "misc.h"
#include "database.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "properties.h"

#define LOG_FILE "Logfile"

/* The log file is closed by default
 */
static FILE *log_file = NULL;

FILE * get_log_file()
{
    return log_file;
}

void
close_log_file(void)
{
   if (log_file != NULL && get_bool_property("USER_LOG_P") == TRUE)
      if (fclose(log_file) != 0) {
	  pips_error("close_log_file", "Could not close\n");
         perror("close_log_file");
         abort();
      }
   log_file = NULL;
}


void
open_log_file(void)
{

    if (log_file != NULL)
	close_log_file();

    if (get_bool_property("USER_LOG_P") == TRUE) 
    {
	string 
	    dir = db_get_current_workspace_directory(),
	    log_file_name = strdup(concatenate(dir, "/", LOG_FILE, 0));
	free(dir);
	
	if ((log_file = fopen(log_file_name, "a")) == NULL) {
	    perror("[open_log_file]");
	    pips_user_error("Cannot open log file in workspace %s. "
			    "Check access rights.");
	}

	free(log_file_name);
    }
}


void
log_on_file(char chaine[])
{
   if (log_file != NULL /* && get_bool_property("USER_LOG_P") == TRUE */) {
      if (fprintf(log_file, "%s", chaine) <= 0) {
         perror("log_on_file");
         abort();
      }
      else
         fflush(log_file);
   }
}
