 /* procedures used in both PIPS top-level, wpips and tpips */
 /* problems to use those procedures with wpips: show_message() and 
    update_props() .
  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <sys/param.h>
/*#include <sys/wait.h>*/
#include <sys/types.h>
#include <dirent.h>
/*#include <sys/timeb.h>*/
#include <sys/stat.h>
#include <ctype.h>
#include <setjmp.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>

#include "genC.h"
#include "ri.h"
#include "graph.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "ri-util.h"
#include "complexity_ri.h"
#include "pipsdbm.h"
#include "tiling.h"

#include "constants.h"
#include "resources.h"
#include "phases.h"

#include "parser_private.h"
#include "dg.h"
#include "property.h"
#include "reduction.h"

#include "pipsmake.h"

#include "top-level.h"

#define LINE_LENGTH 128


/* Return a sorted arg list of workspace names. (For each name, there
   is a name.database directory in the current directory): */
void
pips_get_program_list(int * pargc,
                      char * argv[])
{
   int i;

   /* Find all directories with name ending with ".database": */

   list_files_in_directory(pargc, argv, ".",
                           "^.*\\.database$", directory_exists_p);

   /* Remove the ".database": */
   for (i = 0; i < *pargc; i++) {
      *strchr(argv[i], '.') = '\0';
   }
}


/* Select the true file with names ending in ".f" and return a sorted
   arg list: */
void
pips_get_fortran_list(int * pargc,
                      char * argv[])
{
    list_files_in_directory(pargc, argv, ".", "^.*\\.f$", file_exists_p);
}


char *pips_change_directory(dir)
char *dir;
{
    if (directory_exists_p(dir)) {
	chdir(dir);
	/*
	  log_execl("pwd", NULL);
	  log_execl("/bin/echo Available Fortran Files:", NULL);
	  log_execl("/bin/ls -C *.f", NULL);
	  log_execl("/bin/echo", NULL);
	  log_execl("/bin/echo Available Workspaces:", NULL);
	  log_execl("/bin/ls -C *.DATABASE | sed s/.DATABASE//g", 
	  NULL);
	  */

	return(get_cwd());	
    }

    return(NULL);
}

char *build_view_file(print_type)
char *print_type;
{
    char *module_name = db_get_current_module_name();

    if(module_name != NULL) {
	if ( safe_make_p(print_type, module_name) ) {
	    char * file_name = db_get_file_resource(print_type, module_name, TRUE);
	    return file_name;
	}
    }
    else {
	/* should be show_message */
	user_log("No current module; select a module\n");
    }
    return NULL;
}

char *get_dont_build_view_file(print_type)
char *print_type;
{
    char *module_name = db_get_current_module_name();

    if(module_name != NULL) {
	char * file_name = db_get_file_resource(print_type, module_name, TRUE);
	return file_name;
    }
    else {
	/* should be show_message */
	user_log("No current module; select a module\n");
    }
    return NULL;
}


char *read_line(fd)
FILE *fd;
{
    static char line[LINE_LENGTH];
    if (fgets(line, LINE_LENGTH, fd) != NULL) {
	int l = strlen(line);

	if (l > 0)
	    line[l-1] = '\0';

	return(line);
    }

    return(NULL);
}


void process_user_file(file)
string file;
{
    database pgm;
    FILE *fd;
    char *cwd;
    static char buffer[MAXNAMLEN];
    char *abspath;

    static char *tempfile = NULL;

    if (! file_exists_p(file)) {
	user_warning("process_user_file", "Cannot open file : %s\n", file);
	return;
    }

    if (tempfile == NULL) {
	tempfile = tmpnam(NULL);
    }

    /* the current program is retrieved */
    pgm = db_get_current_program();

    /* the full path of file is calculated */
    abspath = strdup((*file == '/') ? file : 
		     concatenate(get_cwd(), "/", file, NULL));

    /* the new file is registered in the database */
    user_log("Registering file %s\n", file);
    DB_PUT_FILE_RESOURCE(DBR_USER_FILE, database_name(pgm), abspath);

    /* the new file is splitted according to Fortran standard */
    user_log("Splitting file    %s\n", file);
    cwd = strdup(get_cwd());
    chdir(database_directory(pgm));
    /* reverse sort because the list of modules is reversed later */
    system(concatenate("pips-split ", abspath,
		       "| sed -e /zzz00[0-9].f/d | sort -r > ",
		       tempfile, NULL));
    chdir(cwd);
    free(cwd);

    /* the newly created module files are registered in the database */
    fd = safe_fopen(tempfile, "r");
    while (fscanf(fd, "%s", buffer) != EOF) {
	char *modname;
	char *modfullfilename;

	modfullfilename = strdup(concatenate(database_directory(pgm), 
					     "/", buffer, NULL));

	*strchr(buffer, '.') = '\0';
	(void) strupper(buffer, buffer);
	modname = strdup(buffer);

	user_log("  Module         %s\n", modname);
	DB_PUT_FILE_RESOURCE(DBR_SOURCE_FILE, modname, modfullfilename);
    }
    safe_fclose(fd, tempfile);

    /* Next two lines added by BB to remove tempfile. Could be done later, 
     * as exiting the program.
     * 22.03.91
     */
    unlink(tempfile);
    tempfile = NULL;
}
