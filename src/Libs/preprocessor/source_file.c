 /* procedures used in both PIPS top-level, wpips and tpips */
 /* problems to use those procedures with wpips: show_message() and 
    update_props() .
  */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/*#include <sys/param.h>*/
/*#include <sys/wait.h>*/
/*#include <sys/types.h>*/
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
pips_get_workspace_list(int * pargc,
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


/* Return the path of an HPFC file name relative to the current PIPS
   directory. Can be freed by the caller. */
char *
hpfc_generate_path_name_of_file_name(char * file_name)
{
   return concatenate(build_pgmwd(db_get_current_workspace_name()),
                      "/",
                      HPFC_COMPILED_FILE_DIR,
                      "/",
                      file_name,
                      NULL);
}


int
hpfc_get_file_list(int * file_number,
                   char * file_names[],
                   char ** hpfc_directory_name)
{
   static char hpfc_dir[MAXNAMLEN + 1];
   int return_code;
        
   /* Get the HPFC file name list: */
   sprintf(hpfc_dir, "%s/%s",
           build_pgmwd(db_get_current_workspace_name()),
           HPFC_COMPILED_FILE_DIR);
   
   return_code = safe_list_files_in_directory(file_number,
                                              file_names,
                                /* Where is the output of HPFC: */
                                              hpfc_dir,
                                /* generated files start with upercases */
                                              "^[A-Z].*\\.[fh]$",
                                /* Plain files only: */
                                              file_exists_p);
   *hpfc_directory_name = hpfc_dir;

   return return_code;
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

char *
build_view_file(char * print_type)
{
   char * module_name = db_get_current_module_name();

   if(!unloadable_file_p(print_type)) {
       user_error("build_view_file", "resource %s cannot be displayed\n",
		   print_type);
   }

   if(module_name != NULL) {
      if ( safe_make(print_type, module_name) ) {
         static char file_name_in_database[MAXPATHLEN];
           
         char * file_name = db_get_file_resource(print_type, module_name, TRUE);
         sprintf(file_name_in_database, "%s/%s",
                 build_pgmwd(db_get_current_workspace_name()),
                 file_name);
            
         return file_name_in_database;
      }
   }
   else {
      /* should be show_message */
      user_log("No current module; select a module\n");
   }
   return NULL;
}

char *
get_dont_build_view_file(char * print_type)
{
   char *module_name = db_get_current_module_name();

   if(module_name != NULL) {
      /* Allow some place for "/./" and other useless stuff: */
       static char file_name_in_database[MAXPATHLEN];
           
      char * file_name = db_get_file_resource(print_type, module_name, TRUE);

      sprintf(file_name_in_database, "%s/%s",
              build_pgmwd(db_get_current_workspace_name()),
              file_name);
            
      return file_name_in_database;
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


bool process_user_file(file)
string file;
{
    bool success_p = FALSE;
    database pgm;
    FILE *fd;
    char *cwd;
    static char buffer[MAXNAMLEN];
    string abspath = NULL;
    /* string relpath = NULL; */
    static char *tempfile = NULL;
    int err;

    if (! file_exists_p(file)) {
	user_warning("process_user_file", "Cannot open file : %s\n", file);
	return FALSE;
    }

    if (tempfile == NULL) {
	tempfile = tmpnam(NULL);
    }

    pgm = db_get_current_workspace();

    /* the absolute path of file is calculated */
    abspath = strdup((*file == '/') ? file : 
		     concatenate(get_cwd(), "/", file, NULL));
    /* Well, databases are even more relocatable if the absolute name of 
       the user file names are stored in the database
       */
    /*
    relpath = strdup((*file == '/') ? file : 
		     concatenate("../", file, NULL));
		     */

    /* the new file is registered in the database */
    user_log("Registering file %s\n", file);
    /* FI: two problems here
       - the successive calls to DB_PUT_FILE_RESOURCE erase each other...
       - the wiring of the database_name prevents mv of the database (fixed)
       */
    /* DB_PUT_FILE_RESOURCE(DBR_USER_FILE, database_name(pgm), abspath); */
    DB_PUT_FILE_RESOURCE(DBR_USER_FILE, "", abspath);

    /* the new file is splitted according to Fortran standard */
    user_log("Splitting file    %s\n", file);
    cwd = strdup(get_cwd());
    /* chdir(database_directory(pgm)); */
    chdir(db_get_current_workspace_directory());
    /* reverse sort because the list of modules is reversed later */
    /* if two modules have the same name, the first splitted wins
       and the other one is hidden by the call to "sed" since
       fsplit gives it a zzz00n.f name */
    /* Let's hope no user module is called zzz???.f */
    err = safe_system_no_abort
	(concatenate("trap 'exit 123' 2;",
		     "pips-split ", abspath,
		     "| sed -e /zzz[0-9][0-9][0-9].f/d | sort -r > ",
		     tempfile, "; /bin/rm -f zzz???.f", NULL));

    /* Go back unconditionnally to regular directory for execution
     * or you are heading for trouble when the database is closed
     */
    chdir(cwd);
    free(cwd);

    if(err==123) {
	/* Are we or not allowed to use user_error() in top-level? */
	/* 
	user_error("process_user_file",
		   "File splitting interrupted by control-C\n");
		   */
	user_warning("process_user_file",
		     "File splitting interrupted by control-C\n");
	return FALSE;
    }
    else if(err!=0) {
	pips_error("process_user_file",
		   "Unexpected return code from pips-split: %d\n", err);
    }

    /* the newly created module files are registered in the database */
    fd = safe_fopen(tempfile, "r");
    while (fscanf(fd, "%s", buffer) != EOF) {
	char *modname;
	/* char *modfullfilename; */
	char * modrelfilename = NULL;

	success_p = TRUE;
	/*
	modfullfilename = strdup(concatenate(database_directory(pgm), 
					     "/", buffer, NULL)); */
	modrelfilename = strdup(buffer);

	*strchr(buffer, '.') = '\0';
	(void) strupper(buffer, buffer);
	modname = strdup(buffer);

	user_log("  Module         %s\n", modname);

        /* Apply a cleaning procedure on each module: */
        cwd = strdup(get_cwd());
        chdir(db_get_current_workspace_directory());
        err = safe_system_no_abort(concatenate("trap 'exit 123' 2;",
                                               "pips-process-module ",
                                               modrelfilename,
                                               NULL));
        chdir(cwd);
        free(cwd);

        if(err==123) {
           user_warning("process_user_file",
                        "pips-process-module interrupted by control-C\n");
           return FALSE;
    }
        else if(err!=0) {
           pips_error("process_user_file",
                      "Unexpected return code from pips-process-module: %d\n", err);
        }
        
    
	if(DB_PUT_NEW_FILE_RESOURCE(DBR_SOURCE_FILE, 
				    modname, modrelfilename)
	   == resource_undefined) {
	    user_warning("process_user_file", 
			 "Two source codes for module %s."
			 "The second occurence in file %s is ignored\n",
			 modname, file);
	}
    }
    safe_fclose(fd, tempfile);

    unlink(tempfile);
    tempfile = NULL;

    if(!success_p) {
	user_warning("", "No module was found when splitting file %s.\n",
		     abspath);
    }

    return success_p;
}
