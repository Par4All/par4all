/* $RCSfile: file.c,v $ (version $Revision$)
 * $Date: 1997/01/05 22:07:58 $, 
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
/* #include <sys/types.h> */
/* #include <sys/dirent.h> */
#include <sys/stat.h>
#include <sys/param.h>

#ifndef __USE_BSD
#define __USE_BSD
#endif
#include <dirent.h>
#include <setjmp.h>

#include "genC.h"
#include "misc.h"

/* hmmm. ???
 */
extern char *re_comp();
extern int re_exec();

/* Should be in stdlib.h or errno.h: */
extern char * sys_errlist[];

FILE * safe_fopen( filename, what)
char * filename, * what;
{
    FILE * f;

    if((f = fopen( filename, what)) == (FILE *) NULL) {
	pips_error("safe_fopen","fopen failed on file %s\n%s\n",
		   filename, sys_errlist[errno]);
    }
    return(f);
}

int safe_fclose( stream, filename)
FILE * stream;
char * filename;
{
	if(fclose(stream) == EOF) {
	  if(errno==ERNOSPC)
	    user_error("safe_fclose","fclose failed on file %s (%s)\n",
		       filename,
		       sys_errlist[errno]);
	  else
	    pips_error("safe_fclose","fclose failed on file %s (%s)\n",
		       filename,
		       sys_errlist[errno]);
	}
	return(0);
}

int safe_fflush( stream, filename)
FILE * stream;
char * filename;
{
	if(fflush(stream) == EOF) {
	    pips_error("safe_fflush","fflush failed on file %s (%s)\n",
		       filename,
		       sys_errlist[errno]);
	}
	return(0);
}

FILE * safe_freopen( filename, what, stream)
char * filename, * what;
FILE * stream;
{
    FILE *f;

    if((f = freopen( filename, what, stream)) == (FILE *) NULL) {
	pips_error("safe_freopen","freopen failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(f);
}

int safe_fseek( stream, offset, wherefrom, filename)
FILE * stream;
long int offset;
int wherefrom;
char * filename;
{
    if( fseek( stream, offset, wherefrom) != 0) {
	pips_error("safe_fseek","fseek failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(0);
}

long int safe_ftell( stream, filename)
FILE * stream;
char * filename;
{
    long int pt;
    pt = ftell( stream);
    if((pt == -1L) && (errno != 0)) {
	pips_error("safe_ftell","ftell failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(pt);
}

void safe_rewind( stream, filename)
FILE * stream;
char * filename;
{
    rewind( stream );
    if(errno != 0) {
	pips_error("safe_rewind","rewind failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
}

int safe_fgetc( stream, filename)
FILE * stream;
char * filename;
{
    int value;
    
    if((value = fgetc( stream)) == EOF) {
	pips_error("safe_fgetc","fgetc failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(value);
}

int safe_getc( stream, filename)
FILE * stream;
char * filename;
{
    int value;
    
    if((value = getc( stream)) == EOF ) {
	pips_error("safe_getc","getc failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(value);
}

char * safe_fgets( s, n, stream, filename)
char * s, * filename;
int n;
FILE * stream;
{
    if (fgets( s, n, stream) == (char *) NULL) {
	pips_error("safe_fgets","gets failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(s);
}

int safe_fputc( c, stream, filename)
char c;
FILE * stream;
char * filename;
{
    if(fputc( c, stream) == EOF) {
	pips_error("safe_fputc","fputc failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(c);
}

int safe_putc( c, stream, filename)
char c;
FILE * stream;
char * filename;
{
    if(putc( c, stream) == EOF) {
	pips_error("safe_putc","putc failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(c);
}

int safe_fputs( s, stream, filename)
char * s, * filename;
FILE * stream;
{
    if(fputs( s, stream) == EOF) {
	pips_error("safe_fputs","fputs failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(1);
}

int safe_fread( ptr, element_size, count, stream, filename)
char * ptr, * filename;
int element_size;
int count;
FILE * stream;
{
    if(fread(ptr, element_size, count, stream) != count) {
	pips_error("safe_fread","fread failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(count);
}

int safe_fwrite( ptr, element_size, count, stream, filename)
char * ptr, * filename;
int element_size;
int count;
FILE * stream;
{
    if(fwrite(ptr, element_size, count, stream) != count) {
	pips_error("safe_fwrite","fwrite failed on file %s (%s)\n",
		   filename,
		   sys_errlist[errno]);
    }
    return(count);
}

/* returns a sorted arg list of files matching regular expression re
  in directory 'dir' and with file_name_predicate() returning TRUE on
  the file name (for example use directory_exists_p to select
  directories, of file_exists_p to select regular files).  re has the
  ed syntax.

  Return 0 on success, -1 on directory openning error.
  */
int
safe_list_files_in_directory(int * pargc,
                             char * argv[],
                             char * dir,
                             char * re,
                             bool (* file_name_predicate)(char *))
{
   char complete_file_name[MAXNAMLEN + 1];
   list dir_list = NIL;
   DIR * dirp;
   struct dirent * dp;
   char * re_comp_message;
   

   pips_assert("safe_list_files_in_directory", 
               strcmp(dir, "") != 0);

   re_comp_message = re_comp(re);
   if (re_comp_message != NULL)
      user_error("list_files_in_directory",
                 "re_comp() failed to compile \"%s\", %s.\n",
                 re, re_comp_message);
   else {
      dirp = opendir(dir);

      if (dirp != NULL) {
         while((dp = readdir(dirp)) != NULL) {
            if (re_exec(dp->d_name) == 1) {
               (void) sprintf(complete_file_name, "%s/%s", dir, dp->d_name);
               if (file_name_predicate(complete_file_name))
                  dir_list = CONS(STRING, strdup(dp->d_name), dir_list);
            }
         }
         closedir(dirp);
      }
      else
         return -1;
   }

   /* Now sort the file list: */
   list_to_arg(dir_list, pargc, argv);
   args_sort(*pargc, argv);

   return 0;
}


/* The same as the previous safe_list_files_in_directory() but with no
   return code and a call to user error if it cannot open the
   directory.
   */
void
list_files_in_directory(int * pargc,
                        char * argv[],
                        char * dir,
                        char * re,
                        bool (* file_name_predicate)(char *))
{
   int return_code = safe_list_files_in_directory(pargc,
                                                  argv,
                                                  dir,
                                                  re,
                                                  file_name_predicate);
   
   if (return_code == -1)
      user_error("list_files_in_directory",
                 "opendir() failed on directory \"%s\", %s.\n",
                 dir,  sys_errlist[errno]);
}


bool directory_exists_p(name)
char *name;
{
    static struct stat buf;
    return (stat(name, &buf) == 0) && S_ISDIR(buf.st_mode);
}

bool file_exists_p(name)
char *name;
{
    static struct stat buf;
    return (stat(name, &buf) == 0) && S_ISREG(buf.st_mode);
}


bool create_directory(name)
char *name;
{
    bool success;

    if (directory_exists_p(name)) {
	pips_error("create_directory", "existing directory : %s\n", name);
    }

    if (mkdir(name, 0777) == -1) {
	user_warning("create_directory", "cannot create directory : %s\n",
		     name);
	success = FALSE;
    }
    else {
	success = TRUE;
    }
    return success;
}

bool purge_directory(name)
char *name;
{
    bool success = TRUE;

    if (directory_exists_p(name)) {
	if(system(concatenate("/bin/rm -r ", name, (char*) NULL))) {
	    /* FI: this warning should be emitted by a higher-level
	     * routine!
	     */
	    user_warning("purge_directory",
			 "cannot purge directory %s. Check owner rights\n",
			 name);
	    success = FALSE;
	}
	else {
	    success = TRUE;
	}
    }
    else {
	/* Well, it's purged if it does not exist... */
	success = TRUE;
    }

    return success;
}

/* returns the current working directory
*/
/* I couldn't find any header declaring getwd... FC.
 */
extern char * getwd(char *);
char *get_cwd()
{
    static char cwd[MAXPATHLEN];

    return(getwd(cwd));
}
