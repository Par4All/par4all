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
#ifdef HAVE_CONFIG_H
    #include "pips_config.h"
#endif

#include <unistd.h>
#include <stdlib.h>

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <errno.h>
#include <ctype.h>

#include <sys/stat.h>
#include <sys/param.h>
#include <sys/types.h>
#include <sys/wait.h>

#include <dirent.h>
#include <regex.h>

#include "genC.h"
#include "misc.h"
#include "properties.h"

/* @return a file descriptor.
 */
FILE * check_fopen(const char * file, const char * mode)
{
  FILE * fd = fopen(file, mode);
  if (fd==(FILE*)NULL)
  {
    pips_user_warning("fopen failed on file \"%s\" (mode \"%s\")\n%s\n",
                      file, mode, strerror(errno));
  }
  return fd;
}

FILE * safe_fopen(const char *filename, const char *what)
{
  FILE * f;
  if((f = fopen( filename, what)) == (FILE *) NULL) {
    pips_internal_error("fopen failed on file %s\n%s",
                        filename, strerror(errno));
  }
  return f;
}

int safe_fclose(FILE * stream, const char *filename)
{
  if(fclose(stream) == EOF) {
    if(errno==ENOSPC)
	    user_irrecoverable_error("safe_fclose",
                               "fclose failed on file %s (%s)\n",
                               filename,
                               strerror(errno));
    else
	    pips_internal_error("fclose failed on file %s (%s)",
                          filename,
                          strerror(errno));
  }
  return 0;
}

int safe_fflush(FILE * stream, char *filename)
{
  if(fflush(stream) == EOF) {
    pips_internal_error("fflush failed on file %s (%s)",
                        filename,
                        strerror(errno));
  }
  return 0;
}

FILE * safe_freopen(char *filename, char *what, FILE * stream)
{
  FILE *f;

  if((f = freopen( filename, what, stream)) == (FILE *) NULL) {
    pips_internal_error("freopen failed on file %s (%s)",
                        filename,
                        strerror(errno));
  }
  return f;
}

int
safe_fseek(FILE * stream, long int offset, int wherefrom, char *filename)
{
    if( fseek( stream, offset, wherefrom) != 0) {
	pips_internal_error("fseek failed on file %s (%s)",
		   filename,
		   strerror(errno));
    }
    return(0);
}

long int
safe_ftell(FILE * stream, char *filename)
{
    long int pt;
    pt = ftell( stream);
    if((pt == -1L) && (errno != 0)) {
	pips_internal_error("ftell failed on file %s (%s)",
		   filename,
		   strerror(errno));
    }
    return(pt);
}

void
safe_rewind(FILE * stream, char *filename)
{
    rewind( stream );
    if(errno != 0) {
	pips_internal_error("rewind failed on file %s (%s)",
		   filename,
		   strerror(errno));
    }
}

int
safe_fgetc(FILE * stream, char *filename)
{
    int value;
    if((value = fgetc( stream)) == EOF) {
	pips_internal_error("fgetc failed on file %s (%s)",
		   filename, strerror(errno));
    }
    return(value);
}

int
safe_getc(FILE * stream, char *filename)
{
    int value;
    if((value = getc( stream)) == EOF ) {
	pips_internal_error("getc failed on file %s (%s)",
		   filename,
		   strerror(errno));
    }
    return(value);
}

char *
safe_fgets(s, n, stream, filename)
char * s, * filename;
int n;
FILE * stream;
{
    if (fgets( s, n, stream) == (char *) NULL) {
	pips_internal_error("gets failed on file %s (%s)",
		   filename, strerror(errno));
    }
    return(s);
}

int safe_fputc( c, stream, filename)
char c;
FILE * stream;
char * filename;
{
    if(fputc( c, stream) == EOF) {
	pips_internal_error("fputc failed on file %s (%s)",
		   filename, strerror(errno));
    }
    return(c);
}

int safe_putc( c, stream, filename)
char c;
FILE * stream;
char * filename;
{
    if(putc( c, stream) == EOF) {
	pips_internal_error("putc failed on file %s (%s)",
		   filename, strerror(errno));
    }
    return(c);
}

int safe_fputs( s, stream, filename)
char * s, * filename;
FILE * stream;
{
    if(fputs( s, stream) == EOF) {
	pips_internal_error("fputs failed on file %s (%s)",
		   filename, strerror(errno));
    }
    return(1);
}

int safe_fread( ptr, element_size, count, stream, filename)
char * ptr, * filename;
int element_size;
int count;
FILE * stream;
{
    if(((int)fread(ptr, element_size, count, stream)) != count) {
	pips_internal_error("fread failed on file %s (%s)",
		   filename, strerror(errno));
    }
    return(count);
}

int safe_fwrite( ptr, element_size, count, stream, filename)
char * ptr, * filename;
int element_size;
int count;
FILE * stream;
{
    if(((int)fwrite(ptr, element_size, count, stream)) != count) {
	pips_internal_error("fwrite failed on file %s (%s)",
		   filename,
		   strerror(errno));
    }
    return(count);
}

/* returns a sorted arg list of files matching regular expression re
  in directory 'dir' and with file_name_predicate() returning true on
  the file name (for example use directory_exists_p to select
  directories, of file_exists_p to select regular files).  re has the
  ed syntax.

  Return 0 on success, -1 on directory openning error.
  */
int
safe_list_files_in_directory(
    gen_array_t files, /* an allocated array */
    string dir, /* the directory we're interested in */
    string re, /* regular expression */
    bool (*file_name_predicate)(const char *) /* condition to list a file */)
{
    DIR * dirp;
    struct dirent * dp;
    int index = 0;

    pips_assert("some dir", strcmp(dir, "") != 0);

    dirp = opendir(dir);

    if (dirp != NULL)
    {
	regex_t re_compiled;

	if (regcomp(&re_compiled, re, REG_ICASE))
	    pips_user_error("regcomp() failed to compile \"%s\".\n", re);

	while((dp = readdir(dirp)) != NULL) {
	    if (!regexec(&re_compiled, dp->d_name, 0, NULL, 0))
	    {
		char * full_file_name =
		    strdup(concatenate(dir, "/", dp->d_name, NULL));
		if (file_name_predicate(full_file_name))
		    gen_array_dupaddto(files, index++, dp->d_name);
		free(full_file_name);
            }
	}

	regfree(&re_compiled);

	closedir(dirp);
    }
    else
	return -1;

   gen_array_sort(files);
   return 0;
}


/* The same as the previous safe_list_files_in_directory() but with no
   return code and a call to user error if it cannot open the
   directory.
   */
void
list_files_in_directory(
    gen_array_t files,
    string dir,
    string re,
    bool (* file_name_predicate)(const char *))
{
    int return_code = safe_list_files_in_directory
	(files, dir, re, file_name_predicate);

    if (return_code == -1)
	user_error("list_files_in_directory",
		   "opendir() failed on directory \"%s\", %s.\n",
		   dir,  strerror(errno));
}

bool
directory_exists_p(const char * name)
{
    struct stat buf;
    return (stat(name, &buf) == 0) && S_ISDIR(buf.st_mode);
}

bool
file_exists_p(const char * name)
{
    struct stat buf;
    return (stat(name, &buf) == 0) && S_ISREG(buf.st_mode);
}

/* protect a string, for example for use in a system call
 */
char *
strescape (const char *source)
{
    size_t new_size = 1; // for \0
    for(const char *iter=source;*iter;++iter,++new_size)
        if(!isalnum(*iter)&&*iter!='_') ++new_size;
    char *escaped = malloc(sizeof(*escaped)*new_size);
    char *eiter = escaped;
    for(const char *iter=source;*iter;++iter,++eiter) {
        if(!isalnum(*iter)&&*iter!='_') *eiter++='\\';
        *eiter=*iter;
    }
    *eiter=0;
    return escaped;
}

#define COLON ':'
/* Returns the allocated nth path from colon-separated path string.

   @param path_list the string that contains a colon-separated path

   @param n the n-th instance to extract

   @return an allocated string with the n-th part name

   If the path is empty or if n is out-of-bound, NULL is returned.
   The resulting string is *not*escaped, and can contain spaces
*/
string
nth_path(const char * path_list, int n)
{
  int len;

  if (path_list == NULL)
    return NULL;

  /* Find the n-th part: */
  while (*path_list && n > 0)
    if (*path_list++ == COLON)
      n--;

  if (!*path_list)
    /* Out-of-bound... */
    return NULL;

  /* Compute the length up to the COLON or the end of string: */
  for(len = 0; path_list[len] && path_list[len] != COLON; len++)
    ;

  char *unescaped =  strndup(path_list, len);
  return unescaped;
}


static char *
relative_name_if_necessary(const char * name)
{
    if (name[0]=='/' || name[0]=='.') return strdup(name);
    else return strdup(concatenate("./", name, NULL));
}

/* returns an allocated string pointing to the file, possibly
 * with an additional path taken from colon-separated dir_path.
 * returns NULL if no file was found.
 */
char *
find_file_in_directories(const char *file_name, const char *dir_path)
{
    char *path;
    int n=0;
    pips_assert("some file name", file_name);

    if (file_exists_p(file_name))
	return relative_name_if_necessary(file_name);

    if (!dir_path || file_name[0]=='/')
	return (string) NULL;

    /* looks for the file with an additional path ahead.
     */
    while ((path=nth_path(dir_path, n++)))
    {
	char *name = strdup(concatenate(path, "/", file_name, NULL)),
	  *res=NULL;
	free(path);
	if (file_exists_p(name))
	    res = relative_name_if_necessary(name);
	free(name);
	if (res) return res;
    }

    return (string) NULL;
}

bool
file_readable_p(char * name)
{
    struct stat buf;
    return !stat(name, &buf) && (S_IRUSR & buf.st_mode);
}

bool
create_directory(char *name)
{
    bool success = true;

    if (directory_exists_p(name)) {
	pips_internal_error("existing directory : %s", name);
    }

    if (mkdir(name, 0777) == -1) {
	pips_user_warning("cannot create directory : %s (%s)\n",
			  name, strerror(errno));
	success = false;
    }

    return success;
}

bool
purge_directory(char *name)
{
    bool success = true;

    if (directory_exists_p(name)) {
	if(system(concatenate("/bin/rm -r ", name, (char*) NULL))) {
	    /* FI: this warning should be emitted by a higher-level
	     * routine!
	     */
	    user_warning("purge_directory",
			 "cannot purge directory %s. Check owner rights\n",
			 name);
	    success = false;
	}
	else {
	    success = true;
	}
    }
    else {
	/* Well, it's purged if it does not exist... */
	success = true;
    }

    return success;
}

#if !defined(PATH_MAX)
#if defined(_POSIX_PATH_MAX)
#define PATH_MAX _POSIX_PATH_MAX
#else
#define PATH_MAX 255
#endif
#endif

/* returns the current working directory name.
 */
char *
get_cwd(void)
{
    static char cwd[PATH_MAX]; /* argh */
    cwd[PATH_MAX-1] = '\0';
    return getcwd(cwd, PATH_MAX);
}

/* returns the allocated line read, whatever its length.
 * returns NULL on EOF. also some asserts. FC 09/97.
 */
char *
safe_readline(FILE * file)
{
    int i=0, size = 20, c;
    char * buf = (char*) malloc(sizeof(char)*size), * res;
    pips_assert("malloc ok", buf);
    while((c=getc(file)) && c!=EOF && c!='\n')
    {
	if (i==size-1) /* larger for trailing '\0' */
	{
	    size+=20; buf = (char*) realloc((char*) buf, sizeof(char)*size);
	    pips_assert("realloc ok", buf);
	}
	buf[i++] = (char) c;
    }
    if (c==EOF && i==0) { res = NULL; free(buf); }
    else { buf[i++] = '\0'; res = strdup(buf); free(buf); }
    return res;
}

/* returns the file as an allocated string.
 * \n is dropped at the time.
 */
char *
safe_readfile(FILE * file)
{
    char * line, * buf=NULL;
    while ((line=safe_readline(file)))
    {
	if (buf)
	{
	    buf = (char*)
		realloc(buf, sizeof(char)*(strlen(buf)+strlen(line)+2));
	    strcat(buf, " ");
	    strcat(buf, line);
	    free(line);
	}
	else buf = line;
    }
    return buf;
}

void
safe_cat(FILE * out, FILE * in)
{
    int c;
    while ((c=getc(in))!=EOF)
	if (putc(c, out)==EOF)
	    pips_internal_error("cat failed");
}

void
safe_append(
    FILE * out        /* where to output the file content */,
    char *file        /* the content of which is appended */,
    int margin        /* number of spaces for shifting */ ,
    bool but_comments /* do not shift F77 comment lines */)
{
    FILE * in = safe_fopen(file, "r");
    bool first = true;
    int c, i;
    while ((c=getc(in))!=EOF)
    {
	if (first && (!but_comments || (c!='C' && c!='c' && c!='*' && c!='!')))
	{
	    for (i=0; i<margin; i++)
		if (putc(' ', out)==EOF)
		    pips_internal_error("append failed");
	    first = false;
	}
	if (c=='\n')
	    first = true;
	if (putc(c, out)==EOF)
	    pips_internal_error("append failed");
    }
    safe_fclose(in, file);
}

void
safe_copy(char *source, char *target)
{
    FILE * in, * out;
    in = safe_fopen(source, "r");
    out = safe_fopen(target, "w");
    safe_cat(out, in);
    safe_fclose(out, target);
    safe_fclose(in, source);
}

/* Display a file through $PIPS_MORE (or $PAGER) if stdout is a TTY,
   on stdout otherwise.

   Return false if the file couldn't be displayed.
 */
int
safe_display(char *fname)
{
	if (!file_exists_p(fname))
	{
		pips_user_error("View file \"%s\" not found\n", fname);
		return 0;
	}

	if (isatty(fileno(stdout))) {
		int pgpid = fork();
		if (pgpid)
		{
			int status;
			waitpid(pgpid, &status, 0);
			if (WIFEXITED(status) && WEXITSTATUS(status) == 0)
				return 1;
			else
				return 0;
		}
		else
		{
			char *pager = getenv("PIPS_MORE");
			if (!pager)
				pager = getenv("PAGER");
			if (!pager)
				pager = "more";
			execlp(pager, pager, fname, NULL);
			pips_internal_error("running %s %s: %s",
					    pager, fname, strerror(errno));
			exit(127);
		}
	} else {
		FILE * in = safe_fopen(fname, "r");
		safe_cat(stdout, in);
		safe_fclose(in, fname);
		return 1;
	}
}


/* Some OS do not define basename and dirname. Others like DEC OSF1
   do. So define them and use another name for them:

   /some/path/to/file.suffix -> file

   This may create conflicting file names, when the same source
   filename is used in different subdirectory as in:

   create foo mod.c src/mod.c src/init/mod.c src/close/mod.c

   To avoid the problem a larger part of the access path should be
   preserved. This can be done by substituting / by another character.
 */
char * pips_filename(char *fullpath, char *suffix, bool short_p)
{
    int len = strlen(fullpath)-1, i, j;
    char *result;

    if (suffix) /* Drop the suffix */
    {
	int ls = strlen(suffix)-1, le = len;
	while (suffix[ls]==fullpath[le] && ls>=0 && le>=0) ls--, le--;
	if (ls<0) /* ok */ len=le;
    }

    if(short_p) {
      /* Keep the basename only */
      for (i=len; i>=0; i--) if (fullpath[i]=='/') break;
      /* fullpath[i+1:len] */
      result = (char*) malloc(sizeof(char)*(len-i+1));
      for (i++, j=0; i<=len; i++, j++)
	result[j] = fullpath[i];
      result[j++] = '\0';
    }
    else {
    /* Or substitute slashes by a neutral character */
      char * cc;

      if(fullpath[0]=='.' && fullpath[1]=='/')
	result = strndup(fullpath+2, len-1);
      else
	result = strndup(fullpath, len+1);


#define SLASH_SUBSTITUTION_CHARACTER '_'

      for(cc=result; *cc!='\000'; cc++) {
	if(*cc=='/')
	  *cc = SLASH_SUBSTITUTION_CHARACTER;
      }
    }
    return result;
}

char * pips_basename(char *fullpath, char *suffix)
{
  return pips_filename(fullpath, suffix, true);
}

/* The source file name access path is shortened or not depending on
   the property. It is shorten if the name conflicts are not managed. */
char * pips_initial_filename(char *fullpath, char *suffix)
{
  return pips_filename(fullpath, suffix,
		       !get_bool_property("PREPROCESSOR_FILE_NAME_CONFLICT_HANDLING"));
}

/* /some/path/to/file.suffix -> /some/path/to
 */
char * pips_dirname(char *fullpath)
{
    char *result = strdup(fullpath);
    int len = strlen(result);
    while (result[--len]!='/' && len>=0);
    result[len] = '\0';
    return result;
}


/* Delete the given file.

   Throw a pips_internal_error() if it fails.
*/
void
safe_unlink(char *file_name)
{
    if (unlink(file_name))
    {
	perror("[safe_unlink] ");
	pips_internal_error("unlink %s failed", file_name);
    }
}

void
safe_symlink(char *topath, char *frompath)
{
    if (symlink(topath, frompath))
    {
	perror("[safe_symlink] ");
	pips_internal_error("symlink(%s, %s) failed", topath, frompath);
    }
}


/* Create a hard link to topath. That means that the file is accessible
   with the new name frompath too.

   Throw a pips_internal_error() if it fails.
*/
void
safe_link(char *topath, char *frompath)
{
    if (link(frompath, topath))
    {
	perror("[safe_link] ");
	pips_internal_error("link(%s,%s) failed", frompath, topath);
    }
}

/* attempt shell substitutions to what. returns NULL on errors.
 */
char *
safe_system_output(char * what)
{
    char * result;
    FILE * in;

    in = popen(what, "r");

    if (in==NULL) {
	perror("[safe_system_output] ");
	pips_internal_error("popen failed: %s", what);
    }

    result = safe_readfile(in);

    if (pclose(in)) {
	/* on failures, do not stop it anyway...
	 */
	perror("[safe_system_output] ");
	pips_user_warning("\n pclose failed: %s\n", what);
	if (result) free(result), result = NULL;
    }

    return result;
}

/* returns what after variable, command and file substitutions.
 * the returned string is newly allocated. it's NULL on errors.
 */
char *
safe_system_substitute(char * what)
{
    return safe_system_output(concatenate("echo ", what, NULL));
}

/* SunOS forgets to declare this one.
 */
/* extern char * mktemp(char *); */

/* @return a new temporary file name, starting with "prefix".
 * the name is freshly allocated.
 *
 * FI: mkstemp() is being deprecated and it returns an integer, usable as
 * file descriptor, not a character string.
 *
 */
char * safe_new_tmp_file(char * prefix)
{
  string name = strdup(concatenate(prefix, ".XXXXXX", NULL));
  int desc = mkstemp(name);
  pips_assert("could create temporary name", desc!=-1);
  return name;
}

/* utility to open configuration file, (read only!)
 * its name can be found using various ways
 * property and env can be NULL (and ignored)
 * if the file if not found a pips_error is generated
 * canonical_name should be a file name, not a path
 */
#define DEFAULT_CONFIG_DIR "etc"
#define CONFIG_DEFAULT_RIGHT "r"
FILE *
fopen_config(const char* canonical_name,
	     const char* cproperty,
	     const char* cenv)
{
  FILE * fconf;

  // try various combinaison :
  // pips property
  if (cproperty) {
    const char* sproperty = get_string_property(cproperty);
    if (sproperty && (fconf = fopen(sproperty, CONFIG_DEFAULT_RIGHT)))
      return fconf;
  }

  // then pips env var
  if (cenv) {
    string senv = getenv(cenv);
    if (senv && (fconf = fopen(senv, CONFIG_DEFAULT_RIGHT)))
      return fconf;
  }

  // then default, with PIPS_ROOT if set
  string pipsenv = getenv("PIPS_ROOT");
  string sdefault;
  if(pipsenv)
    sdefault =
      concatenate(pipsenv,"/" DEFAULT_CONFIG_DIR "/" , canonical_name, NULL);
  else
    sdefault = concatenate(CONFIG_DIR "/", canonical_name, NULL);

  return safe_fopen(sdefault, CONFIG_DEFAULT_RIGHT);
}
