/* 
 * $Id$
 */

#include <unistd.h>
#include <stdlib.h>

#include <stdio.h>
#include <string.h>
#include <limits.h>
#include <errno.h>

#include <sys/stat.h>
#include <sys/param.h>

#include <dirent.h>
#include <setjmp.h>

#include "rxposix.h"

#include "genC.h"
#include "misc.h"

/* @return a file descriptor. 
 */
FILE * check_fopen(char * file, char * mode)
{
    FILE * fd = fopen(file, mode);
    if (fd==(FILE*)NULL)
    {
	pips_user_warning("fopen failed on file \"%s\" (mode \"%s\")\n%s\n",
			  file, mode, strerror(errno));
    }
    return fd;
}

FILE * safe_fopen(char *filename, char *what)
{
    FILE * f;
    if((f = fopen( filename, what)) == (FILE *) NULL) {
	pips_error("safe_fopen","fopen failed on file %s\n%s\n",
		   filename, strerror(errno));
    }
    return(f);
}

int 
safe_fclose(FILE * stream, char *filename)
{
    if(fclose(stream) == EOF) {
	if(errno==ENOSPC)
	    user_irrecoverable_error("safe_fclose",
				     "fclose failed on file %s (%s)\n",
				     filename,
				     strerror(errno));
	else
	    pips_error("safe_fclose","fclose failed on file %s (%s)\n",
		       filename,
		       strerror(errno));
    }
    return(0);
}

int 
safe_fflush(FILE * stream, char *filename)
{
    if(fflush(stream) == EOF) {
	pips_error("safe_fflush","fflush failed on file %s (%s)\n",
		   filename,
		   strerror(errno));
    }
    return(0);
}

FILE * 
safe_freopen(char *filename, char *what, FILE * stream)
{
    FILE *f;

    if((f = freopen( filename, what, stream)) == (FILE *) NULL) {
	pips_error("safe_freopen","freopen failed on file %s (%s)\n",
		   filename,
		   strerror(errno));
    }
    return(f);
}

int 
safe_fseek(FILE * stream, long int offset, int wherefrom, char *filename)
{
    if( fseek( stream, offset, wherefrom) != 0) {
	pips_error("safe_fseek","fseek failed on file %s (%s)\n",
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
	pips_error("safe_ftell","ftell failed on file %s (%s)\n",
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
	pips_error("safe_rewind","rewind failed on file %s (%s)\n",
		   filename,
		   strerror(errno));
    }
}

int 
safe_fgetc(FILE * stream, char *filename)
{
    int value;
    if((value = fgetc( stream)) == EOF) {
	pips_error("safe_fgetc","fgetc failed on file %s (%s)\n",
		   filename,
		   strerror(errno));
    }
    return(value);
}

int 
safe_getc(FILE * stream, char *filename)
{
    int value;
    if((value = getc( stream)) == EOF ) {
	pips_error("safe_getc","getc failed on file %s (%s)\n",
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
	pips_error("safe_fgets","gets failed on file %s (%s)\n",
		   filename,
		   strerror(errno));
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
		   strerror(errno));
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
		   strerror(errno));
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
		   strerror(errno));
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
	pips_error("safe_fread","fread failed on file %s (%s)\n",
		   filename,
		   strerror(errno));
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
	pips_error("safe_fwrite","fwrite failed on file %s (%s)\n",
		   filename,
		   strerror(errno));
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
safe_list_files_in_directory(
    gen_array_t files, /* an allocated array */
    char *dir, /* the directory we're interested in */
    char *re, /* regular expression */
    bool (*file_name_predicate)(string) /* condition to list a file */)
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
    char * dir,
    char * re,
    bool (* file_name_predicate)(char *))
{
    int return_code = safe_list_files_in_directory
	(files, dir, re, file_name_predicate);
    
    if (return_code == -1)
	user_error("list_files_in_directory",
		   "opendir() failed on directory \"%s\", %s.\n",
		   dir,  strerror(errno));
}

bool 
directory_exists_p(char * name)
{
    struct stat buf;
    return (stat(name, &buf) == 0) && S_ISDIR(buf.st_mode);
}

bool 
file_exists_p(char * name)
{
    struct stat buf;
    return (stat(name, &buf) == 0) && S_ISREG(buf.st_mode);
}

#define COLON ':'
/* returns the allocated nth path in colon-separated path list.
 */
static string
nth_path(char *path_list, int n)
{
    int len=0,i;
    char *result;
    while (*path_list && n>0)
	if (*path_list++==COLON) n--;
    if (!*path_list) return (string) NULL;
    while (path_list[len] && path_list[len]!=COLON) len++;
    result = (string) malloc(len+1);
    pips_assert("malloc ok", result);
    for (i=0; i<len; i++) 
	result[i]=path_list[i];
    result[len]='\0';
    return result;
}

static char *
relative_name_if_necessary(char *name)
{
    if (name[0]=='/' || name[0]=='.') return strdup(name);
    else return strdup(concatenate("./", name, 0));
}

/* returns an allocated string pointing to the file, possibly 
 * with an additional path taken from colon-separated dir_path.
 * returns NULL if no file was found.
 */
char *
find_file_in_directories(char *file_name, char *dir_path)
{
    char *path;
    int n=0;
    pips_assert("some file name", file_name);

    if (file_exists_p(file_name))
	return relative_name_if_necessary(file_name);

    if (!dir_path || file_name[0]=='/')
	return (string) NULL;

    /* looks for the file with an additionnal path ahead.
     */
    while ((path=nth_path(dir_path, n++))) 
    {
	char *name = strdup(concatenate(path, "/", file_name, 0)), *res=NULL;
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
    bool success = TRUE;

    if (directory_exists_p(name)) {
	pips_internal_error("existing directory : %s\n", name);
    }

    if (mkdir(name, 0777) == -1) {
	pips_user_warning("cannot create directory : %s (%s)\n",
			  name, strerror(errno));
	success = FALSE;
    }

    return success;
}

bool 
purge_directory(char *name)
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
    bool first = TRUE;
    int c, i;
    while ((c=getc(in))!=EOF)
    {
	if (first && (!but_comments || (c!='C' && c!='c' && c!='*' && c!='!')))
	{
	    for (i=0; i<margin; i++) 
		if (putc(' ', out)==EOF)
		    pips_internal_error("append failed");
	    first = FALSE;
	}
	if (c=='\n') 
	    first = TRUE;
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


/* Some OS do not define basename and dirname. Others like DEC OSF1
   do. So define them and use another name for them: */
/* /some/path/to/file.suffix -> file
 */
char *
pips_basename(char *fullpath, char *suffix)
{
    int len = strlen(fullpath)-1, i, j;
    char *result;
    if (suffix) /* drop the suffix */
    {
	int ls = strlen(suffix)-1, le = len;
	while (suffix[ls]==fullpath[le] && ls>=0 && le>=0) ls--, le--;
	if (ls<0) /* ok */ len=le;
    }
    for (i=len; i>=0; i--) if (fullpath[i]=='/') break;
    /* fullpath[i+1:len] */
    result = (char*) malloc(sizeof(char)*(len-i+1));
    for (i++, j=0; i<=len; i++, j++) 
	result[j] = fullpath[i];
    result[j++] = '\0';
    return result;
}

/* /some/path/to/file.suffix -> /some/path/to
 */
char *
pips_dirname(char *fullpath)
{
    char *result = strdup(fullpath);
    int len = strlen(result);
    while (result[--len]!='/' && len>=0);
    result[len] = '\0';
    return result;
}

void
safe_unlink(char *file_name)
{
    if (unlink(file_name)) 
    {
	perror("[safe_unlink] ");
	pips_internal_error("unlink %s failed\n", file_name);
    }
}

void
safe_symlink(char *topath, char *frompath)
{
    if (symlink(topath, frompath))
    {
	perror("[safe_symlink] ");
	pips_internal_error("symlink(%s, %s) failed\n", topath, frompath);
    }
}

void
safe_link(char *topath, char *frompath)
{
    if (link(frompath, topath))
    {
	perror("[safe_link] ");
	pips_internal_error("link(%s,%s) failed\n", frompath, topath);
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
	pips_internal_error("popen failed: %s\n", what);
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
    return safe_system_output(concatenate("echo ", what, 0));
}

/* SunOS forgets to declare this one.
 */
/* extern char * mktemp(char *); */

/* @return a new temporary file name, starting with "prefix".
 * the nme is freshly allocated.
 */
char * 
safe_new_tmp_file(char * prefix)
{
    string name = strdup(concatenate(prefix, ".XXXXXX", NULL));
    return mkstemp(name);
}
