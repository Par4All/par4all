#include <stdio.h>
extern int fclose();
extern int fprintf();
extern int fflush();
extern int fseek();
extern int rewind();
extern int fgetc();
extern int fputc();
extern int fputs();
extern int fread();
extern int fscanf();
extern int fwrite();
extern int system();
extern int unlink();
extern int _filbuf();
extern int _flsbuf();
#include <string.h>
extern char toupper(char c);
#include <errno.h>
#include <sys/types.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/param.h>

#include "genC.h"
#include "misc.h"

extern void exit();
extern char * sys_errlist[];
extern char *getwd();

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
		fprintf(stderr,"fclose failed on file %s\n", filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
	return(0);
}

int safe_fflush( stream, filename)
FILE * stream;
char * filename;
{
	if(fflush(stream) == EOF) {
		fprintf(stderr,"fflush failed on file %s\n", filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
	return(0);
}

FILE * safe_freopen( filename, what, stream)
char * filename, * what;
FILE * stream;
{
	FILE * f;

	if((f = freopen( filename, what, stream)) == (FILE *) NULL) {
		fprintf(stderr,"freopen failed on file %s\n", filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
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
		fprintf(stderr,"fseek failed on file %s\n", filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
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
		fprintf(stderr,"ftell failed on file %s\n",filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
	return(pt);
}

void safe_rewind( stream, filename)
FILE * stream;
char * filename;
{
	rewind( stream );
	if(errno != 0) {
		fprintf(stderr,"rewind failed on file %s\n",filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
}

int safe_fgetc( stream, filename)
FILE * stream;
char * filename;
{
	int value;

	if((value = fgetc( stream)) == EOF) {
		fprintf(stderr,"fgetc failed on file %s\n",filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
	return(value);
}

int safe_getc( stream, filename)
FILE * stream;
char * filename;
{
	int value;

	if((value = getc( stream)) == EOF ) {
	    (void) fprintf(stderr,"getc failed on file %s\n",filename);
	    (void) fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
	return(value);
}

char * safe_fgets( s, n, stream, filename)
char * s, * filename;
int n;
FILE * stream;
{
	if (fgets( s, n, stream) == (char *) NULL) {
		fprintf(stderr,"gets failed on file %s\n",filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
	return(s);
}

int safe_fputc( c, stream, filename)
char c;
FILE * stream;
char * filename;
{
	if(fputc( c, stream) == EOF) {
		fprintf(stderr,"fputc failed on file %s\n",filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
	return(c);
}

int safe_putc( c, stream, filename)
char c;
FILE * stream;
char * filename;
{
	if(putc( c, stream) == EOF) {
		fprintf(stderr,"putc failed on file %s\n",filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
	return(c);
}

int safe_fputs( s, stream, filename)
char * s, * filename;
FILE * stream;
{
	if(fputs( s, stream) == EOF) {
		fprintf(stderr,"fputs failed on file %s\n",filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
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
		fprintf(stderr,"fread failed on file %s\n",filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
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
		fprintf(stderr,"fwrite failed on file %s\n",filename);
		fprintf(stderr,"%s\n",sys_errlist[errno]);
		exit(1);
	}
	return(count);
}

/*
  returns a newgen list of files matching regular expression re
  in directory 'dir'. re has the ls syntax.
*/
list list_files_in_directory(dir, re)
char *dir;
char *re;
{
    FILE *fd;
    static char buffer[MAXNAMLEN];
    list *tl, hl;
    static char *tempfile = NULL;

    hl = NIL;
    tl=&hl;

    pips_assert("list_files_in_directory", 
		strcmp(dir, "") != 0);

    if (tempfile == NULL) {
	tempfile = tmpnam(NULL);
    }

    system(concatenate("cd ", dir, "; ls ", re, " >", tempfile, NULL));

    /* build a list from files listed in tempfile */
    fd = safe_fopen(tempfile, "r");
    while (fscanf(fd, "%s", buffer) != EOF) {
	*tl = CONS(STRING, strdup(buffer), NIL);
	tl = &CDR(*tl);
    }
    safe_fclose(fd, tempfile);

    /* Next two lines will remove tempfile.
     * Could be done later, as exiting the program.
     */
    unlink(tempfile);
    tempfile = NULL;

    return(hl);
}

/* 
  returns the list of files from directory 'dir' whose name matches the
  regular expression 'pattern' 
*/
void directory_list(dir, pargc, argv, pattern)
char *dir;
int *pargc;
char *argv[];
char *pattern;
{
    char *regex_error;

    struct dirent *wde;
    DIR *wds = opendir(dir);
    
    *pargc = 0;

    if (wds == NULL)
	return;

    if (pattern != NULL) {
	if ((regex_error = re_comp(pattern)) != NULL) {
	    pips_error("directory_list", "bad pattern: %s\n", 
		    regex_error);
	}
    }

    while ((wde = readdir(wds)) != NULL) {
	char *s = wde->d_name;

	if (*s != '.') {
	    if (pattern != NULL) {
		if (re_exec(s) == 1)
		    args_add(pargc, argv, strdup(s));
	    }
	    else {
		args_add(pargc, argv, strdup(s));
	    }
	}
    }

    closedir(wds);
}

bool directory_exists_p(name)
char *name;
{
    static struct stat buf;

    return((stat(name, &buf) == 0) && (buf.st_mode & S_IFDIR));
}

bool file_exists_p(name)
char *name;
{
    static struct stat buf;

    return((stat(name, &buf) == 0) && (buf.st_mode & S_IFREG));
}


void create_directory(name)
char *name;
{
    if (directory_exists_p(name)) {
	pips_error("create_directory", "existing directory : %s\n", name);
    }

    if (mkdir(name, 0777) == -1) {
	user_error("create_directory", "cannot create directory : %s\n", name);
    }
}

void purge_directory(name)
char *name;
{
    if (directory_exists_p(name)) {
	if(system(concatenate("rm -r ", name, (char*) NULL)))
	    user_error("purge_directory", "cannot purge directory : %s\n", name);
    }
}

/* 
returns the current working directory
*/
char *get_cwd()
{
    static char cwd[MAXPATHLEN];

    return(getwd(cwd));
}
