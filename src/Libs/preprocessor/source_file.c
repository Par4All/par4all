/* 
 * $Id$
 *
 * procedures used in both PIPS top-level, wpips and tpips.
 *
 * problems to use those procedures with wpips: show_message() and 
 * update_props() .
 *
 * $Log: source_file.c,v $
 * Revision 1.107  2003/09/04 11:16:33  coelho
 * test return status of fpp...
 *
 * Revision 1.106  2003/09/03 16:26:42  irigoin
 * return code added to signal error in compilation
 *
 * Revision 1.105  2003/08/18 08:55:13  coelho
 * setjmp out.
 *
 * Revision 1.104  2003/08/14 08:47:54  irigoin
 * Compatibility with LINUX for strdup() declaration
 *
 * Revision 1.103  2003/08/11 13:57:16  coelho
 * hop.
 *
 * Revision 1.102  2003/08/08 16:31:36  irigoin
 * Two currently useless static functions commented out to silence gcc
 *
 * Revision 1.101  2003/08/04 16:54:24  irigoin
 * Mostly, addition of code to process PIPS_SRCPATH. Plus some more error
 * detection and propagation.
 *
 * Revision 1.100  2003/08/01 06:00:25  irigoin
 * Intermediate version installed to let Production link
 *
 * Revision 1.99  2000/03/24 11:37:18  coelho
 * ordering dropped.
 *
 * Revision 1.98  1999/01/08 16:19:15  coelho
 * error in fsplit...
 *
 * Revision 1.97  1998/12/30 14:28:41  irigoin
 * Bug fix in process_thru_cpp()
 *
 * Revision 1.96  1998/12/24 11:10:11  coelho
 * unlink zzz files.
 *
 * Revision 1.95  1998/12/23 14:05:16  coelho
 * zzz -> ###
 *
 * Revision 1.94  1998/11/24 17:48:19  coelho
 * accept empty files. no big deal, just a warning.
 *
 * Revision 1.93  1998/07/10 20:42:56  irigoin
 * Handling of complex constant commented out
 *
 * Revision 1.92  1998/07/10 11:20:46  coelho
 * more comments about entries.
 *
 * Revision 1.91  1998/07/10 08:16:41  coelho
 * modules are put in the right order...
 *
 * Revision 1.90  1998/07/09 18:24:39  coelho
 * hack for entry: file names are preceded by their modules...
 *
 * Revision 1.89  1998/05/29 16:22:07  coelho
 * filter includes for bang comments and hollerith.
 *
 * Revision 1.88  1998/05/22 11:41:11  coelho
 * regex fixed.
 *
 * Revision 1.87  1998/05/22 11:37:33  coelho
 * bug--
 *
 * Revision 1.86  1998/05/14 11:55:15  coelho
 * abort quickly if cpp fails (stderr not empty... with gcc -E at least...)
 *
 * Revision 1.85  1998/04/14 21:22:19  coelho
 * linear.h
 *
 * Revision 1.84  1998/04/02 10:50:11  coelho
 * include cache dropped if include file was not found.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <dirent.h>

#include <sys/stat.h>
#include <ctype.h>
#include <unistd.h>
#include <errno.h>

#include "genC.h"
#include "linear.h"
#include "ri.h"
#include "database.h"
#include "makefile.h"

#include "misc.h"
#include "properties.h"

#include "ri-util.h"
#include "pipsdbm.h"

#include "constants.h"
#include "resources.h"
#include "phases.h"

#include "property.h"
/* #include "pipsmake.h" */
#include "pipsdbm.h"

#include "preprocessor.h"

extern char * strdup(const char *);
extern int putenv(char *); /* Supposedly in stdlib.h */

static bool dot_c_file_p(string);
static bool dot_f_file_p(string);
static bool dot_F_file_p(string);

#define skip_line_p(s) \
  ((*(s))=='\0' || (*(s))=='!' || (*(s))=='*' || (*(s))=='c' || (*(s))=='C')


/* Return a sorted arg list of workspace names. (For each name, there
   is a name.database directory in the current directory): */
void pips_get_workspace_list(gen_array_t array)
{
   int i, n;
   
   /* Find all directories with name ending with ".database": */

   list_files_in_directory(array, ".", "^.*\\.database$", directory_exists_p);
   n = gen_array_nitems(array);
   /* Remove the ".database": */
   for (i = 0; i < n; i++) {
      *strchr(gen_array_item(array, i), '.') = '\0';
   }
}


/* Select the true file with names ending in ".[fF]" and return a sorted
   arg list: */
void pips_get_fortran_list(gen_array_t array)
{
    list_files_in_directory(array, ".", "^.*\\.[fF]$", file_exists_p);
}


/* Return the path of an HPFC file name relative to the current PIPS
   directory. Can be freed by the caller. */
string hpfc_generate_path_name_of_file_name(string file_name)
{
    string dir_name = db_get_current_workspace_directory(),
	name = strdup(concatenate(
	    dir_name, "/", HPFC_COMPILED_FILE_DIR, "/", file_name, 0));
    free(dir_name);
    return name;
}


int hpfc_get_file_list(gen_array_t file_names,
		       char ** hpfc_directory_name)
{
    /* some static but dynamic buffer.
     */
    static int hpfc_bsz = 0;
    static char * hpfc_dir = NULL;

    int return_code, len;    
    char * dir = build_pgmwd(db_get_current_workspace_name());

    len = strlen(dir) + strlen(HPFC_COMPILED_FILE_DIR) + 5;

    if (hpfc_bsz<len) {
	if (hpfc_dir) free(hpfc_dir), hpfc_dir=NULL;
	hpfc_bsz = len;
	hpfc_dir = (char*) malloc(hpfc_bsz);
	message_assert("malloc succeeded", hpfc_dir);
    }

   /* Get the HPFC file name list: */
   sprintf(hpfc_dir, "%s/%s", dir, HPFC_COMPILED_FILE_DIR);
   
   return_code = safe_list_files_in_directory(
       file_names,
       hpfc_dir, /* Where is the output of HPFC: */
       "^[A-Z].*\\.[fh]$", /* generated files start with upercases */
       file_exists_p /* Plain files only: */);
   *hpfc_directory_name = hpfc_dir;

   return return_code;
}


string pips_change_directory(char *dir)
{
    if (directory_exists_p(dir)) {
	chdir(dir);
	return(get_cwd());	
    }

    return NULL;
}

/********************************************************* PIPS SOURCE PATH */

#define SRCPATH "PIPS_SRCPATH"

string pips_srcpath_append(string pathtoadd)
{
    string old_path, new_path;
    old_path = getenv(SRCPATH);
    old_path = strdup(old_path? old_path: "");
    new_path = strdup(concatenate(SRCPATH "=", old_path, ":", pathtoadd, 0));
    putenv(new_path);
    return old_path;
}

void pips_srcpath_set(string path)
{
    putenv(strdup(concatenate(SRCPATH "=", path, 0)));
}


/*************************** MODULE PROCESSING (INCLUDES and IMPLICIT NONE) */

static string user_file_directory = NULL;

#ifdef NO_RX_LIBRARY

static bool pips_process_file(string file_name)
{
    int err = safe_system_no_abort(concatenate
       ("trap 'exit 123' 2; pips-process-module ", file_name, NULL));

    if(err==123) {
	pips_user_warning("pips-process-module interrupted by control-C\n");
	return FALSE;
    }
    else if(err!=0) 
	pips_internal_error
	    ("Unexpected return code from pips-process-module: %d\n", err);

    return TRUE;
}

#else

#include "rxposix.h"

#define IMPLICIT_NONE_RX "^[ \t]*implicit[ \t]*none"
#define INCLUDE_FILE_RX  "^[ \t]*include[ \t]*['\"]\\([^'\"]*\\)['\"]"

/* not recognized: print *, "i = ", (0.,1.)
 * to avoid modifying a character constant...
 * this stuff should be done by hand in split...
 * also the generated lines may be too long...
 *
 * Well, the regex can be improved if necessary.
 *
 * In particular, it does first columned tabbed lines.
 */
#define CMPLX_RX \
"^[^\t!*Cc].....[^\"']*[^a-zA-Z0-9_ \t][ \t]*\\((\\)[-+0-9eE\\. \t]*,[-+0-9eE\\. \t]*)"

#define CMPLX2_RX \
"^[^\t!*Cc].....[ \t]*\\((\\)[-+0-9eE\\. \t]*,[-+0-9eE\\. \t]*)"

#define DCMPLX_RX \
    "^[^\t!*Cc].....[^\"']*[^a-zA-Z0-9_ \t][ \t]*" \
    "\\((\\)[-+0-9dDeE\\. \t]*,[-+0-9dDeE\\. \t]*)"

#define DCMPLX2_RX \
    "^[^\t!*Cc].....[ \t]*\\((\\)[-+0-9dDeE\\. \t]*,[-+0-9dDeE\\. \t]*)"

#define GOTO_RX "g[ \t]*o[ \t]*t[ \t]*o[ \t]*"

static regex_t 
    some_goto_rx,
    implicit_none_rx, 
    include_file_rx,
    complex_cst_rx,
    complex_cst2_rx,
    dcomplex_cst_rx,
    dcomplex_cst2_rx;

static void
insert_at(string * line, /* string to be modified, may be reallocated */
	  int offset, /* where to insert */
	  string what /* to be inserted */)
{
    int i, len=strlen(*line), shift=strlen(what);
    *line = (char*) realloc(*line, sizeof(char)*(len+shift+1));
    if (!*line) pips_internal_error("realloc failed\n");

    for (i=len; i>=offset; i--)
	(*line)[i+shift]=(*line)[i];

    for (shift--; shift>=0; shift--)
	(*line)[offset+shift]=what[shift];
}

#define CONTINUATION "\n     x "

static void add_continuation_if_needed(string * line)
{
    int len = strlen(*line);
    if (len<=73) return; /* nothing to do */
    /* else let us truncate */
    {
	int i = 71;
	while (i>5 && (isalnum((int) (*line)[i]) || (*line)[i]=='_')) 
	    i--;
	pips_assert("still in line", i>5);
	insert_at(line, i, CONTINUATION);
    }
}

/* return if modified
 */
static bool try_this_one(
    regex_t * prx, 
    string * line,
    string replacement, 
    bool was_modified)
{
    bool modified = FALSE;
    regmatch_t matches[2]; /* matched strings */

    while (!regexec(prx, *line, 2, matches, 0)) {
	if (!was_modified && !modified && strlen(*line)>73) {
	    pips_user_warning("line truncated in complex constant handling");
	    (*line)[72] = '\n', (*line)[73] = '\0';
	}
	modified = TRUE;
	insert_at(line, matches[1].rm_so, replacement);
    }
    
    return modified;
}

/* an assigned goto may look like "goto l, (10,20)",
 * and we wish not to consider (10,20) as a complex constant...
 */
/*
static void handle_complex_constants(string * line)
{
    bool diff = FALSE;

    if (!regexec(&some_goto_rx, *line, 0, NULL, 0)) return;

    diff |= try_this_one(&complex_cst_rx, line, IMPLIED_COMPLEX_NAME, diff);
    diff |= try_this_one(&complex_cst2_rx, line, IMPLIED_COMPLEX_NAME, diff);
    diff |= try_this_one(&dcomplex_cst_rx, line, IMPLIED_DCOMPLEX_NAME, diff);
    diff |= try_this_one(&dcomplex_cst2_rx, line, IMPLIED_DCOMPLEX_NAME, diff);

    if (diff) add_continuation_if_needed(line);
}
*/

/* tries several path for a file to include...
 * first rely on $PIPS_SRCPATH, then other directories.
 */
static string find_file(string name)
{
    string srcpath = getenv(SRCPATH), result;
    string path = strdup(concatenate(
	srcpath? srcpath: "", ":", 
	user_file_directory? user_file_directory: "", ":.:..", 0));
    result = find_file_in_directories(name, path);
    free(path);
    return result;
}

/* cache of preprocessed includes
 */
static hash_table processed_cache = hash_table_undefined;
void init_processed_include_cache(void)
{
    pips_assert("undefined cache", hash_table_undefined_p(processed_cache));
    processed_cache = hash_table_make(hash_string, 100);
}

void close_processed_include_cache(void)
{
    if (hash_table_undefined_p(processed_cache)) 
    {
	/* pips may call this without a prior call to 
	 * init_processed_include_cache under some error conditions,
	 * such as a file not found in the initializer, or a failed cpp.
	 */
	pips_user_warning("no 'processed include cache' to close, "
			  "skipping...\n");
	return;
    }
    pips_assert("defined cache", !hash_table_undefined_p(processed_cache));
    HASH_MAP(k, v, { unlink(v); free(v); free(k); }, processed_cache);
    hash_table_free(processed_cache);
    processed_cache = hash_table_undefined;
}

/* returns the processed cached file name, or null if none.
 */
static string get_cached(string s)
{
    string res;
    pips_assert("cache initialized", !hash_table_undefined_p(processed_cache));
    res = hash_get(processed_cache, s);
    return res==HASH_UNDEFINED_VALUE? NULL: res;
}

/* return an allocated unique cache file name.
 */
static string get_new_tmp_file_name(void)
{
    static int unique = 0;
    string dir_name, file_name;
    unsigned int len;
    dir_name = db_get_directory_name_for_module(WORKSPACE_TMP_SPACE);
    len = strlen(dir_name)+20;
    file_name = (char*) malloc(sizeof(char)*len);
    if (!file_name) pips_internal_error("malloc failed\n");
    sprintf(file_name, "%s/cached.%d", dir_name, unique++);
    pips_assert("not too long", strlen(file_name)<len);
    free(dir_name);
    return file_name;
}

/* double recursion (handle_file/handle_file_name) 
 * => forwarded declaration.
 */
static bool handle_file(FILE*, FILE*);

static bool handle_file_name(FILE * out, char * file_name, bool included)
{
    FILE * f;
    string found = find_file(file_name);
    bool ok = FALSE;

    if (!found)
    {
	/* Do not raise a user_error exception,
	   because you are not in the right directory
	   maybe this is not true anymore? FC 01/04/1998
	*/
	pips_user_warning("include file %s not found\n", file_name);
	fprintf(out, 
		"!! ERROR - include \"%s\" was not found\n"
		"      include \"%s\"\n", file_name, file_name);
	return FALSE;
    }

    pips_debug(2, "including file \"%s\"\n", found);
    if (included) fprintf(out, "! include \"%s\"\n", file_name);

    f=safe_fopen(found, "r");
    ok = handle_file(out, f);
    safe_fclose(f, found);

    if (included) fprintf(out, "! end include \"%s\"\n", file_name);
    free(found);
    return ok;
}

static bool handle_include_file(FILE * out, char * file_name)
{
    FILE * in;
    bool ok = TRUE;
    string cached = get_cached(file_name);
    char * error = NULL;

    if (!cached)
    {
	FILE * tmp_out;

	cached = get_new_tmp_file_name();
	tmp_out = safe_fopen(cached, "w");
	ok = handle_file_name(tmp_out, file_name, TRUE);
	safe_fclose(tmp_out, cached);

	/* handle bang comments and hollerith with an additionnal
	 * processing...
	 */
	if (ok)
	{
	    string filtered;
	    FILE * tmp_hbc, * tmp_in;

	    filtered = get_new_tmp_file_name();
	    tmp_hbc = safe_fopen(filtered, "w");
	    tmp_in = safe_fopen(cached, "r");
	    
	    error = process_bang_comments_and_hollerith(tmp_in, tmp_hbc);
	    if (error) ok = FALSE;

	    safe_fclose(tmp_in, cached);
	    safe_fclose(tmp_hbc, filtered);

	    safe_unlink(cached);
	    free(cached);
	    cached = filtered;
	}	    

	/* if ok put in the cache, otherwise drop it. */
	if (ok) hash_put(processed_cache, strdup(file_name), cached);
	else {
	    safe_unlink(cached);
	    free(cached), cached = NULL;
	}
    }

    if (ok) 
    {
	in = safe_fopen(cached, "r");
	safe_cat(out, in);
	safe_fclose(in, cached);
    }

    if (error) pips_user_error("preprocessing error: %s\n", error);

    return ok;
}

/* process f for includes and nones
 */
static bool handle_file(FILE * out, FILE * f) 
{
    string line;
    regmatch_t matches[2]; /* matched strings */

    while ((line = safe_readline(f)))
    {
	if (!skip_line_p(line))
	{
	    if (!regexec(&include_file_rx, line, 2, matches, 0))
	    {
		char c = line[matches[1].rm_eo];
		line[matches[1].rm_eo]='\0';
		
		if (!handle_include_file(out, &line[matches[1].rm_so]))
		    return FALSE; /* error? */

		line[matches[1].rm_eo]=c;
		fprintf(out, "! ");
	    }
	    else if (!regexec(&implicit_none_rx, line, 0, matches, 0))
		    fprintf(out, 
		      "! MIL-STD-1753 Fortran extension not in PIPS\n! ");
	    else {
		/* FI: test for parser */
		/* handle_complex_constants(&line); */
		;
	    }
	}
	fprintf(out, "%s\n", line);
	free(line);
    }
    return TRUE;
}

static void init_rx(void)
{
    static bool done=FALSE;
    if (done) return;
    done=TRUE;
    if (regcomp(&some_goto_rx, GOTO_RX, REG_ICASE)              ||
	regcomp(&implicit_none_rx, IMPLICIT_NONE_RX, REG_ICASE) ||
	regcomp(&include_file_rx, INCLUDE_FILE_RX, REG_ICASE)   ||
	regcomp(&complex_cst_rx, CMPLX_RX, REG_ICASE)           ||
	regcomp(&complex_cst2_rx, CMPLX2_RX, REG_ICASE)         ||
	regcomp(&dcomplex_cst_rx, DCMPLX_RX, REG_ICASE)         ||
	regcomp(&dcomplex_cst2_rx, DCMPLX2_RX, REG_ICASE))
	pips_internal_error("invalid regular expression\n");
}

static bool pips_process_file(string file_name, string new_name)
{
    bool ok = FALSE;
    FILE * out;
    pips_debug(2, "processing file %s\n", file_name);
    init_rx();
    out = safe_fopen(new_name, "w");
    ok = handle_file_name(out, file_name, FALSE);
    safe_fclose(out, new_name);
    return ok;
}

#endif

#define FORTRAN_FILE_SUFFIX ".f"

bool filter_file(string mod_name)
{
    string name, new_name, dir_name, abs_name, abs_new_name;
    name = db_get_memory_resource(DBR_INITIAL_FILE, mod_name, TRUE);

    /* directory is set for finding includes. */
    user_file_directory = 
	pips_dirname(db_get_memory_resource(DBR_USER_FILE, mod_name, TRUE));
    new_name = db_build_file_resource_name
	(DBR_SOURCE_FILE, mod_name, FORTRAN_FILE_SUFFIX);
    
    dir_name = db_get_current_workspace_directory();
    abs_name = strdup(concatenate(dir_name, "/", name, 0));
    abs_new_name = strdup(concatenate(dir_name, "/", new_name, 0));
    free(dir_name);
    
    if (!pips_process_file(abs_name, abs_new_name)) 
    {
	pips_user_warning("initial file filtering of %s failed\n", mod_name);
	safe_unlink(abs_new_name);
	free(abs_new_name); free(abs_name);
	return FALSE;
    }
    free(abs_new_name); free(abs_name);
    free(user_file_directory), user_file_directory = NULL;

    DB_PUT_NEW_FILE_RESOURCE(DBR_SOURCE_FILE, mod_name, new_name);
    return TRUE;
}


/******************************************************************** SPLIT */

static bool zzz_file_p(string s) /* .../zzz???.f */
{ int len = strlen(s)-1;
  return len>=8 && s[len-8]=='/' && s[len-7]=='#' && s[len-6]=='#' && 
      s[len-5]=='#' && s[len-1]=='.' && s[len]=='f'; }
/* static int cmp(const void * x1, const void * x2)
{ return strcmp(*(char**)x1, *(char**)x2);} */
static void clean_file(string name)
{
    FILE *f;
    string line;
    int i=0, size = 20;
    char ** lines = (char**) malloc(sizeof(char*)*size);
    pips_assert("malloc ok", lines);

    f=safe_fopen(name, "r");
    while ((line=safe_readline(f)))
    {
	if (!zzz_file_p(line)) /* drop zzz* files */
	{
	    if (i==size) { /* resize lines[] */
		size*=2;
		lines = (char**) realloc(lines, sizeof(char*)*size);
		pips_assert("realloc ok", lines);
	    }
            lines[i++]=line;
	}
	else 
	{
	    unlink(line);
	    free(line);
	}
    }
    safe_fclose(f, name);

    /* keep order for unsplit. */
    /* qsort(lines, i, sizeof(char*), cmp); */

    f=safe_fopen(name, "w");
    while (i>0) {
	fprintf(f, "%s\n", lines[--i]);
	free(lines[i]);
    }
    free(lines);
    safe_fclose(f, name);
}

static bool pips_split_file(string name, string tempfile)
{
  char * err;
  FILE * out = safe_fopen(tempfile, "w");
  string dir = db_get_current_workspace_directory();

  if (dot_c_file_p(name))
    err = csplit(dir, name, out);
  else if (dot_f_file_p(name) || dot_F_file_p(name)) 
    err = fsplit(dir, name, out);
  else
    pips_user_error("unexpected file name for splitting: %s", name);

  free(dir);
  safe_fclose(out, tempfile);
  clean_file(tempfile); 
  if (err) fprintf(stderr, "split error: %s\n", err);
  return err? TRUE: FALSE;
}

/***************************************** MANAGING .F AND .c FILES WITH CPP */

/* an issue is that the cpp used for .F must be Fortran 77 aware.
 */

#define CPP_FORTRAN_ED		 	".cpp_processed.f"
#define CPP_C_ED		 	".cpp_processed.c"
#define CPP_ERR			".stderr"

/* pre-processor and added options from environment
 */
#define CPP_PIPS_ENV		"PIPS_CPP"
#define CPP_PIPS_OPTIONS_ENV 	"PIPS_CPP_FLAGS"

/* default preprocessor and basic options
 */
#define CPP_CPP			"cpp -C" /* alternative values: "gcc -E -C" or "fpp" */
#define CPP_CPPFLAGS		" -P -D__PIPS__ -D__HPFC__ "

static bool suffix_file_p(string name, char suffix)
{
    int l = strlen(name);
    return l>=2 && name[l-1]==suffix && name[l-2]=='.';
}

static bool dot_F_file_p(string name)
{
  return suffix_file_p(name, 'F');
}

static bool dot_f_file_p(string name)
{
  return suffix_file_p(name, 'f');
}

static bool dot_c_file_p(string name)
{
  return suffix_file_p(name, 'c');
}

/* Returns the newly allocated name if ccp succeeds.
 * Returns NULL if cpp fails.
 */

/* The structure of the string is not checked. Funny results to be expected for strings starting or ending with ':' and containing lots of SPACES*/
static int colon_number(string s)
{
  int number = s? 1: 0;
  string new_s = s;
  char c;

  while((c=*new_s++))
    if(c==':' && new_s!=s+1 && *new_s!='\000')
      number++;

  return number;
}

static string process_thru_C_cpp(string name)
{
    string dir_name, new_name, simpler, cpp_options, cpp, cpp_err;
    int status = 0;
    string includes = getenv(SRCPATH);
    string new_includes = includes; /* pointer towards the current
                                       character in includes. */
    int include_option_length = 5+strlen(includes)+3*colon_number(includes)+1+1;
    /* Dynamic size, gcc only? Use malloc() instead */
    char include_options[include_option_length];
    /* Pointer to the current end of include_options: */
    string new_include_options = string_undefined;
    /* Pointer to the beginning of include_options, for debugging purposes: */
    string old_include_options = &include_options[0]; 

    (void) strcpy(old_include_options, "-I. ");
    new_include_options = include_options+strlen(include_options);
    dir_name = db_get_directory_name_for_module(WORKSPACE_TMP_SPACE);
    simpler = pips_basename(name, ".c");
    new_name = strdup(concatenate(dir_name, "/", simpler, CPP_C_ED, 0));
    cpp_err  = strdup(concatenate(new_name, CPP_ERR, 0));
    free(dir_name);
    free(simpler);

    cpp = getenv(CPP_PIPS_ENV);
    cpp_options = getenv(CPP_PIPS_OPTIONS_ENV);

    /* Convert PIPS_SRCPATH syntax to gcc -I syntax */
    /* Could have reused nth_path()... */
    if (includes && *includes) {
      (void) strcat(old_include_options, " -I");
      new_include_options += 3;
      do {
	/* Skip leading and trailing colons. */
	if(*new_includes==':' && new_includes!=includes+1 && *new_includes!='\000') {
	  new_includes++;
	  *new_include_options = '\000';
	  new_include_options = strcat(new_include_options, " -I");
	  new_include_options += 3;
	}
	else
	  *new_include_options++ = *new_includes++;
      } while(includes && *new_includes);
    }
    *new_include_options++ = ' ';
    *new_include_options++ = '\000';

    pips_assert("include_options is large enough",
		strlen(include_options) < include_option_length);

    pips_debug(1, "PIPS_SRCPATH=\"%s\"\n", includes);
    pips_debug(1, "INCLUDE=\"%s\"\n", include_options);


    status = safe_system_no_abort
      (concatenate("/usr/local/bin/gcc "/* cpp? cpp: CPP_CPP  */,
		   "-E -C " /* CPP_CPPFLAGS, cpp_options? cpp_options: ""*/, 
		   old_include_options,
		   name, " > ", new_name, " 2> ", cpp_err, 0));

    if(status) {
      (void) safe_system_no_abort(concatenate("cat ", cpp_err, 0));

      /* check "test" could be performed before "cat" but not after, and
	 the error file may be useful for the user. Why should we remove
	 it so soon?

			    " && test ! -s ", cpp_err, 
			    " && rm -f ", cpp_err, 0)); */
      free(new_name);
      new_name = NULL;
    }

    return new_name;
}

static string process_thru_fortran_cpp(string name)
{
    string dir_name, new_name, simpler, cpp_options, cpp, cpp_err;
    int status;

    dir_name = db_get_directory_name_for_module(WORKSPACE_TMP_SPACE);
    simpler = pips_basename(name, ".F");
    new_name = strdup(concatenate(dir_name, "/", simpler, CPP_FORTRAN_ED, 0));
    cpp_err  = strdup(concatenate(new_name, CPP_ERR, 0));
    free(dir_name);
    free(simpler);

    cpp = getenv(CPP_PIPS_ENV);
    cpp_options = getenv(CPP_PIPS_OPTIONS_ENV);
    
    /* Note: the cpp used **must** know somehow about Fortran
     * and its lexical and comment conventions. This is ok with gcc
     * when g77 is included. Otherwise, "'" appearing in Fortran comments
     * results in errors to be reported.
     * Well, the return code could be ignored maybe, but I prefer not to.
     */

    /* FI->FC: it should be a safe_system_no_abort(). Errors are
       supposedly displayed... but their display is skipped because of the
       return code of the first process and pips_internal_error is
       executed!. PIPS_SRCPATH is not used to find the include files. A
       pips_user_error is not caught at the process_user_file level.

       See preprocessor/Validation/csplit09.tpips */

    status = safe_system_no_abort(concatenate(cpp? cpp: CPP_CPP, 
				 CPP_CPPFLAGS, cpp_options? cpp_options: "", 
				 name, " > ", new_name, " 2> ", cpp_err, 
				 " && cat ", cpp_err, 
         			 " && test ! -s ", cpp_err, 
			         " && rm -f ", cpp_err, 0));

    /* fpp was wrong... */
    if (status)
    {
      /* show errors */
      (void) safe_system_no_abort(concatenate("cat ", cpp_err, 0));
      free(new_name);
      new_name = NULL;
    }

    return new_name;
}

static string process_thru_cpp(string name)
{
  /* Not much to share between .F and .c? */
  string new_name = string_undefined;

  if(dot_F_file_p(name)) 
    new_name = process_thru_fortran_cpp(name);
  else
    new_name = process_thru_C_cpp(name);

    return new_name;
}

/*************************************************** MANAGING A USER FILE */

/* Fortran compiler triggerred from the environment (PIPS_CHECK_FORTRAN) 
 * or a property (CHECK_FORTRAN_SYNTAX_BEFORE_PIPS)
 */
static int pips_check_fortran(void)
{
    string v = getenv("PIPS_CHECK_FORTRAN");

    if (v && (*v=='o' || *v=='y' || *v=='t' || *v=='v' || *v=='1' ||
	      *v=='O' || *v=='Y' || *v=='T' || *v=='V'))
	return TRUE;

    return get_bool_property("CHECK_FORTRAN_SYNTAX_BEFORE_PIPS");
}

#define SUFFIX ".pips.o"
#define DEFAULT_PIPS_FLINT "f77 -c -ansi"

static bool check_fortran_syntax_before_pips(string file_name)
{
  string pips_flint = getenv("PIPS_FLINT");
  bool syntax_ok_p = TRUE;

  user_log("Checking Fortran syntax of %s\n", file_name);

  if (safe_system_no_abort(concatenate(
				       pips_flint? pips_flint: DEFAULT_PIPS_FLINT, " ",
				       file_name, 
				       " -o ", file_name, SUFFIX,
				       " ; test -f ", file_name, SUFFIX, 
				       " && rm ", file_name, SUFFIX, NULL))) {

    /* f77 is rather silent on errors... which is detected if no
     * file was output as expected.
     */
    pips_user_warning("\n\n\tFortran syntax errors in file %s!\007\n\n", 
		      file_name);
    syntax_ok_p = FALSE;
  }
  return syntax_ok_p;
}

/* "foo bla fun  ./e.database/foo.f" -> "./e.database/foo.f"
 */
static string extract_last_name(string line)
{
    int l = strlen(line);
    do {
	while (l>=0 && line[l]!=' ') l--;
	if (l>=0 && line[l]==' ') line[l]='\0';
    } while (l>=0 && strlen(line+l+1)==0);
    return l>=-1? line+l+1: NULL;
}

bool process_user_file(string file)
{
  FILE *fd;
  bool success_p = FALSE, cpp_processed_p;
  string initial_file, nfile, file_list, a_line,
    dir_name = db_get_current_workspace_directory();

  static int number_of_files = 0;
  static int number_of_modules = 0;

  number_of_files++;
  pips_debug(1, "file %s (number %d)\n", file, number_of_files);

  /* the file is looked for in the pips source path.
   */
  nfile = find_file_in_directories(file, getenv(SRCPATH));

  if (!nfile)
    {
      pips_user_warning("Cannot open file: %s\n", file);
      return FALSE;
    }

  initial_file = nfile;

  /* the new file is registered (well, not really...) in the database.
   */
  user_log("Registering file %s\n", file);

  /* Fortran compiler if required.
   */
  if (pips_check_fortran() && (dot_F_file_p(nfile) || dot_f_file_p(nfile))) {
    bool syntax_ok_p = check_fortran_syntax_before_pips(nfile);

    if(!syntax_ok_p)
      return FALSE;
  }
  else if(FALSE) {
    /*Run the C compiler */
    ;
  }

  /* CPP if file extension if .F or .c 
   * (assumes string_equal_p(nfile, initial_file))
   */
  cpp_processed_p = dot_F_file_p(nfile) || dot_c_file_p(nfile);

  if (cpp_processed_p) {
    user_log("Preprocessing file %s\n", initial_file);
    nfile = process_thru_cpp(initial_file);
    if(nfile==NULL) {
      pips_user_warning("Cannot preprocess file: %s\n", initial_file);
      return FALSE;
    }
  }
  else if (!dot_f_file_p(nfile)) {
    pips_user_error("Unexpected file extension\n");
  }

  /* if two modules have the same name, the first splitted wins
   * and the other one is hidden by the call since fsplit gives 
   * it a zzz00n.f name 
   * Let's hope no user module is called ###???.f 
   */
  file_list = 
    strdup(concatenate(dir_name,
		       dot_c_file_p(nfile)? 
		         "/.csplit_file_list" : "/.fsplit_file_list", 0));
  unlink(file_list);

  user_log("Splitting file    %s\n", nfile);
  if (pips_split_file(nfile, file_list))
    return FALSE;

  /* the newly created module files are registered in the database
   * The file_list allows split to communicate with this function.
   */
  fd = safe_fopen(file_list, "r");
  while ((a_line = safe_readline(fd))) 
    {
      string mod_name = NULL, res_name = NULL, abs_res, file_name;
      list modules = NIL;
      bool renamed=FALSE;

      /* a_line: "MODULE1 ... MODULEn file_name"
       *
       * the liste modules come from entries that might be included
       * in the subroutine. 
       */
      file_name = extract_last_name(a_line);
      success_p = TRUE;
      number_of_modules++;
      pips_debug(2, "module %s (number %d)\n", file_name, number_of_modules);

      while (mod_name!=a_line && (mod_name = extract_last_name(a_line)))
	modules = CONS(STRING, mod_name, modules);

      /* For each Fortran module in the line, put the initial_file and
	 user_file resource. In C, line should have only one entry and a C
	 source file and a user file resources are created. */
      MAP(STRING, mod_name, 
      {
	user_log("  Module         %s\n", mod_name);

	if (!renamed)
	  {
	    if(dot_c_file_p(nfile)) {
	      res_name = db_build_file_resource_name
		(DBR_C_SOURCE_FILE, mod_name, ".c");
	    }
	    else {
	      res_name = db_build_file_resource_name
		(DBR_INITIAL_FILE, mod_name, ".f_initial");
	    }
	    
	    abs_res = strdup(concatenate(dir_name, "/", res_name, 0));
		
	    if (rename(file_name, abs_res))
	      {
		perror("process_user_file");
		pips_internal_error("mv %s %s failed\n", 
				    file_name, res_name);
	      }
	    renamed = TRUE;
	    free(abs_res); 
	  }

	if(dot_c_file_p(nfile)) {
	  DB_PUT_NEW_FILE_RESOURCE(DBR_C_SOURCE_FILE, mod_name, 
				   strdup(res_name));
	}
	else {
	  DB_PUT_NEW_FILE_RESOURCE(DBR_INITIAL_FILE, mod_name, 
				   strdup(res_name));
	}
	/* from which file the initial source was derived.
	 * absolute path to the file so that db moves should be ok?
	 */
	DB_PUT_NEW_FILE_RESOURCE(DBR_USER_FILE, mod_name, strdup(nfile));
      },
	  modules);

      gen_free_list(modules), modules=NIL;

      if (res_name) free(res_name), res_name = NULL;
      free(a_line);
    }

  safe_fclose(fd, file_list);
  unlink(file_list); 
  free(file_list);
  free(dir_name);

  if (cpp_processed_p)
    free(initial_file); /* hey, that's cleaning! */

  if(!success_p)
    pips_user_warning("No module was found when splitting file %s.\n",
		      nfile);

  if(cpp_processed_p) {
    /* nfile is not the initial file */
    pips_debug(1, "Remove output of preprocessing: %s\n", nfile);
    /* Seems to be recorded as a resource, causes problems later when
       closing the workspace... */
    /* unlink(nfile); */
  }
  free(nfile);

  return TRUE; /* well, returns ok whether modules were found or not. */
}
