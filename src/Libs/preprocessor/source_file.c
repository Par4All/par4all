/* 
 * $Id$
 *
 * procedures used in both PIPS top-level, wpips and tpips.
 *
 * problems to use those procedures with wpips: show_message() and 
 * update_props() .
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <dirent.h>

#include <sys/stat.h>
#include <ctype.h>
#include <setjmp.h>
#include <unistd.h>
#include <errno.h>

#include "genC.h"
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
#include "pipsmake.h"

#include "top-level.h"

#define LINE_LENGTH 240

#define skip_line_p(s) \
  ((*(s))=='\0' || (*(s))=='!' || (*(s))=='*' || (*(s))=='c' || (*(s))=='C')


/* Return a sorted arg list of workspace names. (For each name, there
   is a name.database directory in the current directory): */
void
pips_get_workspace_list(gen_array_t array)
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
void
pips_get_fortran_list(gen_array_t array)
{
    list_files_in_directory(array, ".", "^.*\\.[fF]$", file_exists_p);
}


/* Return the path of an HPFC file name relative to the current PIPS
   directory. Can be freed by the caller. */
char *
hpfc_generate_path_name_of_file_name(char * file_name)
{
    string dir_name = db_get_current_workspace_directory(),
	name = strdup(concatenate(
	    dir_name, "/", HPFC_COMPILED_FILE_DIR, "/", file_name, 0));
    free(dir_name);
    return name;
}


int
hpfc_get_file_list(gen_array_t file_names,
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


char *
pips_change_directory(char *dir)
{
    if (directory_exists_p(dir)) {
	chdir(dir);
	return(get_cwd());	
    }

    return NULL;
}

/* returns the allocated full path name 
 */
static char *
get_view_file(char * print_type, bool displayable)
{
   char * module_name = db_get_current_module_name();

   if(displayable && !displayable_file_p(print_type)) {
       pips_user_error("resource %s cannot be displayed\n", print_type);
   }

   if(module_name != NULL)
   {
      if ( safe_make(print_type, module_name) ) 
      {
         char *file_name = db_get_file_resource(print_type, module_name, TRUE);
	 char *pgm_wd = build_pgmwd(db_get_current_workspace_name());
	 char *file_name_in_database = strdup(
	     concatenate(pgm_wd, "/", file_name, NULL));

	 free(pgm_wd); 
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
build_view_file(char * print_type)
{
    return get_view_file(print_type, TRUE);
}

char *
get_dont_build_view_file(char * print_type)
{
    return get_view_file(print_type, FALSE);
}

/*************************** MODULE PROCESSING (INCLUDES and IMPLICIT NONE) */

static string real_file_dir = NULL;
static void reset_real_file_dir(void)
{
    free(real_file_dir);
    real_file_dir = NULL;
}
static void set_real_file_dir(string path)
{
    int l = strlen(path);
    if (real_file_dir) reset_real_file_dir();
    real_file_dir = strdup(path);
    while (l>0 && real_file_dir[l]!='/') l--;
    real_file_dir[l]= '\0';
}

#ifdef NO_RX_LIBRARY

static bool
pips_process_file(string file_name)
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
 */
#define CMPLX_RX \
"^......[^\"']*[^a-zA-Z0-9_ \t][ \t]*\\((\\)[-+0-9eE\\. \t]*,[-+0-9eE\\. \t]*)"

#define CMPLX2_RX \
"^......[ \t]*\\((\\)[-+0-9eE\\. \t]*,[-+0-9eE\\. \t]*)"

#define DCMPLX_RX \
    "^......[^\"']*[^a-zA-Z0-9_ \t][ \t]*" \
    "\\((\\)[-+0-9dDeE\\. \t]*,[-+0-9dDeE\\. \t]*)"

#define DCMPLX2_RX \
    "^......[ \t]*\\((\\)[-+0-9dDeE\\. \t]*,[-+0-9dDeE\\. \t]*)"

static regex_t 
    implicit_none_rx, 
    include_file_rx,
    complex_cst_rx,
    complex_cst2_rx,
    dcomplex_cst_rx,
    dcomplex_cst2_rx;
static FILE * output_file;

static void
insert_at(
    char line[LINE_LENGTH], /* string to be modified */
    int offset, /* where to insert */
    string what /* to be inserted */)
{
    int i, len=strlen(line), shift=strlen(what);
    pips_assert("line large enough", len+shift<LINE_LENGTH);

    for (i=len; i>=offset; i--)
	line[i+shift]=line[i];

    for (shift--; shift>=0; shift--)
	line[offset+shift]=what[shift];
}

#define CONTINUATION "\n     x "

static void
add_continuation_if_needed(char line[LINE_LENGTH])
{
    int len = strlen(line);
    if (len<=73) return; /* nothing to do */
    /* else let us truncate */
    {
	int i = 71;
	while (i>5 && (isalnum(line[i]) || line[i]=='_')) 
	    i--;
	pips_assert("still in line", i>5);
	insert_at(line, i, CONTINUATION);
    }
}

/* return if modified
 */
static bool
try_this_one(
    regex_t * prx, 
    string line,
    string replacement, 
    bool was_modified)
{
    bool modified = FALSE;
    regmatch_t matches[2]; /* matched strings */

    while (!regexec(prx, line, 2, matches, 0)) {
	if (!was_modified && !modified && strlen(line)>73) {
	    pips_user_warning("line truncated in complex constant handling");
	    line[72] = '\n', line[73] = '\0';
	}
	modified = TRUE;
	insert_at(line, matches[1].rm_so, replacement);
    }
    
    return modified;
}

static void
handle_complex_constants(string line)
{
    bool diff = FALSE;

    diff |= try_this_one(&complex_cst_rx, line, IMPLIED_COMPLEX_NAME, diff);
    diff |= try_this_one(&complex_cst2_rx, line, IMPLIED_COMPLEX_NAME, diff);
    diff |= try_this_one(&dcomplex_cst_rx, line, IMPLIED_DCOMPLEX_NAME, diff);
    diff |= try_this_one(&dcomplex_cst2_rx, line, IMPLIED_DCOMPLEX_NAME, diff);

    if (diff) add_continuation_if_needed(line);
}

/* tries several path for a file to include...
 */
static string 
find_file(string name)
{
    string other;

    other = strdup(concatenate(real_file_dir, "/", name, NULL));
    if (file_exists_p(other))
	return other;
    free(other);

    if (file_exists_p(name)) return strdup(name);

    other = strdup(concatenate("../", name, NULL));
    if (file_exists_p(other))
	return other;
    free(other);
    
    return NULL;
}

static bool handle_file(FILE*);
static bool 
handle_file_name(char * file_name, bool comment)
{
    FILE * f;
    string found = find_file(file_name);
    bool ok = FALSE;

    if (!found)
    {
      /* Do not raise a user_error exception,
	 because you are not in the right directory */
	pips_user_warning("include file %s not found\n", file_name);
	fprintf(output_file, "! include \"%s\" not found\n", file_name);
	return FALSE;
    }

    pips_debug(2, "including file \"%s\"\n", found);

    if (comment)
	fprintf(output_file, "! include \"%s\"\n", file_name);

    f=safe_fopen(found, "r");
    ok = handle_file(f);
    safe_fclose(f, found);

    if (comment)
	fprintf(output_file, "! end include \"%s\"\n", file_name);

    free(found);

    return ok;
}

static bool 
handle_file(FILE * f) /* process f for includes and nones */
{
    char line[LINE_LENGTH];
    regmatch_t matches[2]; /* matched strings */

    while (fgets(line, LINE_LENGTH, f))
    {
	if (!skip_line_p(line))
	{
	    if (!regexec(&include_file_rx, line, 2, matches, 0))
	    {
		char c = line[matches[1].rm_eo];
		line[matches[1].rm_eo]='\0';
		
		if (!handle_file_name(&line[matches[1].rm_so], TRUE))
		    return FALSE;

		line[matches[1].rm_eo]=c;
		fprintf(output_file, "! ");
	    }
	    else if (!regexec(&implicit_none_rx, line, 0, matches, 0))
		    fprintf(output_file, 
			    "! MIL-STD-1753 Fortran extension not in PIPS\n! ");
	    else
		handle_complex_constants(line);
	}
	fprintf(output_file, "%s", line);
    }
    return TRUE;
}

static void 
init_rx(void)
{
    static bool done=FALSE;
    if (done) return;
    done=TRUE;
    if (regcomp(&implicit_none_rx, IMPLICIT_NONE_RX, REG_ICASE) ||
	regcomp(&include_file_rx, INCLUDE_FILE_RX, REG_ICASE)   ||
	regcomp(&complex_cst_rx, CMPLX_RX, REG_ICASE)           ||
	regcomp(&complex_cst2_rx, CMPLX2_RX, REG_ICASE)         ||
	regcomp(&dcomplex_cst_rx, DCMPLX_RX, REG_ICASE)         ||
	regcomp(&dcomplex_cst2_rx, DCMPLX2_RX, REG_ICASE))
	pips_internal_error("invalid regular expression\n");
}

static bool 
pips_process_file(string file_name)
{
    string origin = strdup(concatenate(file_name, ".origin", 0));
    bool ok = FALSE;

    pips_debug(2, "processing file %s\n", file_name);
    
    if (rename(file_name, origin)) {
	perror("pips_process_file");
	pips_user_warning("error while renaming %s as %s\n",
			  file_name, origin);
	return FALSE;
    }

    output_file = safe_fopen(file_name, "w");
    init_rx();
    ok = handle_file_name(origin, FALSE);
    safe_fclose(output_file, origin);
    free(origin);

    return ok;
}

#endif

static bool zzz_file_p(string s) /* zzz???.f */
{ return strlen(s)==8 && s[0]=='z' && s[1]=='z' && s[2]=='z' &&
      s[6]=='.' && s[7]=='f'; }
static int cmp(const void * x1, const void * x2)
{ return strcmp(*(char**)x1, *(char**)x2);}
static void sort_file(string name)
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
	else free(line);
    }
    safe_fclose(f, name);

    qsort(lines, i, sizeof(char*), cmp);

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
#ifdef NO_INTERNAL_FSPLIT
    int err = safe_system_no_abort
	(concatenate("trap 'exit 123' 2;",
		     "pips-split ", abspath,
		     "| sed -e /zzz[0-9][0-9][0-9].f/d | sort -r > ",
		     tempfile, "; /bin/rm -f zzz???.f", NULL));

    if(err==123)
	pips_user_warning("File splitting interrupted by control-C\n");
    else if(err!=0)
	pips_internal_error("Unexpected return code from pips-split: %d\n", 
			    err);

    return err;
#else
    int err;
    FILE * out = safe_fopen(tempfile, "w");
    err = fsplit(name, out);
    safe_fclose(out, tempfile);

    sort_file(tempfile);

    return err;
#endif
}

/********************************************** managing .F files with cpp */

#define CPP_FILTERED_SUFFIX 	".cpp_processed"

/* pre-processor and added options from environment
 */
#define CPP_PIPS_ENV		"PIPS_CPP"
#define CPP_PIPS_OPTIONS_ENV 	"PIPS_CPP_FLAGS"

/* default preprocessor and basic options
 */
#define CPP_CPP			"cpp"
#define CPP_CPPFLAGS		" -P -C -D__PIPS__ -D__HPFC__ "

static bool dot_F_file_p(string name)
{
    int l = strlen(name);
    return l>=2 && name[l-1]=='F' && name[l-2]=='.';
}

/* returns the newly allocated name */
static string process_thru_cpp(string name)
{
    string new_name = strdup(concatenate(name, CPP_FILTERED_SUFFIX, NULL));
    string cpp_options = getenv(CPP_PIPS_OPTIONS_ENV);
    string cpp = getenv(CPP_PIPS_ENV);

    safe_system(concatenate(cpp? cpp: CPP_CPP, 
			    CPP_CPPFLAGS, cpp_options? cpp_options: "", " ",
			    name, " > ", new_name, NULL));

    return new_name;
}

/*************************************************** managing a user file */

/* Fortran compiler triggerred from the environment (PIPS_CHECK_FORTRAN) 
 * or a property (CHECK_FORTRAN_SYNTAX_BEFORE_PIPS)
 */
static int
pips_check_fortran(void)
{
    string v = getenv("PIPS_CHECK_FORTRAN");

    if (v && (*v=='o' || *v=='y' || *v=='t' || *v=='v' || *v=='1' ||
	      *v=='O' || *v=='Y' || *v=='T' || *v=='V'))
	return TRUE;

    return get_bool_property("CHECK_FORTRAN_SYNTAX_BEFORE_PIPS");
}

#define suffix ".pips.o"
#define DEFAULT_PIPS_FLINT "f77 -c -ansi"

static void 
check_fortran_syntax_before_pips(
    string file_name)
{
    string pips_flint = getenv("PIPS_FLINT");
    user_log("Checking Fortran syntax of %s\n", file_name);

    if (safe_system_no_abort(concatenate(
	pips_flint? pips_flint: DEFAULT_PIPS_FLINT, " ", file_name, 
	" -o ", file_name, suffix, " ; test -f ", file_name, suffix, 
	" && rm ", file_name, suffix, NULL)))

	/* f77 is rather silent on errors... which is detected if no
	 * file was output as expected.
	 */
	pips_user_warning("\n\n\tFortran syntax errors in file %s!\007\n\n", 
			  file_name);
}

/* for Linux at least
 */
#ifndef MAXNAMLEN
#define MAXNAMLEN (1024)
#endif

bool 
process_user_file(string file)
{
    FILE *fd;
    bool success_p = FALSE, cpp_processed_p;
    string cwd, abspath = NULL, initial_file, tempfile = NULL, nfile, name;
    int err;

    static int number_of_files = 0;
    static int number_of_modules = 0;

    number_of_files++;
    pips_debug(1, "file %s (number %d)\n", file, number_of_files);

    /* the file is looked for in the pips source path.
     */
    nfile = find_file_in_directories(file, getenv("PIPS_SRCPATH"));

    if (!nfile) {
	pips_user_warning("Cannot open file : %s\n", file);
	return FALSE;
    }

    initial_file = nfile;

    /* Fortran compiler.
     */
    if (pips_check_fortran()) 
	check_fortran_syntax_before_pips(nfile);

    /* CPP
     */
    cpp_processed_p = dot_F_file_p(nfile);

    if (cpp_processed_p)
    {
	user_log("Preprocessing file %s\n", initial_file);
	nfile = process_thru_cpp(initial_file);
	
    }

    if (tempfile == NULL) {
	tempfile = tmpnam(NULL);
    }

    /* the absolute path of file is calculated
     */
    abspath = strdup((*nfile == '/') ? nfile : 
		     concatenate(get_cwd(), "/", nfile, NULL));

    set_real_file_dir(abspath);

    /* the new file is registered in the database
     */
    user_log("Registering file %s\n", file);

    /* FI: two problems here
     * - the successive calls to DB_PUT_FILE_RESOURCE erase each other...
     */
    DB_PUT_FILE_RESOURCE(DBR_USER_FILE, "", abspath);

    /* the new file is splitted according to Fortran standard */

    user_log("Splitting file    %s\n", nfile);

    cwd = strdup(get_cwd());
    chdir(db_get_current_workspace_directory());

    /* reverse sort because the list of modules is reversed later */
    /* if two modules have the same name, the first splitted wins
       and the other one is hidden by the call to "sed" since
       fsplit gives it a zzz00n.f name */
    /* Let's hope no user module is called zzz???.f */

    err = pips_split_file(abspath, tempfile);

    /* Go back unconditionnally to regular directory for execution
     * or you are heading for trouble when the database is closed
     */
    chdir(cwd);
    free(cwd);

    if (err) return FALSE;

    /* the newly created module files are registered in the database */
    fd = safe_fopen(tempfile, "r");
    while ((name=safe_readline(fd))) 
    {
	string mod_name, file_name, res_name;

	number_of_modules++;
	pips_debug(2, "module %s (number %d)\n", name, number_of_modules);

	success_p = TRUE;
	file_name = strdup(name);

	*strchr(name, '.') = '\0'; /* there MUST be one one dot... */
	mod_name = strdup(name);
	(void) strupper(mod_name, mod_name);
	free(name);

	user_log("  Module         %s\n", mod_name);

	res_name = db_build_file_resource_name(DBR_SOURCE_FILE,mod_name, ".f");

        /* Apply a cleaning procedure on each module: */
        cwd = strdup(get_cwd());
        chdir(db_get_current_workspace_directory());

	if (rename(file_name, res_name)) {
	    perror("process_user_file");
	    chdir(cwd); free(cwd); 
	    pips_internal_error("mv %s %s failed in %s\n", 
				file_name, res_name, get_cwd());
	}
	free(file_name);

        if (!pips_process_file(res_name)) { /* should be a pass... */
	    chdir(cwd); free(cwd); 
	    return FALSE;
	}

        chdir(cwd); free(cwd);

	DB_PUT_NEW_FILE_RESOURCE(DBR_SOURCE_FILE, mod_name, res_name);
    }
    safe_fclose(fd, tempfile);

    unlink(tempfile); tempfile = NULL;

    if (cpp_processed_p) {
	unlink(nfile); /* remove .cpp_filtered file */
	free(initial_file);
    }

    if(!success_p)
	pips_user_warning("No module was found when splitting file %s.\n",
			  abspath);

    reset_real_file_dir();
    free(nfile);

    return success_p;
}
