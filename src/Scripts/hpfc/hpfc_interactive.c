/* $RCSfile: hpfc_interactive.c,v $ (version $Revision$)
 * $Date: 1995/05/17 12:26:02 $, 
 *
 * interactive interface to hpfc, based on the GNU realine library.
 */

#include <stdio.h>
#include <strings.h>

extern int system();
extern int fprintf();
extern int chdir();
extern char *getenv();
extern int malloc();
extern void free();

#include "readline/readline.h"
#include "readline/history.h"

#define HPFC_PROMPT "hpfc> "
#define HPFC_PREFIX "hpfc"
#define HIST ".hpfc.history"

#define SHELL_ESCAPE "! "
#define CHANGE_DIR   "cd "
#define QUIT         "quit"

/* the lexer is quite simple:-)
 */
#define PREFIX_EQUAL_P(str, prf) (strncmp(str, prf, strlen(prf))==0)

/*  returns the full hpfc history file name, i.e.
 *  - $HPFC_HISTORY (if any)
 *  - $HOME/"HIST"
 */
static char *default_hist_file_name()
{
    char *home, *hist = getenv("HPFC_HISTORY");

    if (hist) return(hist);

    /* else builds the default name.
     */
    home = getenv("HOME");
    return(sprintf((char*) malloc(sizeof(char)*(strlen(home)+strlen(HIST)+2)),
		   "%s/%s", home, HIST));
}

/* main: interactive loop and history management.
 */
int main()
{
    char *last = NULL, *line = NULL, *file_name = default_hist_file_name();
    
    /*  initialize history
     */
    using_history();
    read_history(file_name);
    
    /*  interactive loop
     */
    while ((line=readline(HPFC_PROMPT)))
    {
	if (PREFIX_EQUAL_P(line, QUIT)) 
	    break;
	else if (PREFIX_EQUAL_P(line, CHANGE_DIR))
	{
	    if (chdir(line+strlen(CHANGE_DIR)))
		fprintf(stderr, "error while changing directory\n");
	}
	else if (PREFIX_EQUAL_P(line, SHELL_ESCAPE))
	{
	    /*   the shell escape is directly executed. That easy!
	     */
	    system(line+strlen(SHELL_ESCAPE));
	}
	else
	{
	    /*   calls a script:-)
	     *   All this stuff could be taken care of in the C, but shell
	     *   scripts are much easier to develop:-) 
	     */
	    char *shll = (char*)
		malloc(sizeof(char)*(strlen(HPFC_PREFIX)+strlen(line)+2));

	    system(sprintf(shll, "%s %s", HPFC_PREFIX, line));

	    free(shll);
	}

	/*   add to history if not the same as the last one (in this session)
	 */
	if ((last && line && strcmp(last, line)!=0) || (line && !last))
	    add_history(line), last = line; 
	else
	    free(line), line = NULL;
	    
    }

    if (!line) fprintf(stdout, "\n"); /* for Ctrl-D terminations */

    /*   close history
     */
    write_history(file_name);
    history_truncate_file(file_name, 100);

    return(0);
}

/*   that is all
 */
