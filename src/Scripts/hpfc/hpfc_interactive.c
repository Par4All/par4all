/* $RCSfile: hpfc_interactive.c,v $ (version $Revision$)
 * $Date: 1995/04/26 17:17:27 $, 
 *
 * interactive interface to hpfc, based on the GNU realine library.
 */

#include <stdio.h>
#include <strings.h>

extern int system();
extern int fprintf();
extern char *getenv();
extern int malloc();
extern void free();

#include "readline/readline.h"
#include "readline/history.h"

#define HPFC_PROMPT "hpfc> "
#define HPFC_PREFIX "hpfc"
#define HIST ".hpfc.history"

#define QUIT "qui"

/*  returns the full hpfc history file name, i.e.
 *  - $HPFC_HISTORY (if any)
 *  - $HOME/"HIST"
 */
static char *default_hist_file_name()
{
    char 
	*hist = getenv("HPFC_HISTORY"),
	*home;

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
    char 
	*file_name = default_hist_file_name(),
	*shll = NULL,
	*last = NULL,
	*line = NULL;
    
    /*  initialize history
     */
    using_history();
    read_history(file_name);
    
    /*  interactive loop
     */
    while ((line=readline(HPFC_PROMPT)))
    {
	if (strncmp(line, QUIT, strlen(QUIT))==0) /* quit! */
	    break;

	/*   calls a script:-)
	 *   All this stuff could be taken care of in the C, but shell
	 *   scripts are much easier to develop. 
	 */
	shll = (char*) 
	    malloc(sizeof(char)*(strlen(HPFC_PREFIX)+strlen(line)+2));

	system(sprintf(shll, "%s %s", HPFC_PREFIX, line));

	free(shll), shll = NULL;

	/*   add to history if not the same as the last one (in this session)
	 */
	if ((last && line && strcmp(last, line)!=0) || (line && !last))
	    add_history(line),  last = line; 
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
