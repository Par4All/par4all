/* $RCSfile: hpfc_interactive.c,v $ (version $Revision$)
 * $Date: 1995/04/24 15:43:58 $, 
 */

#include <stdio.h>
#include <strings.h>

extern int system();
extern int fprintf();

#include "readline/readline.h"
#include "readline/history.h"

#define HPFC_PROMPT "hpfc> "
#define BUFFER_SIZE 1024
#define FILE_NAME NULL

#define QUIT "qui"

int main()
{
    char *line, buffer[BUFFER_SIZE];
    
    using_history();
    read_history(FILE_NAME);
    
    while ((line=readline(HPFC_PROMPT)))
    {
	if ((strlen(line)+strlen(HPFC_PROMPT))>=BUFFER_SIZE) /* woh! */
	{
	    fprintf(stderr, "line too long\n");
	    continue;
	}	    

	if (strncmp(line, QUIT, 3)==0) /* berk */
	    break;

	/*   calls a script:-)
	 *   All this stuff could be taken care of in the C, but shell
	 *   scripts are much easier to develop. 
	 */
	system(sprintf(buffer, "hpfc %s", line));

	/*   maybe a memory leak, history is not very well documented.
	 */
	add_history(line); 
    }

    write_history(FILE_NAME);

    fprintf(stdout, "\n"); /* usefull for Ctrl-D terminations */
    return(0);
}

/*   that is all
 */
