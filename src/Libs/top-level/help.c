/*

  $Id$

  Copyright 1989-2009 MINES ParisTech

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
/* $Id$
 */

#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "genC.h"
#include "constants.h"
#include "misc.h"

#include "top-level.h"

#define BEGIN_STR	"BEGIN"
#define END_STR 	"END"

void 
get_help_topics(gen_array_t array)
{
    int index = 0, begin_length;
    FILE *fd;
    static char *help_file = NULL;

    char *line;

    if (help_file == NULL)
	help_file = XV_HELP_FILE;

    begin_length = strlen(BEGIN_STR);

    if ((fd = fopen(help_file, "r")) == NULL) {
	/* should be show_message */
	user_log("Could not open help file (%s)\n", help_file);
    }

    while ((line = safe_readline(fd)) != NULL) {
	if (strncmp(line, BEGIN_STR, begin_length) == 0)
	    gen_array_dupaddto(array, index++, line+begin_length+1);
	free(line);
    }

    fclose(fd);
}

void 
get_help_topic(string topic, gen_array_t array)
{
    FILE *fd;
    int selected = FALSE, index=0;
    static char *help_file;
    int begin_length, end_length;

    char *line;

    if (help_file == NULL)
	help_file = XV_HELP_FILE;

    begin_length = strlen(BEGIN_STR);
    end_length = strlen(END_STR);

    if ((fd = fopen(help_file, "r")) == NULL) {
	perror("Could not open help file\n");
	/* should be show_message */
	user_log("Could not open help file (%s)\n", help_file);
    }
    else {
	while ((line = safe_readline(fd)) != NULL) 
	{
	    if (strncmp(line, BEGIN_STR, begin_length) == 0 &&
		strcmp(line + begin_length + 1, topic) == 0) {
		selected = TRUE;
	    } else if (strncmp(line, END_STR, end_length) == 0) {
		if (selected) break;
	    }
	    else if (selected) 
		gen_array_dupaddto(array, index++, line);
	    free(line);
	}
    }

    if (! selected)
	gen_array_dupaddto(array, index++, "Sorry: no help on this topic");

    fclose(fd);
}

/* add checkings here (FI: why in help.c?)
 */
void 
pips_checks(void)
{
    if (!getenv("PIPS_ROOT")) {
	(void) fprintf(stderr, "PIPS_ROOT environment variable not set. \n"
		       "Set it properly and relaunch.\n");
	exit(1);
    }
}

