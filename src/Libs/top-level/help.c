#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <stdlib.h>

#include "genC.h"
#include "constants.h"
#include "misc.h"

#include "top-level.h"

void get_help_topics(pargc, argv)
int *pargc;
char *argv[];
{
    FILE *fd;

    static char *help_file = NULL;
    static char *begin_string = "BEGIN";
    static int begin_length = 0;

    char *line;

    if (help_file == NULL)
	help_file = XV_HELP_FILE;

    if (begin_length == 0)
	begin_length = strlen(begin_string);

    if ((fd = fopen(help_file, "r")) == NULL) {
	/* should be show_message */
	user_log("Could not open help file (%s)\n", help_file);
    }

    while ((line = read_line(fd)) != NULL) {
	if (strncmp(line, begin_string, begin_length) == 0) {
	    args_add(pargc, argv, strdup(line + begin_length + 1));
	}
    }

    fclose(fd);
}

void get_help_topic(topic, pargc, argv)
char *topic;
int *pargc;
char *argv[];
{
    int selected = FALSE;
    FILE *fd;

    static char *begin_string = "BEGIN";
    static int begin_length = 0;

    static char *end_string = "END";
    static int end_length = 0;

    static char *help_file;

    char *line;

    if (help_file == NULL)
	help_file = XV_HELP_FILE;

    if (begin_length == 0)
	begin_length = strlen(begin_string);

    if (end_length == 0)
	end_length = strlen(end_string);

    if ((fd = fopen(help_file, "r")) == NULL) {
	perror("Could not open help file\n");
	/* should be show_message */
	user_log("Could not open help file (%s)\n", help_file);
    }
    else {
	while ((line = read_line(fd)) != NULL) {
	    if (strncmp(line, begin_string, begin_length) == 0 &&
		strcmp(line + begin_length + 1, topic) == 0) {
		selected = TRUE;
	    }
	    else if (strncmp(line, end_string, end_length) == 0) {
		if (selected)
		    break;
	    }
	    else {
		if (selected) {
		    args_add(pargc, argv, strdup(line));
		}
	    }
	}
    }

    if (! selected) {
	args_add(pargc, argv, "Sorry: no help on this topic");
    }

    fclose(fd);
}
