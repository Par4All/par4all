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

/* Here are all the stuff to interface Pips with Emacs. */

/* Ronan.Keryell@cri.ensmp.fr, 23/05/1995. */

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <strings.h>

#include <unistd.h>
#include <sys/ioctl.h>
#ifndef __linux
#include <sys/filio.h>
#endif

#include <xview/xview.h>
#include <xview/panel.h>
#include <xview/notify.h>

#include "string.h"
#include "genC.h"
#include "misc.h"
#include "linear.h"
#include "ri.h"
#include "database.h"
#include "ri-util.h"
#include "pipsdbm.h"
#include "properties.h"

/* Include the label names: */
#include "wpips-labels.h"

#include "wpips.h"

/* This variable is used to indicate wether wpips is in the Emacs
   mode: */
/* By default, wpips is not called from emacs. RK. */
bool wpips_emacs_mode = 0;

/* The title of the commands used by the emacs interface: */

static char EMACS_AVAILABLE_MODULES_NAME[] = "AVAILABLE_MODULES";
static char EMACS_MODULE_NAME[] = "MODULE_NAME";
static char EMACS_PROMPT_USER[] = "PROMPT_USER";
static char EMACS_SEQUENTIAL_VIEW[] = "Sequential View";
static char EMACS_USER_ERROR[] = "USER_ERROR";
static char EMACS_USER_LOG[] = "USER_LOG";
static char EMACS_USER_WARNING[] = "USER_WARNING";
static char EMACS_WINDOW_NUMBER[] = "WINDOW_NUMBER";
static char EMACS_NEW_DAVINCI_CONTEXT[] = "NEW_DAVINCI_CONTEXT";
static char EMACS_VIEW_DAVINCI_GRAPH[] = "VIEW_DAVINCI_GRAPH";

/* The token to mark the begin and end of command. Use some strings to
   be usually never used in ISO-8859-1: */
static char epips_begin_of_command_token[] = "\200";
static char epips_end_of_command_token[] = "\201";
static char epips_receive_begin_of_command_token[] = "\202";
static char epips_receive_end_of_command_token[] = "\203";

/* At initialization, we are waiting for an input command: */
typedef enum
{
   epips_wait_for_begin,
      epips_wait_for_end
} epips_input_automaton_states;

static epips_input_automaton_states
epips_input_automaton_state = epips_wait_for_begin;



/* Here are described all the functions used to send informations to
   emacs: */


/* Just send some raw text to emacs: */
static void
send_text_to_emacs(char * some_text) 
{
   (void) printf("%s", some_text);
   fflush(stdout);
}


/* The function that frames the command to emacs: */
static void
send_command_to_emacs(char * command_title, char * command_content)
{
   send_text_to_emacs(epips_begin_of_command_token);
   send_text_to_emacs(command_title);
   /* Separate the command name from the content with a ":" : */
   send_text_to_emacs(":");
   send_text_to_emacs(command_content);
   send_text_to_emacs(epips_end_of_command_token);
}


/* Here are all the methods used to send an object to Emacs: */

void
send_module_name_to_emacs(char * some_text)
{
   send_command_to_emacs(EMACS_MODULE_NAME, some_text);
}


/* Tell Emacs about what are the modules available in the current
   workspace, if any: */
void
send_the_names_of_the_available_modules_to_emacs(void)
{
    if (wpips_emacs_mode) {
	char * module_string_list_string = strdup("(");
   
	if (db_get_current_workspace_name() != NULL) {
	    gen_array_t modules = db_get_module_list();
	    int module_list_length = gen_array_nitems(modules), i;
	    for(i = 0; i < module_list_length; i++) {
		char * new_module_string_list_string =
		    strdup(concatenate(module_string_list_string,
				       "\"", gen_array_item(modules, i), "\" ",
				       NULL));
		free(module_string_list_string);
		module_string_list_string = new_module_string_list_string;
	    }
	    gen_array_full_free(modules);
	}
	send_command_to_emacs(EMACS_AVAILABLE_MODULES_NAME,
			      concatenate(module_string_list_string,
					 ")", NULL));
	free(module_string_list_string);
    }
}


void
send_prompt_user_to_emacs(char * some_text)
{
   send_command_to_emacs(EMACS_PROMPT_USER, some_text);
}


void
send_user_error_to_emacs(char * some_text)
{
   send_command_to_emacs(EMACS_USER_ERROR, some_text);
}


void
send_user_log_to_emacs(char * some_text)
{
   send_command_to_emacs(EMACS_USER_LOG, some_text);
}


void
send_user_warning_to_emacs(char * some_text)
{
   send_command_to_emacs(EMACS_USER_WARNING, some_text);
}


void
send_view_to_emacs(char * view_name, char * the_file_name)
{
   unsigned int number_of_characters_written;
   char full_path[1000];
   
   /* Send a complete file path since the current directory in Emacs
      is no the same a priori: */
   (void) sprintf(full_path, "%s/%s%n", get_cwd(), the_file_name,
                  &number_of_characters_written);
   pips_assert("send_view_to_emacs",
               number_of_characters_written < sizeof(full_path));
   
   send_command_to_emacs(view_name, full_path);
}


void
send_window_number_to_emacs(int number)
{
   char a_string[10];

   (void) sprintf(a_string, "%d", number);
   send_command_to_emacs(EMACS_WINDOW_NUMBER, a_string);
}



void
send_notice_prompt_to_emacs(char *first_line, ...)
{
   va_list ap;
   char * prompt_string;
   
   va_start(ap, first_line);

   send_prompt_user_to_emacs(first_line);
   while((prompt_string = va_arg(ap, char *)) != NULL)
      send_prompt_user_to_emacs(prompt_string);

   va_end(ap);
}


void
ask_emacs_to_open_a_new_daVinci_context()
{
    send_command_to_emacs(EMACS_NEW_DAVINCI_CONTEXT, "");
}


void
ask_emacs_to_display_a_graph(string file_name)
{
    send_command_to_emacs(EMACS_VIEW_DAVINCI_GRAPH, file_name);
}


/* Here are described all the functions used to receive informations
   from emacs: */


/* Emacs said to select a module: */
static bool
epips_select_module(char * module_name)
{
    gen_array_t modules;
    int new_module_list_length;
    int module_list_length = 0;
    int i = 0;

   if (db_get_current_workspace_name() == NULL) {
      user_warning("epips_select_module",
                 "No workspace selected or created yet !\n");
      return FALSE;
   }

   modules = db_get_module_list();
   module_list_length = gen_array_nitems(modules);

   if (module_list_length == 0)
   {
      /* If there is no module... */
      prompt_user("No module available in this workspace");
   }
   else {
      /* Just to be sure that the selected module exist: */
      for(i = 0; i < module_list_length; i++) {
         if (strcmp(module_name, gen_array_item(modules, i)) == 0) {
            end_select_module_notify(module_name);
            break;
         }
      }
      if (i == module_list_length)
         user_warning("epips_select_module",
                      "The module \"%s\" does not exist.\n", module_name);
   }
   /* args_free zeroes also its length argument... */
   new_module_list_length = module_list_length;
   gen_array_full_free(modules);

   if (module_list_length == 0 || i == module_list_length)
      /* Something went wrong. */
      return FALSE;

   return TRUE;
}


/* Emacs said to display a sequential view: */
static void
epips_execute_and_display_something(char * view_label, char * module_name)
{
   bool module_selection_result = TRUE;
   
   if (module_name != '\0')
      /* If module_name is not an empty string, we need to select this
         module first: */
      module_selection_result = epips_select_module(module_name);
   
   if (module_selection_result)
      /* Display something only if an eventual module selection has
         been successful: */
      wpips_execute_and_display_something_from_alias(view_label);
}


/* Emacs said to display a sequential view: */
static void
epips_sequential_view(char * module_name)
{
   epips_execute_and_display_something(EMACS_SEQUENTIAL_VIEW, module_name);
}


static void
epips_execute_command(char * command_buffer)
{
   char * command_name, * command_content;
   
   /* Separate the command name from the content with a ":" : */
   char * separator_index = (char *) index(command_buffer, ':');
   debug(2, "epips_execute_command", "Command: \"%s\"\n", command_buffer);

   if (separator_index == NULL) {
      user_warning("epips_execute_command",
                   "Cannot understand command: \"%s\"\n", command_buffer);
      return;
   }

   command_name = command_buffer;
   *separator_index = '\0';
   command_content = separator_index + 1;
   debug(2, "epips_execute_command",
         "command_name: \"%s\", command_content: \"%s\"\n",
         command_name,
         command_content);

   /* Now we can choose what command to execute: */
   if (strcmp(command_name, EMACS_MODULE_NAME) == 0)
      epips_select_module(command_content);
   else if (strcmp(command_name, EMACS_SEQUENTIAL_VIEW) == 0)
      epips_sequential_view(command_content);
   else {
      user_warning("epips_execute_command",
                 "Cannot understand command \"%s\" with argument \"%s\"\n",
                 command_name, command_content);
   }
}


static void
trow_away_epips_input(char * entry_buffer, long int length)
{
   if (length > 0) {
      /* By default, send what we do not understand to stderr: */
      /* I can't remember how to give an argument to %...c ...*/
      (void) write(fileno(stderr), entry_buffer, length);
   }
}


enum { EPIPS_COMMAND_BUFFER_SIZE = 2000 };
static char command_buffer[EPIPS_COMMAND_BUFFER_SIZE];
static unsigned int command_buffer_length;
   

/* Copy a part of the entry_buffer in the command_buffer: */
static void
add_command_to_buffer(char * entry_buffer,
                      unsigned long int length)
{
   pips_assert("add_command_to_buffer in emacs.c: command too big !!!",
               length + command_buffer_length < sizeof(command_buffer) - 1);
   (void) memcpy(command_buffer + command_buffer_length,
                 entry_buffer,
                 length);
   command_buffer_length += length;
   command_buffer[command_buffer_length] = '\0';
}


/* Try to unframe one half of command sent by emacs. Thus needs an
   iterator outside. */
static char *
unframe_commands_from_emacs(char * entry_buffer, long int *length)
{
   if (epips_input_automaton_state == epips_wait_for_begin) {
      /* Wait for a begin of command: */
      char * epips_packet_begin_position = 
	  (char*) index(entry_buffer,
			epips_receive_begin_of_command_token[0]);
      debug(8, "unframe_commands_from_emacs",
            "epips_packet_begin_position = %8X\n", epips_packet_begin_position);
      if (epips_packet_begin_position == NULL) {
         /* No begin of command: */
         trow_away_epips_input(entry_buffer, *length);
         /* Return an empty string: */
         entry_buffer += *length;
         *length = 0;
         return entry_buffer;
      }
      else {
         /* Discard the begin of buffer: */
         trow_away_epips_input(entry_buffer,
                               epips_packet_begin_position - entry_buffer);
         /* Now we wait for an end of command: */
         epips_input_automaton_state = epips_wait_for_end;
         /* Empty the command buffer: */
         command_buffer_length = 0;
         /* Skip the begin token: */
         *length -= epips_packet_begin_position + 1 - entry_buffer;
         return epips_packet_begin_position + 1;
      }
   }
   else {
      /* Look for an end of command: */
      char * epips_packet_end_position = 
	  (char*) index(entry_buffer,
			epips_receive_end_of_command_token[0]);
      debug(8, "unframe_commands_from_emacs",
            "epips_packet_end_position = %8X\n", epips_packet_end_position);
      if (epips_packet_end_position == NULL) {
         /* No end of packet found yet. Add the content to the command
            buffer and keep waiting: */
         add_command_to_buffer(entry_buffer,
                               *length);
         entry_buffer += *length;
         *length = 0;
         return entry_buffer;
      }
      else {
         add_command_to_buffer(entry_buffer,
                               epips_packet_end_position - entry_buffer);
         epips_execute_command(command_buffer);
         /* Go in the wait for begin state: */
         epips_input_automaton_state = epips_wait_for_begin;
         /* Skip the end token: */
         *length -= epips_packet_end_position + 1 - entry_buffer;
         return epips_packet_end_position + 1;
      }
      
   }     
}


/* The function that accept commands from emacs: */
static Notify_value
read_commands_from_emacs(Notify_client client, int fd)
{
   char emacs_entry_buffer[1000];
   long int length;

   debug_on("EPIPS_DEBUG_LEVEL");
   
   if (ioctl(fd, FIONREAD, &length) == -1 || length == 0) {
      /* Nasty thing. In act, if stdin happens to close (for example I
         debugged the stuff with wpips -emacs < a), the notifier loop
         on read_commands_from_emacs(). Thus, in this case, we
         disinterest the notifier from stdin: */
      debug(3, "read_commands_from_emacs",
            "Detach the notifyer from stdin (length to read is %d)\n",
            length);
      (void) notify_set_input_func(client,
                                   NOTIFY_FUNC_NULL,
                                   fileno(stdin));
   }
   else {
      /* We have something to read: */
      long int read_length;
      char * entry_buffer = emacs_entry_buffer;
      do {
         read_length = read(fd,
                           emacs_entry_buffer,
                           sizeof(emacs_entry_buffer) - 1);
         if (read_length > 0) {
            long int analyze_length = read_length;
            
            /* OK, we have read something... */
            emacs_entry_buffer[length] = '\0';
            debug(9, "read_commands_from_emacs", "Read got \"%s\"\n",
                  emacs_entry_buffer);
            /* Try to parse until the buffer is empty: */
            while(*(entry_buffer
                    = unframe_commands_from_emacs(entry_buffer, &analyze_length)) != '\0')
               debug(8, "unframe_commands_from_emacs", "Return \"%s\"\n",
                  entry_buffer);
;
         }
         /* Do not try a blocking read: */
      } while (read_length > 0 && (length -= read_length) > 0);
   }
   debug_off();
   
   return NOTIFY_DONE;
}


/* The function to initialize some things in the emacs mode: */
void
initialize_emacs_mode()
{
   if (wpips_emacs_mode) {
      /* An arbitrary number to identify the notify client function: */
      Notify_client notifier_client = (Notify_client) 1234;
      
     /* The commands from emacs are read from stdin.

         Thus, we need to register stdin in the XView event Notifier: */
      (void) notify_set_input_func(notifier_client,
                                   read_commands_from_emacs,
                                   fileno(stdin));

      /* The user query is redirected to Emacs: */
      /* pips_request_handler = epips_user_request; */

      /* Initialize the epips.el epips-window-number variable: */
      send_window_number_to_emacs(INITIAL_NUMBER_OF_WPIPS_WINDOWS);
      
      /* Ask for Emacs prettyprinting: */
      set_bool_property("PRETTYPRINT_ADD_EMACS_PROPERTIES", TRUE);
      /* Ask Pips to display the declarations from the RI to have
         hypertext functions on the declarations: */
      set_bool_property("PRETTYPRINT_ALL_DECLARATIONS", TRUE);
      /* Since the comments of the declarations are not in the RI,
         pick them in the text: */
      set_bool_property("PRETTYPRINT_HEADER_COMMENTS", TRUE);      
   }
}
