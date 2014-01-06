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
/* Storage for arguments of tpips commands. E.g.
 display PRINTED_FILE[MOD_A, MOD_B]
 display PRINTED_FILE
*/
typedef struct {
  gen_array_t the_owners; /* MOD_A, MOD_B, %ALL after expansion, default
                             value can be PROGRAM or current_module */
  string the_name; /* e.g. PRINTED_FILE */
} res_or_rule;

extern int tp_lex();
extern int tp_parse();
extern void tp_error();
extern void tp_init_lex();
extern void tp_begin_key();
extern void tp_begin_fname();

extern FILE * tp_in;
#ifdef FLEX_SCANNER
extern void tp_restart(FILE *);
#endif

#define TPIPS_PRIMARY_PROMPT 	"tpips> " /* prompt for readline  */
#define TPIPS_REQUEST_PROMPT    "tpips-request> "
#define TPIPS_SECONDARY_PROMPT 	"> "
#define TPIPS_CONTINUATION_CHAR '\\'

#define TPIPS_HISTENV "TPIPS_HISTORY"	/* history file env variable */
#define TPIPS_HISTORY_LENGTH 100	/* max length of history file */
#define TPIPS_COMMENT_PREFIX '#'	/* comment prefix */
#define TPIPS_HISTORY ".tpips.history" 	/* default history file */

#define SHELL_ESCAPE "shell" 		/* ! used for history reference */
#define CHANGE_DIR   "cd"
#define TPIPS_SOURCE "source"

#define SET_ENV	     "setenv"
#define GET_ENV	     "getenv"

#define SET_PROP     "setproperty"
#define GET_PROP     "getproperty"

#define QUIT         "quit"
#define HELP         "help"
/* macro ECHO is reserved by flex */
#define ECHO_N         "echo"

#define skip_blanks(str) \
  while (*str && (*str==' ' || *str=='\t' || *str=='\n')) str++

// redundant declarations to help bootstrap?
extern bool tpips_execution_mode;
extern bool tpips_is_interactive;
extern bool jpips_is_running;
extern bool tpips_init_done;
