/* $Id$ */
  
#include "c_parser_private.h"
extern FILE * c_in; /* the file read in by the c_lexer */

/* Functions declared in clex.l */
extern int get_current_C_line_number(void);
extern void set_current_C_line_number(void);
extern void reset_current_C_line_number(void);

extern string get_current_C_comment(void);
extern string pop_current_C_comment(void);
extern void push_current_C_comment(void);
extern void update_C_comment(string a_comment);
extern void discard_C_comment(void);
extern void reset_C_comment(bool);
extern void clear_C_comment();
extern void init_C_comment();
