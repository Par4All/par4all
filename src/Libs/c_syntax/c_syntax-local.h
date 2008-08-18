/* $Id$ */
  
#include "c_parser_private.h"
extern FILE * c_in; /* the file read in by the c_lexer */

/* Functions declared in clex.l because they use internal variables of the lexer */

extern int get_current_C_line_number(void);
extern void set_current_C_line_number(void);
extern void reset_current_C_line_number(void);
extern void error_reset_current_C_line_number(void);
extern int pop_current_C_line_number();
extern void push_current_C_line_number();

extern string get_current_C_comment(void);
extern string pop_current_C_comment(void);
extern void push_current_C_comment(void);
extern void update_C_comment(string a_comment);
extern void discard_C_comment(void);
extern void reset_C_comment(bool);
extern void clear_C_comment();
extern void init_C_comment();

/* Functions declared in cyacc.y because they are relate to variable
   ContextStack, private to the parser */

/* These functions could be moved elsewhere, e.g. in a future scope.c */
extern string scope_to_block_scope(string);
extern string pop_block_scope(string);
extern bool string_block_scope_p(string);
extern string empty_scope(void);
extern bool empty_scope_p(string);
extern string scope_to_block_scope(string);
/* The following functions are using parser variables*/
extern void InitScope(void);
extern void EnterScope(void);
extern string GetScope(void);
extern c_parser_context GetContext(void);
extern void ExitScope(void);
