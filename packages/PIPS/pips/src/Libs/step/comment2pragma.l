%{
#include "defines-local.h"

static statement current_statement = statement_undefined;

static char *directive_txt = NULL;
static char *comment_txt = NULL;

static void directive_start(int type);
static void directive_reset(void);
static void append(char **buffer);
static void clean_buffer(char **buffer);
static void end_directive(void);
static void end_comment(void);

static int directive_syntaxe = -1;
static int directive_type = -1;

#define DIRECTIVE_STEP 0
#define DIRECTIVE_OMP 1

%}
%option warn
%option noyywrap
%option nounput
%option noinput

 /*
   Attention PIPS supprime les blancs en début de commentaire en syntaxe libre, et insère un saut de ligne supplémentaire :
   |     !txt1
   |     !txt2
   devient :
   |!txt1
   |
   |!txt2
   |
  */

comment_fixe [cC*!]
comment_libre [:blank:]+!

continuation_fixe [^[:blank:]0]
continuation_libre ([:blank:]*&[:blank:]*)?

sentinelle_step $(?i:step)
sentinelle_omp $(?i:omp)
sentinelle {sentinelle_step}|{sentinelle_omp}


%x comment_fixe  directive_fixe  directive_fixe_comment  directive_fixe_continuation
%x comment_libre directive_libre directive_libre_comment directive_libre_continuation
%x other_line

%%

 /*
  Directive à syntaxe libre
 */
<directive_libre>[^&!\n]* {
  append(&directive_txt); // ajout du texte de la directive
  BEGIN(directive_libre_comment);
 }

<directive_libre_comment>&[^!\n]* { // détection d'une continuation du texte de la directive sur la ligne suivante
  BEGIN(directive_libre_continuation);
 }

<directive_libre_continuation>{ // controle de la continuation
  \n{comment_libre}{sentinelle_omp}{continuation_libre} {
    if (directive_type != DIRECTIVE_OMP)
      yy_fatal_error("erreur continuation omp libre\n");
    BEGIN(directive_libre);
  }
  \n{comment_libre}{sentinelle_step}{continuation_libre} {
    if (directive_type != DIRECTIVE_STEP)
      yy_fatal_error("erreur continuation omp libre\n");
    BEGIN(directive_libre);
  }
 }

 /*
  Directive à syntaxe fixe
 */
<directive_fixe>[^!\n]* {
  append(&directive_txt); // ajout du texte de la directive
  BEGIN(directive_fixe_comment);
 }

<directive_fixe_comment>\n{comment_fixe}/({sentinelle}{continuation_fixe}) { // détection d'une continuation du texte de la directive sur la ligne suivante
  BEGIN(directive_fixe_continuation);
 }

<directive_fixe_continuation>{ // controle de la continuation
  {sentinelle_omp}{continuation_fixe} {
    if (directive_type != DIRECTIVE_OMP)
      yy_fatal_error("erreur continuation omp fixe\n");
    BEGIN(directive_fixe);
  }
  {sentinelle_step}{continuation_fixe} {
    if (directive_type != DIRECTIVE_STEP)
      yy_fatal_error("erreur continuation step fixe\n");
    BEGIN(directive_fixe);
  }
 }

 /*
   Collecte d'un commentaire porté par une directive
 */
<directive_fixe_comment,directive_libre_comment,directive_libre_continuation>!.* {
  append(&comment_txt);
 }

 /*
   Détection de la fin d'une directive et de son commentaire associé éventuel
 */
<directive_fixe_comment,directive_libre_comment>\n {
  append(&comment_txt);

  end_directive();

  directive_reset();
  BEGIN(INITIAL);
 }

 /*
   Identification d'une nouvelle directive
 */
<comment_fixe,comment_libre>{
  {sentinelle_omp}[[:blank:]0] {
    directive_start(DIRECTIVE_OMP);
  }
  {sentinelle_step}[[:blank:]0] {
    directive_start(DIRECTIVE_STEP);
  }
 }

 /*
   Identification des commentaires et autres lignes
 */
<comment_fixe,comment_libre>\n|([^$\n].*\n) {
  append(&comment_txt);
  end_comment();
  clean_buffer(&comment_txt);
  BEGIN(INITIAL);
 }
<other_line>.*\n {
  append(&comment_txt);
  end_comment();
  clean_buffer(&comment_txt);
  BEGIN(INITIAL);
 }


<INITIAL>{
  ^{comment_fixe} {
     BEGIN(comment_fixe);
     yymore();
   }
  ^{comment_libre} {
     BEGIN(comment_libre);
     yymore();
   }
  ^\n {
     append(&comment_txt);
     end_comment();
     clean_buffer(&comment_txt);
   }
  ^. {
     BEGIN(other_line);
     yymore();
   }
 }

%%

static void directive_start(int type)
{
  directive_type = type;

  switch (YY_START)
    {
    case comment_fixe:
      directive_syntaxe = directive_fixe;
      BEGIN(directive_fixe);
      break;
    case comment_libre:
      directive_syntaxe = directive_libre;
      BEGIN(directive_libre);
      break;
    default:
      yy_fatal_error("erreur syntaxe\n");
    }
}

static void directive_reset(void)
{
  clean_buffer(&directive_txt);
  clean_buffer(&comment_txt);
  directive_syntaxe = -1;
  directive_type = -1;
}

static void append(char **buffer)
{
  if (*buffer == NULL)
    *buffer = strdup(yytext);
  else
    {
      *buffer = realloc(*buffer, sizeof(*buffer) * (strlen(*buffer) + yyleng));
      *buffer = strncat(*buffer, yytext, yyleng);
    }
}

static void clean_buffer(char **buffer)
{
  if(*buffer != NULL)
    {
      free(*buffer);
      *buffer = NULL;
    }
}

static void remove_blank(void)
{
  char *dest;
  char *src;

  dest = directive_txt;
  for(src=directive_txt; *src !='\0'; src++)
    if (*src != ' ' && *src != '\t')
      {
	*dest=*src;
	    dest++;
      }
  *dest = '\0';
}

static char *get_directive_type(void)
{
  switch (directive_type)
    {
    case  DIRECTIVE_STEP:
      return strdup("step");
      break;
    case DIRECTIVE_OMP:
      return strdup("omp");
      break;
    default:
      return NULL;
      break;
    }
}


static void end_comment(void)
{
  if (!statement_undefined_p(current_statement))
    {
      pips_debug(2,"COMMENTAIRE_TXT-> \"%s\"\n", comment_txt);
      append_comments_to_statement(current_statement, comment_txt);
    }
  else
    {
      ifdebug(1)
	{
	  if (comment_txt !=NULL)
	    pips_debug(1,"%s :%s",(YY_START==other_line)?"OTHER_LINE":"COMMENT", comment_txt);
	  else
	    pips_debug(1,"\n");
	}
    }
}

static void end_directive(void)
{
  string directive_type = get_directive_type();

  // suppression du saut de ligne
  comment_txt[strlen(comment_txt)-1]='\0';

  if (directive_syntaxe == directive_fixe)
    remove_blank();

  pips_debug(2,"DIRECTIVE-> %s \"%s\"\n", directive_type, directive_txt);
  pips_debug(2,"\tCOMMENTAIRE_DIR-> \"%s\"\n", comment_txt);

  if(!statement_undefined_p(current_statement))
    {
      extensions_extension(statement_extensions(current_statement))=gen_nreverse(extensions_extension(statement_extensions(current_statement)));
      add_pragma_str_to_statement(current_statement, strdup(concatenate(directive_type,directive_txt,NULL)), false);
      extensions_extension(statement_extensions(current_statement))=gen_nreverse(extensions_extension(statement_extensions(current_statement)));

      if (strlen(comment_txt))
	{
	  if (!empty_comments_p(statement_comments(current_statement)))
	    {
	      free(statement_comments(current_statement));
	      statement_comments(current_statement) = string_undefined;
	    }
	  append_comments_to_statement(current_statement, comment_txt);
	}
    }
  else
    {
      pips_debug(1, "DIRECTIVE in decls_txt\n");
    }

  free(directive_type);
}

/*
 * For fortran programs
 * converting comments into pragmas
 *
*/

void step_comment2pragma_handle(statement stmt)
{
  pips_debug(1,"begin\n");
  if(!statement_undefined_p(stmt))
    {
      string comment = statement_comments(stmt);

      if (!empty_comments_p(comment))
	{
	  pips_assert("statement undefined", statement_undefined_p(current_statement));
	  current_statement = stmt;
	  statement_comments(stmt) = string_undefined;

	  pips_debug(1,"Comment txt : begin scan\n");
	  yy_scan_string(comment);
	  yylex();
	  pips_debug(1,"Comment txt : end scan\n");

	  free(comment);
	  current_statement = statement_undefined;
	}
      else
	{
	  pips_debug(1,"Comment txt : Empty\n");
	}
    }
  else
    {
      string decls_txt = code_decls_text(entity_code(get_current_module_entity()));
      if (!empty_comments_p(decls_txt))
	{
	  pips_debug(1,"Declaration txt : begin scan\n");
	  yy_scan_string(decls_txt);
	  yylex();
	  pips_debug(1,"Declaration txt : end scan\n");
	}
      else
	{
	  pips_debug(1,"Declaration txt : Empty\n");
	}
    }
  pips_debug(1,"end\n");
}
