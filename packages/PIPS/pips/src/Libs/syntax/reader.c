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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "parser_private.h"
#include "linear.h"
#include "ri.h"

#include "misc.h"
#include "ri-util.h"

#include "syntax.h"

/*
 * Fortran est un langage un peu ancien (je ne dirais pas  "tres  ancien"  car
 * nous  avons,  lui  et  moi, le meme age) dans lequel les blancs ne sont pas
 * significatifs, et dans lequel il n'y a pas de mot clefs  reserves.  Cela  a
 * pour consequence que l'analyse lexicale de Fortran est delicate, et ne peut
 * etre effectuee avec lex.  Plus exactement,  il  semble  que  les  nouvelles
 * versions  de  lex,  avec 'look-ahead', permette d'analyser Fortran, mais je
 * n'ai pas explore cette voie.
 *
 * J'ai prefere utiliser lex  en  lui  fournissant  une  fonction  'getc'  qui
 * permette de lever les difficultes liees a Fortran.
 *
 * La nouvelle fonction getc fonctionne de la facon suivante.  Getc  lit  d'un
 * seul  coup  toutes  les  lignes  d'une instruction Fortran, c'est a dire la
 * ligne initiale et les 19 eventuelles lignes de continuation, et les  stocke
 * dans  le  buffer  'Stmt'.  Au  vol,  getc  repere le label, enleve tous les
 * blancs, detecte les caracteres entre  simples  quotes,  et  met  a  jour  4
 * variables   externes,  qui  representent  pour  l'instruction  courante  la
 * premiere et la derniere ligne commentaire, et la premiere  et  la  derniere
 * ligne  source.  Ensuite,  le  contenu  du  buffer  Stmt  est analyse pour y
 * detecter les mot clefs, c'est a dire traiter les cas des  instructions  IF,
 * ELSEIF,   ASSIGN,  DO,  des  declaratives  IMPLICIT  et  FUNCTION,  et  des
 * operateurs '.XX.' (.EQ., .NEQV., ...).
 *
 * Lorsqu'un mot clef est detecte, il est mis  en  minuscules  dans  le  texte
 * source,  sauf  la  premiere lettre qui reste en majuscule.  Ainsi, lex peut
 * faire  la  difference  entre  le  mot  clef  'Assign'  et  l'identificateur
 * 'ASSIGN'.  Grace  a  la  premiere  lettre, lex peut detecter deux mots clef
 * successifs,   meme   sans   blanc   pour   les    separer,    comme    dans
 * 'IntegerFunctionASSIGN(X)'.
 *
 * Lorsqu'un operateur .XX. est detecte, il est remplace dans le source
 * par '%XX%'.  Ainsi, lex peut faire la difference entre une constante
 * reelle et un operateur, comme dans '(X+1.%EQ%5)'. It used to be '_' but
 * the underscore is replaced by percent to allow safely underscore in
 * identifiers.
 *
 * Nullary operators .TRUE. and .FALSE. are also converted but are later
 * seen as constants instead.
 *
 * Remi Triolet
 *
 * Modifications:
 *
 *  - the double quote character is not part of Fortran character set; it
 *    should not appear, even in comments or formats (Francois Irigoin)
 *
 *  - comments were associated with the wrong statement; iPrevComm and
 *    PrevComm were added to keep the right comment; syntax-local.h
 *    and statement.c were modified (Francois Irigoin)
 *
 *  - ReadLine: there was no check for comment buffer overflow
 *    (Francois Irigoin, 12 February 1992)
 *
 *  - replaced calls to Warning() by calls to pips_error() in CheckParenthesis
 *    to track a bug (Francois Irigoin, 21 February 1992)
 *    (Francois Irigoin, 7 June 1995)
 *
 *  - toupper moved from GetChar into ReadLine to keep lower case letters
 *    in character strings and comments; tex_util/print.c had to be changed
 *    too to avoid putting everything in upper case at prettyprint type;
 *    statement.c was also modified to put the declaration part in upper
 *    case as before while keeping comments in their original case
 *    (Function check_first_statement). See minuscule.f in Validation
 *    (Francois Irigoin, 7 June 1995)
 *
 *  - double quotes can be used instead of simple quotes for character
 *    string constants (Francois Irigoin, 11 novembre 1996)
 *
 *  - empty and invisible lines made of TAB and SPACE characters are preserved
 *    as comments in the executable part as they are in the declaration part
 *    (Francois Irigoin, 25 juillet 1997).

 */

/*-------------------------------------------------------------------------*/
/*
 * macros
 */
#define IS_QUOTED(c) (c>=256)
#define QUOTE(c) (c+256)
#define UNQUOTE(c) (IS_QUOTED(c) ? (c)-256 : (c))

/*-------------------------------------------------------------------------*/
/*
 *  definitions
 */
#define LOCAL static

#define UNDEF -2

#define FIRST_LINE		100
#define CONTINUATION_LINE 	101
#define EOF_LINE 		102

/*-------------------------------------------------------------------------*/
/*
 * declarations de variables externes locales
 */

/*********************************************************** COMMENT BUFFERS */

/* Comm contains the comments for the current statement in ReadStmt().
 * PrevComm contains the comments for the previous statement in ReadStmt(),
 * which is currently being parsed.
 * CurrComm contains the comments attached to the current line in ReadLine()
 */

#define INITIAL_BUFFER_SIZE (128)
char * Comm = NULL, * PrevComm = NULL, * CurrComm;
int iComm = 0, iPrevComm = 0, iCurrComm = 0;
static int CommSize = 0;
static int EofSeen = false;

/* lazy initialization of the comment buffer
 */
static void 
init_comment_buffers(void)
{
    if (CommSize!=0) return; /* if needed */
    pips_debug(9, "allocating comment buffers\n");
    CommSize = INITIAL_BUFFER_SIZE;
    Comm = (char*) malloc(CommSize);
    PrevComm = (char*) malloc(CommSize);
    CurrComm = (char*) malloc(CommSize);
    pips_assert("malloc ok", Comm && PrevComm && CurrComm);
}

static void
resize_comment_buffers(void)
{
    pips_debug(9, "resizing comment buffers\n");
    pips_assert("comment buffer is initialized", CommSize>0);
    CommSize*=2;
    Comm = (char*) realloc(Comm, CommSize);
    PrevComm = (char*) realloc(PrevComm, CommSize);
    CurrComm = (char*) realloc(CurrComm, CommSize);
    pips_assert("realloc ok", Comm && PrevComm && CurrComm);
}


/*********************************************************** GETCHAR BUFFER */

static int * getchar_buffer = NULL;
static int getchar_buffer_size = 0; /* number of elements in the array */

static void
init_getchar_buffer(void)
{
    if (getchar_buffer_size!=0) return; /* if needed */
    pips_debug(9, "allocating getchar buffer\n");
    getchar_buffer_size = INITIAL_BUFFER_SIZE;
    getchar_buffer = (int*) malloc(sizeof(int)*getchar_buffer_size);
    pips_assert("malloc ok", getchar_buffer);
}

static void
resize_getchar_buffer(void)
{
    pips_debug(9, "resizing getchar buffer\n");
    pips_assert("buffer initialized", getchar_buffer_size>0);
    getchar_buffer_size*=2;
    getchar_buffer = (int*) realloc(getchar_buffer, 
				    sizeof(int)*getchar_buffer_size);
    pips_assert("realloc ok", getchar_buffer);
}


static int i_getchar = UNDEF, l_getchar = UNDEF;


/*************************************************************** STMT BUFFER */

/* le buffer contenant le statement courant, l'indice courant et la longueur.
 */
static int * stmt_buffer = NULL;
static int stmt_buffer_size = 0; 

static void
init_stmt_buffer(void)
{
    if (stmt_buffer_size!=0) return; /* if needed */
    pips_debug(9, "allocating stmt buffer\n");
    stmt_buffer_size = INITIAL_BUFFER_SIZE;
    stmt_buffer = (int*) malloc(sizeof(int)*stmt_buffer_size);
    pips_assert("malloc ok", stmt_buffer);
}

static void
resize_stmt_buffer(void)
{
    pips_debug(9, "resizing stmt buffer\n");
    pips_assert("buffer initialized", stmt_buffer_size>0);
    stmt_buffer_size*=2;
    stmt_buffer = (int*) realloc(stmt_buffer, sizeof(int)*stmt_buffer_size);
    pips_assert("realloc ok", stmt_buffer);
}

/* indexes in the buffer...
 */
static size_t iStmt = 0, lStmt = 0;

/*************************************************************** LINE BUFFER */

/* le buffer contenant la ligne que l'on doit lire en avance pour se rendre
 * compte qu'on a finit de lire un statement, l'indice courant et la longueur.
 */
static int * line_buffer = NULL;
static int line_buffer_size = 0;

static void
init_line_buffer(void)
{
    if (line_buffer_size!=0) return; /* if needed */
    pips_debug(9, "allocating line buffer\n");
    line_buffer_size = INITIAL_BUFFER_SIZE;
    line_buffer = (int*) malloc(sizeof(int)*line_buffer_size);
    pips_assert("malloc ok", line_buffer);
}

static void
resize_line_buffer(void)
{
    pips_debug(9, "resizing line buffer\n");
    pips_assert("buffer initialized", line_buffer_size>0);
    line_buffer_size*=2;
    line_buffer = (int*) realloc(line_buffer, sizeof(int)*line_buffer_size);
    pips_assert("realloc ok", line_buffer);
}

static int iLine = 0, lLine = 0; 

void append_data_current_stmt_buffer_to_declarations(void)
{
    size_t i=0, j=0, column=6;
    char * tmp = (char*) malloc(lStmt+200), * ndecls, * odecls;
    code c = EntityCode(get_current_module_entity());

    for (; i<lStmt; i++, j++, column++) 
    {
      if (column==71) 
      {
	tmp[j++] = '\n';
	tmp[j++] = ' ';
	tmp[j++] = ' ';
	tmp[j++] = ' ';
	tmp[j++] = ' ';
	tmp[j++] = ' ';
	tmp[j++] = 'x';
	tmp[j++] = ' ';
	tmp[j++] = ' ';
	tmp[j++] = ' ';
	tmp[j++] = ' ';
	tmp[j++] = ' ';
	column = 10;
      }
      tmp[j] = (char) stmt_buffer[i]; /* int[] */
    }
    stmt_buffer[i]='\0';
    tmp[j] = '\0';

    odecls = code_decls_text(c);
    ndecls = strdup(concatenate(odecls, "! moved up...\n      DATA ", 
				tmp+4, 0));
    free(odecls);
    free(tmp);
    code_decls_text(c) = ndecls;
}

void parser_reset_all_reader_buffers(void)
{
    iLine = 0, lLine = 0;
    iStmt = 0, lStmt = 0;
    iCurrComm = 0;
    iComm = 0;
    iPrevComm = 0;
    i_getchar = UNDEF, l_getchar = UNDEF;
    EofSeen = false;
}

/*
 * Une variable pour traiter les quotes. Petit automate a 3 etats:
 *	NONINQUOTES 	on est a l'exterieur de quotes
 * 	INQUOTES	on etait dans l'etat NONINQUOTES et on a vu une quote 
 * 	INQUOTEQUOTE	on etait dans l'etat INQUOTES et on a vu un quote
 *
 *                           INQUOTEBACKSLASH
 *                              ^      x
 *                              |      |
 *                              \      v
 *      NONINQUOTES -----'----> INQUOTES -------'-----> INQUOTEQUOTE
 *      |   ^     ^             |   ^  ^                |          |
 *      |   |     |             |   |  |                |          |
 *      +-x-+     |             +-x-+  +--------'-------+          |
 *                |                                                |
 *                +----------------------x-------------------------+
 *
 *      x est un caractere quelconque different de '
 *
 * Modifications: 
 *  - la quote peut-etre simple-quote ou double-quote pour faire plaisir a 
 *    Fabien Coelho.
 * L'information est stockee lors de la rentree dans une constante chaine
 * de caracteres (variable QuoteChar).
 *  - ajout de l'etat INQUOTEBACKSLASH pour traiter les extensions 
 * non normalisees similaires aux chaines C
 *
 * Notes:
 *  - il faut rester compatible avec l'analyseur lexical scanner.l
 *  - les declarations sont relues par un autre analyseur pour en garder
 * le texte et rester fidele au source
 */
static int EtatQuotes;
#define NONINQUOTES 1
#define INQUOTES 2
#define INQUOTEQUOTE 3
#define INQUOTEBACKSLASH 4

/*
 * Numero de ligne et de colonne du fichier d'entree courant.
 */
LOCAL int LineNumber, Column;

/*
 * Line number of the statement in ReadStmt(),
 * which is currently being parsed.
 */
LOCAL int StmtLineNumber;


/*
 * Y a t il un '=' ou un ',' non parenthese ?
 */
LOCAL int ProfZeroVirg, ProfZeroEgal;

/* La table des operateurs du type '.XX.'.
 */
static char * OperateurPoints[] = {
		".NOT.",
		".AND.",
		".OR.",
		".EQV.",
		".NEQV.",
		".LT.",
		".LE.",
		".EQ.",
		".NE.",
		".GT.",
		".GE.",
		".TRUE.",
		".FALSE.",
		".INOUT.",
		".IN.",
		".OUT.",
		NULL
};

/*
 * La table keywtbl contient tous les keywords de la grammaire Fortran.  Elle
 * est fabriquee automatiquement a partir du fichier f77keywords et mise dans
 * le fichier keywtbl.h. Le champ 'keywstr' est le nom du mot clef, et le
 * champ 'keywval est sa valeur numerique pour echange entre le scanner et le
 * parser.
 */
struct Skeyword {
	char * keywstr;
	int keywval;
};

#include "keywtbl.h"

/* Une table pour accelerer les recherche des keywords. keywidx[X] indique le
 * rang dans keywtbl du premier mot clef commencant par X.
 */
static int keywidx[26];

/* Variables qui serviront a mettre a jour les numeros de la premiere et de la
 * derniere ligne de commentaire, et les numeros de la premiere et de la
 * derniere ligne du statement.
 */
static int tmp_b_I, tmp_e_I, tmp_b_C, tmp_e_C;
static char tmp_lab_I[6];

/* memoization des properties */

#include "properties.h"

static bool parser_warn_for_columns_73_80 = true;

void 
init_parser_reader_properties()
{
    parser_warn_for_columns_73_80 = 
	get_bool_property("PARSER_WARN_FOR_COLUMNS_73_80");
    init_comment_buffers();
}


/*-------------------------------------------------------------------------*/
/*
 * declarations de vraies variables externes
 */

/*
 * les numeros de la premiere et de la derniere ligne de commentaire, les
 * numeros de la premiere et de la derniere ligne du statement, et le label du
 * statement.
 */
extern int line_b_I, line_e_I, line_b_C, line_e_C;
extern char lab_I[];

/*-------------------------------------------------------------------------*/
/*
 * declarations de fonctions externes
 */

void CheckParenthesis();
void FindIf();
void FindAutre();
void FindPoints();

int 
syn_wrap(void)
{
	return(1);
}

/* La fonction a appeler pour l'analyse d'un nouveau fichier.
 */
void 
ScanNewFile(void)
{
    register int i;
    static int FirstCall = true;
    char letcour, *keywcour;


    if (FirstCall) {
	FirstCall = false;

	/* on initialise la table keywidx */
	for (i = 0; i < 26; i += 1)
	    keywidx[i] = UNDEF;

	/* on met a jour la table keywidx en fonction des keywords */
	letcour = ' ';
	i = 0;
	while ((keywcour = keywtbl[i].keywstr) != NULL) {
	    if (keywcour[0] != letcour) {
		/* premier keyword commencant par keywcour[0] */
		keywidx[(int) keywcour[0]-'A'] = i;
		letcour = keywcour[0];
	    }
	    i += 1;
	}
    }

    /* on initialise les variables externes locales et non locales */
    LineNumber = 1;
    Column = 1;
    StmtLineNumber = 1;
    EtatQuotes = NONINQUOTES;
    iStmt = lStmt = UNDEF;
    iLine = lLine = UNDEF;
}

/* Fonction appelee par sslex sur la reduction de la regle de reconnaissance
 * des mot clefs. Elle recherche si le mot 's' est un mot clef, retourne sa
 * valeur si oui, et indique une erreur si non.
 */
int 
IsCapKeyword(char * s)
{
    register int i, c;
    char *kwcour, *t;
    char buffer[32];

    debug(9, "IsCapKeyword", "%s\n", s);
    
    pips_assert("not too long keyword", strlen(s)<32);

    /* la chaine s est mise en majuscules */
    t = buffer;
    while ( (c = *s++) ) {
	if (islower(c))
	    c = toupper(c);
	*t++ = c;
    }
    *t = '\0';

    i = keywidx[(int) buffer[0]-'A'];

    if (i != UNDEF) {
	while ((kwcour = keywtbl[i].keywstr)!=0 && kwcour[0]==buffer[0]) {
	    if (strcmp(buffer, kwcour) == 0) {
		debug(9, "IsCapKeyword", "%s %d\n", kwcour, i);
		return(keywtbl[i].keywval);
	    }

	    i += 1;
	}
    }

    user_warning("IsCapKeyword", "[scanner] keyword expected near %s\n",
		 buffer);
    ParserError("IsCapKeyword", "Missing keyword.\n");
    
    return(-1); /* just to avoid a gcc warning */
    /*NOTREACHED*/
}

/* Routine de lecture pour l'analyseur lexical, lex ou flex */
int 
PipsGetc(FILE * fp)
{
    int eof = false;
    int c;

    /* SG: UNDEF negative and iStmt of type size_t ...*/
    if (iStmt == UNDEF || iStmt >= lStmt) {
	/*
	 * le statement est vide. On lit et traite le suivant.
	 */
	if (ReadStmt(fp) == EOF) {
	    eof = true;
	}
	else {
	    /*
	     * verifie les parentheses et on recherche les '=' et
	     * les ',' de profondeur zero.
	     */
	    CheckParenthesis();

	    /*
	     * on recherche les operateurs du genre .eq.
	     */
	    FindPoints();

	    if (!FindDo()) {
		if (!FindImplicit()) {
		    if (!FindIfArith()) {
			FindIf();
						
			if (!FindAssign()) {
			    FindAutre();
			}
		    }
		}
	    }

	    iStmt = 0;
	}
    }

    c = stmt_buffer[iStmt++];
    return((eof) ? EOF : UNQUOTE(c));
}

/* Routine de lecture physique
 *
 * In case an error occurs, buffer must be emptied. 
 * Since i_getchar and l_getchar
 * cannot be touched by the error handling routine, changes of fp are tracked
 * in GetChar() and dynamically tested. Kludge suggested by Fabien Coelho to
 * avoid adding more global variables. (FI)
 *
 * Empty (or rather invisible) lines made of TAB and SPACE characters are 
 * replaced by the string "\n".
 */

int 
GetChar(FILE * fp)
{
    int c = UNDEF;
    static int col = 0;
    static FILE * previous_fp = NULL;

    init_getchar_buffer();

    /* This section (probably) is made obsolete by the new function 
     * parser_reset_all_reader_buffers(). The user_warning() is replaced
     * by a pips_error(). Test: Cachan/bug10
     */
    if( previous_fp != fp ) {
	/* If a file has just been opened */
	if( i_getchar < l_getchar ) {
	    /* if the buffer is not empty, which may never occur if 
	     * previous_fp == NULL, perform a buffer reset
	     */
	    i_getchar = l_getchar;
	    pips_internal_error("Unexpected buffer reset."
		       "A parser error must have occured previously.\n");
	}
	previous_fp = fp;
    }

    /* A whole input line is read to process TABs and empty lines */
    while (i_getchar >= l_getchar && c != EOF) {
	int EmptyBuffer = true;
	int LineTooLong = false;
	bool first_column = true;
	bool in_comment = false;

	i_getchar = l_getchar = 0;

	while ((c = getc(fp)) != '\n' && c != EOF) {

	    if (l_getchar>getchar_buffer_size-20) /* large for expansion */
		resize_getchar_buffer();

	    if(first_column) {
		in_comment = (strchr(START_COMMENT_LINE, (char) c)!= NULL);
		first_column = false;
	    }

	    /* Fortran has a limited character set. See standard section 3.1.
	       This cannot be handled here as you do not know if you are
	       in a string constant or not. You cannot convert the double
	       quote into a simple quote because you may generate an illegal
	       string constant. Maybe the best would be to uncomment the
	       next test. FI, 21 February 1992 
	    if( c == '\"')
		FatalError("GetChar","Illegal double quote character");
		" */
	    /* FI: let's delay and do it in ReadLine:
	     * if (islower(c)) c = toupper(c);
	     */

	    if (c == '\t') {
		int i;
		int nspace = 8-col%8;
		/* for (i = 0; i < (8-Column%8); i++) { */
		for (i = 0; i < nspace; i++) {
		    col += 1;
		    getchar_buffer[l_getchar++] = ' ';
		}
	    } else if (c == '\r') {
		/* Ignore carriage returns introduced by VMS, MSDOS or MACOS...*/
		;
	    }
	    else {
		col += 1;
		if(col > 72 && !LineTooLong && !in_comment && 
		   parser_warn_for_columns_73_80 && !(c==' ' || c=='\t')) {
		    user_warning("GetChar",
				 "Line %d truncated, col=%d and l_getchar=%d\n",
				 LineNumber, col, l_getchar);
		    LineTooLong = true;
		}
		/* buffer[l_getchar++] = (col > 72) ? ' ' : c; */
		/* buffer[l_getchar++] = (col > 72) ? '\n' : c; */
		if(col <= 72 || in_comment) {
		  /* last columns cannot be copied because we might be 
		   * inside a character string
		   */
		  getchar_buffer[l_getchar++] = c;
		}
		if (c != ' ')
		    EmptyBuffer = false;
	    }
	}
		
	if (c == EOF) {
	    if (!EmptyBuffer) {
 		user_warning("GetChar",
			     "incomplete last line !!!\n");
		c = '\n';
	    }
	}
	else {
	    if (EmptyBuffer) {
		/* i_getchar = l_getchar = UNDEF; */
		debug(8, "GetChar", "An empty line has been detected\n");
		i_getchar = l_getchar = 0;
		getchar_buffer[l_getchar++] = '\n';
		col = 0;
		/* LineNumber += 1; */
	    }
	    else {
		col = 0;
		getchar_buffer[l_getchar++] = '\n';
	    }
	}
	ifdebug(8) {
	    int i;

	    if(l_getchar==UNDEF) {
		debug(8, "GetChar",
		      "Input line after tab expansion is empty:\n");
	    }
	    else {
		debug(8, "GetChar",
		      "Input line after tab expansion l_getchar=%d, col=%d:\n",
		      l_getchar, col);
	    }
	    for (i=0; i < l_getchar; i++) {
		(void) putc((char) getchar_buffer[i], stderr);
	    }
	    if(l_getchar<=0) {
		(void) putc('\n', stderr);
	    }
	}
    }

    if (c != EOF) {
	if ((c = getchar_buffer[i_getchar++]) == '\n') {
	    Column = 1;
	    LineNumber += 1;
	}
	else {
	    Column += 1;
	}
    }

    return(c);
}

/* All physical lines of a statement are put together in a unique buffer 
 * called "line_buffer". Each character in each physical line is retrieved with
 * GetChar().
 */
int 
ReadLine(FILE * fp)
{
    static char QuoteChar = '\000';
    int TypeOfLine;
    int i, c;
    char label[6];
    int ilabel = 0;

    /* on entre dans ReadLine avec Column = 1 */
    pips_assert("ReadLine", Column == 1);

    init_line_buffer();

    /* Read all comment lines you can */
    while (strchr(START_COMMENT_LINE,(c = GetChar(fp))) != NULL) {
	if (tmp_b_C == UNDEF)
	    tmp_b_C = (c=='\n')?LineNumber-1:LineNumber;

	ifdebug(8) {
	    if(c=='\n')
		debug(8, "ReadLine",
		      "Empty comment line detected at line %d "
		      "for comment starting at line %d\n",
		      LineNumber-1, tmp_b_C);
	}

	while(c!=EOF) {
	    if (iCurrComm >= CommSize-2)
		resize_comment_buffers();
	    CurrComm[iCurrComm++] = c;
	    if(c=='\n') break;
	    c = GetChar(fp);
	}
    }

    CurrComm[iCurrComm] = '\0';

    pips_debug(7, "comment CurrComm: (%d) --%s--\n", iCurrComm, CurrComm);

    if (c != EOF) {
	/* Read label */
	for (i = 0; i < 5; i++) {
	    if (c != ' ') {
		if (isdigit(c)) {
		    label[ilabel++] = c;
		}
		else {
		    pips_user_warning("Unexpected character '%c' (0x%x)\n", 
				 c, (int) c);
		    ParserError("ReadLine",
				"non numeric character in label!\n");
		}
	    }
	    c = GetChar(fp);
	}

	if (ilabel > 0) {
	    label[ilabel] = '\0';
	    strcpy(tmp_lab_I, label);
	}
	else
	    strcpy(tmp_lab_I, "");

	/* Check continuation character */
	TypeOfLine = (c != ' ' && c!= '0') ? CONTINUATION_LINE : FIRST_LINE;

	/* Keep track of the first and last comment lines and of the first and
	 * last statement lines. These two intervals may intersect.
	 *
	 * Append current comment CurrComm to Comm if it is a continuation. Save Comm
	 * in PrevComm and CurrComm in Comm if it is a first statement line.
	 */
	if (TypeOfLine == FIRST_LINE) {
	    if(iComm!=0) {
		Comm[iComm] = '\0';
		(void) strcpy(PrevComm, Comm);
		Comm[0] = '\0';
	    }
	    else {
		PrevComm[0] = '\0';
	    }
	    iPrevComm = iComm;

	    (void) strcpy(Comm, CurrComm);
	    iComm = iCurrComm;
	    iCurrComm = 0;
	    CurrComm[0] = '\0';

	    if (tmp_b_C != UNDEF)
		tmp_e_C = LineNumber - 1;
	    tmp_b_I = LineNumber;
	}
	else if (TypeOfLine == CONTINUATION_LINE){
	    if (iCurrComm+iComm >= CommSize-2)
		resize_comment_buffers();
	    (void) strcat(Comm, CurrComm);
	    iComm += iCurrComm;
	    iCurrComm = 0;
	    CurrComm[0] = '\0';

	    /* FI: this is all wrong */
	    /* Why destroy comments because there are continuation lines? */
	    /* tmp_b_C = tmp_e_C = UNDEF; */
	}

	pips_debug(7, "comment Comm: (%d) --%s--\n", iComm, Comm);
	pips_debug(7, "comment PrevComm: (%d) --%s--\n", iPrevComm, PrevComm);

	/* Read the rest of the line, skipping SPACEs but handling string constants */

	while ((c = GetChar(fp)) != '\n') {
	    if (c == '\'' || c == '"') {
		if (EtatQuotes == INQUOTES) {
		    if(c == QuoteChar)
		        EtatQuotes = INQUOTEQUOTE;
		    else {
		        if (EtatQuotes == INQUOTEQUOTE)
			    EtatQuotes = NONINQUOTES;
		    }
		}
		else if(EtatQuotes == INQUOTEBACKSLASH) 
		    EtatQuotes = INQUOTES;
		else {
		    EtatQuotes = INQUOTES;
		    QuoteChar = c;
		}
	    }
	    else {
		if (EtatQuotes == INQUOTEQUOTE)
		    EtatQuotes = NONINQUOTES;
		else if(EtatQuotes == INQUOTES && c == '\\')
		    EtatQuotes = INQUOTEBACKSLASH;
		else if(EtatQuotes == INQUOTEBACKSLASH)
		    EtatQuotes = INQUOTES;
	    }

	    if (lLine>line_buffer_size-5)
		resize_line_buffer();

	    if (EtatQuotes == NONINQUOTES) {
		if (c != ' ') {
		    line_buffer[lLine++] = islower(c)? toupper(c) : c;
		}
	    }
	    else {
		line_buffer[lLine++] = QUOTE(c);
	    }
				
	}

	if (EtatQuotes == INQUOTEQUOTE)
	    EtatQuotes = NONINQUOTES;
    }
    else {
	TypeOfLine = EOF_LINE;
	if (tmp_b_C != UNDEF)
	    tmp_e_C = LineNumber - 1;
	tmp_b_I = LineNumber;

	if(iComm!=0) {
	    Comm[iComm] = '\0';
	    (void) strcpy(PrevComm, Comm);
	    Comm[0] = '\0';
	}
	else {
	    PrevComm[0] = '\0';
	}
	iPrevComm = iComm;
    }

    pips_debug(9, "Aggregation of continuation lines: '%s'\n", (char*)line_buffer);

    return(TypeOfLine);
}

/* regroupement des lignes du statement en une unique ligne sans continuation */
int 
ReadStmt(FILE * fp)
{
    int TypeOfLine;	
    int result;

    init_stmt_buffer();

    if (EofSeen == true) {
	/*
	 * on a rencontre EOF, et on a deja purge le dernier
	 * statement. On arrete.
	 */
	EofSeen = false;
	result = EOF;
    }
    else {
	/*
	 * on a deja lu la 1ere ligne sauf au moment de l'initialisation
	 */
	if (lLine == UNDEF) {
	    lLine = 0;

	    tmp_b_I = tmp_e_I = UNDEF;
	    tmp_b_C = tmp_e_C = UNDEF;

	    if ((TypeOfLine = ReadLine(fp)) == CONTINUATION_LINE) {
		ParserError("ReadStmt",
		  "[scanner] incorrect continuation line as first line\n");
	    }
	    else if (TypeOfLine == FIRST_LINE) {
		/* It would be nice to move the current comments from
		 * Comm to PrevComm, but it is just too late because of
		 * the repeat until control structure down: ReadLine()
		 * has already been called and read the first line of
		 * the next statement. Hence, CurrComm is needed.
		 */
	    }
	    else if (TypeOfLine == EOF_LINE) {
		result = EOF;
	    }
	}

	line_b_I = tmp_b_I;
	line_b_C = tmp_b_C;
	strcpy(lab_I, tmp_lab_I);
		
	lStmt = 0;
	/* Memorize the line number before to find next Statement*/
	StmtLineNumber = LineNumber;
	do {
	    iLine = 0;
	    while (iLine < lLine) {
		if (lStmt>stmt_buffer_size-20)
		    resize_stmt_buffer();
		stmt_buffer[lStmt++] = line_buffer[iLine++];
	    }
	    lLine = 0;

	    /* Update the current final lines for instruction and comments */
	    line_e_I = tmp_e_I;
	    line_e_C = tmp_e_C;

	    /* Initialize temporary beginning and end line numbers */
	     tmp_b_I = tmp_e_I = UNDEF;
	     tmp_b_C = tmp_e_C = UNDEF;

	} while ((TypeOfLine = ReadLine(fp)) == CONTINUATION_LINE) ;

	stmt_buffer[lStmt++] = '\n';
	iStmt = 0;

	line_e_I = (tmp_b_C == UNDEF) ? tmp_b_I-1 : tmp_b_C-1;
		
	if (TypeOfLine == EOF_LINE)
	    EofSeen = true;

	result = 1;

	ifdebug(7) {
	  int i;
	  pips_debug(7, "stmt: (%td)\n", lStmt);
	  for(i=0; i<lStmt; i++)
	    putc((int) stmt_buffer[i], stderr);
	}
    }

    return(result);
}

void 
CheckParenthesis(void)
{
    register int i;
    int parenthese = 0;

    ProfZeroVirg = ProfZeroEgal = false;

    for (i = 0; i < lStmt; i++) {
	if (!IS_QUOTED(stmt_buffer[i])) {
	    if (parenthese == 0) {
		if (stmt_buffer[i] == ',')
			ProfZeroVirg = true;
		else if (stmt_buffer[i] == '=')
			ProfZeroEgal = true;
	    }
	    if(stmt_buffer[i] == '(') parenthese ++;
	    if(stmt_buffer[i] == ')') parenthese --;
	}
    }
    if(parenthese < 0) {
	for (i=0; i < lStmt; i++)
	    (void) putc((char) stmt_buffer[i], stderr);
	/* Warning("CheckParenthesis", */
	ParserError("CheckParenthesis",
		    "unbalanced paranthesis (too many ')')\n"
		    "Due to line truncation at column 72?\n");
    }
    if(parenthese > 0) {
	for (i=0; i < lStmt; i++)
	    (void) putc((char) stmt_buffer[i], stderr);
	ParserError("CheckParenthesis",
		    "unbalanced paranthesis (too many '(')\n"
		    "Due to line truncation at column 72?\n");
    }
}

/* This function is redundant with FindDo() but much easier to
 * understand. I leave it as documentation. FI.
 */

int 
FindDoWhile(void)
{
    int result = false;

    if (!ProfZeroEgal && StmtEqualString("DOWHILE", iStmt)) {
	(void) CapitalizeStmt("DO", iStmt);
	(void) CapitalizeStmt("WHILE", iStmt+2);
	result = true;
    }

    return(result);
}

int 
FindDo(void)
{
    int result = false;

    if(StmtEqualString("DO", iStmt)) {
	if (ProfZeroVirg && ProfZeroEgal) {
	    (void) CapitalizeStmt("DO", iStmt);
	    result = true;
	}
	else if (!ProfZeroVirg && !ProfZeroEgal) {
	    /* Let's skip a loop label to look for a while construct */
	    int i = iStmt+2;
	    while (isdigit(stmt_buffer[i]))
		i++;

	    if (StmtEqualString("WHILE", i)) {
		(void) CapitalizeStmt("DO", iStmt);
		(void) CapitalizeStmt("WHILE", i);
		result = true;
	    }
	}
    }

    return(result);
}

int 
FindImplicit(void)
{
    int result = false;

    if (!ProfZeroEgal && StmtEqualString("IMPLICIT", iStmt)) {
	iStmt = CapitalizeStmt("IMPLICIT", iStmt);
	while (iStmt < lStmt) {
	    iStmt = NeedKeyword();
	    if ((iStmt = FindProfZero((int) ',')) == UNDEF)
		iStmt = lStmt;
	    else
		iStmt += 1;
	}
	result = true;
    }

    return(result);
}

int 
FindIfArith(void)
{
    int result = false;

    if (StmtEqualString("IF(", iStmt)) {
	int i = FindMatchingPar(iStmt+2)+1;
	if ('0' <= stmt_buffer[i] && stmt_buffer[i] <= '9') {
	    (void) CapitalizeStmt("IF", iStmt);
	    result = true;
	}
    }

    return(result);
}

void 
FindIf(void)
{
    if (StmtEqualString("IF(", iStmt)) {
	int i = FindMatchingPar(iStmt+2)+1;
	if (stmt_buffer[i] != '=') {
	    (void) CapitalizeStmt("IF", iStmt);
	    iStmt = i;
	}
    }
    else if (StmtEqualString("ELSEIF(", iStmt)) {
	int i = FindMatchingPar(iStmt+6)+1;
	if (stmt_buffer[i] != '=') {
	    (void) CapitalizeStmt("ELSEIF", iStmt);
	    iStmt = i;
	}
    }
}

void 
FindAutre(void)
{
    if (!ProfZeroEgal) {
	int i = NeedKeyword();

	/*
	 * on detecte le cas tordu: INTEGER FUNCTION(...) ou encore
	 * plus tordu: CHARACTER*89 FUNCTION(...)
	 */
	if (StmtEqualString("Integer", iStmt) ||
	    StmtEqualString("Real", iStmt) ||
	    StmtEqualString("Character", iStmt) ||
	    StmtEqualString("Complex", iStmt) ||
	    StmtEqualString("Doubleprecision", iStmt) ||
	    StmtEqualString("Logical", iStmt)) {
	    if (stmt_buffer[i] == '*' && isdigit(stmt_buffer[i+1])) {
		i += 2;
		while (isdigit(stmt_buffer[i]))
		    i++;
	    }
	    if (StmtEqualString("FUNCTION", i)) {
		(void) CapitalizeStmt("FUNCTION", i);
	    }
	}
    }
}

int 
FindAssign(void)
{
    int result = false;

    if (!ProfZeroEgal && StmtEqualString("ASSIGN", iStmt)) {
	register int i = iStmt+6;

	if (isdigit(stmt_buffer[i])) {
	    while (i < lStmt && isdigit(stmt_buffer[i]))
		i++;

	    if (StmtEqualString("TO", i)) {
		(void) CapitalizeStmt("ASSIGN", iStmt);
		(void) CapitalizeStmt("TO", i);
		result = true;
	    }
	}
    }

    return(result);
}

void 
FindPoints(void)
{
    register int i = iStmt;

    while (i < lStmt) {
	if (stmt_buffer[i] == '.' && isalpha(stmt_buffer[i+1])) {
	    register int j = 0;

	    while (OperateurPoints[j] != NULL) {
		if (StmtEqualString(OperateurPoints[j], i)) {
		    stmt_buffer[i] = '%';
		    i += strlen(OperateurPoints[j]);
		    stmt_buffer[i-1] = '%';
		    break;
		}
		j += 1;
	    }

	    if (OperateurPoints[j] == NULL)
		i += 2;
	}
	else {
	    i += 1;
	}
    }
}

int 
FindProfZero(int c)
{
    register int i;
    int parenthese = 0;

    for (i = iStmt; i < lStmt; i++) {
	if (!IS_QUOTED(stmt_buffer[i])) {
	    if (parenthese == 0 && stmt_buffer[i] == c)
		break;

	    if(stmt_buffer[i] == '(') parenthese ++;
	    if(stmt_buffer[i] == ')') parenthese --;
	}
    }

    return((i == lStmt) ? UNDEF : i);
}

int 
FindMatchingPar(int i)
{
    int parenthese;

    pips_assert("FindMatchingPar", 
		stmt_buffer[i] == '(' && !IS_QUOTED(stmt_buffer[i]));

    i += 1;
    parenthese = 1;

    while (i < lStmt && parenthese > 0) {
	if (!IS_QUOTED(stmt_buffer[i])) {
	    if(stmt_buffer[i] == '(') parenthese ++;
	    if(stmt_buffer[i] == ')') parenthese --;
	}
	i += 1;
    }

    return((i == lStmt) ? UNDEF : i-1);
}

int 
StmtEqualString(char *s, int i)
{
    int result = false;

    if (strlen(s) <= lStmt-i) {
	while (*s)
	    if (*s != stmt_buffer[i++])
		break;
	    else
		s++;

	result = (*s) ? false : i;
    }

    return(result);
}

int 
CapitalizeStmt(char s[], int i)
{
    int l = i+strlen(s);

    if (l <= lStmt) {
	/* la 1ere lettre n'est pas modifiee */
	i += 1;
	while (i < l) {
	    stmt_buffer[i] = tolower(stmt_buffer[i]);
	    i += 1;
	}
    }
    else {
	ParserError("CapitalizeStmt",
		    "[scanner] internal error in CapitalizeStmt\n");
    }

    return(i);
}

int 
NeedKeyword(void)
{
    register int i, j;
    char * kwcour;

    i = keywidx[(int) stmt_buffer[iStmt]-'A'];

    if (i != UNDEF) {
	while ((kwcour = keywtbl[i].keywstr)!=0 && 
	       kwcour[0]==stmt_buffer[iStmt]) {
	    if (StmtEqualString(kwcour, iStmt) != false) {
		j = CapitalizeStmt(kwcour, iStmt);
		return(j);
	    }
	    i += 1;
	}
    }

    ParserError("NeedKeyword", "[scanner] keyword expected\n");

    return(-1); /* just to avoid a gcc warning */
		  
    /*NOTREACHED*/
}

void dump_current_statement()
{
  int i;
  FILE * syn_in = NULL;

  /* Preprocessed statement: Spaces have been eliminated as well as
     continuation lines, keyword have been emphasized and variables
     capitalized. */
  /*
  for(i=0; i<lStmt; i++)
    fprintf(stderr, "%c", (char) stmt_buffer[i]);
  fprintf(stderr,"\n");
  */

  syn_in = safe_fopen(CurrentFN, "r");

  /* Skip the initial lines */

  /* line_b_I, line_e_I */
  i = 1;
  while(i<line_b_I) {
    int c;
    if((c = getc(syn_in))==(int) '\n') i++;
    pips_assert("The end of file cannot be reached", c!=EOF);
  }

  /* Copy the data lines */
  while(i<=line_e_I) {
    int c;
    if((c = getc(syn_in))==(int) '\n') i++;
    pips_assert("The end of file cannot be reached", c!=EOF);
    putc(c, stderr);
  }

  safe_fclose(syn_in, CurrentFN);
}

/*return the line number of the statement being parsed*/
int get_statement_number () {
  return StmtLineNumber - 1;
}

