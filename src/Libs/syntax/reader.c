#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "genC.h"
#include "parser_private.h"
#include "ri.h"

#include "misc.h"

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
 * Lorsqu'un operateur .XX. est detecte, il est remplace dans  le  source  par
 * '_XX_'.  Ainsi,  lex peut faire la difference entre une constante reelle et
 * un operateur, comme dans '(X+1._EQ_5)'. Modification: underscore is replaced
 * by percent to allow safely underscore in identifiers.
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

/*
 * le buffer contenant le statement courant, l'indice courant et la longueur.
 */
#define STMTLENGTH (4096)
LOCAL int Stmt[STMTLENGTH];
LOCAL int iStmt, lStmt; 

/*
 * le buffer contenant le commentaire courant, l'indice courant.
 */
#define COMMLENGTH (8192)
char Comm[COMMLENGTH];
char PrevComm[COMMLENGTH];
int iComm = 0;
int iPrevComm = 0;

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
 *  - la quote peut-etre ' ou " pour faire plaisir a Fabien Coelho.
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
LOCAL int EtatQuotes;
#define NONINQUOTES 1
#define INQUOTES 2
#define INQUOTEQUOTE 3
#define INQUOTEBACKSLASH 4

/*
 * le buffer contenant la ligne que l'on doit lire en avance pour se rendre
 * compte qu'on a finit de lire un statement, l'indice courant et la longueur.
 */
#define LINELENGTH (128)
LOCAL int Line[LINELENGTH];
LOCAL int iLine, lLine; 

/*
 * Numero de ligne et de colonne du fichier d'entree courant.
 */
LOCAL int LineNumber, Column;

/*
 * Y a t il un '=' ou un ',' non parenthese ?
 */
LOCAL int ProfZeroVirg, ProfZeroEgal;

/*
 * La table des operateurs du type '.XX.'.
 */
char * OperateurPoints[] = {
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
extern struct Skeyword keywtbl[];
#include "keywtbl.h"

/*
 * Une table pour accelerer les recherche des keywords. keywidx[X] indique le
 * rang dans keywtbl du premier mot clef commencant par X.
 */
int keywidx[26];

/*
 * Variables qui serviront a mettre a jour les numeros de la premiere et de la
 * derniere ligne de commentaire, et les numeros de la premiere et de la
 * derniere ligne du statement.
 */
LOCAL int tmp_b_I, tmp_e_I, tmp_b_C, tmp_e_C;
LOCAL char tmp_lab_I[6];

/* memoization des properties */

#include "properties.h"

static bool parser_warn_for_columns_73_80 = TRUE;

void init_parser_reader_properties()
{
  parser_warn_for_columns_73_80 = get_bool_property("PARSER_WARN_FOR_COLUMNS_73_80");
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

int syn_wrap()
{
	return(1);
}

/*
 * La fonction a appeler pour l'analyse d'un nouveau fichier.
 */
void ScanNewFile()
{
    register int i;
    static int FirstCall = TRUE;
    char letcour, *keywcour;


    if (FirstCall) {
	FirstCall = FALSE;

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
    EtatQuotes = NONINQUOTES;
    iStmt = lStmt = UNDEF;
    iLine = lLine = UNDEF;
}

/*
 * Fonction appelee par sslex sur la reduction de la regle de reconnaissance
 * des mot clefs. Elle recherche si le mot 's' est un mot clef, retourne sa
 * valeur si oui, et indique une erreur si non.
 */
int IsCapKeyword(s)
char * s;
{
    register int i, c;
    char *kwcour, *t;
    static char buffer[32];

    debug(9, "", "[IsCapKeyword] %s\n", s);

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
		debug(9, "", "[IsCapKeyword] %s %d\n", kwcour, i);
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
int PipsGetc(fp)
FILE * fp;
{
    int eof = FALSE;
    int c;

    if (iStmt == UNDEF || iStmt >= lStmt) {
	/*
	 * le statement est vide. On lit et traite le suivant.
	 */
	if (ReadStmt(fp) == EOF) {
	    eof = TRUE;
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

    c = Stmt[iStmt++];
    return((eof) ? EOF : UNQUOTE(c));
}

/* Routine de lecture physique */
int GetChar(fp)
FILE * fp;
{
    int c = UNDEF;
    static int buffer[LINELENGTH], ibuffer = UNDEF, lbuffer = UNDEF;
    static col = 0;

    /* on lit toute une ligne d'un coup pour traiter le cas des
     * tabulations et des lignes vides.
     */
    while (ibuffer >= lbuffer && c != EOF) {
	int EmptyBuffer = TRUE;
	int LineTooLong = FALSE;
	bool first_column = TRUE;
	bool in_comment = FALSE;

	ibuffer = lbuffer = 0;

	while ((c = getc(fp)) != '\n' && c != EOF) {

	  if(first_column) {
	    in_comment = (strchr(START_COMMENT_LINE, (char) c)!= NULL);
	    first_column = FALSE;
	  }

	    /* Fortran has a limited character set. See standard section 3.1.
	       This cannot be handled here as you do not know if you are
	       in a string constant or not. You cannot convert the double
	       quote into a simple quote because you may generate an illegal
	       string constant. Maybe the best would be to uncomment the
	       next test. FI, 21 February 1992
	    if( c == '\"')
		FatalError("GetChar","Illegal double quote character");
		*/
	    /* FI: let's delay and do it in ReadLine:
	     * if (islower(c)) c = toupper(c);
	     */

	    if (c == '\t') {
		int i;
		int nspace = 8-col%8;
		/* for (i = 0; i < (8-Column%8); i++) { */
		for (i = 0; i < nspace; i++) {
		    col += 1;
		    buffer[lbuffer++] = ' ';
		}
	    }
	    else {
		col += 1;
		if(col > 72 && !LineTooLong && !in_comment && parser_warn_for_columns_73_80) {
		    user_warning("GetChar",
				 "Line %d truncated, col=%d and lbuffer=%d\n",
				 LineNumber, col, lbuffer);
		    LineTooLong = TRUE;
		}
		/* buffer[lbuffer++] = (col > 72) ? ' ' : c; */
		/* buffer[lbuffer++] = (col > 72) ? '\n' : c; */
		if(col <= 72 || in_comment) {
		  /* last columns cannot be copied because we might be inside a character string */
		  buffer[lbuffer++] = c;
		}
		if (c != ' ')
		    EmptyBuffer = FALSE;
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
		ibuffer = lbuffer = UNDEF;
		col = 0;
		LineNumber += 1;
	    }
	    else {
		col = 0;
		buffer[lbuffer++] = '\n';
	    }
	}
	ifdebug(8) {
	    int i;

	    debug(8, "GetChar",
		  "Input line after tab expansion lbuffer=%d, col=%d:\n",
		  lbuffer, col);
	    for (i=0; i < lbuffer; i++)
		(void) putc((char) buffer[i], stderr);
	}
    }

    if (c != EOF) {
	if ((c = buffer[ibuffer++]) == '\n') {
	    Column = 1;
	    LineNumber += 1;
	}
	else {
	    Column += 1;
	}
    }

    return(c);
}

/* regroupementde la ligne initiale et des lignes suite en un unique buffer, Line */
int ReadLine(fp)
FILE * fp;
{
    static char QuoteChar = '\000';
    int TypeOfLine;
    int i, c;
    char label[6];
    int ilabel = 0;

    /* on entre dans ReadLine avec Column = 1 */
    pips_assert("ReadLine", Column == 1);

    /*
     * on lit le label et le caractere de continuation de la premiere
     * ligne non vide et non ligne de commentaire.
     */
    if(iComm!=0) {
	Comm[iComm] = '\0';
	(void) strcpy(PrevComm, Comm);
	iPrevComm = iComm;
	iComm = 0;
	Comm[0] = '\0';
    }
    else {
	iPrevComm = iComm;
	PrevComm[0] = '\0';
    }
    while (strchr(START_COMMENT_LINE,(c = GetChar(fp))) != NULL) {
      if (tmp_b_C == UNDEF)
	tmp_b_C = LineNumber;

      /* Modif by AP: oct 18th 1995

       	 Deals with comment buffer overflow. If the buffer is full, we
       	 just skip everything else. */
      if(iComm >= COMMLENGTH-2)
	while((c = GetChar(fp)) != '\n') {continue;}
      else {
	do {
	  Comm[iComm++] = c;
	} while((c = GetChar(fp)) != '\n' && iComm < COMMLENGTH-2);
	if(iComm >= COMMLENGTH-2)
	  while((c = GetChar(fp)) != '\n') {continue;}
	else
	  Comm[iComm++] = c;
      }
    }
    if(iComm >= COMMLENGTH-2)
      user_warning("ReadLine", 
		   "Too many comments. Comment buffer overflow\n");

    if (c != EOF) {
	/*
	 * on lit les 5 caracteres du label, et le caractere de
	 * continuation.
	 */
	for (i = 0; i < 5; i++) {
	    if (c != ' ') {
		if (isdigit(c)) {
		    label[ilabel++] = c;
		}
		else {
		    ParserError("ReadLine",
			       "[scanner] non numeric character in label !!!\n");
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

	TypeOfLine = (c != ' ' && c!= '0') ? CONTINUATION_LINE : FIRST_LINE;

	if (TypeOfLine == FIRST_LINE) {
	    if (tmp_b_C != UNDEF)
		tmp_e_C = LineNumber - 1;
	    tmp_b_I = LineNumber;
	}
	else {
	    tmp_b_C = tmp_e_C = UNDEF;
	}

	/*
	 * dans tous les cas on lit jusqu'au newline en sautant les
	 * blancs.
	 */

	while ((c = GetChar(fp)) != '\n') {
	    if (c == '\'' || c == '"') {
		if (EtatQuotes == INQUOTES)
		    if(c == QuoteChar)
		        EtatQuotes = INQUOTEQUOTE;
		    else {
		        if (EtatQuotes == INQUOTEQUOTE)
			    EtatQuotes = NONINQUOTES;
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

	    if (EtatQuotes == NONINQUOTES) {
		if (c != ' ') {
		    Line[lLine++] = islower(c)? toupper(c) : c;
		}
	    }
	    else {
		Line[lLine++] = QUOTE(c);
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
    }

    /* debug(9, "ReadLine", "Aggregation of continuation lines: '%s'\n", Line); */

    return(TypeOfLine);
}

/* regroupement des lignes du statement en une unique ligne sans continuation */
int ReadStmt(fp)
FILE * fp;
{
    static int EofSeen = FALSE;
    int TypeOfLine;	
    int result;

    if (EofSeen == TRUE) {
	/*
	 * on a rencontre EOF, et on a deja purge le dernier
	 * statement. On arrete.
	 */
	EofSeen = FALSE;
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
	    else if (TypeOfLine == EOF_LINE) {
		result = EOF;
	    }
	}

	line_b_I = tmp_b_I; line_e_I = tmp_e_I;
	line_b_C = tmp_b_C; line_e_C = tmp_e_C;
	strcpy(lab_I, tmp_lab_I);

	tmp_b_I = tmp_e_I = UNDEF;
	tmp_b_C = tmp_e_C = UNDEF;
		
	lStmt = 0;
	do {
	    iLine = 0;
	    while (iLine < lLine)
		Stmt[lStmt++] = Line[iLine++];
	    lLine = 0;
	} while ((TypeOfLine = ReadLine(fp)) == CONTINUATION_LINE) ;

	Stmt[lStmt++] = '\n';
	iStmt = 0;

	line_e_I = (tmp_b_C == UNDEF) ? tmp_b_I-1 : tmp_b_C-1;
		
	if (TypeOfLine == EOF_LINE)
	    EofSeen = TRUE;
			
	result = 1;
    }

    return(result);
}

void CheckParenthesis()
{
    register int i;
    int parenthese = 0;

    ProfZeroVirg = ProfZeroEgal = FALSE;

    for (i = 0; i < lStmt; i++) {
	if (!IS_QUOTED(Stmt[i])) {
	    if (parenthese == 0) {
		if (Stmt[i] == ',')
			ProfZeroVirg = TRUE;
		else if (Stmt[i] == '=')
			ProfZeroEgal = TRUE;
	    }
	    if(Stmt[i] == '(') parenthese ++;
	    if(Stmt[i] == ')') parenthese --;
	}
    }
    if(parenthese < 0) {
	for (i=0; i < lStmt; i++)
	    (void) putc((char) Stmt[i], stderr);
	/* Warning("CheckParenthesis", */
	ParserError("CheckParenthesis",
		    "unbalanced paranthesis (too many ')')\n"
		    "Due to line truncation at column 72?\n");
    }
    if(parenthese > 0) {
	for (i=0; i < lStmt; i++)
	    (void) putc((char) Stmt[i], stderr);
	ParserError("CheckParenthesis",
		    "unbalanced paranthesis (too many '(')\n"
		    "Due to line truncation at column 72?\n");
    }
}

int FindDo()
{
    int result = FALSE;

    if (ProfZeroVirg && ProfZeroEgal && StmtEqualString("DO", iStmt)) {
	(void) CapitalizeStmt("DO", iStmt);
	result = TRUE;
    }

    return(result);
}

int FindImplicit()
{
    int result = FALSE;

    if (!ProfZeroEgal && StmtEqualString("IMPLICIT", iStmt)) {
	iStmt = CapitalizeStmt("IMPLICIT", iStmt);
	while (iStmt < lStmt) {
	    iStmt = NeedKeyword();
	    if ((iStmt = FindProfZero((int) ',')) == UNDEF)
		iStmt = lStmt;
	    else
		iStmt += 1;
	}
	result = TRUE;
    }

    return(result);
}

int FindIfArith()
{
    int result = FALSE;

    if (StmtEqualString("IF(", iStmt)) {
	int i = FindMatchingPar(iStmt+2)+1;
	if ('0' <= Stmt[i] && Stmt[i] <= '9') {
	    (void) CapitalizeStmt("IF", iStmt);
	    result = TRUE;
	}
    }

    return(result);
}

void FindIf()
{
    if (StmtEqualString("IF(", iStmt)) {
	int i = FindMatchingPar(iStmt+2)+1;
	if (Stmt[i] != '=') {
	    (void) CapitalizeStmt("IF", iStmt);
	    iStmt = i;
	}
    }
    else if (StmtEqualString("ELSEIF(", iStmt)) {
	int i = FindMatchingPar(iStmt+6)+1;
	if (Stmt[i] != '=') {
	    (void) CapitalizeStmt("ELSEIF", iStmt);
	    iStmt = i;
	}
    }
}

void FindAutre()
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
	    if (Stmt[i] == '*' && isdigit(Stmt[i+1])) {
		i += 2;
		while (isdigit(Stmt[i]))
		    i++;
	    }
	    if (StmtEqualString("FUNCTION", i)) {
		(void) CapitalizeStmt("FUNCTION", i);
	    }
	}
    }
}

int FindAssign()
{
    int result = FALSE;

    if (!ProfZeroEgal && StmtEqualString("ASSIGN", iStmt)) {
	register int i = iStmt+6;

	if (isdigit(Stmt[i])) {
	    while (i < lStmt && isdigit(Stmt[i]))
		i++;

	    if (StmtEqualString("TO", i)) {
		(void) CapitalizeStmt("ASSIGN", iStmt);
		(void) CapitalizeStmt("TO", i);
		result = TRUE;
	    }
	}
    }

    return(result);
}

void FindPoints()
{
    register int i = iStmt;

    while (i < lStmt) {
	if (Stmt[i] == '.' && isalpha(Stmt[i+1])) {
	    register int j = 0;

	    while (OperateurPoints[j] != NULL) {
		if (StmtEqualString(OperateurPoints[j], i)) {
		    Stmt[i] = '%';
		    i += strlen(OperateurPoints[j]);
		    Stmt[i-1] = '%';
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

int FindProfZero(c)
int c;
{
    register int i;
    int parenthese = 0;

    for (i = iStmt; i < lStmt; i++) {
	if (!IS_QUOTED(Stmt[i])) {
	    if (parenthese == 0 && Stmt[i] == c)
		break;

	    if(Stmt[i] == '(') parenthese ++;
	    if(Stmt[i] == ')') parenthese --;
	}
    }

    return((i == lStmt) ? UNDEF : i);
}

int FindMatchingPar(i)
int i;
{
    int parenthese;

    pips_assert("FindMatchingPar", Stmt[i] == '(' && !IS_QUOTED(Stmt[i]));

    i += 1;
    parenthese = 1;

    while (i < lStmt && parenthese > 0) {
	if (!IS_QUOTED(Stmt[i])) {
	    if(Stmt[i] == '(') parenthese ++;
	    if(Stmt[i] == ')') parenthese --;
	}
	i += 1;
    }

    return((i == lStmt) ? UNDEF : i-1);
}

int StmtEqualString(s, i)
char *s;
int i;
{
    int result = FALSE;

    if (strlen(s) <= lStmt-i) {
	while (*s)
	    if (*s != Stmt[i++])
		break;
	    else
		s++;

	result = (*s) ? FALSE : i;
    }

    return(result);
}

int CapitalizeStmt(s, i)
char s[];
int i;
{
    int l = i+strlen(s);

    if (l <= lStmt) {
	/* la 1ere lettre n'est pas modifiee */
	i += 1;
	while (i < l) {
	    Stmt[i] = tolower(Stmt[i]);
	    i += 1;
	}
    }
    else {
	ParserError("CapitalizeStmt",
		    "[scanner] internal error in CapitalizeStmt\n");
    }

    return(i);
}

int NeedKeyword()
{
    register int i, j;
    char * kwcour;

    i = keywidx[(int) Stmt[iStmt]-'A'];

    if (i != UNDEF) {
	while ((kwcour = keywtbl[i].keywstr)!=0 && kwcour[0]==Stmt[iStmt]) {
	    if (StmtEqualString(kwcour, iStmt) != FALSE) {
		j = CapitalizeStmt(kwcour, iStmt);
		return(j);
	    }
	    i += 1;
	}
    }

    /* FatalError("NeedKeyword", "[scanner] keyword expected\n"); */
    ParserError("NeedKeyword", "[scanner] keyword expected\n");

    return(-1); /* just to avoid a gcc warning */
		  
    /*NOTREACHED*/
}
