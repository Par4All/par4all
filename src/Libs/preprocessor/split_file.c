/*
 * $Id$
 *
 * adapted from what can be seen by FC 31/12/96
 * 
 * - static declarations; 
 * - main -> function; 
 * - stdout -> FILE* out;
 * - include unistd added
 * - exit -> return
 * - close ifp
 * - bug labeled end (skipped) in lend()
 * - tab in first columns...
 * - bang comments added
 * - bug name[20] overflow not checked in lname (20 -> 80)
 * - hollerith constants conversion;-)
 * - LINESIZE 80 -> 200...
 * - "PROGRAM MAIN..." added if implicit program name.
 * - extr* stuff dropped.
 * - dir_name for localizing files...
 */

static void hollerith(char *);
#define LINESIZE 200

/*
 * Copyright (c) 1983 The Regents of the University of California.
 * All rights reserved.
 *
 * This code is derived from software contributed to Berkeley by
 * Asa Romberger and Jerry Berkman.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the University of
 *	California, Berkeley and its contributors.
 * 4. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef lint
char fsplit_copyright[] =
"@(#) Copyright (c) 1983 The Regents of the University of California.\n\
 All rights reserved.\n";
#endif /* not lint */

#ifndef lint
char fsplit_sccsid[] = "@(#)fsplit.c	5.5 (Berkeley) 3/12/91";
#endif /* not lint */

#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>

/*
 *	usage:		fsplit [-e efile] ... [file]
 *
 *	split single file containing source for several fortran programs
 *		and/or subprograms into files each containing one
 *		subprogram unit.
 *	each separate file will be named using the corresponding subroutine,
 *		function, block data or program name if one is found; otherwise
 *		the name will be of the form mainNNN.f or blkdtaNNN.f .
 *		If a file of that name exists, it is saved in a name of the
 *		form zzz000.f .
 *	If -e option is used, then only those subprograms named in the -e
 *		option are split off; e.g.:
 *			fsplit -esub1 -e sub2 prog.f
 *		isolates sub1 and sub2 in sub1.f and sub2.f.  The space 
 *		after -e is optional.
 *
 *    Modified Feb., 1983 by Jerry Berkman, Computing Services, U.C. Berkeley.
 *		- added comments
 *		- more function types: double complex, character*(*), etc.
 *		- fixed minor bugs
 *		- instead of all unnamed going into zNNN.f, put mains in
 *		  mainNNN.f, block datas in blkdtaNNN.f, dups in zzzNNN.f .
 */

#define BSZ 512
static char buf[BSZ];
static FILE *ifp;
static char *x, *mainp, *blkp;

#define TRUE 1
#define FALSE 0
static struct stat sbuf;

static char *look(), *skiplab(), *functs();
static int scan_name();
static void get_name();

#define trim(p)	while (*p == ' ' || *p == '\t') p++

static int saveit(name)
char *name;
{
    return(1);
}

static char * 
full_name(char * dir, char * name)
{
    char * full = (char*) malloc(sizeof(char)*(strlen(dir)+strlen(name)+2));
    sprintf(full, "%s/%s", dir, name);
    return full;
}

static void get_name(name, letters)
char *name;
int letters;
{
	register char *ptr;

	while (stat(name, &sbuf) >= 0) {
		for (ptr = name + letters + 2; ptr >= name + letters; ptr--) {
			(*ptr)++;
			if (*ptr <= '9')
				break;
			*ptr = '0';
		}
		if(ptr < name + letters) {
			fprintf( stderr, "fsplit: ran out of file names\n");
			exit(1);
		}
	}
}

static int getline()
{
	register char *ptr;

	for (ptr = buf; ptr < &buf[BSZ]; ) {
		*ptr = getc(ifp);
		if (feof(ifp))
			return (-1);
		if (*ptr++ == '\n') {
			*ptr = 0;
			return (1);
		}
	}
	while (getc(ifp) != '\n' && feof(ifp) == 0) ;
	fprintf(stderr, "line truncated to %d characters\n", BSZ);
	return (1);
}

/* return 1 for 'end' alone on card (up to col. 72),  0 otherwise */
static int lend()
{
	register char *p;
	int tab = FALSE;

	/* if ((p = skiplab(buf)) == 0)
	    return (0) ; */ 

	if (buf[0]!=' ' && buf[0]!='\t') 
	    return 0; /* a comment */
	for (p=buf; p<&buf[6] && !tab; p++)
	{
	    if (*p=='\0') return 0;
	    if (*p=='\t') tab=TRUE;
	}
	
	if (!tab && (buf[5]!=' ' && buf[5]!='\t')) 
	    return 0; /* a continuation */
	    
	trim(p);
	if (*p != 'e' && *p != 'E') return(0);
	p++;
	trim(p);
	if (*p != 'n' && *p != 'N') return(0);
	p++;
	trim(p);
	if (*p != 'd' && *p != 'D') return(0);
	p++;
	trim(p);
	if (p - buf >= 72 || *p == '\n')
		return (1);
	return (0);
}

static int implicit_program; /* FC */
static int implicit_blockdata_name; /* FC */
static int implicit_program_name; /* FC */
static int it_is_a_main; /* FC */

/*		check for keywords for subprograms	
		return 0 if comment card, 1 if found
		name and put in arg string. invent name for unnamed
		block datas and main programs.		*/
static int lname(s)
char *s;
{
	register char *ptr, *p;
	char	line[LINESIZE], *iptr = line;

	implicit_program = 0;
	implicit_blockdata_name = 0;
	implicit_program_name = 0;
	it_is_a_main = 0;

	/* first check for comment cards */
	if(buf[0]=='c' || buf[0]=='C' || buf[0]=='*' || buf[0]=='!') return 0;
	ptr = buf;
	while (*ptr == ' ' || *ptr == '\t') ptr++;
	if(*ptr == '\n') return(0);


	ptr = skiplab(buf);
	if (ptr == 0) return (0);

	/*  copy to buffer and converting to lower case */
	p = ptr;
	while (*p && p <= &buf[71] ) {
	   *iptr = isupper(*p) ? tolower(*p) : *p;
	   iptr++;
	   p++;
	}
	*iptr = '\n';

	if ((ptr = look(line, "subroutine")) != 0 ||
	    (ptr = look(line, "function")) != 0 ||
	    (ptr = functs(line)) != 0) {
		if(scan_name(s, ptr)) return(1);
		strcpy( s, x);
	} else if((ptr = look(line, "program")) != 0) {
	    it_is_a_main = 1;
	    if(scan_name(s, ptr)) return(1);
	    implicit_program_name = 1;
	    get_name( mainp, 4);
	    strcpy( s, mainp);
	} else if((ptr = look(line, "blockdata")) != 0) {
		if(scan_name(s, ptr)) return(1);
		implicit_blockdata_name = 1;
		get_name( blkp, 4);
		strcpy( s, blkp);
	} else if((ptr = functs(line)) != 0) {
		if(scan_name(s, ptr)) return(1);
		strcpy( s, x);
	} else {
	    implicit_program = 1;
	    it_is_a_main = 1;
		get_name( mainp, 4);
		strcpy( s, mainp);
	}
	return(1);
}

static int scan_name(s, ptr)
char *s, *ptr;
{
	char *sptr;

	/* scan off the name */
	trim(ptr);
	sptr = s;
	while (*ptr != '(' && *ptr != '\n') {
		if (*ptr != ' ' && *ptr != '\t')
			*sptr++ = *ptr;
		ptr++;
	}

	if (sptr == s) return(0);

	*sptr++ = '.';
	*sptr++ = 'f';
	*sptr++ = 0;
	return(1);
}

static char *functs(p)
char *p;
{
        register char *ptr;

/*      look for typed functions such as: real*8 function,
                character*16 function, character*(*) function  */

        if((ptr = look(p,"character")) != 0 ||
           (ptr = look(p,"logical")) != 0 ||
           (ptr = look(p,"real")) != 0 ||
           (ptr = look(p,"integer")) != 0 ||
           (ptr = look(p,"doubleprecision")) != 0 ||
           (ptr = look(p,"complex")) != 0 ||
           (ptr = look(p,"doublecomplex")) != 0 ) {
                while ( *ptr == ' ' || *ptr == '\t' || *ptr == '*'
			|| (*ptr >= '0' && *ptr <= '9')
			|| *ptr == '(' || *ptr == ')') ptr++;
		ptr = look(ptr,"function");
		return(ptr);
	}
        else
                return(0);
}

/* 	if first 6 col. blank, return ptr to col. 7,
	if blanks and then tab, return ptr after tab,
	else return 0 (labelled statement, comment or continuation */
static char *skiplab(p)
char *p;
{
	register char *ptr;

	for (ptr = p; ptr < &p[6]; ptr++) {
		if (*ptr == ' ')
			continue;
		if (*ptr == '\t') {
			ptr++;
			break;
		}
		return (0);
	}
	return (ptr);
}

/* 	return 0 if m doesn't match initial part of s;
	otherwise return ptr to next char after m in s */
static char *look(s, m)
char *s, *m;
{
	register char *sp, *mp;

	sp = s; mp = m;
	while (*mp) {
		trim(sp);
		if (*sp++ != *mp++)
			return (0);
	}
	return (sp);
}

static void print_name(FILE * o, char * name, int n) /* FC */
{
    while (n-->0) putc(*name++, o);
}

int 
fsplit(char * dir_name, char * file_name, FILE * out)
{
    register FILE *ofp;	/* output file */
    register rv;	/* 1 if got card in output file, 0 otherwise */
    int nflag,		/* 1 if got name of subprog., 0 otherwise */
	retval;
   /* ??? 20 -> 80 because not checked... smaller than a line is ok ? FC */
    char name[80]; 
    char * main_list = full_name(dir_name, ".fsplit_main_list");
    x = full_name(dir_name, "zzz000.f");
    mainp = full_name(dir_name, "main000.f");
    blkp = full_name(dir_name, "data000.f");
	
    if ((ifp = fopen(file_name, "r")) == NULL) {
	fprintf(stderr, "fsplit: cannot open %s\n", file_name);
	return 0;
    }

    for(;;) {
	/* look for a temp file that doesn't correspond to an existing file */
	get_name(x, 3);
	ofp = fopen(x, "w");
	if (ofp==NULL) {
	    fprintf(stderr, "fopen(\"%s\", ...) failed\n", x);
	    exit(2);
	}
	nflag = 0;
	rv = 0;
	while (getline() > 0) {
	    hollerith(buf); /* FC */
	    if (nflag == 0) /* if no name yet, try and find one */
		nflag = lname(name);
	    if (it_is_a_main) {
		char * c = name;
		FILE * fm = fopen(main_list, "a");
		if (fm==NULL) {
		    fprintf(stderr, "fopen(\"%s\", ...) failed\n", main_list);
		    exit(2);
		}
		while (*c && *c!='.') putc(toupper(*c++), fm);
		putc('\n', fm);
		fclose(fm);
		it_is_a_main = 0;
	    }
	    if (implicit_program==1) /* FC again */ 
	    {
		fprintf(ofp, 
			"! next line added by fsplit() in pips\n"
			"      PROGRAM ");
		print_name(ofp, name, 7);
		putc('\n', ofp);
		implicit_program = 0; /* now we gave it a name! */
	    }
	    
	    if (implicit_blockdata_name==1 || implicit_program_name==1) 
	    {
		fprintf(ofp, 
			"! next line modified by fsplit() in pips\n"
			"      %s ", 
			implicit_program_name==1? "PROGRAM": "BLOCK DATA");
		print_name(ofp, name, 7);
		putc('\n', ofp);
		implicit_blockdata_name = 0;
		implicit_program_name = 0;
	    }
	    else
		fprintf(ofp, "%s", buf);

	    rv = 1;

	    if (lend())		/* look for an 'end' statement */
		break;
	}
	if (fclose(ofp)) {
	    fprintf(stderr, "fclose(ofp) failed\n");
	    exit(2);
	}
	if (rv == 0) {			/* no lines in file, forget the file */
		unlink(x);
		retval = 0;
		if (fclose(ifp)) {
		    fprintf(stderr, "fclose(ifp) failed\n");
		    exit(2);
		}
		return ( retval );
	}
	if (nflag) {			/* rename the file */
		if(saveit(name)) {
		    char * nname = full_name(dir_name, name);
		    if (stat(nname, &sbuf) < 0 ) {
			link(x, nname);
			unlink(x);
			fprintf(out, "%s\n", name);
		    } else if (strcmp(nname, x) == 0) {
				printf("%s\n", x);
		    }
		    else 
			printf("%s already exists, put in %s\n", nname, x);
		    free(nname);
		    continue;
		} else
		    unlink(x);
		continue;
	}
	fprintf(out, "%s\n", x);
    }

    if (fclose(ifp)) {
	fprintf(stderr, "fclose(ifp) failed\n");
	exit(2);
    }
    free(main_list), free(x), free(mainp), free(blkp);
    return 1;
}


/* ADDITION: basic Hollerith constants handling
 * FC 11 Apr 1997
 *
 * bugs:
 *  - under special circonstances, the dilatation of the transformation
 *    may lead continuations to exceed the 19 lines limit. 
 *
 * to improve:
 *  - hack for "real*8 hollerith", but should just forbids start after *?
 *    maybe some other characters?
 */

#define isbegincomment(c) ((c)=='!' || (c)=='*' || (c)=='c' || (c)=='C')
#define issquote(c) ((c)=='\'')
#define isdquote(c) ((c)=='\"')
#define ishH(c) ((c)=='h' || (c)=='H')
#define char2int(c) ((int)((c)-'0'))

/* global state
 */
static int in_squotes=0, in_dquotes=0, in_id=0;

static int blank_line_p(char * line)
{
    if (!line) return 1;
    while (*line)
	if (!isspace(*line++))
	    return 0;
    return 1;
}

static void hollerith(char * line)
{
    int i,j,initial, touched=0;
    
    if (!line) {
	in_squotes=0, in_dquotes=0, in_id=0; /* RESET */
	return;
    }

    if (blank_line_p(line))
	return;

    if (isbegincomment(line[0]))
	return;

    i = (line[0]=='\t')? 1: 6; /* first column to analyze */
    
    for (j=0; j<i; j++)
	if (!line[j]) return;

    if (isspace(line[i-1]))
	in_squotes=0, in_dquotes=0, in_id=0; /* RESET */

    initial=i;

    while (line[i] && initial<72) /* 73.. ignored */
    {
	if (!in_dquotes && issquote(line[i])) 
	    in_squotes = !in_squotes, in_id=0;
	if (!in_squotes && isdquote(line[i])) 
	    in_dquotes = !in_dquotes, in_id=0;
	if (!in_squotes && !in_dquotes)
	{
	    if (isalpha(line[i]))
		in_id=1;
	    else if (!isalnum(line[i]) && !isspace(line[i]) 
		&& line[i]!='*') /* hack for real*8 hollerith */
		in_id=0;
	}

	if (!in_squotes && !in_dquotes && !in_id && isdigit(line[i]))
	{
	    /* looks for [0-9 ]+[hH] 
	     */
	    int len=char2int(line[i]), ni=i;
	    i++, initial++;
	    
	    while (line[i] && initial<72
		   && (isdigit(line[i]) || isspace(line[i])))
	    {
		if (isdigit(line[i]))
		    len=10*len+char2int(line[i]);
		i++, initial++;
	    }

	    if (!line[i] || initial>=72) return;
	    
	    if (ishH(line[i])) /* YEAH, here it is! */
	    {
		char tmp[200];
		int k;

		if (!touched) { /* rm potential 73-80 text */
		    touched=1;
		    line[72]='\n'; 
		    line[73]='\0';			
		}

		j=1;

		tmp[0] = '\''; i++, initial++;
		while (j<200 && line[i] && initial<72 && 
		       line[i]!='\n' && len>0)
		{
		    len--;
		    if (line[i]=='\'')
			tmp[j++]='\'';
		    tmp[j++] = line[i++];
		    initial++;
		}
                
                while (j<199 && len>0) /* padding */
                    tmp[j++]=' ', len--;
		
		tmp[j]='\'';

		/* must insert tmp[<j] in line[ni..]
		 * first, shift the line...
		 */
		
		{
		    int ll = strlen(line), shift = i-(ni+j+1);

		    if (shift>0) /* to the left */
			for (k=0; i+k<=ll; k++) 
			    line[ni+j+1+k] = line[i+k];
		    else /* to the right */
			for (k=ll-i; k>=0; k--)
			    line[ni+j+1+k] = line[i+k];
		}

		i=ni+j+1;

		while(j>=0)
		    line[ni+j]=tmp[j], j--;

	    }
	}
	
	i++, initial++;
    }

    if (touched) {
	int len = strlen(line); /* the new line may exceed the 72 column */
	/* caution, len includes cr... */
	/* the dilatation cannot exceed one line (?) */
	if (len-1>72) /* then shift and continuation... */
	{
	    for (i=len; i>=72; i--) line[i+7] = line[i];
	    line[72]='\n'; line[73]=' '; line[74]=' ';
	    line[75]=' '; line[76]=' '; line[77]=' '; line[78]='x';
	}
    }
}
