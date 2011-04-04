// Nicolas Halbwachs: Linear Relation Analysis, Principles and Recent Progress
// talk at Aussois, 9/12/2010
// http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/pdfs/HALBWACHS.Nicolas.aussois2010.pdf
// slide 16

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (ok == 1)

// tools

#include <stdlib.h>
#include <stdio.h>

int rand_b(void) {
	return rand() % 2;
}
int rand_z(void) {
	return rand() - rand();
}

#define OR(t1, t2) {if (rand_b()) {t1} else {t2}}
#define LOOP(t) {while (rand_b()) {t}}

void deadlock() {
	printf("deadlock\n");
	while (1);
}
#define ASSUME(c) {if (!(c)) deadlock();}

#if DO_CONTROL == 0
#define CONTROL(c)
#else
void control_error(void) {
	fprintf(stderr, "control error");
	exit(1);
}
#define CONTROL(c) {if (!(c)) control_error();}
#endif

#if DO_CHECKING == 0
#define CHECK(c)
#define CHECK_NOT(c)
#else
void checking_error(void) {
	fprintf(stderr, "checking error");
	exit(2);
}
#define CHECK(c) {if (!(c)) checking_error();}
#define CHECK_NOT(c) {if (c) checking_error();}
#endif

#define COMMAND_NOCHECK(g, a) {ASSUME(g); a;}
#ifdef GOOD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK(GOOD);}
#else
#ifdef BAD
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK_NOT(BAD);}
#endif
#endif

// control and commands

#define S1 CONTROL(b == 0)
#define S2 CONTROL(b == 1)

#define G1 (b == 0)
#define A1 {b = 1; x++; if (ok ==  1 && x >= y) {ok = 1;} else {ok = 0;}}
#define C1 COMMAND(G1, A1)

#define G2 (b == 1)
#define A2 {b = 0; y++; if (ok == 1 && x >= y) {ok = 1;} else {ok = 0;}}
#define C2 COMMAND(G2, A2)

#define INI {b = x = y = 0; ok = 1;}

// transition system

void ts_singlestate(void) {
	int b, ok, x, y;
	INI;
	LOOP(OR(C1, C2));
}

void ts_restructured(void) {
	int b, ok, x, y;
	INI;
	S1;
	LOOP(C1; S2; C2; S1);
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

