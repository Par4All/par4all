// Nicolas Halbwachs: Linear Relation Analysis, Principles and Recent Progress
// talk at Aussois, 9/12/2010
// http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/pdfs/HALBWACHS.Nicolas.aussois2010.pdf
// slide 8

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (0 <= y && y <= x && x <= 102 && x + y <= 202)

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
#define CHECK
#else
void checking_error(void) {
	fprintf(stderr, "checking error");
	exit(2);
}
#ifdef GOOD
#define CHECK {if (!(GOOD)) checking_error();}
#else
#ifdef BAD
#define CHECK {if (BAD) checking_error();}
#endif
#endif
#endif

#define COMMAND_NOCHECK(g, a) {ASSUME(g); a;}
#define COMMAND(g, a) {COMMAND_NOCHECK(g, a); CHECK;}

// control and commands

#define S1 CONTROL(x <= 100)
#define S2 CONTROL(x > 100)

#define G1 (x <= 100)
#define G1a (x <= 98)
#define G1b (G1 && x >= 99)
#define U1 {x += 2;}
#define C1 COMMAND(G1, U1)
#define C1a COMMAND(G1a, U1)
#define C1b COMMAND(G1b, U1)

#define G2 (x <= 100)
#define G2a (x <= 99)
#define G2b (x == 100)
#define U2 {x++; y++;}
#define C2 COMMAND(G2, U2)
#define C2a COMMAND(G2a, U2)
#define C2b COMMAND(G2b, U2)

#define INI {x = y = 0;}

// transition system

void ts_singlestate(void) {
	int x, y;
	INI; CHECK;
	LOOP(OR(C1, C2));
}

void ts_restructured(void) {
	int x, y;
	INI; CHECK;
	S1;
	LOOP(OR(C1a; S1, C2a; S1));
	OR(C1b; S2, C2b; S2);
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

