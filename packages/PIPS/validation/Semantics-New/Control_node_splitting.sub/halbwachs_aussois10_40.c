// Nicolas Halbwachs: Linear Relation Analysis, Principles and Recent Progress
// talk at Aussois, 9/12/2010
// http://compilation.gforge.inria.fr/2010_12_Aussois/programpage/pdfs/HALBWACHS.Nicolas.aussois2010.pdf
// slide 40

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (2 * v <= t + 10 && 5 * v >= t - 10)

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

#define S1 CONTROL(x <= 1 && d <= 9)
#define S2 CONTROL(2 <= x && x <= 4 && d <= 9)
#define S3 CONTROL(5 <= x && d <= 9)
#define S4 CONTROL(x <= 1 && d == 10)
#define S5 CONTROL(2 <= x && x <= 4 && d == 10)
#define S6 CONTROL(5 <= x && d == 10)

#define G1 (x <= 4)
#define G1a (x == 0)
#define G1b (x == 1)
#define G1c (x <= 3)
#define G1d (x == 4)
#define A1 {x++; v++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)
#define C1c COMMAND(G1c, A1)
#define C1d COMMAND(G1d, A1)

#define G2 (d <= 9)
#define G2a (d <= 8)
#define G2b (d == 9)
#define A2 {d++; t++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (d == 10 && x >= 2)
#define A3 {x = 0; d = 0;}
#define C3 COMMAND(G3, A3)

#define INI {v = t = x = d = 0;}

// transition system

void ts_singlestate(void) {
	int v, t, x, d;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int v, t, x, d;
	INI; CHECK;
	S1;
	LOOP(
		OR(
			C1a; S1,
		OR(
			C2a; S1,
		OR(
			OR(
				C2b; S4; LOOP(C1a; S4); C1b; S5,
				
				C1b; S2; LOOP(OR(C1c; S2, C2a; S2)); C2b; S5
			);
			LOOP(C1c; S5); C3; S1,
			
			OR(
				C2b; S4; LOOP(C1a; S4); C1b; S5; LOOP(C1c; S5); C1d; S6,

				C1b; S2; LOOP(OR(C1c; S2, C2a; S2));
				OR(C1d; S3; LOOP(C2a; S3); C2b; S6, C2b; S5; LOOP(C1c; S5); C1d; S6);
			);
			C3; S1
		)))
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

