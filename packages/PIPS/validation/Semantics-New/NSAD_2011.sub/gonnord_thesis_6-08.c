// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 6.8

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (0 <= y && y <= x && y <= -x + 202 && 0 <= x && x <= 102)

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

#define S1 CONTROL(s == 1 && x <= 100)
#define S2 CONTROL(s == 1 && x > 100)
#define S3 CONTROL(s == 2 && x > 100)

#define G1 (s == 1 && x <= 100)
#define G1a (s == 1 && x <= 99)
#define G1b (s ==  1 && x == 100)
#define A1 {x++; y++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (s == 1 && x <= 100)
#define G2a (s == 1 && x <= 98)
#define G2b (G2 && x > 98)
#define A2 {x += 2;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define G3 (s == 1)
#define A3 {s = 2;}
#define C3 COMMAND(G3, A3)

#define INI {s = 1; x = y = 0;}

// transition system

void ts_singlestate(void) {
	int s, x, y;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int s, x, y;
	INI; CHECK;
	S1;
	LOOP(OR(C1a; S1, C2a; S1));
	OR(C1b; S2, C2b; S2);
	C3; S3
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

