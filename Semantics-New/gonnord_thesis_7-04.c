// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 7.4

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (x >= 2 * z - 1 && x >= 0 && z >= 0 && z <= x)

// tools

#include <stdlib.h>
#include <stdio.h>

int flip(void) {
	return rand() % 2;
}
#define OR(t1, t2) {if (flip()) {t1} else {t2}}
#define LOOP(t) {while (flip()) {t}}

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

#define S1 CONTROL(x >= 2 * z)
#define S2 CONTROL(x < 2 * z)

#define G1 (x >= 2 * z)
#define G1a (G1 && x + 1 < 2 * z + 2)
#define G1b (G1 && x + 1 >= 2 * z + 2)
#define A1 {x++; z++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (1)
#define A2 {z = 0;}
#define C2 COMMAND(G2, A2)

#define INI {z = x = 0;}

// COMMANDition system

void ts_singlestate(void) {
	int x, z;
	INI;
	LOOP(OR(C1, C2));
}

void ts_restructured(void) {
	int x, z;
	INI;
	S1;
	LOOP(OR(C1b; S1, OR(C2; S1, C1a; S2; C2; S1)));
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

