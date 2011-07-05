// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 6.2
// The system has 2 variables x, y (initially x = y = 0).
// There are 2 possible transitions:
// - y <= 1 ? x++;
// - y >= x - 1 ? x++; y++;
// Difficulty: the set of reachable states (x, y) is not convex.

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (x == 4 && y == 2)

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

#define S1 CONTROL(G1 && G2)
#define S2 CONTROL(G1 && !G2)
#define S3 CONTROL(!G1 && G2)

#define G1 (y <= 1)
#define G1a (G1 && y > x - 1)
#define G1b (G1 && y == x - 1)
#define A1 {x++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (y >= x - 1)
#define G2a (G2 && y < 1)
#define G2b (G2 && y == 1)
#define A2 {x++; y++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)

#define INI {x = y = 0;}

// transition system

void ts_singlestate(void) {
	int x, y;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int x, y;
	INI; CHECK;
	S1;
	LOOP(OR(C1a; S1, C2a; S1));
	OR(C1b; S2; LOOP(C1; S2), C2b; S3; LOOP(C2; S3))
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

