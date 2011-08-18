// Laure Gonnord: Accélération abstraite pour l'amélioration de la précision en
// Analyse des Relations Linéaires
// figure 7.3

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (x >= 0 && z >= 0 && z <= 10 && z <= x)

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

#define S1 CONTROL(z <= 9)
#define S2 CONTROL(z == 10)

#define G1 (z <= 9)
#define G1a (z <= 8)
#define G1b (z == 9)
#define A1 {x++; z++;}
#define C1 COMMAND(G1, A1)
#define C1a COMMAND(G1a, A1)
#define C1b COMMAND(G1b, A1)

#define G2 (z == 10)
#define A2 {z = 0;}
#define C2 COMMAND(G2, A2)

#define INI {z = x = 0;}

// transition system

void ts_singlestate(void) {
	int x, z;
	INI; CHECK;
	LOOP(OR(C1, C2))
}

void ts_restructured(void) {
	int x, z;
	INI; CHECK;
	S1;
	LOOP(OR(C1a; S1, C1b; S2; C2; S1))
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

