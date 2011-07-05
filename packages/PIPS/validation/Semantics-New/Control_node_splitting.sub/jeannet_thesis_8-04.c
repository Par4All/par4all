// Bertrand Jeannet: Partitionnement Dynamique dans l'Analyse de Relations Linéaires
// et Application à la Vérification de Programmes Synchrones
// figure 8.4

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (y <= 2 * x)

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

#define I 1
#define A 2
#define B 3

#define S1 CONTROL(s == I && e == 0)
#define S2 CONTROL(s == A && e == 0)
#define S3 CONTROL(s == B && e == 0)
#define S4 CONTROL(s == I && e == 1)
#define S5 CONTROL(s == A && e == 1)

#define G1 (s == I && e == 1)
#define A1 {s = A;}
#define C1 COMMAND(G1, A1)

#define G2 (s == I && e == 0)
#define A2 {s = B;}
#define C2 COMMAND(G2, A2)

#define G3 (s == A && e == 1)
#define A3 {x++; y += f;}
#define C3 COMMAND(G3, A3)

#define G4 (s == A && e == 0)
#define A4 {s = B;}
#define C4 COMMAND(G4, A4)

#define G5 (s == B)
#define A5 {s = A; x++; y += 2;}
#define C5 COMMAND(G5, A5)

#define INI {\
	s = I;\
	x = y = 0;\
	e = rand_b();\
	f = rand_b();\
}

// transition system

void ts_singlestate(void) {
	int s, x, y, e, f;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, OR(C3, OR(C4, C5)))))
}

void ts_restructured(void) {
	int s, x, y, e, f;
	INI; CHECK;
	if (e == 0) {
		S1;
		C2; S3;
		LOOP(C5; S2; C4; S3);
	}
	else {
		S4;
		C1; S5;
		LOOP(C3; S5);
	}
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

