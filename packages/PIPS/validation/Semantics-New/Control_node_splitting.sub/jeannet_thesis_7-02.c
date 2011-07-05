// Bertrand Jeannet: Partitionnement Dynamique dans l'Analyse de Relations Linéaires
// et Application à la Vérification de Programmes Synchrones
// chapter 7, example 2

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define BAD (b == 0 && s == 2 && x > 4)

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

#define S1 CONTROL(s == 1 && x <= 2 && b == 1)
#define S2 CONTROL(s == 1 && x <= 2 && b == 0)
#define S3 CONTROL(s == 1 && x >= 3)
#define S4 CONTROL(s == 2)

#define G1 (s == 1 && x <= 2 && b == 1)
#define A1 {b = 1; x+= 10; s = 2;}
#define C1 COMMAND(G1, A1)

#define G2 (s == 1 && x <= 2 && b == 0)
#define A2 {x--; s = 2;}
#define C2 COMMAND(G2, A2)

#define G3 (s == 1 && x >= 3)
#define A3 {b = 0; x--; s = 2;}
#define C3 COMMAND(G3, A3)

#define INI {s = 1; b = rand_b(); x = rand(); ASSUME(0 <= x && x <= 5);}

// transition system

void ts_singlestate(void) {
	int s, b, x;
	INI; CHECK;
	LOOP(OR(C1, OR(C2, C3)))
}

void ts_restructured(void) {
	int s, b, x;
	INI; CHECK;
	if (x <= 2) {
		if (b == 1) {
			S1; C1; S4;
		}
		else {
			S2; C2; S4;
		}
	}
	else {
		S3; C3; S4;
	}
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

