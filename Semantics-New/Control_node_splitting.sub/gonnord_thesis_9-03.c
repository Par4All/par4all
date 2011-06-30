// N. Halbwachs. Delay analysis in synchronous programs. In Proceedings of
// conference on Computer-Aided Verification, 1993, pages 333-346, 1993.

// $Id$

// parameters

#define DO_CONTROL 0
#define DO_CHECKING 1
#define GOOD (b - s >= -10 && b - s <= 19)

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

#define T 1
#define L 2
#define B 3
#define S 4

#define G2 (t == T && b > s - 9)
#define G2a (G2 && b == s - 8)
#define G2b (G2 && b > s - 8)
#define G2c (G2 && b == s + 2)
#define G2d (G2 && b > s + 2)
#define A2 {s++;}
#define C2 COMMAND(G2, A2)
#define C2a COMMAND(G2a, A2)
#define C2b COMMAND(G2b, A2)
#define C2c COMMAND(G2c, A2)
#define C2d COMMAND(G2d, A2)

#define G3 (t == T && b == s + 9)
#define A3 {t = B; b++; d = 0;}
#define C3 COMMAND(G3, A3)

#define G4 (t == B && b == s + 1)
#define A4 {t = T; s++; d = 0;}
#define C4 COMMAND(G4, A4)

#define G5 (t == L && b == s - 1)
#define A5 {t = T; b++;}
#define C5 COMMAND(G5, A5)

#define G6 (t == T && b == s - 9)
#define A6 {t = L; s++;}
#define C6 COMMAND(G6, A6)

#define G7 (t == L && b < s - 1)
#define G7a (G7 && b < s - 1)
#define G7b (G7 && b == s - 1)
#define G7c (G7 && b < s - 2)
#define G7d (G7 && b == s - 2)
#define A7 {b++;}
#define C7 COMMAND(G7, A7)
#define C7a COMMAND(G7a, A7)
#define C7b COMMAND(G7b, A7)
#define C7c COMMAND(G7c, A7)
#define C7d COMMAND(G7d, A7)

#define G8 (t == B && d == 9)
#define G8a (G8 && b < s + 8)
#define G8b (G8 && b == s + 8)
#define A8 {t = S; b++;}
#define C8 COMMAND(G8, A8)
#define C8a COMMAND(G8a, A8)
#define C8b COMMAND(G8b, A8)
#define C8c COMMAND(G8c, A8)
#define C8d COMMAND(G8d, A8)

#define G9 (t == B && b > s + 1)
#define G9a (G9 && b == s + 1)
#define G9b (G9 && b > s + 1)
#define G9c (G9 && b == s + 2)
#define G9d (G9 && b > s + 2)
#define A9 {s++;}
#define C9 COMMAND(G9, A9)
#define C9a COMMAND(G9a, A9)
#define C9b COMMAND(G9b, A9)
#define C9c COMMAND(G9c, A9)
#define C9d COMMAND(G9d, A9)

#define G10 (t == B && d < 9)
#define G10a (G10 && d < 8)
#define G10b (G10 && d == 8)
#define G10c (G10 && b < s + 8 && d < 8)
#define G10d (G10 && b < s + 8 && d == 8)
#define G10e (G10 && b == s + 8 && d < 8)
#define G10f (G10 && b == s + 8 && d == 8)
#define A10 {d++; b++;}
#define C10 COMMAND(G10, A10)
#define C10a COMMAND(G10a, A10)
#define C10b COMMAND(G10b, A10)
#define C10c COMMAND(G10c, A10)
#define C10d COMMAND(G10d, A10)
#define C10e COMMAND(G10e, A10)
#define C10f COMMAND(G10f, A10)

#define G11 (t == S && b > s + 1)
#define G11a (G11 && b == s + 10)
#define G11b (G11 && b > s + 10)
#define G11c (G11 && b == s + 2)
#define G11d (G11 && b > s + 2)
#define A11 {s++;}
#define C11 COMMAND(G11, A11)
#define C11a COMMAND(G11a, A11)
#define C11b COMMAND(G11b, A11)
#define C11c COMMAND(G11c, A11)
#define C11d COMMAND(G11d, A11)

#define G12 (t == S && b == s + 1)
#define A12 {t = T; s++; d = 0;}
#define C12 COMMAND(G12, A12)

#define G13 (t == T && b < s + 9)
#define G13a (G13 && b < s - 2)
#define G13b (G13 && b == s - 2)
#define G13c (G13 && b < s + 8)
#define G13d (G13 && b == s + 8)
#define A13 {b++;}
#define C13 COMMAND(G13, A13)
#define C13a COMMAND(G13a, A13)
#define C13b COMMAND(G13b, A13)
#define C13c COMMAND(G13c, A13)
#define C13d COMMAND(G13d, A13)

#define INI {\
	b = s = d = 0;\
	t = T;\
}

void ts_singlestate(void) {
	int b, s, d, t;
	INI; CHECK;
	LOOP(
		OR(C2,
		OR(C3,
		OR(C4,
		OR(C5,
		OR(C6,
		OR(C7,
		OR(C8,
		OR(C9,
		OR(C10,
		OR(C11,
		OR(C12,
		C13)))))))))))
	)
}

#define PROG_1_9 {C13; C13; LOOP(OR(OR(C13d; C2, OR(C13c, C2d)), C2c; C13)); C13d; C13;}
#define PROG_9_9 {OR(C9a; LOOP(C9; LOOP(OR(OR(C9d, C10c), C9c; C10a)); C10e); C10a, OR(C10a, C9b));}
#define PROG_9_13 {OR(C10b, C9a; LOOP(C9; LOOP(OR(OR(C9d, C10c), C9c; C10a)); C10e); C10b);}
#define PROG_9_16 {C9a; LOOP(C9; LOOP(OR(OR(C9d, C10c), C9c; C10a)); C10e); C9; LOOP(OR(OR(C9d, C10c), C9c; C10a)); C10f;}
#define PROG_9_19 {C9a; LOOP(C9; LOOP(OR(OR(C9d, C10c), C9c; C10a)); C10e); C9; LOOP(OR(OR(C9d, C10c), C9c; C10a)); OR(C9c; C10b, C10d);}

void ts_restructured(void) {
	int b, s, d, t;
	INI; CHECK;
	LOOP(
		OR(
			OR(OR(C13; C2, OR(C2; C13, C2; C2; LOOP(OR(C13b; C2, OR(OR(C13a, C2b), C2a; C13))); C13b; C13)), C13; C13; LOOP(OR(OR(C13d; C2, OR(C13c, C2d)), C2c; C13)); C2c; C2),
		OR(
			C2; C2; LOOP(OR(C13b; C2, OR(OR(C13a, C2b), C2a; C13))); C2a; C6;
			LOOP(C7a); C7b; C7; LOOP(C7c); C7d; C5,
		OR(
			PROG_1_9; LOOP(PROG_9_9);
			C9a; LOOP(C9; LOOP(OR(OR(C9d, C10c), C9c; C10a)); C10e); C9; LOOP(OR(OR(C9d, C10c), C9c; C10a)); C9c; C4,
		OR(
			PROG_1_9; LOOP(PROG_9_9);
			OR(
				PROG_9_13; LOOP(C9b); C9a; C9,
			OR(
				PROG_9_16; C9,

				PROG_9_19
			));
			LOOP(C9d); C9c; C4,

			PROG_1_9; LOOP(PROG_9_9);
			OR(
				OR(
					PROG_9_13; LOOP(C9b); OR(C9a; C8, C8),

					PROG_9_16; C8
				);
				LOOP(C11b); C11a; C11,

				OR(
					PROG_9_13; LOOP(C9b); C9a; C9,
				OR(
					PROG_9_16; C9,

					PROG_9_19
				));
				LOOP(C9d); OR(OR(C8a, C9c; C8), C8b; C11)
			);
			LOOP(C11d); C11c; C12;
		))))
	)
}

int main(void) {
	ts_singlestate();
	ts_restructured();
	return 0;
}

