// TRAIN CONTROLER SYSTEM
// N. Halbwachs. Delay analysis in synchronous programs. In Proceedings of
// conference on Computer-Aided Verification, 1993, pages 333-346, 1993.

// $Id$

#define USE_ASSERT 0
#define USE_CHECK 1
#define RESTRUCTURE 1
#define GOOD (b - s >= -10 && b - s <= 19)

// usefull stuff

#include <stdlib.h>

int flip(void) {
	return rand() % 2;
}

void assert_error(void) {
	exit(1);
}

void checking_error(void) {
	exit(2);
}

#if USE_ASSERT == 0
#define assert(e)
#else
#define assert(e) {if (!(e)) assert_error();}
#endif

#if USE_CHECK == 0
#define check(e)
#define check_not(e)
#else
#define check(e) {if (!(e)) checking_error();}
#define check_not(e) {if (e) checking_error();}
#endif

#define freeze {while(1);}

#ifdef GOOD
#define trans(g, c) {if (g) {c; check(GOOD);} else freeze;}
#else
#ifdef BAD
#define trans(g, c) {if (g) {c; check_not(BAD);} else freeze;}
#else
#define trans(g, c) {if (g) {c;} else freeze;}
#endif
#endif

// transition system

#define T 1
#define L 2
#define B 3
#define S 4

#define G2 (t == T && b > s - 9)
#define G2a (G2 && b == s - 8)
#define G2b (G2 && b > s - 8)
#define G2c (G2 && b == s + 2)
#define G2d (G2 && b > s + 2)
#define U2 {s++;}
#define T2 {trans(G2, U2);}
#define T2a {trans(G2a, U2);}
#define T2b {trans(G2b, U2);}
#define T2c {trans(G2c, U2);}
#define T2d {trans(G2d, U2);}

#define G3 (t == T && b == s + 9)
#define U3 {t = B; b++; d = 0;}
#define T3 {trans(G3, U3);}

#define G4 (t == B && b == s + 1)
#define U4 {t = T; s++; d = 0;}
#define T4 {trans(G4, U4);}

#define G5 (t == L && b == s - 1)
#define U5 {t = T; b++;}
#define T5 {trans(G5, U5);}

#define G6 (t == T && b == s - 9)
#define U6 {t = L; s++;}
#define T6 {trans(G6, U6);}

#define G7 (t == L && b < s - 1)
#define G7a (G7 && b < s - 1)
#define G7b (G7 && b == s - 1)
#define G7c (G7 && b < s - 2)
#define G7d (G7 && b == s - 2)
#define U7 {b++;}
#define T7 {trans(G7, U7);}
#define T7a {trans(G7a, U7);}
#define T7b {trans(G7b, U7);}
#define T7c {trans(G7c, U7);}
#define T7d {trans(G7d, U7);}

#define G8 (t == B && d == 9)
#define G8a (G8 && b < s + 8)
#define G8b (G8 && b == s + 8)
#define U8 {t = S; b++;}
#define T8 {trans(G8, U8);}
#define T8a {trans(G8a, U8);}
#define T8b {trans(G8b, U8);}
#define T8c {trans(G8c, U8);}
#define T8d {trans(G8d, U8);}

#define G9 (t == B && b > s + 1)
#define G9a (G9 && b == s + 1)
#define G9b (G9 && b > s + 1)
#define G9c (G9 && b == s + 2)
#define G9d (G9 && b > s + 2)
#define U9 {s++;}
#define T9 {trans(G9, U9);}
#define T9a {trans(G9a, U9);}
#define T9b {trans(G9b, U9);}
#define T9c {trans(G9c, U9);}
#define T9d {trans(G9d, U9);}

#define G10 (t == B && d < 9)
#define G10a (G10 && d < 8)
#define G10b (G10 && d == 8)
#define G10c (G10 && b < s + 8 && d < 8)
#define G10d (G10 && b < s + 8 && d == 8)
#define G10e (G10 && b == s + 8 && d < 8)
#define G10f (G10 && b == s + 8 && d == 8)
#define U10 {d++; b++;}
#define T10 {trans(G10, U10);}
#define T10a {trans(G10a, U10);}
#define T10b {trans(G10b, U10);}
#define T10c {trans(G10c, U10);}
#define T10d {trans(G10d, U10);}
#define T10e {trans(G10e, U10);}
#define T10f {trans(G10f, U10);}

#define G11 (t == S && b > s + 1)
#define G11a (G11 && b == s + 10)
#define G11b (G11 && b > s + 10)
#define G11c (G11 && b == s + 2)
#define G11d (G11 && b > s + 2)
#define U11 {s++;}
#define T11 {trans(G11, U11);}
#define T11a {trans(G11a, U11);}
#define T11b {trans(G11b, U11);}
#define T11c {trans(G11c, U11);}
#define T11d {trans(G11d, U11);}

#define G12 (t == S && b == s + 1)
#define U12 {t = T; s++; d = 0;}
#define T12 {trans(G12, U12);}

#define G13 (t == T && b < s + 9)
#define G13a (G13 && b < s - 2)
#define G13b (G13 && b == s - 2)
#define G13c (G13 && b < s + 8)
#define G13d (G13 && b == s + 8)
#define U13 {b++;}
#define T13 {trans(G13, U13);}
#define T13a {trans(G13a, U13);}
#define T13b {trans(G13b, U13);}
#define T13c {trans(G13c, U13);}
#define T13d {trans(G13d, U13);}

void run(void) {
	
	int b, s, d, t;
	b = s = d = 0;
	t = T;

	while (1) {

#if RESTRUCTURE == 0

		if (flip()) {
			T2;
		}
		else if (flip()) {
			T3;
		}
		else if (flip()) {
			T4;
		}
		else if (flip()) {
			T5;
		}
		else if (flip()) {
			T6;
		}
		else if (flip()) {
			T7;
		}
		else if (flip()) {
			T8;
		}
		else if (flip()) {
			T9;
		}
		else if (flip()) {
			T10;
		}
		else if (flip()) {
			T11;
		}
		else if (flip()) {
			T12;
		}
		else {
			T13;
		}
		
#else

		// can run without transformer lists, with the assertion:
		assert(t == T && b == s && d < 9);
		
		if (flip()) {
			if (rand() % 2) {T13; T13; while (rand() % 2) {if (rand() % 2) {T2c; T13;} else {if (rand() % 2) {T13d; T2;} else {if (rand() % 2) {T2d;} else {T13c;}}}} T2c; T2;} else {if (rand() % 2) {T13; T2;} else {if (rand() % 2) {T2; T13;} else {T2; T2; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T13a;} else {T2b;}} else {T2a; T13;}} else {T13b; T2;}} T13b; T13;}}}
		}

		else if (flip()) {
			T2; T2; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T13a;} else {T2b;}} else {T2a; T13;}} else {T13b; T2;}} T2a; T6;
			while (flip()) {T7a;}
			T7b;
			T7;
			while (flip()) {T7c;}
			T7d;
			T5;
		}
		
		else if (flip()) {
			T13; T13; while (rand() % 2) {if (rand() % 2) {T2c; T13;} else {if (rand() % 2) {T13d; T2;} else {if (rand() % 2) {T2d;} else {T13c;}}}} T13d; T13;
			while (flip()) {if (rand() % 2) {T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T10a;} else {if (rand() % 2) {T9b;} else {T10a;}}}
			T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T9c; T4;
		}

		else if (flip()) {
			T13; T13; while (rand() % 2) {if (rand() % 2) {T2c; T13;} else {if (rand() % 2) {T13d; T2;} else {if (rand() % 2) {T2d;} else {T13c;}}}} T13d; T13;
			while (flip()) {if (rand() % 2) {T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T10a;} else {if (rand() % 2) {T9b;} else {T10a;}}}
			if (flip()) {
				if (rand() % 2) {T10b;} else {T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T10b;}
				while (flip()) {T9b;}
				T9a;
				T9;
			}
			else if (flip()) {
				T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10f;
				T9;
			}
			else {
				T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} if (rand() % 2) {T10d;} else {T9c; T10b;}
			}
			while (flip()) {T9d;}
			T9c; T4;
		}
		
		else {
			T13; T13; while (rand() % 2) {if (rand() % 2) {T2c; T13;} else {if (rand() % 2) {T13d; T2;} else {if (rand() % 2) {T2d;} else {T13c;}}}} T13d; T13;
			while (flip()) {if (rand() % 2) {T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T10a;} else {if (rand() % 2) {T9b;} else {T10a;}}}
			if (flip()) {
				if (rand() % 2) {T10b;} else {T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T10b;}
				while (flip()) {T9b;}
				if (flip()) {
					T9a;
					T9;
					while (flip()) {T9d;}
					if (rand() % 2) {if (rand() % 2) {T9c; T8;} else {T8a;}} else {T8b; T11;}
				}
				else if (flip()) {
					T9a;
					T8;
					while (flip()) {T11b;}
					T11a; T11;
				}
				else {
					T8;
					while (flip()) {T11b;}
					T11a; T11;
				}
			}
			else if (flip()) {
				T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10f;
				if (flip()) {
					T9;
					while (flip()) {T9d;}
					if (rand() % 2) {if (rand() % 2) {T9c; T8;} else {T8a;}} else {T8b; T11;}
				}
				else {
					T8;
					while (flip()) {T11b;}
					T11a; T11;
				}
			}
			else {
				T9a; while (rand() % 2) {T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} T10e;} T9; while (rand() % 2) {if (rand() % 2) {if (rand() % 2) {T10c;} else {T9d;}} else {T9c; T10a;}} if (rand() % 2) {T10d;} else {T9c; T10b;}
				while (flip()) {T9d;}
				if (rand() % 2) {if (rand() % 2) {T9c; T8;} else {T8a;}} else {T8b; T11;}
			}
			while (flip()) {T11d;}
			T11c;
			T12;
		}
		
#endif
		
	}
	
}

int main(void) {
	run();
	return 0;
}

