/* C encoding of mutual exclusion algorithm called "bakery"
 *
 * Derived from Pugh 1995
 *
 * Bug submitted by Vivien Maisonneuve: The value of B should be known
 * exactly when reaching label CW (b==2). Instead, you get 0<=b<=2.
 *
 * The control flow graph is a DAG. There should be no problem to
 * forward preconditions.
 *
 * This has been fixed by replacing return statements by goto
 * statements and return values when necessary.
 */

#include <stdlib.h>

void error() {
	exit(1);
}

int alea(void) {
	return rand() % 2;
}

void run() {
	int a = 0, b = 0;
	float z;

TT:
	a=b+1;
	goto WT;

WT:
	if (alea()) {
		goto CT;
	}
	else {
		b=a+1;
		z = 0.;
		goto WW1;
	}

CT:
	if (alea()) {
	  //goto end;
	  return;
	}
	else {
		b=a+1;
		goto CW;
	}

WW1:
	z = 0.;
	goto CW;

CW:
	z=0.;
end:
	return;
}

int main(void) {
  run();
  return 0;
}

