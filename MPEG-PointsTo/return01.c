/* C necessary.
 */

#include <stdlib.h>

void error() {
	exit(1);
}

int alea(void) {
	return rand() % 2;
}

int run() {
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
	  return a;
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
	return b;
}

int main(void) {
  run();
  return 0;
}

