// run_enum1 and run_enum2 fails with PIPS r18945

enum mybool {
	mybool_true = 1,
	mybool_false = 0
};

void run_int(void) {
	int b;
	b = 0;
	if (b == 1) "unreachable";
}

void run_enum1(void) {
	enum mybool b;
	b = 0;
	if (b == 1) "unreachable";
}

void run_enum2(void) {
	enum mybool b;
	b = mybool_false;
	if (b == mybool_true) "unreachable";
}

int main(void) {
	run_int();
	run_enum1();
	run_enum2();
	return 0;
}

