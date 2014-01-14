
int __lv0;
int __lv1;
int __lv2;

void userfunction(int unused_arg1, int arg2,
		int unused_arg3, int arg4, int arg5, int arg6, int unused_arg7) {

	for (__lv2 = 0; __lv2 < 1; __lv2++) {
		0;
	}

	int unused_1;
	int unused_2;

	if ((arg5 == 1)) {
		0;
	}

	int useless_switch_param = (0);

	switch (arg6) {
	case 1:
		useless_switch_param = (0);
		break;
	case 2:
		useless_switch_param = (1);
		break;
	case 3:
		useless_switch_param = (0);
		break;
	default:
		return;
		break;
	}

	int unused_4;

	for (__lv2 = 0; __lv2 < arg2; __lv2++) {
		for (__lv1 = 0; __lv1 < 1; __lv1++) {
			0;
		}
	}

	int useless_array2[arg2][1 * arg4];

}

int main(int argc, char* argv[]) {


	int useless_in_main;

	userfunction(0, 1,  1, 1, 1, 2, 1);

// Replacing the first call with the second prevents the
// overflow and allow to see what the precondition looks
// like
	userfunction(41874, (int) (1.), 2, 1, 1, 3, 0);
//	userfunction(41873, (int) (1.), 2, 1, 1, 3, 0);

}
