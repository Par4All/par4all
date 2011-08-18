
model halbwachs_aussois10_23 {

	var b0, b1, ok, x, y;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := b0 = 0 && b1 = 0 && x = 0 && y = 0 && ok = 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := b0 = b1 && b0 = 0 && b1 = 0 && ok = 1 && x >= y;
		action := b0' = 1 - b1, b1' = b0, x' = x + 1, ok' = 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := b0 = b1 && b0 = 0 && b1 = 0 && ok = 0;
		action := b0' = 1 - b1, b1' = b0, x' = x + 1, ok' = 0;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := b0 = b1 && b0 = 0 && b1 = 0 && x < y;
		action := b0' = 1 - b1, b1' = b0, x' = x + 1, ok' = 0;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := b0 = b1 && b0 = 1 && b1 = 1 && ok = 1 && x >= y;
		action := b0' = 1 - b1, b1' = b0, x' = x + 1, ok' = 1;
	};

	transition t5 := {
		from := k2;
		to := k2;
		guard := b0 = b1 && b0 = 1 && b1 = 1 && ok = 0;
		action := b0' = 1 - b1, b1' = b0, x' = x + 1, ok' = 0;
	};

	transition t6 := {
		from := k2;
		to := k2;
		guard := b0 = b1 && b0 = 1 && b1 = 1 && x < y;
		action := b0' = 1 - b1, b1' = b0, x' = x + 1, ok' = 0;
	};

	transition t7 := {
		from := k2;
		to := k2;
		guard := b0 != b1 && b0 = 0 && b1 = 1 && ok = 1 && x >= y;
		action := b0' = 1 - b1, b1' = b0, y' = y + 1, ok' = 1;
	};

	transition t8 := {
		from := k2;
		to := k2;
		guard := b0 != b1 && b0 = 0 && b1 = 1 && ok = 0;
		action := b0' = 1 - b1, b1' = b0, y' = y + 1, ok' = 0;
	};

	transition t9 := {
		from := k2;
		to := k2;
		guard := b0 != b1 && b0 = 0 && b1 = 1 && x < y;
		action := b0' = 1 - b1, b1' = b0, y' = y + 1, ok' = 0;
	};
	
	transition t10 := {
		from := k2;
		to := k2;
		guard := b0 != b1 && b0 = 1 && b1 = 0 && ok = 1 && x >= y;
		action := b0' = 1 - b1, b1' = b0, y' = y + 1, ok' = 1;
	};

	transition t11 := {
		from := k2;
		to := k2;
		guard := b0 != b1 && b0 = 1 && b1 = 0 && ok = 0;
		action := b0' = 1 - b1, b1' = b0, y' = y + 1, ok' = 0;
	};

	transition t12 := {
		from := k2;
		to := k2;
		guard := b0 != b1 && b0 = 1 && b1 = 0 && x < y;
		action := b0' = 1 - b1, b1' = b0, y' = y + 1, ok' = 0;
	};
	
}

strategy s {

	Region init := {state = k1};

}

