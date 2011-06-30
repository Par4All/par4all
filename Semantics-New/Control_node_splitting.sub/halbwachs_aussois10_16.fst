
model halbwachs_aussois10_16 {

	var b, ok, x, y;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x = 0 && y = 0 && ok = 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := b = 0 && ok = 1 && x >= y;
		action := b' = 1, x' = x + 1, ok' = 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := b = 0 && ok = 0;
		action := b' = 1, x' = x + 1, ok' = 0;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := b = 0 && x < y;
		action := b' = 1, x' = x + 1, ok' = 0;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := b = 1 && ok = 1 && x >= y;
		action := b' = 0, y' = y + 1, ok' = 1;
	};

	transition t5 := {
		from := k2;
		to := k2;
		guard := b = 1 && ok = 0;
		action := b' = 0, y' = y + 1, ok' = 0;
	};

	transition t6 := {
		from := k2;
		to := k2;
		guard := b = 1 && x < y;
		action := b' = 0, y' = y + 1, ok' = 0;
	};

}

strategy s {

	Region init := {state = k1};

}

