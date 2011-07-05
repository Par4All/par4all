
model jeannet_thesis_4_02 {

	var b0, b1, x, y, ok;
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
		guard := b0 = 1 && b1 = 1 && ok = 1 && x >= y;
		action := b0' = 0, b1' = 1, x' = x + 1, ok' = 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := b0 = 1 && b1 = 0 && ok = 1 && x >= y;
		action := b0' = 1, b1' = 1, y' = y + 1, ok' = 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := b0 = 0 && b1 = 1 && ok = 1 && x >= y;
		action := b0' = 0, b1' = 0, y' = y + 1, ok' = 1;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := b0 = 0 && b1 = 0 && ok = 1 && x >= y;
		action := b0' = 1, b1' = 0, x' = x + 1, ok' = 1;
	};

	transition t5 := {
		from := k2;
		to := k2;
		guard := ok = 1 && x < y;
		action := ok' = 0;
	};

}

strategy s {

	Region init := {state = k1};

}

