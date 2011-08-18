
model jeannet_thesis_4_01 {

	var x, y, ok;
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
		guard := x >= 14 && ok = 1;
		action := x' = x + 1, y' = y + 1, ok' = 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := x >= 10 && y <= 3 && ok = 1;
		action := x' = x + 1, y' = y + 1, ok' = 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := x >= 10 && x <= 13 && y >= 4 && ok = 1;
		action := x' = x + 1, y' = y + 1, ok' = 0;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := x <= 9 && y <= 3 && ok = 1;
		action := x' = x + 1, ok' = 1;
	};

	transition t5 := {
		from := k2;
		to := k2;
		guard := x <= 9 && y >= 4 && ok = 1;
		action := x' = x + 1, ok' = 0;
	};

}

strategy s {

	Region init := {state = k1};

}

