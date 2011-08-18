
model gonnord_thesis_6_02 {

	var x, y;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x = 0 && y = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := y <= 1;
		action := x' = x + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := y >= x - 1;
		action := x' = x + 1, y' = y + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

