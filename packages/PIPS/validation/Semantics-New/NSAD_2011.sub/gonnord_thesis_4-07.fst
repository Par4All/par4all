
model gonnord_thesis_4_07 {

	var i, j, k;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := i = 0 && j = 0 && k = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := i <= 100 && j < 9;
		action := i' = i + 2, k' = k + 1, j' = j + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := i <= 100 && j < 9;
		action := i' = i + 2, j' = j + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := i <= 100 && j = 9;
		action := i' = i + 2, k' = k + 2, j' = 0;
	};

}

strategy s {

	Region init := {state = k1};

}

