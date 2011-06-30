
model gulwani_cav09_1 {

	var x, y, i, m;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x = 0 && y = 0 && i = 0 && m >= 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := x < 100 && y < m;
		action := y' = y + 1, i' = i + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := x < 100 && y >= m;
		action := x' = x + 1, i' = i + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

