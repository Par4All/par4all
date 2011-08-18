
model gulwani_cav09_2 {

	var x, y, m, n, i1, i2;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x = 0 && y = 0 && i1 = 0 && i2 = 0 && m >= 0 && n >= 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := x < n && y < m;
		action := y' = y + 1, i1' = i1 + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := x < n && y >= m;
		action := y' = 0, x' = x + 1, i1' = 0, i2' = i2 + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

