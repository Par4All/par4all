
model gulwani_pldi09_4 {

	var i, m, n, j1, j2;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := m > 0 && n >= 0 && m < n && i = n && j1 = 0 && j2 = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := i > 0 && i < m;
		action := i' = i - 1, j1' = j1 + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := i > 0 && i >= m;
		action := i' = i - m, j2' = j2 + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

