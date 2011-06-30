
model gulwani_pldi09_5 {

	var i, m, n, b, j1, j2;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := m > 0 && n >= 0 && b >= 0 && m < n && b <= 1 && i = m && j1 = 0 && j2 = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := 0 < i && i < n && b = 1;
		action := i' = i + 1, j1' = j1 + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := 0 < i && i < n && b = 0;
		action := i' = i - 1, j2' = j2 + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

