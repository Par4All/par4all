
model gulwani_pldi09_3 {

	var i, j, m, n, l1, l2;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := i = 0 && j = 0 && l1 = 0 && l2 = 0 && m >= 0 && n >= 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := i < n && j < m;
		action := j' = j + 1, l1' = l1 + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := i < n && j >= m;
		action := j' = 0, i' = i + 1, l1' = 0, l2' = l2 + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

