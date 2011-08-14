
model gulwani_pldi10_5 {

	var n0, n, i, f, j, s;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := n0 >= 0 && n = n0 && i = 0 && f >= 0 && f <= 1 && j = 0 && s = 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := i < n && s = 1;
		action := f' = 0, s' = 2, j' = j + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := f = 1 && s = 2;
		action := n' = n - 1, j' = j + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := f = 0 && s = 2;
		action := i' = i + 1, s' = 1;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := f = 1 && s = 2;
		action := s' = 1;
	};

}

strategy s {

	Region init := {state = k1};

}

