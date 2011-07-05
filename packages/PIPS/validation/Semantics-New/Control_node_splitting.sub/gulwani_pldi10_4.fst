
model gulwani_pldi10_4 {

	var n0, n, f, s, i;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := n0 >= 0 && n = n0 && f = 1 && s = 1 && i = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && f = 1;
		action := s' = 2, f' = 0, i' = i + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := n > 0 && s = 2;
		action := n' = n - 1, f' = 1, i' = i + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

