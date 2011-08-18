
model gulwani_pldi10_2 {

	var n0, n, m, s, i;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := n0 > 0 && m > 0 && n = n0 && s = 1 && i = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := n > 0 && m > 0 && s = 1;
		action := n' = n - 1, m' = m - 1, i' = i + 1, s' = 2;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := n > 0 && s = 2;
		action := n' = n - 1, m' = m + 1, i' = i + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

