
model gulwani_pldi10_7 {

	var i, t, n, j;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := i >= 0 && n >= 0 && i <= n && t = i + 1 && j = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := t != i && t <= n;
		action := t' = t + 1, j' = j + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := t != i && t > n;
		action := t' = 0, j' = j + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

