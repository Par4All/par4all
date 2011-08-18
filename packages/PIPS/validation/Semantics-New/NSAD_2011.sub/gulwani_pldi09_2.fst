
model gulwani_pldi09_2 {

	var v1, v2, n, m, i1, i2;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := n > 0 && m > 0 && v1 = n && v2 = 0 && i1 = 0 && i2 = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := v1 > 0 && v2 < m;
		action := v2' = v2 + 1, v1' = v1 - 1, i1' = i1 + 1, i2' = 0;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := v1 > 0 && v2 >= m;
		action := v2' = 0, i2' = i2 + m;
	};

}

strategy s {

	Region init := {state = k1};

}

