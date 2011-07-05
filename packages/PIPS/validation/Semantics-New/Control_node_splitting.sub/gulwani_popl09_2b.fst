
model gulwani_popl09_2b {

	var x0, z0, n, x, z, c1, c2;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x0 >= 0 && z0 >= 0 && n >= 0 && n >= x0 && n >= z0 && x = x0 && z = z0 && c1 = 0 && c2 = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := x < n && z > x;
		action := x' = x + 1, c1' = c1 + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := x < n && z <= x;
		action := z' = z + 1, c2' = c2 + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

