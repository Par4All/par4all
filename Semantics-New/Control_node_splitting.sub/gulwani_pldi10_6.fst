
model gulwani_pldi10_6 {

	var n0, x0, z0, n, x, z, i1, i2;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := n0 >= 0 && x0 >= 0 && z0 >= 0 && n0 >= x0 && n0 >= z0 && n = n0 && x = x0 && z = z0 && i1 = 0 && i2 = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := x < n && z > x;
		action := x' = x + 1, i1' = i1 + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := x < n && z <= x;
		action := z' = z + 1, i2' = i2 + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

