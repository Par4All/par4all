
model katz_isca85 {

	var e, ne, uo, i;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := e = 0 && ne = 0 && uo = 0 && i >= 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := i >= 1;
		action := ne' = ne + e, uo' = uo + 1, i' = i - 1, e' = 0;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := ne + uo >= 1;
		action := i' = i + ne + uo - 1, e' = e + 1, ne' = 0, uo' = 0;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := i >= 1;
		action := i' = i + e + ne + uo - 1, e' = 1, ne' = 0, uo' = 0;
	};

}

strategy s {

	Region init := {state = k1};

}

