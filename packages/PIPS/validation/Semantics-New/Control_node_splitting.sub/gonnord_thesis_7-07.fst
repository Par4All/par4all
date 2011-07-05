
model gonnord_thesis_7_07 {

	var u, t, l, v;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := u = 0 && t = 0 && l = 0 && v = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := u <= 59 && v <= 9;
		action := u' = u + 1, t' = t + 1, l' = l + 1, v' = v + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := u <= 59;
		action := u' = u + 1, t' = t + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := u = 60;
		action := u' = 0, v' = 0;
	};

}

strategy s {

	Region init := {state = k1};

}

