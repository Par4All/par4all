
model gonnord_thesis_2_15 {

	var s, l, t, x;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := s = 0 && l = 0 && t = 0 && x = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 0 && x <= 9;
		action := t' = t + 1, l' = l + 1, x' = x + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 0;
		action := x' = 0, s' = 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 1;
		action := t' = t + 1, x' = x + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

