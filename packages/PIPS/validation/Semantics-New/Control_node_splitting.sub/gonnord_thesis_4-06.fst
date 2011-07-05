
model gonnord_thesis_4_06 {

	var s, d, v, t;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := s = 1 && d = 0 && v = 0 && t = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s = 1 && t <= 2;
		action := v' = 0, t' = t + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := s = 1 && v <= 1 && d <= 8;
		action := v' = v + 1, d' = d + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := s = 1 && t >= 3;
		action := s' = 2;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := s = 1 && d >= 10;
		action := s' = 3;
	};

	transition t5 := {
		from := k2;
		to := k2;
		guard := s = 1 && v >= 3;
		action := s' = 4;
	};

}

strategy s {

	Region init := {state = k1};

}

