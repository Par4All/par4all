
model gonnord_thesis_7_05 {

	var s, t, d;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := s = 0 && t = 0 && d = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s <= 3;
		action := d' = d + 1, s' = s + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := true;
		action := s' = 0, t' = t + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

