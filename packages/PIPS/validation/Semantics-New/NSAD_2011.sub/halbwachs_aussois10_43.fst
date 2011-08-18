
model halbwachs_aussois10_43 {

	var t, s, d;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := t = 0 && s = 0 && d = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := s <= 3;
		action := s' = s + 1, d' = d + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := true;
		action := t' = t + 1, s' = 0;
	};

}

strategy s {

	Region init := {state = k1};

}

