
model gonnord_thesis_9_03 {

	var b, s, d, t;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := b = 0 && s = 0 && d = 0 && t = 1;
		action := ;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := t = 1 && b > s - 9;
		action := s' = s + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := t = 1 && b = s + 9;
		action := t' = 3, b' = b + 1, d' = 0;
	};

	transition t4 := {
		from := k2;
		to := k2;
		guard := t = 3 && b = s + 1;
		action := t' = 1, s' = s + 1, d' = 0;
	};

	transition t5 := {
		from := k2;
		to := k2;
		guard := t = 2 && b = s - 1;
		action := t' = 1, b' = b + 1;
	};

	transition t6 := {
		from := k2;
		to := k2;
		guard := t = 1 && b = s - 9;
		action := t' = 2, s' = s + 1;
	};

	transition t7 := {
		from := k2;
		to := k2;
		guard := t = 2 && b < s - 1;
		action := b' = b + 1;
	};

	transition t8 := {
		from := k2;
		to := k2;
		guard := t = 3 && d = 9;
		action := t' = 4, s' = s + 1;
	};

	transition t9 := {
		from := k2;
		to := k2;
		guard := t = 3 && b > s + 9;
		action := s' = s + 1;
	};

	transition t10 := {
		from := k2;
		to := k2;
		guard := t = 3 && d < 9;
		action := d' = d + 1, b' = b + 1;
	};

	transition t11 := {
		from := k2;
		to := k2;
		guard := t = 4 && b > s + 1;
		action := s' = s + 1;
	};

	transition t12 := {
		from := k2;
		to := k2;
		guard := t = 4 && b = s + 1;
		action := t' = 1, s' = s + 1, d' = 0;
	};

	transition t13 := {
		from := k2;
		to := k2;
		guard := t = 1 && b < s + 9;
		action := b' = b + 1;
	};

}

strategy s {

	Region init := {state = k1};

}

