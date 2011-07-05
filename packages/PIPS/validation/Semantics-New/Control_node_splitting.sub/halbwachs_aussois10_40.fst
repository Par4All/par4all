
model halbwachs_aussois10_40 {

	var v, t, x, d;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := v = 0 && t = 0 && x = 0 && d = 0;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := x <= 4;
		action := x' = x + 1, v' = v + 1;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := d <= 9;
		action := d' = d + 1, t' = t + 1;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := d = 10 && x >= 2;
		action := x' = 0, d' = 0;
	};
	
}

strategy s {

	Region init := {state = k1};

}

