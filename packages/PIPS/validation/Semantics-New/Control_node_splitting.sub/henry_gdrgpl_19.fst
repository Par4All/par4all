
model henry_gdrgpl_19 {

	var x, d;
	states k1, k2;

	transition t_ini := {
		from := k1;
		to := k2;
		guard := x = 0 && d = 1;
		action := ;
	};

	transition t1 := {
		from := k2;
		to := k2;
		guard := x = 0;
		action := d' = 1, x' = x + d;
	};

	transition t2 := {
		from := k2;
		to := k2;
		guard := 1 <= x && x <= 999;
		action := x' = x + d;
	};

	transition t3 := {
		from := k2;
		to := k2;
		guard := x = 1000;
		action := d' = -1, x' = x + d;
	};

}

strategy s {

	Region init := {state = k1};

}

