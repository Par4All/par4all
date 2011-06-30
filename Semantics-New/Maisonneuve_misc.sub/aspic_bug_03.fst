// Something goes wrong with Aspic v3.1 (compiled on Monday 15 March 2010)
// if acceleration is enabled (default case), which makes Aspic
// fallback to an acceleration-less strategy:
// *Hum* A problem occured when accelerating the loops, now trying with -noaccel option

model aspic_bug_03 {

	var i, m, n, j1, j2;
	states k;

	transition t1 := {
		from := k;
		to := k;
		guard := i > 0 && i < m;
		action := i' = i - 1, j1' = j1 + 1;
	};

	transition t2 := {
		from := k;
		to := k;
		guard := i > 0 && i >= m;
		action := i' = i - m, j2' = j2 + 1;
	};

}

strategy s {

	Region init := {state = k && m > 0 && n >= 0 && m < n && i = n && j1 = 0 && j2 = 0};

}

