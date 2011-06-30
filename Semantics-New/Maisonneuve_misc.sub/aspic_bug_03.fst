// Something goes wrong with Aspic v3.1 (compiled on Monday 15 March 2010)
// if acceleration is enabled (default case), which makes Aspic
// fallback to an acceleration-less strategy:
// *Hum* A problem occured when accelerating the loops, now trying with -noaccel option

model aspic_bug_03 {

	var i, m, j;
	states k;

	transition t := {
		from := k;
		to := k;
		guard := i > 0 && i >= m;
		action := i' = i - m, j' = j + 1;
	};

}

strategy s {

	Region init := {state = k && m > 0 && j = 0};

}

