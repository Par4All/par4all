<?php

if(len(sys.argv) < 5) {
 echo "usage : ....";
 exit(1);
}

$texfile = sys.argv[1]
$generator = sys.argv[4]

$input = open(sys.argv[3],"r");
$lines = input.readlines();
close($input);

foreach($line in $lines) {
  $m = preg_match(r"(.*?):\s*(.*)", $line)
  $p = $m[1]
	
  $deps = []
  if($m['lastindex'] == 2) {
    $deps = split(" ", $m.group(2));
  }
  $pipsdep[$p] = deps;
}

#Read properties into a string
$rcfile = sys.argv[2];
$input = open(rcfile,"r");
$lines = input.readlines();
close($input);

$pipsprops = dict()
foreach($line in $lines) {
  $m = preg_match("\s*(.*?)\s+(.*)", $line)
  $d = $m[2]
  $pipsprops[$m.group(1)] = $d
}

#Read input tex file into a string
$input = open(texfile,"r");
$lines = input.readlines();
$rc = "".join(lines);
close($input);


function printPhpMethod($name,$doc) {
	$extraparamsetter = "";
	$extraparamresetter = "";
	$extraparams = "";
	$has_loop_label = False;

	if($name in $pipsdep and len($pipsdep[$name]) > 0 {
		$props = [];
		for(prop in pipsdep[name]) {
			$short_prop = re.sub(r'^' + name + '\_(.*)', r'\1', $prop)
			$arg = $short_prop + "=None" # + pipsprops[prop.upper()]

			if($prop == "loop_label") {
				$has_loop_label = True;
				$extraparamsetter += '\tif self.workspace:self.workspace.cphpips.push_property("LOOP_LABEL",phpipsutils.formatprop(self.label))\n';
				$extraparamresetter = '\t\tif self.workspace:self.workspace.cphpips.pop_property("LOOP_LABEL")\n'  + extraparamresetter;
			} else {
				$props+=$arg
				$extraparamsetter += '\tif '+short_prop+' == None and self.workspace:'+short_prop+'=self.workspace.props.'+prop + '\n';
				$extraparamsetter += '\tif self.workspace:self.workspace.cphpips.push_property("%s",phpipsutils.formatprop(%s))\n' % ( prop.upper(), short_prop);
				$extraparamresetter = '\t\tif self.workspace:self.workspace.cphpips.pop_property("%s")\n' % (prop.upper()) + extraparamresetter;
			}
		}
		if(len($props) > 0) {
			$extraparams = ",".join($props) + ",";
		}
	
	#Some regexp to filter the LaTeX source file, sometimes they work, sometimes they don't,
	#sometimes it's worth than before but they only act one the produced PHP comments
	$doc = re.sub(r'(?ms)(\\begin\{.*?\})|(\\end\{.*?\})|(\\label\{.*?\})','',doc)  #Remove any begin,end and label LaTeX command
	$doc = re.sub(r'(?ms)(\\(.*?)\{.*?\})', r'', doc)#, flags=re.M|re.S) #Remove any other LaTeX command
	$doc = doc.replace("\_","_") #Convert \_ occurences to _
	$doc = doc.replace("~"," ")  #Convert ~ to spaces
	$doc = re.sub(r"\\verb\|(.*?)\|", r"\1", doc)#, flags=re.M|re.S) #Replace \verb|somefile| by somefile
	$doc = re.sub(r"\\verb\/(.*?)\/", r"\1", doc)#, flags=re.M|re.S) #Replace \verb/something/ by something
	$doc = re.sub(r"\\verb\+(.*?)\+", r"\1", doc)#, flags=re.M|re.S) #Replace \verb+something+ by something
	$doc = doc.replace("\PIPS{}","PIPS") #Convert \PIPS{} to PIPS
	$name = re.sub(r'\s',r'_',name)

	$mself = "self"
	if($has_loop_label and $generator == "-loop") {
		$mself = "self.module";
	}
	
	if ($has_loop_label and $generator == "-loop") or (! $has_loop_label and $generator != "-loop") {
		if(generator == "-modules")
			$extraparams = $extraparams + " concurrent=False,";

		echo '\ndef '+$name+'(self,'+$extraparams+' **props):';
		echo '\t"""'+$doc+'"""';
		echo $extraparamsetter;
		echo '\tif '+$mself+'.workspace: old_props = phpipsutils.set_properties(self.workspace,phpipsutils.update_props("'+$name.upper()+'",props))';

		echo '\ttry:';
		if(generator != "-modules")
			echo '\t\tphpipsutils.apply('+mself+',\"'+$name+'\")';
		else {
			echo '\t\tif concurrent: phpipsutils.capply(self,\"'+$name+'\")';
			echo '\t\telse:'
			echo '\t\t\tfor m in self: phpipsutils.apply(m,\"'+$name+'\")';
		}
		echo '\texcept:';
		echo '\t\traise';
		echo '\tfinally:';
		echo '\t\tif '+$mself+'.workspace: phpipsutils.set_properties('+$mself+'.workspace,old_props)';
		echo '\n' + $extraparamresetter;
		echo $generator[1:] + "." + $name + "=" + $name;

foreach($dstr in $doc_strings) {
  $m = preg_match('\{([^\}]+)\}[\n]+(.*)', $dstr, $flags)
  printPHPMethod($m[1], $m[2])
}
