# -*- coding: utf-8 -*-
<!DOCTYPE html>

<%inherit file="skeleton.mako"/>

<%def name="level_head()">
	<script type="text/javascript">
	
	(function($) {
		$.widget("ui.combobox", {
			_create: function(){},
			destroy: function(){}
		});
	})(jQuery);

	$(function(){
		initialize();
		
		$('#combobox').combobox();
	});

	function enable_performing(){
		performed = false;
	}

	// ON SIDE LOADING
	$(document).ready(function() {

		skn = document.getElementById('pdqbox').style;
		skn.visibility = "visible";
		skn.display = "none";
		document.onmousemove = get_mouse;

		upload_prepare_forms();
		load_advanced_analyses();
		load_advanced_properties();
		load_advanced_phases();
	});
	
	function load_advanced_properties(){
		$.ajax({
			type: "POST",
			data: {operation: ${self.operation_id()}},
			cache: false,
			url: "/advanced/load_properties",
			success: function(data){
				$('#operations_properties').html("<h3>PROPERTIES:</h3>" + data);
				$('#operations_properties input:submit').button();
				$('#properties_strs select').combobox();
			}
		});
	};
	
	function load_advanced_phases(){
		$.ajax({
			type: "POST",
			data: {operation: ${self.operation_id()}},
			cache: false,
			url: "/advanced/load_phases",
			success: function(data){
				$('#operations_phases').html("<h3>PHASES:</h3>" + data);
				$('#operations_phases input:submit').button();
			}
		});
	};

	function load_advanced_analyses(){
		$.getJSON("/advanced/load_analyses", {operation: ${self.operation_id()}}, function(data) {
			analyses_groups = data[0];
			$('#operations_analyses').html("<h3>SELECT ANALYSES:</h3>" + data[1]);
			$('#operations_analyses input:submit').button();
		});
	};

	function perform_operation(index, panel_id) {
		clear_result_tab();
		var props = get_checked_properties_values(document.properties.properties_ints, '#int_');
		props += get_checked_properties_values(document.properties.properties_strs, '#str_');
		if (validate_properties(props)) {
			props += get_checked_bool_properties(document.properties.properties_bools);
			if(check_sources(index, panel_id)) {
				if (multiple) {
					choose_function(props);
				} else {
					invoke_operation(document.source1.sourcecode1.value, document.language1.lang1.value, props);
				}
			}
		}
	}

	function invoke_operation(source_code, lang, props) {
		$.ajax({
			type: "POST",
			data: {
				operation: ${self.operation_id()},
				code: source_code,
				language: lang,
				properties: props,
				analyses: get_analyses(),
				phases: get_checked_bool_properties(document.phases.phases)
			},
			cache: false,
			url: "/operations/perform_advanced",
			error: function() {
				$('#resultcode').html("Web error, try again later.");
			},
			success: function(data) {
				$('#resultcode').html(data);
				activate_buttons();
			}
		});
	}

	function choose_function(props) {
		$('#multiple-functions').html(functions);
		$('#multiple-functions input:submit').button();
		$('#multiple-functions input:submit').click(function() {
			$.ajax({
				type: "POST",
				data: {
					function: $(this).attr('value'),
					operation: ${self.operation_id()},
					language: document.language1.lang1.value,
					properties: props,
					analyses: get_analyses(),
					phases: get_checked_bool_properties(document.phases.phases)
				},
				cache: false,
				url: "/operations/perform_multiple_advanced",
				error: function() {
				},
				success: function(data) {
					$('#resultcode').html(data);
					activate_buttons();
				}
			});
		});
		add_choose_function_notification();
	}

	function validate_properties(props) {
		var test = true;
		if (props != "") {
			$.ajax({
				type: "POST",
				data: { properties: props},
				cache: false,
				async: false,
				url: "/advanced/validate",
				error: function(data) {
					alert(data);
					test = false;
				},
				success: function(data) {
					if (data != "") {
						alert(data);
						test = false;
					}
				}
			});
		}
		return test;
	}

	function get_analyses() {
		var analysis = "";
		var types = analyses_groups.toString().split(',');
		for(var i = 0; i < types.length; i++) {
			if (document.forms['analyses'].elements[types[i]].checked) {
				analysis += $('#analysis_' + document.forms['analyses'].elements[types[i]].value).attr('value') + ';';
			}
		}
		return analysis;
	}
		
	function get_checked_bool_properties(properties_form) {
		if (properties_form instanceof NodeList) {
			var props = "";
			for (var i = 0; i < properties_form.length; i++)
				props += properties_form[i].value + " " + properties_form[i].checked + ";";
			return props;
		}
		return properties_form.value + " " + properties_form.checked + ";";
	}

	function get_checked_properties_values(properties_form, prefix) {
		if (properties_form instanceof NodeList) {
			var props = "";
			for (var i = 0; i < properties_form.length; i++) {
				if (properties_form[i].checked) {
					props += properties_form[i].value + " ";
					props += $(prefix + properties_form[i].value).attr('value') + ';';
				}
			}
			return props;
		}
		if (properties_form && properties_form.checked)
			return properties_form.value + " " + $(prefix + properties_form.value).attr('value') + ';';
		else
			return "";
	}
	</script>
	
	<style type="text/css">
		body{ font: 62.5% "Trebuchet MS", sans-serif; margin: 50px;}
        	.demoHeaders { margin-top: 2em; }
                #dialog_link {padding: .4em 1em .4em 20px;text-decoration: none;position: relative;}
                #dialog_link span.ui-icon {margin: 0 5px 0 0;position: absolute;left: .2em;top: 50%;margin-top: -8px;}
                ul#icons {margin: 0; padding: 0;}
                ul#icons li {margin: 2px; position: relative; padding: 4px 0; cursor: pointer; float: left;  list-style: none;}
                ul#icons span.ui-icon {float: left; margin: 0 4px;}
		input.hide { position: absolute; height: 30px; left: -90px; -moz-opacity: 0; filter: alpha(opacity: 0); opacity: 0; z-index: 2;}
		h3 { color: #185172;}
		.left_side_buttons { margin-left:10px;}
		.boxpopup {font-family: Arial, sans-serif; font-size:120%; color:black; background:#DEEDF7;width:200px;text-align:center;padding:4px 5px 4px 5px;font-weight:bold;border:1px solid gray;}
		#pdqbox {position:absolute; visibility:hidden; z-index:200;}
	</style>
</%def>

<%def name="content()">

	<iframe id="iframetoprint" style="height: 0px; width: 0px; position: absolute; -moz-opacity: 0; opacity: 0"></iframe>

	<div id="pdqbox"></div>

	<div id="dialog-error-examples" title="ERROR">
		<p>Error while loading examples, try again!</p>
	</div>		

	<div id="dialog-load-examples" title="Select an example.">
		<div class="select-examples" id="select-examples-buttons">
			<input value="LOAD" type="submit">
		</div>
	</div>		

	<div id="dialog-choose-function" title="Select function to transform.">
		<div class="choose-function" id="choose-function-buttons"></div>
	</div>

	<table class="ui-widget ui-widget-content ui-corner-all"><tr valign="top"><td width="32%">

		<div id="left_side">
			<div id='resizing_source'>
				<div style="text-align: right; width: 100%; float: left;"><input name="Aplus" value="A+" type="submit" onClick="resize(1)"/><input name="Aminus" value="A-" type="submit" onClick="resize(0)"/></div>
			</div>
			<br/><br/>
			<p><h3>TYPE OR SELECT SOME SOURCE CODE FROM:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</h3></p>
			<div class="load_examples left_side_buttons">
				<p><b>A set of classic examples:</b>&nbsp;<input value="BROWSE" type="submit"/></p>
			</div>
			<div class="load_client_file_form left_side_buttons">
				<p><b>Or from your own test cases:</b><br/>
	 			<input type="submit" value="BROWSE" id="pseudobutton"/>
				<input type="file" id="inp" name="file" class="hide"/>
				<input type="text" id="pseudotextfile" readonly="readonly"/></p>
			</div>
			<!--<p align="right"><br/>To get information about property<br/>hover your mouse over its button.</p>-->
			<form action="/" id="properties" name="properties">
				<div class="properties left_side_buttons" id="operations_properties" name="operations_properties">
					<!-- <input type="submit"/> -->
				</div>
			</form>
			<form action="/" id="analyses" name="analyses">
				<div class="analyses left_side_buttons" id="operations_analyses" name="operations_analyses">
					
				</div>
			</form>
			<br/>
			<form action="/" id="phases" name="phases">
				<div class="phases left_side_buttons" id="operations_phases" name="operations_phases">
					
				</div>
			</form>
			<br/>
			<form action="/" id="buttons">
				<div class="operation left_side_buttons" id="operations_buttons">
					<input type="submit"/>
				</div>
			</form>
                        <br/>
                        <br/>
                        <div class="save_results left_side_buttons">
                        	<p><a href="/res/result_file" id="save_button_link" style="text-decoration: none"><input value="SAVE RESULT" type="submit" id="save_button"/></a></p>
                        </div> 
                        <div class="print_results left_side_buttons">
                        	<p><input value="PRINT RESULT" type="submit" id="print_button"/></p>
                        </div>
                        <br/>
                        <div class="left_side_buttons">
                        	<p><a href=${self.link()}><b>basic mode</b></a></p>
                        </div>

		</div>
	</td><td>
	<div id="tabs">
		<ul>
			<li><a href="#tabs-1" id="source_tab_link1">SOURCE</a></li>
			<li><a href="#result" id="result_tab_link">RESULT</a></li>
			<li><a href='#graph'>GRAPH</a></li>
		</ul>
		<div id="tabs-1">
			<form name="language1">
				<label for="lang1">Language: </label>
				<input name="lang1" value="not yet detected." readonly="readonly"/>
			</form>
			<form name="source1">
				<textarea name="sourcecode1" id="sourcecode1" rows="50" cols="130" onkeydown="handle_key_down(this, event)">Put your source code here.</textarea>
			</form>
			C and Fortran supported!
		</div>
		<div id="result">
			<div id="multiple-functions">
			</div>
			<div id="resultcode">
				Choose analysis to get the result.
			</div>
		</div>
		<div id="graph">
			<div id="dependence_graph">
Please wait, it might take long time.
			</div>
		</div>
	</div>
	</td></tr></table>
</%def>
