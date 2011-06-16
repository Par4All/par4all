# -*- coding: utf-8 -*-
<!DOCTYPE html>

<%inherit file="skeleton.mako"/>

<%def name="level_head()">
		<script type="text/javascript">

		// ON SIDE LOADING
		$(document).ready(function() {
			
			upload_prepare_forms();
			load_advanced_buttons();
			load_advanced_properties();

		});
		
		function upload_load(response) {
                	
                        upload_add(response);
                        deactivate_buttons();
		}

		function clear_multiple() {}

		function upload_add_multiple(splited) {}
	
		function load_advanced_buttons(){
			$.ajax({
				type: "POST",
				data: {operation: ${self.operation_id()}},
				cache: false,
				url: "/advanced/load_buttons",
				error: function() {
					//console.debug("ERROR!");
				},
				success: function(data) {
					$('#operations_buttons').html("<h3>SELECT ANALYSIS:</h3>" + data);
					$('#operations_buttons input:submit').button();
					$('#operations_buttons input:submit').click(function(event) { 
						event.preventDefault();
						var op_type = $(this).attr('value');
						clear_result_tab();
						var source_code = document.source1.sourcecode1.value;
                                                $.ajax({
                                                        type: "GET",
                                                        cache: false,
                                                        url: "/operations/get_directory",
                                                        error: function() {
                                                                $('#resultcode').html("Web error, try again!");
                                                        },
                                                        success: function(data) {
                                                                directory = data;
                                                                document.getElementById('save_button_link').href = '/res/' + directory;
                                                        }
                                                });
						if (preprocess_input(source_code, 1, 1, result_id)) {
							var lang = document.language1.lang1.value;
							$.ajax({
								type: "POST",
								data: {
									type: $(this).attr('value').replace(/ /gi, '_'),
									operation: ${self.operation_id()},
									code: source_code,
									language: lang,
									properties: check_properties()
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
						
					});
				}
			});
		};

		function set_analysis(filename) {
			
			$.ajax({
				type: "POST",
				data: {
					name: filename,
					operation: ${self.operation_id()}
				},
				cache: false,
				url: "/examples/get_analysis",
				error: function() {
					//console.debug("WEB error");	
				},
				success: function(data) {
					$('#operations_buttons').html("<h3>ANALYSIS:</h3>" + data);
					$('#operations_buttons input:submit').button();
					$('#operations_buttons input:submit').click(function(event){
						event.preventDefault();
					});
				}
			});
		};

		function check_properties() {
			var props = "";
			if (document.properties.properties_bools instanceof NodeList) {
				for ( i = 0; i < document.properties.properties_bools.length; i++) {
					props += document.properties.properties_bools[i].value + " " + document.properties.properties_bools[i].checked + ";";
				}
			} else {
				props += document.properties.properties_bools.value + " " + document.properties.properties_bools.checked + ";";
			}
			if (document.properties.properties_ints instanceof NodeList) {
				for ( i = 0; i < document.properties.properties_ints.length; i++) {
					if (document.properties.properties_ints[i].checked) {
						props += document.properties.properties_ints[i].value + " ";
						props += $('#int_' + document.properties.properties_ints[i].value).attr('value')+";";
					}
				}
			} else {
				if (document.properties.properties_ints.checked)
					props += document.properties.properties_ints.value + " " + $('#int_' + document.properties.properties_ints.value).attr('value') + ";";
			}
			if (document.properties.properties_strs instanceof NodeList) {
				for ( i = 0; i < document.properties.properties_strs.length; i++) {
					if (document.properties.properties_strs[i].checked) {
						props += document.properties.properties_strs[i].value + " ";
						props += $('#str_' + document.properties.properties_strs[i].value).attr('value')+";";
					}
				}
			} else {
				if (document.properties.properties_strs.checked)
					props += document.properties.properties_strs.value + " " + $('#str_' + document.properties.properties_strs.value).attr('value') + ";";
			}
			return props;
		};

		function load_advanced_properties(){
			$.ajax({
				type: "POST",
				data: {operation: ${self.operation_id()}},
				cache: false,
				url: "/advanced/load_properties",
				error: function() {
					//console.debug("ERROR!");
				},
				success: function(data){
					$('#operations_properties').html("<h3>POSSIBLE PROPERTIES:</h3>" + data);
					$('#operations_properties input:submit').button();
				}
			});
		};

		function set_properties(filename) {
			$.ajax({
				type: "POST",
				data: {
					name: filename,
					operation: ${self.operation_id()}
				},
				cache: false,
				url: "/examples/get_properties",
				error: function() {
					//console.debug("ERROR");
				},
				success: function(data) {
					$('#operations_properties').html("<h3>PROPERTIES:</h3>" + data);
				}
			});
		};

		function perform_operation() {
			clear_result_tab();
			var source_code = document.source1.sourcecode1.value;
                        $.ajax({
                        	type: "GET",
                                cache: false,
                                url: "/operations/get_directory",
                                error: function() {
                                	$('#resultcode').html("Web error, try again!");
                                },
                                success: function(data) {
                                	directory = data;
                                        document.getElementById('save_button_link').href = '/res/' + directory;
                                }
			});
			if (preprocess_input(source_code, 1, 1, result_id)) {
				var lang = document.language1.lang1.value;
				$.ajax({
					type: "POST",
					data: {
						code: source_code,
						language: lang,
						operation: ${self.operation_id()}
					},
					cache: false,
					url: "/operations/perform", 
					error: function() {
						$("#resultcode").html("Web error; try again!");
					},
					success: function(data) {
						$("#resultcode").html(data);
						activate_buttons();
					}
				});
			}
		}
	
		function load_example_success() {
			// set_properties(filename);
			// set_analysis(filename);
		}
		
		$(function(){
			
			initialize();
		});

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
		</style>
</%def>

<%def name="content()">

		<iframe id="iframetoprint" style="height: 0px; width: 0px; position: absolute; -moz-opacity: 0; opacity: 0"></iframe>

		<div id="dialog-error-examples" title="ERROR">
			<p>Error while loading examples, try again!</p>
		</div>		

		<div id="dialog-load-examples" title="Select an example.">
			<div class="select-examples" id="select-examples-buttons">
				<input value="LOAD" type="submit">
			</div>
		</div>		

		<table class="ui-widget ui-widget-content ui-corner-all"><tr valign="top"><td width="30%">

			<div id="left_side">
				<br/><br/>
				<p><h3>TYPE OR SELECT SOME SOURCE CODE FROM:</h3></p>
				<div class="load_examples left_side_buttons">
					<p><b>A set of classic examples:</b>&nbsp;<input value="BROWSE" type="submit"/></p>
				</div>
				<div class="load_client_file_form left_side_buttons">
					<p><b>Or from your own test cases:</b><br/>
	   				<input type="submit" value="BROWSE" id="pseudobutton"/>
					<input type="file" id="inp" name="file" class="hide"/>
					<input type="text" id="pseudotextfile" readonly="readonly"/></p>
				</div>
				<p align="right"><a href="http://www.cri.ensmp.fr/pips/pipsmake-rc.htdoc/" target="_blank">Help! What does it mean?</a></p>
				<form action="/" id="properties" name="properties">
					<div class="properties left_side_buttons" id="operations_properties" name="operations_properties">
						<!-- <input type="submit"/> -->
					</div>
				</form>
				<br/>
				<form action="/" id="buttons">
					<div class="operation left_side_buttons" id="operations_buttons">
						<!-- <input type="submit"/> -->
					</div>
				</form>
                                <br/>
                                <br/>
                                <div class="save_results left_side_buttons">
                                        <p><a href="/res/result_file" id="save_button" style="text-decoration: none"><input value="SAVE RESULT" type="submit" id="save_button"/></a></p>
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
				<li><a href="#tabs-2" id="result_tab_link">RESULT</a></li>
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
			<div id="tabs-2">
				<div id="resultcode">
					Choose analysis to get the result.
				</div>
			</div>
		</div>
		</td></tr></table>
</%def>
