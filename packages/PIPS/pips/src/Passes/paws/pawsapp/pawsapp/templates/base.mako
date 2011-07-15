# -*- coding: utf-8 -*-
<!DOCTYPE html>

<%inherit file="skeleton.mako"/>

<%def name="level_head()">
	<script type="text/javascript">
		
		$(function(){
			initialize();
		});

		$(document).ready(function() {
			upload_prepare_forms();
		});

		function choose_function() {
			$('#multiple-functions').html(functions);
			$('#multiple-functions input:submit').button();
			$('#multiple-functions input:submit').click(function() {
				$.ajax({
					type: "POST",
					data: {
						function: $(this).attr('value'),
						operation: ${self.operation_id()},
						language: document.language1.lang1.value
					},
					cache: false,
					url: "/operations/perform_multiple",
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
		
		function perform_operation(index, panel_id) {
			clear_result_tab();
			if (check_sources(index, panel_id)) { // all files are ok
				if (multiple) {
					choose_function();
				} else {
					invoke_operation(document.source1.sourcecode1.value, document.language1.lang1.value);
				}
			}
		}

		function invoke_operation(source_code, lang) {
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
					$("#resultcode").html("Web error, try again!");
				},
				success: function(data) {
					$("#resultcode").html(data);
					activate_buttons();
				}
			});
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
		h2 { font-size: 1.0em;}
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

		<div id="dialog-choose-function" title="Select function to transform.">
			<div class="choose-function" id="choose-function-buttons"></div>
		</div>

		<table class="ui-widget ui-widget-content ui-corner-all"><tr valign="top"><td width="20%">

			<div id="left_side">
				<div id='resizing_source'>
					<div style="text-align: right; width: 100%; float: left;"><input name="Aplus" value="A+" type="submit" onClick="resize(1)"/><input name="Aminus" value="A-" type="submit" onClick="resize(0)"/></div>
				</div>
				<br/><br/>
				<p><h3>TYPE OR SELECT SOURCE CODE FROM:</h3></p>
				<br/>
				<br/>
				<div class="load_examples left_side_buttons">
					<p><b>A set of classic examples:</b></p>
					<input value="BROWSE" type="submit"/>
				</div>
				<div class="load_client_file_form left_side_buttons">
					<p><b>Or from your own test cases:</b></p>
	   				<input type="submit" value="BROWSE" id="pseudobutton"/>
					<input type="file" id="inp" name="file" class="hide"/>
					<input type="text" id="pseudotextfile" readonly="readonly"/>
				</div>
				<br/><br/>
				<form action="/" id="buttons">
					<div class="operation left_side_buttons">
						<input type="submit"/>
					</div>
				</form>
				<br/>
				<br/>
				<br/>
				<br/>
				<br/>
				<br/>
				<div class="save_results left_side_buttons">
					<p><a href="/res/result_file" id="save_button_link" style="text-decoration: none"><input value="SAVE RESULT" type="submit" id="save_button"/></a></p>
				</div>
				<div class="print_results left_side_buttons">
					<p><input value="PRINT RESULT" type="submit" id="print_button"/></p>
				</div>
				<br/>
				<br/>
				<div class="left_side_buttons">
					<p><a href=${self.link()}><b>advanced mode</b></a></p>
				</div>
			</div>
		</td><td>
		<div id="tabs">
			<ul>
				<li><a href="#tabs-1" id="source_tab_link1">SOURCE</a></li>
				<li><a href="#result"  id="result_tab_link">${self.operation_id()}</a></li>
				<li><a href="#graph">GRAPH</a></li>
			</ul>
			<div id="tabs-1">
				<form name="language1">
					<label for="lang1">Language: </label>
					<input name="lang1" value="not yet detected." readonly="readonly"/>
				</form>
				<table><tr><td>
				<form name="source1">
					<textarea name="sourcecode1" id="sourcecode1" rows="27" cols="120" onkeydown="handle_key_down(this, event)">Put your source code here.</textarea>
				</form></td></tr></table>
			</div>
			<div id="result">
				<div id="multiple-functions">
				</div>
				<div id="resultcode">
					Result of the transformation will be displayed here.
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

