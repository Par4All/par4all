<%doc>
  Generic page for a tool
</%doc>


<%inherit file="base.mako"/>

<%namespace name="w"   file="pawsapp:templates/lib/widgets.mako"/>

<%def name="js_slot()">
${h.javascript_link(request.route_url("routes.js"))}
<script type="text/javascript">
  operation = "demo";
  $(function () {
  $(".pagination li a").popover({html:true, placement: "bottom"});
  });
</script>
</%def>




## ONE COLUMN

<%def name="content()">

<div class="container-fluid">

  <div class="hero-unit" style="padding:.5em 1em">
    <h2>${h.image(request.static_url("pawsapp:static/img/favicon-trans.gif"), u"PAWS icon")}
      ${info["title"]}
    </h2>
    ${info["descr"]|n}
  </div>

  <div class="pagination">
    <ul>
      ## 'Previous' link
      <li class="${h.css_classes([('disabled', step==0)])}">
	${h.link_to(u"«", url=request.route_url(request.matched_route.name, tutorial=name, _query=dict(step=step-1)) if step > 0 else "#")}</li>

      ## Overview
      <li class="${h.css_classes([('active', step==0)])}">${h.link_to(u"Overview", url=request.route_url(request.matched_route.name, tutorial=name))}</li>

      ## Steps 1..n
      % for i in range(1, nb_steps+1):
      <li class="${h.css_classes([('active', step==i)])}">
	<a href="${request.route_url(request.matched_route.name, tutorial=name, _query=dict(step=i))}"
	   data-content="${comments[i-1]}" data-original-title="Step ${i}">${i}</a></li>
      % endfor
      ## 'Next' link
      <li class="${h.css_classes([('disabled', step==nb_steps)])}">
	${h.link_to(u"»", url=request.route_url(request.matched_route.name, tutorial=name, _query=dict(step=step+1)) if step < nb_steps else "#")}</li>
    </ul>
  </div>

  <div id="demo" class="row-fluid">
    <div class="span6">
      <span class="label">SOURCE</span>
      <div>
	% if step == 0:
	<textarea name="sourcecode" id="sourcecode" class="span12" rows=34 style="height:400px"
		  onkeydown="handle_key_down(this, event)">${source}</textarea>
	% else:
	${source|n}
	% endif
      </div>
      <button class="btn">${w.icon("download-alt")} Save</button>
      <button class="btn">${w.icon("print")} Print</button>
    </div>
    <div class="span6">
      <div>
	<span class="label">SCRIPT</span>
      </div>
      <div id="sourcetpips" style="overflow:auto">
	${tpips|n}
      </div>
      <button class="btn">${w.icon("download-alt")} Save</button>
      <button class="btn">${w.icon("print")} Print</button>
    </div>
  </div>

</div>
</%def>

