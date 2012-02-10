<%doc>
  Widgets for advanced mode
</%doc>


<%namespace name="w" file="widgets.mako"/>


## PROPERTIES (advanced mode)

<%def name="properties_fields(props)">

<table>

  ## BOOL properties
  % if "bool" in props:
  <tr style="vertical-align: top">
    <td><span class="label label-success">True/False</span></td>
    <td>
      <% n = 0 %>
      % for p in props["bool"]:
      <label>
	${h.hidden(".properties.bool.%d.id" % n, value=p["name"])}
	${h.checkbox(".properties.bool.%d.checked" % n, value="True", checked=p["val"], class_="watch")}
	${p["name"]}
	<a rel="tooltip" href="#" data-original-title="${p['descr']}">${w.icon("info-sign")}</a>
      </label><br/>
      <% n += 1 %>
      % endfor
    </td>
  </tr>
  % endif

  ## INT properties
  % if "int" in props:
  <tr style="vertical-align: top">
    <td><span class="label label-success">Integer</span></td>
    <td>
      <% n = 0 %>
      % for p in props["int"]:
      <label>
	${h.hidden(".properties.int.%d.id" % n, value=p["name"])}
	${h.checkbox(".properties.int.%d.checked" % n, value="True", checked=True, class_="watch")}
	${p["name"]}
	${h.text(".properties.int.%d.val" % n , value=p["val"], size=5, class_="watch")}
	<a rel="tooltip" href="#" data-original-title="${p['descr']}">${w.icon("info-sign")}</a>
      </label><br/>
      <% n += 1 %>
      % endfor
    </td>
  </tr>
  % endif

  ## STR properties
  % if "str" in props:
  <tr style="vertical-align: top">
    <td><span class="label label-success">String</span></td>
    <td>
      <% n = 0 %>
      % for p in props["str"]:
      <label>
	${h.hidden(".properties.str.%d.id" % n, value=p["name"])}
	${h.checkbox(".properties.str.%d.checked" % n, value="True", checked=True, class_="watch")}
	${p["name"]}
	${h.select(".properties.str.%d.val" % n, "", [(v, v) for v in p["val"]], class_="watch")}
	<a rel="tooltip" href="#" data-original-title="${p['descr']}">${w.icon("info-sign")}</a>
      </label><br/>
      <% n += 1 %>
      % endfor
    </td>
  </tr>
  % endif
  
</table>
</%def>


## ANALYSES (advanced mode)

<%def name="analyses_fields(analyses)">
<% n = 0 %>
% for a in analyses:
<label>
  ${h.hidden(".analyses.%d.id" % n, value=a)}
  ${h.checkbox(".analyses.%d.checked" % n, value="True", checked=True, class_="watch")}
  <span>${a}</span>
  ${h.select(".analyses.%d.val" % n, "", [(v["name"], v["name"]) for v in analyses[a]], class_="watch")}
</label>
<% n += 1 %>
% endfor
</%def>


## PHASES (advanced mode)

<%def name="phases_fields(phases)">
<% n = 0 %>
% for p in phases.get("PHASES", []):
<label>
  ${h.hidden(".phases.%d.id" % n, value=p["name"])}
  ${h.checkbox(".phases.%d.checked" % n, value="True", checked=True, class_="watch")}
  <span>${p["name"]}</span>
</label>
<% n += 1 %>
% endfor
</%def>
