<%doc>
  Widgets for advanced mode
</%doc>


## PROPERTIES (advanced mode)

<%def name="properties_fields(props)">

<table>

  ## BOOL properties
  % if "bool" in props:
  <tr style="vertical-align: top">
    <td><span class="label label-success">True/False</span></td>
    <td>
      % for p in props["bool"]:
      <label>
	${h.checkbox("bools", value=p["name"], checked=p["val"])}
	<span>${p["name"]}</span>
      </label><br/>
      % endfor
    </td>
  </tr>
  % endif

  ## INT properties
  % if "int" in props:
  <tr style="vertical-align: top">
    <td><span class="label label-success">Integer</span></td>
    <td>
      % for p in props["int"]:
      <label>
	${h.checkbox("bools", value=p["name"], checked=True)}
	<span>${p["name"]}</span>
      ${h.text(p["name"], value=p["val"], size=5)}
      </label><br/>
      % endfor
    </td>
  </tr>
  % endif

  ## STR properties
  % if "str" in props:
  <tr style="vertical-align: top">
    <td><span class="label label-success">String</span></td>
    <td>
      % for p in props["str"]:
      <label>
	${h.checkbox("bools", value=p["name"], checked=True)}
	<span>${p["name"]}</span>
      ${h.select(p["name"], "", [(v, v) for v in p["val"]])}
      </label><br/>
      % endfor
    </td>
  </tr>
  % endif
  
</table>
</%def>


## ANALYSES (advanced mode)

<%def name="analyses_fields(analyses)">
% for a in analyses:
<label>
  ${h.checkbox("analyses", value=a, checked=True)}
  <span>${a}</span>
  ${h.select(a, "", [(v["name"], v["name"]) for v in analyses[a]])}
</label>
% endfor
</%def>


## PHASES (advanced mode)

<%def name="phases_fields(phases)">
% for p in phases.get("PHASES", []):
<label>
  ${h.checkbox("phases", value=p["name"], checked=True)}
  <span>${p["name"]}</span>
</label>
% endfor
</%def>
