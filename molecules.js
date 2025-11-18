// index.js
(function() {
  "use strict";

  var state = {
    moleculesIndex: [],
    metamaterialsIndex: [],
    moleculesDetailCache: {},
    metamaterialsDetailCache: {},
    type: "molecule",
    filtered: [],
    page: 1,
    pageSize: 100,
    selectedId: null,
    loadingDetailId: null
  };

  var els = {};

  function qs(id) {
    return document.getElementById(id);
  }

  function initRefs() {
    els.fileMolecules = qs("file-molecules");
    els.fileMetas = qs("file-metas");
    els.dataTypeSelect = qs("data-type-select");
    els.sortSelect = qs("sort-select");
    els.filterFormula = qs("filter-formula");
    els.filterElement = qs("filter-element");
    els.minAtoms = qs("min-atoms");
    els.maxAtoms = qs("max-atoms");
    els.stabilityRange = qs("stability-range");
    els.stabilityRangeValue = qs("stability-range-value");
    els.pageSize = qs("page-size");
    els.applyFiltersBtn = qs("apply-filters-btn");
    els.summaryText = qs("summary-text");
    els.mainHeaderCounter = qs("main-header-counter");
    els.listHeaderMeta = qs("list-header-meta");
    els.listTbody = qs("list-tbody");
    els.paginationInfo = qs("pagination-info");
    els.btnFirst = qs("btn-first");
    els.btnPrev = qs("btn-prev");
    els.btnNext = qs("btn-next");
    els.btnLast = qs("btn-last");
    els.vizCanvas = qs("viz-canvas");
    els.vizTag = qs("viz-tag");
    els.statsBody = qs("stats-body");
    els.statsTitle = qs("stats-title");
  }

  function attachEvents() {
    if (els.fileMolecules) {
      els.fileMolecules.addEventListener("change", function(e) {
        var file = e.target.files[0];
        if (!file) return;
        readJsonFile(file, function(obj) {
          var mols = Array.isArray(obj.molecules) ? obj.molecules : (obj.molecules || obj.molecules_geom || []);
          state.moleculesIndex = normalizeMoleculeIndex(mols);
          state.moleculesDetailCache = {};
          refreshAll();
        });
      });
    }

    if (els.fileMetas) {
      els.fileMetas.addEventListener("change", function(e) {
        var file = e.target.files[0];
        if (!file) return;
        readJsonFile(file, function(obj) {
          var metas = Array.isArray(obj.metamaterials) ? obj.metamaterials : (obj.metamaterials || obj.metas || []);
          state.metamaterialsIndex = normalizeMetamaterialIndex(metas);
          state.metamaterialsDetailCache = {};
          refreshAll();
        });
      });
    }

    els.dataTypeSelect.addEventListener("change", function() {
      state.type = els.dataTypeSelect.value === "metamaterial" ? "metamaterial" : "molecule";
      state.page = 1;
      state.selectedId = null;
      refreshAll();
    });

    els.sortSelect.addEventListener("change", function() {
      state.page = 1;
      applyFilters();
    });

    els.stabilityRange.addEventListener("input", function() {
      els.stabilityRangeValue.textContent = parseFloat(els.stabilityRange.value).toFixed(2);
    });

    els.applyFiltersBtn.addEventListener("click", function() {
      var ps = parseInt(els.pageSize.value, 10);
      if (!isFinite(ps) || ps <= 0) ps = 100;
      state.pageSize = ps;
      state.page = 1;
      applyFilters();
    });

    els.btnFirst.addEventListener("click", function() {
      if (state.page !== 1) {
        state.page = 1;
        renderList();
      }
    });
    els.btnPrev.addEventListener("click", function() {
      if (state.page > 1) {
        state.page -= 1;
        renderList();
      }
    });
    els.btnNext.addEventListener("click", function() {
      var maxPage = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
      if (state.page < maxPage) {
        state.page += 1;
        renderList();
      }
    });
    els.btnLast.addEventListener("click", function() {
      var maxPage = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
      if (state.page !== maxPage) {
        state.page = maxPage;
        renderList();
      }
    });
  }

  function readJsonFile(file, cb) {
    var reader = new FileReader();
    reader.onload = function(ev) {
      try {
        var obj = JSON.parse(ev.target.result);
        cb(obj);
      } catch (e) {
        console.error("Failed to parse JSON file", e);
      }
    };
    reader.readAsText(file, "utf-8");
  }

  function normalizeMoleculeIndex(mols) {
    var out = [];
    for (var i = 0; i < mols.length; i++) {
      var m = mols[i] || {};
      var id = m.id != null ? m.id : (i + 1);
      out.push({
        id: id,
        formula: String(m.formula || ""),
        atoms: safeInt(m.atoms, 0),
        unique_elements: safeInt(m.unique_elements, 0),
        mass_estimate: safeFloat(m.mass_estimate, 0),
        stability_index: safeFloat(m.stability_index, 0)
      });
    }
    return out;
  }

  function normalizeMetamaterialIndex(metas) {
    var out = [];
    for (var i = 0; i < metas.length; i++) {
      var m = metas[i] || {};
      var id = m.id != null ? m.id : (i + 1);
      out.push({
        id: id,
        pattern: String(m.pattern || ""),
        unit_atoms: safeInt(m.unit_atoms, 0),
        element_count: Array.isArray(m.elements) ? m.elements.length : safeInt(m.element_count, 0),
        mass_estimate: safeFloat(m.mass_estimate, 0),
        stability_index: safeFloat(m.stability_index, 0)
      });
    }
    return out;
  }

  function safeFloat(x, d) {
    var v = Number(x);
    if (!isFinite(v)) return d;
    return v;
  }

  function safeInt(x, d) {
    var v = parseInt(x, 10);
    if (!isFinite(v)) return d;
    return v;
  }

  function refreshAll() {
    applyFilters();
    updateSummary();
  }

  function getActiveIndexCollection() {
    return state.type === "metamaterial" ? state.metamaterialsIndex : state.moleculesIndex;
  }

  function applyFilters() {
    var coll = getActiveIndexCollection();
    var formulaQ = (els.filterFormula.value || "").trim().toLowerCase();
    var elemQ = (els.filterElement.value || "").trim();
    elemQ = elemQ.length ? elemQ.toLowerCase() : "";
    var minAtoms = safeInt(els.minAtoms.value, 1);
    var maxAtoms = safeInt(els.maxAtoms.value, 9999999);
    var minStab = safeFloat(els.stabilityRange.value, 0.0);

    if (minAtoms < 1) minAtoms = 1;
    if (maxAtoms < minAtoms) maxAtoms = minAtoms;

    var res = [];
    for (var i = 0; i < coll.length; i++) {
      var item = coll[i];
      var atoms = state.type === "metamaterial" ? item.unit_atoms : item.atoms;
      if (atoms < minAtoms || atoms > maxAtoms) continue;
      var stab = safeFloat(item.stability_index, 0.0);
      if (stab < minStab) continue;

      if (formulaQ.length) {
        var label = state.type === "metamaterial" ? (item.pattern || "") : (item.formula || "");
        if (label.toLowerCase().indexOf(formulaQ) === -1) continue;
      }

      if (elemQ.length) {
        res.push(item);
      } else {
        res.push(item);
      }
    }

    var sortKey = els.sortSelect.value || "id";
    res.sort(function(a, b) {
      if (sortKey === "id") {
        return safeInt(a.id, 0) - safeInt(b.id, 0);
      }
      if (sortKey === "mass") {
        return safeFloat(a.mass_estimate, 0) - safeFloat(b.mass_estimate, 0);
      }
      if (sortKey === "stability") {
        return safeFloat(b.stability_index, 0) - safeFloat(a.stability_index, 0);
      }
      if (sortKey === "atoms") {
        var aa = state.type === "metamaterial" ? a.unit_atoms : a.atoms;
        var ab = state.type === "metamaterial" ? b.unit_atoms : b.atoms;
        return aa - ab;
      }
      return 0;
    });

    state.filtered = res;
    if (state.page < 1) state.page = 1;
    var maxPage = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
    if (state.page > maxPage) state.page = maxPage;
    renderList();
    updateSummary();
    if (state.selectedId != null) {
      var found = null;
      for (var i2 = 0; i2 < state.filtered.length; i2++) {
        if (state.filtered[i2].id === state.selectedId) {
          found = state.filtered[i2];
          break;
        }
      }
      if (!found) {
        state.selectedId = null;
        clearSelection();
      }
    } else {
      clearSelection();
    }
  }

  function updateSummary() {
    var coll = getActiveIndexCollection();
    var total = coll.length;
    var shown = state.filtered.length;
    var label = state.type === "metamaterial" ? "metamaterials" : "molecules";
    els.summaryText.textContent = "Loaded " + total + " " + label + ", " + shown + " match filters.";
    els.mainHeaderCounter.textContent = shown + " " + label;
    els.listHeaderMeta.textContent = shown + " shown";
  }

  function renderList() {
    var tbody = els.listTbody;
    while (tbody.firstChild) tbody.removeChild(tbody.firstChild);

    var start = (state.page - 1) * state.pageSize;
    var end = Math.min(state.filtered.length, start + state.pageSize);

    for (var i = start; i < end; i++) {
      var item = state.filtered[i];
      var tr = document.createElement("tr");
      tr.className = "list-row";
      tr.dataset.id = String(item.id);

      if (state.selectedId != null && item.id === state.selectedId) {
        tr.classList.add("selected");
      }

      var label = state.type === "metamaterial" ? (item.pattern || "") : (item.formula || "");
      var atoms = state.type === "metamaterial" ? item.unit_atoms : item.atoms;

      var tdId = document.createElement("td");
      tdId.textContent = String(item.id);
      tr.appendChild(tdId);

      var tdFormula = document.createElement("td");
      tdFormula.textContent = label || "-";
      tr.appendChild(tdFormula);

      var tdAtoms = document.createElement("td");
      tdAtoms.textContent = String(atoms);
      tr.appendChild(tdAtoms);

      var tdMass = document.createElement("td");
      tdMass.textContent = safeFloat(item.mass_estimate, 0).toFixed(2);
      tr.appendChild(tdMass);

      var tdStab = document.createElement("td");
      tdStab.textContent = safeFloat(item.stability_index, 0).toFixed(3);
      tr.appendChild(tdStab);

      tr.addEventListener("click", onRowClick);
      tbody.appendChild(tr);
    }

    var maxPage = Math.max(1, Math.ceil(state.filtered.length / state.pageSize));
    els.paginationInfo.textContent = "Page " + state.page + " / " + maxPage;
  }

  function onRowClick(e) {
    var tr = e.currentTarget;
    var id = parseInt(tr.dataset.id, 10);
    if (!isFinite(id)) return;
    state.selectedId = id;
    loadDetailAndRender(id);
    var rows = els.listTbody.querySelectorAll(".list-row");
    for (var r = 0; r < rows.length; r++) {
      rows[r].classList.remove("selected");
      if (parseInt(rows[r].dataset.id, 10) === id) {
        rows[r].classList.add("selected");
      }
    }
  }

  function clearSelection() {
    els.statsTitle.textContent = "No selection";
    var ctx = els.vizCanvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, els.vizCanvas.width, els.vizCanvas.height);
    els.vizTag.textContent = "";
    els.vizTag.className = "badge";
    while (els.statsBody.firstChild) els.statsBody.removeChild(els.statsBody.firstChild);
    var div = document.createElement("div");
    div.className = "stats-section-title";
    div.textContent = "Instructions";
    els.statsBody.appendChild(div);
    var p = document.createElement("div");
    p.style.fontSize = "12px";
    p.style.color = "#d8dee9";
    p.textContent = "Select a molecule or metamaterial from the list to view its geometry and metrics.";
    els.statsBody.appendChild(p);
  }

  function loadDetailAndRender(id) {
    var isMeta = state.type === "metamaterial";
    var cache = isMeta ? state.metamaterialsDetailCache : state.moleculesDetailCache;
    if (cache[id]) {
      renderSelection(cache[id], isMeta);
      return;
    }
    if (state.loadingDetailId === id) return;
    state.loadingDetailId = id;
    var ctx = els.vizCanvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, els.vizCanvas.width, els.vizCanvas.height);
    ctx.fillStyle = "#d8dee9";
    ctx.font = "12px system-ui";
    ctx.fillText("Loading detail for ID " + id + "...", 20, 30);
    while (els.statsBody.firstChild) els.statsBody.removeChild(els.statsBody.firstChild);
    var div = document.createElement("div");
    div.className = "stats-section-title";
    div.textContent = "Loading";
    els.statsBody.appendChild(div);
    var p = document.createElement("div");
    p.style.fontSize = "12px";
    p.style.color = "#d8dee9";
    p.textContent = "Fetching detailed geometry and metrics from server.";
    els.statsBody.appendChild(p);
    var url = isMeta ? ("/api/metamaterials/" + id) : ("/api/molecules/" + id);
    fetch(url).then(function(res) {
      if (!res.ok) throw new Error("HTTP " + res.status);
      return res.json();
    }).then(function(obj) {
      state.loadingDetailId = null;
      cache[id] = obj;
      renderSelection(obj, isMeta);
    }).catch(function(err) {
      state.loadingDetailId = null;
      console.error("Failed to load detail", err);
      clearSelection();
      var ctx2 = els.vizCanvas.getContext("2d");
      ctx2.fillStyle = "#000000";
      ctx2.fillRect(0, 0, els.vizCanvas.width, els.vizCanvas.height);
      ctx2.fillStyle = "#bf616a";
      ctx2.font = "12px system-ui";
      ctx2.fillText("Failed to load detail for ID " + id, 20, 30);
    });
  }

  function renderSelection(item, isMeta) {
    els.statsTitle.textContent = (isMeta ? "Metamaterial " : "Molecule ") + (isMeta ? (item.pattern || ("#" + item.id)) : (item.formula || ("#" + item.id)));
    els.vizTag.textContent = isMeta ? "Metamaterial" : "Molecule";
    els.vizTag.className = "badge " + (isMeta ? "badge-metamaterial" : "badge-molecule");
    renderGeometry(item.geometry, isMeta);
    renderStats(item, isMeta);
  }

  function renderGeometry(geom, isMeta) {
    var canvas = els.vizCanvas;
    var ctx = canvas.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (!geom || !Array.isArray(geom.atom_positions) || geom.atom_positions.length === 0) {
      ctx.fillStyle = "#d8dee9";
      ctx.font = "12px system-ui";
      ctx.fillText("No geometry available", 20, 30);
      return;
    }
    var pts = geom.atom_positions;
    var bonds = Array.isArray(geom.bonds) ? geom.bonds : [];
    var xs = [];
    var ys = [];
    var zs = [];
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i] || [0, 0, 0];
      xs.push(safeFloat(p[0], 0));
      ys.push(safeFloat(p[1], 0));
      zs.push(safeFloat(p[2], 0));
    }
    var minX = Math.min.apply(null, xs);
    var maxX = Math.max.apply(null, xs);
    var minY = Math.min.apply(null, ys);
    var maxY = Math.max.apply(null, ys);
    var minZ = Math.min.apply(null, zs);
    var maxZ = Math.max.apply(null, zs);
    var dx = maxX - minX;
    var dy = maxY - minY;
    var pad = 40;
    var w = canvas.width - 2 * pad;
    var h = canvas.height - 2 * pad;
    var sx = dx > 0 ? w / dx : 1;
    var sy = dy > 0 ? h / dy : 1;
    var s = Math.min(sx, sy);
    if (!isFinite(s) || s <= 0) s = 1;
    function project(p) {
      var x = safeFloat(p[0], 0);
      var y = safeFloat(p[1], 0);
      var z = safeFloat(p[2], 0);
      var px = pad + (x - minX) * s;
      var py = pad + h - (y - minY) * s;
      var zn = maxZ > minZ ? (z - minZ) / (maxZ - minZ) : 0.5;
      var r = 4 + 4 * zn;
      return { x: px, y: py, r: r, z: zn };
    }
    ctx.lineWidth = 1;
    for (var bi = 0; bi < bonds.length; bi++) {
      var b = bonds[bi];
      if (!Array.isArray(b) || b.length < 2) continue;
      var i0 = safeInt(b[0], -1);
      var i1 = safeInt(b[1], -1);
      if (i0 < 0 || i0 >= pts.length || i1 < 0 || i1 >= pts.length) continue;
      var p0 = project(pts[i0]);
      var p1 = project(pts[i1]);
      var zMean = (p0.z + p1.z) / 2;
      var shade = Math.round(180 + 75 * zMean);
      var c = "rgb(" + shade + "," + shade + "," + shade + ")";
      ctx.strokeStyle = c;
      ctx.beginPath();
      ctx.moveTo(p0.x, p0.y);
      ctx.lineTo(p1.x, p1.y);
      ctx.stroke();
    }
    for (var ai = 0; ai < pts.length; ai++) {
      var q = project(pts[ai]);
      var ccol = atomColor(ai);
      ctx.fillStyle = ccol;
      ctx.beginPath();
      ctx.arc(q.x, q.y, q.r, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 0.5;
      ctx.stroke();
    }
  }

  function atomColor(idx) {
    var base = idx * 2654435761;
    var r = (base & 0xff);
    var g = ((base >> 8) & 0xff);
    var b = ((base >> 16) & 0xff);
    var min = 80;
    if (r < min) r = min;
    if (g < min) g = min;
    if (b < min) b = min;
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function renderStats(item, isMeta) {
    while (els.statsBody.firstChild) els.statsBody.removeChild(els.statsBody.firstChild);

    var section1 = document.createElement("div");
    section1.className = "stats-section-title";
    section1.textContent = isMeta ? "Metamaterial" : "Molecule";
    els.statsBody.appendChild(section1);

    var table1 = document.createElement("table");
    table1.className = "stats-table";
    addRow(table1, "ID", String(item.id));
    addRow(table1, isMeta ? "Pattern" : "Formula", isMeta ? (item.pattern || "-") : (item.formula || "-"));
    addRow(table1, "Atoms / Unit", String(isMeta ? item.unit_atoms : item.atoms));
    addRow(table1, "Distinct elements", String(isMeta ? (Array.isArray(item.elements) ? item.elements.length : item.element_count || 0) : item.unique_elements || 0));
    addRow(table1, "Mass estimate", safeFloat(item.mass_estimate, 0).toFixed(4));
    addRow(table1, "Stability index", safeFloat(item.stability_index, 0).toFixed(6));
    els.statsBody.appendChild(table1);

    var section2 = document.createElement("div");
    section2.className = "stats-section-title";
    section2.textContent = "Harmonic profile";
    els.statsBody.appendChild(section2);

    var hp = item.harmonic_profile || {};
    var table2 = document.createElement("table");
    table2.className = "stats-table";
    if (isMeta) {
      addRow(table2, "Mean harmonic", safeFloat(hp.mean_harmonic, 0).toFixed(3));
      addRow(table2, "Mean dim_mix", safeFloat(hp.mean_dim_mix, 0).toFixed(3));
      addRow(table2, "Harmonic contrast", safeFloat(hp.harmonic_contrast, 0).toFixed(3));
      addRow(table2, "Dim_mix contrast", safeFloat(hp.dim_mix_contrast, 0).toFixed(3));
    } else {
      addRow(table2, "Effective harmonic", safeFloat(hp.effective_harmonic, 0).toFixed(3));
      addRow(table2, "Effective dim_mix", safeFloat(hp.effective_dim_mix, 0).toFixed(3));
      addRow(table2, "Stable isotope fraction", safeFloat(hp.avg_stable_iso_fraction, 0).toFixed(3));
    }
    els.statsBody.appendChild(table2);

    var section3 = document.createElement("div");
    section3.className = "stats-section-title";
    section3.textContent = "Composition";
    els.statsBody.appendChild(section3);

    var table3 = document.createElement("table");
    table3.className = "stats-table";
    if (isMeta) {
      var elArr = Array.isArray(item.elements) ? item.elements : [];
      for (var i = 0; i < elArr.length; i++) {
        var e = elArr[i] || {};
        var label = (e.symbol || "?") + " (" + (e.name || "") + ")";
        addRow(table3, label, "count=" + String(e.count != null ? e.count : "?"));
      }
    } else {
      var comp = Array.isArray(item.composition) ? item.composition : [];
      for (var j = 0; j < comp.length; j++) {
        var c = comp[j] || {};
        var label2 = (c.symbol || "?") + " (" + (c.name || "") + ")";
        addRow(table3, label2, "count=" + String(c.count != null ? c.count : "?"));
      }
    }
    els.statsBody.appendChild(table3);

    if (!isMeta) {
      var section4 = document.createElement("div");
      section4.className = "stats-section-title";
      section4.textContent = "Valence model";
      els.statsBody.appendChild(section4);

      var table4 = document.createElement("table");
      table4.className = "stats-table";
      var vm = item.valence_model || {};
      var caps = Array.isArray(vm.atom_capacities) ? vm.atom_capacities : [];
      addRow(table4, "Atoms in model", String(caps.length));
      addRow(table4, "Total capacity", String(caps.reduce(function(a, b) { return a + safeInt(b, 0); }, 0)));
      addRow(table4, "Construction valid", String(vm.construction_valid === false ? false : true));
      els.statsBody.appendChild(table4);
    }

    var section5 = document.createElement("div");
    section5.className = "stats-section-title";
    section5.textContent = "Geometry summary";
    els.statsBody.appendChild(section5);

    var table5 = document.createElement("table");
    table5.className = "stats-table";
    var g = item.geometry || {};
    var bbox = g.bounding_box || {};
    var bbMin = bbox.min || [0, 0, 0];
    var bbMax = bbox.max || [0, 0, 0];
    addRow(table5, "Primitive", String(g.primitive || "unknown"));
    addRow(table5, "Dimension", String(g.dimension != null ? g.dimension : 3));
    addRow(table5, "Layers", String(g.layers != null ? g.layers : "-"));
    addRow(table5, "Base vertices", String(g.base_vertices != null ? g.base_vertices : "-"));
    addRow(table5, "BBox min", "[" + bbMin.map(function(x) { return safeFloat(x, 0).toFixed(2); }).join(", ") + "]");
    addRow(table5, "BBox max", "[" + bbMax.map(function(x) { return safeFloat(x, 0).toFixed(2); }).join(", ") + "]");
    els.statsBody.appendChild(table5);
  }

  function addRow(table, key, val) {
    var tr = document.createElement("tr");
    var td1 = document.createElement("td");
    var td2 = document.createElement("td");
    td1.textContent = key;
    td2.textContent = val;
    tr.appendChild(td1);
    tr.appendChild(td2);
    table.appendChild(tr);
  }

  function autoFetchDefault() {
    fetchIfExists("/api/molecules/index", function(obj) {
      var items = Array.isArray(obj.items) ? obj.items : [];
      state.moleculesIndex = items;
      refreshAll();
    });
    fetchIfExists("/api/metamaterials/index", function(obj) {
      var items = Array.isArray(obj.items) ? obj.items : [];
      state.metamaterialsIndex = items;
      refreshAll();
    });
  }

  function fetchIfExists(path, cb) {
    if (!window.fetch) return;
    fetch(path).then(function(res) {
      if (!res.ok) return null;
      return res.text();
    }).then(function(txt) {
      if (!txt) return;
      try {
        var obj = JSON.parse(txt);
        cb(obj);
      } catch (e) {
        console.error("Failed to parse", path, e);
      }
    }).catch(function() {});
  }

  window.addEventListener("load", function() {
    initRefs();
    attachEvents();
    els.stabilityRangeValue.textContent = parseFloat(els.stabilityRange.value).toFixed(2);
    autoFetchDefault();
  });
})();
