// server.js
"use strict";

const express = require("express");
const path = require("path");
const fs = require("fs");

const app = express();
const PORT = process.env.PORT || 3000;
const ROOT_DIR = __dirname;

// ============= helpers =============
function safeFloat(x, d) {
  const v = Number(x);
  return Number.isFinite(v) ? v : d;
}

function safeInt(x, d) {
  const v = parseInt(x, 10);
  return Number.isFinite(v) ? v : d;
}

function loadJsonSafe(p) {
  try {
    const txt = fs.readFileSync(p, "utf8");
    return JSON.parse(txt);
  } catch (_) {
    return null;
  }
}

// ============= data stores =============
function buildMoleculeStore() {
  const p = path.join(ROOT_DIR, "molecules_geom.json");
  const raw = loadJsonSafe(p);
  if (!raw) {
    return { index: [], byId: {} };
  }

  const arr = Array.isArray(raw.molecules)
    ? raw.molecules
    : Array.isArray(raw)
    ? raw
    : [];

  const index = [];
  const byId = {};

  for (let i = 0; i < arr.length; i++) {
    const m = arr[i] || {};
    const id = m.id != null ? m.id : i + 1;

    index.push({
      id,
      formula: String(m.formula || ""),
      atoms: safeInt(m.atoms, 0),
      unique_elements: safeInt(m.unique_elements, 0),
      mass_estimate: safeFloat(m.mass_estimate, 0),
      stability_index: safeFloat(m.stability_index, 0)
    });

    byId[id] = {
      id,
      formula: String(m.formula || ""),
      atoms: safeInt(m.atoms, 0),
      unique_elements: safeInt(m.unique_elements, 0),
      mass_estimate: safeFloat(m.mass_estimate, 0),
      stability_index: safeFloat(m.stability_index, 0),
      composition: Array.isArray(m.composition) ? m.composition : [],
      harmonic_profile: m.harmonic_profile || {},
      geometry: m.geometry || {},
      valence_model: m.valence_model || {}
    };
  }

  return { index, byId };
}

function buildMetamaterialStore() {
  const p = path.join(ROOT_DIR, "metamaterials_geom.json");
  const raw = loadJsonSafe(p);
  if (!raw) {
    return { index: [], byId: {} };
  }

  const arr = Array.isArray(raw.metamaterials)
    ? raw.metamaterials
    : Array.isArray(raw)
    ? raw
    : [];

  const index = [];
  const byId = {};

  for (let i = 0; i < arr.length; i++) {
    const m = arr[i] || {};
    const id = m.id != null ? m.id : i + 1;
    const elements = Array.isArray(m.elements) ? m.elements : [];

    index.push({
      id,
      pattern: String(m.pattern || ""),
      unit_atoms: safeInt(m.unit_atoms, 0),
      element_count: elements.length,
      mass_estimate: safeFloat(m.mass_estimate, 0),
      stability_index: safeFloat(m.stability_index, 0)
    });

    byId[id] = {
      id,
      pattern: String(m.pattern || ""),
      unit_atoms: safeInt(m.unit_atoms, 0),
      elements,
      mass_estimate: safeFloat(m.mass_estimate, 0),
      stability_index: safeFloat(m.stability_index, 0),
      harmonic_profile: m.harmonic_profile || {},
      geometry: m.geometry || {}
    };
  }

  return { index, byId };
}

const moleculesStore = buildMoleculeStore();
const metamaterialsStore = buildMetamaterialStore();

// ============= middleware =============
app.use(express.static(ROOT_DIR));

// ============= routes =============
app.get("/", (req, res) => {
  res.sendFile(path.join(ROOT_DIR, "molecules.html"));
});

app.get("/api/molecules/index", (req, res) => {
  res.json({ items: moleculesStore.index });
});

app.get("/api/molecules/:id", (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "Invalid ID" });
    return;
  }
  const item = moleculesStore.byId[id];
  if (!item) {
    res.status(404).json({ error: "Not found" });
    return;
  }
  res.json(item);
});

app.get("/api/metamaterials/index", (req, res) => {
  res.json({ items: metamaterialsStore.index });
});

app.get("/api/metamaterials/:id", (req, res) => {
  const id = parseInt(req.params.id, 10);
  if (!Number.isFinite(id)) {
    res.status(400).json({ error: "Invalid ID" });
    return;
  }
  const item = metamaterialsStore.byId[id];
  if (!item) {
    res.status(404).json({ error: "Not found" });
    return;
  }
  res.json(item);
});

// ============= fallback 404 =============
app.use((req, res) => {
  res.status(404).json({ error: "Not found" });
});

// ============= start =============
app.listen(PORT, () => {
  process.stdout.write(`Molecule explorer running at http://localhost:${PORT}\n`);
});
