/* ============================================================
   MolParam — script.js
   Con visualizador 3D usando 3Dmol.js
   ============================================================ */

const fileInput      = document.getElementById("fileInput");
const dropzone       = document.getElementById("dropzone");
const fileInfo       = document.getElementById("fileInfo");
const btnPredict     = document.getElementById("btnPredict");
const loader         = document.getElementById("loader");
const errorBox       = document.getElementById("errorBox");
const resultsSection = document.getElementById("resultsSection");

let currentFile    = null;
let currentData    = null;
let currentMol2    = null;   // contenido raw del .mol2 para el visualizador
let viewer         = null;
let spinActive     = false;
let currentStyle   = "stick";

// Colores por atomtype para el visualizador
const ATOMTYPE_COLORS = {
  "C":    "#a8d8a8",  // verde suave
  "HC":   "#c8e6c9",  // verde muy claro
  "H":    "#ffffff",  // blanco
  "HS14": "#e8f5e9",  // verde casi blanco
  "O":    "#ef9a9a",  // rojo suave
  "OA":   "#e53935",  // rojo
  "OE":   "#ff7043",  // naranja-rojo
  "OM":   "#f44336",  // rojo intenso
  "N":    "#90caf9",  // azul suave
  "NT":   "#42a5f5",  // azul
  "CH1":  "#ffcc80",  // naranja suave
  "CH2":  "#ffa726",  // naranja
  "CH3":  "#fb8c00",  // naranja intenso
  "F":    "#80cbc4",  // teal
  "CL":   "#a5d6a7",  // verde
  "S":    "#fff176",  // amarillo
  "default": "#90a4ae"  // gris azulado
};

// ── Drag & drop ──────────────────────────────────────────────
dropzone.addEventListener("dragover", (e) => {
  e.preventDefault();
  dropzone.classList.add("dragover");
});

dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

dropzone.addEventListener("drop", (e) => {
  e.preventDefault();
  dropzone.classList.remove("dragover");
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

dropzone.addEventListener("click", (e) => {
  if (e.target !== fileInput) fileInput.click();
});

fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (!file.name.endsWith(".mol2")) {
    showError("El archivo debe tener extensión .mol2");
    return;
  }
  currentFile = file;
  fileInfo.style.display = "block";
  fileInfo.textContent   = `✔ ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
  btnPredict.disabled    = false;
  hideResults();
  hideError();

  // Leer el mol2 para el visualizador
  const reader = new FileReader();
  reader.onload = (e) => { currentMol2 = e.target.result; };
  reader.readAsText(file);
}

// ── Predict ──────────────────────────────────────────────────
btnPredict.addEventListener("click", async () => {
  if (!currentFile) return;

  showLoader();
  hideResults();
  hideError();

  const formData = new FormData();
  formData.append("mol2file", currentFile);

  try {
    const res  = await fetch("/predict", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      showError(data.error);
    } else {
      currentData = data;
      renderResults(data);
    }
  } catch (err) {
    showError("Error de conexión con el servidor: " + err.message);
  } finally {
    hideLoader();
  }
});

// ── Render ───────────────────────────────────────────────────
function renderResults(data) {
  document.getElementById("molName").textContent = data.molecule;
  renderTable("AA", data.AA);
  renderTable("UA", data.UA);
  resultsSection.style.display = "block";

  // Inicializar visualizador después de que el DOM esté visible
  setTimeout(() => initViewer(data), 100);
}

function renderTable(scheme, rows) {
  const tbody = document.querySelector(`#table${scheme} tbody`);
  const stats = document.getElementById(`stats${scheme}`);
  tbody.innerHTML = "";

  if (!rows || rows.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" style="text-align:center;color:var(--text-muted);padding:2rem">Sin átomos para este esquema</td></tr>`;
    stats.innerHTML = "";
    return;
  }

  const nAtoms    = rows.length;
  const types     = [...new Set(rows.map(r => r.atomtype_predicho))].length;
  const charges   = rows.map(r => r.charge_predicha);
  const avgCharge = (charges.reduce((a, b) => a + b, 0) / charges.length).toFixed(3);
  const minC      = Math.min(...charges).toFixed(3);
  const maxC      = Math.max(...charges).toFixed(3);

  stats.innerHTML = `
    <div class="stat-chip">Átomos: <span>${nAtoms}</span></div>
    <div class="stat-chip">Clases atomtype: <span>${types}</span></div>
    <div class="stat-chip">Charge media: <span>${avgCharge} e</span></div>
    <div class="stat-chip">Rango: <span>[${minC}, ${maxC}] e</span></div>
  `;

  rows.forEach(row => {
    const charge      = parseFloat(row.charge_predicha);
    const chargeClass = charge > 0.05 ? "charge-pos" : charge < -0.05 ? "charge-neg" : "charge-neu";
    const tr          = document.createElement("tr");
    tr.innerHTML = `
      <td>${row.molecule}</td>
      <td>${row.atom_id}</td>
      <td>${row.atom_name}</td>
      <td>${row.element}</td>
      <td class="atomtype">${row.atomtype_predicho}</td>
      <td class="${chargeClass}">${charge >= 0 ? "+" : ""}${charge.toFixed(4)}</td>
    `;
    tbody.appendChild(tr);
  });
}

// ── Visualizador 3D ──────────────────────────────────────────
function initViewer(data) {
  if (!currentMol2 || !window.$3Dmol) return;

  const container = document.getElementById("mol3d-container");
  container.innerHTML = "";

  // Crear viewer con fondo oscuro
  viewer = $3Dmol.createViewer(container, {
    backgroundColor: "#050810",
    antialias: true,
  });

  // Cargar la molécula desde el mol2 raw
  viewer.addModel(currentMol2, "mol2");

  // Construir mapa atom_id → atomtype desde los resultados AA
  const atomtypeMap = {};
  if (data.AA) {
    data.AA.forEach(row => {
      atomtypeMap[row.atom_id] = row.atomtype_predicho;
    });
  }

  // Aplicar estilo inicial coloreado por atomtype
  applyAtomtypeColors(atomtypeMap);

  viewer.zoomTo();
  viewer.render();

  // Esperar a que el render termine antes de registrar clicks
setTimeout(() => {
  viewer.setClickable({}, true, (atom) => {
    const atomId   = atom.serial;
    const atomtype = atomtypeMap[atomId] || "?";
    const aaRow    = data.AA ? data.AA.find(r => r.atom_id === atomId) : null;
    const charge   = aaRow ? aaRow.charge_predicha.toFixed(4) : "?";
    const tooltip  = document.getElementById("atomTooltip");
    tooltip.style.display = "block";
    tooltip.innerHTML = `
      <strong>${atom.atom || atom.resn || "?"}</strong> &nbsp;|&nbsp;
      atomtype: <span style="color:var(--accent)">${atomtype}</span> &nbsp;|&nbsp;
      charge: <span style="color:${parseFloat(charge) > 0 ? '#34d399' : '#f87171'}">${parseFloat(charge) >= 0 ? '+' : ''}${charge} e</span>
    `;
  });
  viewer.render();
}, 500);

  // Leyenda
  buildLegend(atomtypeMap);
}

function applyAtomtypeColors(atomtypeMap) {
  if (!viewer) return;

  viewer.setStyle({}, {});

  Object.entries(atomtypeMap).forEach(([atomId, atomtype]) => {
    const color = ATOMTYPE_COLORS[atomtype] || ATOMTYPE_COLORS["default"];
    const id    = parseInt(atomId);

    if (currentStyle === "sphere") {
      viewer.setStyle({ serial: id }, {
        sphere: { color: color, radius: 0.4 }
      });
    } else if (currentStyle === "surface") {
      viewer.setStyle({ serial: id }, {
        stick: { color: color, radius: 0.12, hidden: true },
        sphere: { color: color, radius: 0.01 }
      });
    } else {
      // stick — esferas pequeñas invisibles para capturar clicks
      viewer.setStyle({ serial: id }, {
        stick:  { color: color, radius: 0.12 },
        sphere: { color: color, radius: 0.22 }
      });
    }
  });

  if (currentStyle === "surface") {
    viewer.addSurface($3Dmol.SurfaceType.VDW, {
      opacity: 0.7,
      colorscheme: { prop: "serial", map: Object.fromEntries(
        Object.entries(atomtypeMap).map(([id, at]) => [
          id, ATOMTYPE_COLORS[at] || ATOMTYPE_COLORS["default"]
        ])
      )}
    });
  }

  viewer.render();
}

function setStyle(style) {
  currentStyle = style;
  document.querySelectorAll(".btn-style").forEach(b => {
    b.classList.toggle("active", b.dataset.style === style);
  });

  if (!viewer || !currentData) return;

  const atomtypeMap = {};
  if (currentData.AA) {
    currentData.AA.forEach(row => { atomtypeMap[row.atom_id] = row.atomtype_predicho; });
  }
  applyAtomtypeColors(atomtypeMap);
}

function resetView() {
  if (!viewer) return;
  viewer.zoomTo();
  viewer.render();
}

function toggleSpin() {
  if (!viewer) return;
  spinActive = !spinActive;
  if (spinActive) {
    viewer.spin("y", 1);
  } else {
    viewer.spin(false);
  }
}

function buildLegend(atomtypeMap) {
  const legend  = document.getElementById("viewerLegend");
  const present = [...new Set(Object.values(atomtypeMap))].sort();
  legend.innerHTML = present.map(at => {
    const color = ATOMTYPE_COLORS[at] || ATOMTYPE_COLORS["default"];
    return `<div class="legend-item">
      <div class="legend-dot" style="background:${color}"></div>
      <span>${at}</span>
    </div>`;
  }).join("");
}

// ── Tabs ─────────────────────────────────────────────────────
document.querySelectorAll(".tab").forEach(tab => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(c => c.classList.remove("active"));
    tab.classList.add("active");
    document.getElementById(`tab-${tab.dataset.tab}`).classList.add("active");
  });
});

// ── Download CSV ─────────────────────────────────────────────
document.getElementById("downloadAA").addEventListener("click", () => {
  if (currentData) downloadCSV(currentData.AA, `${currentData.molecule}_AA.csv`);
});

document.getElementById("downloadUA").addEventListener("click", () => {
  if (currentData) downloadCSV(currentData.UA, `${currentData.molecule}_UA.csv`);
});

function downloadCSV(rows, filename) {
  if (!rows || rows.length === 0) return;
  const headers = Object.keys(rows[0]).join(",");
  const lines   = rows.map(r => Object.values(r).join(","));
  const csv     = [headers, ...lines].join("\n");
  const blob    = new Blob([csv], { type: "text/csv" });
  const url     = URL.createObjectURL(blob);
  const a       = document.createElement("a");
  a.href        = url;
  a.download    = filename;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Helpers ──────────────────────────────────────────────────
function showLoader()  { loader.style.display = "block"; }
function hideLoader()  { loader.style.display = "none";  }
function hideResults() { resultsSection.style.display = "none"; }
function hideError()   { errorBox.style.display = "none"; }

function showError(msg) {
  errorBox.style.display = "block";
  errorBox.textContent   = "⚠ " + msg;
}
