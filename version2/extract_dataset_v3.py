"""
extract_dataset_v3.py
---------------------
Lee archivos .mol2 e .itp del ATB y genera un CSV con todas las features
por átomo + atomtype y charge como targets.

Solución al problema de RDKit con mol2 del ATB:
  - El ATB usa atom_name (H1, C3, etc.) como identificador en el mol2,
    no el símbolo del elemento. RDKit no puede leerlos directamente.
  - Este script parsea el mol2 manualmente y construye la molécula
    RDKit desde cero (RWMol) para acceder a anillos, aromaticidad y
    cargas de Gasteiger.

Match mol2 ↔ itp:
  - Siempre por índice posicional (confirmado por la tutora:
    el orden de átomos en mol2 e itp es siempre el mismo).

Uso:


Dependencias:
    pip install rdkit pandas numpy
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem import rdPartialCharges, AllChem
except ImportError:
    sys.exit("RDKit no encontrado. Instalalo con: pip install rdkit")


# =============================================================================
# CONFIG
# =============================================================================

ATOMIC_MASSES = {
    "H": 1.008,  "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06,  "Cl": 35.45,
    "Br": 79.904,"I": 126.904,"B": 10.811, "Si": 28.085,
    "Se": 78.971,"Fe": 55.845,"Zn": 65.38, "Ca": 40.078,
    "Mg": 24.305,"Na": 22.990,"K":  39.098,
}

PLANAR_TYPES    = {"C.2", "C.ar", "N.ar", "N.2", "N.pl3", "O.2", "S.2", "N.am"}
ELECTRONEGATIVE = {"N", "O", "F", "Cl", "S", "Br", "I"}

# Normalización de elementos no estándar del ATB → símbolo estándar para RDKit
ELEMENT_NORMALIZE = {
    "HC": "H",   # H unido a C en mol2 united-atom
    "HO": "H",   # H unido a O
    "HN": "H",   # H unido a N
    "CL": "Cl",  # Cloro en mayúsculas
    "BR": "Br",  # Bromo en mayúsculas
    "SI": "Si",
    "FE": "Fe",
    "ZN": "Zn",
    "CA": "Ca",
    "MG": "Mg",
    "NA": "Na",
}

# Mapeo de bond type del mol2 → RDKit BondType
BOND_TYPE_MAP = {
    "1":   Chem.BondType.SINGLE,
    "2":   Chem.BondType.DOUBLE,
    "3":   Chem.BondType.TRIPLE,
    "ar":  Chem.BondType.AROMATIC,
    "am":  Chem.BondType.SINGLE,   # amide → single para RDKit
    "du":  Chem.BondType.SINGLE,   # dummy
    "un":  Chem.BondType.UNSPECIFIED,
    "nc":  Chem.BondType.UNSPECIFIED,
}


# =============================================================================
# PARSERS
# =============================================================================

def parse_mol2(mol2_path: Path):
    """
    Parsea @<TRIPOS>ATOM y @<TRIPOS>BOND del mol2.

    Devuelve:
        atoms: {atom_id (1-based): {atom_name, tripos_type, element, xyz}}
        bonds: [(a1, a2, bond_type_str)]
    """
    atoms = {}
    bonds = []
    section = None

    with open(mol2_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith("@<TRIPOS>ATOM"):
                section = "atom"
                continue
            if line.startswith("@<TRIPOS>BOND"):
                section = "bond"
                continue
            if line.startswith("@<TRIPOS>"):
                section = None
                continue

            if section == "atom" and line.strip():
                parts = line.split()
                if len(parts) < 6:
                    continue
                atom_id     = int(parts[0])
                atom_name   = parts[1]
                x, y, z     = float(parts[2]), float(parts[3]), float(parts[4])
                tripos_type = parts[5]
                # Elemento: parte antes del punto en tripos_type, normalizado
                elem_raw = tripos_type.split(".")[0]
                elem_raw = ''.join(c for c in elem_raw if c.isalpha())
                element  = ELEMENT_NORMALIZE.get(elem_raw.upper(),
                               elem_raw.capitalize() if len(elem_raw) > 1 else elem_raw)
                atoms[atom_id] = {
                    "atom_name":   atom_name,
                    "tripos_type": tripos_type,
                    "element":     element,
                    "xyz":         (x, y, z),
                }

            if section == "bond" and line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    bonds.append((int(parts[1]), int(parts[2]), parts[3]))

    return atoms, bonds


def parse_itp(itp_path: Path) -> dict:
    """
    Parsea [ atoms ] del itp.
    Devuelve {atom_idx (1-based): {atomtype, charge}}

    El match se realiza siempre por índice posicional, ya que la tutora
    confirmó que el orden de átomos en mol2 e itp es siempre el mismo.
    """
    by_index = {}
    in_block = False

    with open(itp_path) as f:
        for line in f:
            line = line.strip()
            if "[ atoms ]" in line:
                in_block = True
                continue
            if in_block and line.startswith("["):
                break
            if not line or line.startswith(";"):
                continue
            if in_block:
                parts = line.split()
                if len(parts) >= 7:
                    atom_idx           = int(parts[0])
                    by_index[atom_idx] = {
                        "atomtype": parts[1],
                        "charge":   float(parts[6]),
                    }

    return by_index


# =============================================================================
# CONSTRUIR MOL RDKIT DESDE EL PARSER PROPIO
# =============================================================================

def build_rdkit_mol(atoms: dict, bonds: list):
    """
    Construye un RWMol de RDKit a partir de los dicts del parser propio.
    Devuelve (mol, atom_id_to_rdkit_idx) o (None, {}) si falla.
    atom_id es 1-based (como en el mol2), rdkit_idx es 0-based.
    """
    rw = Chem.RWMol()
    id_to_idx = {}

    for atom_id in sorted(atoms.keys()):
        atom  = atoms[atom_id]
        elem  = atom["element"]
        try:
            rd_atom = Chem.Atom(elem)
        except Exception:
            rd_atom = Chem.Atom("C")
        idx = rw.AddAtom(rd_atom)
        id_to_idx[atom_id] = idx

    for a1, a2, btype_str in bonds:
        if a1 not in id_to_idx or a2 not in id_to_idx:
            continue
        i1 = id_to_idx[a1]
        i2 = id_to_idx[a2]
        bt = BOND_TYPE_MAP.get(btype_str.lower(), Chem.BondType.SINGLE)
        try:
            rw.AddBond(i1, i2, bt)
        except Exception:
            pass

    try:
        Chem.SanitizeMol(rw)
    except Exception:
        try:
            Chem.SanitizeMol(rw, Chem.SanitizeFlags.SANITIZE_ALL ^
                                  Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            return None, {}

    return rw.GetMol(), id_to_idx


# =============================================================================
# GEOMETRÍA
# =============================================================================

def bond_length(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def angle_deg(a, b, c):
    ba  = np.array(a) - np.array(b)
    bc  = np.array(c) - np.array(b)
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-12)
    return math.degrees(math.acos(np.clip(cos, -1.0, 1.0)))


def is_planar_geometry(center, neighbors, threshold=0.1):
    if len(neighbors) < 3:
        return True
    pts = np.array([center] + list(neighbors))
    pts -= pts.mean(axis=0)
    _, s, _ = np.linalg.svd(pts)
    return float(s[-1]) < threshold


# =============================================================================
# EXTRACCIÓN POR MOLÉCULA
# =============================================================================

def extract_molecule(mol2_path: Path, itp_path: Path) -> list:
    mol_name      = mol2_path.stem
    atoms, bonds  = parse_mol2(mol2_path)
    itp_by_index  = parse_itp(itp_path) if itp_path.exists() else {}

    if not atoms:
        print(f"  [WARN] No se parsearon átomos de {mol2_path.name}")
        return []

    # Construir grafo propio
    graph = {aid: [] for aid in atoms}
    for a1, a2, btype in bonds:
        if a1 in graph and a2 in graph:
            graph[a1].append((a2, btype))
            graph[a2].append((a1, btype))

    # Construir mol RDKit
    mol, id_to_idx = build_rdkit_mol(atoms, bonds)

    has_rdkit     = mol is not None
    has_gasteiger = False
    ring_info     = None

    if has_rdkit:
        try:
            rdPartialCharges.ComputeGasteigerCharges(mol)
            has_gasteiger = True
        except Exception:
            pass
        ring_info = mol.GetRingInfo()

    # n_dihedrals por átomo
    dihedral_counts = {aid: 0 for aid in atoms}
    for a1, a2, _ in bonds:
        if a1 not in graph or a2 not in graph:
            continue
        b, c = a1, a2
        for (a, _) in graph[b]:
            if a == c:
                continue
            for (d, _) in graph[c]:
                if d == b:
                    continue
                for idx in (a, b, c, d):
                    if idx in dihedral_counts:
                        dihedral_counts[idx] += 1

    rows      = []
    unmatched = []

    for atom_id in sorted(atoms.keys()):
        atom        = atoms[atom_id]
        tripos_type = atom["tripos_type"]
        atom_name   = atom["atom_name"]
        element     = atom["element"]
        xyz_i       = atom["xyz"]
        mass        = ATOMIC_MASSES.get(element, 0.0)

        neighbors    = graph[atom_id]
        coordination = len(neighbors)

        neighbor_info    = []
        bond_lengths     = []
        neighbor_xyzs    = []
        elem_counts      = {"C": 0, "H": 0, "O": 0, "N": 0}
        bond_type_counts = {"single": 0, "double": 0, "aromatic": 0}
        n_electroneg_1   = 0
        bonded_elements  = []

        for nb_id, btype_str in neighbors:
            nb       = atoms[nb_id]
            nb_elem  = nb["element"]
            nb_xyz   = nb["xyz"]

            bonded_elements.append(nb_elem)
            bond_lengths.append(bond_length(xyz_i, nb_xyz))
            neighbor_xyzs.append(nb_xyz)

            if nb_elem in elem_counts:
                elem_counts[nb_elem] += 1
            if nb_elem in ELECTRONEGATIVE:
                n_electroneg_1 += 1

            btype_norm = btype_str.lower()
            if btype_norm == "ar":
                bond_type_counts["aromatic"] += 1
            elif btype_norm in ("2", "3"):
                bond_type_counts["double"] += 1
            else:
                bond_type_counts["single"] += 1

            neighbor_info.append(f"{nb_elem}{btype_str}")

        # bonded_to_element
        if coordination == 1:
            bonded_to_element = bonded_elements[0] if bonded_elements else ""
        elif coordination > 1:
            bonded_to_element = ",".join(sorted(bonded_elements))
        else:
            bonded_to_element = ""

        neighbor_config = "-".join(sorted(neighbor_info))
        avg_bond_len    = float(np.mean(bond_lengths)) if bond_lengths else 0.0

        # Vecinos a 2 saltos
        seen2 = set()
        neighbors2_elems = []
        for nb_id, _ in neighbors:
            for nb2_id, _ in graph[nb_id]:
                if nb2_id != atom_id and nb2_id not in seen2:
                    seen2.add(nb2_id)
                    neighbors2_elems.append(atoms[nb2_id]["element"])

        neighbor2_config = "-".join(sorted(neighbors2_elems))
        n_electroneg_2   = sum(1 for e in neighbors2_elems if e in ELECTRONEGATIVE)

        # Ángulos
        nb_list = [nb_id for nb_id, _ in neighbors]
        angles  = [
            angle_deg(atoms[nb_list[j]]["xyz"], xyz_i, atoms[nb_list[k]]["xyz"])
            for j in range(len(nb_list))
            for k in range(j + 1, len(nb_list))
        ]
        avg_angle = float(np.mean(angles)) if angles else 0.0

        # Planaridad
        if tripos_type in PLANAR_TYPES:
            planar = True
        elif coordination >= 3:
            planar = is_planar_geometry(xyz_i, neighbor_xyzs)
        else:
            planar = False

        # Features RDKit
        rdkit_idx = id_to_idx.get(atom_id)
        if has_rdkit and rdkit_idx is not None:
            rd_atom       = mol.GetAtomWithIdx(rdkit_idx)
            is_in_ring    = ring_info.NumAtomRings(rdkit_idx) > 0
            atom_rings    = [r for r in ring_info.AtomRings() if rdkit_idx in r]
            ring_size     = min(len(r) for r in atom_rings) if atom_rings else 0
            is_aromatic   = rd_atom.GetIsAromatic()
            formal_charge = rd_atom.GetFormalCharge()
            if has_gasteiger:
                try:
                    g = float(rd_atom.GetDoubleProp("_GasteigerCharge"))
                    gasteiger_charge = 0.0 if (math.isnan(g) or math.isinf(g)) else g
                except Exception:
                    gasteiger_charge = 0.0
            else:
                gasteiger_charge = 0.0
        else:
            is_in_ring       = tripos_type in {"C.ar", "N.ar"}
            ring_size        = 6 if is_in_ring else 0
            is_aromatic      = is_in_ring
            formal_charge    = 0
            gasteiger_charge = 0.0

        degree_of_unsat = bond_type_counts["double"] + bond_type_counts["aromatic"]

        # Targets del ITP — siempre por índice posicional
        itp_atom = itp_by_index.get(atom_id)
        if itp_atom is None and itp_by_index:
            unmatched.append(atom_name)
        atomtype = itp_atom["atomtype"] if itp_atom else None
        charge   = itp_atom["charge"]   if itp_atom else None

        rows.append({
            "molecule":                mol_name,
            "atom_id":                 atom_id,
            "atom_name":               atom_name,
            "tripos_type":             tripos_type,
            "element":                 element,
            "mass":                    round(mass, 4),
            "coordination":            coordination,
            "avg_bond_len_A":          round(avg_bond_len, 4),
            "avg_angle_deg":           round(avg_angle, 4),
            "is_planar":               planar,
            "C_count":                 elem_counts["C"],
            "H_count":                 elem_counts["H"],
            "O_count":                 elem_counts["O"],
            "N_count":                 elem_counts["N"],
            "neighbor_config":         neighbor_config,
            "neighbor2_config":        neighbor2_config,
            "n_dihedrals":             dihedral_counts[atom_id],
            "n_electroneg_neighbors":  n_electroneg_1,
            "n_electroneg_neighbors2": n_electroneg_2,
            "bonds_single":            bond_type_counts["single"],
            "bonds_double":            bond_type_counts["double"],
            "bonds_aromatic":          bond_type_counts["aromatic"],
            "is_in_ring":              is_in_ring,
            "ring_size":               ring_size,
            "is_aromatic":             is_aromatic,
            "formal_charge":           formal_charge,
            "degree_of_unsat":         degree_of_unsat,
            "gasteiger_charge":        round(gasteiger_charge, 6),
            "bonded_to_element":       bonded_to_element,
            "atomtype":                atomtype,
            "charge":                  charge,
        })

    if unmatched:
        print(f"  [WARN] {mol2_path.name}: {len(unmatched)} sin match en .itp → {unmatched}")

    return rows


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extrae features de mol2/itp del ATB → CSV"
    )
    parser.add_argument("--input",  "-i", required=True, help="Carpeta con .mol2 e .itp")
    parser.add_argument("--output", "-o", default="dataset.csv", help="CSV de salida")
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        sys.exit(f"Error: '{input_dir}' no es una carpeta válida.")

    mol2_files = sorted(input_dir.glob("*.mol2"))
    if not mol2_files:
        sys.exit(f"No se encontraron .mol2 en '{input_dir}'.")

    print(f"Encontrados {len(mol2_files)} archivos .mol2\n")

    all_rows = []
    skipped  = []

    for mol2_path in mol2_files:
        itp_path = mol2_path.with_suffix(".itp")
        if not itp_path.exists():
            print(f"  [SKIP] Sin .itp: {mol2_path.name}")
            skipped.append(mol2_path.name)
            continue

        print(f"  ✔ {mol2_path.name} ...", end=" ", flush=True)
        rows = extract_molecule(mol2_path, itp_path)
        all_rows.extend(rows)
        print(f"{len(rows)} átomos")

    if not all_rows:
        sys.exit("No se generaron filas. Revisá los archivos.")

    df = pd.DataFrame(all_rows).dropna(subset=["atomtype", "charge"])
    df.to_csv(args.output, index=False)

    print(f"\n✔ Dataset guardado en: {args.output}")
    print(f"  {len(df)} átomos · {df['molecule'].nunique()} moléculas · {df.shape[1]} columnas")
    if skipped:
        print(f"  Saltadas (sin .itp): {skipped}")


if __name__ == "__main__":
    main()