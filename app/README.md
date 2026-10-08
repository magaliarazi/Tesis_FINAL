# MolParam — Predictor de Parámetros Atómicos

App web para predicción automática de `atomtype` y `charge` parcial
a partir de archivos `.mol2`, usando modelos Random Forest entrenados
con el campo de fuerzas GROMOS.

## Estructura

```
app/
├── app.py              ← servidor Flask
├── predictor.py        ← pipeline de predicción
├── templates/
│   └── index.html      ← interfaz
├── static/
│   ├── style.css
│   └── script.js
└── models/             ← carpeta donde van los modelos
    ├── clf_AA.joblib
    ├── reg_AA.joblib
    ├── clf_UA.joblib
    ├── reg_UA.joblib
    └── preprocessing_artifacts.json
```

## Instalación

```bash
pip install flask pandas numpy scikit-learn joblib rdkit
```

## Configuración

Copiá los modelos entrenados a la carpeta `models/`:

```bash
cp results_AA/clf_AA.joblib          app/models/
cp results_AA/reg_AA.joblib          app/models/
cp results_UA/clf_UA.joblib          app/models/
cp results_UA/reg_UA.joblib          app/models/
cp processed/preprocessing_artifacts.json  app/models/
```

## Uso

```bash
cd app/
python app.py
```

Abrí el navegador en: http://localhost:5000

## Output

La app devuelve por cada átomo:
- `molecule`: nombre del archivo .mol2
- `atom_id`: índice del átomo
- `atom_name`: nombre del átomo
- `element`: elemento químico
- `atomtype_predicho`: tipo atómico predicho por el modelo
- `charge_predicha`: carga parcial predicha (en unidades de carga del electrón)

Los resultados se muestran en dos pestañas (AA y UA) y se pueden
descargar como CSV.
