import pandas as pd
import numpy as np

"""
Projects the 2043/2045 population for Oberbayern, Niederbayern and Schwaben.

Replaces the notebooks create_population_2040.ipynb and create_population_2040_munich.ipynb.

Methodology:
- Landkreis-level 2043 totals are linearly interpolated from the 2024 and 2044
  figures published in "Regionalisierte Bevölkerungsvorausberechnung für Bayern
  bis 2044" (Bayerisches Landesamt für Statistik, April 2026; region14.pdf, pp. 24-25).
- Each Landkreis 2043 total is distributed to its communes using the 2024
  commune-level population shares; the 2024 sex × age_class distribution within
  each commune is preserved.
- The city of Munich (commune 091620000000) is replaced with the Planungsprognose
  2045 from the "Demografiebericht München – Teil 1A 2025" (p. 51, Tabelle 11).
  The sex split within each census age class is taken from 2024 Munich data.
"""

# Regionalisierte Bevölkerungsvorausberechnung für Bayern bis 2044
# Bayerisches Landesamt für Statistik, April 2026 (region14.pdf), Table pp. 24-25
# Values are in 1 000 persons.  2043 interpolated as: p24 + (p44 - p24) * 19/20
_LANDKREIS_POP = {  # departement_id: (pop_2024_k, pop_2044_k)
    # Oberbayern
    "09161": (141.2, 147.6), "09162": (1505.0, 1575.5), "09163": (65.3, 67.4),
    "09171": (113.9, 118.8), "09172": (105.3, 107.2),  "09173": (130.2, 131.7),
    "09174": (153.6, 157.0), "09175": (144.2, 148.5),  "09176": (135.7, 140.4),
    "09177": (140.1, 146.9), "09178": (184.6, 191.6),  "09179": (218.2, 215.2),
    "09180": (89.3,  87.8),  "09181": (122.1, 128.4),  "09182": (97.2,  96.6),
    "09183": (122.0, 129.9), "09184": (354.4, 354.9),  "09185": (99.4,  105.5),
    "09186": (130.8, 139.1), "09187": (258.6, 266.6),  "09188": (139.3, 137.5),
    "09189": (175.1, 178.1), "09190": (139.0, 144.8),
    # Niederbayern
    "09261": (71.9,  75.2),  "09262": (53.0,  55.5),   "09263": (49.0,  50.3),
    "09271": (121.8, 121.7), "09272": (78.7,  78.3),   "09273": (126.0, 131.9),
    "09274": (162.2, 172.7), "09275": (194.7, 199.1),  "09276": (77.3,  74.2),
    "09277": (120.5, 125.4), "09278": (103.0, 107.3),  "09279": (101.0, 107.0),
    # Schwaben
    "09761": (301.1, 312.5), "09762": (46.2,  49.9),   "09763": (67.6,  69.1),
    "09764": (44.2,  47.5),  "09771": (136.8, 142.5),  "09772": (262.8, 280.4),
    "09773": (98.9,  105.1), "09774": (129.8, 137.9),  "09775": (183.6, 195.4),
    "09776": (82.4,  84.2),  "09777": (142.0, 151.0),  "09778": (147.3, 159.8),
    "09779": (135.1, 138.1), "09780": (155.4, 159.5),
}

# Demografiebericht München – Teil 1A 2025, p. 51, Tabelle 11 (Planungsprognose 2045)
# (age_lo, age_hi_inclusive, total)
_MUNICH_2045 = [
    (0,  2,   56184),   # Kinderbetreuung
    (3,  5,   50839),   # Kindergarten
    (6,  9,   62569),   # Grundschule
    (10, 15,  88973),   # Sekundarstufe I
    (16, 18,  45229),   # Sekundarstufe II / berufliche Bildung
    (19, 24, 139079),   # FH, Uni, berufliche Qualifikation
    (25, 39, 507915),   # Haushaltsgründung, Erwerbstätigkeit
    (40, 64, 571871),   # Erwerbstätigkeit
    (65, 74, 146578),   # Senioren
    (75, 99, 160284),   # Hochbetagte (75+)
]

_MUNICH_COMMUNE_ID = "091620000000"


def configure(context):
    context.stage("bavaria.data.census.population_2024")


def execute(context):
    df_2024 = context.stage("bavaria.data.census.population_2024").copy()
    df_2024["commune_id"] = df_2024["commune_id"].astype(str)
    df_2024["departement_id"] = df_2024["commune_id"].str[:5]

    # ── 1. Scale non-Munich communes by the Landkreis 2043/2024 growth factor ──
    # For each Landkreis, growth_factor = pop_2043 / pop_2024 (from statistics office).
    # All weights within a commune are multiplied by its Landkreis's growth factor,
    # which preserves the 2024 sex × age_class distribution within the commune
    # while matching the projected 2043 Landkreis total.

    growth_factor = {
        k: (p24 + (p44 - p24) * 19.0 / 20.0) / p24
        for k, (p24, p44) in _LANDKREIS_POP.items()
    }

    df_result = df_2024[df_2024["commune_id"] != _MUNICH_COMMUNE_ID].copy()
    df_result["growth"] = df_result["departement_id"].map(growth_factor).fillna(1.0)
    df_result["weight"] = df_result["weight"] * df_result["growth"]
    df_result = df_result.drop(columns=["departement_id", "growth"])

    # ── 2. Build Munich 2045 data ──────────────────────────────────────────────
    # Step 2a: distribute each Munich report age group uniformly over single ages.
    pop_per_age = {}
    for lo, hi, total in _MUNICH_2045:
        val = total / (hi - lo + 1)
        for a in range(lo, hi + 1):
            pop_per_age[a] = val

    # Step 2b: aggregate single-age values to census age class intervals.
    df_munich_2024 = df_2024[df_2024["commune_id"] == _MUNICH_COMMUNE_ID]
    age_classes = sorted(df_munich_2024["age_class"].unique())
    age_intervals = [
        (age_classes[i], age_classes[i + 1] - 1 if i < len(age_classes) - 1 else 99)
        for i in range(len(age_classes))
    ]

    age_class_total = {}
    for lo, hi in age_intervals:
        age_class_total[lo] = sum(
            pop_per_age.get(a, 0.0) for a in range(lo, hi + 1)
        )

    # Step 2c: female share per age class from 2024 Munich census data.
    female_share = (
        df_munich_2024.groupby(["age_class", "sex"])["weight"].sum()
        .unstack(fill_value=0.0)
    )
    if "female" not in female_share.columns:
        female_share["female"] = 0.0
    if "male" not in female_share.columns:
        female_share["male"] = 0.0
    female_share["total"] = female_share["female"] + female_share["male"]
    female_share["f_share"] = (
        female_share["female"] / female_share["total"].replace(0.0, np.nan)
    ).fillna(0.5)

    # Step 2d: assemble Munich 2045 rows.
    munich_rows = []
    for age_class, total in age_class_total.items():
        f_sh = female_share.loc[age_class, "f_share"] if age_class in female_share.index else 0.5
        munich_rows.append({
            "commune_id": _MUNICH_COMMUNE_ID,
            "sex": "female",
            "age_class": age_class,
            "weight": total * f_sh,
        })
        munich_rows.append({
            "commune_id": _MUNICH_COMMUNE_ID,
            "sex": "male",
            "age_class": age_class,
            "weight": total * (1.0 - f_sh),
        })

    df_munich_2045 = pd.DataFrame(munich_rows)
    df_munich_2045["sex"] = df_munich_2045["sex"].astype("category")

    # ── 3. Combine ─────────────────────────────────────────────────────────────
    df_result["commune_id"] = df_result["commune_id"].astype("category")
    df_munich_2045["commune_id"] = df_munich_2045["commune_id"].astype("category")

    return pd.concat([df_result, df_munich_2045], ignore_index=True)[
        ["commune_id", "sex", "age_class", "weight"]
    ]
