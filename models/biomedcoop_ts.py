"""
BiomedCoOp-style class prompting for MedTsLLM (decoder-only classification).

This module ports *only the prompting strategy* of BiomedCoOp
(https://github.com/HealthX-Lab/BiomedCoOp) into the decoder-only MedTsLLM
classification path. It does **not** port the CLIP dual-encoder, the learnable
context vectors, the statistics-based prompt selection (SPS), or the
distillation losses (SCCM / KDSP), because those require an image-text
contrastive model that MedTsLLM does not have.

What BiomedCoOp's prompting strategy is
---------------------------------------
BiomedCoOp builds, for every class, an *ensemble of LLM-generated descriptive
sentences* (their ``BIOMEDCOOP_TEMPLATES`` dict in
``trainers/prompt_templates.py``). Each sentence describes the characteristic
appearance of that class. Instead of a single hand-written template
("a photo of a {}."), the model is conditioned on many rich, class-specific
descriptions.

Adaptation to MedTsLLM
----------------------
MedTsLLM is a decoder-only LLM that consumes a *text* prompt followed by the
reprogrammed time-series patch embeddings. We therefore inject the per-class
descriptions directly into that text prompt (see ``MedTsLLM.build_class_prompt``
in ``models/medtsllm.py``). At each step a few descriptions per class are
sampled (the ensemble-sampling analogue of BiomedCoOp's prompt ensemble),
giving the LLM explicit, expert-style knowledge of what each diagnostic
category looks like before it sees the signal.

``BIOMEDCOOP_TS_TEMPLATES`` below mirrors the structure of BiomedCoOp's
``BIOMEDCOOP_TEMPLATES`` (``class -> list[str]``), populated with the
characteristic 12-lead ECG features of the five PTB-XL diagnostic superclasses
(NORM, MI, STTC, CD, HYP).

Public API
----------
``load_class_prompts(path, class_codes) -> list[list[str]]``
    Returns one list of description strings per class, in the order of
    ``class_codes``. ``path`` may point to a JSON file (``code -> list[str]``)
    to override / extend the built-in templates; pass ``""`` / ``"none"`` /
    ``"default"`` to use the built-ins.
"""

import json
import os

__all__ = ["load_class_prompts", "BIOMEDCOOP_TS_TEMPLATES", "CODE_TO_NAME"]


# Canonical PTB-XL superclass code -> human-readable diagnostic name.
CODE_TO_NAME = {
    "NORM": "Normal ECG",
    "MI": "Myocardial Infarction",
    "STTC": "ST/T Change",
    "CD": "Conduction Disturbance",
    "HYP": "Hypertrophy",
}


# Per-class ensemble of descriptive prompts, mirroring BiomedCoOp's
# BIOMEDCOOP_TEMPLATES (class -> list[str]). Descriptions state the
# well-established characteristic 12-lead ECG features of each superclass.
BIOMEDCOOP_TS_TEMPLATES = {
    "NORM": [
        "A normal ECG shows a regular sinus rhythm with a heart rate between 60 and 100 beats per minute.",
        "Each beat is preceded by an upright P wave in lead II, indicating normal sinus node activation.",
        "The PR interval is constant and falls within 120 to 200 milliseconds.",
        "The QRS complex is narrow, typically under 120 milliseconds, reflecting normal ventricular conduction.",
        "The ST segment is isoelectric, sitting level with the baseline with no elevation or depression.",
        "T waves are upright in most leads and concordant with the QRS complex.",
        "The QT interval is within normal limits when corrected for heart rate.",
        "R-wave progression across the precordial leads is smooth and orderly from V1 to V6.",
        "There are no pathological Q waves in a normal ECG.",
        "The rhythm is regular with even spacing between consecutive R waves.",
        "P waves are uniform in shape and precede every QRS complex one-to-one.",
        "The frontal-plane QRS axis falls within the normal range.",
        "No signs of chamber enlargement or hypertrophy are present.",
        "The baseline is stable without abnormal deflections between beats.",
        "A normal ECG has no ectopic beats or conduction blocks.",
        "Overall, the tracing shows organized, physiologic electrical activity with no abnormalities.",
    ],
    "MI": [
        "Myocardial infarction is characterized by ST-segment elevation in the leads facing the infarcted region.",
        "Pathological Q waves develop, reflecting established myocardial necrosis.",
        "In acute infarction, hyperacute peaked T waves may appear early before ST elevation.",
        "Reciprocal ST-segment depression is often seen in leads opposite the infarct.",
        "T-wave inversion evolves over time as the infarction progresses.",
        "Anterior infarction shows changes in the precordial leads V1 through V4.",
        "Inferior infarction produces changes in leads II, III, and aVF.",
        "Lateral infarction is reflected in leads I, aVL, V5, and V6.",
        "Loss of R-wave amplitude can accompany the development of Q waves.",
        "ST elevation in myocardial infarction is typically convex or dome-shaped.",
        "The location of ST elevation helps localize the coronary artery involved.",
        "Old or prior infarction is suggested by persistent Q waves without acute ST changes.",
        "The combination of Q waves, ST elevation, and T-wave inversion suggests an evolving infarction.",
        "Posterior infarction shows tall R waves and ST depression in V1 to V3 as reciprocal changes.",
        "A regional rather than diffuse pattern of abnormality is typical of infarction.",
        "Overall, myocardial infarction shows a regional pattern of ST-segment, Q-wave, and T-wave abnormalities.",
    ],
    "STTC": [
        "ST/T changes refer to abnormalities of the ST segment and T wave without diagnostic Q waves.",
        "ST-segment depression may indicate myocardial ischemia or strain.",
        "T-wave inversion can reflect ischemia, strain, or a non-specific repolarization change.",
        "Flattening of the T wave is a common non-specific ST/T abnormality.",
        "ST depression is often horizontal or downsloping in ischemia.",
        "Diffuse ST/T changes may be non-specific and not localize to a single territory.",
        "Repolarization abnormalities can be secondary to hypertrophy or conduction abnormalities.",
        "Symmetric, deep T-wave inversions may suggest active ischemia.",
        "ST/T changes can be dynamic and vary between recordings.",
        "Non-specific ST/T changes deviate from normal but do not meet criteria for infarction.",
        "ST-segment depression greater than one millimetre is generally considered significant.",
        "Subtle T-wave abnormalities may be the only sign of a repolarization disturbance.",
        "Drug effects and electrolyte imbalance can also produce ST/T changes.",
        "The ST segment may show subtle upsloping, downsloping, or horizontal depression.",
        "ST/T change describes repolarization abnormality without the injury current of infarction.",
        "Overall, ST/T change describes repolarization abnormalities of the ST segment and T wave.",
    ],
    "CD": [
        "Conduction disturbances involve delayed or blocked propagation of the cardiac impulse.",
        "Right bundle branch block produces a wide QRS with an rSR' pattern in V1.",
        "Left bundle branch block produces a wide QRS with broad, notched R waves in the lateral leads.",
        "First-degree AV block shows a prolonged PR interval beyond 200 milliseconds.",
        "Second-degree AV block shows intermittent failure of P waves to conduct to the ventricles.",
        "Third-degree (complete) AV block shows AV dissociation with independent P waves and QRS complexes.",
        "Bundle branch block widens the QRS complex beyond 120 milliseconds.",
        "Left anterior fascicular block causes left axis deviation with a characteristic QRS morphology.",
        "Left posterior fascicular block causes right axis deviation in the appropriate setting.",
        "Intraventricular conduction delay widens the QRS without meeting specific bundle branch criteria.",
        "In right bundle branch block, there is a wide, slurred S wave in leads I and V6.",
        "Conduction disturbances often produce secondary ST/T changes opposite to the QRS.",
        "Mobitz I (Wenckebach) block shows progressive PR prolongation before a dropped beat.",
        "Mobitz II block shows sudden non-conducted P waves without progressive PR change.",
        "The hallmark of conduction disturbance is an abnormal QRS width or AV relationship.",
        "Overall, conduction disturbances are recognized by widened QRS complexes or altered AV conduction.",
    ],
    "HYP": [
        "Hypertrophy on ECG reflects increased myocardial mass of the atria or ventricles.",
        "Left ventricular hypertrophy shows increased QRS voltage in the precordial leads.",
        "The Sokolow-Lyon criterion sums the S wave in V1 and the R wave in V5 or V6.",
        "Left ventricular hypertrophy is often accompanied by a strain pattern of ST depression and T inversion.",
        "Right ventricular hypertrophy shows a dominant R wave in V1 and right axis deviation.",
        "Left atrial enlargement produces a broad, notched P wave in lead II, known as P mitrale.",
        "Right atrial enlargement produces a tall, peaked P wave in lead II, known as P pulmonale.",
        "Increased R-wave amplitude in the lateral leads suggests left ventricular hypertrophy.",
        "Left ventricular hypertrophy may be associated with left axis deviation.",
        "The strain pattern shows downsloping ST depression with asymmetric T-wave inversion.",
        "Right ventricular hypertrophy may show a deep S wave in the lateral precordial leads.",
        "Voltage criteria combined with repolarization changes increase confidence in hypertrophy.",
        "Hypertrophy increases the amplitude and may widen the QRS complex.",
        "Biventricular hypertrophy shows features of both left and right ventricular enlargement.",
        "Chamber enlargement alters P-wave morphology and QRS voltage.",
        "Overall, hypertrophy is recognized by increased voltages and chamber-enlargement patterns.",
    ],
}


def _normalize_templates(raw):
    """Coerce a loaded JSON object into ``{key: list[str]}`` with string values."""
    if not isinstance(raw, dict):
        raise ValueError(
            "BiomedCoOp class-prompt file must be a JSON object mapping "
            "class code/name -> list of description strings."
        )
    out = {}
    for k, v in raw.items():
        if isinstance(v, str):
            v = [v]
        if not isinstance(v, (list, tuple)) or len(v) == 0:
            raise ValueError(f"Class '{k}' must map to a non-empty list of strings.")
        out[k] = [str(s) for s in v]
    return out


def load_class_prompts(path, class_codes):
    """Load per-class BiomedCoOp-style descriptive prompts.

    Args:
        path: Path to a JSON file mapping class code (or name) -> list of
            description strings. Pass ``""``, ``"none"``, ``"default"``, or
            ``"builtin"`` (or ``None``) to use the built-in
            ``BIOMEDCOOP_TS_TEMPLATES``. A JSON file may define a subset of
            classes; any class it does not define falls back to the built-ins.
        class_codes: Ordered iterable of class codes (e.g. PTB-XL superclasses
            ``["NORM", "MI", "STTC", "CD", "HYP"]``). The returned list follows
            this exact order so it lines up with the model's class index order.

    Returns:
        ``list[list[str]]`` of length ``len(class_codes)``: each element is the
        list of description strings for the corresponding class.

    Raises:
        FileNotFoundError: if ``path`` is given but does not exist.
        KeyError: if a requested class has no descriptions in either the file
            or the built-in templates.
    """
    # Start from the built-in templates, then let a JSON file override/extend.
    templates = dict(BIOMEDCOOP_TS_TEMPLATES)

    use_builtin = path is None or str(path).strip().lower() in (
        "", "none", "default", "builtin", "built-in",
    )
    if not use_builtin:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"BiomedCoOp class-prompt file not found: {path!r}. "
                f"Set prompting.class_prompts_path to a valid JSON file, or to "
                f'"" to use the built-in ECG templates.'
            )
        with open(path, "r", encoding="utf-8") as f:
            file_templates = _normalize_templates(json.load(f))
        templates.update(file_templates)

    # Case-insensitive lookup that accepts either codes ("MI") or names
    # ("Myocardial Infarction").
    lookup = {str(k).strip().lower(): v for k, v in templates.items()}
    for code, name in CODE_TO_NAME.items():
        if code in templates and name.strip().lower() not in lookup:
            lookup[name.strip().lower()] = templates[code]

    descriptions = []
    for code in class_codes:
        key = str(code).strip().lower()
        if key not in lookup:
            name = CODE_TO_NAME.get(code)
            if name is not None and name.strip().lower() in lookup:
                key = name.strip().lower()
            else:
                raise KeyError(
                    f"No BiomedCoOp class prompts found for class {code!r}. "
                    f"Available keys: {sorted(templates.keys())}. Add an entry "
                    f"for this class to your class_prompts JSON file or to "
                    f"BIOMEDCOOP_TS_TEMPLATES."
                )
        descriptions.append(list(lookup[key]))

    return descriptions
