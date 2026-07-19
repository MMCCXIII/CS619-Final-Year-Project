# Professional Dataset Notes

The training CSV is generated reproducibly by `scripts/generate_professional_dataset.py`.

## Current Dataset

- Total training entities: 4,800 patient-case rows.
- Disease classes: 160.
- Medicine catalog entries: 195.
- Safety-rule entries: 358.
- Class balance: 30 rows per disease.
- Fields: `case_id`, `symptoms`, `disease`, `age_group`, `severity`, `source_note`.
- Source type: structured synthetic academic data, not real patient records.

## Design Rationale

The generated cases use disease-specific symptom profiles, supporting symptoms, red-flag symptoms, age-group weighting, severity weighting, and controlled overlap between related disease groups. The expanded catalog gives the ML pipeline a broader training surface than the original 64-row sample while keeping the project reproducible and privacy-safe.

Because this is not a licensed clinical dataset, the app still shows safety warnings, avoids dosage advice, flags prescription medicines, and treats recommendations as decision-support only.

## Reference Anchors

- CDC dengue clinical care guidance highlights warning signs, supportive hydration, acetaminophen for fever, and avoiding aspirin/NSAIDs: https://www.cdc.gov/dengue/hcp/clinical-care/index.html
- CDC flu treatment guidance describes antiviral medicines as prescription treatments that work best when started early and notes that antibiotics do not treat influenza viruses: https://www.cdc.gov/flu/treatment/
- MedlinePlus asthma information lists chest tightness, coughing, shortness of breath, and wheezing as core symptoms: https://medlineplus.gov/asthma.html
- MedlinePlus UTI information lists burning urination, fever/tiredness, frequent urge, lower-belly pressure, cloudy/reddish urine, and back/side pain as symptoms: https://medlineplus.gov/urinarytractinfections.html
