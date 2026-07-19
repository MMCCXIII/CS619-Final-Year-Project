# Test Cases

| Test case | Purpose | Expected result |
| --- | --- | --- |
| Symptom extraction | Verify free text symptoms are normalized against the vocabulary. | High fever, rash, and pain behind eyes are detected. |
| Database seeding | Verify schema and seed data are complete. | At least 150 diseases, 150 medicines, and 150 contraindication rules exist. |
| Dataset scale | Verify expanded training data has professional-scale volume for the project. | At least 2,000 training entities and 150 disease classes exist. |
| Dengue safety ranking | Verify severe dengue symptoms are identified and NSAID-like unsafe suggestions are not prioritized. | Dengue Fever is top condition and bleeding warning is shown. |
| Model training | Verify multiple models train and metrics are saved. | `models/disease_model.joblib` and `models/model_metrics.json` are created. |
| Streamlit recommendation | Verify user can enter symptoms and see ranked outputs. | Likely conditions and ranked medicines display in the Recommendation tab. |
