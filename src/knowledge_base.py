from dataclasses import dataclass


@dataclass(frozen=True)
class DiseaseProfile:
    name: str
    group: str
    core_symptoms: tuple[str, ...]
    supporting_symptoms: tuple[str, ...]
    red_flag_symptoms: tuple[str, ...]
    age_weights: dict[str, float]
    severity_weights: dict[str, float]


RELATED_NOISE = {
    "respiratory": ("fatigue", "headache", "mild fever", "sore throat", "runny nose", "chest discomfort"),
    "gastro": ("nausea", "loss of appetite", "weakness", "dizziness", "fatigue"),
    "cardiometabolic": ("fatigue", "headache", "dizziness", "weakness", "poor sleep"),
    "neuro_pain": ("fatigue", "dizziness", "poor sleep", "stress", "neck stiffness"),
    "infectious_tropical": ("fatigue", "weakness", "headache", "nausea", "loss of appetite"),
    "urinary_reproductive": ("fatigue", "nausea", "lower abdominal pain", "back pain"),
    "skin_allergy": ("itching", "redness", "swelling", "skin tenderness"),
    "eye_ent": ("eye irritation", "ear pressure", "headache", "facial discomfort"),
    "pediatric_general": ("irritability", "poor feeding", "fever", "fatigue"),
    "emergency_redflag": ("severe weakness", "confusion", "rapid breathing", "fainting"),
}


GROUP_TEMPLATES = {
    "respiratory": {
        "core": ("cough", "sore throat", "fever", "nasal congestion", "fatigue"),
        "supporting": ("runny nose", "chest tightness", "wheezing", "headache", "body aches", "mucus"),
        "red_flags": ("shortness of breath", "chest pain", "low oxygen", "confusion"),
        "age": {"child": 0.25, "adult": 0.50, "older adult": 0.25},
        "severity": {"mild": 0.35, "moderate": 0.48, "severe": 0.17},
    },
    "gastro": {
        "core": ("nausea", "upper abdominal pain", "bloating", "loss of appetite", "vomiting"),
        "supporting": ("heartburn", "diarrhea", "constipation", "belching", "weakness", "dizziness"),
        "red_flags": ("black stool", "severe abdominal pain", "vomiting blood", "fainting"),
        "age": {"child": 0.15, "adult": 0.60, "older adult": 0.25},
        "severity": {"mild": 0.36, "moderate": 0.50, "severe": 0.14},
    },
    "cardiometabolic": {
        "core": ("fatigue", "dizziness", "headache", "weakness", "blurred vision"),
        "supporting": ("palpitations", "shortness of breath on exertion", "poor sleep", "swelling ankles", "excessive thirst"),
        "red_flags": ("chest pain", "shortness of breath", "confusion", "fainting"),
        "age": {"child": 0.05, "adult": 0.50, "older adult": 0.45},
        "severity": {"mild": 0.24, "moderate": 0.56, "severe": 0.20},
    },
    "neuro_pain": {
        "core": ("pain", "fatigue", "dizziness", "poor sleep", "stress"),
        "supporting": ("neck stiffness", "tingling", "muscle pain", "joint pain", "anxiety", "weakness"),
        "red_flags": ("confusion", "new neurological weakness", "worst headache", "seizure"),
        "age": {"child": 0.10, "adult": 0.64, "older adult": 0.26},
        "severity": {"mild": 0.38, "moderate": 0.47, "severe": 0.15},
    },
    "infectious_tropical": {
        "core": ("fever", "chills", "headache", "body aches", "fatigue"),
        "supporting": ("rash", "vomiting", "sweating", "loss of appetite", "weakness", "travel exposure"),
        "red_flags": ("confusion", "persistent vomiting", "bleeding gums", "severe weakness"),
        "age": {"child": 0.24, "adult": 0.56, "older adult": 0.20},
        "severity": {"mild": 0.16, "moderate": 0.46, "severe": 0.38},
    },
    "urinary_reproductive": {
        "core": ("lower abdominal pain", "pelvic pain", "fatigue", "back pain", "nausea"),
        "supporting": ("urinary urgency", "burning urination", "fever", "night urination", "pressure in lower belly"),
        "red_flags": ("high fever", "severe pelvic pain", "fainting", "pregnancy with urinary or fever symptoms"),
        "age": {"child": 0.12, "adult": 0.62, "older adult": 0.26},
        "severity": {"mild": 0.30, "moderate": 0.52, "severe": 0.18},
    },
    "skin_allergy": {
        "core": ("rash", "itching", "redness", "swelling", "skin tenderness"),
        "supporting": ("dry skin", "blisters", "scaling", "warm skin", "painful skin", "discharge"),
        "red_flags": ("facial swelling", "rapidly spreading redness", "fever", "severe allergic reaction"),
        "age": {"child": 0.22, "adult": 0.55, "older adult": 0.23},
        "severity": {"mild": 0.50, "moderate": 0.38, "severe": 0.12},
    },
    "eye_ent": {
        "core": ("eye irritation", "ear pressure", "facial discomfort", "headache", "localized pain"),
        "supporting": ("red eye", "watery eyes", "ear pain", "hearing change", "mouth pain", "nasal congestion"),
        "red_flags": ("vision loss", "severe eye pain", "facial swelling", "trouble breathing"),
        "age": {"child": 0.20, "adult": 0.56, "older adult": 0.24},
        "severity": {"mild": 0.48, "moderate": 0.41, "severe": 0.11},
    },
    "pediatric_general": {
        "core": ("fever", "irritability", "poor feeding", "fatigue", "sleep disturbance"),
        "supporting": ("rash", "cough", "abdominal pain", "ear pain", "dehydration signs", "crying"),
        "red_flags": ("rapid breathing", "seizure", "severe dehydration", "lethargy"),
        "age": {"child": 0.82, "adult": 0.16, "older adult": 0.02},
        "severity": {"mild": 0.42, "moderate": 0.44, "severe": 0.14},
    },
    "emergency_redflag": {
        "core": ("severe pain", "confusion", "rapid breathing", "fainting", "severe weakness"),
        "supporting": ("chest pain", "shortness of breath", "bleeding", "severe abdominal pain", "low oxygen"),
        "red_flags": ("loss of consciousness", "severe bleeding", "blue lips", "one sided weakness"),
        "age": {"child": 0.18, "adult": 0.48, "older adult": 0.34},
        "severity": {"mild": 0.02, "moderate": 0.18, "severe": 0.80},
    },
}


DISEASE_GROUPS = {
    "respiratory": [
        "Common Cold", "Influenza", "COVID-19", "Bronchitis", "Pneumonia", "Asthma",
        "Allergic Rhinitis", "Sinusitis", "Pharyngitis", "Tonsillitis", "Laryngitis",
        "Otitis Media", "COPD Exacerbation", "Tuberculosis", "Pertussis", "RSV Infection",
    ],
    "gastro": [
        "Gastritis", "GERD", "Gastroenteritis", "Peptic Ulcer Disease", "Irritable Bowel Syndrome",
        "Constipation", "Diarrheal Illness", "Food Poisoning", "Hepatitis A", "Gallbladder Disease",
        "Pancreatitis", "Appendicitis", "Hemorrhoids", "Diverticulitis", "Lactose Intolerance",
        "Inflammatory Bowel Flare",
    ],
    "cardiometabolic": [
        "Hypertension", "Type 2 Diabetes", "Hyperlipidemia", "Hypothyroidism", "Hyperthyroidism",
        "Anemia", "Obesity", "Metabolic Syndrome", "Gout", "Chronic Kidney Disease", "Heart Failure",
        "Angina", "Atrial Fibrillation", "Peripheral Artery Disease", "Vitamin D Deficiency",
        "Electrolyte Imbalance",
    ],
    "neuro_pain": [
        "Migraine", "Tension Headache", "Cluster Headache", "Vertigo", "Epilepsy", "Neuropathy",
        "Sciatica", "Low Back Pain", "Osteoarthritis", "Rheumatoid Arthritis", "Fibromyalgia",
        "Neck Strain", "Carpal Tunnel Syndrome", "Insomnia", "Anxiety Disorder", "Depression",
    ],
    "infectious_tropical": [
        "Malaria", "Dengue Fever", "Typhoid Fever", "Chikungunya", "Zika Virus Infection",
        "Leptospirosis", "Cholera", "Tuberculosis Exposure", "Measles", "Chickenpox", "Mumps",
        "Rubella", "Tetanus Risk Wound", "Rabies Exposure", "Scabies", "Fungal Skin Infection",
    ],
    "urinary_reproductive": [
        "Urinary Tract Infection", "Kidney Stone", "Pyelonephritis", "Prostatitis", "Vaginitis",
        "Pelvic Inflammatory Disease", "Dysmenorrhea", "Menopause Symptoms", "Pregnancy Nausea",
        "Bacterial Vaginosis", "Candidiasis", "Endometriosis", "Polycystic Ovary Syndrome",
        "Erectile Dysfunction", "Benign Prostatic Hyperplasia", "Testicular Pain",
    ],
    "skin_allergy": [
        "Acne", "Eczema", "Psoriasis", "Urticaria", "Cellulitis", "Impetigo", "Shingles",
        "Herpes Simplex", "Contact Dermatitis", "Sunburn", "Minor Burn", "Insect Bite Reaction",
        "Head Lice", "Rosacea", "Seborrheic Dermatitis", "Skin Abscess",
    ],
    "eye_ent": [
        "Conjunctivitis", "Dry Eye Syndrome", "Glaucoma Alert", "Cataract Symptoms", "Stye",
        "Ear Wax Impaction", "Otitis Externa", "Tinnitus", "Hearing Loss", "Dental Caries",
        "Gingivitis", "Mouth Ulcer", "Oral Thrush", "Nosebleed", "Allergic Conjunctivitis",
        "Motion Sickness",
    ],
    "pediatric_general": [
        "Hand Foot and Mouth Disease", "Croup", "Febrile Seizure", "Diaper Rash", "Colic",
        "Teething Pain", "Pinworm Infection", "Growth Pain", "Dehydration Child", "Pediatric Fever",
        "Worm Infestation", "Nutritional Anemia Child", "Ear Infection Child", "Childhood Asthma Flare",
        "School Anxiety", "Bedwetting",
    ],
    "emergency_redflag": [
        "Stroke Warning", "Heart Attack Warning", "Sepsis Warning", "Severe Allergic Reaction",
        "Severe Dehydration", "Meningitis Warning", "Diabetic Ketoacidosis Warning", "Hypoglycemia",
        "Heat Stroke", "Poisoning Exposure", "Severe Burn", "Fracture Suspected",
        "Deep Vein Thrombosis", "Pulmonary Embolism Warning", "Severe Bleeding", "Acute Abdomen",
    ],
}


SPECIFIC_SYMPTOMS_BY_KEYWORD = {
    "cold": ("runny nose", "sneezing", "nasal congestion"),
    "influenza": ("high fever", "chills", "body aches"),
    "covid": ("loss of smell", "loss of taste", "dry cough"),
    "bronchitis": ("productive cough", "mucus", "wheezing"),
    "pneumonia": ("productive cough", "rapid breathing", "chest pain"),
    "asthma": ("wheezing", "shortness of breath", "night cough"),
    "rhinitis": ("itchy eyes", "sneezing", "watery eyes"),
    "sinus": ("facial pressure", "thick nasal discharge", "reduced smell"),
    "pharyngitis": ("throat pain", "painful swallowing", "fever"),
    "tonsillitis": ("swollen tonsils", "throat pain", "fever"),
    "laryngitis": ("hoarse voice", "voice loss", "throat irritation"),
    "otitis": ("ear pain", "ear fullness", "hearing change"),
    "copd": ("chronic cough", "wheezing", "shortness of breath on exertion"),
    "tuberculosis": ("night sweats", "weight loss", "persistent cough"),
    "pertussis": ("paroxysmal cough", "whooping cough", "post cough vomiting"),
    "rsv": ("wheezing", "runny nose", "rapid breathing"),
    "gastritis": ("burning stomach", "upper abdominal pain", "nausea"),
    "gerd": ("heartburn", "acid reflux", "regurgitation"),
    "gastroenteritis": ("diarrhea", "vomiting", "abdominal cramps"),
    "ulcer": ("burning stomach", "black stool", "early fullness"),
    "irritable": ("abdominal cramps", "bloating", "changed bowel habits"),
    "constipation": ("hard stool", "straining", "infrequent bowel movement"),
    "diarrheal": ("watery diarrhea", "abdominal cramps", "dehydration signs"),
    "poisoning": ("vomiting", "diarrhea", "food exposure"),
    "hepatitis": ("jaundice", "dark urine", "right upper abdominal pain"),
    "gallbladder": ("right upper abdominal pain", "pain after fatty meals", "nausea"),
    "pancreatitis": ("severe upper abdominal pain", "pain radiating to back", "vomiting"),
    "appendicitis": ("right lower abdominal pain", "loss of appetite", "fever"),
    "hemorrhoids": ("rectal bleeding", "anal itching", "painful bowel movement"),
    "diverticulitis": ("left lower abdominal pain", "fever", "changed bowel habits"),
    "lactose": ("bloating after dairy", "diarrhea after dairy", "gas"),
    "bowel flare": ("bloody diarrhea", "abdominal cramps", "weight loss"),
    "hypertension": ("high blood pressure", "morning headache", "dizziness"),
    "diabetes": ("frequent urination", "excessive thirst", "dry mouth"),
    "hyperlipidemia": ("high cholesterol history", "xanthelasma", "cardiovascular risk"),
    "hypothyroidism": ("cold intolerance", "weight gain", "dry skin"),
    "hyperthyroidism": ("heat intolerance", "weight loss", "palpitations"),
    "anemia": ("pale skin", "fatigue", "shortness of breath on exertion"),
    "obesity": ("weight gain", "snoring", "joint pain"),
    "metabolic": ("waist weight gain", "high blood pressure", "high blood sugar"),
    "gout": ("big toe pain", "red swollen joint", "sudden joint pain"),
    "kidney": ("flank pain", "swelling ankles", "foamy urine"),
    "heart failure": ("swelling ankles", "shortness of breath lying flat", "fatigue"),
    "angina": ("chest pressure", "pain with exertion", "left arm discomfort"),
    "atrial": ("irregular heartbeat", "palpitations", "dizziness"),
    "peripheral artery": ("leg pain walking", "cold feet", "slow wound healing"),
    "vitamin d": ("bone pain", "muscle weakness", "low vitamin d history"),
    "electrolyte": ("muscle cramps", "palpitations", "weakness"),
    "migraine": ("one sided headache", "light sensitivity", "visual aura"),
    "tension": ("pressure around head", "dull headache", "stress"),
    "cluster": ("eye tearing", "severe one sided eye pain", "nasal stuffiness"),
    "vertigo": ("spinning sensation", "balance problem", "nausea"),
    "epilepsy": ("seizure", "loss of awareness", "post seizure confusion"),
    "neuropathy": ("burning feet", "tingling feet", "numbness"),
    "sciatica": ("shooting leg pain", "low back pain", "numb leg"),
    "low back": ("low back pain", "stiff back", "pain worse bending"),
    "osteoarthritis": ("joint stiffness", "joint pain with use", "reduced range of motion"),
    "rheumatoid": ("morning joint stiffness", "swollen small joints", "joint warmth"),
    "fibromyalgia": ("widespread pain", "fatigue", "sleep disturbance"),
    "neck strain": ("neck pain", "muscle spasm", "reduced neck movement"),
    "carpal": ("hand numbness", "wrist tingling", "night hand pain"),
    "insomnia": ("difficulty sleeping", "early awakening", "daytime fatigue"),
    "anxiety": ("excess worry", "palpitations", "restlessness"),
    "depression": ("low mood", "loss of interest", "sleep change"),
    "malaria": ("cyclic fever", "chills", "sweating"),
    "dengue": ("high fever", "severe joint pain", "pain behind eyes"),
    "typhoid": ("stepwise fever", "abdominal pain", "constipation"),
    "chikungunya": ("severe joint pain", "fever", "rash"),
    "zika": ("rash", "conjunctivitis", "joint pain"),
    "leptospirosis": ("calf pain", "red eyes", "jaundice"),
    "cholera": ("profuse watery diarrhea", "severe dehydration", "leg cramps"),
    "measles": ("fever", "cough", "koplik spots"),
    "chickenpox": ("itchy blisters", "fever", "skin vesicles"),
    "mumps": ("parotid swelling", "jaw pain", "fever"),
    "rubella": ("fine rash", "lymph node swelling", "mild fever"),
    "tetanus": ("jaw stiffness", "muscle spasms", "dirty wound"),
    "rabies": ("animal bite", "tingling at bite site", "fear of water"),
    "scabies": ("night itching", "burrow tracks", "itchy rash"),
    "fungal": ("ring shaped rash", "scaling", "itching"),
    "urinary tract": ("burning urination", "urinary urgency", "cloudy urine"),
    "stone": ("severe flank pain", "blood in urine", "pain radiating to groin"),
    "pyelonephritis": ("fever", "flank pain", "back pain"),
    "prostatitis": ("pelvic pain", "painful urination", "fever"),
    "vaginitis": ("vaginal itching", "vaginal discharge", "irritation"),
    "pelvic inflammatory": ("pelvic pain", "fever", "abnormal discharge"),
    "dysmenorrhea": ("menstrual cramps", "lower abdominal pain", "back pain"),
    "menopause": ("hot flashes", "night sweats", "sleep disturbance"),
    "pregnancy nausea": ("morning nausea", "vomiting", "pregnancy"),
    "bacterial vaginosis": ("thin discharge", "fishy odor", "vaginal irritation"),
    "candidiasis": ("thick white discharge", "vaginal itching", "burning"),
    "endometriosis": ("pelvic pain", "painful periods", "pain with intercourse"),
    "polycystic": ("irregular periods", "acne", "excess hair growth"),
    "erectile": ("difficulty maintaining erection", "sexual performance concern", "stress"),
    "prostatic": ("weak urine stream", "night urination", "hesitancy"),
    "testicular": ("testicular pain", "scrotal swelling", "nausea"),
    "acne": ("pimples", "blackheads", "oily skin"),
    "eczema": ("dry itchy skin", "scaling", "skin cracks"),
    "psoriasis": ("silvery scales", "thick plaques", "itching"),
    "urticaria": ("hives", "itching", "welts"),
    "cellulitis": ("warm red skin", "skin swelling", "fever"),
    "impetigo": ("honey colored crust", "skin sores", "itching"),
    "shingles": ("painful blisters", "one sided rash", "burning skin pain"),
    "herpes": ("painful grouped blisters", "tingling before rash", "recurrent sores"),
    "contact": ("rash after exposure", "itching", "redness"),
    "sunburn": ("red painful skin", "sun exposure", "peeling skin"),
    "burn": ("burn pain", "blistering", "skin redness"),
    "insect": ("bite mark", "localized swelling", "itching"),
    "lice": ("scalp itching", "visible nits", "scratch marks"),
    "rosacea": ("facial redness", "flushing", "visible facial vessels"),
    "seborrheic": ("dandruff", "greasy scaling", "itchy scalp"),
    "abscess": ("pus collection", "painful lump", "warm skin"),
    "conjunctivitis": ("red eye", "eye discharge", "gritty feeling"),
    "dry eye": ("dry eyes", "burning eyes", "foreign body sensation"),
    "glaucoma": ("severe eye pain", "halos around lights", "vision loss"),
    "cataract": ("cloudy vision", "glare", "slow vision change"),
    "stye": ("eyelid lump", "eyelid pain", "localized swelling"),
    "wax": ("ear fullness", "reduced hearing", "ear pressure"),
    "externa": ("ear canal pain", "pain pulling ear", "ear discharge"),
    "tinnitus": ("ringing ears", "hearing change", "sleep disturbance"),
    "hearing": ("reduced hearing", "asking repetition", "ear fullness"),
    "dental": ("tooth pain", "sensitivity to sweets", "visible cavity"),
    "gingivitis": ("bleeding gums", "gum swelling", "bad breath"),
    "mouth ulcer": ("painful mouth sore", "burning mouth pain", "difficulty eating"),
    "oral thrush": ("white mouth patches", "burning tongue", "altered taste"),
    "nosebleed": ("bleeding from nose", "nasal dryness", "recent nose picking"),
    "motion": ("nausea with travel", "dizziness with travel", "vomiting"),
    "hand foot": ("mouth sores", "hand rash", "foot rash"),
    "croup": ("barking cough", "hoarse voice", "stridor"),
    "febrile seizure": ("seizure with fever", "brief shaking episode", "post seizure sleepiness"),
    "diaper": ("diaper area rash", "skin redness", "irritability"),
    "colic": ("prolonged crying", "evening fussiness", "drawing legs up"),
    "teething": ("gum swelling", "drooling", "chewing objects"),
    "pinworm": ("anal itching at night", "restless sleep", "visible tiny worms"),
    "growth pain": ("leg pain at night", "normal daytime activity", "bilateral leg ache"),
    "bedwetting": ("nighttime urination", "wet bed", "deep sleep"),
    "stroke": ("one sided weakness", "face droop", "slurred speech"),
    "heart attack": ("crushing chest pain", "left arm pain", "sweating"),
    "sepsis": ("high fever", "confusion", "rapid breathing"),
    "allergic reaction": ("facial swelling", "wheezing", "hives"),
    "dehydration": ("very low urine", "dry mouth", "sunken eyes"),
    "meningitis": ("neck stiffness", "fever", "light sensitivity"),
    "ketoacidosis": ("fruity breath", "rapid breathing", "vomiting"),
    "hypoglycemia": ("sweating", "shaking", "confusion"),
    "heat stroke": ("high body temperature", "confusion", "hot dry skin"),
    "fracture": ("bone deformity", "severe limb pain", "swelling after injury"),
    "deep vein": ("calf swelling", "calf pain", "warm leg"),
    "pulmonary embolism": ("sudden shortness of breath", "chest pain", "coughing blood"),
    "bleeding": ("active bleeding", "dizziness", "pale skin"),
    "acute abdomen": ("severe abdominal pain", "guarding", "vomiting"),
}


MEDICINE_ROWS = [
    ("Acetaminophen", "Pain and fever relief", 0, 0.74),
    ("Ibuprofen", "NSAID pain and inflammation relief", 0, 0.68),
    ("Naproxen", "NSAID pain and inflammation relief", 0, 0.66),
    ("Aspirin", "NSAID antiplatelet and pain relief", 0, 0.62),
    ("Diclofenac Gel", "Topical NSAID pain relief", 0, 0.58),
    ("Celecoxib", "NSAID pain and inflammation relief", 1, 0.64),
    ("Ketorolac", "NSAID pain and inflammation relief", 1, 0.62),
    ("Mefenamic Acid", "NSAID pain and menstrual cramp relief", 1, 0.61),
    ("Topical Lidocaine", "Local pain relief", 0, 0.55),
    ("Capsaicin Cream", "Topical pain relief", 0, 0.50),
    ("Cetirizine", "Antihistamine", 0, 0.72),
    ("Loratadine", "Antihistamine", 0, 0.70),
    ("Fexofenadine", "Antihistamine", 0, 0.70),
    ("Diphenhydramine", "Sedating antihistamine", 0, 0.58),
    ("Chlorpheniramine", "Sedating antihistamine", 0, 0.56),
    ("Fluticasone Nasal Spray", "Nasal corticosteroid", 0, 0.68),
    ("Saline Nasal Spray", "Nasal supportive care", 0, 0.46),
    ("Oxymetazoline Nasal Spray", "Topical nasal decongestant", 0, 0.52),
    ("Pseudoephedrine", "Decongestant", 0, 0.55),
    ("Phenylephrine", "Decongestant", 0, 0.42),
    ("Dextromethorphan", "Cough suppressant", 0, 0.50),
    ("Guaifenesin", "Expectorant", 0, 0.50),
    ("Throat Lozenges", "Throat supportive care", 0, 0.42),
    ("Salbutamol Inhaler", "Respiratory bronchodilator", 1, 0.82),
    ("Ipratropium Inhaler", "Respiratory bronchodilator", 1, 0.66),
    ("Budesonide Inhaler", "Inhaled corticosteroid", 1, 0.70),
    ("Fluticasone Inhaler", "Inhaled corticosteroid", 1, 0.70),
    ("Montelukast", "Respiratory leukotriene modifier", 1, 0.62),
    ("Prednisone", "Systemic corticosteroid", 1, 0.62),
    ("Tiotropium Inhaler", "Respiratory bronchodilator", 1, 0.66),
    ("Oral Rehydration Salts", "Hydration support", 0, 0.76),
    ("Antacid", "Gastrointestinal therapy", 0, 0.64),
    ("Omeprazole", "Proton pump inhibitor", 0, 0.69),
    ("Pantoprazole", "Proton pump inhibitor", 1, 0.68),
    ("Famotidine", "H2 blocker acid relief", 0, 0.62),
    ("Sucralfate", "Gastrointestinal mucosal protectant", 1, 0.56),
    ("Ondansetron", "Antiemetic", 1, 0.70),
    ("Metoclopramide", "Antiemetic and prokinetic", 1, 0.54),
    ("Loperamide", "Antidiarrheal", 0, 0.56),
    ("Psyllium Fiber", "Constipation support", 0, 0.52),
    ("Polyethylene Glycol", "Osmotic laxative", 0, 0.58),
    ("Lactulose", "Osmotic laxative", 1, 0.55),
    ("Simethicone", "Gas relief", 0, 0.42),
    ("Bismuth Subsalicylate", "Gastrointestinal therapy", 0, 0.48),
    ("Hyoscine Butylbromide", "Antispasmodic", 1, 0.50),
    ("Nitrofurantoin", "Antibiotic", 1, 0.78),
    ("Amoxicillin-Clavulanate", "Penicillin antibiotic", 1, 0.76),
    ("Azithromycin", "Macrolide antibiotic", 1, 0.66),
    ("Doxycycline", "Tetracycline antibiotic", 1, 0.66),
    ("Cephalexin", "Cephalosporin antibiotic", 1, 0.66),
    ("Cefixime", "Cephalosporin antibiotic", 1, 0.64),
    ("Ceftriaxone", "Cephalosporin antibiotic", 1, 0.74),
    ("Ciprofloxacin", "Fluoroquinolone antibiotic", 1, 0.64),
    ("Levofloxacin", "Fluoroquinolone antibiotic", 1, 0.64),
    ("Metronidazole", "Antibiotic and antiprotozoal", 1, 0.68),
    ("Clindamycin", "Antibiotic", 1, 0.60),
    ("Trimethoprim-Sulfamethoxazole", "Sulfonamide antibiotic", 1, 0.66),
    ("Fosfomycin", "Antibiotic", 1, 0.64),
    ("Amoxicillin", "Penicillin antibiotic", 1, 0.70),
    ("Penicillin V", "Penicillin antibiotic", 1, 0.62),
    ("Clarithromycin", "Macrolide antibiotic", 1, 0.62),
    ("Erythromycin", "Macrolide antibiotic", 1, 0.58),
    ("Vancomycin", "Antibiotic", 1, 0.70),
    ("Meropenem", "Antibiotic", 1, 0.72),
    ("Linezolid", "Antibiotic", 1, 0.68),
    ("Oseltamivir", "Antiviral", 1, 0.70),
    ("Acyclovir", "Antiviral", 1, 0.68),
    ("Valacyclovir", "Antiviral", 1, 0.70),
    ("Nirmatrelvir-Ritonavir", "Antiviral", 1, 0.76),
    ("Remdesivir", "Antiviral", 1, 0.74),
    ("Baloxavir", "Antiviral", 1, 0.68),
    ("Zanamivir", "Antiviral", 1, 0.64),
    ("Tenofovir", "Antiviral", 1, 0.62),
    ("Sofosbuvir", "Antiviral", 1, 0.70),
    ("Entecavir", "Antiviral", 1, 0.62),
    ("Fluconazole", "Antifungal", 1, 0.68),
    ("Clotrimazole Cream", "Topical antifungal", 0, 0.60),
    ("Terbinafine Cream", "Topical antifungal", 0, 0.62),
    ("Nystatin", "Antifungal", 1, 0.60),
    ("Griseofulvin", "Antifungal", 1, 0.56),
    ("Itraconazole", "Antifungal", 1, 0.62),
    ("Artemisinin-Based Therapy", "Antimalarial", 1, 0.84),
    ("Chloroquine", "Antimalarial", 1, 0.54),
    ("Primaquine", "Antimalarial", 1, 0.58),
    ("Quinine", "Antimalarial", 1, 0.56),
    ("Albendazole", "Antiparasitic", 1, 0.62),
    ("Mebendazole", "Antiparasitic", 1, 0.60),
    ("Ivermectin", "Antiparasitic", 1, 0.58),
    ("Permethrin Cream", "Topical antiparasitic", 0, 0.66),
    ("Amlodipine", "Cardiovascular therapy", 1, 0.72),
    ("Lisinopril", "ACE inhibitor cardiovascular therapy", 1, 0.72),
    ("Losartan", "ARB cardiovascular therapy", 1, 0.72),
    ("Hydrochlorothiazide", "Diuretic cardiovascular therapy", 1, 0.62),
    ("Furosemide", "Diuretic cardiovascular therapy", 1, 0.66),
    ("Spironolactone", "Diuretic cardiovascular therapy", 1, 0.64),
    ("Atenolol", "Beta blocker cardiovascular therapy", 1, 0.60),
    ("Metoprolol", "Beta blocker cardiovascular therapy", 1, 0.66),
    ("Carvedilol", "Beta blocker cardiovascular therapy", 1, 0.64),
    ("Atorvastatin", "Lipid lowering therapy", 1, 0.70),
    ("Rosuvastatin", "Lipid lowering therapy", 1, 0.70),
    ("Simvastatin", "Lipid lowering therapy", 1, 0.62),
    ("Aspirin Low Dose", "Antiplatelet therapy", 1, 0.62),
    ("Clopidogrel", "Antiplatelet therapy", 1, 0.66),
    ("Warfarin", "Anticoagulant", 1, 0.66),
    ("Apixaban", "Anticoagulant", 1, 0.70),
    ("Nitroglycerin", "Antianginal cardiovascular therapy", 1, 0.70),
    ("Isosorbide Mononitrate", "Antianginal cardiovascular therapy", 1, 0.62),
    ("Digoxin", "Cardiovascular therapy", 1, 0.55),
    ("Amiodarone", "Antiarrhythmic cardiovascular therapy", 1, 0.62),
    ("Metformin", "Diabetes therapy", 1, 0.73),
    ("Glipizide", "Diabetes therapy", 1, 0.58),
    ("Gliclazide", "Diabetes therapy", 1, 0.58),
    ("Insulin Glargine", "Diabetes therapy", 1, 0.78),
    ("Regular Insulin", "Diabetes therapy", 1, 0.78),
    ("Sitagliptin", "Diabetes therapy", 1, 0.58),
    ("Empagliflozin", "Diabetes therapy", 1, 0.62),
    ("Semaglutide", "Diabetes therapy", 1, 0.66),
    ("Levothyroxine", "Endocrine therapy", 1, 0.72),
    ("Methimazole", "Endocrine therapy", 1, 0.66),
    ("Propylthiouracil", "Endocrine therapy", 1, 0.60),
    ("Vitamin D3", "Vitamin and mineral", 0, 0.56),
    ("Calcium Carbonate", "Vitamin and mineral", 0, 0.52),
    ("Ferrous Sulfate", "Vitamin and mineral", 0, 0.60),
    ("Folic Acid", "Vitamin and mineral", 0, 0.58),
    ("Cyanocobalamin", "Vitamin and mineral", 0, 0.58),
    ("Potassium Chloride", "Electrolyte replacement", 1, 0.58),
    ("Magnesium Supplement", "Vitamin and mineral", 0, 0.48),
    ("Allopurinol", "Gout therapy", 1, 0.64),
    ("Colchicine", "Gout therapy", 1, 0.60),
    ("Sumatriptan", "Migraine-specific therapy", 1, 0.76),
    ("Propranolol", "Beta blocker neurology therapy", 1, 0.62),
    ("Topiramate", "Neurology therapy", 1, 0.62),
    ("Valproate", "Neurology therapy", 1, 0.62),
    ("Carbamazepine", "Neurology therapy", 1, 0.62),
    ("Levetiracetam", "Neurology therapy", 1, 0.68),
    ("Gabapentin", "Neurology pain therapy", 1, 0.62),
    ("Pregabalin", "Neurology pain therapy", 1, 0.62),
    ("Amitriptyline", "Neurology and mental health therapy", 1, 0.56),
    ("Sertraline", "Mental health therapy", 1, 0.62),
    ("Fluoxetine", "Mental health therapy", 1, 0.62),
    ("Escitalopram", "Mental health therapy", 1, 0.62),
    ("Duloxetine", "Mental health and pain therapy", 1, 0.60),
    ("Melatonin", "Sleep support", 0, 0.48),
    ("Meclizine", "Vertigo and motion sickness therapy", 0, 0.56),
    ("Betahistine", "Vertigo therapy", 1, 0.50),
    ("Diazepam", "Neurology and emergency therapy", 1, 0.52),
    ("Hydroxyzine", "Sedating antihistamine", 1, 0.52),
    ("Trazodone", "Mental health and sleep therapy", 1, 0.54),
    ("Hydrocortisone Cream", "Dermatology corticosteroid", 0, 0.58),
    ("Betamethasone Cream", "Dermatology corticosteroid", 1, 0.60),
    ("Mupirocin Ointment", "Topical antibiotic", 1, 0.64),
    ("Fusidic Acid Cream", "Topical antibiotic", 1, 0.58),
    ("Benzoyl Peroxide", "Acne therapy", 0, 0.56),
    ("Adapalene Gel", "Acne therapy", 0, 0.58),
    ("Calamine Lotion", "Dermatology supportive care", 0, 0.48),
    ("Zinc Oxide Cream", "Dermatology barrier therapy", 0, 0.50),
    ("Silver Sulfadiazine Cream", "Topical sulfonamide antimicrobial", 1, 0.60),
    ("Acyclovir Cream", "Topical antiviral", 0, 0.52),
    ("Ketoconazole Shampoo", "Topical antifungal", 0, 0.56),
    ("Selenium Sulfide Shampoo", "Dermatology antifungal support", 0, 0.50),
    ("Artificial Tears", "Eye and ear therapy", 0, 0.52),
    ("Olopatadine Eye Drops", "Eye antihistamine", 0, 0.56),
    ("Ciprofloxacin Eye Drops", "Eye antibiotic", 1, 0.62),
    ("Chloramphenicol Eye Drops", "Eye antibiotic", 1, 0.58),
    ("Timolol Eye Drops", "Eye beta blocker", 1, 0.58),
    ("Carbamide Peroxide Ear Drops", "Ear wax therapy", 0, 0.48),
    ("Ofloxacin Ear Drops", "Ear antibiotic", 1, 0.60),
    ("Phenazopyridine", "Urinary analgesic", 0, 0.48),
    ("Tamsulosin", "Urology therapy", 1, 0.62),
    ("Finasteride", "Urology therapy", 1, 0.58),
    ("Clotrimazole Vaginal", "Gynecology antifungal", 0, 0.60),
    ("Metronidazole Vaginal Gel", "Gynecology antimicrobial", 1, 0.62),
    ("Tranexamic Acid", "Bleeding control therapy", 1, 0.58),
    ("Combined Oral Contraceptive", "Reproductive hormone therapy", 1, 0.56),
    ("Progesterone", "Reproductive hormone therapy", 1, 0.56),
    ("Sildenafil", "Urology therapy", 1, 0.58),
    ("Epinephrine Auto-Injector", "Emergency and supportive care", 1, 0.86),
    ("Glucagon Rescue Kit", "Emergency and supportive care", 1, 0.78),
    ("Activated Charcoal", "Emergency and supportive care", 1, 0.50),
    ("Naloxone Spray", "Emergency and supportive care", 1, 0.78),
    ("Oral Glucose Gel", "Emergency and supportive care", 0, 0.72),
    ("Normal Saline", "Hydration support", 1, 0.70),
    ("Ringer Lactate", "Hydration support", 1, 0.70),
    ("Oxygen Therapy", "Emergency and supportive care", 1, 0.82),
    ("Tetanus Vaccine", "Vaccine and immunoglobulin", 1, 0.74),
    ("Rabies Vaccine", "Vaccine and immunoglobulin", 1, 0.80),
    ("Rabies Immunoglobulin", "Vaccine and immunoglobulin", 1, 0.82),
    ("Measles Vaccine", "Vaccine and immunoglobulin", 1, 0.74),
    ("Varicella Vaccine", "Vaccine and immunoglobulin", 1, 0.72),
    ("Wound Cleaning Solution", "Emergency and supportive care", 0, 0.46),
    ("Sterile Dressing", "Emergency and supportive care", 0, 0.44),
    ("Elastic Bandage", "Emergency and supportive care", 0, 0.42),
    ("Eye Wash Solution", "Emergency and supportive care", 0, 0.44),
]


GROUP_MEDICINE_POOLS = {
    "respiratory": ["Acetaminophen", "Saline Nasal Spray", "Cetirizine", "Dextromethorphan", "Guaifenesin", "Salbutamol Inhaler", "Budesonide Inhaler", "Oseltamivir", "Azithromycin"],
    "gastro": ["Oral Rehydration Salts", "Omeprazole", "Famotidine", "Antacid", "Ondansetron", "Polyethylene Glycol", "Loperamide", "Metronidazole", "Hyoscine Butylbromide"],
    "cardiometabolic": ["Amlodipine", "Metformin", "Atorvastatin", "Lisinopril", "Losartan", "Furosemide", "Insulin Glargine", "Levothyroxine", "Ferrous Sulfate"],
    "neuro_pain": ["Acetaminophen", "Topical Lidocaine", "Sumatriptan", "Gabapentin", "Pregabalin", "Meclizine", "Sertraline", "Melatonin", "Propranolol"],
    "infectious_tropical": ["Acetaminophen", "Oral Rehydration Salts", "Artemisinin-Based Therapy", "Doxycycline", "Ceftriaxone", "Acyclovir", "Permethrin Cream", "Albendazole", "Rabies Vaccine"],
    "urinary_reproductive": ["Nitrofurantoin", "Fosfomycin", "Phenazopyridine", "Tamsulosin", "Clotrimazole Vaginal", "Metronidazole Vaginal Gel", "Ondansetron", "Mefenamic Acid", "Progesterone"],
    "skin_allergy": ["Hydrocortisone Cream", "Cetirizine", "Mupirocin Ointment", "Clotrimazole Cream", "Permethrin Cream", "Calamine Lotion", "Acyclovir", "Benzoyl Peroxide", "Betamethasone Cream"],
    "eye_ent": ["Artificial Tears", "Olopatadine Eye Drops", "Ciprofloxacin Eye Drops", "Carbamide Peroxide Ear Drops", "Ofloxacin Ear Drops", "Acetaminophen", "Amoxicillin-Clavulanate", "Chlorhexidine Mouthwash", "Meclizine"],
    "pediatric_general": ["Acetaminophen", "Oral Rehydration Salts", "Zinc Oxide Cream", "Permethrin Cream", "Mebendazole", "Salbutamol Inhaler", "Ferrous Sulfate", "Melatonin", "Medical Referral"],
    "emergency_redflag": ["Oxygen Therapy", "Normal Saline", "Ringer Lactate", "Epinephrine Auto-Injector", "Oral Glucose Gel", "Glucagon Rescue Kit", "Activated Charcoal", "Sterile Dressing", "Medical Referral"],
}


ADDITIONAL_SUPPORT_ITEMS = [
    ("Chlorhexidine Mouthwash", "Dental antiseptic", 0, 0.46),
    ("Medical Referral", "Emergency and supportive care", 0, 0.95),
]


SPECIAL_DISEASE_MEDICINES = {
    "Dengue Fever": [
        ("Oral Rehydration Salts", 0.86, "Hydration support; warning signs require urgent care."),
        ("Acetaminophen", 0.70, "Fever relief when label-appropriate; avoid NSAIDs unless a clinician says otherwise."),
        ("Medical Referral", 0.95, "Dengue warning signs should be assessed by a healthcare professional."),
    ],
    "Asthma": [
        ("Salbutamol Inhaler", 0.88, "Rescue inhaler therapy within a clinician action plan."),
        ("Budesonide Inhaler", 0.70, "Controller therapy requires clinician review."),
        ("Medical Referral", 0.65, "Breathing symptoms may need urgent assessment."),
    ],
    "Urinary Tract Infection": [
        ("Nitrofurantoin", 0.84, "Prescription antibiotic option for selected uncomplicated UTI."),
        ("Fosfomycin", 0.72, "Prescription antibiotic option after clinician evaluation."),
        ("Oral Rehydration Salts", 0.35, "Hydration support while seeking care."),
    ],
    "Influenza": [
        ("Oseltamivir", 0.82, "Clinician-directed antiviral option for selected patients."),
        ("Acetaminophen", 0.62, "Fever and body ache relief when label-appropriate."),
        ("Oral Rehydration Salts", 0.40, "Hydration support during fever."),
    ],
    "Common Cold": [
        ("Acetaminophen", 0.55, "Fever or aches relief when label-appropriate."),
        ("Saline Nasal Spray", 0.52, "Supportive nasal symptom relief."),
        ("Cetirizine", 0.44, "Can help runny nose and sneezing when allergy overlap is present."),
    ],
    "Pneumonia": [
        ("Amoxicillin-Clavulanate", 0.80, "Prescription antibiotic option when bacterial pneumonia is suspected."),
        ("Azithromycin", 0.64, "Prescription antibiotic option in selected cases."),
        ("Medical Referral", 0.90, "Pneumonia symptoms should be reviewed promptly."),
    ],
    "Migraine": [
        ("Sumatriptan", 0.82, "Migraine-specific prescription therapy when clinically suitable."),
        ("Acetaminophen", 0.56, "Pain relief option when label-appropriate."),
        ("Topical Lidocaine", 0.32, "Supportive local comfort option."),
    ],
    "Hypertension": [
        ("Amlodipine", 0.82, "Prescription therapy after blood pressure confirmation."),
        ("Losartan", 0.70, "Prescription blood pressure option requiring clinician monitoring."),
        ("Medical Referral", 0.65, "Blood pressure treatment needs measurement confirmation."),
    ],
    "Type 2 Diabetes": [
        ("Metformin", 0.82, "Prescription therapy after diagnostic confirmation and kidney review."),
        ("Insulin Glargine", 0.74, "Insulin therapy requires clinician management."),
        ("Medical Referral", 0.70, "Diabetes symptoms need diagnostic testing and monitoring."),
    ],
}


def _unique(items: tuple[str, ...] | list[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for item in items if item))


def _specific_symptoms(name: str) -> tuple[str, ...]:
    lower = name.lower()
    symptoms: list[str] = []
    for keyword, values in SPECIFIC_SYMPTOMS_BY_KEYWORD.items():
        if keyword in lower:
            symptoms.extend(values)
    if not symptoms:
        words = [word for word in lower.replace("-", " ").split() if len(word) > 3]
        symptoms.extend(f"{word} related symptom" for word in words[:2])
    return _unique(symptoms)


def build_disease_profiles() -> list[DiseaseProfile]:
    profiles: list[DiseaseProfile] = []
    for group, names in DISEASE_GROUPS.items():
        template = GROUP_TEMPLATES[group]
        for name in names:
            specific = _specific_symptoms(name)
            core = _unique(specific[:3] + template["core"])
            supporting = _unique(specific[3:] + template["supporting"])
            red_flags = _unique(template["red_flags"])
            profiles.append(
                DiseaseProfile(
                    name=name,
                    group=group,
                    core_symptoms=core,
                    supporting_symptoms=supporting,
                    red_flag_symptoms=red_flags,
                    age_weights=template["age"],
                    severity_weights=template["severity"],
                )
            )
    return profiles


def build_diseases() -> dict[str, dict]:
    diseases = {}
    for profile in build_disease_profiles():
        group_label = profile.group.replace("_", " ")
        diseases[profile.name] = {
            "description": (
                f"{profile.name} educational profile with {group_label} symptoms. "
                "Recommendations are decision-support only and require clinical judgment."
            ),
            "severity": max(profile.severity_weights, key=profile.severity_weights.get),
            "flags": list(_unique(profile.red_flag_symptoms)),
        }
    return diseases


def _medicine_notes(name: str, category: str, rx: int) -> str:
    rx_note = "Prescription-only; clinician evaluation and monitoring are required." if rx else "Follow the product label and seek clinician advice when symptoms persist."
    if "Antibiotic" in category or "antibiotic" in category:
        return "Use only for clinician-confirmed bacterial indications; inappropriate use can cause harm and resistance."
    if "NSAID" in category:
        return "Avoid with bleeding risk, stomach ulcer, severe kidney disease, dengue concern, or clinician-advised NSAID restriction."
    if "Anticoagulant" in category or "Antiplatelet" in category:
        return "Bleeding risk must be reviewed by a clinician before use."
    if "Emergency" in category or name == "Medical Referral":
        return "Urgent or supervised care item; use according to emergency or clinician guidance."
    return rx_note


def build_medicines() -> dict[str, dict]:
    rows = MEDICINE_ROWS + ADDITIONAL_SUPPORT_ITEMS
    medicines = {}
    for name, category, rx, score in rows:
        medicines[name] = {
            "category": category,
            "rx": int(rx),
            "score": float(score),
            "notes": _medicine_notes(name, category, int(rx)),
        }
    return medicines


def build_disease_medicine_map() -> dict[str, list[tuple[str, float, str]]]:
    mapping = {}
    for profile in build_disease_profiles():
        if profile.name in SPECIAL_DISEASE_MEDICINES:
            mapping[profile.name] = SPECIAL_DISEASE_MEDICINES[profile.name]
            continue
        pool = GROUP_MEDICINE_POOLS[profile.group]
        selected = pool[:4]
        mapping[profile.name] = [
            (
                medicine,
                round(0.78 - index * 0.08, 2),
                "Educational candidate based on symptom group, relevance, and safety screening.",
            )
            for index, medicine in enumerate(selected)
        ]
    return mapping


def _rule(condition: str, severity: str, reason: str) -> tuple[str, str, str]:
    return (condition, severity, reason)


def _rules_for_medicine(name: str, category: str, rx: int) -> list[tuple[str, str, str]]:
    lower_name = name.lower()
    lower_category = category.lower()
    rules: list[tuple[str, str, str]] = []

    if rx:
        rules.append(_rule("pregnancy", "medium", "Prescription medicines in pregnancy require clinician review."))

    if "nsaid" in lower_category or lower_name in {"ibuprofen", "naproxen", "aspirin", "ketorolac", "celecoxib"}:
        rules.extend(
            [
                _rule("kidney disease", "high", "NSAIDs can worsen kidney function."),
                _rule("gastritis", "medium", "NSAIDs can worsen stomach irritation."),
                _rule("stomach ulcer", "high", "NSAIDs can increase gastrointestinal bleeding risk."),
                _rule("anticoagulant use", "high", "Combined bleeding risk can be dangerous."),
                _rule("dengue fever", "high", "Dengue can increase bleeding risk; NSAIDs are usually avoided."),
                _rule("hypertension", "medium", "NSAIDs can worsen blood pressure control."),
            ]
        )

    if name == "Acetaminophen":
        rules.append(_rule("liver disease", "high", "Higher risk of liver injury; clinician guidance is required."))

    if "antibiotic" in lower_category or "antimicrobial" in lower_category:
        rules.append(_rule("antibiotic allergy", "high", "Possible allergy risk; clinician review is required."))
        if "penicillin" in lower_category or "amoxicillin" in lower_name:
            rules.append(_rule("penicillin allergy", "high", "Avoid when penicillin allergy is present."))
        if "sulfonamide" in lower_category or "sulf" in lower_name:
            rules.append(_rule("sulfa allergy", "high", "Avoid when sulfonamide allergy is present."))
        if "macrolide" in lower_category:
            rules.append(_rule("heart rhythm disorder", "medium", "Macrolides may be unsuitable with some rhythm risks."))
        if "fluoroquinolone" in lower_category:
            rules.extend(
                [
                    _rule("tendon disorder", "medium", "Fluoroquinolones can worsen tendon problems."),
                    _rule("heart rhythm disorder", "medium", "Rhythm risk should be reviewed before use."),
                ]
            )

    if "antiviral" in lower_category or "antimalarial" in lower_category or "antiparasitic" in lower_category:
        rules.append(_rule("kidney disease", "medium", "Dose choice and suitability may depend on kidney function."))

    if "antifungal" in lower_category:
        rules.extend(
            [
                _rule("liver disease", "high", "Some antifungals can worsen liver risk."),
                _rule("pregnancy", "medium", "Antifungal selection in pregnancy requires clinician review."),
            ]
        )

    if "decongestant" in lower_category:
        rules.extend(
            [
                _rule("hypertension", "high", "Decongestants can raise blood pressure."),
                _rule("heart disease", "medium", "Decongestants may worsen heart symptoms."),
                _rule("glaucoma", "medium", "Some decongestants can worsen glaucoma risk."),
            ]
        )

    if "sedating antihistamine" in lower_category:
        rules.extend(
            [
                _rule("older adult", "medium", "Sedating medicines can increase fall or confusion risk."),
                _rule("glaucoma", "medium", "Anticholinergic effects may worsen glaucoma."),
                _rule("prostate enlargement", "medium", "May worsen urinary retention."),
            ]
        )

    if "beta blocker" in lower_category or name in {"Propranolol", "Atenolol", "Metoprolol", "Carvedilol", "Timolol Eye Drops"}:
        rules.extend(
            [
                _rule("asthma", "medium", "Beta blockers can worsen bronchospasm in some patients."),
                _rule("bradycardia", "high", "Can worsen slow heart rate."),
            ]
        )

    if "ace inhibitor" in lower_category or "arb" in lower_category:
        rules.extend(
            [
                _rule("pregnancy", "high", "ACE inhibitor or ARB use in pregnancy can be unsafe."),
                _rule("kidney disease", "medium", "Kidney function and potassium require monitoring."),
            ]
        )

    if "diuretic" in lower_category:
        rules.extend(
            [
                _rule("kidney disease", "medium", "Fluid and electrolyte monitoring is required."),
                _rule("electrolyte imbalance", "medium", "May worsen electrolyte abnormalities."),
            ]
        )

    if "diabetes therapy" in lower_category:
        rules.append(_rule("kidney disease", "medium", "Diabetes medicine selection may depend on kidney function."))
        if "insulin" in lower_name or "glipizide" in lower_name or "gliclazide" in lower_name:
            rules.append(_rule("hypoglycemia", "high", "Can worsen low blood sugar risk."))

    if "anticoagulant" in lower_category or "antiplatelet" in lower_category or "bleeding control" in lower_category:
        rules.extend(
            [
                _rule("bleeding risk", "high", "Bleeding risk must be reviewed before use."),
                _rule("anticoagulant use", "high", "Combined blood-thinning effects can be dangerous."),
                _rule("dengue fever", "high", "Dengue can increase bleeding risk."),
            ]
        )

    if "corticosteroid" in lower_category or name == "Prednisone":
        rules.extend(
            [
                _rule("diabetes", "medium", "Steroids can increase blood sugar."),
                _rule("active infection", "medium", "Steroids can mask or worsen some infections."),
            ]
        )

    if "migraine" in lower_category or name == "Sumatriptan":
        rules.extend(
            [
                _rule("hypertension", "high", "Can be unsafe with uncontrolled blood pressure."),
                _rule("heart disease", "high", "Cardiovascular suitability must be reviewed."),
            ]
        )

    if "neurology therapy" in lower_category or name in {"Valproate", "Carbamazepine", "Topiramate"}:
        rules.append(_rule("pregnancy", "high", "Some neurology medicines can be unsafe in pregnancy."))

    if "emergency" in lower_category or "vaccine" in lower_category:
        rules.append(_rule("severe allergic reaction", "medium", "Prior severe reaction requires supervised care."))

    return list(dict.fromkeys(rules))


def build_contraindications() -> dict[str, list[tuple[str, str, str]]]:
    return {
        name: _rules_for_medicine(name, payload["category"], payload["rx"])
        for name, payload in build_medicines().items()
    }
