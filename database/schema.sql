PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS diseases (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  description TEXT NOT NULL,
  default_severity TEXT NOT NULL,
  emergency_flags TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS medicines (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL UNIQUE,
  category TEXT NOT NULL,
  prescription_required INTEGER NOT NULL DEFAULT 0,
  effectiveness_score REAL NOT NULL,
  safety_notes TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS disease_medicines (
  disease_id INTEGER NOT NULL,
  medicine_id INTEGER NOT NULL,
  relevance_score REAL NOT NULL,
  rationale TEXT NOT NULL,
  PRIMARY KEY (disease_id, medicine_id),
  FOREIGN KEY (disease_id) REFERENCES diseases(id) ON DELETE CASCADE,
  FOREIGN KEY (medicine_id) REFERENCES medicines(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS medicine_contraindications (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  medicine_id INTEGER NOT NULL,
  condition_key TEXT NOT NULL,
  severity TEXT NOT NULL,
  reason TEXT NOT NULL,
  FOREIGN KEY (medicine_id) REFERENCES medicines(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS training_cases (
  id INTEGER PRIMARY KEY,
  disease TEXT NOT NULL,
  symptoms TEXT NOT NULL,
  age_group TEXT NOT NULL,
  severity TEXT NOT NULL,
  source_note TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_logs (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  symptoms TEXT NOT NULL,
  age_group TEXT NOT NULL,
  profile_conditions TEXT NOT NULL,
  top_disease TEXT NOT NULL,
  model_confidence REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS recommendation_feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  symptoms TEXT NOT NULL,
  top_disease TEXT NOT NULL,
  medicine_name TEXT NOT NULL DEFAULT '',
  rating TEXT NOT NULL,
  irrelevant_medicine INTEGER NOT NULL DEFAULT 0,
  comments TEXT NOT NULL
);
