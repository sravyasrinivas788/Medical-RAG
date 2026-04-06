import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

def get_conn():
    return psycopg2.connect(os.getenv("POSTGRES_URL"))

def setup_tables():
    conn = get_conn()
    cur  = conn.cursor()

    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id          SERIAL PRIMARY KEY,
            name        TEXT NOT NULL,
            file_type   TEXT NOT NULL,
            content     BYTEA NOT NULL,
            uploaded_at TIMESTAMP DEFAULT NOW()
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS drugs (
            id               SERIAL PRIMARY KEY,
            name             TEXT NOT NULL,
            category         TEXT,
            indication       TEXT,
            dosage           TEXT,
            contraindications TEXT,
            side_effects     TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS clinic_policies (
            id          SERIAL PRIMARY KEY,
            topic       TEXT NOT NULL,
            description TEXT NOT NULL
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS lab_ranges (
            id          SERIAL PRIMARY KEY,
            test_name   TEXT NOT NULL,
            normal_range TEXT NOT NULL,
            unit        TEXT,
            notes       TEXT
        );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Tables created.")

def seed_dummy_data():
    conn = get_conn()
    cur  = conn.cursor()

    # check if already seeded
    cur.execute("SELECT COUNT(*) FROM drugs")
    if cur.fetchone()[0] > 0:
        print("Data already seeded, skipping.")
        cur.close()
        conn.close()
        return

    # --- drugs ---
    drugs = [
        ("Metformin", "Antidiabetic", "Type 2 diabetes management",
         "500mg–2000mg daily in divided doses with meals",
         "Renal impairment (eGFR < 30), hypersensitivity, metabolic acidosis",
         "Nausea, diarrhea, abdominal discomfort, lactic acidosis (rare)"),

        ("Ibuprofen", "NSAID", "Pain, inflammation, fever",
         "200mg–400mg every 4–6 hours, max 1200mg/day OTC",
         "Active GI bleeding, severe renal impairment, concomitant warfarin use",
         "GI upset, headache, dizziness, increased bleeding risk"),

        ("Amoxicillin", "Antibiotic", "Bacterial infections — respiratory, UTI, skin",
         "250mg–500mg every 8 hours for 5–10 days",
         "Penicillin allergy, mononucleosis",
         "Rash, diarrhea, nausea, allergic reaction"),

        ("Warfarin", "Anticoagulant", "DVT, pulmonary embolism, atrial fibrillation",
         "Individualized based on INR — typically 2mg–10mg daily",
         "Active bleeding, pregnancy, severe liver disease",
         "Bleeding, bruising, hair loss — requires regular INR monitoring"),

        ("Atorvastatin", "Statin", "High cholesterol, cardiovascular disease prevention",
         "10mg–80mg once daily",
         "Active liver disease, pregnancy, concurrent strong CYP3A4 inhibitors",
         "Muscle pain, liver enzyme elevation, headache"),

        ("Amlodipine", "Calcium channel blocker", "Hypertension, angina",
         "5mg–10mg once daily",
         "Severe hypotension, cardiogenic shock",
         "Ankle swelling, flushing, palpitations, dizziness"),
    ]
    cur.executemany("""
        INSERT INTO drugs (name, category, indication, dosage, contraindications, side_effects)
        VALUES (%s,%s,%s,%s,%s,%s)
    """, drugs)

    # --- clinic policies ---
    policies = [
        ("Missed appointment",
         "Patients must cancel at least 24 hours in advance. Three missed appointments without notice result in suspension from non-emergency scheduling."),
        ("Prescription refill",
         "Prescription refills require a consultation if the last visit was more than 6 months ago. Controlled substances cannot be refilled without an in-person visit."),
        ("Emergency walk-in",
         "Walk-in patients with acute symptoms are triaged immediately. Non-emergency walk-ins are seen after scheduled patients."),
        ("Patient records request",
         "Medical records requests are processed within 5 business days. Patients must submit a signed release form."),
        ("Referral process",
         "Specialist referrals are initiated by the attending physician. Patients receive referral letters within 48 hours of the consultation."),
    ]
    cur.executemany("""
        INSERT INTO clinic_policies (topic, description) VALUES (%s,%s)
    """, policies)

    # --- lab ranges ---
    labs = [
        ("Fasting Blood Glucose", "70–99", "mg/dL",
         "100–125 mg/dL indicates prediabetes. 126+ mg/dL on two tests indicates diabetes."),
        ("HbA1c", "Below 5.7", "%",
         "5.7–6.4% prediabetes. 6.5%+ diabetes. Target for diabetics: below 7%."),
        ("Total Cholesterol", "Below 200", "mg/dL",
         "200–239 borderline high. 240+ high. LDL should be below 100 mg/dL ideally."),
        ("Systolic Blood Pressure", "Below 120", "mmHg",
         "120–129 elevated. 130–139 stage 1 hypertension. 140+ stage 2 hypertension."),
        ("Hemoglobin", "13.5–17.5 (male), 12–15.5 (female)", "g/dL",
         "Below normal indicates anemia. Above normal may indicate polycythemia."),
        ("Creatinine", "0.7–1.3 (male), 0.6–1.1 (female)", "mg/dL",
         "Elevated creatinine indicates reduced kidney function. Critical for Metformin dosing."),
        ("INR", "0.8–1.1 (normal), 2.0–3.0 (on warfarin)", "ratio",
         "INR must be monitored closely in warfarin patients. Above 4.0 is dangerous."),
    ]
    cur.executemany("""
        INSERT INTO lab_ranges (test_name, normal_range, unit, notes) VALUES (%s,%s,%s,%s)
    """, labs)

    conn.commit()
    cur.close()
    conn.close()
    print("Dummy data seeded.")

def save_file(name: str, file_type: str, content: bytes) -> int:
    conn = get_conn()
    cur  = conn.cursor()
    cur.execute(
        "INSERT INTO documents (name, file_type, content) VALUES (%s,%s,%s) RETURNING id",
        (name, file_type, content)
    )
    doc_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return doc_id

def get_all_files():
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT id, name, file_type, content FROM documents")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_all_drugs():
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM drugs")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_all_policies():
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM clinic_policies")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows

def get_all_lab_ranges():
    conn = get_conn()
    cur  = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM lab_ranges")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows