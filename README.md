# 🥍 University of Delaware Men's Lacrosse - Expected Goals (xG) Engine

#link = https://expected-goals-xyfvwsgt8qm3fqsfeecref.streamlit.app/

### 🎓 Internship Project Notice
This project was developed during my Data Science & Sports Analytics Internship for the **University of Delaware Men's Lacrosse Team**. 

⚠️ **Data Privacy:** Because this application is powered by confidential, proprietary scouting data, all raw datasets (`.csv`, `.xlsx`) have been strictly excluded from this public repository. This page serves as a technical portfolio to demonstrate the system's architecture, machine learning pipeline, and user interface.

---

## 🍼 The "Explain it Simply" Breakdown
Traditional sports stats often lie. If a player takes a terrible shot from 20 yards away, but it bounces off a defender's helmet and goes in, the stat sheet says "Great Job: 1 Goal." If a player takes a perfect, wide-open shot from 3 yards away, but the goalie makes a miracle save, the stat sheet says "Bad Job: 0 Goals." 

I built an Artificial Intelligence app to fix this. 
* **The Goal:** Grade the *decision* to shoot, not just the result.
* **How it Works:** We feed the AI over 1,500 historic shots. It learns the physics of the game.
* **The Result:** When a player shoots, the app ignores whether it went in or not. Instead, it looks at distance, angle, and defensive pressure, and spits out a percentage (e.g., "That shot had a 28% chance of going in"). 
* **The Business Value:** Coaches can now mathematically prove which players are getting lucky, which are getting unlucky, and which players are making the smartest decisions on the field.

---

## 📸 System Previews

### 1. Data Janitor & Auto-Cleaning
*(The system ingests raw game files, automatically handles missing values, and allows coaches to click one button to remove statistical Z-score outliers.)*
### 2. Feature Engineering & AI Training
*(Coaches select base stats, and the app automatically calculates complex physics metrics like 'Spatial Danger' before training the XGBoost brain.)*
### 3. Head-to-Head Match Scouting
*(The AI hides the target game, tests it blindly, and generates a side-by-side 'Goals Above Expected' leaderboard for both teams.)*
---

## ⚙️ Technical Architecture (The Nuts & Bolts)

### 1. Feature Engineering (Translating Physics to Math)
* **`Spatial_Danger`**: Calculated as `Distance × |Angle|`. This mathematically maps the goalie's shrinking field of vision from the outside.
* **`Shooter_Mechanics`**: A binary interaction feature (`Hands_Free × Feet_Set`) to flag optimal kinetic energy transfer.

### 2. Machine Learning Engine
* **Algorithm:** XGBoost Classifier (chosen for sequential error correction).
* **Cost-Sensitive Learning:** Applied a **3.0x penalty weight** to missed shots that were defensively challenged. This corrected an early data bias where elite D-1 players making lucky, low-percentage shots confused the AI into thinking defense didn't matter.

### 3. Blind Validation Protocol
* **Methodology:** Leave-One-Group-Out Cross-Validation (LOGO-CV).
* **Execution:** If the coach selects "Maryland" to analyze, the system completely removes the Maryland game from its memory, trains itself on the remaining 1,500+ shots, and predicts the Maryland game completely blind to prevent data leakage.

### 4. MLOps & SaaS Deployment
* **Framework:** Streamlit
* **Automation:** Designed a zero-code drag-and-drop interface. The coaching staff can drop in a new post-game Excel file, and the app autonomously detects data types, builds the interaction terms, and outputs the scouting report in seconds.

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/43435323-31ee-46d5-a667-916999111379" />

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/afa29dc2-0d70-4594-ad10-a0e695b4e845" />

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/4b9b9e38-1bfb-4b69-80f5-8d8b372fab79" />

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/265b65df-8b0d-49f5-92dd-3739f959a2f2" />





