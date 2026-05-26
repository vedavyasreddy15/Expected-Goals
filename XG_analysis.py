import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from scipy import stats

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Dynamic xG Match Analyzer", page_icon="🥍", layout="wide")
st.title("🥍 Dynamic Match Analyzer & Head-to-Head Scouting")

# Helper function
def load_uploaded_file(uploaded_file):
    if uploaded_file.name.endswith('.csv'):
        try:
            return pd.read_csv(uploaded_file, encoding='utf-8')
        except UnicodeDecodeError:
            return pd.read_csv(uploaded_file, encoding='latin-1')
    elif uploaded_file.name.endswith('.xlsx'):
        return pd.read_excel(uploaded_file)
    return None

# --- 1. MULTI-FILE UPLOAD & COMBINE ---
st.sidebar.header("📂 Step 1: Upload Data")
uploaded_files = st.sidebar.file_uploader(
    "Upload All Datasets (CSV/Excel)", 
    type=["csv", "xlsx"], 
    accept_multiple_files=True
)

if uploaded_files:
    with st.spinner("Combining datasets and sanitizing..."):
        dfs = [load_uploaded_file(file) for file in uploaded_files]
        df = pd.concat(dfs, ignore_index=True)
        
        # Immediate Drop of blank targets
        if 'Goal' not in df.columns:
            st.error("⚠️ At least one dataset must contain a 'Goal' column.")
            st.stop()
            
        df = df.dropna(subset=['Goal']).copy()
        df['Goal'] = df['Goal'].astype(int) 
            
        # Standardize Booleans
        for col in df.columns:
            if df[col].astype(str).str.lower().isin(['true', 'false', 'yes', 'no']).any():
                df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0, 'yes': 1, 'no': 0})
        
        if 'Opp' not in df.columns: df['Opp'] = np.nan
            
        if 'UDEL Shot' in df.columns:
            df['Is_UD'] = df['UDEL Shot'].fillna(1).astype(int)
        elif 'UD Shot' in df.columns:
            df['Is_UD'] = df['UD Shot'].fillna(1).astype(int)
        else:
            df['Is_UD'] = 1

    # --- 2. DYNAMIC FEATURE SELECTION ---
    st.divider()
    st.subheader("⚙️ Step 2: Feature Selection")
    
    available_features = df.select_dtypes(include=[np.number]).columns.tolist()
    hidden_cols = ['Goal', 'Player', 'Opp', 'Is_UD', 'Shot ID', 'QTR', 'Week']
    valid_features = [col for col in available_features if col not in hidden_cols]
    
    default_selections = [col for col in valid_features if col in ['Shot_Distance', 'Shot_Angle', 'Hands_Free', 'Feet_Set', 'Challenged']]
    if not default_selections: default_selections = valid_features[:3]
    
    selected_features = st.multiselect(
        "Select base features for ML:", 
        valid_features, 
        default=default_selections
    )
    
    st.info("💡 **Interaction Terms:** `Spatial_Danger` and `Shooter_Mechanics` will be automatically generated if applicable.")

    # --- GENERATE INTERACTION TERMS IMMEDIATELY ---
    model_df = df.copy()
    active_features = selected_features.copy()
    
    if 'Shot_Distance' in model_df.columns and 'Shot_Angle' in model_df.columns:
        model_df['Spatial_Danger'] = model_df['Shot_Distance'] * model_df['Shot_Angle'].abs()
        if 'Spatial_Danger' not in active_features: active_features.append('Spatial_Danger')
        
    if 'Hands_Free' in model_df.columns and 'Feet_Set' in model_df.columns:
        model_df['Shooter_Mechanics'] = model_df['Hands_Free'] * model_df['Feet_Set']
        if 'Shooter_Mechanics' not in active_features: active_features.append('Shooter_Mechanics')

    # --- 3. DATA HEALTH & OUTLIER DETECTION ---
    st.divider()
    st.subheader("🧹 Step 3: Dataset Health & Cleaning")
    
    cols_to_check = active_features + ['Opp']
    null_count = model_df[cols_to_check].isnull().sum().sum()
    
    continuous_active = [col for col in active_features if model_df[col].nunique() > 2]
    
    outlier_mask = pd.Series(False, index=model_df.index)
    if continuous_active:
        temp_df = model_df[continuous_active].fillna(model_df[continuous_active].median())
        z_scores = np.abs(stats.zscore(temp_df))
        outlier_mask = (z_scores > 3).any(axis=1)
        
    outlier_count = outlier_mask.sum()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Usable Shots", len(model_df))
    c2.metric("Missing Values in Selection", null_count)
    c3.metric("Detected Outliers", outlier_count)
    
    if outlier_count > 0:
        with st.expander(f"👀 View the {outlier_count} Detected Outliers"):
            st.markdown("These shots are mathematically abnormal (e.g., extreme distance or angle).")
            display_cols = ['Opp', 'Goal'] + continuous_active
            st.dataframe(model_df[outlier_mask][display_cols])
    
    clean_choice = st.radio(
        "Data Cleaning Action:",
        ("Keep Raw Data", "Remove Outliers & Nulls")
    )

    # --- 4. ML EXECUTION ---
    st.divider()
    st.subheader("🎯 Step 4: Head-to-Head Analysis")
    
    # 🚨 THE FIX: ORDER OF OPERATIONS 🚨
    if clean_choice == "Remove Outliers & Nulls":
        model_df = model_df[~outlier_mask]            # 1. Drop outliers FIRST while sizes still match
        model_df = model_df.dropna(subset=cols_to_check) # 2. THEN drop the missing values
        
    opponents = sorted(model_df['Opp'].dropna().unique().tolist())
    
    if not opponents:
        st.warning("No opponents detected. Cannot run Match Analysis.")
    else:
        selected_opp = st.selectbox("Select Match to Analyze:", opponents)
        
        if st.button("Run ML Pipeline"):
            # Split Data
            test_data = model_df[model_df['Opp'] == selected_opp].copy()
            train_data = model_df[model_df['Opp'] != selected_opp].copy()
            
            # SAFETY CHECKS
            if len(test_data) == 0:
                st.error(f"❌ Error: 0 valid shots left for {selected_opp} after cleaning. Try keeping raw data.")
                st.stop()
            if len(train_data) == 0:
                st.error("❌ Error: No training data available after cleaning.")
                st.stop()

            X_train = train_data[active_features]
            y_train = train_data['Goal']
            X_test = test_data[active_features]
            y_test = test_data['Goal']
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = xgb.XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.01, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            test_data['xG_Prob'] = model.predict_proba(X_test_scaled)[:, 1]
            
            # Output
            st.divider()
            ud_data = test_data[test_data['Is_UD'] == 1]
            opp_data = test_data[test_data['Is_UD'] == 0]
            
            ud_actual = ud_data['Goal'].sum()
            ud_xg = ud_data['xG_Prob'].sum()
            opp_actual = opp_data['Goal'].sum()
            opp_xg = opp_data['xG_Prob'].sum()
            
            st.markdown(f"### 📊 Final Score: Delaware {ud_actual} - {opp_actual} {selected_opp}")
            
            ud_col, opp_col = st.columns(2)
            
            with ud_col:
                st.markdown("### 🔵 Delaware Offense")
                st.metric("Expected Goals (xG)", f"{ud_xg:.1f}", f"{(ud_actual - ud_xg):+.1f} GAxE")
                if not ud_data.empty and 'Player' in ud_data.columns:
                    ud_players = ud_data.groupby('Player').agg({'Goal':'sum', 'xG_Prob':'sum'})
                    ud_players['GAxE'] = (ud_players['Goal'] - ud_players['xG_Prob']).round(2)
                    st.dataframe(ud_players.sort_values('GAxE', ascending=False).head(3))

            with opp_col:
                st.markdown(f"### 🔴 {selected_opp} Offense")
                st.metric("Expected Goals (xG)", f"{opp_xg:.1f}", f"{(opp_actual - opp_xg):+.1f} GAxE")
                if not opp_data.empty and 'Player' in opp_data.columns:
                    opp_players = opp_data.groupby('Player').agg({'Goal':'sum', 'xG_Prob':'sum'})
                    opp_players['GAxE'] = (opp_players['Goal'] - opp_players['xG_Prob']).round(2)
                    st.dataframe(opp_players.sort_values('GAxE', ascending=False).head(3))

else:
    st.info("👋 Upload datasets to begin.")