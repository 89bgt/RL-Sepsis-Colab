"""
Custom data loader for training DQN+LSTM on real patient data
"""
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_real_data(data_path='data/training_setA', max_files=None, balance_ratio=0.5):
    """
    Load the patient data from .psv files
    
    Args:
        data_path: Path to the directory containing patient data files
        max_files: Maximum number of files to load (for testing, set to None to load all)
        balance_ratio: Target ratio of sepsis to non-sepsis patients (0.5 means 50% sepsis patients)
    
    Returns:
        Combined DataFrame with all patient data and additional columns for rewards
    """
    print(f"Loading data from {data_path}...")
    all_files = [f for f in os.listdir(data_path) if f.endswith('.psv')]
    
    if max_files:
        all_files = all_files[:max_files]
    
    print(f"Processing {len(all_files)} patient files...")
    
    # First pass: identify sepsis and non-sepsis patients
    sepsis_files = []
    non_sepsis_files = []
    
    # Sample a small set to identify sepsis/non-sepsis files
    sample_size = min(1000, len(all_files))
    sample_files = np.random.choice(all_files, sample_size, replace=False)
    
    for file in tqdm(sample_files, desc="Identifying sepsis patients"):
        file_path = os.path.join(data_path, file)
        df = pd.read_csv(file_path, sep='|')
        
        # Check if patient has sepsis
        if df['SepsisLabel'].sum() > 0:
            sepsis_files.append(file)
        else:
            non_sepsis_files.append(file)
    
    print(f"Found {len(sepsis_files)} sepsis patients and {len(non_sepsis_files)} non-sepsis patients in sample")
    
    # Calculate balanced sampling based on found ratio
    sepsis_ratio = len(sepsis_files) / sample_size
    print(f"Sepsis prevalence in sample: {sepsis_ratio:.2%}")
    
    # Second pass: load all files with balanced sampling if needed
    if max_files is None:
        max_files = len(all_files)
    
    # Adjust balance_ratio to be reasonable (avoid extreme values)
    balance_ratio = min(max(balance_ratio, 0.1), 0.9)
    
    # Initialize list to store all patient dataframes
    dfs = []
    
    # Use all sepsis patients if there are fewer than desired
    all_sepsis_count = int(len(sepsis_files) / sepsis_ratio * sample_size / len(all_files) * max_files)
    desired_sepsis_count = int(balance_ratio * max_files)
    
    if all_sepsis_count < desired_sepsis_count:
        print(f"Warning: Not enough sepsis patients to achieve {balance_ratio:.0%} balance")
        print(f"Using all {all_sepsis_count} estimated sepsis patients")
        # Use all files, but make sure we process sepsis files first
        files_to_process = sepsis_files + [f for f in all_files if f not in sepsis_files]
        files_to_process = files_to_process[:max_files]
    else:
        # Calculate how many files of each type to use
        n_sepsis = min(desired_sepsis_count, int(len(sepsis_files) / sepsis_ratio * sample_size / len(all_files) * max_files))
        n_non_sepsis = max_files - n_sepsis
        
        # Randomly sample from both categories
        if n_sepsis > len(sepsis_files):
            sepsis_to_use = sepsis_files
            # Sample additional sepsis files from remainder
            remaining_files = [f for f in all_files if f not in sepsis_files and f not in non_sepsis_files]
            additional_sepsis = []
            
            for file in tqdm(remaining_files, desc="Finding additional sepsis patients"):
                if len(sepsis_to_use) >= n_sepsis:
                    break
                    
                file_path = os.path.join(data_path, file)
                try:
                    df = pd.read_csv(file_path, sep='|')
                    if df['SepsisLabel'].sum() > 0:
                        sepsis_to_use.append(file)
                except:
                    continue
        else:
            sepsis_to_use = np.random.choice(sepsis_files, n_sepsis, replace=False)
            
        if n_non_sepsis > len(non_sepsis_files):
            non_sepsis_to_use = non_sepsis_files
            # Get remaining files
            remaining_files = [f for f in all_files if f not in sepsis_files and f not in non_sepsis_files]
            # Take as many as needed
            additional_non_sepsis = remaining_files[:min(len(remaining_files), n_non_sepsis - len(non_sepsis_files))]
            non_sepsis_to_use = non_sepsis_to_use + additional_non_sepsis
        else:
            non_sepsis_to_use = np.random.choice(non_sepsis_files, n_non_sepsis, replace=False)
            
        files_to_process = list(sepsis_to_use) + list(non_sepsis_to_use)
    
    print(f"Loading {len(files_to_process)} files ({sum(1 for f in files_to_process if f in sepsis_files)} sepsis)")
    
    # Load each patient file
    for file in tqdm(files_to_process, desc="Loading patient data"):
        file_path = os.path.join(data_path, file)
        # Extract patient ID from filename (e.g., p000001.psv -> 1)
        patient_id = int(file.split('.')[0][1:])
        
        try:
            # Read the PSV file
            df = pd.read_csv(file_path, sep='|')
            
            # Add patient identifier
            df['patient'] = patient_id
            
            # Add index column (hour)
            df['index'] = range(len(df))
            
            # Append to list of dataframes
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    
    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Calculate missing values for each feature
    print("Analyzing missing values in features...")
    missing_percentage = combined_df.isnull().mean() * 100
    print("Missing percentage for each feature:")
    for col, pct in missing_percentage.items():
        print(f"{col}: {pct:.2f}%")
        
    # Exclude features with too many missing values (> 70%)
    high_missing_cols = missing_percentage[missing_percentage > 70].index.tolist()
    print(f"\nExcluding {len(high_missing_cols)} features with >70% missing values:")
    for col in high_missing_cols:
        print(f"  - {col}")
    
    # Keep only features with reasonable amount of data
    usable_cols = [col for col in combined_df.columns if col not in high_missing_cols]
    combined_df = combined_df[usable_cols]
    
    # Handle missing values
    combined_df = handle_missing_values(combined_df)
    
    # Add temporal features to capture trends
    combined_df = add_temporal_features(combined_df)
    
    # Add reward columns with more aggressive penalties for missed sepsis
    combined_df = add_rewards(combined_df)
    
    # Add end episode flag
    combined_df = add_end_episode_flag(combined_df)
    
    # Calculate final class balance
    sepsis_rows = combined_df[combined_df['SepsisLabel'] == 1]
    sepsis_patients = sepsis_rows['patient'].nunique()
    total_patients = combined_df['patient'].nunique()
    
    print(f"\nFinal dataset statistics:")
    print(f"  - Total rows: {len(combined_df)}")
    print(f"  - Total patients: {total_patients}")
    print(f"  - Patients with sepsis: {sepsis_patients} ({sepsis_patients/total_patients:.2%})")
    print(f"  - Rows with sepsis: {len(sepsis_rows)} ({len(sepsis_rows)/len(combined_df):.2%})")
    
    return combined_df

def handle_missing_values(df):
    """
    Handle missing values in the dataset with improved methods
    """
    # Get list of columns with numeric data (excluding patient, index, SepsisLabel, etc.)
    non_feature_cols = ['patient', 'index', 'SepsisLabel', 'zeros_reward', 'ones_reward', 'end_episode']
    numeric_cols = [col for col in df.columns if col not in non_feature_cols]
    
    print("Handling missing values...")
    
    # Create a copy of the dataframe to avoid modifying the original
    df_processed = df.copy()
    
    # Get unique patients
    patients = df['patient'].unique()
    
    # Step 1: Forward fill within each patient time series (for temporal consistency)
    for patient in tqdm(patients, desc="Forward filling patient data"):
        patient_mask = df['patient'] == patient
        df_processed.loc[patient_mask, numeric_cols] = df_processed.loc[patient_mask, numeric_cols].ffill()
    
    # Step 2: Backward fill for remaining NaNs at the beginning of time series
    for patient in tqdm(patients, desc="Backward filling patient data"):
        patient_mask = df['patient'] == patient
        df_processed.loc[patient_mask, numeric_cols] = df_processed.loc[patient_mask, numeric_cols].bfill()
    
    # Step 3: For any remaining NaNs, fill with global median for that column
    missing_after_fill = df_processed[numeric_cols].isnull().sum()
    
    # Only calculate and fill medians for columns that still have missing values
    cols_with_missing = [col for col in numeric_cols if missing_after_fill[col] > 0]
    
    if cols_with_missing:
        print(f"Filling remaining NaNs with column medians for {len(cols_with_missing)} columns")
        for col in cols_with_missing:
            median_val = df_processed[col].median()
            if pd.isna(median_val):  # If median is also NaN (all values are NaN)
                median_val = 0  # Use 0 as a last resort
            df_processed[col] = df_processed[col].fillna(median_val)
    
    # Final check for any remaining NaNs
    final_missing = df_processed.isnull().sum().sum()
    if final_missing > 0:
        print(f"Warning: {final_missing} missing values remain after all fill methods")
        # Fill any remaining NaNs with 0 as a last resort
        df_processed = df_processed.fillna(0)
    else:
        print("All missing values have been handled successfully")
    
    return df_processed

def add_rewards(df):
    """
    Add clinically relevant reward columns based on SepsisLabel and risk factors
    
    Enhanced reward structure:
    - For non-sepsis prediction (action=0):
      - Reward if patient doesn't have sepsis (scaled by time from onset)
      - Large penalty if patient has sepsis (missed detection)
    - For sepsis prediction (action=1):
      - Large reward if patient has sepsis (higher reward for earlier detection)
      - Penalty if patient doesn't have sepsis (scaled by clinical risk)
    """
    # Initialize reward columns
    df['zeros_reward'] = 0.0  # Reward for predicting no sepsis (action=0)
    df['ones_reward'] = 0.0   # Reward for predicting sepsis (action=1)
    
    print("Adding clinically relevant reward structure...")
    
    # Get unique patients
    patients = df['patient'].unique()
    
    # Enhanced reward parameters to address class imbalance and clinical priorities
    SEPSIS_MISS_PENALTY = -15.0     # Increased penalty for missing sepsis
    SEPSIS_DETECT_REWARD = 8.0      # Increased reward for detecting sepsis
    EARLY_DETECTION_BONUS = 4.0     # Additional bonus for early detection
    NON_SEPSIS_CORRECT = 0.2        # Reward for correctly saying no sepsis
    FALSE_ALARM_PENALTY = -0.3      # Penalty for false alarm
    
    # Clinical risk adjustment - if present in the data
    has_risk_score = 'clinical_risk_score' in df.columns or 'SIRS_score' in df.columns or 'qSOFA_score' in df.columns
    
    # Determine which risk score to use
    if 'clinical_risk_score' in df.columns:
        risk_column = 'clinical_risk_score'
    elif 'SIRS_score' in df.columns:
        risk_column = 'SIRS_score'
    elif 'qSOFA_score' in df.columns:
        risk_column = 'qSOFA_score'
    else:
        risk_column = None
    
    for patient in tqdm(patients, desc="Calculating patient-specific rewards"):
        patient_mask = df['patient'] == patient
        patient_df = df[patient_mask]
        
        # Check if patient ever develops sepsis
        sepsis_rows = patient_df[patient_df['SepsisLabel'] == 1]
        
        if len(sepsis_rows) > 0:
            # Patient develops sepsis at some point
            sepsis_onset = sepsis_rows['index'].min()
            
            # For each time step in this patient's data
            for idx, row in patient_df.iterrows():
                current_hour = row['index']
                
                # Get clinical risk factor if available
                clinical_risk_factor = 1.0  # Default factor
                if risk_column and risk_column in row:
                    # Normalize risk score to be between 1.0 and 2.0
                    risk_value = row[risk_column]
                    if risk_column == 'SIRS_score':
                        # SIRS is usually 0-4
                        clinical_risk_factor = 1.0 + (risk_value / 4.0)
                    elif risk_column == 'qSOFA_score':
                        # qSOFA is usually 0-3
                        clinical_risk_factor = 1.0 + (risk_value / 3.0)
                    else:
                        # Assume range 0-5 for general clinical_risk_score
                        clinical_risk_factor = 1.0 + (min(risk_value, 5.0) / 5.0)
                
                if row['SepsisLabel'] == 1:
                    # Patient has sepsis at this time step
                    # Scale penalty by clinical risk factor
                    df.loc[idx, 'zeros_reward'] = SEPSIS_MISS_PENALTY * clinical_risk_factor
                    
                    # Reward for detection, with higher reward for detecting very early
                    time_from_onset = current_hour - sepsis_onset
                    if time_from_onset == 0:  # First hour of sepsis
                        df.loc[idx, 'ones_reward'] = SEPSIS_DETECT_REWARD + EARLY_DETECTION_BONUS
                    else:
                        df.loc[idx, 'ones_reward'] = SEPSIS_DETECT_REWARD
                else:
                    # Patient doesn't have sepsis at this time step
                    if current_hour < sepsis_onset:
                        # Before sepsis onset
                        time_to_sepsis = sepsis_onset - current_hour
                        
                        # Create a more gradual scale for pre-sepsis rewards
                        # The closer to sepsis onset, the more we want to reward early prediction
                        if time_to_sepsis <= 24:  # Within 24 hours of sepsis
                            # Time factor: scales from 0 (24h away) to 1 (at onset)
                            time_factor = 1.0 - (time_to_sepsis / 24.0)
                            
                            # Apply clinical risk factor if available
                            # High risk patients get smaller penalties for false alarms
                            false_alarm_adjustment = clinical_risk_factor * time_factor
                            
                            # Gradually reduce reward for saying no sepsis as we get closer to onset
                            # And reduce penalty for saying sepsis (false alarm) when close to onset and high risk
                            df.loc[idx, 'zeros_reward'] = NON_SEPSIS_CORRECT * (1.0 - time_factor * 0.8)
                            
                            # False alarm penalty is reduced when:
                            # 1. We're close to sepsis onset (time_factor near 1)
                            # 2. Patient has high clinical risk (clinical_risk_factor > 1)
                            df.loc[idx, 'ones_reward'] = FALSE_ALARM_PENALTY * (1.0 - false_alarm_adjustment * 0.7)
                        else:
                            # Far from sepsis onset, use standard values
                            df.loc[idx, 'zeros_reward'] = NON_SEPSIS_CORRECT
                            df.loc[idx, 'ones_reward'] = FALSE_ALARM_PENALTY
                    else:
                        # This shouldn't happen (time steps after sepsis onset should have SepsisLabel=1)
                        # But just in case, treat it like a non-sepsis patient
                        df.loc[idx, 'zeros_reward'] = NON_SEPSIS_CORRECT
                        df.loc[idx, 'ones_reward'] = FALSE_ALARM_PENALTY
        else:
            # Patient never develops sepsis
            indices = df.index[patient_mask].tolist()
            
            # Default rewards for non-sepsis patients
            df.loc[indices, 'zeros_reward'] = NON_SEPSIS_CORRECT
            df.loc[indices, 'ones_reward'] = FALSE_ALARM_PENALTY
            
            # If we have clinical risk information, adjust rewards based on risk
            if risk_column and risk_column in df.columns:
                for idx in indices:
                    risk_value = df.loc[idx, risk_column]
                    
                    # Normalize risk score to a factor between 1.0 and 1.5
                    if risk_column == 'SIRS_score':
                        risk_factor = 1.0 + (risk_value / 8.0)  # Less aggressive for SIRS
                    elif risk_column == 'qSOFA_score':
                        risk_factor = 1.0 + (risk_value / 6.0)  # Less aggressive for qSOFA
                    else:
                        risk_factor = 1.0 + (min(risk_value, 5.0) / 10.0)
                    
                    # For high-risk patients who never develop sepsis:
                    # - Increase reward for correctly saying no sepsis (to reinforce learning)
                    # - Reduce penalty for false alarms (since they were clinically at risk)
                    df.loc[idx, 'zeros_reward'] = NON_SEPSIS_CORRECT * risk_factor
                    df.loc[idx, 'ones_reward'] = FALSE_ALARM_PENALTY * (1.0 - (risk_factor - 1.0) * 0.5)
    
    # Calculate some statistics on the rewards
    zeros_mean = df['zeros_reward'].mean()
    zeros_min = df['zeros_reward'].min()
    zeros_max = df['zeros_reward'].max()
    ones_mean = df['ones_reward'].mean()
    ones_min = df['ones_reward'].min()
    ones_max = df['ones_reward'].max()
    
    print(f"Reward statistics:")
    print(f"Action 0 (no sepsis) - Min: {zeros_min:.2f}, Mean: {zeros_mean:.2f}, Max: {zeros_max:.2f}")
    print(f"Action 1 (sepsis) - Min: {ones_min:.2f}, Mean: {ones_mean:.2f}, Max: {ones_max:.2f}")
    
    return df

def add_temporal_features(df):
    """
    Add features that capture changes over time for vital signs and lab values
    """
    print("Adding temporal features...")
    
    # Define important vital signs to track
    vital_signs = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']
    lab_values = ['Lactate', 'WBC', 'Creatinine', 'Glucose', 'Platelets', 'HCO3', 'pH', 'PaCO2']
    
    # Use only columns that exist in the dataframe
    vital_signs = [col for col in vital_signs if col in df.columns]
    lab_values = [col for col in lab_values if col in df.columns]
    
    # Get unique patients
    patients = df['patient'].unique()
    
    # Create new dataframe to avoid SettingWithCopyWarning
    df_with_features = df.copy()
    
    # For each patient, calculate temporal features
    for patient in tqdm(patients, desc="Calculating temporal features"):
        patient_mask = df['patient'] == patient
        patient_df = df.loc[patient_mask].copy()
        
        # Only process patients with at least 3 time steps
        if len(patient_df) < 3:
            continue
        
        # Add rolling window features for vital signs
        for col in vital_signs:
            # 1. Add rolling mean (smoothed value over time)
            df_with_features.loc[patient_mask, f'{col}_rolling_mean_3h'] = \
                patient_df[col].rolling(window=3, min_periods=1).mean()
                
            # 2. Add rate of change (first derivative)
            df_with_features.loc[patient_mask, f'{col}_rate_of_change'] = \
                patient_df[col].diff().fillna(0)
            
            # 3. Add acceleration (second derivative)
            df_with_features.loc[patient_mask, f'{col}_acceleration'] = \
                patient_df[col].diff().diff().fillna(0)
                
            # 4. Add volatility (rolling standard deviation)
            df_with_features.loc[patient_mask, f'{col}_volatility_3h'] = \
                patient_df[col].rolling(window=3, min_periods=1).std().fillna(0)
        
        # Add rolling window features for lab values (less frequent)
        for col in lab_values:
            # For lab values, mostly interested in significant changes
            df_with_features.loc[patient_mask, f'{col}_change'] = \
                patient_df[col].diff().fillna(0)
    
    # Add clinical risk score based on SIRS criteria (Systemic Inflammatory Response Syndrome)
    # This is a simple example - medical domain knowledge could improve this
    sirs_score = pd.Series(0, index=df_with_features.index)
    
    # SIRS criteria: HR > 90
    if 'HR' in df_with_features.columns:
        sirs_score += (df_with_features['HR'] > 90).astype(int)
    
    # SIRS criteria: Temp > 38°C or < 36°C
    if 'Temp' in df_with_features.columns:
        sirs_score += ((df_with_features['Temp'] > 38) | (df_with_features['Temp'] < 36)).astype(int)
    
    # SIRS criteria: Respiratory rate > 20
    if 'Resp' in df_with_features.columns:
        sirs_score += (df_with_features['Resp'] > 20).astype(int)
    
    # SIRS criteria: WBC > 12,000 or < 4,000
    if 'WBC' in df_with_features.columns:
        sirs_score += ((df_with_features['WBC'] > 12) | (df_with_features['WBC'] < 4)).astype(int)
    
    # Add SIRS score
    df_with_features['SIRS_score'] = sirs_score
    
    # Add qSOFA score components (quick Sepsis-related Organ Failure Assessment)
    qsofa_score = pd.Series(0, index=df_with_features.index)
    
    # qSOFA criteria: SBP ≤ 100 mmHg
    if 'SBP' in df_with_features.columns:
        qsofa_score += (df_with_features['SBP'] <= 100).astype(int)
    
    # qSOFA criteria: Respiratory rate ≥ 22/min
    if 'Resp' in df_with_features.columns:
        qsofa_score += (df_with_features['Resp'] >= 22).astype(int)
    
    # Add qSOFA score
    df_with_features['qSOFA_score'] = qsofa_score
    
    # Create a composite risk score
    clinical_cols = [col for col in ['SIRS_score', 'qSOFA_score'] if col in df_with_features.columns]
    if clinical_cols:
        df_with_features['clinical_risk_score'] = df_with_features[clinical_cols].sum(axis=1)
    
    # Check how many new features were created
    new_cols = [col for col in df_with_features.columns if col not in df.columns]
    print(f"Added {len(new_cols)} new temporal and clinical features")
    
    return df_with_features

def add_end_episode_flag(df):
    """
    Add a flag to indicate the end of each patient episode
    """
    # Initialize end_episode column with zeros
    df['end_episode'] = 0
    
    # Get unique patients
    patients = df['patient'].unique()
    
    for patient in patients:
        # Get the last row for this patient
        patient_mask = df['patient'] == patient
        last_row_idx = df[patient_mask].index[-1]
        
        # Set end_episode to 1 for the last row
        df.loc[last_row_idx, 'end_episode'] = 1
    
    return df
