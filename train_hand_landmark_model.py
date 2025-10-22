import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_prepare_data(csv_path):
    """
    Load the hand landmarks CSV and prepare it for training.
    Handles mixed format CSV files (old 65-column and new 128-column formats).
    
    Args:
        csv_path: Path to the CSV file containing hand landmarks
        
    Returns:
        X: Feature matrix (hand landmarks)
        y: Target labels (gesture classes)
        label_encoder: Fitted LabelEncoder for inverse transform
    """
    print(f"Loading data from: {csv_path}")
    
    # Read CSV manually to handle mixed formats
    import csv
    rows_old = []  # 65 columns: class, hand_label, 63 features
    rows_new = []  # 128 columns: class, hands_used, 126 features
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header_old = None
        header_new = None
        
        for i, row in enumerate(reader):
            if i == 0 and len(row) == 65:
                # Old format header
                header_old = row
                continue
            elif i == 0 and len(row) == 128:
                # New format header
                header_new = row
                continue
            elif i == 0:
                print(f"⚠️  Unexpected header with {len(row)} columns, skipping...")
                continue
            
            # Data rows
            if len(row) == 65:
                rows_old.append(row)
            elif len(row) == 128:
                rows_new.append(row)
            else:
                print(f"⚠️  Skipping row {i+1} with unexpected {len(row)} columns")
    
    print(f"Found {len(rows_old)} old-format samples (single hand)")
    print(f"Found {len(rows_new)} new-format samples (two hands)")
    
    # Convert old format to new format (expand to 128 columns)
    converted_rows = []
    for row in rows_old:
        class_label = row[0]
        hand_label = row[1].lower()  # "Left" or "Right"
        features = [float(x) for x in row[2:]]  # 63 features
        
        # Create 126-feature row with zeros for missing hand
        if hand_label == 'left':
            left_features = features
            right_features = [0.0] * 63
            hands_used = 'left'
        else:  # 'right'
            left_features = [0.0] * 63
            right_features = features
            hands_used = 'right'
        
        # Row format: [class, hands_used, 63 left features, 63 right features]
        converted_rows.append([class_label, hands_used] + left_features + right_features)
    
    # Combine converted old data with new data
    # New data format: [class, hands_used, 63 left features, 63 right features]
    new_data_rows = []
    for row in rows_new:
        class_label = row[0]
        hands_used = row[1]
        # Rest are float features (126 features)
        features = [float(x) for x in row[2:]]
        new_data_rows.append([class_label, hands_used] + features)
    
    all_rows = converted_rows + new_data_rows
    
    # Create DataFrame
    header = ['class', 'hands_used']
    for i in range(21):
        header.extend([f'left_x{i}', f'left_y{i}', f'left_z{i}'])
    for i in range(21):
        header.extend([f'right_x{i}', f'right_y{i}', f'right_z{i}'])
    
    df = pd.DataFrame(all_rows, columns=header)
    
    # Combine similar-looking classes
    print(f"\n🔄 Combining similar-looking gesture classes...")
    class_mapping = {
        '0': 'o(0)',
        'o': 'o(0)',
        '7': 'l(7)',
        'l': 'l(7)',
        '2': 'v(2)',
        'v': 'v(2)'
    }
    
    # Count samples being combined
    combined_counts = {}
    for old_class, new_class in class_mapping.items():
        count = (df['class'] == old_class).sum()
        if count > 0:
            if new_class not in combined_counts:
                combined_counts[new_class] = {}
            combined_counts[new_class][old_class] = count
    
    # Apply the mapping
    df['class'] = df['class'].replace(class_mapping)
    
    # Show what was combined
    if combined_counts:
        print("Combined classes:")
        for new_class, old_classes in combined_counts.items():
            total = sum(old_classes.values())
            details = ", ".join([f"{k}={v}" for k, v in old_classes.items()])
            print(f"  {new_class}: {total} samples ({details})")
    
    print(f"\n✓ Successfully loaded and unified data")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of classes: {df['class'].nunique()}")
    print(f"Classes: {sorted(df['class'].unique())}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts().sort_index())
    print(f"\nHand usage distribution:")
    print(df['hands_used'].value_counts())
    
    # Separate features and labels
    feature_columns = [col for col in df.columns if col not in ['class', 'hands_used']]
    print(f"\nFeature columns (first 10): {feature_columns[:10]}")
    print(f"Total feature columns: {len(feature_columns)}")
    
    X = df[feature_columns].values
    
    # Debug: Check if X contains any non-numeric values
    print(f"\nChecking X data type...")
    print(f"X dtype: {X.dtype}")
    if X.dtype == 'object':
        print("⚠️  WARNING: X contains non-numeric data!")
        # Find problematic rows
        for i in range(min(5, len(X))):
            for j, val in enumerate(X[i]):
                if isinstance(val, str):
                    print(f"  Row {i}, Column {j} ({feature_columns[j]}): '{val}' (string)")
    
    # Ensure X is numeric
    X = X.astype(float)
    
    # Encode class labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['class'])
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Number of features per sample: {X.shape[1]}")
    print(f"Labels shape: {y.shape}")
    
    return X, y, label_encoder, df

def train_model(X_train, y_train, X_test, y_test, label_encoder):
    """
    Train a Random Forest classifier on the hand landmark data.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        label_encoder: LabelEncoder for class names
        
    Returns:
        model: Trained model
        scaler: Fitted StandardScaler
        metrics: Dictionary containing evaluation metrics
    """
    print("\n" + "="*60)
    print("Training Random Forest Classifier...")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Classification report
    class_names = label_encoder.classes_
    print(f"\n{'='*60}")
    print("Classification Report:")
    print(f"{'='*60}")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    metrics = {
        'accuracy': accuracy,
        'y_test': y_test,
        'y_pred': y_pred,
        'class_names': class_names
    }
    
    return model, scaler, metrics

def plot_confusion_matrix(y_test, y_pred, class_names, output_path):
    """
    Plot and save confusion matrix.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        class_names: List of class names
        output_path: Path to save the plot
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix - Hand Landmark Classification', fontsize=16, pad=20)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to: {output_path}")
    plt.close()

def plot_feature_importance(model, output_path, top_n=30):
    """
    Plot feature importance from the Random Forest model.
    
    Args:
        model: Trained Random Forest model
        output_path: Path to save the plot
        top_n: Number of top features to display
    """
    feature_importance = model.feature_importances_
    
    # Create feature names for two-hand format (126 features)
    feature_names = []
    # Left hand features (63)
    for i in range(21):
        feature_names.extend([f'left_x{i}', f'left_y{i}', f'left_z{i}'])
    # Right hand features (63)
    for i in range(21):
        feature_names.extend([f'right_x{i}', f'right_y{i}', f'right_z{i}'])
    
    # Sort by importance
    indices = np.argsort(feature_importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importance = feature_importance[indices]
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), top_importance[::-1], color='steelblue')
    plt.yticks(range(top_n), top_features[::-1])
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Top {top_n} Most Important Hand Landmark Features', fontsize=14, pad=20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved to: {output_path}")
    plt.close()

def plot_class_distribution(df, output_path):
    """
    Plot the distribution of samples across classes.
    
    Args:
        df: DataFrame containing the data
        output_path: Path to save the plot
    """
    class_counts = df['class'].value_counts().sort_index()
    
    plt.figure(figsize=(16, 6))
    plt.bar(range(len(class_counts)), class_counts.values, color='steelblue', edgecolor='black')
    plt.xticks(range(len(class_counts)), class_counts.index)
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.title('Class Distribution in Training Dataset', fontsize=14, pad=20)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Class distribution plot saved to: {output_path}")
    plt.close()

def main():
    # Configuration
    csv_path = "outputs/hand_landmarks.csv"
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("outputs", f"hand_landmark_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("SASL Hand Landmark Model Training")
    print("="*60)
    print(f"Output directory: {output_dir}")
    
    # Load data
    X, y, label_encoder, df = load_and_prepare_data(csv_path)
    
    # Filter out classes with too few samples (less than 2)
    class_counts = df['class'].value_counts()
    classes_to_keep = class_counts[class_counts >= 2].index
    
    if len(classes_to_keep) < len(class_counts):
        print(f"\nFiltering out classes with less than 2 samples:")
        removed_classes = class_counts[class_counts < 2].index.tolist()
        print(f"  Removed classes: {removed_classes}")
        
        # Filter dataframe
        df = df[df['class'].isin(classes_to_keep)]
        
        # Recreate X, y, and label_encoder
        feature_columns = [col for col in df.columns if col not in ['class', 'hands_used']]
        X = df[feature_columns].values.astype(float)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df['class'])
        
        print(f"  New dataset size: {len(df)} samples")
        print(f"  Remaining classes: {len(label_encoder.classes_)}")
    
    # Split data
    print("\nSplitting data into train and test sets (80/20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Train model
    model, scaler, metrics = train_model(X_train, y_train, X_test, y_test, label_encoder)
    
    # Save model and preprocessing objects
    model_path = os.path.join(output_dir, "random_forest_model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    encoder_path = os.path.join(output_dir, "label_encoder.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    
    print(f"\nModel saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Label encoder saved to: {encoder_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(metrics['y_test'], metrics['y_pred'], 
                         metrics['class_names'], cm_path)
    
    # Feature importance
    fi_path = os.path.join(output_dir, "feature_importance.png")
    plot_feature_importance(model, fi_path)
    
    # Class distribution
    cd_path = os.path.join(output_dir, "class_distribution.png")
    plot_class_distribution(df, cd_path)
    
    # Save metrics summary
    summary_path = os.path.join(output_dir, "training_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("SASL Hand Landmark Model Training Summary\n")
        f.write("="*60 + "\n\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"CSV File: {csv_path}\n\n")
        f.write(f"Dataset Statistics:\n")
        f.write(f"  Total samples: {len(X)}\n")
        f.write(f"  Number of features: {X.shape[1]}\n")
        f.write(f"  Number of classes: {len(label_encoder.classes_)}\n")
        f.write(f"  Classes: {', '.join(label_encoder.classes_)}\n\n")
        f.write(f"Training Configuration:\n")
        f.write(f"  Model: Random Forest Classifier\n")
        f.write(f"  Train samples: {len(X_train)}\n")
        f.write(f"  Test samples: {len(X_test)}\n")
        f.write(f"  Test split: 20%\n\n")
        f.write(f"Results:\n")
        f.write(f"  Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
    
    print(f"Training summary saved to: {summary_path}")
    
    print("\n" + "="*60)
    print("Training Complete! All files saved successfully.")
    print("="*60)
    print(f"\nTo use this model for prediction:")
    print(f"  1. Load the model: model = joblib.load('{model_path}')")
    print(f"  2. Load the scaler: scaler = joblib.load('{scaler_path}')")
    print(f"  3. Load the encoder: encoder = joblib.load('{encoder_path}')")
    print(f"  4. Scale your features: X_scaled = scaler.transform(X)")
    print(f"  5. Make predictions: predictions = model.predict(X_scaled)")
    print(f"  6. Decode labels: class_names = encoder.inverse_transform(predictions)")

if __name__ == "__main__":
    main()
