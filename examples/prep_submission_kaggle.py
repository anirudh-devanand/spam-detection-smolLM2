import pandas as pd
import numpy as np
import argparse
import os

def prep_kaggle_submission(input_csv_path, output_csv_path='kaggle_submission.csv'):
    """
    Process a dataset with data_index,prob_ham,prob_spam columns
    and create a Kaggle submission CSV with ID,SPAM/HAM columns.
    
    Args:
        input_csv_path (str): Path to input CSV with probability predictions
        output_csv_path (str): Path for output Kaggle submission CSV
    """
    
    # Read the probability dataset
    df = pd.read_csv(input_csv_path)
    
    # Validate input columns
    required_columns = ['data_index', 'prob_ham', 'prob_spam']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    print(f"Input dataset shape: {df.shape}")
    print(f"Input columns: {df.columns.tolist()}")
    print("\nFirst few rows of input:")
    print(df.head())
    
    # Create submission dataframe
    submission_df = pd.DataFrame()
    
    # Set ID column from data_index
    submission_df['ID'] = df['data_index'].astype(int)
    
    # Determine SPAM/HAM based on which probability is higher
    # If prob_spam > prob_ham, classify as spam (1), else ham (0)
    spam_predictions = (df['prob_spam'] > df['prob_ham']).astype(int)
    
    submission_df['SPAM/HAM'] = spam_predictions
    
    # Display prediction statistics
    print(f"\nPrediction distribution:")
    print(submission_df['SPAM/HAM'].value_counts())
    
    # Show probability statistics
    print(f"\nProbability statistics:")
    print(f"Average prob_ham: {df['prob_ham'].mean():.4f}")
    print(f"Average prob_spam: {df['prob_spam'].mean():.4f}")
    
    print(f"\nFirst few rows of submission:")
    print(submission_df.head(10))
    
    # Save submission file
    submission_df.to_csv(output_csv_path, index=False)
    print(f"\nKaggle submission saved to: {output_csv_path}")
    
    return submission_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Kaggle submission from probability predictions")
    parser.add_argument("--input", type=str, default="bayes_inverse_probs/test_dataset_probs.csv",
                       help="Path to input CSV with data_index,prob_ham,prob_spam columns")
    parser.add_argument("--output", type=str, default="kaggle_submission.csv",
                       help="Path for output Kaggle submission CSV")
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file {args.input} not found.")
        print("Make sure to run the bayes_inverse.py script first to generate probability predictions.")
        exit(1)
    
    # Process the file
    submission_df = prep_kaggle_submission(args.input, args.output)