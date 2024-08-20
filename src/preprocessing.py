'''
PART 1: PRE-PROCESSING
- Tailor the code scaffolding below to load and process the data
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import pandas as pd
import ast

def load_data():
    '''
    Load data from CSV files
    
    Returns:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genres_df (pd.DataFrame): DataFrame containing genre information
    '''
    # Load the data from the correct file paths
    model_pred_df = pd.read_csv('data/prediction_model_03.csv')
    genres_df = pd.read_csv('data/genres.csv')
    
    # Debugging: Check a few rows of the DataFrame
    print("model_pred_df head:")
    print(model_pred_df.head())
    
    print("genres_df head:")
    print(genres_df.head())
    
    return model_pred_df, genres_df


def process_data(model_pred_df, genres_df):
    '''
    Process data to get genre lists and count dictionaries
    
    Returns:
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    '''
    # Extract the list of unique genres
    genre_list = genres_df['genre'].unique().tolist()
    
    # Initialize dictionaries to hold the counts
    genre_true_counts = {genre: 0 for genre in genre_list}
    genre_tp_counts = {genre: 0 for genre in genre_list}
    genre_fp_counts = {genre: 0 for genre in genre_list}
    
    # Iterate over each row to calculate the true, TP, and FP counts
    for _, row in model_pred_df.iterrows():
        # Use ast.literal_eval to convert the string representation to a list
        true_genres = ast.literal_eval(row['actual genres'])
        pred_genres = row['predicted'].split(',')
        
        # Strip whitespace and compare
        true_genres = [genre.strip() for genre in true_genres]
        pred_genres = [genre.strip() for genre in pred_genres]
        
        for genre in genre_list:
            if genre in true_genres:
                genre_true_counts[genre] += 1
                if genre in pred_genres:
                    genre_tp_counts[genre] += 1
            if genre in pred_genres and genre not in true_genres:
                genre_fp_counts[genre] += 1
    
    return genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts
