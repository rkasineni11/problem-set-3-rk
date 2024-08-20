'''
PART 2: METRICS CALCULATION
- Tailor the code scaffolding below to calculate various metrics
- Write the functions below
    - Further info and hints are provided in the docstrings
    - These should return values when called by the main.py
'''

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import ast

def calculate_metrics(model_pred_df, genre_list, genre_true_counts, genre_tp_counts, genre_fp_counts):
    '''
    Calculate micro and macro metrics
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions
        genre_list (list): List of unique genres
        genre_true_counts (dict): Dictionary of true genre counts
        genre_tp_counts (dict): Dictionary of true positive genre counts
        genre_fp_counts (dict): Dictionary of false positive genre counts
    
    Returns:
        tuple: Micro precision, recall, F1 score
        lists of macro precision, recall, and F1 scores
    
    Hint #1: 
    tp -> true positives
    fp -> false positives
    tn -> true negatives
    fn -> false negatives

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    Hint #2: Micro metrics are tuples, macro metrics are lists
    '''

    tp_sum = sum(genre_tp_counts.values())
    fp_sum = sum(genre_fp_counts.values())
    fn_sum = sum(genre_true_counts.values()) - tp_sum
    
    micro_precision = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
    micro_recall = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # Macro metrics calculation
    macro_prec_list = []
    macro_recall_list = []
    macro_f1_list = []
    
    for genre in genre_list:
        tp = genre_tp_counts[genre]
        fp = genre_fp_counts[genre]
        fn = genre_true_counts[genre] - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        macro_prec_list.append(precision)
        macro_recall_list.append(recall)
        macro_f1_list.append(f1_score)
    
    return micro_precision, micro_recall, micro_f1, macro_prec_list, macro_recall_list, macro_f1_list


def calculate_sklearn_metrics(model_pred_df, genre_list):
    '''
    Calculate metrics using sklearn's precision_recall_fscore_support.
    
    Args:
        model_pred_df (pd.DataFrame): DataFrame containing model predictions.
        genre_list (list): List of unique genres.
    
    Returns:
        tuple: Macro precision, recall, F1 score, and micro precision, recall, F1 score.
    
    Hint #1: You'll need these two lists
    pred_rows = []
    true_rows = []
    
    Hint #2: And a little later you'll need these two matrices for sk-learn
    pred_matrix = pd.DataFrame(pred_rows)
    true_matrix = pd.DataFrame(true_rows)
    '''
    pred_rows = []
    true_rows = []

    # Iterate over the DataFrame rows
    for _, row in model_pred_df.iterrows():
        # Use ast.literal_eval to convert the string representation to a list
        true_genres = ast.literal_eval(row['actual genres'])
        pred_genres = row['predicted'].split(',')
        
        # Strip whitespace and compare
        true_genres = [genre.strip() for genre in true_genres]
        pred_genres = [genre.strip() for genre in pred_genres]
        
        # Create binary row representations for true and predicted genres
        true_row = [1 if genre in true_genres else 0 for genre in genre_list]
        pred_row = [1 if genre in pred_genres else 0 for genre in genre_list]
        
        # Append rows to the lists
        true_rows.append(true_row)
        pred_rows.append(pred_row)
    
    # Convert lists to DataFrame matrices
    true_matrix = pd.DataFrame(true_rows, columns=genre_list)
    pred_matrix = pd.DataFrame(pred_rows, columns=genre_list)
    
    # Calculate macro and micro metrics using sklearn, with zero_division set to 0 to avoid warnings
    macro_prec, macro_rec, macro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='macro', zero_division=0)
    micro_prec, micro_rec, micro_f1, _ = precision_recall_fscore_support(true_matrix, pred_matrix, average='micro', zero_division=0)
    
    return macro_prec, macro_rec, macro_f1, micro_prec, micro_rec, micro_f1
