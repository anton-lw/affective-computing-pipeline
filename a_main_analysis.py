# a_main_analysis.py
import os
import sys
from dotenv import load_dotenv

# Import modules
import b_data_loader
import c_feature_annotator
import d_temporal_analyzer
import e_visualizer

# --- CONFIGURATION ---
INPUT_FILE = 'Chronological_Archive.txt'
OUTPUT_ANNOTATED_CSV = 'annotated_messages.csv'

def main():
    """
    Main function to run the entire well-being analysis pipeline.
    """
    load_dotenv() # Load API keys from .env file

    # --- ANNOTATION PHASE ---
    if not os.path.exists(OUTPUT_ANNOTATED_CSV):
        print(f"Annotated file '{OUTPUT_ANNOTATED_CSV}' not found. Starting full annotation process.")
        
        # Load data
        df = b_data_loader.load_and_parse_data(INPUT_FILE)
        
        # Validate data loading
        if df is None or df.empty:
            print("\nCRITICAL ERROR: Data loading failed. The DataFrame is empty.")
            print(f"Please check the format of '{INPUT_FILE}'.")
            sys.exit(1)
        
        # Run annotation modules
        nrc_lexicon = c_feature_annotator.download_and_prepare_nrc_lexicon()
        df = c_feature_annotator.annotate_linguistic_features(df, nrc_lexicon)
        df = c_feature_annotator.annotate_deep_learning_emotions(df)
        df = c_feature_annotator.annotate_llm_perma(df)
        
        # Save results
        df.to_csv(OUTPUT_ANNOTATED_CSV, index=False)
        print(f"\nAnnotation complete. Data saved to '{OUTPUT_ANNOTATED_CSV}'.")
    else:
        print(f"Found existing annotated file '{OUTPUT_ANNOTATED_CSV}'. Skipping annotation.")

    # --- ANALYSIS & VISUALIZATION PHASE ---
    print("\n--- Starting analysis & visualization phase ---")
    annotated_df = pd.read_csv(OUTPUT_ANNOTATED_CSV, parse_dates=['timestamp'])
    
    # Temporal analysis
    daily_data = d_temporal_analyzer.aggregate_to_daily_time_series(annotated_df)
    
    # Change point detection
    signal, change_points = d_temporal_analyzer.detect_change_points(daily_data)
    
    # Visualization
    e_visualizer.plot_wellbeing_trajectory(signal, change_points)
    e_visualizer.plot_multivariate_markers(daily_data)
    e_visualizer.plot_perma_sentiment(annotated_df)
    
    # Display all plots at once
    e_visualizer.show_plots()

if __name__ == '__main__':
    main()