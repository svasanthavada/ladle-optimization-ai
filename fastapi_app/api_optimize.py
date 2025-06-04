from fastapi import UploadFile
import pandas as pd
from src.preprocessing import preprocess_pipeline
from src.api_optimization_service import run_full_optimization

def optimize_alloys(file: UploadFile):
    df, summary_df = preprocess_pipeline(file.file)
    result = run_full_optimization(df, summary_df)
    return result
