import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool, cv
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

def clean_numeric_data(df, columns):
    """
    Clean numeric data by properly handling string conversions.
    """
    for col in columns:
        if col in df.columns:
            mask = df[col].astype(str).str.contains(',', na=False)
            df.loc[mask, col] = df.loc[mask, col].astype(str).str.replace(',', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calculate_sector_intensity(df, sector_col, emissions_cols, revenue_col):
    """
    Calculate sector-level emissions intensity (emissions per dollar of revenue).
    """
    try:
        df[emissions_cols] = df[emissions_cols].apply(pd.to_numeric, errors='coerce')
        df[revenue_col] = pd.to_numeric(df[revenue_col], errors='coerce')

        df['total_emissions'] = df[emissions_cols].sum(axis=1)
        sector_totals = df.groupby(sector_col).agg(
            total_emissions=('total_emissions', 'sum'),
            total_revenue=(revenue_col, 'sum')
        )
        sector_totals['Intensity'] = sector_totals.apply(
            lambda row: row['total_emissions'] / row['total_revenue'] if row['total_revenue'] > 0 else 0, axis=1
        )
        return sector_totals[['Intensity']].reset_index()
    except Exception as e:
        logger.error(f"Error in calculate_sector_intensity: {str(e)}")
        return pd.DataFrame(columns=[sector_col, 'Intensity'])

def fill_missing_values_using_sector_intensity(df, sector_col, emissions_cols, sector_intensities_df, revenue_col):
    """
    Fill missing values for Scope 1, Scope 2, and Scope 3 using sector-level intensity.
    """
    try:
        for col in emissions_cols:
            if col in df.columns and not sector_intensities_df.empty:
                for sector, intensity in sector_intensities_df.values:
                    missing_rows = (df[sector_col] == sector) & (df[col].isnull())
                    df.loc[missing_rows, col] = intensity * df.loc[missing_rows, revenue_col]
        return df
    except Exception as e:
        logger.error(f"Error in filling values using sector intensity: {str(e)}")
        return df

def idw_interpolation(df, sector_col, emissions_cols, weight_col):
    """
    Implements Inverse Distance Weighted (IDW) interpolation for Scope 1 and Scope 2.
    """
    try:
        for col in emissions_cols:
            if col not in df.columns:
                continue
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')

            for sector in df[sector_col].unique():
                sector_data = df[df[sector_col] == sector]
                known_data = sector_data[~sector_data[col].isnull()]
                missing_data = sector_data[sector_data[col].isnull()]

                if not missing_data.empty and not known_data.empty:
                    weights = 1 / (known_data[weight_col] + 1e-6)
                    weights /= weights.sum()
                    for idx in missing_data.index:
                        df.loc[idx, col] = (known_data[col] * weights).sum()
        return df
    except Exception as e:
        logger.error(f"Error in IDW interpolation: {str(e)}")
        return df

def normalize_by_sector(df, sector_col, columns_to_normalize):
    """
    Implement sector-based normalization.
    """
    normalized_df = df.copy()
    for sector in df[sector_col].unique():
        sector_mask = df[sector_col] == sector
        for col in columns_to_normalize:
            if col in df.columns:
                sector_data = df.loc[sector_mask, col].fillna(0)
                median = sector_data.median()
                mad = np.median(np.abs(sector_data - median))
                if mad > 0:
                    normalized_df.loc[sector_mask, f'{col}_normalized'] = (sector_data - median) / mad
                else:
                    normalized_df.loc[sector_mask, f'{col}_normalized'] = 0
    return normalized_df

def prepare_features(df, emissions_cols, revenue_col, sector_col):
    """
    Enhanced feature preparation.
    """
    # Clean numeric data
    df = clean_numeric_data(df, emissions_cols + [revenue_col, "Shares"])

    # Apply robust scaling
    scaler = RobustScaler()
    for col in emissions_cols + [revenue_col]:
        if col in df.columns:
            df[f'{col}_robust'] = scaler.fit_transform(df[[col]].fillna(0))

    # Sector-based normalization
    df = normalize_by_sector(df, sector_col, emissions_cols + [revenue_col])

    # Create log transformations
    for col in emissions_cols + [revenue_col]:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0).clip(lower=1e-6))

    return df

def train_sector_specific_models(train_df, test_df, sector_col, target_col, features):
    """
    Train separate models for each sector.
    """
    sector_models = {}
    sector_predictions = pd.Series(index=test_df.index, dtype=float)

    for sector in train_df[sector_col].unique():
        if pd.isna(sector):
            continue

        sector_mask_train = train_df[sector_col] == sector
        sector_mask_test = test_df[sector_col] == sector

        if sector_mask_train.sum() < 10:
            continue

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=10,
            random_state=42,
            verbose=0
        )

        X_train = train_df.loc[sector_mask_train, features].fillna(0)
        y_train = train_df.loc[sector_mask_train, target_col].fillna(0)

        if len(X_train) > 0 and len(y_train) > 0:
            model.fit(X_train, y_train)
            sector_models[sector] = model

            if sector_mask_test.any():
                X_test = test_df.loc[sector_mask_test, features].fillna(0)
                predictions = model.predict(X_test)
                predictions = np.maximum(predictions, 0)
                sector_predictions.loc[sector_mask_test] = predictions

    return sector_predictions, sector_models

def k_fold_cross_validation(df, features, target_col, params, folds=5):
    """
    Perform K-Fold Cross-Validation using CatBoost.
    """
    # Ensure the target column has no NaN values
    df.loc[:, target_col] = df[target_col].fillna(0)

    # Fill missing values in features
    df.loc[:, features] = df[features].fillna(0)

    # Create Pool for CatBoost
    catboost_data = Pool(df[features], label=df[target_col])

    try:
        # Perform k-fold cross-validation
        cv_results = cv(
            pool=catboost_data,
            params=params,
            fold_count=folds,
            early_stopping_rounds=50,
            verbose=False
        )

        # Check if cv_results is valid
        if cv_results is not None and not cv_results.empty:
            best_iteration = len(cv_results['test-RMSE-mean'])
            best_rmse = cv_results['test-RMSE-mean'].iloc[-1]
            logger.info(f"Best Iteration: {best_iteration}")
            logger.info(f"Best RMSE: {best_rmse:.2f}")
            return cv_results
        else:
            logger.warning("Cross-validation returned no results. Check your data or parameters.")
            return None

    except Exception as e:
        logger.error(f"Error during cross-validation: {e}")
        return None