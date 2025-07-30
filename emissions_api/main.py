import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor, Pool, cv
import traceback

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
        return pd.DataFrame(columns=['Sector', 'Intensity'])

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
    df = clean_numeric_data(df, emissions_cols + [revenue_col, "Shares"])

    scaler = RobustScaler()
    for col in emissions_cols + [revenue_col]:
        if col in df.columns:
            df[f'{col}_robust'] = scaler.fit_transform(df[[col]].fillna(0))

    df = normalize_by_sector(df, sector_col, emissions_cols + [revenue_col])

    for col in emissions_cols + [revenue_col]:
        if col in df.columns:
            df[f'log_{col}'] = np.log1p(df[col].fillna(0).clip(lower=1e-6))

    return df

def main(input_df):
    try:
        df = input_df.copy()
        print(f"Original columns: {list(df.columns)}")  # Debug info

        df.rename(columns={
            '#NAME?': 'Ticker',
            'Carbon Emissions Scope 1 FY 2023': 'Carbon Emissions Scope 1 (2023)',
            'Carbon Emissions Scope 2 FY 2023': 'Carbon Emissions Scope 2 (2023)',
            'Carbon Emissions Scope3  FY2023': 'Carbon Emissions Scope 3 (2023)',
            'Carbon Emissions Scope 1 FY 2022': 'Carbon Emissions Scope 1 (2022)',
            'Carbon Emissions Scope 2 FY 2022': 'Carbon Emissions Scope 2 (2022)',
            'Carbon Emissions Scope3  FY2022': 'Carbon Emissions Scope 3 (2022)',
            'Carbon Emissions Scope 1 FY 2021': 'Carbon Emissions Scope 1 (2021)',
            'Carbon Emissions Scope 2 FY 2021': 'Carbon Emissions Scope 2 (2021)',
            'Carbon Emissions Scope3  FY 2021': 'Carbon Emissions Scope 3 (2021)',
            'SalesUSDrecent FY 2023': 'Revenue (2023)',
            'SalesUSDrecent FY 2022': 'Revenue (2022)',
            'SalesUSDrecent FY 2021': 'Revenue (2021)',
        }, inplace=True)
        
        print(f"After renaming: {list(df.columns)}")  # Debug info

        numeric_cols = [
            'Carbon Emissions Scope 1 (2023)', 'Carbon Emissions Scope 2 (2023)', 'Carbon Emissions Scope 3 (2023)',
            'Carbon Emissions Scope 1 (2022)', 'Carbon Emissions Scope 2 (2022)', 'Carbon Emissions Scope 3 (2022)',
            'Carbon Emissions Scope 1 (2021)', 'Carbon Emissions Scope 2 (2021)', 'Carbon Emissions Scope 3 (2021)',
            'Revenue (2023)', 'Revenue (2022)', 'Revenue (2021)', 'market value', 'Notional Value', 'Shares'
        ]
        df = clean_numeric_data(df, numeric_cols)

        # Filter value_vars to only include columns that actually exist in the dataframe
        available_value_vars = [col for col in numeric_cols if col != 'market value' and col != 'Notional Value' and col != 'Shares' and col in df.columns]
        print(f"Available value vars: {available_value_vars}")  # Debug info
        
        df_long = df.melt(
            id_vars=[
                'Ticker', 'Name', 'Sector', 'Asset Class', 'Country of Headquarters',
                'Geographical Region', 'market value', 'Notional Value', 'Shares',
                'Cusip', 'ISIN', 'Sedol', 'Sector_Group', 'Region_Group'
            ],
            value_vars=available_value_vars,
            var_name='Metric_Year',
            value_name='Value'
        )

        df_long['Year'] = df_long['Metric_Year'].str.extract(r'(\d{4})').astype(int)
        df_long['Metric'] = df_long['Metric_Year'].str.extract(r'(Scope \d|Revenue)').fillna('Unknown')
        df_long.drop(columns=['Metric_Year'], inplace=True)

        df_pivot = df_long.pivot_table(
            index=[col for col in df_long.columns if col not in ['Metric', 'Value']],
            columns='Metric',
            values='Value'
        ).reset_index()
        
        print(f"After pivot - columns: {list(df_pivot.columns)}")  # Debug info
        print(f"Pivot shape: {df_pivot.shape}")  # Debug info
        print(f"Years in data: {df_pivot['Year'].unique() if 'Year' in df_pivot.columns else 'No Year column'}")  # Debug info

        emissions_cols = ['Scope 1', 'Scope 2', 'Scope 3']
        revenue_col = 'Revenue'
        sector_col = 'Sector'

        sector_intensities = calculate_sector_intensity(df_pivot, sector_col, emissions_cols, revenue_col)
        df_pivot = fill_missing_values_using_sector_intensity(
            df_pivot, sector_col, emissions_cols, sector_intensities, revenue_col
        )
        df_pivot = idw_interpolation(df_pivot, sector_col, emissions_cols, 'market value')
        df_pivot = prepare_features(df_pivot, emissions_cols, revenue_col, sector_col)

        base_features = [col for col in df_pivot.columns if any(
            x in col for x in ['_robust', '_normalized', 'log_', '_ratio']
        )]

        train_df = df_pivot[df_pivot['Year'].isin([2021, 2022])]
        test_df = df_pivot[df_pivot['Year'] == 2023]
        
        print(f"Train data shape: {train_df.shape}")  # Debug info
        print(f"Test data shape: {test_df.shape}")  # Debug info

        results = {}
        final_predictions = test_df.copy()

        small_params = {
            'iterations': 1000,
            'learning_rate': 0.01,
            'depth': 4,
            'l2_leaf_reg': 5,
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'RMSE'
        }

        large_params = {
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': 42,
            'verbose': False,
            'loss_function': 'RMSE'
        }

        for target_col in emissions_cols:
            results[target_col] = {}

            small_threshold = train_df[target_col].quantile(0.25)
            medium_threshold = train_df[target_col].quantile(0.75)
            min_nonzero = train_df[train_df[target_col] > 0][target_col].min() if not train_df[train_df[target_col] > 0].empty else 0

            segments = [
                ('small', lambda x: x <= small_threshold, small_params),
                ('medium', lambda x: (x > small_threshold) & (x <= medium_threshold), small_params),
                ('large', lambda x: x > medium_threshold, large_params)
            ]

            for segment_name, segment_filter, params in segments:
                train_mask = segment_filter(train_df[target_col])
                if train_mask.sum() < 10:
                    continue

                X_train = train_df.loc[train_mask, base_features].fillna(0)
                y_train = np.log1p(train_df.loc[train_mask, target_col].fillna(0))

                model = CatBoostRegressor(**params)
                model.fit(X_train, y_train)

                test_mask = segment_filter(test_df.get(target_col, pd.Series(0, index=test_df.index)))
                X_test = test_df.loc[test_mask, base_features].fillna(0)
                if not X_test.empty:
                    predictions = np.expm1(model.predict(X_test))
                    predictions = np.maximum(predictions, min_nonzero)
                    if segment_name == 'small' and predictions.mean() != 0:
                        scale_factor = train_df.loc[train_mask, target_col].mean() / predictions.mean()
                        predictions = predictions * scale_factor

                    final_predictions.loc[test_mask, f'Predicted_{target_col}'] = predictions

                    if target_col in test_df.columns:
                        actuals = test_df.loc[test_mask, target_col].fillna(0)
                        if len(actuals) > 0:
                            metrics = {
                                'RMSE': np.sqrt(mean_squared_error(actuals, predictions)),
                                'R²': r2_score(actuals, predictions),
                                'MAPE': np.mean(np.abs((actuals - predictions) / (actuals + 1e-6))) * 100,
                                'SMAPE': np.mean(np.abs(actuals - predictions) / (np.abs(actuals) + np.abs(predictions) + 1e-6)) * 100
                            }
                            results[target_col][segment_name] = metrics

        # Clean any infinite or NaN values before returning
        final_predictions = final_predictions.replace([np.inf, -np.inf], np.nan)
        final_predictions = final_predictions.fillna(0)
        
        # Clean results dictionary as well
        for scope, scope_results in results.items():
            for segment, metrics in scope_results.items():
                for metric, value in metrics.items():
                    if np.isnan(value) or np.isinf(value):
                        results[scope][segment][metric] = 0.0

        return final_predictions, results

    except Exception as e:
        traceback.print_exc()
        return None, {"error": str(e)}