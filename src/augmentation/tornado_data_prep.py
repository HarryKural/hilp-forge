"""
Tornado Data Preparation for Diffusion Model
Extracts, cleans, normalizes tornado data for augmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import os


class TornadoDataPreparator:
    """Prepare tornado data for diffusion model training"""
    
    def __init__(self, data_path='data/processed/storm_events_full.csv'):
        self.data_path = data_path
        self.df = None
        self.tornado_df = None
        self.hilp_tornado_df = None
        self.feature_cols = None
        self.scaler = None
        
    def load_data(self):
        """Load processed storm data"""
        print("Loading processed data...")
        self.df = pd.read_csv(self.data_path, low_memory=False)
        print(f"  Loaded {len(self.df):,} events")
        
        # extract tornadoes
        self.tornado_df = self.df[self.df['event_type'] == 'Tornado'].copy()
        print(f"  Found {len(self.tornado_df):,} tornado events")
        
        # extract HILP tornadoes
        if 'is_hilp' in self.tornado_df.columns:
            self.hilp_tornado_df = self.tornado_df[
                self.tornado_df['is_hilp'] == True
            ].copy()
        else:
            # if no HILP flag, use top tornadoes by impact
            print("  No 'is_hilp' column found, selecting top 5% by impact...")
            if 'impact_score' in self.tornado_df.columns:
                threshold = self.tornado_df['impact_score'].quantile(0.95)
                self.hilp_tornado_df = self.tornado_df[
                    self.tornado_df['impact_score'] >= threshold
                ].copy()
            else:
                # fallback: use all tornadoes with casualties or significant damage
                self.hilp_tornado_df = self.tornado_df[
                    (self.tornado_df['total_casualties'] > 0) | 
                    (self.tornado_df['total_damage'] > 1e6)
                ].copy()
        
        print(f"  Found {len(self.hilp_tornado_df):,} HILP tornado events\n")
        
        if len(self.hilp_tornado_df) == 0:
            raise ValueError("No HILP tornado events found! Check your data.")
        
        return self.tornado_df, self.hilp_tornado_df
    
    def select_features(self):
        """Select relevant features for augmentation"""
        print("Selecting features for augmentation...")
        
        # define potential feature columns
        potential_features = [
            'magnitude',           # wind speed
            'duration_hours',      # event duration
            'spatial_extent_km',   # path length
            'total_deaths',        # fatalities
            'total_injuries',      # injuries
            'total_damage',        # economic damage
            'month',               # seasonality
            'hour',                # time of day
            'begin_lat',           # latitude
            'begin_lon',           # longitude
        ]
        
        # check which features exist and have data
        available_features = []
        for f in potential_features:
            if f in self.hilp_tornado_df.columns:
                non_null_count = self.hilp_tornado_df[f].notna().sum()
                if non_null_count > 0:
                    available_features.append(f)
                    print(f"  {f}: {non_null_count}/{len(self.hilp_tornado_df)} values")
                else:
                    print(f"  {f}: all null values, skipping")
            else:
                print(f"  {f}: column not found")
        
        if len(available_features) == 0:
            raise ValueError("No valid features found!")
        
        self.feature_cols = available_features
        print(f"\n  Selected {len(self.feature_cols)} features\n")
        
        return self.feature_cols
    
    def clean_data(self):
        """Clean and validate tornado data"""
        print("Cleaning tornado data...")
        
        df = self.hilp_tornado_df.copy()
        initial_count = len(df)
        
        print(f"  Initial samples: {initial_count}")
        
        # handle missing values in feature columns FIRST
        print("\n  Handling missing values...")
        for col in self.feature_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    # fill with median for numeric columns
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col] = df[col].fillna(median_val)
                        print(f"    - {col}: filled {missing_count} missing with median ({median_val:.2f})")
                    else:
                        # if median is NaN, fill with 0
                        df[col] = df[col].fillna(0)
                        print(f"    - {col}: filled {missing_count} missing with 0")
        
        # remove rows with any remaining NaN in feature columns
        before_dropna = len(df)
        df = df.dropna(subset=self.feature_cols)
        dropped_na = before_dropna - len(df)
        if dropped_na > 0:
            print(f"    - Removed {dropped_na} rows with remaining NaN values")
        
        # replace infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=self.feature_cols)
        
        # clean specific features
        print("\n  Cleaning specific features...")
        
        # duration: remove negatives, cap at reasonable max
        if 'duration_hours' in df.columns:
            negative_dur = (df['duration_hours'] < 0).sum()
            if negative_dur > 0:
                print(f"    - duration_hours: setting {negative_dur} negative values to 0")
                df['duration_hours'] = df['duration_hours'].clip(lower=0)
            # cap at 24 hours (most tornadoes are much shorter)
            df['duration_hours'] = df['duration_hours'].clip(upper=24)
        
        # magnitude: should be positive
        if 'magnitude' in df.columns:
            df['magnitude'] = df['magnitude'].clip(lower=0, upper=300)  # Max wind ~300 knots
        
        # spatial extent: should be positive
        if 'spatial_extent_km' in df.columns:
            df['spatial_extent_km'] = df['spatial_extent_km'].clip(lower=0, upper=500)
        
        # deaths/injuries: should be non-negative integers
        if 'total_deaths' in df.columns:
            df['total_deaths'] = df['total_deaths'].clip(lower=0)
        if 'total_injuries' in df.columns:
            df['total_injuries'] = df['total_injuries'].clip(lower=0)
        
        # damage: should be non-negative
        if 'total_damage' in df.columns:
            df['total_damage'] = df['total_damage'].clip(lower=0)
        
        # month: should be 1-12
        if 'month' in df.columns:
            df = df[df['month'].between(1, 12)]
        
        # hour: should be 0-23
        if 'hour' in df.columns:
            df = df[df['hour'].between(0, 23)]
        
        # latitude: should be reasonable for US (approx 25-50)
        if 'begin_lat' in df.columns:
            df = df[df['begin_lat'].between(20, 55)]
        
        # longitude: should be reasonable for US (approx -125 to -65)
        if 'begin_lon' in df.columns:
            df = df[df['begin_lon'].between(-130, -60)]
        
        final_count = len(df)
        removed_count = initial_count - final_count
        
        print(f"\n  Cleaned data: {initial_count} → {final_count} samples")
        print(f"  Removed {removed_count} problematic records ({removed_count/initial_count*100:.1f}%)")
        
        if final_count < 50:
            print(f"\n  WARNING: Only {final_count} samples remaining!")
            print(f"  This may not be enough for good model training.")
            print(f"  Consider using all tornadoes instead of just HILP events.\n")
        
        self.hilp_tornado_df = df
        return df
    
    def visualize_distributions(self, output_dir='results/tornado/figures'):
        """Visualize feature distributions before normalization"""
        print("Visualizing feature distributions...")
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.hilp_tornado_df
        n_features = len(self.feature_cols)
        
        # create subplots
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, col in enumerate(self.feature_cols):
            ax = axes[i]
            
            # histogram
            ax.hist(df[col], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax.set_xlabel(col, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Distribution: {col}', fontsize=11, fontweight='bold')
            ax.grid(alpha=0.3)
            
            # add statistics
            mean_val = df[col].mean()
            median_val = df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                      label=f'Median: {median_val:.2f}')
            ax.legend(fontsize=8)
        
        # hide extra subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, 'feature_distributions_raw.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {output_path}\n")
        
        # correlation matrix
        print("Creating correlation matrix...")
        fig, ax = plt.subplots(figsize=(12, 10))
        
        corr_matrix = df[self.feature_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=ax,
                   cbar_kws={"shrink": 0.8})
        ax.set_title('Feature Correlation Matrix (Raw Data)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        corr_path = os.path.join(output_dir, 'correlation_matrix_raw.png')
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {corr_path}\n")
        
        return corr_matrix
    
    def normalize_data(self, method='standard'):
        """Normalize features for model training"""
        print(f"Normalizing data using {method} scaling...")
        
        df = self.hilp_tornado_df[self.feature_cols].copy()
        
        print(f"  Data shape before normalization: {df.shape}")
        
        if len(df) == 0:
            raise ValueError("No data to normalize! All samples were filtered out during cleaning.")
        
        # choose scaler
        if method == 'standard':
            # zero mean, unit variance
            self.scaler = StandardScaler()
        elif method == 'minmax':
            # scale to [0, 1]
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # fit and transform
        normalized_data = self.scaler.fit_transform(df)
        
        # create normalized dataframe
        normalized_df = pd.DataFrame(
            normalized_data,
            columns=self.feature_cols,
            index=df.index
        )
        
        print(f"  Normalized {len(self.feature_cols)} features")
        print(f"  Shape: {normalized_df.shape}")
        
        # show normalization statistics
        print("\n  Normalized data statistics:")
        print(normalized_df.describe().round(3))
        print()
        
        return normalized_df
    
    def save_prepared_data(self, normalized_df, output_dir='data/processed'):
        """Save prepared data and scaler"""
        print("Saving prepared data...")
        os.makedirs(output_dir, exist_ok=True)
        
        # save normalized data
        data_path = os.path.join(output_dir, 'tornado_hilp_normalized.csv')
        normalized_df.to_csv(data_path, index=False)
        print(f"  Saved normalized data: {data_path}")
        
        # save scaler
        scaler_path = os.path.join(output_dir, 'tornado_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"  Saved scaler: {scaler_path}")
        
        # save feature names
        feature_path = os.path.join(output_dir, 'tornado_features.txt')
        with open(feature_path, 'w') as f:
            f.write('\n'.join(self.feature_cols))
        print(f"  Saved feature list: {feature_path}")
        
        # save original data (for comparison)
        original_path = os.path.join(output_dir, 'tornado_hilp_original.csv')
        self.hilp_tornado_df[self.feature_cols].to_csv(original_path, index=False)
        print(f"  Saved original data: {original_path}\n")
        
        return data_path, scaler_path
    
    def generate_summary_statistics(self):
        """Generate summary statistics for tornado data"""
        print("="*60)
        print("TORNADO DATA SUMMARY")
        print("="*60)
        
        df = self.hilp_tornado_df
        
        print(f"\nTotal HILP Tornadoes: {len(df):,}")
        
        if 'begin_date_time' in df.columns:
            print(f"\nDate Range: {df['begin_date_time'].min()} to {df['begin_date_time'].max()}")
        
        print("\nKey Statistics:")
        print("-"*60)
        
        stats = {}
        
        if 'total_deaths' in df.columns:
            stats['Total Deaths'] = int(df['total_deaths'].sum())
            stats['Avg Deaths per Event'] = f"{df['total_deaths'].mean():.2f}"
        
        if 'total_injuries' in df.columns:
            stats['Total Injuries'] = int(df['total_injuries'].sum())
            stats['Avg Injuries per Event'] = f"{df['total_injuries'].mean():.2f}"
        
        if 'total_damage' in df.columns:
            stats['Total Damage'] = f"${df['total_damage'].sum()/1e9:.2f}B"
            stats['Avg Damage per Event'] = f"${df['total_damage'].mean()/1e6:.2f}M"
        
        if 'duration_hours' in df.columns:
            stats['Avg Duration'] = f"{df['duration_hours'].mean():.2f} hours"
        
        if 'spatial_extent_km' in df.columns:
            stats['Avg Path Length'] = f"{df['spatial_extent_km'].mean():.2f} km"
        
        for key, value in stats.items():
            print(f"  {key:<30} {value}")
        
        # F-scale distribution
        if 'tor_f_scale' in df.columns:
            print("\nF-Scale Distribution:")
            print("-"*60)
            fscale_dist = df['tor_f_scale'].value_counts().sort_index()
            for scale, count in fscale_dist.items():
                if pd.notna(scale):
                    print(f"  {scale}: {count:,} ({count/len(df)*100:.1f}%)")
        
        # monthly distribution
        if 'month' in df.columns:
            print("\nMonthly Distribution:")
            print("-"*60)
            monthly = df['month'].value_counts().sort_index()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month, count in monthly.items():
                if pd.notna(month) and 1 <= int(month) <= 12:
                    print(f"  {month_names[int(month)-1]}: {count:,} ({count/len(df)*100:.1f}%)")
        
        print("\n" + "="*60 + "\n")
    
    def run_full_preparation(self):
        """Run complete data preparation pipeline"""
        print("\n" + "="*60)
        print("TORNADO DATA PREPARATION PIPELINE")
        print("="*60 + "\n")
        
        # load data
        self.load_data()
        
        # select features
        self.select_features()
        
        # clean data
        self.clean_data()
        
        # generate initial summary
        self.generate_summary_statistics()
        
        # visualize distributions
        corr_matrix = self.visualize_distributions()
        
        # normalize data
        normalized_df = self.normalize_data(method='standard')
        
        # save prepared data
        data_path, scaler_path = self.save_prepared_data(normalized_df)
        
        print("="*60)
        print("DATA PREPARATION COMPLETE")
        print("="*60)
        print(f"Training samples: {len(normalized_df):,}")
        print(f"Features: {len(self.feature_cols)}")
        print("="*60 + "\n")
        
        return normalized_df, self.scaler


def main():
    """Main execution"""
    preparator = TornadoDataPreparator()
    normalized_df, scaler = preparator.run_full_preparation()
    
    print("\nPreparation complete!")
    print(f"Normalized data shape: {normalized_df.shape}")
    print(f"Features: {list(normalized_df.columns)}")


if __name__ == "__main__":
    main()