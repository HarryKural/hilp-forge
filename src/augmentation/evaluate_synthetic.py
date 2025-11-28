"""
Evaluation Module for Synthetic Tornado Data
Compares real vs synthetic data using statistical tests and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import jensenshannon
import joblib
import os


class SyntheticDataEvaluator:
    """Evaluate quality of synthetic data"""
    
    def __init__(self, real_data, synthetic_data, feature_names, scaler=None):
        """
        Args:
            real_data: Real data (normalized or original)
            synthetic_data: Synthetic data (normalized)
            feature_names: List of feature names
            scaler: Scaler object to denormalize data
        """
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.feature_names = feature_names
        self.scaler = scaler
        
        # convert to DataFrame if needed
        if not isinstance(self.real_data, pd.DataFrame):
            self.real_data = pd.DataFrame(self.real_data, columns=feature_names)
        if not isinstance(self.synthetic_data, pd.DataFrame):
            self.synthetic_data = pd.DataFrame(self.synthetic_data, columns=feature_names)
        
        self.metrics = {}
    
    def denormalize_data(self):
        """Denormalize data using scaler"""
        if self.scaler is not None:
            print("Denormalizing data...")
            real_denorm = self.scaler.inverse_transform(self.real_data)
            synth_denorm = self.scaler.inverse_transform(self.synthetic_data)
            
            self.real_data_denorm = pd.DataFrame(real_denorm, columns=self.feature_names)
            self.synthetic_data_denorm = pd.DataFrame(synth_denorm, columns=self.feature_names)
            
            print("  Data denormalized for visualization\n")
        else:
            self.real_data_denorm = self.real_data.copy()
            self.synthetic_data_denorm = self.synthetic_data.copy()
    
    def correlation_comparison(self):
        """Compare correlation matrices"""
        print("Comparing correlation matrices...")
        
        real_corr = self.real_data.corr()
        synth_corr = self.synthetic_data.corr()
        
        # Correlation difference
        corr_diff = np.abs(real_corr - synth_corr)
        mean_corr_diff = corr_diff.values[np.triu_indices_from(corr_diff.values, k=1)].mean()
        
        self.metrics['mean_correlation_difference'] = mean_corr_diff
        
        print(f"  Mean absolute correlation difference: {mean_corr_diff:.4f}")
        print(f"  Lower is better (0 = perfect match)\n")
        
        return real_corr, synth_corr, corr_diff
    
    def statistical_tests(self):
        """Perform statistical tests on each feature"""
        print("Performing statistical tests...")
        
        results = []
        
        for feature in self.feature_names:
            real_values = self.real_data[feature].values
            synth_values = self.synthetic_data[feature].values
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_pval = stats.ks_2samp(real_values, synth_values)
            
            # Wasserstein distance (Earth Mover's Distance)
            wasserstein_dist = stats.wasserstein_distance(real_values, synth_values)
            
            # Jensen-Shannon divergence
            # create histograms for both distributions
            bins = np.linspace(
                min(real_values.min(), synth_values.min()),
                max(real_values.max(), synth_values.max()),
                50
            )
            real_hist, _ = np.histogram(real_values, bins=bins, density=True)
            synth_hist, _ = np.histogram(synth_values, bins=bins, density=True)
            
            # add small constant to avoid log(0)
            real_hist = real_hist + 1e-10
            synth_hist = synth_hist + 1e-10
            
            # normalize
            real_hist = real_hist / real_hist.sum()
            synth_hist = synth_hist / synth_hist.sum()
            
            js_div = jensenshannon(real_hist, synth_hist)
            
            results.append({
                'Feature': feature,
                'KS_Statistic': ks_stat,
                'KS_P-Value': ks_pval,
                'Wasserstein_Distance': wasserstein_dist,
                'JS_Divergence': js_div
            })
        
        results_df = pd.DataFrame(results)
        
        # store average metrics
        self.metrics['mean_ks_statistic'] = results_df['KS_Statistic'].mean()
        self.metrics['mean_wasserstein'] = results_df['Wasserstein_Distance'].mean()
        self.metrics['mean_js_divergence'] = results_df['JS_Divergence'].mean()
        
        print(results_df.to_string(index=False))
        print(f"\n  Average KS Statistic: {self.metrics['mean_ks_statistic']:.4f}")
        print(f"  Average Wasserstein Distance: {self.metrics['mean_wasserstein']:.4f}")
        print(f"  Average JS Divergence: {self.metrics['mean_js_divergence']:.4f}")
        print("  Lower values indicate better similarity\n")
        
        return results_df
    
    def distribution_comparison_plots(self, output_dir='results/tornado/figures'):
        """Create distribution comparison plots"""
        print("Creating distribution comparison plots...")
        os.makedirs(output_dir, exist_ok=True)
        
        # use denormalized data for better interpretability
        real_plot = self.real_data_denorm
        synth_plot = self.synthetic_data_denorm
        
        n_features = len(self.feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            
            # histograms
            ax.hist(real_plot[feature], bins=30, alpha=0.6, label='Real',
                   color='steelblue', edgecolor='black', density=True)
            ax.hist(synth_plot[feature], bins=30, alpha=0.6, label='Synthetic',
                   color='crimson', edgecolor='black', density=True)
            
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'{feature} Distribution', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # hide extra subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        dist_path = os.path.join(output_dir, 'distribution_comparison.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {dist_path}\n")
        
        return dist_path
    
    def qq_plots(self, output_dir='results/tornado/figures'):
        """Create Q-Q plots for each feature"""
        print("Creating Q-Q plots...")
        os.makedirs(output_dir, exist_ok=True)
        
        n_features = len(self.feature_names)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, feature in enumerate(self.feature_names):
            ax = axes[i]
            
            real_values = self.real_data[feature].values
            synth_values = self.synthetic_data[feature].values
            
            # get quantiles
            quantiles = np.linspace(0, 1, 100)
            real_quantiles = np.quantile(real_values, quantiles)
            synth_quantiles = np.quantile(synth_values, quantiles)
            
            # plot q-q
            ax.scatter(real_quantiles, synth_quantiles, alpha=0.6, s=20)
            
            # add diagonal line
            min_val = min(real_quantiles.min(), synth_quantiles.min())
            max_val = max(real_quantiles.max(), synth_quantiles.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Match')
            
            ax.set_xlabel('Real Data Quantiles', fontsize=10)
            ax.set_ylabel('Synthetic Data Quantiles', fontsize=10)
            ax.set_title(f'Q-Q Plot: {feature}', fontsize=11, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        # hide extra subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        qq_path = os.path.join(output_dir, 'qq_plots.png')
        plt.savefig(qq_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {qq_path}\n")
        
        return qq_path
    
    def correlation_heatmaps(self, output_dir='results/tornado/figures'):
        """Create correlation heatmaps for comparison"""
        print("Creating correlation heatmaps...")
        os.makedirs(output_dir, exist_ok=True)
        
        real_corr = self.real_data.corr()
        synth_corr = self.synthetic_data.corr()
        corr_diff = np.abs(real_corr - synth_corr)
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        # real correlation
        sns.heatmap(real_corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=axes[0],
                   cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
        axes[0].set_title('Real Data Correlation', fontsize=14, fontweight='bold')
        
        # synthetic correlation
        sns.heatmap(synth_corr, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, ax=axes[1],
                   cbar_kws={"shrink": 0.8}, vmin=-1, vmax=1)
        axes[1].set_title('Synthetic Data Correlation', fontsize=14, fontweight='bold')
        
        # difference
        sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='Reds',
                   square=True, linewidths=1, ax=axes[2],
                   cbar_kws={"shrink": 0.8}, vmin=0, vmax=1)
        axes[2].set_title('Absolute Difference', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        corr_path = os.path.join(output_dir, 'correlation_comparison.png')
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {corr_path}\n")
        
        return corr_path
    
    def summary_statistics_comparison(self):
        """Compare summary statistics"""
        print("Comparing summary statistics...")
        
        # use denormalized data
        real_stats = self.real_data_denorm.describe().T
        synth_stats = self.synthetic_data_denorm.describe().T
        
        # key statistics
        comparison = pd.DataFrame({
            'Real_Mean': real_stats['mean'],
            'Synth_Mean': synth_stats['mean'],
            'Real_Std': real_stats['std'],
            'Synth_Std': synth_stats['std'],
            'Real_Min': real_stats['min'],
            'Synth_Min': synth_stats['min'],
            'Real_Max': real_stats['max'],
            'Synth_Max': synth_stats['max'],
        })
        
        # calculate differences
        comparison['Mean_Diff_%'] = np.abs(
            (comparison['Synth_Mean'] - comparison['Real_Mean']) / comparison['Real_Mean'] * 100
        )
        comparison['Std_Diff_%'] = np.abs(
            (comparison['Synth_Std'] - comparison['Real_Std']) / comparison['Real_Std'] * 100
        )
        
        print(comparison.round(3).to_string())
        print()
        
        return comparison
    
    def generate_evaluation_report(self, output_dir='results/tornado'):
        """Generate comprehensive evaluation report"""
        print("\n" + "="*60)
        print("SYNTHETIC DATA EVALUATION REPORT")
        print("="*60 + "\n")
        
        # denormalize data
        self.denormalize_data()
        
        # statistical tests
        stat_results = self.statistical_tests()
        
        # correlation comparison
        real_corr, synth_corr, corr_diff = self.correlation_comparison()
        
        # summary statistics
        stats_comparison = self.summary_statistics_comparison()
        
        # visualizations
        self.distribution_comparison_plots(os.path.join(output_dir, 'figures'))
        self.qq_plots(os.path.join(output_dir, 'figures'))
        self.correlation_heatmaps(os.path.join(output_dir, 'figures'))
        
        # save results
        os.makedirs(output_dir, exist_ok=True)
        
        # save statistical test results
        stat_path = os.path.join(output_dir, 'statistical_tests.csv')
        stat_results.to_csv(stat_path, index=False)
        print(f"Saved statistical tests: {stat_path}")
        
        # save summary comparison
        stats_path = os.path.join(output_dir, 'summary_statistics_comparison.csv')
        stats_comparison.to_csv(stats_path)
        print(f"Saved summary statistics: {stats_path}")
        
        # save synthetic data
        synth_denorm_path = os.path.join(output_dir, 'synthetic_data', 'synthetic_tornadoes.csv')
        os.makedirs(os.path.dirname(synth_denorm_path), exist_ok=True)
        self.synthetic_data_denorm.to_csv(synth_denorm_path, index=False)
        print(f"Saved synthetic data: {synth_denorm_path}")
        
        # save evaluation metrics
        metrics_path = os.path.join(output_dir, 'evaluation_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("SYNTHETIC DATA QUALITY METRICS\n")
            f.write("="*60 + "\n\n")
            f.write(f"Number of real samples: {len(self.real_data)}\n")
            f.write(f"Number of synthetic samples: {len(self.synthetic_data)}\n\n")
            f.write("Overall Similarity Metrics:\n")
            f.write("-"*60 + "\n")
            for key, value in self.metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        print(f"Saved evaluation metrics: {metrics_path}")
        
        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"\nQuality Assessment:")
        print(f"  Mean Correlation Difference: {self.metrics['mean_correlation_difference']:.4f}")
        print(f"  Mean KS Statistic: {self.metrics['mean_ks_statistic']:.4f}")
        print(f"  Mean Wasserstein Distance: {self.metrics['mean_wasserstein']:.4f}")
        print(f"  Mean JS Divergence: {self.metrics['mean_js_divergence']:.4f}")
        print(f"\n  All evaluation results saved to: {output_dir}")
        print("="*60 + "\n")
        
        return self.metrics


def main():
    """Example usage"""
    # load real data
    real_data = pd.read_csv('data/processed/tornado_hilp_normalized.csv')
    
    # load synthetic data (assuming already generated)
    synthetic_data = pd.read_csv('results/tornado/synthetic_data/synthetic_tornadoes_normalized.csv')
    
    # load scaler
    scaler = joblib.load('data/processed/tornado_scaler.pkl')
    
    # load feature names
    with open('data/processed/tornado_features.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    # evaluate synthetic data
    evaluator = SyntheticDataEvaluator(real_data, synthetic_data, feature_names, scaler)
    metrics = evaluator.generate_evaluation_report()
    
    print("Evaluation complete!")


if __name__ == "__main__":
    main()