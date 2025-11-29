"""
Complete Augmentation Pipeline
Runs data preparation, training, generation, and evaluation
"""

import sys
import os
import argparse
from datetime import datetime
import joblib
import pandas as pd

# Add src to path
sys.path.append('src')
sys.path.append('src/augmentation')

from tornado_data_prep import TornadoDataPreparator
from diffusion_model import train_diffusion_model, load_trained_model
from evaluate_synthetic import SyntheticDataEvaluator


def print_banner(text):
    """Print formatted banner"""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def step1_prepare_data():
    """Step 1: Prepare tornado data"""
    print_banner("STEP 1: DATA PREPARATION")
    
    preparator = TornadoDataPreparator()
    normalized_df, scaler = preparator.run_full_preparation()
    
    return normalized_df, scaler


def step2_train_model(normalized_df, num_epochs=1000, batch_size=64):
    """Step 2: Train diffusion model"""
    print_banner("STEP 2: TRAIN DIFFUSION MODEL")
    
    model, trainer = train_diffusion_model(
        normalized_df,
        num_epochs=num_epochs,
        batch_size=batch_size,
        lr=1e-3,
        hidden_dims=[256, 512, 512, 256],
        num_timesteps=1000,
        save_path='models/tornado_diffusion.pt'
    )
    
    return model, trainer


def step3_generate_samples(trainer, num_samples=1000):
    """Step 3: Generate synthetic samples"""
    print_banner("STEP 3: GENERATE SYNTHETIC SAMPLES")
    
    print(f"Generating {num_samples} synthetic tornado events...")
    synthetic_samples = trainer.sample(num_samples=num_samples, verbose=True)
    
    print(f"\nGenerated {len(synthetic_samples)} samples")
    print(f"  Shape: {synthetic_samples.shape}")
    
    # load feature names
    with open('data/processed/tornado_features.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    # create DataFrame
    synthetic_df = pd.DataFrame(synthetic_samples, columns=feature_names)
    
    # save normalized synthetic data
    os.makedirs('results/tornado/synthetic_data', exist_ok=True)
    synth_norm_path = 'results/tornado/synthetic_data/synthetic_tornadoes_normalized.csv'
    synthetic_df.to_csv(synth_norm_path, index=False)
    print(f"Saved normalized synthetic data: {synth_norm_path}\n")
    
    return synthetic_df


def step4_evaluate(real_data, synthetic_data, scaler):
    """Step 4: Evaluate synthetic data"""
    print_banner("STEP 4: EVALUATE SYNTHETIC DATA")
    
    # load feature names
    with open('data/processed/tornado_features.txt', 'r') as f:
        feature_names = [line.strip() for line in f]
    
    # create evaluator
    evaluator = SyntheticDataEvaluator(
        real_data=real_data,
        synthetic_data=synthetic_data,
        feature_names=feature_names,
        scaler=scaler
    )
    
    # generate evaluation report
    metrics = evaluator.generate_evaluation_report(output_dir='results/tornado')
    
    return metrics


def run_full_pipeline(prepare_data=True, train_model=True, num_epochs=1000,
                     batch_size=64, num_samples=1000, model_path=None):
    """
    Run the complete augmentation pipeline
    
    Args:
        prepare_data: Whether to prepare data (or use existing)
        train_model: Whether to train model (or load existing)
        num_epochs: Number of training epochs
        batch_size: Training batch size
        num_samples: Number of synthetic samples to generate
        model_path: Path to existing model (if train_model=False)
    """
    start_time = datetime.now()
    
    print_banner("HILP TORNADO DATA AUGMENTATION PIPELINE")
    print(f"Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Prepare Data: {prepare_data}")
    print(f"  Train Model: {train_model}")
    print(f"  Epochs: {num_epochs if train_model else 'N/A'}")
    print(f"  Batch Size: {batch_size if train_model else 'N/A'}")
    print(f"  Samples to Generate: {num_samples}")
    
    try:
        # step 1: data preparation
        if prepare_data:
            normalized_df, scaler = step1_prepare_data()
        else:
            print_banner("STEP 1: LOADING EXISTING DATA")
            normalized_df = pd.read_csv('data/processed/tornado_hilp_normalized.csv')
            scaler = joblib.load('data/processed/tornado_scaler.pkl')
            print(f"Loaded normalized data: {normalized_df.shape}")
            print(f"Loaded scaler\n")
        
        # step 2: model training
        if train_model:
            model, trainer = step2_train_model(normalized_df, num_epochs, batch_size)
        else:
            print_banner("STEP 2: LOADING EXISTING MODEL")
            if model_path is None:
                model_path = 'models/tornado_diffusion.pt'
            model, trainer = load_trained_model(model_path)
            print(f"Model loaded and ready\n")
        
        # step 3: generate synthetic samples
        synthetic_df = step3_generate_samples(trainer, num_samples)
        
        # step 4: evaluation
        metrics = step4_evaluate(normalized_df, synthetic_df, scaler)
        
        # final summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_banner("PIPELINE COMPLETE")
        
        print("Summary:")
        print(f"  Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        print("\nData:")
        print(f"  Real Samples: {len(normalized_df):,}")
        print(f"  Synthetic Samples: {len(synthetic_df):,}")
        print(f"  Features: {len(normalized_df.columns)}")
        
        print("\nEvaluation Metrics:")
        print(f"  Mean Correlation Difference: {metrics['mean_correlation_difference']:.4f}")
        print(f"  Mean KS Statistic: {metrics['mean_ks_statistic']:.4f}")
        print(f"  Mean Wasserstein Distance: {metrics['mean_wasserstein']:.4f}")
        print(f"  Mean JS Divergence: {metrics['mean_js_divergence']:.4f}")
        
        print("\nOutput Locations:")
        print(f"  Model: models/tornado_diffusion.pt")
        print(f"  Synthetic Data: results/tornado/synthetic_data/")
        print(f"  Evaluation Figures: results/tornado/figures/")
        print(f"  Evaluation Metrics: results/tornado/evaluation_metrics.txt")
        
        print("\nAugmentation pipeline completed successfully!")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main execution with command line arguments"""
    parser = argparse.ArgumentParser(
        description='HILP Tornado Data Augmentation with Diffusion Models'
    )
    
    parser.add_argument(
        '--skip-prep',
        action='store_true',
        help='Skip data preparation (use existing prepared data)'
    )
    
    parser.add_argument(
        '--skip-train',
        action='store_true',
        help='Skip model training (use existing model)'
    )
    
    parser.add_argument(
        '--model-path',
        type=str,
        default='models/tornado_diffusion.pt',
        help='Path to model (for loading or saving)'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=1000,
        help='Number of training epochs (default: 1000)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Training batch size (default: 64)'
    )
    
    parser.add_argument(
        '--num-samples',
        type=int,
        default=1000,
        help='Number of synthetic samples to generate (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # run pipeline
    success = run_full_pipeline(
        prepare_data=not args.skip_prep,
        train_model=not args.skip_train,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        model_path=args.model_path if args.skip_train else None
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()