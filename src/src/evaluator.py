"""
Evaluation utilities for TrOCR HTR models.
Implements HTR-specific metrics like CER, WER, and BLEU.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional
import unicodedata

import numpy as np
from jiwer import wer, cer
from rapidfuzz import fuzz
import editdistance
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTREvaluator:
    """Evaluator for Handwritten Text Recognition tasks."""
    
    def __init__(self, eval_config: Dict):
        self.config = eval_config
        self.case_sensitive = eval_config.get('case_sensitive', True)
        self.normalize_whitespace = eval_config.get('normalize_whitespace', True)
        
    def normalize_text(self, text: str) -> str:
        """
        Normalize text for evaluation.
        
        Args:
            text: Raw text string
            
        Returns:
            Normalized text
        """
        # Remove excessive whitespace
        if self.normalize_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        # Case normalization
        if not self.case_sensitive:
            text = text.lower()
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        return text
    
    def compute_cer(self, prediction: str, reference: str) -> float:
        """
        Compute Character Error Rate.
        
        Args:
            prediction: Predicted text
            reference: Reference text
            
        Returns:
            CER value (0.0 = perfect, 1.0 = completely wrong)
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        if len(ref_norm) == 0:
            return 0.0 if len(pred_norm) == 0 else 1.0
            
        return cer(ref_norm, pred_norm)
    
    def compute_wer(self, prediction: str, reference: str) -> float:
        """
        Compute Word Error Rate.
        
        Args:
            prediction: Predicted text
            reference: Reference text
            
        Returns:
            WER value (0.0 = perfect, 1.0 = completely wrong)
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        if len(ref_norm.split()) == 0:
            return 0.0 if len(pred_norm.split()) == 0 else 1.0
            
        return wer(ref_norm, pred_norm)
    
    def compute_exact_match(self, prediction: str, reference: str) -> float:
        """
        Compute exact match accuracy.
        
        Args:
            prediction: Predicted text
            reference: Reference text
            
        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        return 1.0 if pred_norm == ref_norm else 0.0
    
    def compute_edit_distance(self, prediction: str, reference: str) -> int:
        """
        Compute Levenshtein edit distance.
        
        Args:
            prediction: Predicted text
            reference: Reference text
            
        Returns:
            Edit distance (number of character operations)
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        return editdistance.eval(pred_norm, ref_norm)
    
    def compute_similarity_scores(self, prediction: str, reference: str) -> Dict[str, float]:
        """
        Compute various text similarity scores.
        
        Args:
            prediction: Predicted text
            reference: Reference text
            
        Returns:
            Dictionary with similarity scores
        """
        pred_norm = self.normalize_text(prediction)
        ref_norm = self.normalize_text(reference)
        
        return {
            'ratio': fuzz.ratio(pred_norm, ref_norm) / 100.0,
            'partial_ratio': fuzz.partial_ratio(pred_norm, ref_norm) / 100.0,
            'token_sort_ratio': fuzz.token_sort_ratio(pred_norm, ref_norm) / 100.0,
            'token_set_ratio': fuzz.token_set_ratio(pred_norm, ref_norm) / 100.0
        }
    
    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """
        Compute BLEU score for the entire dataset.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            BLEU score (0.0 to 1.0)
        """
        try:
            from nltk.translate.bleu_score import corpus_bleu
            import nltk
            nltk.download('punkt', quiet=True)
            
            # Tokenize
            ref_tokens = []
            pred_tokens = []
            
            for pred, ref in zip(predictions, references):
                pred_norm = self.normalize_text(pred)
                ref_norm = self.normalize_text(ref)
                
                pred_tokens.append(pred_norm.split())
                ref_tokens.append([ref_norm.split()])  # BLEU expects list of references
            
            return corpus_bleu(ref_tokens, pred_tokens)
            
        except ImportError:
            logger.warning("NLTK not available, BLEU score set to 0.0")
            return 0.0
    
    def compute_length_statistics(self, predictions: List[str], references: List[str]) -> Dict:
        """
        Compute length-related statistics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with length statistics
        """
        pred_lengths = [len(self.normalize_text(pred)) for pred in predictions]
        ref_lengths = [len(self.normalize_text(ref)) for ref in references]
        
        return {
            'avg_pred_length': np.mean(pred_lengths),
            'avg_ref_length': np.mean(ref_lengths),
            'length_ratio': np.mean(pred_lengths) / np.mean(ref_lengths) if np.mean(ref_lengths) > 0 else 0.0,
            'pred_length_std': np.std(pred_lengths),
            'ref_length_std': np.std(ref_lengths)
        }
    
    def compute_metrics(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute comprehensive evaluation metrics.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            
        Returns:
            Dictionary with all computed metrics
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        metrics = {}
        
        # Character and word error rates
        cers = [self.compute_cer(pred, ref) for pred, ref in zip(predictions, references)]
        wers = [self.compute_wer(pred, ref) for pred, ref in zip(predictions, references)]
        
        metrics['cer'] = np.mean(cers)
        metrics['wer'] = np.mean(wers)
        metrics['cer_std'] = np.std(cers)
        metrics['wer_std'] = np.std(wers)
        
        # Exact match accuracy
        exact_matches = [self.compute_exact_match(pred, ref) for pred, ref in zip(predictions, references)]
        metrics['exact_match'] = np.mean(exact_matches)
        
        # Edit distance
        edit_distances = [self.compute_edit_distance(pred, ref) for pred, ref in zip(predictions, references)]
        metrics['avg_edit_distance'] = np.mean(edit_distances)
        metrics['edit_distance_std'] = np.std(edit_distances)
        
        # BLEU score
        if 'bleu' in self.config.get('metrics', []):
            metrics['bleu'] = self.compute_bleu_score(predictions, references)
        
        # Similarity scores (average across all samples)
        similarity_scores = [self.compute_similarity_scores(pred, ref) for pred, ref in zip(predictions, references)]
        for score_type in ['ratio', 'partial_ratio', 'token_sort_ratio', 'token_set_ratio']:
            metrics[f'similarity_{score_type}'] = np.mean([sim[score_type] for sim in similarity_scores])
        
        # Length statistics
        length_stats = self.compute_length_statistics(predictions, references)
        metrics.update(length_stats)
        
        # Performance by length bins
        metrics.update(self._compute_length_binned_metrics(predictions, references))
        
        return metrics
    
    def _compute_length_binned_metrics(self, predictions: List[str], references: List[str]) -> Dict:
        """Compute metrics binned by reference text length."""
        binned_metrics = {}
        
        # Define length bins
        ref_lengths = [len(self.normalize_text(ref)) for ref in references]
        bins = [0, 50, 100, 200, float('inf')]
        bin_labels = ['0-50', '50-100', '100-200', '200+']
        
        for i, (bin_start, bin_end) in enumerate(zip(bins[:-1], bins[1:])):
            # Filter samples in this bin
            bin_indices = [j for j, length in enumerate(ref_lengths) 
                          if bin_start <= length < bin_end]
            
            if not bin_indices:
                continue
                
            bin_preds = [predictions[j] for j in bin_indices]
            bin_refs = [references[j] for j in bin_indices]
            
            # Compute metrics for this bin
            bin_cers = [self.compute_cer(pred, ref) for pred, ref in zip(bin_preds, bin_refs)]
            bin_wers = [self.compute_wer(pred, ref) for pred, ref in zip(bin_preds, bin_refs)]
            
            label = bin_labels[i]
            binned_metrics[f'cer_{label}'] = np.mean(bin_cers)
            binned_metrics[f'wer_{label}'] = np.mean(bin_wers)
            binned_metrics[f'count_{label}'] = len(bin_indices)
        
        return binned_metrics
    
    def generate_detailed_report(self, predictions: List[str], references: List[str], 
                               image_paths: Optional[List[str]] = None) -> Dict:
        """
        Generate detailed evaluation report with sample analysis.
        
        Args:
            predictions: List of predicted texts
            references: List of reference texts
            image_paths: Optional list of image paths for reference
            
        Returns:
            Detailed report dictionary
        """
        metrics = self.compute_metrics(predictions, references)
        
        # Find best and worst samples
        cers = [self.compute_cer(pred, ref) for pred, ref in zip(predictions, references)]
        
        # Best samples (lowest CER)
        best_indices = np.argsort(cers)[:5]
        worst_indices = np.argsort(cers)[-5:]
        
        report = {
            'overall_metrics': metrics,
            'total_samples': len(predictions),
            'best_samples': [],
            'worst_samples': [],
            'error_analysis': self._analyze_errors(predictions, references)
        }
        
        # Add best samples
        for idx in best_indices:
            sample = {
                'index': int(idx),
                'cer': float(cers[idx]),
                'prediction': predictions[idx],
                'reference': references[idx]
            }
            if image_paths:
                sample['image_path'] = image_paths[idx]
            report['best_samples'].append(sample)
        
        # Add worst samples
        for idx in worst_indices:
            sample = {
                'index': int(idx),
                'cer': float(cers[idx]),
                'prediction': predictions[idx],
                'reference': references[idx]
            }
            if image_paths:
                sample['image_path'] = image_paths[idx]
            report['worst_samples'].append(sample)
        
        return report
    
    def _analyze_errors(self, predictions: List[str], references: List[str]) -> Dict:
        """Analyze common error patterns."""
        error_patterns = {
            'common_substitutions': {},
            'common_insertions': {},
            'common_deletions': {},
            'avg_errors_per_sample': 0.0
        }
        
        total_errors = 0
        
        for pred, ref in zip(predictions, references):
            pred_norm = self.normalize_text(pred)
            ref_norm = self.normalize_text(ref)
            
            # Simple character-level error analysis
            edit_ops = self._get_edit_operations(pred_norm, ref_norm)
            total_errors += len(edit_ops)
            
            for op, char1, char2 in edit_ops:
                if op == 'substitute':
                    key = f"{char1} -> {char2}"
                    error_patterns['common_substitutions'][key] = error_patterns['common_substitutions'].get(key, 0) + 1
                elif op == 'insert':
                    error_patterns['common_insertions'][char2] = error_patterns['common_insertions'].get(char2, 0) + 1
                elif op == 'delete':
                    error_patterns['common_deletions'][char1] = error_patterns['common_deletions'].get(char1, 0) + 1
        
        error_patterns['avg_errors_per_sample'] = total_errors / len(predictions) if predictions else 0.0
        
        # Sort by frequency
        for key in ['common_substitutions', 'common_insertions', 'common_deletions']:
            error_patterns[key] = dict(sorted(error_patterns[key].items(), key=lambda x: x[1], reverse=True)[:10])
        
        return error_patterns
    
    def _get_edit_operations(self, pred: str, ref: str) -> List[Tuple[str, str, str]]:
        """Get list of edit operations (simplified implementation)."""
        # This is a simplified version - for detailed analysis, 
        # you might want to use a proper alignment algorithm
        operations = []
        
        if len(pred) != len(ref):
            # Length difference indicates insertions/deletions
            if len(pred) > len(ref):
                diff = len(pred) - len(ref)
                operations.extend([('insert', '', pred[i]) for i in range(len(ref), len(pred))])
            else:
                diff = len(ref) - len(pred)
                operations.extend([('delete', ref[i], '') for i in range(len(pred), len(ref))])
        
        # Character substitutions
        min_len = min(len(pred), len(ref))
        for i in range(min_len):
            if pred[i] != ref[i]:
                operations.append(('substitute', ref[i], pred[i]))
        
        return operations


if __name__ == "__main__":
    # Test evaluator
    evaluator = HTREvaluator({'case_sensitive': True, 'normalize_whitespace': True})
    
    # Test data
    predictions = [
        "Hello world",
        "This is a test",
        "Constance de Salm"
    ]
    
    references = [
        "Hello world",
        "This is a text",
        "Constance de Salme"
    ]
    
    # Compute metrics
    metrics = evaluator.compute_metrics(predictions, references)
    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Generate detailed report
    report = evaluator.generate_detailed_report(predictions, references)
    print(f"\nTotal samples: {report['total_samples']}")
    print(f"Overall CER: {report['overall_metrics']['cer']:.4f}")
