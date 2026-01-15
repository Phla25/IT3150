import time
import numpy as np
from models.MyGaussian_Base import MyGaussianNB_Base

class GaussianNB_MLE(MyGaussianNB_Base):
    """
    METHOD 1: Pure MLE (No Regularization)
    
    Công thức: σ² = σ²_MLE (không cộng thêm gì)
    Epsilon chỉ để tránh division by zero (≈ 0)
    """
    
    def __init__(self):
        super().__init__(name="MLE")
        
    def fit(self, X, y):
        t0 = time.time()
        
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # Tính statistics
        var_per_feature = np.var(X, axis=0)
        max_var = np.max(var_per_feature)
        mean_var = np.mean(var_per_feature)
        
        # ═══════════════════════════════════════
        # MLE: Epsilon CỰC NHỎ (chỉ tránh /0)
        # ═══════════════════════════════════════
        self.epsilon = 1e-15
        
        print(f"\n{'='*70}")
        print(f"[METHOD 1: MLE - Pure Maximum Likelihood]")
        print(f"{'='*70}")
        print(f"  Formula:       σ² = σ²_MLE (no regularization)")
        print(f"  Epsilon:       {self.epsilon:.2e} (numerical safety only)")
        print(f"  Max variance:  {max_var:.2e}")
        print(f"  Mean variance: {mean_var:.2e}")
        print(f"{'='*70}\n")
        
        # Train từng class
        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c)
            
            self.class_prior[c] = (n_c + 1) / (n_samples + len(self.classes))
            self.mean[c] = np.mean(X_c, axis=0)
            
            # Pure MLE variance
            var_mle = np.var(X_c, axis=0)
            self.var[c] = var_mle + self.epsilon
            
            print(f"  Class '{c}': N={n_c:>6,} | Avg Var={np.mean(var_mle):.3e}")
        
        self.train_time = time.time() - t0
        print(f"\n✓ Training done in {self.train_time:.2f}s\n")
        return self