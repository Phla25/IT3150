import numpy as np
from models.MyGaussian_Base import MyGaussianNB_Base
import time
class GaussianNB_MAP(MyGaussianNB_Base):
    """
    METHOD 2: MAP Estimation với Inverse-Gamma Prior
    
    Công thức (simplified): ε = mean_var / N
    Lý thuyết: σ²_MAP ≈ σ²_MLE + (2β - σ²_MLE(2α+2)) / N
    """
    
    def __init__(self, alpha=1.0, use_full_formula=False):
        super().__init__(name="MAP")
        self.alpha = alpha
        self.use_full_formula = use_full_formula
        
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
        # MAP: Epsilon từ Bayesian Prior
        # ═══════════════════════════════════════
        if self.use_full_formula:
            # Full formula với correction term
            beta = mean_var
            epsilon_pos = 2 * beta / n_samples
            epsilon_neg = mean_var * (2*self.alpha + 2) / n_samples
            self.epsilon = epsilon_pos - epsilon_neg
            self.epsilon = max(self.epsilon, 1e-9 * max_var)
            formula = f"ε = (2β - σ²(2α+2)) / N"
        else:
            # Simplified formula (chỉ term dương)
            self.epsilon = mean_var / n_samples
            formula = f"ε = mean_var / N"
        
        # Safety net
        min_epsilon = 1e-15 * max_var
        if self.epsilon < min_epsilon:
            self.epsilon = min_epsilon
        
        print(f"\n{'='*70}")
        print(f"[METHOD 2: MAP - Bayesian Estimation]")
        print(f"{'='*70}")
        print(f"  Formula:       {formula}")
        print(f"  Prior:         InverseGamma(α={self.alpha})")
        print(f"  N samples:     {n_samples:,}")
        print(f"  Mean variance: {mean_var:.3e}")
        print(f"  Epsilon:       {self.epsilon:.3e}")
        print(f"  Ratio ε/max:   {self.epsilon/max_var:.3e}")
        print(f"{'='*70}\n")
        
        # Train từng class
        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c)
            
            self.class_prior[c] = (n_c + 1) / (n_samples + len(self.classes))
            self.mean[c] = np.mean(X_c, axis=0)
            
            var_mle = np.var(X_c, axis=0)
            self.var[c] = var_mle + self.epsilon
            
            print(f"  Class '{c}': N={n_c:>6,} | "
                  f"Var(MLE)={np.mean(var_mle):.3e} | "
                  f"Var(MAP)={np.mean(self.var[c]):.3e}")
        
        self.train_time = time.time() - t0
        print(f"\n✓ Training done in {self.train_time:.2f}s\n")
        return self