import numpy as np
import time
from models.MyGaussian_Base import MyGaussianNB_Base
class GaussianNB_Sklearn(MyGaussianNB_Base):
    """
    METHOD 3: Sklearn-style Fixed Smoothing
    
    Công thức: ε = var_smoothing × max(variance)
    Default: var_smoothing = 1e-9
    """
    
    def __init__(self, var_smoothing=1e-9):
        super().__init__(name="Sklearn")
        self.var_smoothing = var_smoothing
        
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
        # SKLEARN: Fixed ratio smoothing
        # ═══════════════════════════════════════
        self.epsilon = self.var_smoothing * max_var
        
        print(f"\n{'='*70}")
        print(f"[METHOD 3: Sklearn - Fixed Smoothing]")
        print(f"{'='*70}")
        print(f"  Formula:       ε = var_smoothing × max(var)")
        print(f"  var_smoothing: {self.var_smoothing:.2e}")
        print(f"  Max variance:  {max_var:.3e}")
        print(f"  Epsilon:       {self.epsilon:.3e}")
        print(f"  (Independent of N)")
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
                  f"Var(Final)={np.mean(self.var[c]):.3e}")
        
        self.train_time = time.time() - t0
        print(f"\n✓ Training done in {self.train_time:.2f}s\n")
        return self
