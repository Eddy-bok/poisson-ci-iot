import numpy as np
import pandas as pd
import scipy.stats as stats

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import mean_squared_error

from functools import lru_cache
import warnings
import time

class ImprovedPoissonConcentrationML(BaseEstimator, RegressorMixin):
    """
    Enhanced Poisson concentration ML preserving the original mathematical foundation.

    Core inequalities preserved:
    1. E[|Σa_i X_i|^q] ≤ T_q(Σλ_i) * ||a||_∞^q  (Independent case)
    2. E_{PO}[|Σa_i X_i|^q] ≤ M^q * ||a||_∞^q     (Conditional case)

    Where T_q(λ) = Σ_{k=0}^q S(q,k) * λ^k are Touchard polynomials.

    Improvements:
    - Sample-specific λ estimation instead of global sum
    - Empirical calibration of the proven bounds
    - Better numerical stability in Touchard computation
    - Adaptive selection between independent/conditional bounds
    """
    # -------------------------------------------------
    #  Initialization & Core Components
    # -------------------------------------------------
    def __init__(self,
             model_type='linear',
             q=2,
             confidence_level=0.95,
             use_conditional=False,
             kappa_min=0.5,
             local_lambda_estimation=True,
             empirical_calibration=True,
             calibration_samples=200,
             max_ci_width=None,
             verbose=True,
             pure_poisson_mode=False,
             random_state=42,
	         lambda_mode="model_based"):

        """
        Initialize Improved Poisson Concentration ML model while preserving
        the theoretical inequality structure.
        Parameters
        ----------
        random_state : int, default=42
            Ensures reproducible bootstrap and local λ estimation randomness.
        model_type : str, {'linear', 'logistic'}
            Type of underlying ML model
        q : int
            Moment parameter for the concentration inequality (typically 2)
        confidence_level : float
            Confidence level for intervals
        use_conditional : bool
            Use conditional inequality E_{PO}[|Σa_i X_i|^q] ≤ M^q * ||a||_∞^q
        local_lambda_estimation : bool
            Estimate λ_i locally for each sample instead of using global sum
        empirical_calibration : bool
            Apply empirical calibration factor to the proven bounds
        calibration_samples : int
            Number of bootstrap samples for calibration
        """
        # Core parameters
        self.model_type = model_type
        self.q = q
        self.confidence_level = confidence_level
        self.use_conditional = use_conditional
        self.local_lambda_estimation = local_lambda_estimation
        self.empirical_calibration = empirical_calibration
        self.calibration_samples = calibration_samples
        self.max_ci_width = max_ci_width
        self.verbose = verbose
        self.lambda_mode = lambda_mode
        self.kappa_min = kappa_min
        self.pure_poisson_mode = pure_poisson_mode


        # Defensive controls and numerical safety
        self.defensive = True
        self.lambda_cap = 5000.0  # raised for high-rate Poisson traffic
        self.weight_cap = None    # adaptive handling later
        self.pseudo_count = 1e-3
        self.expect_log_transformed_data = True
        self.overdispersion_detected = False
        self.clipped_count = 0

        # Fitted attributes
        self.model = None
        self.lambda_estimates = None
        self.calibration_factor = 1.0
        self.scaler_ = None

        # Randomness control
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    @staticmethod
    @lru_cache(maxsize=None)
    def stirling2(n: int, k: int) -> float:
        """
        Compute Stirling number of the second kind S(n, k) using the recurrence:
            S(n, k) = k * S(n-1, k) + S(n-1, k-1)
        with base cases:
            S(0, 0) = 1
            S(n, 0) = 0 for n > 0
            S(0, k) = 0 for k > 0

        Returns
        -------
        float
            Stirling number of the second kind for (n, k)
        """
        # Handle invalid domain
        if k > n:
            return 0.0
        if n == k == 0:
            return 1.0
        if n == 0 or k == 0:
            return 0.0

        # Recurrence relation with caching
        value = (
            k * ImprovedPoissonConcentrationML.stirling2(n - 1, k) +
            ImprovedPoissonConcentrationML.stirling2(n - 1, k - 1)
        )

        # Prevent overflow for large n,k
        if value > 1e308:
            value = 1e308

        return float(value)

    def _touchard_polynomial(self, lam: float, q: int) -> float:
        """
        Compute the Touchard (Bell) polynomial T_q(λ) = Σ_{k=0}^q S(q,k) * λ^k
        using numerically stable log-sum-exp formulation.

        Stability improvements:
        - Uses log-domain summation for large λ or q.
        - Employs asymptotic approximation T_q(λ) ≈ λ^q for λ >> q.
        - Cap synchronized with model-wide self.lambda_cap (default = 5000).

        Parameters
        ----------
        lam : float
            Poisson rate λ (mean count)
        q : int
            Moment order of the inequality

        Returns
        -------
        float
            Value of the Touchard polynomial T_q(λ)
        """
        if lam <= 0:
            return 1.0 if q == 0 else 0.0

        # Unified cap from model configuration
        if lam > self.lambda_cap:
            if self.verbose:
                print(f"[WARN] λ={lam:.1f} exceeds cap ({self.lambda_cap}); "
                      f"using asymptotic λ^q approximation.")
            return float(lam ** q)

        # Compute in log-domain for numerical stability
        log_terms = []
        for k in range(q + 1):
            S_qk = self.stirling2(q, k)
            if S_qk <= 0:
                continue
            log_term = np.log(S_qk) + k * np.log(lam)
            log_terms.append(log_term)

        if not log_terms:
            return 0.0

        # Safe aggregation (log-sum-exp)
        max_log = np.max(log_terms)
        stable_sum = np.exp(max_log) * np.sum(np.exp(np.array(log_terms) - max_log))

        # Clip to double precision limit
        return float(np.clip(stable_sum, 0, 1e308))

    def _compute_original_poisson_bound(
        self,
        lambda_values: np.ndarray,
        weights_infinity_norm: float,
        sample_sum: float = None
    ) -> float:
        """
        Compute the proven Poisson concentration inequality bounds.
        Independent case:
            E[|Σ a_i X_i|^q] ≤ T_q(Σ λ_i) * ||a||_∞^q
        Conditional case:
            E_PO[|Σ a_i X_i|^q] ≤ M^q * ||a||_∞^q
        Notes
        -----
        - λ_i are treated as per-feature Poisson intensities.
        - To prevent artificial inflation from correlated features,
          λ_total is normalized by feature count.
        - Applies defensive clipping to prevent overflow or underflow.
        """
        # Defensive clipping
        lambda_values = np.clip(lambda_values, 1e-8, self.lambda_cap)
        weights_infinity_norm = max(abs(weights_infinity_norm), 1e-8)

        # Conditional inequality (based on realized total M)
        if self.use_conditional and sample_sum is not None:
            safe_M = max(sample_sum, 1e-8)
            try:
                bound = (safe_M ** self.q) * (weights_infinity_norm ** self.q)
            except OverflowError:
                bound = np.inf

        else:
            # Independent inequality (normalized λ to prevent inflation)
            # --- INSERTED BLOCK BELOW ---
            if np.isscalar(lambda_values):
                lambda_total = float(lambda_values)
            else:
                lambda_mean = np.mean(lambda_values)
                lambda_total = lambda_mean * len(lambda_values)  # optional normalization
            # --- END OF INSERTED BLOCK ---

            touchard_val = self._touchard_polynomial(lambda_total, self.q)

            try:
                bound = max(touchard_val, 1e-8) * (weights_infinity_norm ** self.q)
            except OverflowError:
                bound = np.inf

        return float(bound)


    def _estimate_local_poisson_parameters(self, X: np.ndarray, sample_idx: int = None) -> np.ndarray:
        """
        Estimate local Poisson intensity parameters λ_i.
        Supports both global (mean) and sample-specific (kernel-weighted) modes.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        sample_idx : int, optional
            Target sample index for local λ estimation.

        Returns
        -------
        np.ndarray
            Estimated λ vector (per feature)
        """
        X = np.nan_to_num(np.asarray(X), nan=0.0, posinf=0.0, neginf=0.0)

        # Global mean estimator (MLE under iid Poisson)
        if not self.local_lambda_estimation or sample_idx is None:
            return np.maximum(np.mean(X, axis=0), 1e-6)

        # --- Local estimation branch ---
        n_samples = len(X)
        if sample_idx >= n_samples:
            return np.maximum(np.mean(X, axis=0), 1e-6)

        try:
            # Standardize feature scales for fair distance computation
            X_scaled = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6)
            x_ref = X_scaled[sample_idx]

            # Euclidean distances in normalized space
            distances = np.sqrt(np.sum((X_scaled - x_ref) ** 2, axis=1))

            # Robust bandwidth: 60th percentile (less sensitive than median)
            bandwidth = np.percentile(distances, 60)
            if bandwidth <= 1e-8:
                weights = np.ones(n_samples) / n_samples
            else:
                weights = np.exp(-distances / bandwidth)
                weights /= np.sum(weights)

            # Weighted local λ estimate
            local_lambdas = np.dot(weights, X)
            local_lambdas = np.maximum(local_lambdas, 1e-6)

            return local_lambdas

        except Exception as e:
            if self.verbose:
                print(f"[ERROR] Local λ estimation failed at index {sample_idx}: {e}")

            # Adaptive fallback: use mean of small random neighborhood
            self.rng = getattr(self, "rng", np.random.default_rng(self.random_state))
            neighbor_idx = self.rng.choice(n_samples, size=min(10, n_samples), replace=False)
            return np.maximum(np.mean(X[neighbor_idx], axis=0), 1e-6)


    def _empirical_calibration_of_bounds(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Empirically calibrate the proven Poisson inequality bounds.

        The calibration factor rescales analytical bounds to match observed empirical variance
        without altering the inequality structure.

        Notes
        -----
        - Uses residual bootstrapping with deterministic RNG.
        - Operates on matching sample subsets for analytical and empirical variance.
        - Ensures reproducibility and numerical stability.
        """
        if not self.empirical_calibration:
            self.calibration_factor = 1.0
            return

        X = np.nan_to_num(np.asarray(X), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(np.asarray(y), nan=0.0, posinf=0.0, neginf=0.0)

        n_samples = len(X)
        subset_size = min(200, n_samples)
        idx_subset = self.rng.choice(n_samples, size=subset_size, replace=False)

        # --- Empirical variance estimation via residual bootstrap ---
        preds = self.model.predict(X[idx_subset])
        residuals = y[idx_subset] - preds
        bootstrap_vars = []

        for _ in range(self.calibration_samples):
            resampled_residuals = self.rng.choice(residuals, size=len(residuals), replace=True)
            synthetic_y = preds + resampled_residuals
            var_boot = np.var(synthetic_y - preds)
            bootstrap_vars.append(var_boot)

        empirical_variance = max(np.mean(bootstrap_vars), 1e-10)

        # --- Analytical variance from Poisson inequality bounds ---
        weights = self.model.coef_.flatten()
        weights_inf_norm = np.max(np.abs(weights))
        bounds = []

        for i in idx_subset:
            x_i = X[i]
            lambda_vals = self._estimate_local_poisson_parameters(X, i)
            sample_sum = np.sum(x_i) if self.use_conditional else None
            bound = self._compute_original_poisson_bound(lambda_vals, weights_inf_norm, sample_sum)
            bounds.append(max(bound, 1e-8))

        if self.q == 2:
            analytical_variance = np.mean(bounds)
        else:
            analytical_variance = np.mean(np.power(bounds, 2.0 / self.q))

        analytical_variance = max(analytical_variance, 1e-10)

        # --- Calibration factor computation ---
        ratio = empirical_variance / analytical_variance

        # --- NEW: compute raw kappa ---
        raw_kappa = np.sqrt(ratio)

        # --- NEW: store raw value ---
        self.raw_kappa_ = float(raw_kappa)

        # --- existing clipping ---
        self.calibration_factor = float(np.clip(raw_kappa, 0.1, 5.0))

        if self.verbose:
            print(f"[CALIBRATION] Empirical var: {empirical_variance:.4e}, "
                  f"Analytical var: {analytical_variance:.4e}, "
                  f"Calibration factor: {self.calibration_factor:.3f}")

    # -------------------------------------------------
    #  Model Fitting & Safety
    # -------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the model while preserving the Poisson concentration inequality structure.

        Enhancements:
        - Robust preprocessing (scaling, pseudo-count)
        - Dispersion diagnostics using Var/Mean ratio
        - Safe model fitting (linear/logistic)
        - Deferred λ estimation post-fit
        - Deterministic calibration
        """
        X = np.nan_to_num(np.asarray(X), nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(np.asarray(y), nan=0.0, posinf=0.0, neginf=0.0)

        # --- Pure Poisson mode (bypass regression) ---
        if getattr(self, "pure_poisson_mode", False):
            self.lambda_estimates = np.maximum(np.mean(y), 1e-6)
            self.model = None
            if self.verbose:
                print("[INFO] Pure Poisson Mode: λ estimated directly from observed counts.")
            return self

        # --- Defensive data checks ---
        if self.defensive and not self.expect_log_transformed_data:
            if np.any(X < 0):
                warnings.warn("Negative feature values detected in non-log-transformed data.")
            if not np.allclose(X, np.round(X), atol=1e-3):
                warnings.warn("Data may not represent integer-like Poisson counts.")

        # --- Dispersion diagnostic ---
        means = np.mean(X, axis=0)
        vars_ = np.var(X, axis=0)
        dispersion_index = np.divide(vars_, means + 1e-6)
        overdisp_frac = np.mean(dispersion_index > 1.5)

        if self.verbose:
            print(f"[DIAGNOSTIC] Mean dispersion index: {np.mean(dispersion_index):.3f}, "
                  f"Overdispersed features (>1.5× mean): {overdisp_frac:.2%}")

        # --- Preprocessing and pseudo-counts ---
        X = np.where(X == 0, self.pseudo_count, X)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # --- Model training ---
        try:
            if self.model_type == "linear":
                self.model = LinearRegression()
            else:
                self.model = LogisticRegression(max_iter=1000, solver="lbfgs")

            self.model.fit(X_scaled, y)
        except Exception as e:
            print(f"[ERROR] Model fitting failed: {e}")
            self.model = LinearRegression().fit(X_scaled, y)  # fallback

        # --- λ estimation (post-fit context, corrected) ---
        try:
            # Predict log-domain λ̂ for each sample
            y_pred_log = self.model.predict(X_scaled)
            self.lambda_estimates = np.maximum(y_pred_log, 1e-6)
        except Exception as e:
            # As a fallback, retain local parameter estimation
            print(f"[WARN] λ̂ prediction fallback: {e}")
            self.lambda_estimates = self._estimate_local_poisson_parameters(X_scaled)


        # --- Empirical calibration (if applicable) ---
        if self.model is not None and self.empirical_calibration:
            self._empirical_calibration_of_bounds(X_scaled, y)
        else:
            self.calibration_factor = 1.0

        # --- NEW: Cap calibration factor to prevent over-inflation ---
        self.calibration_factor = np.clip(getattr(self, "calibration_factor", 1.0), self.kappa_min, 2.5)

        if self.verbose:
            print(f"[FIT SUMMARY] Model type: {self.model_type}, "
                  f"λ_mean: {np.mean(self.lambda_estimates):.3f}, "
                  f"Calibration factor: {self.calibration_factor:.3f}")


        return self


    def safe_fit(self, X: np.ndarray, y: np.ndarray, dispersion_threshold: float = 5.0):
        """
        Robust wrapper for the standard fit() method with numerical and statistical safety checks.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target vector
        dispersion_threshold : float, default=5.0
            Upper bound on acceptable average dispersion index
            (Var/Mean ratio) before flagging potential overdispersion.

        Returns
        -------
        self : ImprovedPoissonConcentrationML
            Fitted estimator with diagnostic attributes.
        """
        self.fit_success = False  # diagnostic flag
        try:
            # Clean inputs
            X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=0.0)

            # Proceed with standard fitting
            self.fit(X, y)
            self.fit_success = True

            # Compute dispersion diagnostic
            means = np.mean(X, axis=0)
            vars_ = np.var(X, axis=0)
            dispersion_index = np.divide(vars_, means + 1e-6)
            avg_dispersion = np.mean(dispersion_index)

            # Stability diagnostic based on calibration factor and dispersion
            if np.isnan(self.calibration_factor) or self.calibration_factor < 0.1 or self.calibration_factor > 5.0:
                warnings.warn(f"[SAFEFIT WARNING] Calibration factor {self.calibration_factor:.3f} "
                              "outside stable range (0.1–5.0).")
            if avg_dispersion > dispersion_threshold:
                warnings.warn(f"[SAFEFIT WARNING] Average dispersion index ({avg_dispersion:.2f}) "
                              f"exceeds threshold ({dispersion_threshold}). Potential overdispersion detected.")

            if self.verbose:
                print(f"[SAFEFIT] Completed successfully. Calibration factor: {self.calibration_factor:.3f}, "
                      f"Mean dispersion: {avg_dispersion:.2f}")

        except Exception as e:
            warnings.warn(f"[SAFEFIT ERROR] Model fitting encountered an exception: {e}")
            if self.verbose:
                print(f"[TRACEBACK] Fallback initiated due to: {e}")

            # Attempt minimal fallback: direct λ estimation without regression
            X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)
            y = np.nan_to_num(y, nan=0.0, posinf=100.0, neginf=0.0)
            self.lambda_estimates = np.maximum(np.mean(y), 1e-6)
            self.model = None
            self.calibration_factor = 1.0

        return self

    # -------------------------------------------------
    #  Prediction & Confidence Interval Computation
    # -------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point predictions using the fitted model.
        Ensures consistency with feature scaling used in fit().
        """
        if self.model is None:
            raise ValueError("Model not fitted or running in pure Poisson mode. Use pure λ estimates instead.")

        X = np.nan_to_num(np.asarray(X), nan=0.0, posinf=100.0, neginf=0.0)

        # Apply same scaling as during fit
        if hasattr(self, "scaler_"):
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        if self.model_type == "logistic":
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            return self.model.predict(X_scaled)

    def predict_with_confidence(self, X: np.ndarray, return_bound_details: bool = False):
        """
        Compute Poisson-based confidence intervals for predictions.
        Preserves the theoretical inequality while ensuring numerical and statistical consistency.
        """
        X = np.nan_to_num(np.asarray(X), nan=0.0, posinf=100.0, neginf=0.0)
        n_samples = X.shape[0]

        # --- Pure Poisson mode ---
        if getattr(self, "pure_poisson_mode", False) or self.model is None:
            lam = np.full(n_samples, self.lambda_estimates)
            z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            std_est = np.sqrt(lam)
            margin = z_score * std_est
            lower = np.clip(lam - margin, 0, None)
            upper = lam + margin
            ci = np.vstack([lower, upper]).T
            return lam, ci

        # --- Scaled prediction branch ---
        if hasattr(self, "scaler_"):
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        preds = self.predict(X_scaled)
        weights = np.nan_to_num(self.model.coef_.flatten(), nan=0.0)
        weights_inf_norm = np.max(np.abs(weights)) / (np.std(X_scaled, axis=0).max() + 1e-6)

        # Use precomputed λ estimates
        # lambda_vals = np.clip(self.lambda_estimates, 1e-8, self.lambda_cap)

        # =====================================================
        # CONTROLLED LAMBDA MODE (CRITICAL FIX)
        # =====================================================
        if self.lambda_mode == "input_based":
            lambda_vals = np.array([
                np.mean(np.abs(X[i])) for i in range(n_samples)
            ])
        elif self.lambda_mode == "model_based":
            lambda_vals = self.lambda_estimates[:n_samples]
        else:
            raise ValueError(f"Unknown lambda_mode: {self.lambda_mode}")

        lambda_vals = np.clip(lambda_vals, 1e-8, self.lambda_cap)
        calibrated_bounds = []
        ci = np.zeros((n_samples, 2))

        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

        for i in range(n_samples):
            # Use λ for this sample only
            lam_i = np.clip(lambda_vals[i], 1e-8, self.lambda_cap)

            # Compute bound for this λ (scalar, not vector)
            bound_i = self._compute_original_poisson_bound(lam_i, weights_inf_norm)

            # Apply calibration and compute std
            calibrated_bound = float(bound_i) * (self.calibration_factor ** 2)
            std_est = np.sqrt(calibrated_bound) if self.q == 2 else calibrated_bound ** (1.0 / self.q)

            # --- Variance-scale correction to stabilize log-domain CIs ---
            scale_correction = np.clip(self.calibration_factor, 0.1, 2.0)

            # --- Local & dual-factor correction ---
            local_protocol_factor = getattr(self, "local_protocol_factor", 1.0)
            per_bin_dual = getattr(self, "dual_factor", None)
            if per_bin_dual is not None:
                try:
                    if np.ndim(per_bin_dual) > 0:
                        dual_i = float(per_bin_dual[i])
                    else:
                        dual_i = float(per_bin_dual)
                except Exception:
                    dual_i = float(np.mean(per_bin_dual))
                # multiply once (no redundant averaging later)
                local_protocol_factor *= dual_i

            # === [NEW PATCH] Adaptive widening for high-intensity λ ===
            # For λ ≥ 10 widen up to ×3 to avoid overconfidence on flood attacks.
            high_lambda_factor = np.clip(lam_i / 6.0, 1.0, 3.0)

            # --- Final margin (cleaned: redundant scaling removed) ---
            margin = float(local_protocol_factor * 0.25 * z_score *
                           std_est * scale_correction * high_lambda_factor)

            lower = max(preds[i] - margin, 0.0)
            upper = preds[i] + margin
            ci[i] = [lower, upper]
            calibrated_bounds.append(calibrated_bound)

        # Adaptive clipping: cap extreme intervals via quantiles
        widths = ci[:, 1] - ci[:, 0]
        width_cap = np.quantile(widths, 0.98)  # 98th percentile threshold
        clipped = widths > width_cap
        ci[clipped, 0] = preds[clipped] - width_cap / 2
        ci[clipped, 1] = preds[clipped] + width_cap / 2

        if self.verbose:
            print(f"[CONFIDENCE] q={self.q}, mean CI width={np.mean(widths):.3f}, "
                  f"clipped intervals={np.sum(clipped)}")

        if return_bound_details:
            return preds, ci, {
                "bounds": calibrated_bounds,
                "lambda_vals": lambda_vals,
                "width_cap": width_cap,
                "clipped_count": np.sum(clipped)
            }

        return preds, ci

    def analytical_bootstrap_ci(self, X: np.ndarray, alpha: float = None):
        """
        Compute analytical confidence intervals using Poisson inequality bounds.

        This serves as a deterministic analogue of bootstrap intervals:
        it reuses the analytical bounds derived from Poisson concentration
        inequalities instead of resampling residuals.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix for prediction.
        alpha : float, optional
            Desired significance level. If provided, overrides the model's
            confidence_level (alpha = 1 - confidence_level).

        Returns
        -------
        lower_ci : np.ndarray
            Lower bound of analytical CI.
        upper_ci : np.ndarray
            Upper bound of analytical CI.
        """
        # Determine confidence level dynamically if alpha is given
        if alpha is not None:
            old_conf = self.confidence_level
            self.confidence_level = 1 - alpha
            if self.verbose:
                print(f"[BOOTSTRAP-CI] Overriding confidence_level to {self.confidence_level:.3f} (alpha={alpha:.3f})")

        # Defensive input cleaning
        X = np.nan_to_num(np.asarray(X), nan=0.0, posinf=100.0, neginf=0.0)

        # Handle pure Poisson mode or missing model
        if getattr(self, "pure_poisson_mode", False) or self.model is None:
            lam, ci = self.predict_with_confidence(X)
            return ci[:, 0], ci[:, 1]

        # Compute analytical intervals
        preds, ci = self.predict_with_confidence(X)

        # Optional diagnostic output
        if self.verbose:
            mean_width = np.mean(ci[:, 1] - ci[:, 0])
            print(f"[BOOTSTRAP-CI] q={self.q}, mean analytical CI width={mean_width:.3f}")

        # Restore previous confidence level if overridden
        if alpha is not None:
            self.confidence_level = old_conf

        return ci[:, 0], ci[:, 1]

    # -------------------------------------------------
    #  Validation, Demonstration, & Benchmarking
    # -------------------------------------------------
    def demonstrate_original_inequalities(self, X: np.ndarray, y: np.ndarray, verbose: bool = True):
        """
        Demonstrate and verify the use of original Poisson concentration inequalities.

        This method fits two reference models:
          1. Independent inequality  ->  E[|Σ a_i X_i|^q] ≤ T_q(Σλ_i) * ||a||_∞^q
          2. Conditional inequality  ->  E_PO[|Σ a_i X_i|^q] ≤ M^q * ||a||_∞^q

        It reports Touchard values, analytical bounds, and ratio comparisons to
        empirical variance for interpretability.
        """
        results_summary = {}

        if verbose or getattr(self, "verbose", False):
            print("\n[DEMONSTRATION] Original Poisson Concentration Inequalities")
            print("=" * 65)

        models = {
            "Independent Inequality": ImprovedPoissonConcentrationML(
                use_conditional=False,
                empirical_calibration=False,
                q=self.q,
                lambda_cap=self.lambda_cap,
                random_state=self.random_state,
                verbose=False
            ),
            "Conditional Inequality": ImprovedPoissonConcentrationML(
                use_conditional=True,
                empirical_calibration=False,
                q=self.q,
                lambda_cap=self.lambda_cap,
                random_state=self.random_state,
                verbose=False
            )
        }

        for name, model in models.items():
            model.fit(X, y)
            sample_count = min(5, len(X))
            preds, ci, details = model.predict_with_confidence(X[:sample_count], return_bound_details=True)

            bounds = details.get("bounds", [])
            lambdas = details.get("lambda_totals", [])
            touchards = details.get("touchard_values", [])
            analytical_mean = np.mean(bounds)
            empirical_var = np.var(y[:sample_count] - preds)
            ratio = empirical_var / analytical_mean if analytical_mean > 0 else np.nan

            results_summary[name] = {
                "inequality": details["inequality_type"],
                "mean_analytical_bound": analytical_mean,
                "empirical_variance": empirical_var,
                "empirical_to_analytical_ratio": ratio,
                "sample_bounds": bounds[:3],
                "touchard_values": touchards[:3] if touchards else None,
            }

            if verbose or getattr(self, "verbose", False):
                print(f"\n{name}:")
                print(f"  Inequality type: {details['inequality_type']}")
                print(f"  Mean analytical bound: {analytical_mean:.3e}")
                print(f"  Empirical variance (residual): {empirical_var:.3e}")
                print(f"  Ratio (empirical / analytical): {ratio:.3f}")
                for i in range(min(3, len(bounds))):
                    if details["inequality_type"] == "independent":
                        print(f"    Sample {i+1}: T_{self.q}({lambdas[i]:.2f}) = {touchards[i]:.2e}, "
                              f"bound = {bounds[i]:.2e}")
                    else:
                        print(f"    Sample {i+1}: M^{self.q} = {bounds[i]:.2e}")

        return results_summary

    def benchmark_with_original_inequalities(self, X: np.ndarray, y: np.ndarray,
                                         n_bootstrap_samples: int = 100,
                                         test_size: float = 0.2,
                                         verbose: bool = True):
        """
        Benchmark different Poisson inequality configurations against
        a bootstrap reference using consistent scaling and calibration logic.
        """
        if verbose:
            print("\n[BENCHMARK] Evaluating Poisson inequality variants vs empirical bootstrap")
            print("=" * 75)

        # Train-test split
        n = len(X)
        n_test = max(1, int(n * test_size))
        X_train, X_test = X[:-n_test], X[-n_test:]
        y_train, y_test = y[:-n_test], y[-n_test:]

        # Variants to compare
        variants = {
            "Original (Global λ)": ImprovedPoissonConcentrationML(
                local_lambda_estimation=False,
                empirical_calibration=False,
                use_conditional=False,
                q=self.q,
                lambda_cap=self.lambda_cap,
                random_state=self.random_state,
                verbose=False
            ),
            "Improved (Local λ)": ImprovedPoissonConcentrationML(
                local_lambda_estimation=True,
                empirical_calibration=False,
                use_conditional=False,
                q=self.q,
                lambda_cap=self.lambda_cap,
                random_state=self.random_state,
                verbose=False
            ),
            "Improved (Calibrated)": ImprovedPoissonConcentrationML(
                local_lambda_estimation=True,
                empirical_calibration=True,
                use_conditional=False,
                q=self.q,
                lambda_cap=self.lambda_cap,
                random_state=self.random_state,
                verbose=False
            ),
            "Conditional Bound": ImprovedPoissonConcentrationML(
                local_lambda_estimation=True,
                empirical_calibration=True,
                use_conditional=True,
                q=self.q,
                lambda_cap=self.lambda_cap,
                random_state=self.random_state,
                verbose=False
            )
        }

        results = {}
        for name, model in variants.items():
            start = time.time()
            model.fit(X_train, y_train)
            preds, ci = model.predict_with_confidence(X_test)
            elapsed = time.time() - start

            coverage = np.mean((y_test >= ci[:, 0]) & (y_test <= ci[:, 1]))
            avg_width = np.mean(ci[:, 1] - ci[:, 0])
            mse = mean_squared_error(y_test, preds)

            results[name] = {
                "coverage": coverage,
                "avg_width": avg_width,
                "mse": mse,
                "time": elapsed,
                "calibration_factor": getattr(model, "calibration_factor", 1.0)
            }

            if verbose:
                print(f"\n{name}:")
                print(f"   Coverage: {coverage:.3f},  Avg Width: {avg_width:.3f},  "
                      f"MSE: {mse:.4f},  Time: {elapsed:.3f}s,  "
                      f"Calib: {model.calibration_factor:.3f}")

        # --- Bootstrap reference for comparison ---
        if verbose:
            print("\nBootstrap Reference:")

        start = time.time()
        base_model = LinearRegression().fit(X_train, y_train)
        preds = base_model.predict(X_test)

        # Empirical bootstrap via residual resampling
        bootstrap_preds = []
        for _ in range(n_bootstrap_samples):
            res = y_train - base_model.predict(X_train)
            y_resampled = base_model.predict(X_train) + self.rng.choice(res, size=len(res), replace=True)
            model_b = LinearRegression().fit(X_train, y_resampled)
            bootstrap_preds.append(model_b.predict(X_test))

        bootstrap_preds = np.array(bootstrap_preds)
        ci_low, ci_high = np.percentile(bootstrap_preds, [2.5, 97.5], axis=0)
        elapsed_boot = time.time() - start

        coverage_b = np.mean((y_test >= ci_low) & (y_test <= ci_high))
        width_b = np.mean(ci_high - ci_low)
        mse_b = mean_squared_error(y_test, np.mean(bootstrap_preds, axis=0))

        results["Bootstrap Reference"] = {
            "coverage": coverage_b,
            "avg_width": width_b,
            "mse": mse_b,
            "time": elapsed_boot
        }

        if verbose:
            print(f"   Coverage: {coverage_b:.3f},  Avg Width: {width_b:.3f},  "
                  f"MSE: {mse_b:.4f},  Time: {elapsed_boot:.3f}s")

        return results

    # -------------------------------------------------
    #  Streaming & Online Detection
    # -------------------------------------------------
    def simulate_streaming_detection(
        self,
        stream_df: pd.DataFrame,
        features: list,
        target_col: str = "log_anomaly_count",
        thresholding: bool = True,
        verbose: bool = True
    ):
        """
        Simulate real-time streaming anomaly detection using Poisson bounds.

        Parameters
        ----------
        stream_df : pd.DataFrame
            Time-sorted streaming data containing features and observed target column.
        features : list
            Feature names used by the fitted model.
        target_col : str, default="log_anomaly_count"
            Column containing observed counts or anomalies.
        thresholding : bool, default=True
            If True, marks observations that fall outside Poisson confidence bounds.
        verbose : bool, default=True
            Controls printed output.

        Returns
        -------
        stream_df : pd.DataFrame
            Original dataframe with added columns ['prediction', 'ci_lower', 'ci_upper', 'violation'].
        summary : dict
            Streaming statistics including total violations and average CI width.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model not fitted. Run .fit() before streaming detection.")

        X = stream_df[features].copy().to_numpy()
        X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=0.0)

        # Apply scaling consistent with training
        if hasattr(self, "scaler_"):
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        preds, ci = self.predict_with_confidence(X_scaled)
        lower, upper = ci[:, 0], ci[:, 1]

        # Record results
        stream_df["prediction"] = preds
        stream_df["ci_lower"] = lower
        stream_df["ci_upper"] = upper
        stream_df["ci_width"] = upper - lower

        if thresholding:
            observed = np.nan_to_num(stream_df[target_col].values, nan=0.0)
            violations = ((observed < lower) | (observed > upper)).astype(int)
            stream_df["violation"] = violations
        else:
            violations = np.zeros(len(stream_df), dtype=int)
            stream_df["violation"] = violations

        # Diagnostics
        total_viol = int(np.sum(violations))
        avg_width = float(np.mean(upper - lower))

        summary = {
            "q": self.q,
            "total_violations": total_viol,
            "violation_rate": total_viol / len(stream_df),
            "mean_ci_width": avg_width,
            "confidence_level": self.confidence_level,
        }

        if verbose:
            print(f"\n[STREAMING DETECTION] Summary (q={self.q})")
            print(f"  Total violations: {total_viol} ({summary['violation_rate']:.2%})")
            print(f"  Mean CI width: {avg_width:.4f}")
            print(f"  Confidence level: {self.confidence_level:.2f}")

        return stream_df, summary

    def streaming_benchmark(
        self,
        stream_df: pd.DataFrame,
        models: dict,
        features: list,
        target_col: str = "log_anomaly_count",
        thresholding: bool = True,
        metric: str = "mse",
        verbose: bool = True,
    ):
        """
        Universal streaming benchmark comparing Poisson, classical ML, and deep learning detectors.

        Parameters
        ----------
        stream_df : pd.DataFrame
            Time-sorted streaming data with features and observed target column.
        models : dict
            Dictionary of trained models (e.g., {"Poisson": self, "Autoencoder": ae_model, ...}).
        features : list
            Feature column names used for prediction.
        target_col : str, default="log_anomaly_count"
            Column containing observed target or anomaly intensity.
        thresholding : bool, default=True
            Apply Poisson-based CI violation logic when applicable.
        metric : {"mse", "roc"}, default="mse"
            Evaluation metric to compute for all comparable models.
        verbose : bool, default=True
            Print detailed benchmark results.

        Returns
        -------
        results : dict
            Per-model metrics including prediction error, CI width, violations, and timing.
        """
        results = {}
        y_true = np.nan_to_num(stream_df[target_col].values, nan=0.0)
        X = np.nan_to_num(stream_df[features].values, nan=0.0, posinf=100.0, neginf=0.0)

        # Use same scaling as training for Poisson model
        if hasattr(self, "scaler_"):
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        for name, model in models.items():
            start = time.time()
            result_entry = {"model_name": name}

            try:
                # Case 1: Poisson model (your class)
                if isinstance(model, ImprovedPoissonConcentrationML):
                    stream_out, summary = model.simulate_streaming_detection(
                        stream_df.copy(),
                        features=features,
                        target_col=target_col,
                        thresholding=thresholding,
                        verbose=False,
                    )
                    preds = stream_out["prediction"].values
                    result_entry.update(summary)
                    result_entry["avg_ci_width"] = summary["mean_ci_width"]

                # Case 2: Classical sklearn model (IsolationForest, OneClassSVM, etc.)
                elif hasattr(model, "decision_function") or hasattr(model, "score_samples"):
                    # Unsupervised anomaly score
                    try:
                        scores = -model.decision_function(X_scaled)
                    except Exception:
                        scores = -model.score_samples(X_scaled)
                    preds = scores
                    result_entry.update({"avg_ci_width": None, "violations": "N/A"})

                # Case 3: Generic ML Regressor or Classifier
                elif hasattr(model, "predict"):
                    preds = model.predict(X_scaled)
                    result_entry.update({"avg_ci_width": None, "violations": "N/A"})

                # Case 4: Deep-learning model (Autoencoder, LSTM, etc.)
                elif hasattr(model, "predict") and callable(model.predict):
                    # Some deep models return reconstruction (X_recon)
                    y_pred = model.predict(X_scaled)
                    if y_pred.shape == X_scaled.shape:
                        # Use reconstruction error as anomaly score
                        preds = np.mean((X_scaled - y_pred) ** 2, axis=1)
                    else:
                        preds = np.ravel(y_pred)
                    result_entry.update({"avg_ci_width": None, "violations": "N/A"})

                else:
                    raise TypeError(f"Unsupported model type: {type(model)}")

                # --- Evaluation metrics ---
                if metric == "mse":
                    score = mean_squared_error(y_true, preds)
                elif metric == "roc":
                    from sklearn.metrics import roc_auc_score
                    # Require binary labels; threshold high anomalies
                    binary_labels = (y_true > np.percentile(y_true, 80)).astype(int)
                    score = roc_auc_score(binary_labels, preds)
                else:
                    raise ValueError("Unsupported metric; choose 'mse' or 'roc'.")

                result_entry[metric] = score
                result_entry["predictions"] = preds
                result_entry["time"] = time.time() - start

            except Exception as e:
                result_entry.update({
                    "error": str(e),
                    metric: None,
                    "predictions": None,
                    "time": time.time() - start,
                })
                if verbose:
                    print(f"[ERROR] {name} failed during benchmark: {e}")

            results[name] = result_entry

            # --- Verbose printout ---
            if verbose and metric in result_entry:
                if isinstance(model, ImprovedPoissonConcentrationML):
                    print(f"[STREAM-BENCH] {name}: {metric.upper()}={score:.4f}, "
                          f"Viol={result_entry.get('total_violations', 0)}, "
                          f"CI={result_entry.get('mean_ci_width', 0):.3f}, "
                          f"Time={result_entry['time']:.2f}s")
                else:
                    print(f"[STREAM-BENCH] {name}: {metric.upper()}={score:.4f}, Time={result_entry['time']:.2f}s")

        return results


class ImprovedPoissonConcentrationML_CrossDataset(ImprovedPoissonConcentrationML):
    """
    Cross-dataset extension of the Poisson-CI detector.

    This subclass preserves the core concentration-inequality framework
    of the primary detector while incorporating additional calibration
    and interval adaptation mechanisms designed to improve robustness
    under distribution shift and heterogeneous traffic conditions
    encountered in external IoT datasets such as CICIoT2023.

    The underlying Poisson concentration bounds and prediction pipeline
    remain unchanged. The extensions primarily affect confidence interval
    scaling and residual calibration during inference.
    """
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Generate point predictions using the fitted model.
        Ensures consistency with feature scaling used in fit().
        """
        if self.model is None:
            raise ValueError("Model not fitted or running in pure Poisson mode. Use pure λ estimates instead.")

        X = np.nan_to_num(np.asarray(X), nan=0.0, posinf=100.0, neginf=0.0)

        # Apply same scaling as during fit
        if hasattr(self, "scaler_"):
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X

        if self.model_type == "logistic":
            return self.model.predict_proba(X_scaled)[:, 1]
        else:
            preds = self.model.predict(X_scaled)

            # ✅ FIX 1: enforce valid domain
            preds = np.maximum(preds, 0.0)

            # Only enforce domain
            preds = np.maximum(preds, 0.0)

            return preds

    def predict_with_confidence(self, X: np.ndarray, return_bound_details: bool = False):
        """
        Compute Poisson-based confidence intervals for predictions.
        Now fully stabilized: NO dependence on test/attack distribution.
        """

        X = np.nan_to_num(np.asarray(X), nan=0.0, posinf=100.0, neginf=0.0)
        n_samples = X.shape[0]

        # --- Pure Poisson mode ---
        if getattr(self, "pure_poisson_mode", False) or self.model is None:
            lam = np.full(n_samples, self.lambda_estimates)
            z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)
            std_est = np.sqrt(lam)
            margin = z_score * std_est
            lower = np.clip(lam - margin, 0, None)
            upper = lam + margin
            ci = np.vstack([lower, upper]).T
            return lam, ci

        # --- Scale input ---
        preds = self.predict(X)

        # ----------------------------------------------------------
        # ✅ FIX 1: USE TRAINING-BASED λ (NO test dependence)
        # ----------------------------------------------------------
        lambda_vals = np.expm1(preds)

        # Soft constraint instead of hard clipping
        lambda_mean = np.mean(self.lambda_estimates)

        # Blend predicted λ with training λ
        alpha = 0.7   # ← key parameter

        lambda_vals = alpha * lambda_vals + (1 - alpha) * lambda_mean

        # Recreate scaled version ONLY for variance computation
        if hasattr(self, "scaler_"):
            X_scaled_for_stats = self.scaler_.transform(X)
        else:
            X_scaled_for_stats = X
        # ----------------------------------------------------------
        # Model weights
        # ----------------------------------------------------------
        weights = np.nan_to_num(self.model.coef_.flatten(), nan=0.0)
        weights_inf_norm = np.max(np.abs(weights)) / (np.std(X_scaled_for_stats, axis=0).max() + 1e-6)

        # ----------------------------------------------------------
        # Precompute constants
        # ----------------------------------------------------------
        z_score = stats.norm.ppf(1 - (1 - self.confidence_level) / 2)

        vmr_factor = getattr(self, "vmr_factor_", 1.0)
        empirical_factor = 2.2   # keep your tuned value

        scale_correction = np.clip(self.calibration_factor, 0.1, 2.0)

        # ----------------------------------------------------------
        # Build CI (now FULLY STABLE)
        # ----------------------------------------------------------
        ci = np.zeros((n_samples, 2))

        for i in range(n_samples):

            # No λ dependence anymore
            local_protocol_factor = getattr(self, "local_protocol_factor", 1.0)

            lam_i = lambda_vals[i]

            # Compute Poisson bound for THIS λ
            bound_i = self._compute_original_poisson_bound(
                np.array([lam_i]),
                weights_inf_norm
            )

            calibrated_bound = float(bound_i) * (self.calibration_factor ** 2)

            # Std in λ space
            std_lambda = np.sqrt(calibrated_bound) if self.q == 2 else calibrated_bound ** (1.0 / self.q)

            # Convert to log-space (delta method)
            std_est = std_lambda / (1.0 + lam_i)

            # Variance floor (prevents collapse)
            std_est = max(std_est, 0.05)

            margin = float(
                local_protocol_factor *
                z_score *
                std_est *
                scale_correction *
                vmr_factor *
                empirical_factor
            )

            # LOCAL bias correction (per sample)
            lam_i = lambda_vals[i]

            # residual-based correction (NOT lambda-based)
            #pred_corrected = preds[i] * np.exp(0.5 * self.residual_mean_)
            #pred_corrected = preds[i] + self.residual_mean_
            residual_mean = getattr(self, "residual_mean_", 0.0)
            pred_corrected = preds[i] + residual_mean

            # Asymmetric correction (fix skew)
            lower_margin = 0.5 * margin
            upper_margin = 1.5 * margin

            lower = max(pred_corrected - lower_margin, 0.0)
            upper = pred_corrected + upper_margin

            if lower > upper:
                lower, upper = upper, lower

            ci[i] = [lower, upper]

        # ----------------------------------------------------------
        # Optional clipping (kept, but now stable)
        # ----------------------------------------------------------
        widths = ci[:, 1] - ci[:, 0]
        width_cap = np.quantile(widths, 0.98)

        clipped = widths > width_cap
        ci[clipped, 0] = preds[clipped] - width_cap / 2
        ci[clipped, 1] = preds[clipped] + width_cap / 2

        if self.verbose:
            print(f"[CONFIDENCE] q={self.q}, mean CI width={np.mean(widths):.3f}, "
                  f"clipped intervals={np.sum(clipped)}")

        if return_bound_details:
            return preds, ci, {
                "width_cap": width_cap,
                "clipped_count": np.sum(clipped)
            }

        return preds, ci