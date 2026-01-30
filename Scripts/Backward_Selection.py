import pandas as pd
import numpy as np
from scipy.special import gammaln, expit
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
import os
from numpy.linalg import inv
from tqdm.auto import tqdm
from scipy.stats import nbinom, binom, norm
import statsmodels.api as sm
from scipy.stats import norm
# ================================================================
# DATA CLASS
# ================================================================
class Data:
    def __init__(self, df = None, path="../Data/Model_Input/Uheld_LINKS.csv"):
        if df is None:
            # --- Load and preprocess data ---
            data = pd.read_csv(path)
    
            # Exposure (vehicle-km)
            data['exposure_vehicle_km'] = (data['Traf_Car_Van'] + data['Traf_Truck']) * data['Length']
    
            # Remove links with no traffic
            data = data[data['exposure_vehicle_km'] > 0]
            # Remove AAR < 2023
            data = data[data['AAR'] ==  2023]

            # Replace missing speeds where no traffic
            data.loc[data['Traf_Truck'] == 0, 'AvgSpeed_Truck'] = data['FreeSpeed']
            data.loc[data['Traf_Car_Van'] == 0, 'AvgSpeed_Car_Van'] = data['FreeSpeed']
    
            # --- Feature engineering ---
            total_traffic = data['Traf_Car_Van'] + data['Traf_Truck']
            data['Frac_Truck']        = data['Traf_Truck'] / total_traffic
            data['SpeedDiff_Car_Van'] = data['FreeSpeed'] - data['AvgSpeed_Car_Van']
            data['SpeedDiff_Truck']   = np.minimum(data['FreeSpeed'], 80) - data['AvgSpeed_Truck']
    
            # --- Accident rates ---
            data['Personskade_rate']   = data['Personskade']   / data['exposure_vehicle_km']
            data['Materielskade_rate'] = data['Materielskade'] / data['exposure_vehicle_km']
        else:
            data = df

        # Store as instance variable
        self.data = data

    def var_extract(self,
                    features_X = ['Frac_Truck','SpeedDiff_Car_Van','SpeedDiff_Truck', 'FreeSpeed'
                        'URBAN','Link_group_0','Link_group_1','Link_group_2','Link_group_3',
                        'Region Hovedstaden','Region Sjælland','Region Syddanmark',
                        'Region Midtjylland','Region Nordjylland'],
                    features_Z = ['Frac_Truck','SpeedDiff_Car_Van','SpeedDiff_Truck', 'FreeSpeed'
                        'URBAN','Link_group_0','Link_group_1','Link_group_2','Link_group_3',
                        'Region Hovedstaden','Region Sjælland','Region Syddanmark',
                        'Region Midtjylland','Region Nordjylland'],
                    features_W = ['Frac_Truck','SpeedDiff_Car_Van','SpeedDiff_Truck', 'FreeSpeed'
                        'URBAN','Link_group_0','Link_group_1','Link_group_2','Link_group_3',
                        'Region Hovedstaden','Region Sjælland','Region Syddanmark',
                        'Region Midtjylland','Region Nordjylland']):
        data = self.data.copy()

        exposure = data['exposure_vehicle_km'].values
        y = data['Personskade'].values + data['Materielskade'].values
        y_p = data['Personskade'].values
        z = data[['slight', 'serious', 'fatal']].values # injury severity 


        X = data[features_X].copy()
        X.insert(0,'Intercept',1)


        Z = data[features_Z].copy()
        Z.insert(0,'Intercept',1)

        W = data[features_W].copy()
        W.insert(0,'Intercept',1)
        return y, y_p, z, X.values, Z.values, W.values, exposure

# ================================================================
# MODEL CLASS
# ================================================================
class Model:
    @staticmethod
    def Binomial(gamma, y_total, y_person, Z, return_hessian=False):
        """
        Binomial log-likelihood with logit link.
        y_person ~ Binomial(n=y_total, p=sigmoid(Z @ gamma))

        Returns:
            ll
            (grad, hess) if return_hessian=True
        """
        eta = Z @ gamma
        # Stable sigmoid and clipping to avoid log(0)
        p = expit(eta)
        p = np.clip(p, 1e-12, 1 - 1e-12)

        # Validate counts
        if np.any(y_person > y_total):
            raise ValueError("Found y_person > y_total.")

        # Log-likelihood (with combinatorial term for completeness)
        ll_i = (gammaln(y_total + 1)
                - gammaln(y_person + 1)
                - gammaln(y_total - y_person + 1)
                + y_person * np.log(p)
                + (y_total - y_person) * np.log(1 - p))
        ll = np.sum(ll_i)

        if not return_hessian:
            return ll

        # Gradient wrt gamma: Z^T (y - n p)
        score_i = y_person - y_total * p
        grad = Z.T @ score_i

        # Hessian wrt gamma: - Z^T diag(n p (1-p)) Z
        w = y_total * p * (1.0 - p)              # shape (n,)
        hess = -(Z.T @ (Z * w[:, None]))         # X^T diag(w) X without forming diag

        return ll, grad, hess


    @staticmethod
    def NB(beta, log_alpha, target, X, offset=None, return_hessian=False):
        """
        Negative Binomial (NB2) log-likelihood, gradient, and Hessian.
        Var(Y) = mu * (1 + alpha * mu).

        Parameters
        ----------
        beta : (k,) array
            Coefficients for the linear predictor.
        log_alpha : float
            Log-dispersion parameter (alpha = exp(log_alpha)).
        target : (n,) array
            Observed counts.
        X : (n, k) array
            Design matrix.
        offset : (n,) array, optional
            Offset term (log exposure, etc.).
        return_hessian : bool
            If True, return both gradient and Hessian.

        Returns
        -------
        ll : float
            Log-likelihood sum.
        grad : (k,) array, optional
            Gradient vector wrt beta.
        hess : (k, k) array, optional
            Hessian matrix wrt beta.
        """
        # --- Parameters ---
        alpha = np.exp(np.clip(log_alpha, -10, 10))

        if offset is None:
            eta = X @ beta
        else:
            eta = offset + X @ beta

        # numerical safety
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)

        r = 1.0 / (alpha + 1e-16)
        p = r / (r + mu)
        p = np.clip(p, 1e-12, 1 - 1e-12)

        # --- Log-likelihood ---
        ll_i = (
            gammaln(target + r) - gammaln(r) - gammaln(target + 1)
            + r * np.log(p)
            + target * np.log1p(-p)
        )
        ll = np.nansum(ll_i)

        if not return_hessian:
            return ll

        # =======================================================
        # Gradient wrt beta
        # =======================================================
        # dℓ/dη_i = (y_i - μ_i) / (1 + α μ_i)
        score_i = (target - mu) / (1 + alpha * mu)
        grad = X.T @ score_i

        # =======================================================
        # Hessian wrt beta
        # =======================================================
        # d²ℓ/dη²_i = - (1 + α y_i) μ_i / (1 + α μ_i)²
        d2_i = -(1 + alpha * target) * mu / (1 + alpha * mu) ** 2
        hess = X.T @ (X * d2_i[:, None]) # X^T diag(d) X

        return ll, grad, hess
    @staticmethod
    def ZINB(beta, gamma, log_alpha, target, X, Z, offset=None, return_hessian=False):
        """
        Zero-Inflated Negative Binomial (ZINB2) log-likelihood and Hessian.
    
        Parameters
        ----------
        beta : (k,) array
            NB mean coefficients.
        gamma : (m,) array
            Zero-inflation (logit) coefficients.
        log_alpha : float
            Log-dispersion parameter (alpha = exp(log_alpha)).
        target : (n,) array
            Observed counts.
        X : (n, k) array
            Design matrix for NB mean.
        Z : (n, m) array
            Design matrix for zero inflation.
        offset : (n,) array, optional
            Offset for NB mean.
        return_hessian : bool
            If True, return Hessian blocks for beta and gamma.
    
        Returns
        -------
        ll : float
            Total log-likelihood.
        (Optional)
        H_bb : (k,k) array
            Hessian wrt beta.
        H_gg : (m,m) array
            Hessian wrt gamma.
        H_bg : (k,m) array
            Cross Hessian wrt (beta, gamma).
        """
    
        # ---------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------
        alpha = np.exp(np.clip(log_alpha, -10, 10))
    
        # NB mean
        if offset is None:
            eta = X @ beta
        else:
            eta = offset + X @ beta
    
        eta = np.clip(eta, -20, 20)
        mu = np.exp(eta)
    
        # NB params
        r = 1.0 / (alpha + 1e-16)
        p = r / (r + mu)
        p = np.clip(p, 1e-12, 1 - 1e-12)
    
        # Zero-inflation
        psi = Z @ gamma
        psi = np.clip(psi, -20, 20)
        pi = 1.0 / (1.0 + np.exp(-psi))
        one_minus_pi = 1 - pi
    
        # ---------------------------------------------------------
        # NB log pmf
        # ---------------------------------------------------------
        ll_nb = (
            gammaln(target + r) - gammaln(r) - gammaln(target + 1)
            + r * np.log(p)
            + target * np.log1p(-p)
        )
    
        # NB zero-probability
        f0 = p**r
        f0 = np.clip(f0, 1e-32, 1.0)
    
        # ---------------------------------------------------------
        # ZINB log-likelihood
        # ---------------------------------------------------------
        is_zero = (target == 0)
        is_pos  = ~is_zero
    
        # zeros: log( pi + (1-pi)*f0 )
        ll_zero = np.log(pi + one_minus_pi * f0)
    
        # positive: log(1-pi) + nb
        ll_pos  = np.log(one_minus_pi) + ll_nb
    
        ll = np.sum(ll_zero[is_zero]) + np.sum(ll_pos[is_pos])
    
        if not return_hessian:
            return ll
    
        # =====================================================================
        # HESSIAN COMPUTATION
        # =====================================================================
    
        # ---------------------------------------------------------
        # Useful NB derivatives (scalar per observation)
        # ---------------------------------------------------------
        # NB score in eta-space
        nb_eta = (target - mu) / (1 + alpha * mu)
    
        # NB curvature in eta-space
        nb_eta2 = -(1 + alpha * target) * mu / (1 + alpha * mu)**2
    
        # For f0, f0', f0''
        nb_eta_zero = -mu / (1 + alpha * mu)
        nb_eta2_zero = -mu / (1 + alpha * mu)**2
    
        f0_prime = f0 * nb_eta_zero
        f0_double = f0 * (nb_eta2_zero + nb_eta_zero**2)
    
        # ---------------------------------------------------------
        # H(+): Non-zero part
        # ---------------------------------------------------------
        # H_bb(+) = sum h_etaeta * x x'
        h_etaeta_pos = nb_eta2[is_pos]      # scalar curvature
        X_pos = X[is_pos]
        H_bb_pos = X_pos.T @ (X_pos * h_etaeta_pos[:, None])
    
        # H_gg(+) = sum -pi(1-pi) * z z'
        h_psipsi_pos = -pi[is_pos] * (1 - pi[is_pos])
        Z_pos = Z[is_pos]
        H_gg_pos = Z_pos.T @ (Z_pos * h_psipsi_pos[:, None])
    
        # cross-block is zero
        H_bg_pos = np.zeros((X.shape[1], Z.shape[1]))
    
        # ---------------------------------------------------------
        # H(0): Zero part
        # ---------------------------------------------------------
        X_zero = X[is_zero]
        Z_zero = Z[is_zero]
    
        pi_z = pi[is_zero]
        one_minus_pi_z = 1 - pi_z
        f0_z = f0[is_zero]
        f0p_z = f0_prime[is_zero]
        f0pp_z = f0_double[is_zero]
    
        A = pi_z + one_minus_pi_z * f0_z
    
        # ---- h_etaeta(0)
        h_etaeta_zero = (
            one_minus_pi_z * (f0pp_z / A)
            - (one_minus_pi_z**2) * (f0p_z**2) / (A**2)
        )
    
        H_bb_zero = X_zero.T @ (X_zero * h_etaeta_zero[:, None])
    
        # ---- h_psipsi(0)
        B = (1 - f0_z) * pi_z * one_minus_pi_z
        Bp = (1 - f0_z) * pi_z * one_minus_pi_z * (1 - 2*pi_z)
    
        h_psipsi_zero = (A*Bp - B*B) / (A**2)
    
        H_gg_zero = Z_zero.T @ (Z_zero * h_psipsi_zero[:, None])
    
        # ---- h_etapsi(0)
        C = one_minus_pi_z
        h_etapsi_zero = - f0p_z * pi_z * C * (A + (1 - f0_z)*C) / (A**2)
    
        H_bg_zero = X_zero.T @ (Z_zero * h_etapsi_zero[:, None])
    
        # ---------------------------------------------------------
        # TOTAL HESSIAN BLOCKS
        # ---------------------------------------------------------
        H_bb = H_bb_pos + H_bb_zero
        H_gg = H_gg_pos + H_gg_zero
        H_bg = H_bg_pos + H_bg_zero
    
        return ll, H_bb, H_gg, H_bg

    @staticmethod
    def Gaussian(param, sigma):
        """
        Log-prior under independent zero-mean Gaussian priors.
        Returns the sum of log p(param | sigma).
        """
        # Ensure sigma is array-like (per-parameter or scalar)
        sigma = np.asarray(sigma)
        logp = -0.5 * np.sum((param / sigma)**2)
        return logp


    # ----------------------------------------------------
    # 3. Posterior likelihood (sum of all components)
    # ----------------------------------------------------
    def Posterior(self, params, sigmas, y, y_p, z, X, Z, W, exposure):
        """
        Joint log-posterior = data log-likelihood + log Gaussian priors.
        sigmas: dictionary or tuple of prior stds for each parameter block.
        """
        k, l, m = X.shape[1], Z.shape[1], W.shape[1]
        
        # --- unpack parameters ---
        beta = params[:k]
        log_alpha_y = params[k]
        
        gamma = params[k+1 : k+1+l]
        idx = k + 1 + l
        
        delta_slight = params[idx : idx + m]
        log_alpha_slight = params[idx + m]
        idx += m + 1
        
        delta_serious = params[idx : idx + m]
        log_alpha_serious = params[idx + m]
        idx += m + 1
        
        delta_fatal = params[idx : idx + m]
        log_alpha_fatal = params[idx + m]
    
        # --- data log-likelihood ---
        ll_total = (
            self.NB(beta, log_alpha_y, y, X, offset=np.log(exposure + 1e-12))
            + self.Binomial(gamma, y, y_p, Z)
            + self.NB(delta_slight,  log_alpha_slight,  z[:,0], W, offset=np.log(y_p + 1e-12))
            + self.NB(delta_serious, log_alpha_serious, z[:,1], W, offset=np.log(y_p + 1e-12))
            + self.NB(delta_fatal,   log_alpha_fatal,   z[:,2], W, offset=np.log(y_p + 1e-12))
        )
    
        # --- Gaussian log-priors (exclude intercepts & log_alphas) ---
        # intercepts are the first column of each design matrix → index 0 of each block
        logp_prior = 0.0
        logp_prior += self.Gaussian(beta[1:],   sigmas['beta'])
        logp_prior += self.Gaussian(gamma[1:],  sigmas['gamma'])
        logp_prior += self.Gaussian(delta_slight[1:],  sigmas['delta_slight'])
        logp_prior += self.Gaussian(delta_serious[1:], sigmas['delta_serious'])
        logp_prior += self.Gaussian(delta_fatal[1:],   sigmas['delta_fatal'])
    
        # --- total log posterior ---
        log_post = ll_total + logp_prior
        return -log_post  # negative for minimization


    # ----------------------------------------------------
    # 4. Optimizer
    # ----------------------------------------------------
    def optimize(self, y, y_p, z, X, Z, W, exposure, verbose=True,
             tol=1e-3, max_outer=50, damping=0.3,
             theta_init=None):

        """
        Empirical Bayes (Type-II ML) with Laplace updates of per-parameter sigmas.
        """
        k = X.shape[1]; l = Z.shape[1]; m = W.shape[1]
        n_params = k + 1 + l + 3*(m + 1)
        
        if theta_init is None:
            # --- Initialize parameters
            theta0 = np.zeros(n_params)
            theta0[:k] = -5.0
            theta0[k]  = np.log(0.3)  # log alpha_y
            theta0[k+1:k+1+l] = 0.0
            start = k + 1 + l
            theta0[start:start + 3*m] = 0.0
            for i in range(3):
                theta0[start + i*(m + 1) + m] = np.log(0.3)  # each injury dispersion
        else:
            # Warm start
            theta0 = theta_init.copy()
    
        # --- Prior stds (vectors per block, intercept excluded)
        sigmas = {
            'beta'         : np.ones(k-1),
            'gamma'        : np.ones(l-1),
            'delta_slight' : np.ones(m-1),
            'delta_serious': np.ones(m-1),
            'delta_fatal'  : np.ones(m-1),
        }
    
        def prior_precision_vec(s):  # 1/sigma^2 elementwise
            s = np.asarray(s)
            return 1.0 / (s**2 + 1e-12)
    
        def block_post_cov_diag(H_ll_block, sigmas_block):
            """
            Build posterior Hessian for penalized coef only (exclude intercept),
            invert (pinv) and return diag of the inverse.
            """
            # H_ll_block is (d x d) Hessian of log-likelihood for the whole block
            # Extract penalized subblock (drop intercept index 0)
            H_ll_pen = H_ll_block[1:, 1:]
            # Posterior Hessian (positive definite ideally)
            Prec_prior = np.diag(prior_precision_vec(sigmas_block))
            H_post = (-H_ll_pen) + Prec_prior
            # Numerical guard
            # Use pinv to avoid crashes if nearly singular
            H_post += np.eye(H_post.shape[0]) * 1e-6
            H_post_inv = np.linalg.pinv(H_post)

            return np.diag(H_post_inv)
    
        # Outer EB loop
        pbar = tqdm(range(max_outer), desc="Empirical Bayes", leave=False)
        for outer in pbar:  
            # ========== MAP step (optimize theta with current sigmas) ==========
            res = minimize(
                fun=self.Posterior,
                x0=theta0,
                args=(sigmas, y, y_p, z, X, Z, W, exposure),
                method="L-BFGS-B",
                tol=1e-8,
                options={'maxfun': 500000, 'maxiter': 500000}
            )
            theta_hat = res.x
    
            # Unpack
            i = 0
            beta_hat = theta_hat[i:i+k]; i += k
            loga_y   = theta_hat[i];     i += 1
            gamma_hat= theta_hat[i:i+l]; i += l
    
            delta_s_hat  = theta_hat[i:i+m]; i += m
            la_s_hat     = theta_hat[i];     i += 1
            delta_se_hat = theta_hat[i:i+m]; i += m
            la_se_hat    = theta_hat[i];     i += 1
            delta_f_hat  = theta_hat[i:i+m]; i += m
            la_f_hat     = theta_hat[i];     i += 1
    
            # ========== Likelihood Hessians per block at MAP ==========
            _, _, H_beta  = self.NB(beta_hat,  loga_y,   y,      X, offset=np.log(exposure + 1e-12),  return_hessian=True)
            _, _, H_gamma = self.Binomial(gamma_hat,     y, y_p, Z,                                   return_hessian=True)
            _, _, H_s     = self.NB(delta_s_hat,  la_s_hat,  z[:,0], W, offset=np.log(y_p + 1e-12),   return_hessian=True)
            _, _, H_se    = self.NB(delta_se_hat, la_se_hat, z[:,1], W, offset=np.log(y_p + 1e-12),   return_hessian=True)
            _, _, H_f     = self.NB(delta_f_hat,  la_f_hat,  z[:,2], W, offset=np.log(y_p + 1e-12),   return_hessian=True)
    
            # ========== Laplace σ update (moment-matching style) ==========
            # posterior variance (diag of Σ = H_post^{-1})
            var_beta   = block_post_cov_diag(H_beta,  sigmas['beta'])
            var_gamma  = block_post_cov_diag(H_gamma, sigmas['gamma'])
            var_s      = block_post_cov_diag(H_s,     sigmas['delta_slight'])
            var_se     = block_post_cov_diag(H_se,    sigmas['delta_serious'])
            var_f      = block_post_cov_diag(H_f,     sigmas['delta_fatal'])
    
            # new sigma^2 := theta^2 + Var (per penalized coefficient)
            sig2_beta_new  = beta_hat[1:]**2      + var_beta
            sig2_gamma_new = gamma_hat[1:]**2     + var_gamma
            sig2_s_new     = delta_s_hat[1:]**2   + var_s
            sig2_se_new    = delta_se_hat[1:]**2  + var_se
            sig2_f_new     = delta_f_hat[1:]**2   + var_f
    
            # take sqrt to get sigma
            sigmas_new = {
                'beta'         : np.sqrt(np.maximum(sig2_beta_new, 1e-18)),
                'gamma'        : np.sqrt(np.maximum(sig2_gamma_new,1e-18)),
                'delta_slight' : np.sqrt(np.maximum(sig2_s_new,    1e-18)),
                'delta_serious': np.sqrt(np.maximum(sig2_se_new,   1e-18)),
                'delta_fatal'  : np.sqrt(np.maximum(sig2_f_new,    1e-18)),
            }
    
            # Damped update for stability
            for kkey in sigmas:
                sigmas_new[kkey] = (1 - damping) * sigmas[kkey] + damping * sigmas_new[kkey]
    
            # Convergence check across all blocks
            diff = 0.0
            for kkey in sigmas:
                a = sigmas[kkey].ravel(); b = sigmas_new[kkey].ravel()
                diff += np.sum((a - b)**2)
    
            if diff < tol:
                if verbose:
                    print(f"EB converged in {outer+1} outer iters; ||Δσ||²={diff:.3e}")
                break
    
            # Prepare next EB iteration
            sigmas = sigmas_new
            theta0 = theta_hat.copy()
        pbar.close()
        # Store and return final MAP result
        self.result_ = res
        if verbose:
            print(f"Optimization success: {res.success}")
            print(f"Final objective (−log posterior): {res.fun:.3f}; inner iters: {res.nit}")
        # Also store final sigmas for inspection
        self.sigmas_ = sigmas
        self.theta_hat_ = theta_hat
        # --- Posterior standard deviations from last iteration ---
        self.sigma_post_ = {
            "beta": np.concatenate([[np.nan], np.sqrt(var_beta)]),  # include NaN for intercept
            "gamma": np.concatenate([[np.nan], np.sqrt(var_gamma)]),
            "delta_slight": np.concatenate([[np.nan], np.sqrt(var_s)]),
            "delta_serious": np.concatenate([[np.nan], np.sqrt(var_se)]),
            "delta_fatal": np.concatenate([[np.nan], np.sqrt(var_f)])
        }

        return res

    
    
    def predict_summary(self, y, y_p, z, X, Z, W, exposure, feature_names_X=None, feature_names_Z=None, feature_names_W=None):
        """
        Compute, print, and return detailed summary of observed vs predicted totals and parameters.
        """
        params = self.result_.x
        k = X.shape[1]
        l = Z.shape[1]
        m = W.shape[1]
    
        # --- Extract parameter blocks
        beta = params[0:k]
        log_alpha_y = params[k]
        gamma = params[k+1 : k+1+l]
    
        idx = k + 1 + l
        delta_slight = params[idx : idx + m]
        log_alpha_slight = params[idx + m]
        idx += m + 1
    
        delta_serious = params[idx : idx + m]
        log_alpha_serious = params[idx + m]
        idx += m + 1
    
        delta_fatal = params[idx : idx + m]
        log_alpha_fatal = params[idx + m]
    
        # --- Convert dispersion parameters
        alpha_y = np.exp(log_alpha_y)
        alpha_slight = np.exp(log_alpha_slight)
        alpha_serious = np.exp(log_alpha_serious)
        alpha_fatal = np.exp(log_alpha_fatal)
    
        # --- Predict components (with stability clipping)
        eta_y     = np.clip(X @ beta, -20, 20)
        eta_gamma = np.clip(Z @ gamma, -20, 20)
        eta_s     = np.clip(W @ delta_slight,  -10, 10)
        eta_se    = np.clip(W @ delta_serious, -10, 10)
        eta_f     = np.clip(W @ delta_fatal,   -10, 10)
    
        mu_y     = np.clip(exposure, 1e-6, None) * np.exp(eta_y)
        pi       = expit(eta_gamma)
        mu_y_p   = mu_y * pi
        mu_z_slight  = np.exp(eta_s)  * mu_y_p
        mu_z_serious = np.exp(eta_se) * mu_y_p
        mu_z_fatal   = np.exp(eta_f)  * mu_y_p
    
        # --- Aggregate results
        obs_total = y.sum()
        pred_total = mu_y.sum()
    
        obs_injury = y_p.sum()
        pred_injury = mu_y_p.sum()
    
        obs_slight, obs_serious, obs_fatal = z.sum(axis=0)
        pred_slight, pred_serious, pred_fatal = (
            mu_z_slight.sum(),
            mu_z_serious.sum(),
            mu_z_fatal.sum()
        )
    
        # --- Print summary
        print("\n=== Prediction Summary ===")
        print(f"{'Component':<20}{'Observed':>15}{'Predicted':>15}{'Ratio (pred/obs)':>20}")
        print("-" * 70)
        print(f"{'Total accidents':<20}{obs_total:15.2f}{pred_total:15.2f}{pred_total/obs_total:20.3f}")
        print(f"{'Injury accidents':<20}{obs_injury:15.2f}{pred_injury:15.2f}{pred_injury/obs_injury:20.3f}")
        print(f"{'Slight injuries':<20}{obs_slight:15.2f}{pred_slight:15.2f}{pred_slight/obs_slight:20.3f}")
        print(f"{'Serious injuries':<20}{obs_serious:15.2f}{pred_serious:15.2f}{pred_serious/obs_serious:20.3f}")
        print(f"{'Fatal injuries':<20}{obs_fatal:15.2f}{pred_fatal:15.2f}{pred_fatal/obs_fatal:20.3f}")
    
        # --- Print dispersions
        print("\nDispersion parameters (α = exp(log_alpha)):")
        print(f"  α_y (frequency):     {alpha_y:.4f}")
        print(f"  α_slight (injuries): {alpha_slight:.4f}")
        print(f"  α_serious:           {alpha_serious:.4f}")
        print(f"  α_fatal:             {alpha_fatal:.4f}")
    
        # --- Optional: attach feature names for readability
        if feature_names_X is None:
            feature_names_X = [f"x{i}" for i in range(k)]
        if feature_names_Z is None:
            feature_names_Z = [f"z{i}" for i in range(l)]
        if feature_names_W is None:
            feature_names_W = [f"w{i}" for i in range(m)]
    
        df_params = pd.DataFrame({
            "param_type": (["beta"] * k) + (["gamma"] * l)
                           + (["delta_slight"] * m)
                           + (["delta_serious"] * m)
                           + (["delta_fatal"] * m),
            "name": feature_names_X + feature_names_Z + feature_names_W * 3,
            "value": np.concatenate([beta, gamma, delta_slight, delta_serious, delta_fatal])
        })
    
        # --- Construct prediction DataFrame
        df_pred = pd.DataFrame({
            "y_obs": y,
            "y_pred": mu_y,
            "y_p_obs": y_p,
            "y_p_pred": mu_y_p,
            "z_slight_obs": z[:, 0],
            "z_slight_pred": mu_z_slight,
            "z_serious_obs": z[:, 1],
            "z_serious_pred": mu_z_serious,
            "z_fatal_obs": z[:, 2],
            "z_fatal_pred": mu_z_fatal,
            "exposure": exposure
        })
    
        # --- Return everything neatly structured
        return {
            "params": {
                "beta": beta,
                "gamma": gamma,
                "delta_slight": delta_slight,
                "delta_serious": delta_serious,
                "delta_fatal": delta_fatal,
                "alpha_y": alpha_y,
                "alpha_slight": alpha_slight,
                "alpha_serious": alpha_serious,
                "alpha_fatal": alpha_fatal,
                "param_table": df_params
            },
            "predictions": {
                "y_pred": mu_y,
                "y_p_pred": mu_y_p,
                "z_pred": np.column_stack([mu_z_slight, mu_z_serious, mu_z_fatal]),
                "pred_table": df_pred
            },
            "summary": {
                "obs_total": obs_total,
                "pred_total": pred_total,
                "obs_injury": obs_injury,
                "pred_injury": pred_injury,
                "obs_slight": obs_slight,
                "pred_slight": pred_slight,
                "obs_serious": obs_serious,
                "pred_serious": pred_serious,
                "obs_fatal": obs_fatal,
                "pred_fatal": pred_fatal
            }
        }

class BackwardSelection:
    def __init__(self, data_obj):
        self.data_obj = data_obj
    
    def run(self, features_X, features_Z, features_W, verbose=True):
        removed_params = []
        iteration = 0
        theta_prev = None   # <---------------- NEW
    
        while True:
            iteration += 1
            if verbose:
                print(f"\n========== Backward Iteration {iteration} ==========")
    
            # ---- Extract matrices ----
            data_obj = self.data_obj
            y, y_p, z, X, Z, W, exposure = data_obj.var_extract(
                features_X=features_X,
                features_Z=features_Z,
                features_W=features_W
            )
    
            # ---- Fit model (WARM START) ----
            model = Model()
            model.optimize(
                y, y_p, z, X, Z, W, exposure,
                verbose=False,
                theta_init=theta_prev   # <---- NEW
            )
            theta_prev = model.theta_hat_.copy()  

            # ---- store parameter names ----
            k = X.shape[1]; l = Z.shape[1]; m = W.shape[1]
            model.param_names_ = (
                ["beta:"  + nm for nm in ["Intercept"] + features_X] +
                ["log_alpha_y"] +
                ["gamma:" + nm for nm in ["Intercept"] + features_Z] +
                ["delta_slight:"  + nm for nm in ["Intercept"] + features_W] +
                ["log_alpha_slight"] +
                ["delta_serious:" + nm for nm in ["Intercept"] + features_W] +
                ["log_alpha_serious"] +
                ["delta_fatal:"   + nm for nm in ["Intercept"] + features_W] +
                ["log_alpha_fatal"]
            )



            # ---- Build CI table ----
            df = build_param_table(
                model,
                feature_names_X=["Intercept"] + features_X,
                feature_names_Z=["Intercept"] + features_Z,
                feature_names_W=["Intercept"] + features_W
            )

            # ---- Find lowest significance (closest CI to zero) ----
            df_no_intercept = df[df["name"] != "Intercept"].copy()


            df_no_intercept["CI_over_zero"] = (df_no_intercept["CI_low"] > 0) | (df_no_intercept["CI_high"] < 0)
            nonsig = df_no_intercept[~df_no_intercept["CI_over_zero"]].copy()

            # If everything is significant → stop
            if nonsig.empty:
                print("\n🎉 All remaining parameters significant. Stopping.")
                break

            # Pick parameter with smallest |value|/sigma (weakest effect)
            nonsig["t_abs"] = (nonsig["value"].abs() / nonsig["sigma_post"])
            to_remove = nonsig.iloc[nonsig["t_abs"].argmin()]

            if verbose:
                print(f"❌ Removing parameter: {to_remove['name']} (type {to_remove['param_type']})")

            removed_params.append(to_remove['name'])

           # ---- remove feature ----
            param_to_remove = f"{to_remove['param_type']}:{to_remove['name']}"
            
            # Build new param names
            old_names = model.param_names_
            new_names = [nm for nm in old_names if nm != param_to_remove]
            
            # Build new theta_prev (mapped warm-start)
            idx_map = [old_names.index(nm) for nm in new_names]
            theta_prev = theta_prev[idx_map]
            
            # Update feature lists
            if to_remove['param_type'] == 'beta':
                features_X = [f for f in features_X if f != to_remove['name']]
            elif to_remove['param_type'] == 'gamma':
                features_Z = [f for f in features_Z if f != to_remove['name']]
            elif "delta" in to_remove['param_type']:
                features_W = [f for f in features_W if f != to_remove['name']]


        return features_X, features_Z, features_W, removed_params

def build_param_table(model, feature_names_X, feature_names_Z, feature_names_W , alpha = 0.2):
    params = model.theta_hat_
    sigma_post = model.sigma_post_

    k = len(feature_names_X)
    l = len(feature_names_Z)
    m = len(feature_names_W)

    # Extract MAP parameters from theta_hat_
    beta   = params[0:k]
    gamma  = params[k+1 : k+1+l]  # skip log_alpha_y
    offs   = k + 1 + l

    delta_s = params[offs : offs + m]
    offs += m + 1
    delta_se = params[offs : offs + m]
    offs += m + 1
    delta_f = params[offs : offs + m]

    # Convert sigmas
    sigma_beta   = sigma_post["beta"]
    sigma_gamma  = sigma_post["gamma"]
    sigma_s      = sigma_post["delta_slight"]
    sigma_se     = sigma_post["delta_serious"]
    sigma_f      = sigma_post["delta_fatal"]

    # Build combined dataframe
    df = pd.DataFrame({
        "param_type":  ["beta"]*k + ["gamma"]*l + ["delta_slight"]*m + ["delta_serious"]*m + ["delta_fatal"]*m,
        "name":        feature_names_X + feature_names_Z + feature_names_W + feature_names_W + feature_names_W,
        "value":       np.concatenate([beta, gamma, delta_s, delta_se, delta_f]),
        "sigma_post":  np.concatenate([sigma_beta, sigma_gamma, sigma_s, sigma_se, sigma_f])
    })
    # Compute z-value for the given alpha
    z = norm.ppf(1 - alpha / 2)
    df["CI_low"]  = df["value"] - z *df["sigma_post"]
    df["CI_high"] = df["value"] + z *df["sigma_post"]

    return df

def plot_param_CI(df):
    df = df.sort_values(["param_type", "name"])

    plt.figure(figsize=(10, 12))
    y_positions = np.arange(len(df))

    plt.errorbar(
        df["value"], y_positions,
        xerr=1.96 * df["sigma_post"],
        fmt="o", color="blue", ecolor="black", capsize=3
    )

    plt.yticks(y_positions, df["name"])
    plt.axvline(0, linestyle="--", color="gray")
    plt.title("Posterior Parameter Estimates with 95% Credible Intervals")
    plt.xlabel("Value")
    plt.ylabel("Parameter")
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    data_obj = Data()

    initial_X = [#'Frac_Truck','SpeedDiff_Car_Van','SpeedDiff_Truck','FreeSpeed',
                 'URBAN','Link_group_0','Link_group_1','Link_group_2','Link_group_3',
                 'Region Hovedstaden','Region Sjælland','Region Syddanmark',
                 'Region Midtjylland','Region Nordjylland']

    initial_Z = initial_X.copy()
    initial_W = initial_X.copy()

    selector = BackwardSelection(data_obj)
    final_X, final_Z, final_W, removed = selector.run(initial_X, initial_Z, initial_W)

    print("\n===== Backward Selection Complete =====")
    print("Removed parameters:", removed)
    print("Remaining in X:", final_X)
    print("Remaining in Z:", final_Z)
    print("Remaining in W:", final_W)



