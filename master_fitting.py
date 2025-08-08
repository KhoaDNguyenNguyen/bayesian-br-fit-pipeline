#python -m numpy.f2py -c --f77flags="-ffixed-line-length-none" master_combined_for_f2py.f -m fqmodel --f90flags="-mcmodel=large --no-pie"

import config
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
import logging
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
import time
import os
import fqmodel
import math
import argparse
import dynesty
from scipy.stats import norm
import corner
from concurrent.futures import ProcessPoolExecutor
import pickle

class Logger:
    """Handles logging of messages and results to a file and the console."""
    def __init__(self, log_file="[UPDATE]fitting_log.txt"):
        self.log_file = log_file
        if os.path.exists(self.log_file):
            os.remove(self.log_file)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file, mode='w'),
                logging.StreamHandler()
            ]
        )

    def log(self, message, level="info"):
        getattr(logging, level.lower())(message)

    def log_result(self, params, chi2_value):
        mc, rho = params
        self.log("\n" + "="*50)
        self.log("OPTIMIZATION COMPLETE")
        self.log(f"  - Minimum Chi-Squared (χ²): {chi2_value:.6f}")
        self.log(f"  - Best-fit mc:  {mc:.6f} GeV")
        self.log(f"  - Best-fit rho: {rho:.6f}")
        self.log("="*50 + "\n")

class DummyLogger:
    """A logger that does nothing. Used to silence child processes."""
    def log(self, message, level="info"):
        pass

class PhysicsModel:
    """
    Interfaces with the compiled Fortran (fqmodel) library to calculate
    theoretical predictions for branching ratios.
    """
    def __init__(self, logger, sampled_properties=None):
        """
        This constructor can now operate in two modes:
        1.  Default mode (sampled_properties=None): Loads the CENTRAL MEAN values
            for all particle masses and widths from the config.py file. This is used
            during the main chi-squared fitting process.
        2.  Monte Carlo mode (sampled_properties is a dict): Uses the provided
            dictionary of randomly sampled particle properties. This is used by the
            ErrorPropagator worker to account for systematic uncertainties.
        Args:
            logger: The logger instance.
            sampled_properties (dict, optional): A dictionary containing sampled
                values for particle masses and widths. Defaults to None.
        """
        self.logger = logger
        self.alpEM = config.PHYSICAL_CONSTANTS["alpEM"]
        self.ec = config.PHYSICAL_CONSTANTS["ec"]
        self.pi = config.PHYSICAL_CONSTANTS["pi"]

        if sampled_properties is None:
            props = config.PARTICLE_PROPERTIES
            self.xJpsimas = props['Jpsi']['mass']['mean']
            self.etacmas = props['etac']['mass']['mean']
            self.Psi2mas = props['psi2S']['mass']['mean']
            self.chiC0mas = props['chiC0']['mass']['mean']
            self.hcmas = props['hc']['mass']['mean']
            self.chiC1mas = props['chiC1']['mass']['mean']
            self.chiC2mas = props['chiC2']['mass']['mean']
            
            self.GamJpsitot = props['Jpsi']['width']['mean']
            self.GamChiC0tot = props['chiC0']['width']['mean']
            self.GamPsi2tot = props['psi2S']['width']['mean']
            self.Gamhctot = props['hc']['width']['mean']
            self.GamEtactot = props['etac']['width']['mean']
            self.GamChiC1tot = props['chiC1']['width']['mean']
            self.GamChiC2tot = props['chiC2']['width']['mean']
        else:
            self.xJpsimas = sampled_properties['Jpsi_mass']
            self.etacmas = sampled_properties['etac_mass']
            self.Psi2mas = sampled_properties['psi2S_mass']
            self.chiC0mas = sampled_properties['chiC0_mass']
            self.hcmas = sampled_properties['hc_mass']
            self.chiC1mas = sampled_properties['chiC1_mass']
            self.chiC2mas = sampled_properties['chiC2_mass']
            
            self.GamJpsitot = sampled_properties['Jpsi_width']
            self.GamChiC0tot = sampled_properties['chiC0_width']
            self.GamPsi2tot = sampled_properties['psi2S_width']
            self.Gamhctot = sampled_properties['hc_width']
            self.GamEtactot = sampled_properties['etac_width']
            self.GamChiC1tot = sampled_properties['chiC1_width']
            self.GamChiC2tot = sampled_properties['chiC2_width']

        try:
            fortran_defaults = config.FORTRAN_MODEL_DEFAULTS
            fqmodel.accuracy.eps = fortran_defaults["eps"]
            fqmodel.confinement.conf = fortran_defaults["conf"]
            fqmodel.confinement.xlam = fortran_defaults["xlam"]
            fqmodel.confinement.pi = self.pi
        except Exception as e:
            self.logger.log(f"Error initializing Fortran common blocks: {e}", "error")
            raise

        if not isinstance(logger, DummyLogger) and sampled_properties is None:
            self.logger.log("PhysicsModel initialized with central constants from config.py.")


    def get_jpsi_br(self, mc, rho):
        try:
            if self.xJpsimas <= self.etacmas:
                return float('nan')
            xJpsilam = self.xJpsimas * rho
            etaclam = self.etacmas * rho
            ss1, ss2 = 1.0 / xJpsilam**2, 1.0 / etaclam**2
            rJpsi = 3.0 * fqmodel.fvv_jpsietac(self.xJpsimas, ss1, mc, mc)
            gnJpsi = 1.0 / np.sqrt(rJpsi) if rJpsi > 0 else 0
            retac = 3.0 * fqmodel.fpp_jpsietac(self.etacmas, ss2, mc, mc)
            gnetac = 1.0 / np.sqrt(retac) if retac > 0 else 0
            factor = gnJpsi * gnetac * self.ec
            matrix_element = factor * (-24.0 * mc) * fqmodel.fjpsietac_jpsietac(
                self.xJpsimas, ss1, self.etacmas, ss2, mc)
            qCMS = (self.xJpsimas**2 - self.etacmas**2) / (2.0 * self.xJpsimas)
            decay_width = (self.alpEM / 3.0) * qCMS**3 * matrix_element**2
            return decay_width / self.GamJpsitot
        except Exception:
            return float('nan')
        

    def get_chic0_br(self, mc, rho):
        try:
            if self.chiC0mas <= self.xJpsimas:
                return float('nan')
            chiC0lam = self.chiC0mas * rho
            xJpsilam = self.xJpsimas * rho
            hm_jpsi, ss_jpsi = self.xJpsimas, 1.0 / xJpsilam**2
            rJpsi = 3.0 * fqmodel.fvv_chic0jpsi(hm_jpsi, ss_jpsi, mc, mc)
            gnJpsi = 1.0 / np.sqrt(rJpsi) if rJpsi > 0 else 0
            hm_chic0, ss_chic0 = self.chiC0mas, 1.0 / chiC0lam**2
            rchiC0 = 3.0 * fqmodel.fss_chic0jpsi(hm_chic0, ss_chic0, mc, mc)
            gnchiC0 = 1.0 / np.sqrt(rchiC0) if rchiC0 > 0 else 0
            hm1, hm2 = self.chiC0mas, self.xJpsimas
            ss1, ss2 = ss_chic0, ss_jpsi
            factor = gnchiC0 * gnJpsi * self.ec
            A1loop = factor*(-6.0)*fqmodel.floopa1_chic0jpsi(hm1,ss1,hm2,ss2,mc)
            A2loop = factor*(-6.0)*fqmodel.floopa2_chic0jpsi(hm1,ss1,hm2,ss2,mc)
            A1bub1 = factor*(+6.0)*fqmodel.fbub1a1_chic0jpsi(hm1,ss1,hm2,ss2,mc)
            A2bub1 = factor*(+6.0)*fqmodel.fbub1a2_chic0jpsi(hm1,ss1,hm2,ss2,mc)
            A1bub2 = factor*(+6.0)*fqmodel.fbub2a1_chic0jpsi(hm1,ss1,hm2,ss2,mc)
            A2bub2 = factor*(+6.0)*fqmodel.fbub2a2_chic0jpsi(hm1,ss1,hm2,ss2,mc)
            A1 = A1loop + A1bub1 + A1bub2
            A2 = A2loop + A2bub1 + A2bub2
            p2q =(hm1**2-hm2**2)/2.0
            squareME = A2**2 * (-hm1**-2 * p2q**2) + A1**2 * (3.0 * hm1**2)
            p2CMS = p2q/hm1
            decay_width = (self.alpEM/2.0)*(p2CMS/hm1**2)*squareME
            return decay_width / self.GamChiC0tot
        except Exception:
            return float('nan')

    def get_psi2s_br(self, mc, rho):
        try:
            if self.Psi2mas <= self.chiC0mas:
                return float('nan')
            Psi2lam = self.Psi2mas * rho
            chiC0lam = self.chiC0mas * rho
            xJpsilam = self.xJpsimas * rho
            ssPsi2 = 1.0 / Psi2lam**2 if Psi2lam != 0 else float('inf')
            ssC0 = 1.0 / chiC0lam**2 if chiC0lam != 0 else float('inf')
            ssJpsi = 1.0 / xJpsilam**2 if xJpsilam != 0 else float('inf')
            hm_chic0 = self.chiC0mas
            rchiC0 = 3.0*fqmodel.fss_psi2schic0(hm_chic0,ssC0,mc,mc)
            gnchiC0 = 1.0/np.sqrt(rchiC0) if rchiC0 > 0 else 0
            hm_psi2s = self.Psi2mas
            ssum = ssJpsi + ssPsi2
            VtoV = 3.0*fqmodel.fvtov_psi2schic0(hm_psi2s,ssum,mc)
            VtoVk2 = 3.0*fqmodel.fvtovk2_psi2schic0(hm_psi2s,ssum,mc)
            rat = -VtoV/VtoVk2 if VtoVk2 != 0 else 0
            CC1 = rat/ssPsi2 if ssPsi2 != 0 else 0
            rPsi2 = 3.0*fqmodel.fv1v1_psi2schic0(hm_psi2s,ssPsi2,mc,CC1)
            gnPsi2 = 1.0/np.sqrt(rPsi2) if rPsi2 > 0 else 0
            hm1, hm2 = self.Psi2mas, self.chiC0mas
            ss1, ss2 = ssPsi2, ssC0
            factor = gnPsi2 * gnchiC0 * self.ec
            A1loop = factor*(-6.0)*fqmodel.floopa1_psi2schic0(hm1,ss1,hm2,ss2,mc,CC1)
            A2loop = factor*(-6.0)*fqmodel.floopa2_psi2schic0(hm1,ss1,hm2,ss2,mc,CC1)
            A1bub1 = factor*(+6.0)*fqmodel.fbub1a1_psi2schic0(hm1,ss1,hm2,ss2,mc,CC1)
            A2bub1 = factor*(+6.0)*fqmodel.fbub1a2_psi2schic0(hm1,ss1,hm2,ss2,mc,CC1)
            A1bub2 = factor*(+6.0)*fqmodel.fbub2a1_psi2schic0(hm1,ss1,hm2,ss2,mc,CC1)
            A2bub2 = factor*(+6.0)*fqmodel.fbub2a2_psi2schic0(hm1,ss1,hm2,ss2,mc,CC1)
            A1 = A1loop + A1bub1 + A1bub2
            A2 = A2loop + A2bub1 + A2bub2
            qCMS=(hm1**2-hm2**2)/(2.0*hm1)
            AA1=-(qCMS/hm1)*A2
            decay_width = (self.alpEM/3.0)*qCMS*AA1**2
            return decay_width/self.GamPsi2tot
        except Exception:
            return float('nan')
            
    def get_hc_br(self, mc, rho):
        try:
            if self.hcmas <= self.etacmas:
                return float('nan')
            hclam, etaclam = self.hcmas*rho, self.etacmas*rho
            ss1, ss2 = 1.0/hclam**2, 1.0/etaclam**2
            rhc = 12.0*fqmodel.fhc_hcetac(self.hcmas, ss1, mc, mc)
            gnhc = 1.0/np.sqrt(rhc) if rhc>0 else 0
            retac = 3.0*fqmodel.fpp_hcetac(self.etacmas, ss2, mc, mc)
            gnetac = 1.0/np.sqrt(retac) if retac>0 else 0
            factor = 6.0*gnhc*gnetac*self.ec
            A1loop = -factor*fqmodel.floopa1_hcetac(self.hcmas, ss1, self.etacmas, ss2, mc)
            A2loop = -factor*fqmodel.floopa2_hcetac(self.hcmas, ss1, self.etacmas, ss2, mc)
            A1bub1a = 0.0
            A2bub1a = -factor*fqmodel.fbub1aa2_hcetac(self.hcmas, ss1, self.etacmas, ss2, mc)
            A1bub1b = +factor*fqmodel.fbub1ba1_hcetac(self.hcmas, ss1, self.etacmas, ss2, mc)
            A2bub1b = +factor*fqmodel.fbub1ba2_hcetac(self.hcmas, ss1, self.etacmas, ss2, mc)
            A1bub2 = +factor*fqmodel.fbub2a1_hcetac(self.hcmas, ss1, self.etacmas, ss2, mc)
            A2bub2 = +factor*fqmodel.fbub2a2_hcetac(self.hcmas, ss1, self.etacmas, ss2, mc)
            A1_total = A1loop + A1bub1a + A1bub1b + A1bub2
            qCMS = (self.hcmas**2-self.etacmas**2)/(2.0*self.hcmas)
            decay_width = (self.alpEM/3.0)*qCMS**3*A1_total**2
            return decay_width/self.Gamhctot
        except Exception:
            return float('nan')

    def get_etac_br(self, mc, rho):
        try:
            etaclam = self.etacmas*rho
            ss = 1.0/etaclam**2
            retac = 3.0*fqmodel.fpp_etacgg(self.etacmas, ss, mc, mc)
            gnetac = 1.0/np.sqrt(retac) if retac > 0 else 0
            matrix_element = gnetac*self.ec**2*(-24.0*mc)*fqmodel.fetacgg_etacgg(self.etacmas, ss, mc)
            decay_width = (self.pi/4.0)*self.alpEM**2*self.etacmas**3*matrix_element**2
            return decay_width/self.GamEtactot
        except Exception:
            return float('nan')

    def get_psi2s_to_chic1_br(self, mc, rho):
        """
        Calculates the theoretical branching ratio for Psi(2S) -> gamma + chi_c1.
        This channel is not used in the fit but predicted for error analysis.
        """
        try:
            if self.Psi2mas <= self.chiC1mas:
                return float('nan')
            psi_2s_lambda = self.Psi2mas * rho
            chi_c1_lambda = self.chiC1mas * rho
            j_psi_lambda = self.xJpsimas * rho

            ss_jpsi = 1.0 / j_psi_lambda**2
            
            ss_chic1 = 1.0 / chi_c1_lambda**2
            r_chic1 = 3.0 * fqmodel.faa_psi2schic1(self.chiC1mas, ss_chic1, mc, mc)

            ss_psi2s = 1.0 / psi_2s_lambda**2
            ss_sum = ss_jpsi + ss_psi2s
            v_to_v = 3.0 * fqmodel.fvtov_psi2schic1(self.Psi2mas, ss_sum, mc)
            v_to_vk2 = 3.0 * fqmodel.fvtovk2_psi2schic1(self.Psi2mas, ss_sum, mc)
            ratio = -v_to_v / v_to_vk2 if v_to_vk2 != 0 else 0.0
            orthogonality_coeff = ratio / ss_psi2s
            r_psi2s = 3.0 * fqmodel.fv1v1_psi2schic1(self.Psi2mas, ss_psi2s, mc, orthogonality_coeff)
            
            gn_chic1 = 1.0 / math.sqrt(r_chic1) if r_chic1 > 0 else 0.0
            gn_psi2s = 1.0 / math.sqrt(r_psi2s) if r_psi2s > 0 else 0.0


            hm1, hm2 = self.Psi2mas, self.chiC1mas
            ss1, ss2 = ss_psi2s, ss_chic1
            common_factor = gn_psi2s * gn_chic1 * self.ec

            w_components = {}
            for j in range(1, 6):
                floop_func = getattr(fqmodel, f"floopw{j}_psi2schic1")
                fbub1_func = getattr(fqmodel, f"fbub1w{j}_psi2schic1")
                fbub2_func = getattr(fqmodel, f"fbub2w{j}_psi2schic1")
                w_components[f'W{j}_loop'] = common_factor * (-6.0) * floop_func(hm1, ss1, hm2, ss2, mc, orthogonality_coeff)
                w_components[f'W{j}_bub1'] = common_factor * (+6.0) * fbub1_func(hm1, ss1, hm2, ss2, mc, orthogonality_coeff)
                w_components[f'W{j}_bub2'] = common_factor * (+6.0) * fbub2_func(hm1, ss1, hm2, ss2, mc, orthogonality_coeff)

            W1 = w_components['W1_loop'] + w_components['W1_bub1'] + w_components['W1_bub2']
            W2 = w_components['W2_loop'] + w_components['W2_bub1'] + w_components['W2_bub2']
            W3 = w_components['W3_loop'] + w_components['W3_bub1'] + w_components['W3_bub2']
            W5 = w_components['W5_loop'] + w_components['W5_bub1'] + w_components['W5_bub2']

            q_cms = (hm1**2 - hm2**2) / (2.0 * hm1)
            de_factor = W1 + W3 + (hm2**2 / (hm1 * q_cms)) * W5
            dm_factor = W1 + W2 + (1.0 + hm2**2 / (hm1 * q_cms)) * W5
            

            decay_width = (self.alpEM / 3.0) * q_cms**5 * (dm_factor**2 + (hm1**2 / hm2**2) * de_factor**2)
            
            return decay_width / self.GamPsi2tot
            
        except Exception:
            return float('nan')

    def get_psi2s_to_chic2_br(self, mc, rho):
        """
        Calculates the theoretical branching ratio for Psi(2S) -> gamma + chi_c2.
        The logic is adapted from the unit_test.py file. This channel is not
        used in the fit but is predicted for error analysis.
        """
        try:
            if self.Psi2mas <= self.chiC2mas:
                return float('nan')
            psi_2s_lambda = self.Psi2mas * rho
            chi_c2_lambda = self.chiC2mas * rho
            j_psi_lambda = self.xJpsimas * rho

            hm_jpsi = self.xJpsimas
            ss_jpsi = 1.0 / j_psi_lambda**2
            
            hm_psi2s = self.Psi2mas
            ss_psi2s = 1.0 / psi_2s_lambda**2
            ss_sum = ss_jpsi + ss_psi2s
            v_to_v = 3.0 * fqmodel.fvtov_psi2schic2(hm_psi2s, ss_sum, mc)
            v_to_vk2 = 3.0 * fqmodel.fvtovk2_psi2schic2(hm_psi2s, ss_sum, mc)
            ratio = -v_to_v / v_to_vk2 if v_to_vk2 != 0 else 0.0
            orthogonality_coeff = ratio / ss_psi2s
            r_psi2s = 3.0 * fqmodel.fv1v1_psi2schic2(hm_psi2s, ss_psi2s, mc, orthogonality_coeff)
            gn_psi2s = 1.0 / math.sqrt(r_psi2s) if r_psi2s > 0 else 0.0

            hm_chic2 = self.chiC2mas
            ss_chic2 = 1.0 / chi_c2_lambda**2
            r_chic2 = 3.0 * fqmodel.ftt_psi2schic2(hm_chic2, ss_chic2, mc, mc)
            gn_chic2 = 1.0 / math.sqrt(r_chic2) if r_chic2 > 0 else 0.0

            factor = 6.0 * gn_psi2s * gn_chic2 * self.ec
            
            w1_loop = -factor * fqmodel.floopw1_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)
            w2_loop = -factor * fqmodel.floopw2_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)
            w3_loop = -factor * fqmodel.floopw3_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)
            
            w3_bub1a = -factor * fqmodel.fbub1aw3_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)
            w5_bub1a = -factor * fqmodel.fbub1aw5_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)

            w1_bub1b = +factor * fqmodel.fbub1bw1_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)
            w2_bub1b = +factor * fqmodel.fbub1bw2_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)
            w3_bub1b = +factor * fqmodel.fbub1bw3_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)

            w1_bub2 = +factor * fqmodel.fbub2w1_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)
            w2_bub2 = +factor * fqmodel.fbub2w2_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)
            w3_bub2 = +factor * fqmodel.fbub2w3_psi2schic2(hm_psi2s, ss_psi2s, hm_chic2, ss_chic2, mc, orthogonality_coeff)

            W1 = w1_loop + w1_bub1b + w1_bub2
            W2 = w2_loop + w2_bub1b + w2_bub2
            W3 = w3_loop + w3_bub1a + w3_bub1b + w3_bub2
            
            p2q = (hm_psi2s**2 - hm_chic2**2) / 2.0
            F1 = W1
            F2 = 2.0 * W2
            F3 = 2.0 * W3

            # Calculate the squared matrix element (ME_sq)
            p2q2, p2q3, p2q4, p2q5, p2q6, p2q7 = [p2q**i for i in range(2, 8)]
            hm1_n2, hm2_n2, hm2_n4 = hm_psi2s**(-2), hm_chic2**(-2), hm_chic2**(-4)
            me_sq = (
                + p2q2 * F2**2 * (13.0/6.0 + 7.0/6.0 * hm1_n2 * hm_chic2**2)
                + p2q3 * F2 * F3 * (2.0/3.0 * hm1_n2 + 8.0/3.0 * hm2_n2)
                + p2q3 * F2**2 * (7.0/3.0 * hm1_n2)
                + p2q4 * F3**2 * (5.0/3.0 * hm1_n2 * hm2_n2 + 2.0/3.0 * hm2_n4)
                + p2q4 * F2 * F3 * (-2.0/3.0 * hm1_n2 * hm2_n2)
                + p2q4 * F2**2 * (hm1_n2 * hm2_n2)
                + p2q4 * F1 * F2 * (2.0/3.0 * hm1_n2 + 2.0/3.0 * hm2_n2)
                + p2q5 * F3**2 * (4.0/3.0 * hm1_n2 * hm2_n4)
                + p2q5 * F1 * F3 * (4.0/3.0 * hm1_n2 * hm2_n2 + 4.0/3.0 * hm2_n4)
                + p2q5 * F1 * F2 * (4.0/3.0 * hm1_n2 * hm2_n2)
                + p2q6 * F1 * F3 * (8.0/3.0 * hm1_n2 * hm2_n4)
                + p2q6 * F1**2 * (2.0/3.0 * hm1_n2 * hm2_n2 + 2.0/3.0 * hm2_n4)
                + p2q7 * F1**2 * (4.0/3.0 * hm1_n2 * hm2_n4)
            )
            
            p2_cms = p2q / hm_psi2s
            decay_width = (self.alpEM / 6.0) * (p2_cms / hm_psi2s**2) * me_sq
            
            return decay_width / self.GamPsi2tot

        except Exception:
            return float('nan')

    def get_chic1_to_jpsi_br(self, mc, rho):
        """
        Calculates the theoretical branching ratio for chi_c1 -> gamma + J/psi.
        The logic is adapted from the unit_test.py file. 
        """
        try:
            if self.chiC1mas <= self.xJpsimas:
                return float('nan')
            chi_c1_lambda = self.chiC1mas * rho
            j_psi_lambda = self.xJpsimas * rho

            ss_jpsi = 1.0 / j_psi_lambda**2
            r_jpsi = 3.0 * fqmodel.fvv_chic1jpsi(self.xJpsimas, ss_jpsi, mc, mc)
            gn_jpsi = 1.0 / math.sqrt(r_jpsi) if r_jpsi > 0 else 0.0

            ss_chic1 = 1.0 / chi_c1_lambda**2
            r_chic1 = 3.0 * fqmodel.faa_chic1jpsi(self.chiC1mas, ss_chic1, mc, mc)
            gn_chic1 = 1.0 / math.sqrt(r_chic1) if r_chic1 > 0 else 0.0

            hm1, hm2 = self.chiC1mas, self.xJpsimas
            ss1, ss2 = ss_chic1, ss_jpsi
            common_factor = gn_chic1 * gn_jpsi * self.ec

            w_components = {}
            for j in range(1, 6):
                w_components[f'W{j}_loop'] = common_factor * (-6.0) * getattr(fqmodel, f"floopw{j}_chic1jpsi")(hm1, ss1, hm2, ss2, mc)
                w_components[f'W{j}_bub1'] = common_factor * (+6.0) * getattr(fqmodel, f"fbub1w{j}_chic1jpsi")(hm1, ss1, hm2, ss2, mc)
                w_components[f'W{j}_bub2'] = common_factor * (+6.0) * getattr(fqmodel, f"fbub2w{j}_chic1jpsi")(hm1, ss1, hm2, ss2, mc)

            W1 = w_components['W1_loop'] + w_components['W1_bub1'] + w_components['W1_bub2']
            W2 = w_components['W2_loop'] + w_components['W2_bub1'] + w_components['W2_bub2']
            W3 = w_components['W3_loop'] + w_components['W3_bub1'] + w_components['W3_bub2']
            W5 = w_components['W5_loop'] + w_components['W5_bub1'] + w_components['W5_bub2']

            q_cms = (hm1**2 - hm2**2) / (2.0 * hm1)
            
            de_factor = W1 + W3 + (hm2**2 / (hm1 * q_cms)) * W5
            dm_factor = W1 + W2 + (1.0 + hm2**2 / (hm1 * q_cms)) * W5
            decay_width = (self.alpEM / 3.0) * q_cms**5 * (dm_factor**2 + (hm1**2 / hm2**2) * de_factor**2)
            
            return decay_width / self.GamChiC1tot
            
        except Exception:
            return float('nan')

    def get_chic2_to_jpsi_br(self, mc, rho):
        """
        Calculates the theoretical branching ratio for chi_c2 -> gamma + J/psi.
        The logic is adapted from the unit_test.py file. This channel is not
        used in the fit but is predicted for error analysis.
        """
        try:
            if self.chiC2mas <= self.xJpsimas:
                return float('nan')
            chi_c2_lambda = self.chiC2mas * rho
            j_psi_lambda = self.xJpsimas * rho

            ss_jpsi = 1.0 / j_psi_lambda**2
            r_jpsi = 3.0 * fqmodel.fvv_chic2jpsi(self.xJpsimas, ss_jpsi, mc, mc)
            gn_jpsi = 1.0 / math.sqrt(r_jpsi) if r_jpsi > 0 else 0.0

            ss_chic2 = 1.0 / chi_c2_lambda**2
            r_chic2 = 3.0 * fqmodel.ftt_chic2jpsi(self.chiC2mas, ss_chic2, mc, mc)
            gn_chic2 = 1.0 / math.sqrt(r_chic2) if r_chic2 > 0 else 0.0

            hm1, hm2 = self.chiC2mas, self.xJpsimas
            ss1, ss2 = ss_chic2, ss_jpsi
            factor = 6.0 * gn_chic2 * gn_jpsi * self.ec
            
            w1_loop = -factor * fqmodel.floopw1_chic2jpsi(hm1, ss1, hm2, ss2, mc)
            w2_loop = -factor * fqmodel.floopw2_chic2jpsi(hm1, ss1, hm2, ss2, mc)
            w3_loop = -factor * fqmodel.floopw3_chic2jpsi(hm1, ss1, hm2, ss2, mc)

            w3_bub1a = -factor * fqmodel.fbub1aw3_chic2jpsi(hm1, ss1, hm2, ss2, mc)
            w5_bub1a = -factor * fqmodel.fbub1aw5_chic2jpsi(hm1, ss1, hm2, ss2, mc)

            w1_bub1b = +factor * fqmodel.fbub1bw1_chic2jpsi(hm1, ss1, hm2, ss2, mc)
            w2_bub1b = +factor * fqmodel.fbub1bw2_chic2jpsi(hm1, ss1, hm2, ss2, mc)
            w3_bub1b = +factor * fqmodel.fbub1bw3_chic2jpsi(hm1, ss1, hm2, ss2, mc)
            
            w1_bub2 = +factor * fqmodel.fbub2w1_chic2jpsi(hm1, ss1, hm2, ss2, mc)
            w2_bub2 = +factor * fqmodel.fbub2w2_chic2jpsi(hm1, ss1, hm2, ss2, mc)
            w3_bub2 = +factor * fqmodel.fbub2w3_chic2jpsi(hm1, ss1, hm2, ss2, mc)

            W1 = w1_loop + w1_bub1b + w1_bub2
            W2 = w2_loop + w2_bub1b + w2_bub2
            W3 = w3_loop + w3_bub1a + w3_bub1b + w3_bub2
            
            p2q = (hm1**2 - hm2**2) / 2.0
            F1 = W1
            F2 = 2.0 * W2
            F3 = 2.0 * W3

            p2q2, p2q3, p2q4, p2q5, p2q6 = [p2q**j for j in range(2, 7)]
            hm1_n2, hm1_n4 = hm1**(-2), hm1**(-4)
            hm2_2, hm2_4 = hm2**2, hm2**4
            hm2_n2 = hm2**(-2)

            me_sq = (
                + p2q2 * F2**2 * (7.0/6.0 + (2.0/3.0) * hm1_n4 * hm2_4 + (3.0/2.0) * hm1_n2 * hm2_2)
                + p2q3 * F2 * F3 * (-(4.0/3.0) * hm1_n4 * hm2_2 + (11.0/3.0) * hm1_n2 + hm2_n2)
                + p2q3 * F2**2 * ((8.0/3.0) * hm1_n4 * hm2_2 + 3.0 * hm1_n2)
                + p2q4 * F3**2 * (2.0 * hm1_n4 + (1.0/3.0) * hm1_n2 * hm2_n2)
                + p2q4 * F2 * F3 * (-4.0 * hm1_n4 + (4.0/3.0) * hm1_n2 * hm2_n2)
                + p2q4 * F2**2 * ((10.0/3.0) * hm1_n4 + (1.0/3.0) * hm1_n2 * hm2_n2)
                + p2q4 * F1 * F2 * (-(4.0/3.0) * hm1_n4 * hm2_2 + (8.0/3.0) * hm1_n2)
                + p2q5 * F3**2 * ((4.0/3.0) * hm1_n4 * hm2_n2)
                + p2q5 * F2 * F3 * (-(8.0/3.0) * hm1_n4 * hm2_n2)
                + p2q5 * F2**2 * ((4.0/3.0) * hm1_n4 * hm2_n2)
                + p2q5 * F1 * F3 * ((8.0/3.0) * hm1_n4)
                + p2q5 * F1 * F2 * (-(8.0/3.0) * hm1_n4)
                + p2q6 * F1**2 * ((4.0/3.0) * hm1_n4)
            )

            p2_cms = p2q / hm1
            decay_width = (self.alpEM / 10.0) * (p2_cms / hm1**2) * me_sq
            
            return decay_width / self.GamChiC2tot
            
        except Exception:
            return float('nan')
        

def _calculate_chi2_static(params):
    """
    Static worker function to calculate chi-squared for a given (mc, rho) pair.
    This is used by the ProcessPoolExecutor for parallel grid scans.
    It reads experimental data directly from the config file.
    """
    model = PhysicsModel(logger=DummyLogger())
    mc, rho = params

    exp_data = config.EXPERIMENTAL_DATA

    br_jpsi_th = model.get_jpsi_br(mc, rho)
    br_chic0_th = model.get_chic0_br(mc, rho)
    br_psi2s_th = model.get_psi2s_br(mc, rho)

    if math.isnan(br_jpsi_th) or math.isnan(br_chic0_th) or math.isnan(br_psi2s_th):
        return params, float('inf')

    chi2_jpsi = ((br_jpsi_th - exp_data['jpsi']['br']) / exp_data['jpsi']['sigma'])**2
    chi2_chic0 = ((br_chic0_th - exp_data['chic0']['br']) / exp_data['chic0']['sigma'])**2
    chi2_psi2s = ((br_psi2s_th - exp_data['psi2s']['br']) / exp_data['psi2s']['sigma'])**2

    total_chi2 = chi2_chic0 + chi2_psi2s + chi2_jpsi

    if not np.isfinite(total_chi2):
        return params, float('inf')

    return params, total_chi2


class Fitter:
    def __init__(self, model, logger):
        self.model = model
        self.logger = logger
        self.iter_count = 0
        self.exp_data = config.EXPERIMENTAL_DATA
        
    def calculate_chi2_terms(self, params):
        mc, rho = params
        br_jpsi_th = self.model.get_jpsi_br(mc, rho)
        br_chic0_th = self.model.get_chic0_br(mc, rho)
        br_psi2s_th = self.model.get_psi2s_br(mc, rho)
        
        chi2_jpsi = ((br_jpsi_th - self.exp_data['jpsi']['br']) / self.exp_data['jpsi']['sigma'])**2 if not math.isnan(br_jpsi_th) else float('inf')
        chi2_chic0 = ((br_chic0_th - self.exp_data['chic0']['br']) / self.exp_data['chic0']['sigma'])**2 if not math.isnan(br_chic0_th) else float('inf')
        chi2_psi2s = ((br_psi2s_th - self.exp_data['psi2s']['br']) / self.exp_data['psi2s']['sigma'])**2 if not math.isnan(br_psi2s_th) else float('inf')
        
        return {'jpsi': chi2_jpsi, 'chic0': chi2_chic0, 'psi2s': chi2_psi2s}

    def calculate_chi2(self, params):
        total_chi2 = sum(self.calculate_chi2_terms(params).values())
        return total_chi2 if np.isfinite(total_chi2) else 1e30

    def _callback(self, xk):
        self.iter_count += 1
        chi2 = self.calculate_chi2(xk)
        self.logger.log(f"  Iter {self.iter_count:3d}: mc={xk[0]:.6f}, rho={xk[1]:.6f}, χ²={chi2:.6f}")

    def find_best_fit(self):
        self.logger.log("Starting efficient optimization...")
        self.iter_count = 0
        bounds = config.FIT_CONFIG_2D["bounds"]
        initial_guess = config.FIT_CONFIG_2D["initial_guess"]
        
        result = optimize.minimize(
            self.calculate_chi2, initial_guess, method='L-BFGS-B',
            bounds=bounds, callback=self._callback,
            options={'disp': False, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        self.logger.log_result(result.x, result.fun)
        return result

    def generate_chi2_map(self, resolution=25):
        self.logger.log(f"Starting exhaustive grid scan ({resolution}x{resolution} points)...")
        start_time = time.time()
        mc_range = np.linspace(1.6, 1.9, resolution)
        rho_range = np.linspace(0.3, 1.2, resolution)
        mc_grid, rho_grid = np.meshgrid(mc_range, rho_range)
        param_pairs = list(product(mc_range, rho_range))
        
        results_map = {}
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_calculate_chi2_static, pair) for pair in param_pairs]
            for i, future in enumerate(as_completed(futures)):
                try:
                    params, chi2_val = future.result()
                    results_map[params] = chi2_val
                except Exception as e:
                    self.logger.log(f"A worker process failed for params {param_pairs[i]}: {e}", "error")
                    results_map[param_pairs[i]] = float('inf')

                print(f"  ... scanned {i + 1}/{len(param_pairs)} points", end='\r')
        
        print() 
        self.logger.log("Grid scan complete. Assembling results...")
        chi2_values = [results_map.get(tuple(pair), float('inf')) for pair in param_pairs]
        chi2_grid = np.array(chi2_values).reshape((resolution, resolution)).T

        end_time = time.time()
        self.logger.log(f"Grid scan completed in {end_time - start_time:.2f} seconds.")
        return mc_grid, rho_grid, chi2_grid

    def analyze_best_fit(self, best_params):
        self.logger.log("\nAnalyzing chi-squared contributions at best-fit point...")
        terms = self.calculate_chi2_terms(best_params)
        total_chi2 = sum(terms.values())
        if not np.isfinite(total_chi2):
            self.logger.log("Best-fit point resulted in invalid chi-squared; cannot analyze contributions.", "warning")
            return
            
        self.logger.log(f"  {'Channel':<20} | {'χ² Contribution':<20} | {'Percentage':<15}")
        self.logger.log(f"  {'-'*20} | {'-'*20} | {'-'*15}")
        for key, value in terms.items():
            name = self.exp_data[key]['name']
            percentage = (value / total_chi2) * 100 if total_chi2 > 0 else 0
            self.logger.log(f"  {name:<20} | {value:<20.4f} | {percentage:<15.2f}%")
        self.logger.log(f"  {'-'*20} | {'-'*20} | {'-'*15}")
        self.logger.log(f"  {'Total χ²':<20} | {total_chi2:<20.4f} | {'100.00':<15}%")

    def generate_zoomed_chi2_map(self, resolution, center_mc, center_rho, width_mc, width_rho):
        self.logger.log(f"\nStarting ZOOMED grid scan ({resolution}x{resolution} points) around mc={center_mc:.4f}, rho={center_rho:.4f}")
        start_time = time.time()
        
        mc_range = np.linspace(center_mc - width_mc, center_mc + width_mc, resolution)
        rho_range = np.linspace(center_rho - width_rho, center_rho + width_rho, resolution)
        
        mc_grid, rho_grid = np.meshgrid(mc_range, rho_range)
        param_pairs = list(product(mc_range, rho_range))
        
        results_map = {}
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(_calculate_chi2_static, pair) for pair in param_pairs]
            for i, future in enumerate(as_completed(futures)):
                try:
                    params, chi2_val = future.result()
                    results_map[params] = chi2_val
                except Exception as e:
                    self.logger.log(f"A worker process failed for params {param_pairs[i]}: {e}", "error")
                    results_map[param_pairs[i]] = float('inf')

                print(f"  ... scanned {i + 1}/{len(param_pairs)} points", end='\r')
        
        print() 
        self.logger.log("Zoomed grid scan complete. Assembling results...")
        chi2_values = [results_map.get(tuple(pair), float('inf')) for pair in param_pairs]
        chi2_grid = np.array(chi2_values).reshape((resolution, resolution)).T

        end_time = time.time()
        self.logger.log(f"Zoomed grid scan completed in {end_time - start_time:.2f} seconds.")
        return mc_grid, rho_grid, chi2_grid


# --- Utility: Define the parameter space precisely ---
# The order of parameters is crucial and must be consistent across all functions.
# Order: [mc, rho, 14 nuisance parameters from PDG]
NUISANCE_PARAM_ORDER = [
    ('Jpsi', 'mass'), ('Jpsi', 'width'),
    ('etac', 'mass'), ('etac', 'width'),
    ('psi2S', 'mass'), ('psi2S', 'width'),
    ('chiC0', 'mass'), ('chiC0', 'width'),
    ('hc', 'mass'), ('hc', 'width'),
    ('chiC1', 'mass'), ('chiC1', 'width'),
    ('chiC2', 'mass'), ('chiC2', 'width'),
]

MC_BOUNDS = config.FIT_CONFIG_2D["bounds"][0]
RHO_BOUNDS = config.FIT_CONFIG_2D["bounds"][1]

N_DIM = 2 + len(NUISANCE_PARAM_ORDER) # 2 main params + 14 nuisance params = 16 dimensions


def calculate_chi2_robust(params):
    """
    The manual kinematic check has been REMOVED because:
    1. It makes sampling from independent priors nearly impossible.
    2. The underlying PhysicsModel class already handles these checks internally
       by returning NaN, which correctly results in chi2=inf.
    3. The outer try/except block serves as the final safety net.
    
    Total Chi2 = Chi2_decays + Chi2_nuisance_penalty

    Args:
        params (list or np.array): A 14-element list of parameters in the defined order.
    Returns:
        float: The calculated total chi-squared, or float('inf') if any error occurs.
    """
    mc, rho = params[0], params[1]

    try:
        # --- Unpack parameters and build the properties dictionary for the model ---
        sampled_properties = {}
        for i, (particle, prop_type) in enumerate(NUISANCE_PARAM_ORDER):
            key = f"{particle}_{prop_type}" 
            sampled_properties[key] = params[i + 2]
        model = PhysicsModel(logger=DummyLogger(), sampled_properties=sampled_properties)
        fitter_instance = Fitter(model, DummyLogger())
        chi2_decays = fitter_instance.calculate_chi2([mc, rho])

        # If the decay calculation fails (e.g., due to kinematics), chi2 will be inf.
        # We can stop here, as inf + anything is still inf.
        if not np.isfinite(chi2_decays):
            return float('inf')
        chi2_penalty = 0.0
        for i, (particle, prop_type) in enumerate(NUISANCE_PARAM_ORDER):
            sampled_value = params[i + 2]
            mean = config.PARTICLE_PROPERTIES[particle][prop_type]['mean']
            error = config.PARTICLE_PROPERTIES[particle][prop_type]['error']
            chi2_penalty += ((sampled_value - mean) / error)**2

        total_chi2 = chi2_decays + chi2_penalty
        return total_chi2

    except Exception:
        # This is the ultimate safety net for any unexpected crashes
        # from the Fortran code or other issues.
        return float('inf')
# =============================================================================
# STAGE 1: BAYESIAN INFERENCE WITH DYNESTY
# =============================================================================
def prior_transform(u):
    """
     Transforms the unit cube `u` to the physical parameter space.
    This version uses UNIFORM priors for all 16 parameters to ensure the sampler
    can easily find valid points. The Gaussian constraints are moved into the
    likelihood function as penalty terms.

    Args:
        u (np.array): 14-element array of values in [0, 1].

    Returns:
        np.array: 14-element array of physical parameter values.
    """
    params = np.zeros_like(u)

    params[0] = u[0] * (MC_BOUNDS[1] - MC_BOUNDS[0]) + MC_BOUNDS[0]  # mc
    params[1] = u[1] * (RHO_BOUNDS[1] - RHO_BOUNDS[0]) + RHO_BOUNDS[0] # rho

    # --- Transform nuisance parameters (NOW ALSO UNIFORM PRIORS) ---
    # This creates a 16-dimensional hyper-rectangle that is easy to sample from.
    for i, (particle, prop_type) in enumerate(NUISANCE_PARAM_ORDER):
        mean = config.PARTICLE_PROPERTIES[particle][prop_type]['mean']
        error = config.PARTICLE_PROPERTIES[particle][prop_type]['error']
        
        low_bound = mean - 7 * error
        high_bound = mean + 7 * error

        # Ensure the lower bound is not negative for widths
        if prop_type == 'width':
            low_bound = max(0, low_bound)

        params[i + 2] = u[i + 2] * (high_bound - low_bound) + low_bound

    return params

def log_likelihood(params):
    """
    The log-likelihood function for dynesty.
    It's simply -0.5 * chi-squared.

    Args:
        params (np.array): 16-element array of physical parameter values.

    Returns:
        float: The log-likelihood value.
    """
    chi2 = calculate_chi2_robust(params)
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2


def run_bayesian_analysis(executor):
    """
    Executes the main Nested Sampling run using dynesty.
    Args:
        executor (ProcessPoolExecutor): The executor object for parallel processing.
    """
    logger = Logger(log_file="[BAYESIAN]_dynesty_log.txt")
    logger.log("--- Starting Stage 1: Bayesian Inference with dynesty ---")
    
    n_workers = executor._max_workers
    logger.log(f"Parameter space dimension: {N_DIM}")
    logger.log(f"Using {n_workers} CPU cores for parallel processing.")


    sampler = dynesty.NestedSampler(log_likelihood, prior_transform, N_DIM,
                                    nlive=1500,       # More live points for higher-D
                                    bound='multi',
                                    sample='rwalk',   # Random walk is good for correlated params
                                    pool=executor,    # Pass the executor object itself
                                    queue_size=n_workers)

    # dlogz is the stopping criterion. Lower values mean a more thorough run.
    sampler.run_nested(dlogz=0.05)
    results = sampler.results

    with open('pkl/dynesty_results.pkl', 'wb') as f:
        pickle.dump(results, f)

    logger.log("--- Dynesty run complete. Results saved to 'dynesty_results.pkl' ---")
    return results

# =============================================================================
# STAGE 2: POST-ANALYSIS AND GOODNESS-OF-FIT
# =============================================================================

def predict_br(params):
    """
    Worker function for posterior predictive checks.
    Calculates the theoretical branching ratios for a given parameter set.
    Returns:
        tuple: (br_chic0, br_psi2s) or (nan, nan) on failure.
    """
    mc, rho = params[0], params[1]
    try:
        sampled_properties = {}
        for i, (particle, prop_type) in enumerate(NUISANCE_PARAM_ORDER):
            key = f"{particle}_{prop_type}"
            sampled_properties[key] = params[i + 2]

        model = PhysicsModel(logger=DummyLogger(), sampled_properties=sampled_properties)

        br_jpsi = model.get_jpsi_br(mc, rho)
        br_chic0 = model.get_chic0_br(mc, rho)
        br_psi2s = model.get_psi2s_br(mc, rho)

        return (br_jpsi, br_chic0, br_psi2s)
    except Exception:
        return (float('nan'), float('nan'), float('nan'))

def run_post_analysis(results, pool):
    """
    Performs posterior predictive checks and evaluates goodness-of-fit.
    [MODIFIED TO FULLY EMBED LATEX FOR PLOTTING]
    """
    logger = Logger(log_file="[POST_ANALYSIS]_log.txt")
    logger.log("--- Starting Stage 2: Post-Analysis and Goodness-of-Fit ---")
    samples = results.samples
    weights = np.exp(results.logwt - results.logz[-1])
    mean, cov = dynesty.utils.mean_and_cov(samples, weights)
    
    logger.log("\n--- Parameter Estimation Summary ---")
    param_names = ['mc', 'rho'] + [f"{p}_{t}" for p, t in NUISANCE_PARAM_ORDER]
    for i in range(N_DIM):
        quantiles = dynesty.utils.quantile(samples[:, i], [0.15865, 0.5, 0.84135], weights=weights)
        median, err_low, err_high = quantiles[1], quantiles[1] - quantiles[0], quantiles[2] - quantiles[1]
        logger.log(f"{param_names[i]:>12s} = {median:.5f} +{err_high:.5f} / -{err_low:.5f}")

    max_logl_index = np.argmax(results.logl)
    chi2_min = -2 * results.logl[max_logl_index]
    best_fit_params = results.samples[max_logl_index]
    dof = len(config.EXPERIMENTAL_DATA) - 2
    
    logger.log("\n--- Goodness-of-Fit Summary ---")
    logger.log(f"Best-fit parameters found by dynesty: mc={best_fit_params[0]:.5f}, rho={best_fit_params[1]:.5f}")
    logger.log(f"Minimum Chi-Squared (from max log-likelihood): {chi2_min:.4f}")
    logger.log(f"Degrees of Freedom (d.o.f): {dof}")
    if dof > 0:
        chi2_per_dof = chi2_min / dof
        logger.log(f"Chi-Squared / d.o.f: {chi2_per_dof:.4f}")
        if chi2_per_dof > 1.5:
            logger.log("WARNING: Chi2/dof is high, suggesting tension between the model and data.", "warning")
    else:
        logger.log("Chi-Squared / d.o.f: Not applicable (d.o.f is not positive).")
        if abs(chi2_min) > 1e-6:
            logger.log("WARNING: Minimum chi-squared is not close to zero, which is unexpected for dof=0.", "warning")

    logger.log("\n--- Performing Posterior Predictive Check (PPC) ---")
    ppc_samples = results.samples_equal()
    logger.log(f"Calculating theoretical predictions for {len(ppc_samples)} posterior samples...")
    predictions = list(pool.map(predict_br, ppc_samples))
    with open('pkl/ppc_predictions.pkl', 'wb') as f:
        pickle.dump(predictions, f)
    logger.log("PPC predictions calculated and saved to 'ppc_predictions.pkl'.")
    logger.log("Plotting code can now use this file to compare predicted distributions to data.")

    logger.log("Generating corner plot for main parameters...")

    try:
        plt.rcParams.update({
            "text.usetex": True,                     
            "font.family": "serif",                 
            "font.serif": ["Computer Modern Roman"], 
            "font.size": 12,                      
            "axes.labelsize": 14,                 
            "xtick.labelsize": 12,                 
            "ytick.labelsize": 12,
            "text.latex.preamble": r"""
                \usepackage{amsmath}
                \usepackage{siunitx}  % Gói để định dạng đơn vị chuẩn
            """,
        })
        logger.log("Successfully configured Matplotlib to use LaTeX for rendering.")
    except Exception as e:
        logger.log(f"WARNING: Failed to configure LaTeX. Using default backend. Error: {e}", "warning")

    corner_samples = dynesty.utils.resample_equal(samples, weights)[:, :2]
    
    latex_labels = [r'$m_c~[\mathrm{GeV}]$', r'$\rho$']

    fig = corner.corner(corner_samples, 
                        labels=latex_labels,  
                        quantiles=[0.16, 0.5, 0.84], 
                        show_titles=True,
                        title_kwargs={"fontsize": 12}) 
    
    fig.savefig("corner_plot_mc_rho.eps", format='eps', bbox_inches='tight')
    logger.log("Corner plot saved as 'corner_plot_mc_rho.eps'.")
    
# =============================================================================
# STAGE 3: FREQUENTIST CROSS-CHECK VIA PROFILE LIKELIHOOD
# =============================================================================
GLOBAL_CHI2_MIN = float('inf')

def chi2_for_profiling(variable_params, fixed_param_value, fixed_param_index):
    """
    The objective function for `differential_evolution`.
    It constructs the full 16-param vector and calls the robust chi2 calculator.
    
    Args:
        variable_params (np.array): 15 parameters to be optimized.
        fixed_param_value (float): The value of the parameter being profiled.
        fixed_param_index (int): The index (0 for mc, 1 for rho) of the fixed parameter.
        
    Returns:
        float: Calculated chi-squared.
    """
    full_params = np.insert(variable_params, fixed_param_index, fixed_param_value)
    return calculate_chi2_robust(full_params)

def calculate_profile_point(fixed_param_value, fixed_param_index):
    """
    A worker function that finds the minimum chi2 for a single point on the profile.
    It optimizes the other 15 "nuisance" parameters.
    """
    bounds = []
    param_index_counter = 0
    # mc and rho bounds
    if fixed_param_index != 0: bounds.append(MC_BOUNDS)
    if fixed_param_index != 1: bounds.append(RHO_BOUNDS)
    # Nuisance parameter bounds
    for i, (p, t) in enumerate(NUISANCE_PARAM_ORDER):
        mean = config.PARTICLE_PROPERTIES[p][t]['mean']
        err = config.PARTICLE_PROPERTIES[p][t]['error']
        bounds.append((mean - 10 * err, mean + 10 * err))

    result = optimize.differential_evolution(
        chi2_for_profiling,
        bounds=bounds,
        args=(fixed_param_value, fixed_param_index),
        strategy='best1bin',
        maxiter=300,
        popsize=20,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=False
    )
    
    print(f"  ... profiled point {fixed_param_value:.4f}, found min chi2 = {result.fun:.4f}")
    return (fixed_param_value, result.fun)
    
def run_profile_likelihood_crosscheck(pool):
    """
    Performs the profile likelihood analysis for mc and rho.
    """
    logger = Logger(log_file="[PROFILE_LIKELIHOOD]_log.txt")
    logger.log("--- Starting Stage 3: Frequentist Cross-Check via Profile Likelihood ---")

    logger.log("Finding global minimum chi-squared...")
    all_bounds = [MC_BOUNDS, RHO_BOUNDS]
    for i, (p, t) in enumerate(NUISANCE_PARAM_ORDER):
        mean = config.PARTICLE_PROPERTIES[p][t]['mean']
        err = config.PARTICLE_PROPERTIES[p][t]['error']
        all_bounds.append((mean - 3 * err, mean + 3 * err))
    
    global_min_result = optimize.differential_evolution(calculate_chi2_robust, bounds=all_bounds, disp=True)
    
    global GLOBAL_CHI2_MIN
    GLOBAL_CHI2_MIN = global_min_result.fun
    
    logger.log(f"Global minimum chi-squared found: {GLOBAL_CHI2_MIN:.4f}")
    
    # --- Profile mc ---
    logger.log("\n--- Profiling mc ---")
    mc_grid = np.linspace(MC_BOUNDS[0], MC_BOUNDS[1], 50) 
    
    from functools import partial
    worker_for_mc = partial(calculate_profile_point, fixed_param_index=0)
    
    mc_profile_results = list(pool.map(worker_for_mc, mc_grid))
    with open('pkl/profile_mc.pkl', 'wb') as f:
        pickle.dump(mc_profile_results, f)
    logger.log("mc profile complete. Results saved to 'profile_mc.pkl'.")


    logger.log("\n--- Profiling rho ---")
    rho_grid = np.linspace(RHO_BOUNDS[0], RHO_BOUNDS[1], 50)
    worker_for_rho = partial(calculate_profile_point, fixed_param_index=1)
    
    rho_profile_results = list(pool.map(worker_for_rho, rho_grid))
    with open('pkl/profile_rho.pkl', 'wb') as f:
        pickle.dump(rho_profile_results, f)
    logger.log("rho profile complete. Results saved to 'pkl/profile_rho.pkl'.")
    logger.log("\nCross-check finished. Plotting code can now use the .pkl files to draw Delta Chi2 plots.")

# =============================================================================
# MAIN ANALYSIS DRIVER
# =============================================================================
if __name__ == '__main__':
    """
    Main execution block to run the full analysis pipeline.
    """
    print("=====================================================")
    print("=== GOLD STANDARD CHARMONIUM ANALYSIS PIPELINE ====")
    print("=====================================================")

    # This automatically uses all available CPU cores
    with ProcessPoolExecutor() as executor:
        print("\n--- STAGE 1: EXECUTING BAYESIAN INFERENCE ---")

        results = run_bayesian_analysis(executor)

        # To reload results instead of running again, uncomment the following lines:
        # print("--- STAGE 1: SKIPPED (loading from 'pkl/dynesty_results.pkl') ---")
        with open('pkl/dynesty_results.pkl', 'rb') as f:
            results = pickle.load(f)

        print("\n--- STAGE 2: EXECUTING POST-ANALYSIS & GOF ---")
        run_post_analysis(results, executor)

        print("\n--- STAGE 3: EXECUTING PROFILE LIKELIHOOD CROSS-CHECK ---")
        run_profile_likelihood_crosscheck(executor)

    print("\n=====================================================")
    print("=== ANALYSIS PIPELINE COMPLETE ===")
    print("=====================================================")