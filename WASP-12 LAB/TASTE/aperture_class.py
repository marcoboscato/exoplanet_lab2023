import numpy as np
import pickle
from astropy.io import fits

class AperturePhotometry:
    def __init__(self):
        #self.data_path = './data/'
        
        self.readout_noise = 7.4  # [e] photoelectrons
        self.gain = 1.91 # [e/ADU]
        
        self.bias_std = 1.3 # [e] photoelectrons
        self.median_bias = pickle.load(open('output_files/median_bias.p', 'rb'))
        self.median_bias_errors = pickle.load(open('output_files/median_bias_error.p', 'rb'))
        # put the value here if you decided to use a constant value
        self.median_normalized_flat = pickle.load(open('output_files/median_normalized_flat.p', 'rb'))
        self.median_normalized_flat_errors = pickle.load(open('output_files/median_normalized_flat_errors.p', 'rb'))
        
        self.science_path = '../data/'
        self.science_list = np.genfromtxt(self.science_path + 'data.list', dtype=str)
        self.science_size = len(self.science_list)
        
        ylen, xlen  = np.shape(self.median_bias)
        self.X_axis = np.arange(0, xlen, 1)
        self.Y_axis = np.arange(0, ylen, 1)
        self.X, self.Y = np.meshgrid(self.X_axis, self.Y_axis)

    #---------------------------------------------------------------------------------------------------
    # 1)
    def correct_science_frame(self, science_frame):
        science_debiased = science_frame - self.median_bias
        science_corrected = science_debiased / self.median_normalized_flat

        ## Error associated to the science corrected frame
        science_debiased_errors = np.sqrt(self.readout_noise**2 + self.bias_std**2 + science_debiased)
        science_corrected_errors = science_corrected * np.sqrt((science_debiased_errors/science_debiased)**2 + (self.median_normalized_flat_errors/self.median_normalized_flat)**2)
        
        return science_corrected, science_corrected_errors
    
    # 2)
    def compute_centroid(self, science_frame, x_target_initial, y_target_initial, maximum_number_of_iterations=20):

        for i_iter in range(0, maximum_number_of_iterations):

            if i_iter == 0:
                # first iteration
                x_target_previous = x_target_initial
                y_target_previous = y_target_initial
            else:
                # using the previous result as starting point
                x_target_previous = x_target_refined
                y_target_previous = y_target_refined

            # 2D array with the distance of each pixel from the target star 
            target_distance = np.sqrt((self.X-x_target_previous)**2 + (self.Y-y_target_previous)**2)

            # Selection of the pixels within the inner radius
            annulus_sel = (target_distance < self.sky_inner_radius)

            # Weighted sum of coordinates
            weighted_X = np.sum(science_frame[annulus_sel]*self.X[annulus_sel])
            weighted_Y = np.sum(science_frame[annulus_sel]*self.Y[annulus_sel])

            # Sum of the weights
            total_flux = np.sum(science_frame[annulus_sel])


            # Refined determination of coordinates
            x_target_refined = weighted_X/total_flux
            y_target_refined = weighted_Y/total_flux

            percent_variance_x = (x_target_refined-x_target_previous)/(x_target_previous) * 100.
            percent_variance_y = (y_target_refined-y_target_previous)/(y_target_previous) * 100.
            # exit condition: both percent variance are smaller than 0.1%
            if np.abs(percent_variance_x)<0.1 and  np.abs(percent_variance_y)<0.1:
                  break

        return x_target_refined, y_target_refined
    
    # 3)
    def compute_sky_background(self, science_frame, science_frame_err, x_pos, y_pos):
        target_distance = np.sqrt((self.X-x_pos)**2 + (self.Y-y_pos)**2)

        annulus_selection = (target_distance > self.sky_inner_radius) & (target_distance<=self.sky_outer_radius)

        sky_flux_average = np.sum(science_frame[annulus_selection]) / np.sum(annulus_selection)
        sky_flux_err_average = np.sum(science_frame_err[annulus_selection])/np.sum(annulus_selection)
        return sky_flux_average, sky_flux_err_average
    
    # 4) 
    def calculate_FWHM(self, reference_axis, normalized_cumulative_distribution):
        # Find the closest point to NCD= 0.15865 (-1 sigma)
        NCD_index_left = np.argmin(np.abs(normalized_cumulative_distribution-0.15865))
    
        # Find the closest point to NCD= 0.84135 (+1 sigma)
        NCD_index_right = np.argmin(np.abs(normalized_cumulative_distribution-0.84135))

        # We model the NCD around the -1sgima value with a polynomial curve. 
        # The independet variable is actually the normalized cumulative distribution, 
        # the depedent variable is the pixel position
        p_fitted = np.polynomial.Polynomial.fit(normalized_cumulative_distribution[NCD_index_left-1: NCD_index_left+2],
                                                reference_axis[NCD_index_left-1: NCD_index_left+2],
                                                deg=2)

        # We get a more precise estimate of the pixel value corresponding to the -1sigma position
        pixel_left = p_fitted(0.15865)

        # We repeat the step for the 1sigma value
        p_fitted = np.polynomial.Polynomial.fit(normalized_cumulative_distribution[NCD_index_right-1: NCD_index_right+2],
                                                reference_axis[NCD_index_right-1: NCD_index_right+2],
                                                deg=2)
        pixel_right = p_fitted(0.84135)

        #print(pixel_left, pixel_right)
    
        FWHM_factor = 2 * np.sqrt(2 * np.log(2)) # = 2.35482
        FWHM = (pixel_right-pixel_left)/2. * FWHM_factor

        return FWHM
    
    # --------------------------------------------------------------------------------------------------------------------------
    # Final Function for aperture photometry
    def aperture_photometry(self, sky_inner_radius, sky_outer_radius, aperture_radius, x_initial, y_initial):
        #provide aperture parameters
        self.sky_inner_radius = sky_inner_radius 
        self.sky_outer_radius = sky_outer_radius 
        self.aperture_radius = aperture_radius 
        self.x_initial = x_initial
        self.y_initial = y_initial

        # list of empty numpy's arrays
        self.airmass = np.empty(self.science_size)
        self.exptime = np.empty(self.science_size)
        self.julian_date = np.empty(self.science_size)

        self.aperture = np.empty(self.science_size)
        self.aperture_errors = np.empty(self.science_size)

        self.sky_background = np.empty(self.science_size)
        self.sky_background_errors = np.empty(self.science_size)

        self.x_refined = np.empty(self.science_size)
        self.y_refined = np.empty(self.science_size)

        self.x_fwhm = np.empty(self.science_size)
        self.y_fwhm = np.empty(self.science_size)

        # starting the computation
        for ii_science, science_name in enumerate(self.science_list):
            science_fits = fits.open(self.science_path + science_name)
            
            #You must read the info from the header before closing the file
            self.airmass[ii_science] = science_fits[0].header['AIRMASS']
            self.exptime[ii_science] = science_fits[0].header['EXPTIME']
            self.julian_date[ii_science] = science_fits[0].header['JD']
            
            science_data = science_fits[0].data * self.gain # save the data from the first HDU 
            science_fits.close()

            # 1) step
            science_corrected, science_corrected_errors = self.correct_science_frame(science_data)

            #2) step
            self.x_refined[ii_science], self.y_refined[ii_science] = self.compute_centroid(science_corrected, self.x_initial, self.y_initial)
            
            #3) step
            self.sky_background[ii_science], self.sky_background_errors[ii_science] = self.compute_sky_background(science_frame=science_corrected,
                                                                                                                  science_frame_err=science_corrected_errors,
                                                                                                                  x_pos=self.x_refined[ii_science],
                                                                                                                  y_pos=self.y_refined[ii_science])

            science_sky_corrected = science_corrected- self.sky_background[ii_science]
            science_sky_corrected_errors = np.sqrt(science_corrected_errors**2. + self.sky_background_errors[ii_science]**2.)
            

            target_distance = np.sqrt((self.X-self.x_refined[ii_science])**2 + (self.Y-self.y_refined[ii_science])**2)

            aperture_selection = (target_distance < self.aperture_radius)
            self.aperture[ii_science] =  np.sum(science_sky_corrected[aperture_selection])
            self.aperture_errors[ii_science] = np.sqrt(np.sum(science_sky_corrected_errors[aperture_selection]**2.)) #da controllare

            #4) step
            #target_distance = np.sqrt((self.X-x_target_refined)**2 + (self.Y-y_target_refined)**2) già ce l'ho
            annulus_sel = (target_distance < self.sky_inner_radius)

            # We compute the sum of the total flux within the inner radius.
            total_flux = np.nansum(science_corrected*annulus_sel)

            # We compute the sum of the flux along each axis, within the inner radius.
            flux_x = np.nansum(science_corrected*annulus_sel, axis=0) 
            flux_y = np.nansum(science_corrected*annulus_sel, axis=1)

            # we compute the cumulative sum along each axis, normalized to the total flux
            cumulative_sum_x = np.cumsum(flux_x)/total_flux
            cumulative_sum_y = np.cumsum(flux_y)/total_flux

            self.x_fwhm[ii_science] = self.calculate_FWHM(self.X_axis, cumulative_sum_x)
            self.y_fwhm[ii_science] = self.calculate_FWHM(self.Y_axis, cumulative_sum_y)



