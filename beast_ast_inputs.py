# M31 AST input lists
# June 29, 2021
# from SMIDGE:
# beast_ast_inputs.beast_ast_inputs('13659_SMIDGE-COMBO-v1')
# Had to correct the assignment of coordinates in BEAST's make_ast_xy_list.py.
# The SMIDGE catalog used was 13659_SMIDGE-COMBO-v1 without X and Y columns
# per ~ April 15 conversation with Cliff Johnson on the Scylla Slack that this was best.
# beast_ast_inputs.beast_ast_inputs('13659_SMIDGE-COMBO-v1')

# Usually run in batch, e.g.:
# fields3 = ['data/M31-B05-EAST.phot.fits', 'data/M31-B05-WEST.phot.fits',
# 'data/M31-B07-EAST.phot.fits', 'data/M31-B07-WEST.phot.fits',
# 'data/M31-B09-EAST.phot.fits', 'data/M31-B09-WEST.phot.fits']
# ref_imgs3 = ['data/M31-B05-EAST_F475W_drz.chip1.fits', 'data/M31-B05-WEST_F475W_drz.chip1.fits',
# 'data/M31-B07-EAST_F475W_drz.chip1.fits', 'data/M31-B07-WEST_F475W_drz.chip1.fits',
# 'data/M31-B09-EAST_F475W_drz.chip1.fits', 'data/M31-B09-WEST_F475W_drz.chip1.fits']

# To assign source density bins to generated AST input lists, can do:
# For speed of reading:
# obs_cat = 'data/M31-B05-EAST.phot_with_sourceden.fits'
# cat_t = Table.read(obs_cat)
# cat_x = cat_t['X']
# cat_y = cat_t['Y']
# sd = cat_t['SD_20-22']
# catxl = list(cat_x)
# catyl = list(cat_y)
# sdl = list(sd)
# field = 'M31-B05-East'
# ast_input_file = 'M31_ast_inputs/M31-B05-EAST_inputAST.txt'
#
# beast_ast_inputs.plot_ast_check(ast_input_file, field, obs_cat=obs_cat, plot_scr_den=True, cat_t=cat_t, pix_rad=3,
# catxl=catxl, catyl=catyl, sdl=sdl, fast_loop=True, write_ast_sd=True, match_obs=False, savefig=True, src_den_str=src_den_str)

import numpy as np
import glob
import os
import sys
import types
import time

import argparse

import matplotlib.pyplot as plt
plt.ion()

from beast.tools.run import (
    create_physicsmodel,
    make_ast_inputs,
    create_filenames,
)

from beast.plotting import plot_mag_hist
from beast.tools import (
    beast_settings,
    create_background_density_map,
    split_ast_input_file,
    setup_batch_beast_fit,
)

from beast.tools.density_map import BinnedDensityMap
from beast.observationmodel.observations import Observations
from beast.physicsmodel.grid import SEDGrid
from beast.observationmodel.vega import Vega
from beast.plotting import plot_mag_hist, plot_ast_histogram

from astropy.io import fits, ascii
from astropy.wcs import WCS
from astropy.table import Table
from astropy.coordinates import Angle
from astropy import units as u


def beast_ast_inputs(field_name=None, ref_image="None",
                     peak_mags=None, plot_ast_img=False, cmap='magma',
                     vmin=None, vmax=None, results_dir='M31_ast_inputs',
                     sed_grid='M31-seds.grid.hd5', copy_sed_grid=False,
                     make_sourceden_fits_only=False, ref_filter=["F814W"],
                     flag_filter=["F814W"], mag_range=[20, 22],
                     move_files_to_dir=False, gst_with_sourceden=True):
    """
    This does all of the steps for generating AST inputs and can be used
    as a wrapper to automatically do most steps for multiple fields.
    * make field's beast_settings file
    * make source density map
    * optional: make background density map
    * split catalog by source density
    * make physics model (SED grid)
    * make input list for ASTs
    * # optional: prune input ASTs

    ----
    Inputs:

    field_name (str):  name of field (i.e., "M31-B01-WEST")
    ref_image (str):   path to reference image (i.e., "M31-B01-WEST_F475W_drz.chip1.fits")
    filter_ids (list): list of indexes corresponding to the filters in the
                       observation, referenced to the master list below.
    galaxy (str):      name of target galaxy (e.g., 'M31')
    ref_filter (str):  The reference filter whose range in magnitudes will be used to
                       generate the range of input ASTs. Used by BEAST's make_source_dens_map
                       to designate a filter based on whose mags the src dens will be determined
    flag_filter (str): Used by beast.tools.cut_catalogs to select flagged sources in a filter
                       according to which to cut the catalog.
    mag_range (tuple):

    PHAT specific variables added by Petia to deal with big files and file structure
    sed_grid (str):    the model grid can be provided directly if it has already been generated
                       for speed of executing code if code is run miltiple times. Only used if
                       "copy_sed_grid" is used.
    copy_sed_grid (bool) PHAT specific; used to copy the SED grid into each individual half-brick dir
    make_sourceden_fits_only (bool): Only generate fits files w/ src density, and exit.
    move_files_to_dir (bool): Specific to PHAT file structure to organize files.
    gst_with_sourceden (bool): If reading in from a 'phot_with_sourceden.fits' photometry file,
                        copy this temporarily into a phot.fits; also copy existing into the
                        last magrange this was created with.
    ----

    Places for user to manually do things:
    * editing code before use
        - here: list the catalog filter names with the corresponding BEAST names
        - here: choose settings (pixel size, filter, mag range) for the source density map
        - here: choose settings (pixel size, reference image) for the background map

    """

    # Make a folder for the brick files
    if not os.path.isdir(field_name):
        os.mkdir(field_name)

    # PHAT-specific
    # Move existing files into F475 dir
    # only used to move all half brick files
    if move_files_to_dir:
        move_files_to_dir = "./{0}/done_with_f814w_ref_filter_magrange_20_22/".format(field_name)
        print("Moving existing files to:", move_files_to_dir)

        if not os.path.isdir(move_files_to_dir):
            os.mkdir(move_files_to_dir)
        os.system("mv ./{0}/M31*_input* ".format(field_name) + move_files_to_dir)
        os.system("mv ./{0}/M31*_img* ".format(field_name) + move_files_to_dir)
        os.system("mv ./{0}/M31*_AST* ".format(field_name) + move_files_to_dir)

    # PHAT-specific
    # Temporarily copy the SED model grid into the new directory, with a new name
    if copy_sed_grid:
        os.system('cp ' + sed_grid + ' ./{0}/{0}_seds.grid.hd5 '.format(field_name))

    # PHAT-specific
    # if reading in from a 'phot_with_sourceden.fits' photometry file,
    # copy this temporarily into a phot.fits; also copy existing into
    # the last magrange this was created with.
    if gst_with_sourceden:
        filtstr = ref_filter[0]
        magrstr = "_magrange_" + str(mag_range[0]) + "_" + str(mag_range[1])

        # Take care of previous source densities + remove SourceDensity column for clean phot.fits
        gst_sd_in = "./data/{0}.phot_with_sourceden.fits".format(field_name)
        t = Table.read(gst_sd_in)
        cols = t.colnames

        if 'SOURCEDENSITY' or 'SourceDensity' in cols:
            print("Changing Source Density column names...")
            # This is from magrange = [20, 21.5]; first limit suggested
            if 'SOURCEDENSITY' in cols:
                t['SOURCEDENSITY'].name = 'SD_20-21.5'
            # This is from magrange = [20, 22]; second limit suggested due to no structure
            if 'SourceDensity' in cols:
                t['SourceDensity'].name = 'SD_20-22'

            hdu = fits.BinTableHDU(data=t)
            hdu.writeto(gst_sd_in, overwrite=True)

        os.system('cp ' + gst_sd_in + ' ./data/{0}.phot.fits '.format(field_name))

    # the list of fields
    field_names = [field_name]

    # choose a filter to use for removing artifacts
    # (remove catalog sources with filter_FLAG > 99)
    # This is the flag_filter

    # number of fields
    n_field = len(field_names)

    # Need to know what the correspondence is between filter names in the
    # catalog and the BEAST filter names.
    #
    # These will be used to automatically determine the filters present in
    # each GST file and fill in the beast settings file.  The order doesn't
    # matter, as long as the order in one list matches the order in the other
    # list.
    #
    gst_filter_names = [
        "F475W",
        "F814W",
        "F275W",
        "F336W",
        "F110W",
        "F160W",
    ]
    beast_filter_names = [
        "HST_ACS_WFC_F475W",
        "HST_ACS_WFC_F814W",
        "HST_WFC3_F275W",
        "HST_WFC3_F336W",
        "HST_WFC3_F110W",
        "HST_WFC3_F160W",
    ]

    # filter_ids = [int(i) for i in filter_ids]
    #
    # gst_filter_names = [gst_filter_names[i] for i in filter_ids]
    # beast_filter_names = [beast_filter_names[i] for i in filter_ids]

    # for b in range(n_field):
    for b in range(1):

        print("********")
        print("field " + field_names[b])
        print("********")

        # -----------------
        # data file names
        # -----------------

        # paths for the data/AST files

        gst_file = "./data/" + field_names[b] + ".phot.fits"
        # ast_file = "./data/" + field_names[b] + ".st.fake.fits"
        print('gst file:', gst_file)

        # -----------------
        # 0. make beast settings file
        # -----------------

        # check if a settings file exists:
        if not os.path.isfile("settings/beast_settings_asts_" + field_names[b] + ".txt"):
            print("")
            print("creating beast settings file for", field_names[b])
            print("")

            create_beast_settings(
                gst_file,
                ref_image
            )

            # move to settings dir
            os.system("mv beast_settings_asts_" + field_names[b] + ".txt settings/")

        # load in beast settings to get number of subgrids
        settings = beast_settings.beast_settings(
            "settings/beast_settings_asts_" + field_names[b] + ".txt"
        )

        # -----------------
        # 1a. make magnitude histograms
        # -----------------

        # check if peak_magnitudes in each filter are supplied as inputs to avoid re-generating mag hists
        if peak_mags is None:
            print("")
            print("making magnitude histograms")
            print("")

            peak_mags = plot_mag_hist.plot_mag_hist(gst_file, stars_per_bin=70, max_bins=75)

        print(peak_mags)

        # The line below returning peak mags can be uncommented if one is testing the code
        # and does not want to take time to generate histograms with peak magnitudes every time
        # return peak_mags

        # -----------------
        # 1b. make a source density map
        # -----------------

        if not os.path.isfile(gst_file.replace(".fits", "_sourceden_map.hd5")):
            print("")
            print("making source density map")
            print("")

            use_bg_info = False
            if use_bg_info:
                background_args = types.SimpleNamespace(
                    subcommand="background",
                    catfile=gst_file,
                    pixsize=5,
                    npix=None,
                    reference=ref_image,
                    mask_radius=10,
                    ann_width=20,
                    cat_filter=[ref_filter, "90"],
                )
                create_background_density_map.main_make_map(background_args)

        if mag_range is None:
            mag_range = [17, peak_mags[ref_filter[0]] - 0.5]
        print('mag_range check:', mag_range)

        # but we are doing source density bins!
        if not os.path.isfile(gst_file.replace(".fits", "_source_den_image.fits")):
            # - pixel size of 10 arcsec
            # - use ref_filter[b] between vega mags of 17 and peak_mags[ref_filter[b]]-0.5
            sourceden_args = types.SimpleNamespace(
                subcommand="sourceden",
                catfile=gst_file,
                erode_boundary=settings.ast_erode_selection_region,
                pixsize=5,
                npix=None,
                mag_name=ref_filter[0] + "_VEGA",
                mag_cut=mag_range,
                flag_name=flag_filter[0] + "_FLAG",
            )
            create_background_density_map.main_make_map(sourceden_args)

        # new file name with the source density column
        gst_file_sd = gst_file.replace(".fits", "_with_sourceden.fits")

        # PHAT-specific
        # If the rest of the code is not needed, only generate fits files w/ src density, and exit.
        if make_sourceden_fits_only:
            print('sourceden.fits is made. Exiting.')
            return 1
        print("Done with source density work.")

        # ------------------------------
        # 2. make or fetch physics model
        # ------------------------------

        # see which subgrid files already exist
        gs_str = ""
        if settings.n_subgrid > 1:
            gs_str = "sub*"

        # try to fetch the list of SED files (physics models)
        model_grid_files = sorted(
            glob.glob(
                "./{0}/{0}_seds.grid*.hd5".format(
                    field_names[b],
                )
            )
        )
        print('SED grid file(s):', model_grid_files)

        # only make the physics model if they don't already exist
        if len(model_grid_files) < settings.n_subgrid:
            print("")
            print("making physics model")
            print("")
            # directly create physics model grids
            create_physicsmodel.create_physicsmodel(
                settings, nprocs=1, nsubs=settings.n_subgrid
            )

        # fetch the list of SED files again (physics models)
        model_grid_files = sorted(
            glob.glob(
                "./{0}/{0}_seds.grid*.hd5".format(
                    field_names[b],
                )
            )
        )

        # -------------------
        # 3. make AST inputs
        # -------------------

        # only create an AST input list if the ASTs don't already exist
        ast_input_file = "./" + field_names[b] + "/" + field_names[b] + "_inputAST.txt"
        print('\nast_input_file is/will be', ast_input_file, "\n")

        # Generating the AST inputs
        if not os.path.isfile(ast_input_file):
            make_ast_inputs.make_ast_inputs(settings, pick_method="flux_bin_method")

        # ------------------------
        # 4. Make diagnostic plots
        # ------------------------

        # compare magnitude histograms of ASTs with catalog
        plot_ast_histogram.plot_ast_histogram(
            ast_file=ast_input_file, sed_grid_file=model_grid_files[0]
        )

        # plot AST input list onto reference image
        if plot_ast_img:
            plot_ast_check(ast_input_file, field_names[b], ref_image, cmap, vmin, vmax)

        # move three files - AST list & two images - to combined results directory
        os.system('cp ' + ast_input_file + ' ' + results_dir)
        if plot_ast_img:
            os.system('cp ./{0}/{0}_img_asts.png '.format(field_name, field_name) + results_dir)
            os.system('cp ./{0}/{0}_img.png '.format(field_name, field_name) + results_dir)
        os.system('cp ./{0}/{0}_inputAST.png '.format(field_name, field_name) + results_dir)

        # PHAT-specific
        # remove temporary photometry and SED grid file
        if gst_with_sourceden:
            os.system('rm ./data/{0}.phot.fits '.format(field_names[b]))
        os.system('rm ./{0}/{0}_seds.grid.hd5'.format(field_name, field_name))

        print("now go check the diagnostic plots!")
        return peak_mags


def input_ast_bin_stats(settings, ast_input_file, field_name):

    # Load input ast file
    ast_input = Table.read(ast_input_file, format="ascii")

    # Set reference and source density images
    reference_image = settings.ast_reference_image
    source_density_image = settings.obsfile.replace(".fits", "_source_den_image.fits")

    # Check stats
    map_file = settings.ast_density_table # '...sourceden_map.hd5'
    bdm = BinnedDensityMap.create(
        map_file,
        bin_mode=settings.sd_binmode,
        N_bins=settings.sd_Nbins,
        bin_width=settings.sd_binwidth,
        custom_bins=settings.sd_custom,
    )

    # Add RA and Dec information to the input AST file (which is just an ascii filewith only X,Y positions)
    hdu_ref = fits.open(reference_image)
    wcs_ref = WCS(hdu_ref[0].header)
    source_astin = wcs_ref.wcs_pix2world(ast_input["X"], ast_input["Y"], 0)

    # Compute source coordinates in SD image frame
    hdu_sd = fits.open(source_density_image)
    wcs_sd = WCS(hdu_sd[0].header)
    source_sdin = wcs_sd.wcs_world2pix(source_astin[0], source_astin[1], 0)

    # Import filter information from the BEAST settings file
    filters = settings.filters.copy()
    # Count number of filters and decide how many rows to plot
    ncmds = len(filters)
    nrows = int(ncmds / 2) + 1

    # Figure out what the bins are
    bin_foreach_source = np.zeros(len(ast_input), dtype=int)
    for i in range(len(ast_input)):
        bin_foreach_source[i] = bdm.bin_for_position(
            source_astin[0][i], source_astin[1][i]
        )
    # compute the AST input indices for each bin
    binnrs = np.unique(bin_foreach_source)
    bin_idxs = []
    for b in binnrs:
        sources_for_bin = bin_foreach_source == b
        bin_idxs.append([sources_for_bin])

    for k in range(len(binnrs)):
        cat = ast_input[bin_idxs[k]]
        print(binnrs[k], np.shape(cat["zeros"]))


def create_beast_settings(
    gst_file,
    ref_image="None",
):
    """
    Create a beast_settings file for the given field.  This will open the file to
    determine the filters present - the `*_filter_label` inputs are references
    to properly interpret the file's information.

    Parameters
    ----------
    gst_file : string
        the path+name of the GST file

    ast_file : string
        the path+name of the AST file

    gst_filter_label : list of strings
        Labels used to represent each filter in the photometry catalog

    beast_filter_label : list of strings
        The corresponding full labels used by the BEAST

    ref_image : string (default='None')
        path+name of image to use as reference for ASTs

    Returns
    -------
    nothing

    """

    # read in the catalog
    cat = Table.read(gst_file)
    # extract field name
    field_name = gst_file.split("/")[-1].split(".")[0]

    # get the list of filters
    # filter_list_base = []
    # filter_list_long = []
    # for f in range(len(gst_filter_label)):
    #     filt_exist = [gst_filter_label[f] in c for c in cat.colnames]
    #     if np.sum(filt_exist) > 0:
    #         filter_list_base.append(gst_filter_label[f])
    #         filter_list_long.append(beast_filter_label[f])

    # read in the template settings file
    template_file = "beast_settings_asts_M31_template.txt"

    orig_file = open(template_file, "r")
    settings_lines = np.array(orig_file.readlines())
    orig_file.close()

    # write out an edited beast_settings
    new_file = open("beast_settings_asts_" + field_name + ".txt", "w")

    for i in range(len(settings_lines)):

        # replace project name with the field ID
        if settings_lines[i][0:10] == "project = ":
            new_file.write('project = "' + field_name + '" \n')
        # obsfile
        elif settings_lines[i][0:10] == "obsfile = ":
            new_file.write('obsfile = "' + gst_file + '"\n')
        # AST file name
        # elif settings_lines[i][0:10] == "astfile = ":
        #     new_file.write('astfile = "' + ast_file + '"\n')
        # BEAST filter names
        # elif settings_lines[i][0:10] == "filters = ":
        #     new_file.write("filters = ['" + "','".join(filter_list_long) + "'] \n")
        # catalog filter names
        # elif settings_lines[i][0:14] == "basefilters = ":
        #     new_file.write("basefilters = ['" + "','".join(filter_list_base) + "'] \n")
        # AST stuff
        elif settings_lines[i][0:20] == "ast_density_table = ":
            new_file.write(
                'ast_density_table = "'
                + gst_file.replace(".fits", "_sourceden_map.hd5")
                + '" \n'
            )
        elif settings_lines[i][0:22] == "ast_reference_image = ":
            new_file.write('ast_reference_image = "' + ref_image + '" \n')
        # none of those -> write line as-is
        else:
            new_file.write(settings_lines[i])

    print('Generate new settings file:', new_file)
    new_file.close()


def plot_ast_check(ast_input_file,
                   field_name,
                   ref_img=None,
                   obs_cat=None,
                   plot_scr_den=False,
                   scatter=True,
                   sdcheck=False,
                   match_obs=False,
                   cat_t=None,
                   xfl=None,
                   yfl=None,
                   catxl=None,
                   catyl=None,
                   sdl=None,
                   pix_rad=2,
                   cmap='magma',
                   vmin=None,
                   vmax=None,
                   savefig=False,
                   write_ast_sd=False,
                   return_sd=False,
                   fast_loop=False,
                   obs_cb=False,
                   quick_rematch=True,
                   src_den_str='SD_20-22',
                   memory_intensive=True):
    """
    Added by Petia to check AST input lists results against observations catalog.

    Parameters
    ----------
    ast_input_file
    field_name
    ref_img
    obs_cat
    cmap
    vmin
    vmax

    Returns
    -------

    """

    # Plot ASTs onto observed catalog

    start_code = time.time()
    if plot_scr_den:
        if cat_t is None:
            cat_t = Table.read(obs_cat)
        cat_x = cat_t['X']
        cat_y = cat_t['Y']
        sd = cat_t[src_den_str]

        if match_obs:
            plt.figure(figsize=(10, 8))

            if scatter:
                cb = plt.scatter(cat_x, cat_y, c=sd, cmap='inferno')
                if obs_cb:
                    cbar = plt.colorbar(cb)
                    cbar.ax.tick_params(labelsize=14)
                    cbar.set_label(label='SD obs', size=16)
            else:
                plt.plot(cat_x, cat_y, ',', ls='')
            end_plot_obs = time.time()
            print("Plotting obs took", (end_plot_obs-start_code)/60., "min")
    # Plot ASTs onto image
    else:
        if ref_img is None or ref_img is "None":
            ref_img = glob.glob("./data/{0}_F475W_drz.chip1.fits".format(field_name))

            im = fits.open(ref_img)
            im = im[0].data

        plt.imshow(im, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.savefig("./{0}/{0}_img.png".format(field_name, field_name))

    if sdcheck is False:
        asts = Table.read(ast_input_file, format='ascii')
        print("Matching %s ASTs" % len(asts))

        # Find a good periodic recording point based on the number of ASTs
        if len(asts) > 10000.:
            rec_interval = int(np.around(len(asts) / 500., decimals=5))
        elif len(asts) > 1000. and len(asts) <= 10000:
            rec_interval = int(np.around(len(asts) / 50., decimals=5))
        elif len(asts) <= 1000:
            rec_interval = int(np.around(len(asts) / 5., decimals=5))

        print("Recording interval is", rec_interval)

        check_asts_in = time.time()
        if xfl is None:
            f = open(ast_input_file, 'r')
            rows = f.readlines()[1:]
            xfl = []
            yfl = []
            for i in rows:
                ast_x = float(i.split(' ')[2])
                ast_y = float(i.split(' ')[3])
                xfl.append(ast_x)
                yfl.append(ast_y)
                f.close()
        done_asts_in_check = time.time()
        print("ASTs read-in took", (done_asts_in_check - check_asts_in) / 60., "min")

        # All these take awhile
        if catxl is None:
            catxl = list(cat_x)
        if catyl is None:
            catyl = list(cat_y)
        if sdl is None:
            sdl = list(sd)

        beg_loop = time.time()
        combo_inds = []
        # Try a faster loop
        if fast_loop:
            print("Faster loop through obs and asts...")

            obs = [catxl, catyl, sdl]
            obsa = np.array(obs)

            obsx = np.array(catxl)
            obsy = np.array(catyl)
            done_arr_conv = time.time()
            print("Converting into arrays took", (done_arr_conv-beg_loop)/60., "min")

            # Latest
            astx = np.array(xfl)
            asty = np.array(yfl)
            xmatch = []
            ymatch = []

            # Get all X/Y sources +/- pixel radius
            # This bit takes awhile, and is memory-intensive. :. need to write periodically to disk.
            ast_range_beg = 0
            ast_range_end = 0

            for i in range(astx.shape[0]):
                # the length of x/y-thismatch is len(obs)
                xthismatch = ((astx[i] - pix_rad) <= obsx) & (obsx <= (astx[i] + pix_rad))
                ythismatch = ((asty[i] - pix_rad) <= obsy) & (obsy <= (asty[i] + pix_rad))

                xmatch.append(xthismatch)
                ymatch.append(ythismatch)

                if memory_intensive:
                    ast_range_end += 1
                    # print("ast range beg and end", ast_range_beg, ast_range_end)

                    # Perdiodically record matched ASTs for memory purposes
                    if (i % rec_interval == 0 and i > 0) \
                            or i == astx.shape[0] - 1 \
                            or i > astx.shape[0] - rec_interval:
                        if (i % rec_interval) == 0:
                            print("\ni =", i)

                        # print("Recording batch matched SDs, again; i =", i)
                        # provide only the remaining unexplored range of astx/y
                        astx_this = astx[ast_range_beg:]
                        asty_this = asty[ast_range_beg:]
                        # print("astx provided", astx_this)
                        combo_inds = match_ast_to_obs(combo_inds, xmatch, ymatch, obsa, astx_this, asty_this)

                        # print("Length of combo_inds:", len(combo_inds))
                        # print("Combo inds", combo_inds)

                        # Reset the arrays and start the next for loop iteration
                        xmatch = []
                        ymatch = []
                        ast_range_beg = ast_range_end
                        continue

                else:
                    xmatcha = np.array(xmatch)
                    ymatcha = np.array(ymatch)
                    obsa_t = np.transpose(obsa)

                    print("%s matched sources" % xmatcha.shape[0])

                    # Select observations within matched ASTs X/Y ranges
                    for i in range(xmatcha.shape[0]):
                        matchedobs = obsa_t[np.where((xmatcha[i] == True) & (ymatcha[i] == True))[0]]
                        if len(matchedobs) > 0:
                            # Attribute one out of a number of observations to one AST
                            # Find the minimum difference between X/Y-obs and X/Y-ast
                            if len(matchedobs) > 1:
                                diff = [20, 20]
                                take_match = 0
                                for j in range(len(matchedobs)):
                                    diffnow = [np.abs(matchedobs[j, 0] - astx[i]),
                                               np.abs(matchedobs[j, 1] - asty[i])]
                                    if (diffnow[0] < diff[0]) & (diffnow[1] < diff[1]):
                                        diff = diffnow
                                        take_match = j
                                combo_inds.append([astx[i], asty[i], matchedobs[take_match, 2]])
                            else:
                                combo_inds.append([astx[i], asty[i], matchedobs[0, 2]])

            end_loop = time.time()
            print("The fast loop took %s for %s ASTs" % ((end_loop-beg_loop)/60., len(astx)))

        # Else do slow loop
        else:
            print("Start to loop through obs and asts...")
            time_printed = 0
            for j in range(len(xfl)):
                for i in range(len(catxl)):
                    if (catxl[i] >= (xfl[j] - pix_rad)) & (catxl[i] <= (xfl[j] + pix_rad)) & \
                       (catyl[i] >= (yfl[j] - pix_rad)) & (catyl[i] <= (yfl[j] + pix_rad)):
                        combo_inds.append([xfl[j], yfl[j], i])
                        med_loop = time.time()
                        # if ((med_loop - beg_loop)/60. > 1) & time_printed < ...:
                        #     print("Time so far:", (med_loop - beg_loop)/60.)
                        #     time_printed += 1
            end_loop = time.time()
            print("Loop took %s min" % np.around((end_loop-beg_loop)/60., decimals=2))

        ci_arr = np.array(combo_inds)

        # Remove duplicate AST entries; Only needed with slow loop approach
        # Removes any occurrences beyond the first, so may not be entirely correct
        if fast_loop is False:
            for i in range(ci_arr.shape[1]):
                _, return_index = np.unique(ci_arr[:, i], return_index=True)
                ci_arr = ci_arr[return_index]

        # Print checks
        print("Input ASTs (%s); Matched Obs to ASTs(%s)" %
              (len(xfl), len(combo_inds)))

        # Check if there are enough matches. If not, find new ones.
        if len(ci_arr) < len(xfl):
            # find how many more are needed
            extra_matches_needed = len(xfl) - len(ci_arr)
            print("NOT ENOUGH MATCHES, Need %s more" % extra_matches_needed)

            # print("ci_arr shape", np.shape(ci_arr))
            # quick match fill by random copy of existing matched observations
            if quick_rematch:
                # pick a random ci_arr entry to replicate
                fake_list = np.linspace(0, np.shape(ci_arr)[0]-1, np.shape(ci_arr)[0])
                extra_matches = np.random.choice(fake_list, extra_matches_needed, replace=False)

                print("extra_matches", extra_matches)

                combo_inds_arr = np.array(combo_inds)
                for k in range(len(extra_matches)):
                    combo_inds.append(combo_inds_arr[int(extra_matches[k])])

                # extra_matches_int = np.array([int(x) for x in extra_matches])
                # print("type extra_matches_int", type(extra_matches_int[0]))
                # print("extra matches", extra_matches_int)
                #
                # combo_inds_arr = np.array(combo_inds)
                # print("combo_inds_arr element 5", combo_inds_arr[5])
                # extras = [combo_inds_arr[x] for x in extra_matches]
                # print("extras", extras)
                # print("combo_inds_arr[extra_matches]", combo_inds_arr[extra_matches])
                # combo_inds.append(combo_inds_arr[extra_matches])
                # print("These are the new matches", combo_inds)
            # print("previous length of ci_arr", len(ci_arr))
            ci_arr = np.array(combo_inds)
            # print("new length of ci_arr", len(ci_arr))

        if match_obs is False:
            # print("opening figure")
            plt.figure(figsize=(10, 8))
        if scatter:
            # print("scatter plot")
            sd_inds = ci_arr[:, 2].astype(int)
            # cb2 = plt.scatter(ci_arr[:, 0], ci_arr[:, 1], c=sd_arr[sd_inds], cmap='jet')
            cb2 = plt.scatter(ci_arr[:, 0], ci_arr[:, 1], c=ci_arr[:, 2], cmap='jet')
            plt.xlabel('X', fontsize=16)
            plt.ylabel('Y', fontsize=16)
            plt.title('%s' % field_name, fontsize=16)
            cbar2 = plt.colorbar(cb2)
            cbar2.ax.tick_params(labelsize=14)
            cbar2.set_label(label='SD ASTs', size=16)
        else:
            plt.plot(xfl, yfl, '.', markersize=2, ls='', c='cyan', label='$N_{AST}=%s$' % len(xfl))
            plt.legend()

    if savefig:
        print("saving figure")
        if plot_scr_den:
            plt.savefig("./{0}/{0}_srcden_asts.png".format(field_name, field_name))
        else:
            plt.savefig("./{0}/{0}_img_asts.png".format(field_name, field_name))
        plt.close()


    # Record a table with AST x/y values and source density
    if write_ast_sd:
        sd = np.around(ci_arr[:, 2], decimals=3)
        # First make a copy of the ASTs input list file
        ast_input_file_w_sd = ast_input_file.replace(".txt", "_sd.txt")
        os.system("cp " + ast_input_file + " " + ast_input_file_w_sd)
        if 'SrcDen' in asts.colnames:
            print("SrcDen column already exists. Will not write to file.")
        else:
            print("Writing SD column to ASTs input list")
            asts_t = Table.read(ast_input_file_w_sd, format='ascii')
            asts_t.add_column(sd, name='SrcDen', index=10)
            ascii.write(asts_t, ast_input_file_w_sd, overwrite=True)

    if return_sd:
        sd = np.around(ci_arr[:, 2], decimals=3)
        return sd, ci_arr


def match_ast_to_obs(combo_inds, xmatch, ymatch, obsa, astx, asty):
    """
    Goes with plot check and records the assigned AST SD periodically
    """

    xmatcha = np.array(xmatch)
    ymatcha = np.array(ymatch)
    obsa_t = np.transpose(obsa)

    # print("%s matched sources" % xmatcha.shape[0])

    # Select observations within matched ASTs X/Y ranges
    for i in range(xmatcha.shape[0]):
        matchedobs = obsa_t[np.where((xmatcha[i] == True) & (ymatcha[i] == True))[0]]
        if len(matchedobs) > 0:
            # print("matched observation for AST #%s = %s" % (i, matchedobs))
            # Attribute one out of a number of observations to one AST
            # Find the minimum difference between X/Y-obs and X/Y-ast
            if len(matchedobs) > 1:
                diff = [20, 20]
                take_match = 0
                for j in range(len(matchedobs)):
                    diffnow = [np.abs(matchedobs[j, 0] - astx[i]),
                               np.abs(matchedobs[j, 1] - asty[i])]
                    if (diffnow[0] < diff[0]) & (diffnow[1] < diff[1]):
                        diff = diffnow
                        take_match = j
                combo_inds.append([astx[i], asty[i], matchedobs[take_match, 2]])
                # print("astx[i], asty[i], matchedobs[take_match, 2] = ", astx[i], asty[i], matchedobs[take_match, 2])
                # print("Check on obs: obsx[i], obsy[i], matchedobs[take_match, 2] = ", matchedobs[take_match, 0], matchedobs[take_match, 1], matchedobs[take_match, 2])
                # return astx[i], asty[i], matchedobs[take_match, 2]
            else:
                combo_inds.append([astx[i], asty[i], matchedobs[0, 2]])
                # return astx[i], asty[i], matchedobs[0, 2]

    return combo_inds



if __name__ == "__main__":
    # commandline parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "field_name",
        type=str,
        help="name of target field",
    )
    parser.add_argument(
        "--ref_image",
        type=str,
        default=None,
        help="path to reference image",
    )
    parser.add_argument(
        "--filter_ids",
        type=list,
        default=None,
        help="indexes of filters",
    )
    parser.add_argument(
        "--galaxy",
        type=str,
        default=None,
        help="target galaxy",
    )

    args = parser.parse_args()

    beast_ast_inputs(
        field_name=args.field_name,
        ref_image=args.ref_image,
        filter_ids=args.filter_ids,
        galaxy=args.galaxy,
    )
