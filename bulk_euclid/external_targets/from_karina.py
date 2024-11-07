import requests
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as colors
import os

path_cat='path to the catalog'
path_fits='path to store the tile fits files '
path_cutouts='path to store the cutouts '


target_csv_name = 'targets.csv' # to save information related to targets I found a match 
mosaic_csv_name = 'mosaic.csv'  # to read the tile name and path to download it. 
#modify the following kw with the name of columns that contains ra, dec, and ID of the targets:
ra_kw = 'ra_2'
dec_kw = 'dec_2'
id_kw = 'name'

cutout_size = 100

fdf = pd.read_csv(path_cat+mosaic_csv_name)

#here select the data you would like to download NISP/VIS/Other-instrument-available
fdf=fdf[(fdf['instrument_name']=='NISP') | (fdf['instrument_name']=='VIS')]


#targets df have only one time the coordinates of the lens candidates that have any mosaic data in the euclid data base.
targets = fdf.groupby([id_kw]).size().reset_index(name='Count')

# some additional information I would like to add to the final csv file:
targets['IDp'] = 'target'
targets['fits_mosaic_VIS'] = 100
targets['fits_mosaic_NIR_H'] = 100
targets['fits_mosaic_NIR_J'] = 100
targets['fits_mosaic_NIR_Y'] = 100
targets['cutout_name_VIS'] = 100
targets['cutout_name_NIR_H'] = 100
targets['cutout_name_NIR_J'] = 100
targets['cutout_name_NIR_Y'] = 100
targets[ra_kw] = 100
targets[dec_kw] = 100

# Here the script start

login_url = 'https://easotf.esac.esa.int/sas-dd/login'


# Define the data payload for the login request
data = {
    'username': 'username',
    'password': 'password'
}

response = requests.post(login_url, data=data, allow_redirects=True, verify=False)

# Check if the login was successful
if response.status_code == 200:
    print('Login successful.')
    # Extract cookies from the response
    cookies = response.cookies
else:
    print('Login failed:', response.status_code)
    # If login failed, exit or handle the error accordingly
    exit()

    
for index in range(0,4):#len(targets)):
        # re-login after several target search to avoid losing the conection to euclid server. In this case every 2000 targets will re-connect 
    if index % 2000 == 0 and index != 0:
        print('reconecting just in case')
        
        login_url = 'https://easotf.esac.esa.int/sas-dd/login'

        data = {
                'username': 'username',
                'password': 'password'
            }

        response = requests.post(login_url, data=data, allow_redirects=True, verify=False)

        # Check if the login was successful
        if response.status_code == 200:
            print('Login successful.')
            # Extract cookies from the response
            cookies = response.cookies
        else:
            print('Login failed:', response.status_code)
            # If login failed, exit or handle the error accordingly
            exit()

        response = requests.post(login_url, data=data, allow_redirects=True, verify=False)

        # Check if the login was successful
        if response.status_code == 200:
            print('Login successful.')
            # Extract cookies from the response
            cookies = response.cookies
        else:
            print('Login failed:', response.status_code)
            # If login failed, exit or handle the error accordingly
            exit()

    temp=fdf[(fdf[id_kw]==targets[id_kw].iloc[index])]
    minsep=temp.sort_values(by='Separation')['Separation'].min()
    temp=temp[temp['Separation']==minsep]
    targets[ra_kw].iloc[index] = temp[ra_kw].iloc[0]
    targets[dec_kw].iloc[index] = temp[dec_kw].iloc[0]
    bands=temp['filter_name'].values #np.asarray(['VIS'])#
    f2d = temp['file_name'].values

    for j in range(len(bands)):

        targets['fits_mosaic_'+bands[j]].iloc[index] = f2d[j]

        file_path = os.path.join(path_fits, f2d[j])
        if not os.path.exists(file_path):
            print(f"Downloading '{f2d[j]}' as file doesn't exist.")
            # Define the file download URL    
            file_download_url = 'https://easotf.esac.esa.int/sas-dd/data?file_name='+f2d[j]+'&release=sedm&RETRIEVAL_TYPE=FILE'
            # Perform the file download request using the obtained cookies
            response = requests.get(file_download_url, cookies=cookies, allow_redirects=True, verify=False)
            # Check if the file download was successful
            if response.status_code == 200:
                print('File downloaded successfully.')
                # Save the downloaded file
                with open(path_fits+f2d[j], 'wb') as f:
                    f.write(response.content)
            else:
                print('File download failed:', response.status_code)
        else:
            print(f"File '{f2d[j]}' already exists.")

        try:
            hdul = fits.open(path_fits+f2d[j])

            # Access the data and header
            data = hdul[0].data
            header = hdul[0].header
            min_cutout_size = 70

            ra_center = targets[ra_kw].iloc[index] # Replace with your actual RA
            dec_center = targets[dec_kw].iloc[index]

            wcs = WCS(header)
            x_center, y_center = wcs.all_world2pix(ra_center, dec_center, 0)
            # Check if the central coordinates are within the image boundaries
            if (x_center >= 0 and x_center < header['NAXIS1'] and
                    y_center >= 0 and y_center < header['NAXIS2']):
                # Calculate the cutout boundaries
                x_start, x_end = int(x_center - cutout_size / 2), int(x_center + cutout_size / 2)
                y_start, y_end = int(y_center - cutout_size / 2), int(y_center + cutout_size / 2)

                # Check if the cutout size is feasible
                if (x_end - x_start >= min_cutout_size and y_end - y_start >= min_cutout_size):
                    # Make the cutout
                    cutout_data = data[y_start:y_end, x_start:x_end]
                else:
                    print("Cutout size too small. Skipping.")
                    cutout_data = np.asarray([0,0])
            else:
                print("Central coordinates outside image boundaries. Skipping.")
                cutout_data = np.asarray([0,0])
            
            if (cutout_data.mean()<=0):
                print('No signal in the cutout')
            else:
                cutout_name = 'EUC_'+str(index).zfill(4)+'_'+targets.iloc[index]['IDp']
                if not os.path.exists(path_cutouts+cutout_name):
                    os.makedirs(path_cutouts+cutout_name)#create a folder with the same name of the cutout
                fits.writeto(path_cutouts+cutout_name+'/'+cutout_name+'_'+bands[j]+'.fits', cutout_data, header, overwrite=True)
                hdul.close()
                print('Cutout saved')
            #os.system('rm '+path_fits+f2d[j])
            targets['cutout_name_'+bands[j]].iloc[index] = cutout_name
        except FileNotFoundError:
            print("File not found. Skipping this file.")
            targets['cutout_name_'+bands[j]].iloc[index] = 'None'            
            continue  # This will skip to the next iteration of the loop
        except Exception as e:
            print(f"An error occurred: {e}")
            targets['cutout_name_'+bands[j]].iloc[index] = 'None'       
            continue  # This will also skip to the next iteration of the loop
        print('updating catalog')
        targets.to_csv(path_cat+target_csv_name, index=False)

print('done!')
