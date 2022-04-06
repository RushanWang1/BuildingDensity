# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import argparse
from osgeo import gdal
import csv
import numpy as np
import math
import os


def write_geoloc_image(image, output_path, geo_ref_path):
    source = gdal.Open(geo_ref_path)

    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(output_path, image.shape[2], image.shape[1], 2, gdal.GDT_Float32)
    outdata.SetGeoTransform(source.GetGeoTransform())  
    outdata.SetProjection(source.GetProjection())  
    outdata.GetRasterBand(1).WriteArray(image[0])
    outdata.GetRasterBand(2).WriteArray(image[1])
    #save to disk
    outdata.FlushCache()  
    outdata = None
    ds = None


def compute_percentage_of_built_up_area(regions_path, csv_dir, output_path):
    # Get csv paths
    #csv_paths = [os.path.join(csv_dir, elem) for elem in os.listdir(csv_dir) if elem.endswith(".csv")]
    csv_paths = [csv_dir]
    # Read image
    img_obj = gdal.Open(regions_path)
    # Get image metadata
    xmin, xpixel, _, ymax, _, ypixel = img_obj.GetGeoTransform()
    width, height = img_obj.RasterXSize, img_obj.RasterYSize
    xmax = xmin + width * xpixel
    ymin = ymax + height * ypixel
    print("ymin {} ymax {} xmin {}, xmax {}, width {}, height {}, xpixel {} ypixel {}".format(ymin, ymax, xmin, xmax,
                                                                                              width, height, xpixel,
                                                                                              ypixel))

    output = np.zeros((2,height, width)).astype(np.float32)

    for csv_path in csv_paths:
        with open(csv_path, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if i > 0:
                    latitude = float(row[0])
                    longitude = float(row[1])
                    area = float(row[2])
                    confidence = float(row[3])

                    pix_y = int(math.floor((latitude - ymax) / -abs(ypixel)))
                    pix_x = int(math.floor((longitude - xmin) / xpixel))

                    if (pix_y >= 0 and pix_y < height) and (pix_x >= 0 and pix_x < width) and (confidence>0.7):
                        #building count
                        output[0, pix_y, pix_x] += 1 
                        #add mean building area feature
                        output[1, pix_y, pix_x] = (output[1, pix_y, pix_x]+area)/(output[0, pix_y, pix_x])

    print(np.shape(output))
    write_geoloc_image(output, output_path, regions_path)

    return output


def main():
    #parser = argparse.ArgumentParser()
    #parser.add_argument("rst_wp_boundaries_path", type=str,
    #                    help="Raster of WorldPop administrative boundaries information")
    #parser.add_argument("csv_dir", type=str, help="CSV directory")
    #parser.add_argument("output_path", type=str, help="Output path (tif file)")
    #args = parser.parse_args()



    #compute_percentage_of_built_up_area(args.rst_wp_boundaries_path, args.csv_dir, args.output_path)
    compute_percentage_of_built_up_area("data/Sentinel-2_northkivuwest.tif","data/Westkivu.csv","Kivuwest.tif")


if __name__ == "__main__":
    main()
