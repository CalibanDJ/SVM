import os
import sys
import copy
import math
import argparse
import numpy as np
#import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from skimage import io, feature, color, transform
from PIL import Image
from PIL import ImageEnhance

result_path="Results"

"""
Open the image file associated to the path given
"""
def read_image(path):
    image = io.imread(path)
    return image


def get_vanishing_points(image, lines_pts : dict):
        """
            Calculate vanishing points from a dictionary of endpoints using Bob Collin's method.
        """
        vanishing_points = {
            "x": [],
            "y": [],
            "z": []
        }
        for key in lines_pts:
            lines = []
            # Find line equation from every pair of endpoints
            for pts in lines_pts[key]:
                e1, e2 = pts
                # Homogeneous coordinates, w variable is set to 1 by default
                e1 = list(e1) + [1]
                e2 = list(e2) + [1]
                lines.append(np.cross(e1, e2))
            M = np.zeros((3, 3), dtype='float64')
            for i in range(len(lines_pts[key])):
                a, b, c = lines[i]
                M += np.array([[a * a, a * b, a * c], [a * b, b * b, b * c], [a * c, b * c, c * c]])
            # Compute vanishing points
            eig_values, eig_vectors = np.linalg.eig(M)
            vanishing = eig_vectors[:, np.argmin(eig_values)]
            vanishing = vanishing / vanishing[-1]
            vanishing_points[key] = vanishing
                
        return vanishing_points

"""
Save images with vanishing points and lines
"""
def visualize_vanishing_points(vps, image, lines_pts : dict, colors, fig_name):
    i = 0
    #Create a figure for each VP
    for key in lines_pts:
        plt.imshow(image)
        #Draw each line of VP with specific color
        for line in lines_pts[key]:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), colors[i])
        
        plt.axis('off')
        plt.tight_layout()
        
        plt.savefig("." + fig_name.split('.')[1] + "_lines_vp" + str(i) + '.png') 
        plt.plot([vps[key][0]], [vps[key][1]], colors[i]+'X', markersize=5)
        plt.savefig("." + fig_name.split('.')[1] + "_vp" + str(i) + '.png') 
        plt.close()
        i = i + 1

    #Create a figure with the 3 VP
    plt.imshow(image)
    i = 0
    for key in lines_pts:
        #Add all lines of the vp
        for line in lines_pts[key]:
            p0, p1 = line
            plt.plot((p0[0], p1[0]), (p0[1], p1[1]), colors[i])
        plt.plot([vps[key][0]], [vps[key][1]], colors[i]+'X', markersize=5)
        i = i + 1

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(fig_name) 
    plt.close() 

def storage_calibration_result(vps, projection_matrix):
    with open('./'+result_path +'/SVM.txt', "w") as f:
        f.write("-----------Vanishing Points (in pixel)-----------")
        for key in vps:
            f.write("\n")
            f.write("vp_" + key + ":\r")
            f.write("x: " + str(vps[key][0]))
            f.write("\r")
            f.write("y: " + str(vps[key][1]))
            # f.write("\n")
        f.write("\n")
        f.write("\n")
        f.write("----------- Projection matrix -----------")
        for i in range(len(projection_matrix)):
            f.write("\n| ")
            for j in range(len(projection_matrix[i])):
                f.write(str(projection_matrix[i][j]) + " ")
            f.write(" |")
        f.write("\n")    
        f.close()
    print("Finish Computation!")


def findProjectionMatrix(vx, vy, vz, o, rx, ry, rz):
        proj_matrix = np.stack(
            [vx, vy, vz, o], axis=1)
        # Now we'll find the scales associated to x, y and z
    
        ref_lengthx = np.linalg.norm(rx - o.reshape(-1, 1))
        ax, _, _, _ = np.linalg.lstsq(vx.reshape(-1, 1) - rx, rx - o.reshape(-1, 1))
        scalesx = ax[0, 0] / ref_lengthx
        
        ref_lengthy = np.linalg.norm(ry - o.reshape(-1, 1))
        ay, _, _, _ = np.linalg.lstsq(vy.reshape(-1, 1) - ry, (ry - o.reshape(-1, 1)))
        scalesy = ay[0, 0] / ref_lengthy

        ref_lengthz = np.linalg.norm(rz - o.reshape(-1, 1))
        az, _, _, _ = np.linalg.lstsq(vz.reshape(-1, 1) - rz, (rz - o.reshape(-1, 1)))
        scalesz = az[0, 0] / ref_lengthz

        # Use the scales to get the correct matrix
        proj_matrix[:, 0] = proj_matrix[:, 0] * scalesx
        proj_matrix[:, 1] = proj_matrix[:, 1] * scalesy
        proj_matrix[:, 2] = proj_matrix[:, 2] * scalesz
        return proj_matrix

"""
The idea was to found the coordinate of warped points in order to croped 
the warped texture to create the 3 object
"""

def getWarpedPoint(p, matrix):
    w = (matrix[2][0]*p[0] + matrix[2][1]*p[1] + matrix[2][2])
    px = (matrix[0][0]*p[0] + matrix[0][1]*p[1] + matrix[0][2]) / w
    py = (matrix[1][0]*p[0] + matrix[1][1]*p[1] + matrix[1][2]) / w
    return (int(px), int(py))



"""
Return the textures map associated to the orthogonal planes
"""
    
def getTextureMap(image, image_name, proj_matrix) :

    #Planar perspective Map
    homography_xy = proj_matrix[:, [0, 1, 3]].copy()
    homography_yz = proj_matrix[:, [1, 2, 3]].copy()
    homography_xz = proj_matrix[:, [0, 2, 3]].copy()
    
    #Warp image with homography
    warped_xy = transform.warp(image, homography_xy)
    warped_yz = transform.warp(image, homography_yz)
    warped_xz = transform.warp(image, homography_xz)
    
    #Save image
    io.imsave('./'+result_path+"/warped_"+image_name+"_xy.png", warped_xy)
    io.imsave('./'+result_path+"/warped_"+image_name+"yz.png", warped_yz)
    io.imsave('./'+result_path+"/warped_"+image_name+"xz.png", warped_xz)

    return True
    

"""
Main.
"""
def main(image_path, puv, px, py, pz, lines : dict, textures : dict):
    #Get image info
    img_name = os.path.basename(image_path).split('.')[0]
    image = read_image(image_path)
    
    #Compute vanishing points from lines 
    vps = get_vanishing_points(image, lines)

    #Display Lines and VP
    colors_vp = ['r', 'g', 'b']
    
    fig_name = "./"+ result_path + "/"+'{}_vanishing_point.png'\
                .format(img_name)
    visualize_vanishing_points(vps, image, lines, colors_vp, fig_name)


    proj = findProjectionMatrix(vps["x"],
                                vps["y"],
                                vps["z"],
                            np.array([puv[0], puv[1], 1]),
                            np.array([px[0], px[1], 1]).reshape(-1, 1),
                            np.array([py[0], py[1], 1]).reshape(-1, 1),
                            np.array([pz[0], pz[1], 1]).reshape(-1, 1))
    
    getTextureMap(image, img_name, proj)
    
    storage_calibration_result(vps, proj)
    return True
    

