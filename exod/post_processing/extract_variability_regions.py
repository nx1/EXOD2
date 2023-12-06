from skimage.measure import label, regionprops
import numpy as np

def extract_variability_regions(variability_map, threshold):
    variable_pixels = (variability_map > threshold*np.median(variability_map)).astype(int)
    labeled_variability_map = label(variable_pixels)

    tab_centersofmass=[]
    tab_boundingboxes=[]
    for source in range(1,np.max(labeled_variability_map)):
        source_properties = regionprops(label_image=(labeled_variability_map==source).astype(int),
                                        intensity_image=variability_map)
        tab_centersofmass.append(source_properties[0].weighted_centroid)
        tab_boundingboxes.append(source_properties[0].bbox)
    return tab_centersofmass, tab_boundingboxes

def plot_variability_with_regions(variability_map, threshold, outfile):
    fig, ax = plt.subplots()
    m1=ax.imshow(variability_map, norm=LogNorm())
    plt.colorbar(mappable=m1, ax=ax)
    centers, bboxes = extract_variability_regions(variability_map, threshold)
    for center, bbox in zip(centers, bboxes):
        min_error = 10
        width= bbox[3]-bbox[1]
        height =  bbox[2]-bbox[0]
        shiftx=0
        shifty=0
        if width<min_error:
            shiftx = min_error-width
            width = min_error
        if height < min_error:
            shifty = min_error - height
            height = min_error
        rect = patches.Rectangle((bbox[1]-1-shiftx/2, bbox[0]-1-shifty/2), width, height, linewidth=1, edgecolor='r',
                                 facecolor='none')
        ax.add_patch(rect)
    plt.savefig(outfile)


if __name__=='__main__':
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    from exod.pre_processing.read_events_files import read_EPIC_events_file
    from exod.processing.variability_computation import compute_pixel_variability
    from matplotlib.colors import LogNorm
    import os
    from exod.utils.path import data_processed

    cube = read_EPIC_events_file('0831790701', 3, 100,3, gti_only=True)
    variability_map = compute_pixel_variability(cube)
    plot_variability_with_regions(variability_map, 8,
                                  os.path.join(data_processed,'0831790701','plot_test_varregions.png'))