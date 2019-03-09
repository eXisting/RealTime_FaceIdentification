from preprocess import preprocesses
import shutil
import os

def processFeedData():
    input_datadir = './data/feed'
    output_datadir = './data/cropped'

    # remove all cropped images if they are present
    if os.path.exists(output_datadir) and os.path.isdir(output_datadir):
        shutil.rmtree(output_datadir)

    print("Parsing feed into cropped image set...")

    obj = preprocesses(input_datadir, output_datadir)
    nrof_images_total, nrof_successfully_aligned = obj.collect_data()

    print('Total number of images: %d' % nrof_images_total)
    print('Number of successfully aligned images: %d' % nrof_successfully_aligned)