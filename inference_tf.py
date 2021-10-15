import argparse
import numpy as np
from tensorflow import keras
from instance_nor import *
from losscustom import *
import nibabel as nib
from resize import *

def normalize(volume):
    """Normalize the volume"""
    min = -50
    max = 200
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume



def applicator(args):
    # Get input image and output file name.
    input_image=args.input_image
    output_filename=args.output_filename

    # Load the pretrained model
    ## The custom layer used to develop the model must be included as custom objects
    model=args.model
    model = keras.models.load_model(model, custom_objects={'InstanceNormalization':InstanceNormalization, 'tversky_loss':tversky_loss, 'dice_nogb':dice_nogb})

    # Load the data. Please keep in mind that the data must be cropped.
    # The code can be modified to allow for sliding window inference to accommodate a whole CT volume.

    image=nib.load(input_image)
    data=image.get_fdata()
    # Get the affine transform. Will be used when the final output is saved
    affine=image.get_affine()

    # Apply preprocessing transformation

    ## 1. Re orient
    ### check orientation
    print(f'Input Image Orientation {nib.aff2axcodes(image.affine)}')

    ### Convert to orientation expected. The model expects 'R','A','S'
    ornt = nib.orientations.axcodes2ornt(nib.aff2axcodes(image.affine))
    refOrnt = nib.orientations.axcodes2ornt(('R','A','S'))
    newOrnt = nib.orientations.ornt_transform(ornt,refOrnt)
    segImgData = nib.apply_orientation(data,newOrnt)

    ## Resize to model specifications
    x_original,y_original, z_original=np.shape(segImgData)
    print (f'Original Image Dimensions {x_original,y_original, z_original=}')
    segImgData=resize_volume(segImgData, desired_depth = 64, desired_width = 256, desired_height = 128)
    x_resize,y_resize, z_resize=np.shape(segImgData)
    print (f'Resized Image Dimensions {x_resize,y_resize, z_resize}')

    ## Apply window
    segImgData=normalize(segImgData)

    ## Convert to tensor
    segImgData = tf.convert_to_tensor(segImgData, np.float32)
    ## Add dimentions to allow for inference
    segImgData=tf.expand_dims(segImgData, axis=0)
    segImgData=tf.expand_dims(segImgData, axis=-1)
    segData=model.predict(segImgData)
    ### The model predicts 5 classes (includes background)
    ## Perform arg max operation
    segData=segData.argmax(axis=-1)


    # Invert all the transormations so mask can match the original input.

    ## Resize
    segData=resize_volume(segData[0,:,:,:], desired_depth =z_original , desired_width = x_original, desired_height = y_original)

    ## Reorient
    ornt = nib.orientations.axcodes2ornt(nib.aff2axcodes(image.affine))
    refOrnt = nib.orientations.axcodes2ornt(('R','A','S'))
    newOrnt = nib.orientations.ornt_transform(refOrnt,ornt)
    segData = nib.apply_orientation(segData,newOrnt)

    ## Save
    image=nib.Nifti1Image(segData.astype(np.int16), affine)
    nib.save(image, output_filename)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Apply the cor/med model to a a cropped area originating from a CT image. ')
    parser.add_argument("--input_image", type=str, default='case_00000_image_reo_crop.nii.gz')
    parser.add_argument("--output_filename", type=str, default='case_00000_image_reo_maks.nii.gz')
    parser.add_argument("--model", type=str, default='weights.h5')

    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    parser.add_argument("-q", "--quiet",
                        action="store_false", dest="verbose",
                        default=True,
                        help="don't print status messages to stdout")
    args = parser.parse_args()
    applicator(args)