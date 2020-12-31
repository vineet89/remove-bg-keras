# A background removal app built on top of keras segmentation models

Uses PSPNet pre trained on ADE20K data set to identify specified objects, removes everything else in the scene and outputs the final image.

### How to run

Place images in sample_images folder. Run psp.py located in remove-bg/.
The program will go through all images in sample_images folder and write the output in sample_images/output.

3 images are provided for reference with their output.

Currently, the program identifies cars and removes other objects. It can be tuned to identify other objects by adjusting locator RGB values in psp.py

### TODO

- Train the model to remove bg directly and remove the intermediate step
- Identify foreground and background based on size of object
- Train the model on a bigger vehicle dataset
