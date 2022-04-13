import argparse
import os 
parser = argparse.ArgumentParser()
parser.add_argument("--AnnotatedPath", type=str)
parser.add_argument("--RootDir",type = str)
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('~/processedAnnotation/'),
                    help="Training status")
parser.add_argument('--imageType', type = str, default = '.png')
parser.add_argument('--labelType',type = str,default ='.txt')
parser.add_argument('--installReq',type = bool,default = False)

opt = parser.parse_args()

txtFile = []
imgFile = []
deleteImg = []
folderpath = "/content/gdrive/Shareddrives/Cytosim/for_machine_learning/river"
path = opt.AnnotatedPath

# Export Locations 
labels = opt.expPath + 'labels'
images = opt.expPath + 'images'
text_files = [f for f in os.listdir(path) if f.endswith(opt.labelType)]
img_files = [f for f in os.listdir(path) if f.endswith(opt.imageType)]
#Files to keep 
keep_files = []

for i in range(len(text_files)):
    text_files[i] = (text_files[i].split('.'))[0]
for i in range(len(img_files)):
    img_files[i] = (img_files[i].split('.'))[0]

for element in text_files:
    if element in img_files:
        newPathFile = str(path) + '/' + path + "_" + str(element) + opt.imageType
        oldPathFile = str(path) + '/' + str(element) + opt.imageType
        keep_files.append(str(path) + '/' + path + "_" + str(element) + opt.imageType)

delete_files = [f for f in img_files if f not in keep_files]
for i in range(len(delete_files)):
    delete_files[i] = path + delete_files[i] + opt.imageType

 