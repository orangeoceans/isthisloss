from PIL import Image, ImageOps
from random import random
from os import listdir,mkdir,path


def generate_a_comic(dims, images, color = (255)):
    comic = Image.new('L', (dims[0]*(64+30),dims[1]*(64+30)), color)
    imgidx = 0
    for i in range(dims[0]):
        for j in range(dims[1]):
            xoffset = int((20*random()))
            yoffset = int((20*random()))
            woffset = int((40*random()))
            hoffset = int((40*random()))
            this_img = images[imgidx].resize((64+woffset,64+hoffset))
            this_img = ImageOps.grayscale(this_img)
            comic.paste(this_img, (((64+30)*i)+xoffset,((64+30)*j)+yoffset))
            
            imgidx+=1
    return comic.resize((128,128))
            
def select_images(number,all_imgs):
    image_count = len(all_imgs)
    selected_imgs = []
    for n in range(number):
        imgidx = int(random()*image_count)
        selected_file = all_imgs[imgidx]
        selected_imgs.append(Image.open(directory+"/"+selected_file))
    return selected_imgs

directory = "../tiny-imagenet-200/train/n01443537/images"
all_imgs = listdir(directory)
train_comic_count = 4000
test_comic_count = 2000
valid_comic_count = 2000

for c in range(train_comic_count):
    dims = (1+int(random()*4),1+int(random()*4))
    image_count = dims[0]*dims[1]
    selected_imgs = select_images(image_count,all_imgs)
    new_comic = generate_a_comic(dims, selected_imgs)
    if not path.exists("traincomics/%d"%(image_count)):
        mkdir("traincomics/%d"%(image_count))
    new_comic.save("traincomics/%d/%d.jpg"%(image_count, c))

for c in range(test_comic_count):
    dims = (1+int(random()*4),1+int(random()*4))
    image_count = dims[0]*dims[1]
    selected_imgs = select_images(image_count,all_imgs)
    new_comic = generate_a_comic(dims, selected_imgs)
    if not path.exists("testcomics/%d"%(image_count)):
        mkdir("testcomics/%d"%(image_count))
    new_comic.save("testcomics/%d/%d.jpg"%(image_count, c))
    
for c in range(valid_comic_count):
    dims = (1+int(random()*4),1+int(random()*4))
    image_count = dims[0]*dims[1]
    selected_imgs = select_images(image_count,all_imgs)
    new_comic = generate_a_comic(dims, selected_imgs)
    if not path.exists("validcomics/%d"%(image_count)):
        mkdir("validcomics/%d"%(image_count))
    new_comic.save("validcomics/%d/%d.jpg"%(image_count, c))