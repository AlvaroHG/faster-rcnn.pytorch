
import PIL
from PIL import Image
from os import path

if __name__ == '__main__':
    dir_name = 'dpi300'
    image_name = 'black.pdf-dpi300-page0009.png'
    file_path = path.join(dir_name, image_name)
    basewidth = 800
    img = Image.open(file_path)
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), PIL.Image.ANTIALIAS)
    img.save('resized_{}'.format(image_name))