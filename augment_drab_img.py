import os
import random
from PIL import Image
from PIL import ImageFile
import shutil

def main():

	# Prevent errors
	Image.MAX_IMAGE_PIXELS = None
	ImageFile.LOAD_TRUNCATED_IMAGES = True
	
	check_img_count() # Checks the number of images in Sensational and Drab folder
	num_flip = num_img_to_flip() # Get the difference in images count
	flip_img_180(num_flip) # Flip X many images, where X = difference in image count
	check_img_count()

def check_img_count():
	try:
		print(str(len(os.listdir('Sensational'))) + " -- Num of Sensational Images")
		print(str(len(os.listdir('Drab'))) + " -- Num of Drab Images")
	except: 
		print("Cannot find Sensational/Drab folder!")
	
def num_img_to_flip():
	drab_len = len(os.listdir('Drab'))
	sen_len = len(os.listdir('Sensational'))
	diff = sen_len - drab_len 
	return diff
	
def flip_img_180(num_flip):
	drab_files = os.listdir('Drab')
	img_to_flip = random.sample(drab_files,k=num_flip)
	category_dir = 'Drab'
	
	i = 1
	for fname in img_to_flip:
		image = Image.open(os.path.join(category_dir, fname))
		print("" + str(i) + "/" + str(num_flip) + " Rotating " + fname + " by 180 degrees", end='\r')
		image_rot_180 = image.rotate(180)
		image_rot_180.save(os.path.join(category_dir,"180_"+fname))
		i += 1
	
	print("\nFlipping done!")

		
if __name__ == '__main__':
	main()