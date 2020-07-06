from time import time
t = time()
import numpy as np


from TagGenerator import TagGenerator

x = TagGenerator()
arr = x.generate('/home/rizwan/Downloads/Test Images/birds4.jpg')
# arr = x.generate("D:Wallpapers/wp5236607-volvo-truck-wallpapers.jpg")

all_names = 'Animals,Birds,Books,Buildings,Butterfly,Cats,Cows,Dogs,Elephants,Hens,Horses,Pandas,Roads,Sheep,Spiders,Sports,Squirrels,Vehicles,Water'.split(
                ',')
all_names = np.array(all_names)
sorted_indexes = np.lexsort([arr])

print(arr)
print(all_names)
print(all_names[sorted_indexes[-1:-4:-1]])
print(time()-t)