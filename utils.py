
import matplotlib.pyplot as plt

import os
from PIL import Image
from IPython.display import Image as Img
from IPython.display import display

def generate_gif(path, file_name):
    folder = os.listdir(path)
    img_list = [name for name in os.listdir(os.path.join(path,folder[0])) if "png" in name]
    img_list = sorted(img_list, key=lambda x: int(x.split('/')[-1].split('.')[0]))
    img_list = [os.path.join(path,folder[0]) + '/' + x for x in img_list]
    images = [Image.open(x) for x in img_list]
    
    im = images[0]
    im.save(f'{file_name}.gif', save_all=True, append_images=images[1:],loop=0xff, duration=25)
    # loop 반복 횟수
    # duration 프레임 전환 속도 (500 = 0.5초)
    return Img(url=f'{file_name}.gif')


def capture(screen, observation, reward, terminated, truncated, action, done,time_step, file_name):
    plt.imshow(screen)
    plt.annotate(f"cart position {observation[0]:.3f}",(350,50))
    plt.annotate(f"cart velocity {observation[1]:.3f}",(350,64))
    plt.annotate(f"pole angle {observation[2]:.3f}",(350,78))
    plt.annotate(f"pole velocity at tip {observation[3]:.3f}",(350,92))

    if action.item()==1:
        plt.title(f"{time_step} reward({reward}) terminated({terminated}) truncated({truncated})")
        plt.annotate(f"Key: ⬅   done :{done}",(300,350))
        
    if action.item()==0:
        plt.title(f"{time_step} reward({reward}) terminated({terminated}) truncated({truncated})")
        plt.annotate(f"Key: ➡   done :{done}",(300,350))
    
    plt.savefig(f"{file_name}.png")
    plt.close()
    