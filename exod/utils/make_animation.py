from PIL import Image
def make_gif(list_of_images, outputfile, duration=100):
    frames = [Image.open(image) for image in list_of_images]
    frame_one = frames[0]
    frame_one.save(outputfile, format="GIF", append_images=frames,
               save_all=True, duration=duration, loop=0)