from nataili.inference.diffusers.inpainting import inpainting
from PIL import Image

original = Image.open("./inpaint_original.png")
mask = Image.open("./inpaint_mask.png")

generator = inpainting("cuda", "output_dir")
generator.generate("a mecha robot sitting on a bench", original, mask)
image = generator.images[0]["image"]
image.save("robot_sitting_on_a_bench.png", format="PNG")
