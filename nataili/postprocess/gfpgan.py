from nataili.postprocessor import *


class gfpgan(PostProcessor):
    def set_filename_append(self):
        self.filename_append = "gfpgan"

    def process(self, img, img_array, **kwargs):
        strength = kwargs.get("strength", 1.0)
        _, _, output = self.model.enhance(img_array, weight=strength)
        output_array = np.array(output)
        output_image = PIL.Image.fromarray(output_array)
        return output_image
