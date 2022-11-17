from nataili.postprocessor import *


class realesrgan(PostProcessor):
    def set_filename_append(self):
        self.filename_append = "realesrgan"

    def process(self, img, img_array, **kwargs):
        output, _ = self.model.enhance(img_array)
        output_array = np.array(output)
        output_image = PIL.Image.fromarray(output_array)
        return output_image
