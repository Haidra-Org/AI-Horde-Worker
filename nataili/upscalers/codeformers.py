from nataili.postprocessor import *


class CodeFormers(PostProcessor):
    def set_filename_append(self):
        self.filename_append = "CodeFormers"

    def process(self, img: PIL.Image, img_array, **kwargs) -> PIL.Image:
        output = self.model(img)
        output_array = np.array(output)
        output_image = PIL.Image.fromarray(output_array)
        return output_image
