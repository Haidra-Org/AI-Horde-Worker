import PIL
import numpy as np
import skimage
import torch
import k_diffusion as K
import tqdm

def process_init_mask(init_mask: PIL.Image):
    if init_mask.mode == "RGBA":
        init_mask = init_mask.convert("RGBA")
        background = PIL.Image.new("RGBA", init_mask.size, (0, 0, 0))
        init_mask = PIL.Image.alpha_composite(background, init_mask)
        init_mask = init_mask.convert("RGB")
    return init_mask

def resize_image(resize_mode, im, width, height):
    LANCZOS = PIL.Image.Resampling.LANCZOS if hasattr(PIL.Image, "Resampling") else PIL.Image.LANCZOS
    if resize_mode == "resize":
        res = im.resize((width, height), resample=LANCZOS)
    elif resize_mode == "crop":
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio > src_ratio else im.width * height // im.height
        src_h = height if ratio <= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = PIL.Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))
    else:
        ratio = width / height
        src_ratio = im.width / im.height

        src_w = width if ratio < src_ratio else im.width * height // im.height
        src_h = height if ratio >= src_ratio else im.height * width // im.width

        resized = im.resize((src_w, src_h), resample=LANCZOS)
        res = PIL.Image.new("RGBA", (width, height))
        res.paste(resized, box=(width // 2 - src_w // 2, height // 2 - src_h // 2))

        if ratio < src_ratio:
            fill_height = height // 2 - src_h // 2
            res.paste(
                resized.resize((width, fill_height), box=(0, 0, width, 0)),
                box=(0, 0),
            )
            res.paste(
                resized.resize(
                    (width, fill_height),
                    box=(0, resized.height, width, resized.height),
                ),
                box=(0, fill_height + src_h),
            )
        elif ratio > src_ratio:
            fill_width = width // 2 - src_w // 2
            res.paste(
                resized.resize((fill_width, height), box=(0, 0, 0, height)),
                box=(0, 0),
            )
            res.paste(
                resized.resize(
                    (fill_width, height),
                    box=(resized.width, 0, resized.width, height),
                ),
                box=(fill_width + src_w, 0),
            )

    return res

# helper fft routines that keep ortho normalization and auto-shift before and after fft
def _fft2(data):
    if data.ndim > 2:  # has channels
        out_fft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:, :, c]
            out_fft[:, :, c] = np.fft.fft2(np.fft.fftshift(c_data), norm="ortho")
            out_fft[:, :, c] = np.fft.ifftshift(out_fft[:, :, c])
    else:  # one channel
        out_fft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_fft[:, :] = np.fft.fft2(np.fft.fftshift(data), norm="ortho")
        out_fft[:, :] = np.fft.ifftshift(out_fft[:, :])

    return out_fft

def _ifft2(data):
    if data.ndim > 2:  # has channels
        out_ifft = np.zeros((data.shape[0], data.shape[1], data.shape[2]), dtype=np.complex128)
        for c in range(data.shape[2]):
            c_data = data[:, :, c]
            out_ifft[:, :, c] = np.fft.ifft2(np.fft.fftshift(c_data), norm="ortho")
            out_ifft[:, :, c] = np.fft.ifftshift(out_ifft[:, :, c])
    else:  # one channel
        out_ifft = np.zeros((data.shape[0], data.shape[1]), dtype=np.complex128)
        out_ifft[:, :] = np.fft.ifft2(np.fft.fftshift(data), norm="ortho")
        out_ifft[:, :] = np.fft.ifftshift(out_ifft[:, :])

    return out_ifft

def _get_gaussian_window(width, height, std=3.14, mode=0):

    window_scale_x = float(width / min(width, height))
    window_scale_y = float(height / min(width, height))

    window = np.zeros((width, height))
    x = (np.arange(width) / width * 2.0 - 1.0) * window_scale_x
    for y in range(height):
        fy = (y / height * 2.0 - 1.0) * window_scale_y
        if mode == 0:
            window[:, y] = np.exp(-(x**2 + fy**2) * std)
        else:
            window[:, y] = (1 / ((x**2 + 1.0) * (fy**2 + 1.0))) ** (
                std / 3.14
            )  # hey wait a minute that's not gaussian

    return window

def _get_masked_window_rgb(np_mask_grey, hardness=1.0):
    np_mask_rgb = np.zeros((np_mask_grey.shape[0], np_mask_grey.shape[1], 3))
    if hardness != 1.0:
        hardened = np_mask_grey[:] ** hardness
    else:
        hardened = np_mask_grey[:]
    for c in range(3):
        np_mask_rgb[:, :, c] = hardened[:]
    return np_mask_rgb

def get_matched_noise(_np_src_image, np_mask_rgb, noise_q, color_variation):
    """
    Explanation:
    Getting good results in/out-painting with stable diffusion can be challenging.
    Although there are simpler effective solutions for in-painting, out-painting can be especially challenging
    because there is no color data in the masked area to help prompt the generator.

    Ideally, even for in-painting we'd like work effectively without that data as well.
    Provided here is my take on a potential solution to this problem.

    By taking a fourier transform of the masked src img we get a function that tells us the presence and
    orientation of each feature scale in the unmasked src.
    Shaping the init/seed noise for in/outpainting to the same distribution of feature scales, orientations,
    and positions increases output coherence by helping keep features aligned.
    This technique is applicable to any continuous generation task such as audio or video, each of which can
    be conceptualized as a series of out-painting steps where the last half of the input "frame" is erased.
    For multi-channel data such as color or stereo sound the "color tone" or histogram of the seed noise
    can be matched to improve quality (using scikit-image currently)
    This method is quite robust and has the added benefit of being fast independently of the size of the
    out-painted area.
    The effects of this method include things like helping the generator integrate the pre-existing
    view distance and camera angle.

    Carefully managing color and brightness with histogram matching is also essential to achieving good coherence.

    noise_q controls the exponent in the fall-off of the distribution can be any positive number,
    lower values means higher detail (range > 0, default 1.)
    color_variation controls how much freedom is allowed for the colors/palette of the out-painted area
    (range 0..1, default 0.01)
    This code is provided as is under the Unlicense (https://unlicense.org/)
    Although you have no obligation to do so, if you found this code helpful please find it in your heart
    to credit me [parlance-zz].

    Questions or comments can be sent to parlance@fifth-harmonic.com (https://github.com/parlance-zz/)
    This code is part of a new branch of a discord bot I am working on integrating with diffusers
    (https://github.com/parlance-zz/g-diffuser-bot)

    """

    global DEBUG_MODE
    global TMP_ROOT_PATH

    width = _np_src_image.shape[0]
    height = _np_src_image.shape[1]
    num_channels = _np_src_image.shape[2]

    # FIXME: the commented lines are never used. remove?
    # np_src_image = _np_src_image[:] * (1.0 - np_mask_rgb)
    np_mask_grey = np.sum(np_mask_rgb, axis=2) / 3.0
    # np_src_grey = np.sum(np_src_image, axis=2) / 3.0
    # all_mask = np.ones((width, height), dtype=bool)
    img_mask = np_mask_grey > 1e-6
    ref_mask = np_mask_grey < 1e-3

    windowed_image = _np_src_image * (1.0 - _get_masked_window_rgb(np_mask_grey))
    windowed_image /= np.max(windowed_image)
    windowed_image += np.average(_np_src_image) * np_mask_rgb
    # / (1.-np.average(np_mask_rgb))  # rather than leave the masked area black,
    # we get better results from fft by filling the average unmasked color
    # windowed_image += np.average(_np_src_image) * (np_mask_rgb * (1.- np_mask_rgb)) /
    # (1.-np.average(np_mask_rgb)) # compensate for darkening across the mask transition area
    # _save_debug_img(windowed_image, "windowed_src_img")

    src_fft = _fft2(windowed_image)  # get feature statistics from masked src img
    src_dist = np.absolute(src_fft)
    src_phase = src_fft / src_dist
    # _save_debug_img(src_dist, "windowed_src_dist")

    noise_window = _get_gaussian_window(width, height, mode=1)  # start with simple gaussian noise
    noise_rgb = np.random.random_sample((width, height, num_channels))
    noise_grey = np.sum(noise_rgb, axis=2) / 3.0
    noise_rgb *= color_variation  # the colorfulness of the starting noise is blended to greyscale with a parameter
    for c in range(num_channels):
        noise_rgb[:, :, c] += (1.0 - color_variation) * noise_grey

    noise_fft = _fft2(noise_rgb)
    for c in range(num_channels):
        noise_fft[:, :, c] *= noise_window
    noise_rgb = np.real(_ifft2(noise_fft))
    shaped_noise_fft = _fft2(noise_rgb)
    shaped_noise_fft[:, :, :] = (
        np.absolute(shaped_noise_fft[:, :, :]) ** 2 * (src_dist**noise_q) * src_phase
    )  # perform the actual shaping

    brightness_variation = 0.0  # color_variation
    # todo: temporarily tieing brightness variation to color variation for now
    contrast_adjusted_np_src = _np_src_image[:] * (brightness_variation + 1.0) - brightness_variation * 2.0

    # scikit-image is used for histogram matching, very convenient!
    shaped_noise = np.real(_ifft2(shaped_noise_fft))
    shaped_noise -= np.min(shaped_noise)
    shaped_noise /= np.max(shaped_noise)
    shaped_noise[img_mask, :] = skimage.exposure.match_histograms(
        shaped_noise[img_mask, :] ** 1.0,
        contrast_adjusted_np_src[ref_mask, :],
        channel_axis=1,
    )
    shaped_noise = _np_src_image[:] * (1.0 - np_mask_rgb) + shaped_noise * np_mask_rgb
    # _save_debug_img(shaped_noise, "shaped_noise")

    matched_noise = np.zeros((width, height, num_channels))
    matched_noise = shaped_noise[:]
    # matched_noise[all_mask,:] = skimage.exposure.match_histograms(shaped_noise[all_mask,:],
    # _np_src_image[ref_mask,:], channel_axis=1)
    # matched_noise = _np_src_image[:] * (1. - np_mask_rgb) + matched_noise * np_mask_rgb

    # _save_debug_img(matched_noise, "matched_noise")

    """
    todo:
    color_variation doesnt have to be a single number,
    the overall color tone of the out-painted area could be param controlled
    """

    return np.clip(matched_noise, 0.0, 1.0)

def find_noise_for_image(
    model,
    device,
    init_image,
    prompt,
    steps=200,
    cond_scale=2.0,
    verbose=False,
    normalize=False,
    generation_callback=None,
):
    image = np.array(init_image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.0 * image - 1.0
    image = image.to(device)
    x = model.get_first_stage_encoding(model.encode_first_stage(image))

    uncond = model.get_learned_conditioning([""])
    cond = model.get_learned_conditioning([prompt])

    s_in = x.new_ones([x.shape[0]])
    dnw = K.external.CompVisDenoiser(model)
    sigmas = dnw.get_sigmas(steps).flip(0)

    if verbose:
        print(sigmas)

    for i in tqdm.trange(1, len(sigmas)):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigmas[i - 1] * s_in] * 2)
        cond_in = torch.cat([uncond, cond])

        c_out, c_in = [K.utils.append_dims(k, x_in.ndim) for k in dnw.get_scalings(sigma_in)]

        if i == 1:
            t = dnw.sigma_to_t(torch.cat([sigmas[i] * s_in] * 2))
        else:
            t = dnw.sigma_to_t(sigma_in)

        eps = model.apply_model(x_in * c_in, t, cond=cond_in)
        denoised_uncond, denoised_cond = (x_in + eps * c_out).chunk(2)

        denoised = denoised_uncond + (denoised_cond - denoised_uncond) * cond_scale

        if i == 1:
            d = (x - denoised) / (2 * sigmas[i])
        else:
            d = (x - denoised) / sigmas[i - 1]

        dt = sigmas[i] - sigmas[i - 1]
        x = x + d * dt

    return x / sigmas[-1]

