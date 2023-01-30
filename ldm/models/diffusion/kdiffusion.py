import einops
import k_diffusion as K
import torch
import torch.nn as nn

from nataili.util.performance import performance


class KDiffusionSampler:
    def __init__(self, m, sampler, callback=None):
        self.model = m
        self.model_wrap = K.external.CompVisDenoiser(m)
        self.schedule = sampler
        self.generation_callback = callback
    def get_sampler_name(self):
        return self.schedule
    def sample(self, S, conditioning, unconditional_guidance_scale, unconditional_conditioning, x_T,
              karras=False, sigma_override: dict = None, extra_args = None
        ):
        if sigma_override:
            if 'min' not in sigma_override:
                raise ValueError("sigma_override must have a 'min' key")
            if 'max' not in sigma_override:
                raise ValueError("sigma_override must have a 'max' key")
            if 'rho' not in sigma_override:
                raise ValueError("sigma_override must have a 'rho' key")
        if extra_args is None:
            extra_args={'cond': conditioning, 'uncond': unconditional_conditioning,'cond_scale': unconditional_guidance_scale}
    
        sigma_min=self.model_wrap.sigmas[0] if sigma_override is None else sigma_override['min']
        sigma_max=self.model_wrap.sigmas[-1] if sigma_override is None else sigma_override['max']
        sigmas = None
        if karras:
            if sigma_override is None:
                if S > 8:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.0292, 14.6146, 7., self.model.device)
                elif S == 8:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.0936, 14.6146, 7., self.model.device)
                elif S <= 7 and S > 5:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.1072, 14.6146, 7., self.model.device)
                elif S <= 5:
                    sigmas = K.sampling.get_sigmas_karras(S, 0.1072, 7.0796, 9., self.model.device)
            else:
                sigmas = K.sampling.get_sigmas_karras(S, sigma_override['min'], sigma_override['max'], sigma_override['rho'], self.model.device)
        else:
            sigmas = self.model_wrap.get_sigmas(S)
        x = x_T * sigmas[0]
        if extra_args is not None:
            model_wrap_cfg = CFGDenoiser(self.model_wrap)
        else:
            model_wrap_cfg = CFGPix2PixDenoiser(self.model_wrap)
        samples_ddim = None
        if self.schedule == "dpm_fast":
            samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigma_min, sigma_max, S, extra_args=extra_args, disable=False, callback=self.generation_callback)
        elif self.schedule == "dpm_adaptive":
            samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigma_min, sigma_max, extra_args=extra_args, disable=False, callback=self.generation_callback)
        else:
            samples_ddim = K.sampling.__dict__[f'sample_{self.schedule}'](model_wrap_cfg, x, sigmas, extra_args=extra_args, disable=False, callback=self.generation_callback)
        #
        return samples_ddim, None
class CFGMaskedDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale, mask, x0, xi):
        x_in = x
        x_in = torch.cat([x_in] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        denoised = uncond + (cond - uncond) * cond_scale

        if mask is not None:
            assert x0 is not None
            img_orig = x0
            mask_inv = 1. - mask
            denoised = (img_orig * mask_inv) + (mask * denoised)

        return denoised

class CFGDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, x, sigma, uncond, cond, cond_scale):
        x_in = torch.cat([x] * 2)
        sigma_in = torch.cat([sigma] * 2)
        cond_in = torch.cat([uncond, cond])
        uncond, cond = self.inner_model(x_in, sigma_in, cond=cond_in).chunk(2)
        return uncond + (cond - uncond) * cond_scale
    
class CFGPix2PixDenoiser(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model

    def forward(self, z, sigma, cond, uncond, text_cfg_scale, image_cfg_scale):
        cfg_z = einops.repeat(z, "1 ... -> n ...", n=3)
        cfg_sigma = einops.repeat(sigma, "1 ... -> n ...", n=3)
        cfg_cond = {
            "c_crossattn": [torch.cat([cond["c_crossattn"][0], uncond["c_crossattn"][0], uncond["c_crossattn"][0]])],
            "c_concat": [torch.cat([cond["c_concat"][0], cond["c_concat"][0], uncond["c_concat"][0]])],
        }
        out_cond, out_img_cond, out_uncond = self.inner_model(cfg_z, cfg_sigma, cond=cfg_cond).chunk(3)
        return out_uncond + text_cfg_scale * (out_cond - out_img_cond) + image_cfg_scale * (out_img_cond - out_uncond)
