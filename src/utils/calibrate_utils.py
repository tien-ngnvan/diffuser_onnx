import re
import torch



def get_smoothquant_config(model, quant_level=3):
    quant_config = {
        "quant_cfg": {},
        "algorithm": "smoothquant",
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if (
            w_name in quant_config["quant_cfg"].keys()  # type: ignore
            or i_name in quant_config["quant_cfg"].keys()  # type: ignore
        ):
            continue
        if filter_func(name):
            continue
        if isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}  # type: ignore
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}  # type: ignore
        elif isinstance(module, torch.nn.Conv2d):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}  # type: ignore
            quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": None}  # type: ignore
    return quant_config

def filter_func(name):
    pattern = re.compile(
        r".*(time_emb_proj|time_embedding|conv_in|conv_out|conv_shortcut|add_embedding).*"
    )
    return pattern.match(name) is not None


def quantize_lvl(unet, quant_level=2.5):
    for name, module in unet.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
        elif isinstance(module, torch.nn.Linear):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (quant_level >= 2.5 and ("to_q" in name or "to_k" in name or "to_v" in name))
                or quant_level == 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()

# yaml_config['calib_config']['calibrate_size']
def do_calibrate(base, calibration_prompts, dynamic_input, yaml_config, **kwargs):
    if yaml_config['calib_config']['calibrate_size'] >= len(calibration_prompts):
        epoch = len(calibration_prompts)
    elif yaml_config['calib_config']['calibrate_size'] < len(calibration_prompts):
        epoch = yaml_config['calib_config']['calibrate_size']
        
    for i_th in range(int(epoch)):
        image = None
        try:
            # Check if "calibration_imgs" exists in kwargs and is not None
            calibration_imgs = kwargs.get("calibration_imgs", None)
            if calibration_imgs is not None:
                if i_th < len(calibration_imgs):
                    # Handle image
                    image = calibration_imgs[i_th]
                    for key in dynamic_input:
                        if 'image' in key:  # Check if 'image' is in the key
                            dynamic_input[key] = image
        except KeyError as e:
            raise ValueError(f"KeyError: Missing key in kwargs: {e}")
        except Exception as e:
            raise ValueError(f"Unexpected error during image handling: {e}")
            
        # Handle prompt
        prompt = calibration_prompts[i_th]
        negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]* len(prompt)
        dynamic_input['prompt'] = prompt
        dynamic_input['negative_prompt'] = negative_prompt
    
        # Handle others input
        for key, value in yaml_config['input_pipeline'].items():
            dynamic_input[key] = value


        base(
            **dynamic_input
        ).images