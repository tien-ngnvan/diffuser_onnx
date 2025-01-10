import yaml
import inspect
import re
import os
from diffusers.utils import load_image
import torch
from pathlib import Path
import onnx
import torch



""" MY UTILS """

# def extract_input_pipeline(sd_pipeline):
#     inputs = inspect.signature(sd_pipeline.__call__)
#     filtered_params = [
#         param for param in inputs.parameters.values()
#         if ("Optional" not in str(param.annotation)) and (
#             param.name == "prompt" or ("image" in param.name and "ip_adapter_image" not in param.name)
#         )
        
#     ]
#     filtered_param_names = [param.name for param in filtered_params]
#     return filtered_param_names


# def check_pipeline_input_name(dict_base, dict_sample):
#     # Find elements from dict_base that are missing in dict_sample
#     missing_elements = {
#         key: value for key, value in dict_base.items()
#         if key not in dict_sample or dict_sample[key] != value
#     }
#     return missing_elements


def extract_and_validate_pipeline(sd_pipeline, yaml_input_pipeline):
    inputs = inspect.signature(sd_pipeline.__call__)
    dict_sample = {}
    missing_keys = set(yaml_input_pipeline.keys())  
    for param in inputs.parameters.values():
        if param.name in missing_keys:
            missing_keys.remove(param.name)  
        if ("Optional" not in str(param.annotation)) and (
            param.name == "prompt" or ("image" in param.name and "ip_adapter_image" not in param.name)
        ):
          dict_sample[param.name] = param.annotation

    return dict_sample, missing_keys, inputs


def check_image_in_input(list_input_sd, yaml_config_input_pipeline):
    sd_image = False
    cnet_image = False
    for sd_input in list_input_sd:
        if sd_input == "image": 
            sd_image = True
            if yaml_config_input_pipeline['image_sd_folder_path'] is None:
                raise ValueError("Error: 'image_sd_folder_path' cannot be None - Because your Img2Img pipeline require a imgage as input. Please provide a valid input path.")

        elif "image" in sd_input: 
            cnet_image = True
            if yaml_config_input_pipeline['image_cnet_folder_path'] is None:
                raise ValueError ("Error: 'image_cnet_folder_path' cannot be None - Because your controlnet sd pipeline require a imgage as input. Please provide a valid input path.")
    return sd_image, cnet_image


# def validate_yaml_structure(logger, file_path):
#     required_sections = {"image_prompt_path", "input_pipeline", "calib_config", "output"}
    
#     try:
#         # Load the YAML file
#         with open(file_path, 'r') as file:
#             data = yaml.safe_load(file)

#         # Get the top-level keys
#         yaml_sections = set(data.keys())
        
#         # Check if all required sections are present
#         if yaml_sections == required_sections:
#             logger.info("YAML structure is VALID.")
#             return data
#         else:
#             missing = required_sections - yaml_sections
#             extra = yaml_sections - required_sections
#             if missing:
#                 raise ValueError(f"Missing sections: {missing}. Double-check the structure of .yaml file, base on the docs in .yaml file.")
#             if extra:
#                 logger.warning(f"Unexpected sections: {extra}. This/These extra section(s) won't be use during this process.")
#                 return data
#     except Exception as e:
#         raise ValueError(f"Error validating YAML: {e}")


def validate_yaml_structure(logger, file_path):
    # Define the required sections with their respective subsections
    required_structure = {
        "image_prompt_path": {"prompt_file_path", "image_sd_folder_path", "image_cnet_folder_path"},
        "input_pipeline": set(),  # No specific subsections to validate for this section
        "calib_config": {"batch_size", "quant_level", "calibrate_size", "export_onnx"},
        "output": {"output_model_path"}
    }

    try:
        # Load the YAML file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Validate top-level sections
        yaml_sections = set(data.keys())
        required_sections = set(required_structure.keys())
        if yaml_sections != required_sections:
            missing = required_sections - yaml_sections
            extra = yaml_sections - required_sections
            if missing:
                raise ValueError(f"Missing sections: {missing}. Ensure your YAML has the required sections.")
            if extra:
                print(extra)
                logger.info(f"Unexpected sections: {extra}. These extra sections won't be used.")

        # Validate subsections
        for section, expected_subsections in required_structure.items():
            if section in data:
                current_subsections = set(data[section].keys()) if isinstance(data[section], dict) else set()
                if current_subsections != expected_subsections:
                    missing = expected_subsections - current_subsections
                    extra = current_subsections - expected_subsections
                    if missing:
                        raise ValueError(
                            f"Missing subsections in '{section}': {missing}. Ensure it matches the expected structure."
                        )
                    if extra:
                        logger.warning(f"Unexpected subsections in '{section}': {extra}. These won't be used.")

        logger.info("YAML structure and subsections are VALID.")
        return data

    except Exception as e:
        raise ValueError(f"Error validating YAML: {e}")
    


""" TENSORT UTILS """

def load_calibration_images(folder_path, batch_size):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            image = load_image(img_path)
            if image is not None:
                images.append(image)

    # Group images into batches of size `batch_size`
    return [images[i : i + batch_size] for i in range(0, len(images), batch_size)]

def load_calib_prompts(batch_size, calib_data_path):
    with open(calib_data_path, "r") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]

def load_calibration_images(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if os.path.isfile(img_path):
            image = load_image(img_path)
            if image is not None:
                images.append(image)
    return images

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
                
                
                
""" CALIBRATION UTILS """

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
        
# Define the ONNX export function
def export_to_onnx(logger, model, device, output_folder):
    try:
        # Generate dummy inputs
        sample = torch.randn((1, 4, 128, 128), dtype=torch.float32, device=device)
        timestep = torch.rand(1, dtype=torch.float32, device=device)
        encoder_hidden_state = torch.randn((1, 77, 768), dtype=torch.float32, device=device)

        # Create output paths
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)
        onnx_output_path = output_path / "unet_calibrate" / "model.onnx"
        onnx_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export the model to ONNX
        torch.onnx.export(
            model,
            (sample, timestep, encoder_hidden_state),
            str(onnx_output_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['sample', 'timestep', 'encoder_hidden_state'],
            output_names=['predict_noise'],
            dynamic_axes={
                "sample": {0: "B", 2: "W", 3: 'H'},
                "encoder_hidden_state": {0: "B", 1: "S", 2: 'D'},
                "predict_noise": {0: 'B', 2: "W", 3: 'H'}
            }
        )
        logger.info(f"Model successfully exported to ONNX at {onnx_output_path}")

        # Optimize and save the ONNX model
        unet_opt_graph = onnx.load(str(onnx_output_path))
        unet_optimize_path = output_path / "unet_optimize"
        unet_optimize_path.mkdir(parents=True, exist_ok=True)
        unet_optimize_file = unet_optimize_path / "model.onnx"

        onnx.save_model(
            unet_opt_graph,
            str(unet_optimize_file),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="weights.pb",
        )
        logger.info(f"Optimized ONNX model saved to {unet_optimize_file}")

    except Exception as e:
        raise RuntimeError(f"ONNX export failed on device {device}: {e}")