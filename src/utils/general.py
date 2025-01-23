import os
import yaml
import inspect
import logging

from diffusers.utils import load_image


LOGGING_NAME = 'Diffuser'
LOGGER = logging.getLogger(LOGGING_NAME)  # define globally (used in train.py, val.py, detect.py, etc.)

REQ_STRUCT = {
    "image_prompt_path": {"prompt_file_path", "image_sd_folder_path", "image_cnet_folder_path"},
    "input_pipeline": set(),  # No specific subsections to validate for this section
    "calib_config": {"batch_size", "quant_level", "calibrate_size", "export_onnx"},
    "output": {"output_model_path"}
}



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


def load_calibration_images(folder_path, batch_size:int = 1):
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


def validate_yaml_structure(logger, file_path):
    # Define the required sections with their respective subsections
    try:
        # Load the YAML file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Validate top-level sections
        yaml_sections = set(data.keys())
        required_sections = set(REQ_STRUCT.keys())
        if yaml_sections != required_sections:
            missing = required_sections - yaml_sections
            extra = yaml_sections - required_sections
            if missing:
                raise ValueError(f"Missing sections: {missing}. Ensure your YAML has the required sections.")
            if extra:
                print(extra)
                logger.info(f"Unexpected sections: {extra}. These extra sections won't be used.")

        # Validate subsections
        for section, expected_subsections in REQ_STRUCT.items():
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