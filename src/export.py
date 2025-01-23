import os
import torch
import onnx

from pathlib import Path
from typing import Union

from utils.general import LOGGER
from utils.torch_utils import select_device
from utils.general import validate_yaml_structure, extract_and_validate_pipeline, check_image_in_input, load_calib_prompts, load_calibration_images
from utils.calibrate_utils import get_smoothquant_config



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
    
    
    
def export(
    diffusion_pipeline,
    yaml_path: str = None,
    device: Union[str, list[str]] = None,
):
    device = select_device(device)
    
    if not os.path.exists(yaml_path):
        raise ValueError(f"The file path '{yaml_path}' does not exist.")
    yaml_config = validate_yaml_structure(LOGGER, yaml_path)
    
    # Check matching between the pipeline and the yaml['input_pipeline']
    inps_pipeline, missing_keys, inps = extract_and_validate_pipeline(diffusion_pipeline, yaml_config['input_pipeline'])
    
    if missing_keys:
        raise ValueError(f"Inputs are wrong. Can not find '{missing_keys}' in your pipeline.__call__ input. Please provie correct input variable name, double-check with the name of your pipeline input log up there.")
    
    sd_image, cnet_image = check_image_in_input(inps_pipeline, yaml_config['input_pipeline'])
    
    bz, quant_level, calib_size, inp_prompt_file_path, inp_sd_img_folder_path, inp_cnet_img_folder_path, opt_model_folder_path = (
        yaml_config['calib_config']['batch_size'],
        yaml_config['calib_config']['quant_level'],
        yaml_config['calib_config']['calibrate_size'],
        yaml_config['image_prompt_path']['prompt_file_path'],
        yaml_config['image_prompt_path']['image_sd_folder_path'],
        yaml_config['image_prompt_path']['image_cnet_folder_path'],
        yaml_config['output']['output_model_path']
    )
    
    # Setup model
    diffusion_pipeline = diffusion_pipeline.to(device)
    calib_sd_imgs, calib_cnet_imgs = None, None

    calib_prompts_list = load_calib_prompts(batch_size=bz, calib_data_path=inp_prompt_file_path)
    
    if sd_image:
        calib_sd_imgs = load_calibration_images(batch_size=bz, folder_path=inp_sd_img_folder_path)
        
        if len(calib_prompts_list) != len(calib_sd_imgs):
            raise ValueError(f"Length of list calib_prompt and calib_sd_img is not equal.\n Length of calib_prompt is:{len(calib_prompts_list)}.\n Length of calib_sd_img is {len(calib_sd_imgs)}")
        
    if cnet_image:
        calib_cnet_imgs = load_calibration_images(batch_size=bz, folder_path=inp_cnet_img_folder_path)
        
        if len(calib_prompts_list) != len(calib_cnet_imgs):
            raise ValueError(f"Length of list calib_prompt and calib_cnet_img is not equal.\n Length of calib_prompt is:{len(calib_prompts_list)}.\n Length of calib_cnet_img is {len(calib_cnet_imgs)}")
        
    # Quantize_config
    quant_config = get_smoothquant_config(diffusion_pipeline.unet, quant_level=quant_level)
    # Create a dict base on the yaml['input_pipeline']
    dynamic_inputs = {input_name: None for input_name in yaml_config['input_pipeline']}
    
    
    