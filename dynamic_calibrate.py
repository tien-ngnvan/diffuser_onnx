
import os
import torch
import gc
from diffusers import StableDiffusionPipeline
import ammo.torch.quantization as atq 
import ammo.torch.opt as ato
import logging
from dynamic_calibrate_utils import *

""" Logger """
logging.basicConfig(filename="/home/tiennv/trang/model_unet_quantize_full_onnx_2/repo/Quantize-Calibration-int8/log_calibrate/calibrate_unet.log",
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filemode='w')
logger = logging.getLogger('Logging_for_calibrate_process')


def main(sd_pipeline=None,
         yaml_path=None,
         use_gpu=False):
    
    print("~~~~~~~~~~~~~~~~~~~~~~~~ START | CALIBRATION ~~~~~~~~~~~~~~~~~~~~~~~~")
    
    """
    Main function to calibrate model UNET of SD pipeline.
    
    Note:
    1. Your pipeline must have a __call__ function
    2. Your pipeline must have a instance attributes: unet. Example: "pipeline.unet" must be exist.

    Args:
        1/ pipeline (SD Class): The processing pipeline object. MUST BE A STABLE-DIFFUSION PIPELINE
        2/ config_yaml: path to yaml file where used to defined all the necessary argument. Read docs in .yaml file for more informantion
        3/ use_gpu: Option to use gpus or not. Default is FALSE 
    """
    
    """ CHECK VALID VARIABLE """
    
    # 1/ Check function input
    
    # 1.1/ Check pipeline exist
    if sd_pipeline is None:
        raise ValueError("Error: 'pipeline' cannot be None. Please provide a valid Stable diffusion pipeline class.")
    
    # 1.2/ Check for .yaml file path exist
    if os.path.exists(yaml_path):
        logger.info(f"The file path '{yaml_path}' exists.")
    else:
        raise ValueError(f"The file path '{yaml_path}' does not exist.")
    
    # 1.3/ Check for GPUS
    if use_gpu==True:
        # device = "cuda:" + ",".join([str(i) for i in range(torch.cuda.device_count())])
        device = 'cuda:0'
        logger.warning(f"Current device: {device} | Use GPUs for this process")
    else:
        device = "cpu"
        logger.warning(f"Current device: {device} | GPUs won't be used in this process")
        
    # 2/ Check arguments in yaml file
    
    # Check for .yaml file structure
    yaml_config = validate_yaml_structure(logger, yaml_path)

    # Check yaml['input_pipeline'] name match with the pipeline.__call__ variable or not 
    yaml_config_input_pipeline = yaml_config['input_pipeline']
    dict_input_pipeline, missing_keys, inputs = extract_and_validate_pipeline(sd_pipeline, yaml_config_input_pipeline)
    if not missing_keys:
        logger.info("All inputs are correct !")
    else:
        print("===============================")
        print("Your pipeline.__call__ inputs: ")
        print(inputs)
        print("===============================")
        
        raise ValueError(f"Inputs are wrong. Can not find '{missing_keys}' in your pipeline.__call__ input. Please provie correct input variable name, double-check with the name of your pipeline input log up there.")
        
    # Check for image_path exist (if pipeline needed)
    # sd_image, cnet_image: If these variable = True -> pipeline require image for sd or cnet. Else the pipeline not require image
    yaml_config_image_prompt_path = yaml_config['image_prompt_path']
    sd_image, cnet_image = check_image_in_input(dict_input_pipeline, yaml_config_image_prompt_path)



    """ MAIN PROCESS """

    # CONFIG
    BATCH_SIZE = yaml_config['calib_config']['batch_size']
    QUANT_LEVEL = yaml_config['calib_config']['quant_level']
    CALIBRATION_SIZE = yaml_config['calib_config']['calibrate_size']
    
    INPUT_PROMPT_FILE_PATH = yaml_config['image_prompt_path']['prompt_file_path']
    INPUT_SD_IMG_FOLDER_PATH = yaml_config['image_prompt_path']['image_sd_folder_path']
    INPUT_CNEXT_IMG_FOLDER_PATH = yaml_config['image_prompt_path']['image_cnet_folder_path']
    
    OUTPUT_MODEL_FOLDER_PATH = yaml_config['output']['output_model_path']

    # PIPELINE
    pipeline = sd_pipeline
    pipeline.to(device)

    # LOAD IMAGE to list image object. Read image py 'load_image' of diffusers
    calib_prompts_list = load_calib_prompts(calib_data_path=INPUT_PROMPT_FILE_PATH)
    calib_sd_image_list = None
    calib_cnet_image_list = None
    if sd_image == True:
        calib_sd_image_list = load_calibration_images(folder_path=INPUT_SD_IMG_FOLDER_PATH)
        if len(calib_prompts_list)!=len(calib_sd_image_list):
            raise ValueError(f"Length of list calib_prompt and calib_sd_img is not equal.\n Length of calib_prompt: {len(calib_prompts_list)}.\n Length of calib_sd_img: {len(calib_sd_image_list)}")
    if cnet_image == True:
        calib_cnet_image_list = load_calibration_images(folder_path=INPUT_CNEXT_IMG_FOLDER_PATH)
        if len(calib_prompts_list)!=len(calib_cnet_image_list):
            raise ValueError(f"Length of list calib_prompt and calib_cnet_img is not equal.\n Length of calib_prompt: {len(calib_prompts_list)}\n Length of calib_cnet_img: {len(calib_cnet_image_list)}")
    
    # Quantize_config
    quant_config = get_smoothquant_config(pipeline.unet, quant_level=QUANT_LEVEL)

    # Create a dict base on the yaml['input_pipeline']
    dynamic_inputs = {input_name: None for input_name in yaml_config_input_pipeline}

    def calibration_loop_sd():
        do_calibrate(
            base=pipeline,
            dynamic_input=dynamic_inputs,
            yaml_config= yaml_config,
            calibration_prompts=calib_prompts_list,
            sd_images=sd_image,
            cnet_images=cnet_image,
            calibration_imgs=calib_sd_image_list,
            calib_size=CALIBRATION_SIZE,
        )
    
    def calibration_loop_cnet():
        do_calibrate(
            base=pipeline,
            dynamic_input=dynamic_inputs,
            yaml_config= yaml_config,
            calibration_prompts=calib_prompts_list,
            sd_images=sd_image,
            cnet_images=cnet_image,
            calibration_imgs=calib_cnet_image_list,
            calib_size=CALIBRATION_SIZE,
        )

    def calibration_loop():
        do_calibrate(
            base=pipeline,
            dynamic_input=dynamic_inputs,
            yaml_config= yaml_config,
            calibration_prompts=calib_prompts_list,
            sd_images=sd_image,
            cnet_images=cnet_image,
            calib_size=CALIBRATION_SIZE,
        )

    if sd_image == True:
        logger.info("Run WITH image as input for Img2Img pipeline")
        quantized_model = atq.quantize(pipeline.unet, quant_config, forward_loop = calibration_loop_sd)
    if cnet_image == True:
        logger.info("Run WITH image as input for ControlNet pipeline")
        quantized_model = atq.quantize(pipeline.unet, quant_config, forward_loop = calibration_loop_cnet)
    else:
        logger.info("Run WITHOUT image as input")
        quantized_model = atq.quantize(pipeline.unet, quant_config, forward_loop = calibration_loop)
    
    # Save UNET int8 as .pt
    unet_int8_pt = "unet_int8.pt"
    os.makedirs(OUTPUT_MODEL_FOLDER_PATH, exist_ok=True)
    save_path = os.path.join(OUTPUT_MODEL_FOLDER_PATH, unet_int8_pt)
    print("INFO: Saving model UNET int8 as .pt model")
    ato.save(quantized_model, save_path)

    # Apply quantize level to export to ONNX
    quantize_lvl(quantized_model, quant_level=QUANT_LEVEL)
    atq.disable_quantizer(quantized_model, filter_func) 
    
    # Free GPU resources
    del pipeline  # Free the pipeline
    torch.cuda.empty_cache()  # Clear GPU memory
    gc.collect()  # Force garbage collection
    
    # EXPORT MODEL TO ONNX
    try:
        logger.info("Exporting the model to ONNX using GPU...")
        quantized_model = quantized_model.to(torch.float32).to("cuda:0")
        export_to_onnx(logger, quantized_model, "cuda:0", OUTPUT_MODEL_FOLDER_PATH)
    except RuntimeError as gpu_error:
        logger.warning(f"GPU export failed: {gpu_error}. Retrying on CPU...")
        quantized_model = quantized_model.to(torch.float32).to("cpu")
        export_to_onnx(logger, quantized_model, "cpu", OUTPUT_MODEL_FOLDER_PATH)


  
    logger.info("CALIBRATION PROCESS SUCCESSFUL")
    print("~~~~~~~~~~~~~~~~~~~~~~~~ END | CALIBRATION ~~~~~~~~~~~~~~~~~~~~~~~~")

if __name__ == "__main__":
    
    # pipeline = StableDiffusionPipeline.from_pretrained("wyyadd/sd-1.5", torch_dtype=torch.float16)
    
    
    from utils import tools
    pipeline = tools.get_pipeline(
        "neta-art/neta-xl-2.0",
        "Eugeoter/controlnext-sdxl-anime-canny",
        "Eugeoter/controlnext-sdxl-anime-canny",
        vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix",
        lora_path=None,
        load_weight_increasement=False,
        enable_xformers_memory_efficient_attention=False,
        revision=None,
        variant=None,
        hf_cache_dir=None,
        use_safetensors=True,
        device='cuda',
    )
    
    # Read YAML file
    config_yaml_path = "/home/tiennv/trang/Convert-_Unet_int8_Rebuild/Diffusion/config.yaml"
    
    main(sd_pipeline=pipeline, 
        yaml_path=config_yaml_path,
        use_gpu = False,
        )