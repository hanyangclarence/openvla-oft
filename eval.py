from transformers import AutoProcessor, AutoModelForVision2Seq
from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast
import torch
from PIL import Image
from peft import LoraConfig, PeftModel
from typing import Optional
import glob
import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfgt
from pathlib import Path
from typing import Optional, Union
from dataclasses import dataclass

from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
from prismatic.vla.action_tokenizer import ActionTokenizer


def _process_pose_to_state(pose_dict):
    # Extract position (3D) and orientation (4D quaternion)
    pos = np.array(pose_dict['pos'], dtype=np.float32)  # [x, y, z]
    ori = np.array(pose_dict['ori'], dtype=np.float32)  # [qx, qy, qz, qw]
    
    # Convert gripper state to float (1D)
    gripper = 0.0 if pose_dict['gripper_open'] else 1.0
    
    # Concatenate into 8D vector
    state = np.concatenate([pos, ori, [gripper]], dtype=np.float32)
    return state[None, ...]  # (1, 8)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return 1 - actions


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "openvla"                    # Model family
    pretrained_checkpoint: Union[str, Path] = ""     # Pretrained checkpoint path

    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusion==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 2                     # Number of images in the VLA input (default: 1)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)
    num_open_loop_steps: int = 8                     # Number of actions to execute open-loop before requerying policy

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = ""                # Action un-normalization key

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = ""                        # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    num_trials_per_task: int = 50                    # Number of rollouts per task
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # Local directory for eval logs

    use_wandb: bool = False                          # Whether to also log results in Weights & Biases
    wandb_entity: str = "your-wandb-entity"          # Name of WandB entity
    wandb_project: str = "your-wandb-project"        # Name of WandB project

    seed: int = 7                                    # Random Seed (for reproducibility)


saved_model_directory = "logs/openvla-7b+rlbencho1+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug--v0--4500_chkpt"
dataset_metadata_path = "logs/openvla-7b+rlbencho1+b10+lr-0.0005+lora-r32+dropout-0.0/dataset_statistics.json"

cfg = GenerateConfig(
    pretrained_checkpoint = saved_model_directory,
    use_l1_regression = False,
    use_diffusion = False,
    use_film = True,
    num_images_in_input = 2,
    use_proprio = True,
    load_in_8bit = False,
    load_in_4bit = False,
    center_crop = True,
    num_open_loop_steps = NUM_ACTIONS_CHUNK,
    unnorm_key = "rlbencho1",
)

import pdb
pdb.set_trace()

# Load OpenVLA-OFT policy and inputs processor
vla = get_vla(cfg)
processor = get_processor(cfg)

# Load proprio projector to map proprio to language embedding space
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
# processor = AutoProcessor.from_pretrained(
#     saved_model_directory,
#     trust_remote_code=True
# )
# model = AutoModelForVision2Seq.from_pretrained(
#     saved_model_directory,
#     torch_dtype=torch.bfloat16,    # Or your preferred dtype for inference (e.g., torch.float32)
#     low_cpu_mem_usage=True,        # Optional: helps with memory if loading large models
#     trust_remote_code=True,
# ).to(device)


# Create Action Tokenizer
action_tokenizer = ActionTokenizer(processor.tokenizer)

dataset_metadata = vla.norm_stats['rlbencho1']

correct_action_token_count = 0
correct_transition_token_count = 0
correct_rotation_token_count = 0
correct_gripper_token_count = 0
correct_format_count = 0
incorrect_format_count = 0
l1_dist_list = []
transition_l1_dist_list = []
rotation_l1_dist_list = []
gripper_l1_dist_list = []
all_transitions = glob.glob("/gpfs/yanghan/data/runs_vla_data/val/*/0/video/*")
for idx, path in enumerate(all_transitions):
    if "expert" in path:
        obs_path = f"{path}/front_rgb/begin.png"
        wrist_obs_path = f"{path}/wrist_rgb/begin.png"
    elif "perturb" in path:
        obs_path = f"{path}/front_rgb/end.png"
        wrist_obs_path = f"{path}/wrist_rgb/end.png"
    else:
        continue
    json_path = f"{path}/info.json"
    if not os.path.exists(obs_path) or not os.path.exists(json_path):
        continue
    
    json_data = json.load(open(json_path, 'r'))
    task_instruction = json_data['lang_goal']
    
    image = Image.open(obs_path).convert("RGB")
    wrist_image = Image.open(wrist_obs_path).convert("RGB")
    if "expert" in path:
        state = _process_pose_to_state(json_data['prev_pose'])
    else:
        state = _process_pose_to_state(json_data['current_pose'])
    
    observation = {
        "full_image": np.array(image).astype(np.uint8),
        "wrist_image": np.array(wrist_image).astype(np.uint8),
        "state": state,
        "task_description": task_instruction,
    }
    actions = get_vla_action(
        cfg=cfg, vla=vla,  processor=processor, obs=observation, task_label=task_instruction,
        proprio_projector=proprio_projector, use_film=cfg.use_film
    )
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    prompt = f"In: What action should the robot take to {task_instruction}?\nOut:"
    
    
    inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)
    generated_ids = model.generate(**inputs, max_new_tokens=200)[0]
    pred_action_ids = generated_ids[generated_ids > action_tokenizer.action_token_begin_idx].cpu().numpy()
    
    output_str = processor.tokenizer.decode(
        generated_ids.cpu().numpy().tolist(),
        skip_special_tokens=False
    )
    
    if "expert" in path:
        curr_pose = _process_pose_to_state(json_data['prev_pose'])
        next_pose = _process_pose_to_state(json_data['current_pose'])
    else:
        assert "perturb" in path
        curr_pose = _process_pose_to_state(json_data['current_pose'])
        next_pose = _process_pose_to_state(json_data['correct_pose'])
    
    eef_position_proprio, eef_orientation_proprio, gripper_proprio = tf.split(curr_pose, [3,4,1], axis=1)  # (T,3) (T,4) (T,1)
    eef_position_control, eef_orientation_control, gripper_control = tf.split(next_pose, [3,4,1], axis=1)  # (T,3) (T,4) (T,1)
    
    action_gripper = invert_gripper_actions(gripper_control) # +1 = open, 0 = close
    
    action_delta_xyz = eef_position_control - eef_position_proprio # (T, 3)
    
    # quaternions in rlbench and tfgraphics are all in format xyzw, so we don't need further conversion
    delta_eef_orientation_proprio = tfgt.quaternion.multiply(
        eef_orientation_control, tfgt.quaternion.inverse(eef_orientation_proprio)
    )
    delta_eef_orientation_proprio = tfgt.quaternion.normalize(delta_eef_orientation_proprio)
    action_delta_rpy = tfgt.euler.from_quaternion(delta_eef_orientation_proprio)
    
    # resolve NaN values in action_delta_rpy
    action_delta_rpy = tf.where(tf.math.is_nan(action_delta_rpy), tf.zeros_like(action_delta_rpy), action_delta_rpy)
    
    gt_action = tf.concat([action_delta_xyz, action_delta_rpy, action_gripper], axis=-1)
    
    # normalize action with metadata, with default q99 method
    low = tf.constant(dataset_metadata['action']['q01'], dtype=tf.float32)
    high = tf.constant(dataset_metadata['action']['q99'], dtype=tf.float32)
    mask = tf.constant(dataset_metadata['action']['mask'], dtype=tf.bool)
    
    gt_action = tf.where(
        mask,
        tf.clip_by_value(2 * (gt_action - low) / (high - low + 1e-8) - 1, -1, 1),
        gt_action
    )
    
    gt_action = np.array(gt_action, dtype=np.float32)[0]
    
    # Tokenize the action into ids
    gt_action_ids = np.digitize(
        np.clip(gt_action, a_min=action_tokenizer.min_action, a_max=action_tokenizer.max_action),
        action_tokenizer.bins,
    )
    gt_action_ids = action_tokenizer.tokenizer_len - gt_action_ids
    
    if len(pred_action_ids) == len(gt_action_ids):
        correct_action_token_count += np.sum(pred_action_ids == gt_action_ids)
        correct_transition_token_count += np.sum(pred_action_ids[:3] == gt_action_ids[:3])
        correct_rotation_token_count += np.sum(pred_action_ids[3:6] == gt_action_ids[3:6])
        correct_gripper_token_count += np.sum(pred_action_ids[6] == gt_action_ids[6])
        correct_format_count += 1
    else:
        incorrect_format_count += 1
    
    if len(pred_action_ids) == len(gt_action_ids):
        pred_action = action_tokenizer.decode_token_ids_to_actions(pred_action_ids)
        action_l1_distance = np.mean(np.abs(pred_action - gt_action))
        transition_l1_distance = np.mean(np.abs(pred_action[:3] - gt_action[:3]))
        rotation_l1_distance = np.mean(np.abs(pred_action[3:6] - gt_action[3:6]))
        gripper_l1_distance = np.mean(np.abs(pred_action[6] - gt_action[6]))
        transition_l1_dist_list.append(transition_l1_distance)
        rotation_l1_dist_list.append(rotation_l1_distance)
        gripper_l1_dist_list.append(gripper_l1_distance)
        l1_dist_list.append(action_l1_distance)

    action_accuracy = correct_action_token_count / (correct_format_count * 7)
    transition_accuracy = correct_transition_token_count / (correct_format_count * 3)
    rotation_accuracy = correct_rotation_token_count / (correct_format_count * 3)
    gripper_accuracy = correct_gripper_token_count / (correct_format_count * 1)
    transition_accuracy 
    print(f"{idx + 1}/{len(all_transitions)}: Action Accuracy: {action_accuracy:.4f}, {transition_accuracy:.4f}, "
          f"{rotation_accuracy:.4f}, {gripper_accuracy:.4f}, "
          f"Incorrect Count: {incorrect_format_count}, "
          f"L1 Distance: {np.mean(l1_dist_list) if l1_dist_list else 0:.4f}, {np.mean(transition_l1_dist_list) if transition_l1_dist_list else 0:.4f}, "
          f"{np.mean(rotation_l1_dist_list) if rotation_l1_dist_list else 0:.4f}, "
            f"{np.mean(gripper_l1_dist_list) if gripper_l1_dist_list else 0:.4f}")
