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

from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM


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


class ActionTokenizer:
    def __init__(
        self,
        tokenizer,
        bins: int = 256,
        min_action: int = -1,
        max_action: int = 1,
        use_extra: bool = False,
    ) -> None:
        """
        Discretizes continuous robot actions into N bins per dimension and maps to the least used tokens.

        NOTE =>> by default, assumes a BPE-style tokenizer akin to the LlamaTokenizer, where *the least used tokens*
                 appear at the end of the vocabulary!

        :param tokenizer: Base LLM/VLM tokenizer to extend.
        :param bins: Number of bins for each continuous value; we'll adopt a uniform binning strategy.
        :param min_action: Minimum action value (for clipping, setting lower bound on bin interval).
        :param max_action: Maximum action value (for clipping, setting upper bound on bin interval).
        :param use_extra: Use the extra tokens (not just the last ones), only implemented for Qwen2
        """
        self.tokenizer, self.n_bins, self.min_action, self.max_action = tokenizer, bins, min_action, max_action

        # Create Uniform Bins + Compute Bin Centers
        self.bins = np.linspace(min_action, max_action, self.n_bins)
        self.bin_centers = (self.bins[:-1] + self.bins[1:]) / 2.0

        self.tokenizer_len = self.tokenizer.vocab_size
        if isinstance(tokenizer, Qwen2TokenizerFast) and use_extra:
            self.tokenizer_len = len(self.tokenizer)
        elif use_extra:
            raise NotImplementedError("Cannot use extra tokens for this tokenizer!")

        # [Contract] Set "action_token_begin_idx" based on `self.tokenizer.vocab_size - (self.n_bins + 1)`
        #   =>> Assumes we're always overwriting the final `n_bins` tokens of the vocabulary!
        self.action_token_begin_idx: int = int(self.tokenizer_len - (self.n_bins + 1))
        self.action_token_end_idx: int = int(self.tokenizer_len)

    def __call__(self, action: np.ndarray):
        """Clip & bin actions to *the last `n_bins` tokens* of the vocabulary (e.g., tokenizer.vocab[-256:])."""
        action = np.clip(action, a_min=float(self.min_action), a_max=float(self.max_action))
        discretized_action = np.digitize(action, self.bins)

        # Handle single element vs. batch
        if len(discretized_action.shape) <= 1:
            return self.tokenizer.decode(list(self.tokenizer_len - discretized_action))
        else:
            return self.tokenizer.batch_decode((self.tokenizer_len - discretized_action).tolist())

    def decode_token_ids_to_actions(self, action_token_ids: np.ndarray) -> np.ndarray:
        """
        Returns continuous actions for discrete action token IDs.

        NOTE =>> Because of the way the actions are discretized w.r.t. the bins (and not the bin centers), the
                 digitization returns bin indices between [1, # bins], inclusive, when there are actually only
                 (# bins - 1) bin intervals.

                 Therefore, if the digitization returns the last possible index, we map this to the last bin interval.

        EXAMPLE =>> Let's say self._bins has 256 values. Then self._bin_centers has 255 values. Digitization returns
                    indices between [1, 256]. We subtract 1 from all indices so that they are between [0, 255]. There
                    is still one index (i==255) that would cause an out-of-bounds error if used to index into
                    self._bin_centers. Therefore, if i==255, we subtract 1 from it so that it just becomes the index of
                    the last bin center. We implement this simply via clipping between [0, 255 - 1].
        """
        discretized_actions = self.tokenizer_len - action_token_ids
        discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=self.bin_centers.shape[0] - 1)

        return self.bin_centers[discretized_actions]

    @property
    def vocab_size(self) -> int:
        return self.n_bins

    @property
    def required_future_horizon(self) -> int:
        # the number of future action horizon elements
        return 0


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

dataset_metadata = json.load(open(dataset_metadata_path, 'r'))['rlbencho1']

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
        "full_image": image,
        "wrist_image": wrist_image,
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
