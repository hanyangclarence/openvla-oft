from typing import Iterator, Tuple, Any
import os
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import json
from tqdm import tqdm

tfds.core.utils.gcs_utils._is_gcs_disabled = True
os.environ['NO_GCE_CHECK'] = 'true'

class RLBenchO1Dataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('2.0.0')
    RELEASE_NOTES = {
      '2.0.0': 'Initial release.',
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'depth_image': tfds.features.Image(
                            shape=(512, 512, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float32,
                            doc='Robot state, qpos or RTX version: consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(8,),
                        dtype=np.float32,
                        doc='Robot action, qpos or RTX version: consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'language_reason': tfds.features.Text(
                        doc='Language Reason.'
                    ),
                    'is_perturb': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='Whether this step is from a perturbed trajectory.'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='Path to the original data file.'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/gpfs/yanghan/data/runs_vla_data/train'),
            'val': self._generate_examples(path='/gpfs/yanghan/data/runs_vla_data/val'),
        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""
        
        error_count = 0
        error_log_path = f"errors_{path.replace('/', '_')}.txt"
        
        def _get_episode_paths(path):
            episode_paths = []
            tasks = os.listdir(path)
            for task in tasks:
                task_path = os.path.join(path, task)
                if not os.path.isdir(task_path):
                    continue
                episodes = os.listdir(task_path)
                for episode in episodes:
                    episode_path = os.path.join(task_path, episode, "video")
                    if not os.path.isdir(episode_path):
                        continue
                    episode_paths.append(episode_path)
            return episode_paths
        
        def _process_pose_to_state(pose_dict):
            """Convert pose dictionary to 8-dimensional state vector
            
            Args:
                pose_dict: Dictionary containing 'pos', 'ori', and 'gripper_open'
                
            Returns:
                np.array: 8-dimensional state vector [pos(3) + ori(4) + gripper(1)]
            """
            # Extract position (3D) and orientation (4D quaternion)
            pos = np.array(pose_dict['pos'], dtype=np.float32)  # [x, y, z]
            ori = np.array(pose_dict['ori'], dtype=np.float32)  # [qx, qy, qz, qw]
            
            # Convert gripper state to float (1D)
            gripper = 0.0 if pose_dict['gripper_open'] else 1.0
            
            # Concatenate into 8D vector
            state = np.concatenate([pos, ori, [gripper]], dtype=np.float32)
            return state
    
        def _parse_example(episode_path):
            nonlocal error_count
            data = []
            
            for subdir in os.listdir(episode_path):
                try:
                    subdir_path = os.path.join(episode_path, subdir)
                    if not os.path.isdir(subdir_path):
                        continue
                        
                    # Get image paths rather than loading them
                    if 'expert' in subdir:
                        rgb_path = os.path.join(subdir_path, 'front_rgb', 'begin.png')
                        depth_path = os.path.join(subdir_path, 'front_depth', 'begin.png')
                        wrist_rgb_path = os.path.join(subdir_path, 'wrist_rgb', 'begin.png')
                        info_path = os.path.join(subdir_path, 'info.json')
                        
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        
                        subgoal = info['subgoal']
                        subgoal = subgoal.lower().strip()
                        if not subgoal.endswith('.'):
                            subgoal = subgoal + '.'
                        if subgoal.startswith('the robot'):
                            subgoal = subgoal[10:]
                        subgoal = subgoal.strip().capitalize()
                            
                        sample = {
                            'observation': {
                                'image': rgb_path,  # Just store the path
                                'depth_image': depth_path,  # Just store the path
                                'wrist_image': wrist_rgb_path,
                                'state': _process_pose_to_state(info['prev_pose'])
                            },
                            'action': _process_pose_to_state(info['current_pose']),
                            'language_instruction': info['lang_goal'],
                            'language_reason': f"ACTION SUCCESS:\nTrue\n\nCURRENT GOAL:\n{subgoal}",
                            'is_perturb':False
                        }
                        
                    elif 'perturb' in subdir:
                        
                        rgb_path = os.path.join(subdir_path, 'front_rgb', 'end.png')
                        depth_path = os.path.join(subdir_path, 'front_depth', 'end.png')
                        wrist_rgb_path = os.path.join(subdir_path, 'wrist_rgb', 'end.png')
                        info_path = os.path.join(subdir_path, 'info.json')
                        
                        with open(info_path, 'r') as f:
                            info = json.load(f)
                        
                        failure_reason = info['failure_reason_gpt']
                        currection_instruction = info['correction_instruction_gpt']
                        
                        sample = {
                            'observation': {
                                'image': rgb_path,  # Just store the path
                                'depth_image': depth_path,  # Just store the path
                                'wrist_image': wrist_rgb_path,
                                'state': _process_pose_to_state(info['current_pose'])
                            },
                            'action': _process_pose_to_state(info['correct_pose']),
                            'language_instruction': info['lang_goal'],
                            'language_reason': f"ACTION SUCCESS:\nFalse\n\nFAILURE REASON:\n{failure_reason}\n\nCORRECTION INSTRUCTION:\n{currection_instruction}",
                            'is_perturb':True
                        }
                        
                    if 'expert' in subdir or 'perturb' in subdir:
                        data.append(sample)
                        
                except Exception as e:
                    error_count += 1
                    error_msg = f"Error processing episode {episode_path}, subdir {subdir}: {str(e)}\n"
                    # print(error_msg, end='')
                    with open(error_log_path, 'a') as f:
                        f.write(error_msg)
                    continue
            
            # create output data sample
            samples = {
                'steps': data,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }
                    
            return episode_path, samples

        episode_paths = _get_episode_paths(path)
        
        # Clear existing log file
        open(error_log_path, 'w').close()
        
        # Add tqdm progress bar
        for episode_path in tqdm(episode_paths, desc=f"Processing {path}", unit="episode"):
            yield _parse_example(episode_path)
            
        print(f"\nTotal errors encountered: {error_count}")
        print(f"Error details saved to: {error_log_path}")

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )


