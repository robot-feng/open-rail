from enum import Enum
from ml_collections import ConfigDict
import os
from pathlib import Path

class ModelType(str, Enum):
    MOCK = 'mock'
    ACT = 'act'
    GR00T_N1 = 'gr00t_n1'
    GR00T_N1_5 = 'gr00t_n1_5'
    GR00T_N1_6 = 'gr00t_n1_6'
    RDT = 'rdt'
    SMOLVLA = 'smolvla'
    GO1 = 'go1'
    PI0 = 'pi0'
    PI05 = 'pi05'
    PI05_PYTORCH = 'pi05_pytorch'
    TAO = 'tao'
    DM05 = 'dm05'
    WALLX = 'wallx'
    STARVLA = 'starvla'
    XPOLICYLAB = 'xpolicylab'

def get_mock_config():
    """Generate configuration for MOCK model.
    
    Returns:
        ConfigDict: Configuration dictionary containing model path for MOCK.
    """
    config = ConfigDict()
    config.model_path = '/path/to/model'
    return config

def get_gr00t_config():
    """Generate configuration for GR00T model.
    
    Returns:
        ConfigDict: Configuration dictionary containing model path for GR00T.
    """
    config = ConfigDict()
    config.model_path = '/path/to/model'
    config.embodiment_tag = 'a2d'
    config.data_config_key = 'a2d_arms_only'
    return config

def get_act_config():
    """Generate configuration for ACT model.
    
    Returns:
        ConfigDict: Configuration dictionary containing model path for ACT.
    """
    config = ConfigDict()
    config.model_path = '/path/to/model'
    return config

def get_rdt_config():
    """Generate configuration for RDT model.
    
    Returns:
        ConfigDict: Configuration dictionary containing paths for RDT model, config file, vision encoder, and language embeddings.
    """
    config = ConfigDict()
    config.model_path = '/path/to/model'
    config.config_path = '/path/to/config'
    config.vision_encoder_name_or_path = '/path/to/vision_encoder'
    config.lang_embd_path = '/path/to/lang_embds/place_bottle.pt'
    return config

def get_smolvla_config():
    """Generate configuration for RDT model.
    
    Returns:
        ConfigDict: Configuration dictionary containing paths for RDT model, config file, vision encoder, and language embeddings.
    """
    config = ConfigDict()
    config.model_path = '/path/to/model'
    config.root_path = '/path/to/root'
    return config

def get_go1_config():
    cfg = ConfigDict()
    cfg.model_path = "/path/to/model"      # 权重目录

    cfg.exp_path   = "/path/to/exppath" #  能 import evaluate.deploy
    cfg.data_stats_path = cfg.data_stats_path = os.path.join(cfg.model_path, "dataset_stats.json")                  
    return cfg

def get_pi0_config():
    """Generate configuration for pi model.
    
    Returns:
        ConfigDict: Configuration dictionary containing model path for pi.
    """
    config = ConfigDict()
    config.config = 'pi0_a2d'
    config.model_path = '/path/to/model'
    config.is_hand = False
    return config


def get_pi05_config():
    """Generate configuration for pi model.
    
    Returns:
        ConfigDict: Configuration dictionary containing model path for pi.
    """
    config = ConfigDict()
    config.config = 'pi05_a2d'
    config.model_path = '/path/to/model'
    config.is_hand = False
    return config


def get_pi05_pytorch_config():
    """PyTorch pi0.5 with the training config from pi05_base."""
    config = ConfigDict()
    config.config = 'pi05_a2d'
    config.model_path = '/path/to/model'  # Contains model.safetensors and assets/.
    config.compile_mode = 'max-autotune'  # Options: None (off, no quotes), 'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'.
    config.execute_horizon = 50
    return config


def get_tao_config():
    """Generate configuration for TAO model.
    
    Returns:
        ConfigDict: Configuration dictionary containing model path for TAO.
    """
    config = ConfigDict()
    config.model_path = '/path/to/model'
    config.embodiment_tag = 'a2d_gripper'
    return config

def get_dm05_config():
    """Generate configuration for the DM0.5 (OpenDM) model.

    Returns:
        ConfigDict: Configuration dictionary for the DM05 in-process policy.
    """
    config = ConfigDict()
    config.model_path = '/path/to/model'
    config.repo_path = '/path/to/repo'
    config.action_mode = 'absolute' # relative | absolute
    config.backend = 'fast'  # 'fast' | 'default' | 'compile'
    config.vision_trt_engine_path = '/path/to/dm05_vision.engine'
    config.diffusion_steps = 10
    config.vision_attn_implementation = 'flash_attention_2'
    config.execute_horizon = 16  # steps actually sent to the robot per cycle
    return config


def get_wallx_config():
    config = ConfigDict()
    config.model_path = '/path/to/model'
    config.repo_path = str(Path(__file__).resolve().parents[2] / 'wall-x')
    config.embodiment_tag = 'a2d_gripper'
    config.device = 'cuda:0'
    config.flow_steps = 10
    return config


def get_starvla_config():
    """Generate configuration for the StarVLA framework adapter."""
    config = ConfigDict()
    config.model_path = '/path/to/starvla/checkpoint'
    config.repo_path = os.environ.get('STARVLA_ROOT', '/data/EvaClient-StarVLA/starVLA')
    config.device = 'cuda'
    config.use_bf16 = False
    config.unnorm_key = None
    config.config_overrides = []
    # Camera order must match the checkpoint's training-time image order.
    config.camera_keys = ['cam.head', 'cam.hand_left', 'cam.hand_right']
    config.send_state = True
    config.state_key = 'state'
    config.language_key = 'language'
    config.expected_state_dim = None
    config.expected_action_dim = None
    return config


def get_xpolicylab_config():
    """Generate configuration for a remote XPolicyLab policy server."""
    config = ConfigDict()
    # XPolicyLab loads the selected policy/checkpoint in its own server process.
    config.model_path = 'remote://xpolicylab'
    config.repo_path = os.environ.get(
        'XPOLICYLAB_ROOT', str(Path(__file__).resolve().parents[1] / 'test' / 'XPolicyLab')
    )
    config.server_url = 'ws://127.0.0.1:19000'
    config.evaluation_id = 'open-rail'
    config.trial_id = 'open-rail'
    config.action_case_id = None
    config.prepare_case = False
    config.case_meta = {}
    config.ws_ping_interval_s = 20.0
    config.ws_ping_timeout_s = 20.0
    config.connect_timeout_s = 30.0
    config.handshake_timeout_s = 60.0
    config.request_timeout_s = 120.0
    config.max_connect_attempts = 180
    config.connect_retry_delay_s = 5.0
    config.max_connect_seconds = 900.0
    config.close_timeout_s = 10.0
    # Map Open-Rail's flat camera fields to XPolicyLab's standard vision tree.
    config.camera_map = [
        {'source': 'cam.head', 'target': 'cam_head'},
        {'source': 'cam.hand_left', 'target': 'cam_left_wrist'},
        {'source': 'cam.hand_right', 'target': 'cam_right_wrist'},
    ]
    # Default bimanual joint/gripper mapping for a 16-D Open-Rail state/action.
    # Edit these slices and action_keys to match the target robot and XPL policy.
    config.state_fields = [
        {'name': 'left_arm_joint_state', 'start': 0, 'end': 7},
        {'name': 'right_arm_joint_state', 'start': 7, 'end': 14},
        {'name': 'left_ee_joint_state', 'start': 14, 'end': 15},
        {'name': 'right_ee_joint_state', 'start': 15, 'end': 16},
    ]
    config.action_keys = [
        'left_arm_joint_state',
        'right_arm_joint_state',
        'left_ee_joint_state',
        'right_ee_joint_state',
    ]
    config.expected_action_dim = None
    config.instruction_key = 'language'
    return config


def get_models_config():
    """Generate configuration for all available VLA models.
    
    This function creates a comprehensive configuration dictionary that includes
    settings for all supported VLA models (GR00T, ACT, RDT) and specifies
    which model type to use by default.
    
    Returns:
        ConfigDict: Configuration dictionary containing:
            - type: Default model type to use
            - gr00t: GR00T model configuration
            - act: ACT model configuration  
            - rdt: RDT model configuration
    """
    config = ConfigDict()
    config.type = ModelType.GR00T_N1_5
    config.mock = get_mock_config()
    config.gr00t = get_gr00t_config()
    config.act = get_act_config()
    config.rdt = get_rdt_config()
    config.go1 = get_go1_config()
    config.smolvla = get_smolvla_config()
    config.pi0 = get_pi0_config()
    config.pi05 = get_pi05_config()
    config.pi05_pytorch = get_pi05_pytorch_config()
    config.tao = get_tao_config()
    config.dm05 = get_dm05_config()
    config.wallx = get_wallx_config()
    config.starvla = get_starvla_config()
    config.xpolicylab = get_xpolicylab_config()
    return config
