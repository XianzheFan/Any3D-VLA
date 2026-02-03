from urchin import URDF, Collision, Sphere, Geometry  # type: ignore

import os
if 'DEBUG_PORT' in os.environ:
    import debugpy
    debugpy.listen(int(os.environ['DEBUG_PORT']))
    print(f'waiting for debugger to attach...')
    debugpy.wait_for_client()

import argparse
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--port", type=str, required=True)
arg_parser.add_argument("--batching-delay", type=float, default=80)
arg_parser.add_argument("--batch-size", type=int, default=1)
arg_parser.add_argument("--dataset-statistics", type=str, required=True)
arg_parser.add_argument("--path", type=str, required=True)
arg_parser.add_argument("--openloop", action="store_true")
arg_parser.add_argument("--top_k", type=int, default=None)
arg_parser.add_argument("--compile", action="store_true")


import PIL
import io
import os
from typing import List
import zmq
import pickle
import time
import numpy as np
from tqdm import tqdm
from vla_network.model.vla import VLAAgent
from vla_network.dataset.prompt import COT_PROMPT
import torch
torch.autograd.set_grad_enabled(False)

from gx_utils.logger import log


def decode_depth_png_f32_rgba(png_bytes: bytes, expected_shape=None) -> np.ndarray:
    import PIL.Image
    img = PIL.Image.open(io.BytesIO(png_bytes))
    rgba = np.array(img, dtype=np.uint8)  # (H,W,4), uint8
    h, w = rgba.shape[:2]

    depth = np.ascontiguousarray(rgba).view('<f4').reshape(h, w)  # (H,W) float32

    if expected_shape is not None:
        eh, ew, ec = expected_shape
        assert (eh, ew) == (h, w), f"depth shape mismatch: got {(h,w)}, expected {(eh,ew)}"
        if ec == 1:
            depth = depth[..., None]  # (H,W,1)
    else:
        depth = depth[..., None]

    return depth


def interpolate_delta_actions(delta_actions, n):
    """
    Interpolate m delta_actions to m*n delta_actions.

    actions: list of actions, each action is (delta x, delta y, delta z, delta roll, delta pitch, delta yaw, gripper open/close).
    """
    import transforms3d as t3d
    ret = []
    for delta_action in delta_actions:
        xyzs = 1 / n * np.array([delta_action[:3]]*n)
        axangle_ax, axangle_angle = t3d.euler.euler2axangle(*delta_action[3:6])
        eulers = [t3d.euler.axangle2euler(axangle_ax, axangle_angle / n)]*n
        grippers = np.array([[0.]] * (n-1) + [[delta_action[-1]]])  # 0 for no change of gripper state
        ret.extend(np.concatenate([xyzs, eulers, grippers], axis=-1))
    return ret


def batch_process(vla_model: VLAAgent, batch: List[dict]):
    input_batch = []
    for sample in batch:
        if sample.get('compressed', False):
            for key in ['image_array', 'image_wrist_array']:
                decompressed_image_array = []
                for compressed_image in sample[key]:
                    decompressed_image_array.append(np.array(PIL.Image.open(io.BytesIO(compressed_image))))
                sample[key] = decompressed_image_array
            depth_encoding = 'png_f32_rgba'
            depth_shape = tuple((256, 256, 1))
            for key in ['depth_array', 'depth_wrist_array']:
                if key in sample:
                    decompressed_depth_array = []
                    for compressed_depth in sample[key]:
                        if depth_encoding == 'png_f32_rgba':
                            depth = decode_depth_png_f32_rgba(compressed_depth, expected_shape=depth_shape)
                            decompressed_depth_array.append(depth)
                        else:
                            img = PIL.Image.open(io.BytesIO(compressed_depth))
                            arr = np.array(img)
                            if arr.dtype != np.float32:
                                arr = arr.astype(np.float32) / 255.0
                            decompressed_depth_array.append(arr.reshape(img.height, img.width, 1))
                    sample[key] = decompressed_depth_array
            sample['compressed'] = False
            
            default_depth_shape = (256, 256, 1)
            if 'depth_array' not in sample:
                log.warn("'depth_array' not in sample, using default zeros.")
                sample['depth_array'] = [np.zeros(default_depth_shape, dtype=np.float32)]
            if 'depth_wrist_array' not in sample:
                log.warn("'depth_wrist_array' not in sample, using default zeros.")
                sample['depth_wrist_array'] = [np.zeros(default_depth_shape, dtype=np.float32)]
    
    for sample in batch:
        proprio_array = np.array([sample['proprio_array'][-4], sample['proprio_array'][-1]])
        input_batch.append({
            **sample,
            'text': sample['text'],
            'proprio_array': proprio_array,
            'image_array': [sample['image_array'][-1]],
            'image_wrist_array': [sample['image_wrist_array'][-1]],
            'depth_array': [sample['depth_array'][-1]],
            'depth_wrist_array': [sample['depth_wrist_array'][-1]],
        })
    results = vla_model(input_batch)
    ret = []
    for result, input_sample in zip(results, batch):
        action = result['action']
        last_dim = action[:, -1]
        last_dim = np.where(last_dim < -0.5, -1, np.where(last_dim > 0.5, 1, 0))
        action = np.concatenate([action[:, :-1], last_dim[:, None]], axis=-1)
        action = interpolate_delta_actions(action, vla_model.data_cfg.dt_steps)
        debug = {}
        if 'goal' in result:
            debug['pose'] = result['goal']
        if 'bbox' in result:
            debug['bbox'] = result['bbox']
        ret.append({
            'result': action,
            'env_id': input_sample['env_id'],
            'debug': debug,
        })
    return ret


def warmup(vla_model: VLAAgent):
    SAMPLES = [
        {
            'text': 'pick up elephant',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'depth_wrist_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])]*4,
            'traj_metadata': None,
            'env_id': 1,
        },
        {
            'text': 'pick up toy large elephant',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'depth_wrist_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])]*4,
            'traj_metadata': None,
            'env_id': 2,
        },
        {
            'text': 'pick up toy car',
            'image_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'image_wrist_array': [np.zeros((256, 256, 3), dtype=np.uint8)],
            'depth_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'depth_wrist_array': [np.zeros((256, 256, 1), dtype=np.float32)],
            'proprio_array': [np.concatenate([np.zeros((6,), dtype=np.float32), np.ones((1,), dtype=np.float32)])]*4,
            'traj_metadata': None,
            'env_id': 3,
        },
    ]
    NUM_TESTS = 5
    print('warming up...')
    for i in tqdm(range(NUM_TESTS)):
        ret = batch_process(vla_model, [SAMPLES[i%len(SAMPLES)]])
    print('check the latency after warm up:')
    for i in tqdm(range(NUM_TESTS)):
        ret = batch_process(vla_model, [SAMPLES[i%len(SAMPLES)]])


def main():
    args = arg_parser.parse_args()
    vla_model = VLAAgent(args.path, compile=args.compile)
    vla_model.preprocessor.config.robot_rep = "identity"

    assert vla_model.data_cfg.action_rel_len == 0

    warmup(vla_model)

    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)
    socket.bind(f"tcp://*:{args.port}")

    requests = []
    first_arrive_time = None

    log.info('start serving')
    while True:
        current_time = time.time() * 1000
        if (len(requests) >= args.batch_size or
            ((first_arrive_time is not None) and
             (current_time - first_arrive_time > args.batching_delay) and
             len(requests) > 0)):
            data_num = min(args.batch_size, len(requests))
            client_ids, data_batch = zip(*requests[:data_num])

            tbegin = time.time()
            log.info(f'start processing {len(requests)} requests')
            results = batch_process(vla_model, data_batch)
            tend = time.time()
            log.info(f'finished {len(requests)} requests in {tend - tbegin:.3f}s')

            for client_id, result in zip(client_ids, results):
                socket.send_multipart([
                    client_id,
                    b'',
                    pickle.dumps({
                        'info': 'success',
                        'env_id': result['env_id'],
                        'result': result['result'],
                        'debug': result['debug'],
                    })
                ])

            requests = requests[data_num:]
            if len(requests) == 0:
                first_arrive_time = None

        # try getting new sample
        try:
            client_id, empty, data = socket.recv_multipart(zmq.DONTWAIT)
            if len(requests) == 0:
                first_arrive_time = time.time() * 1000

            data = pickle.loads(data)
            requests.append((client_id, data))
        except zmq.Again:
            pass


if __name__ == "__main__":
    main()