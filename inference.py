import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import sys
from pathlib import Path
from torchvision.utils import save_image

from models.vffc_model import VolumetricFFCModel
from data.sub_volume_dataset import GridFullPatchedSubVolumeDataset
from data.papyrus_ink_detection import detect_papyrus_ink


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--papyrus_path', type=str)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)

    parser.add_argument('--patch_size', type=int, default=256)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--z_start', type=int, default=24)
    parser.add_argument('--z_size', type=int, default=16)
    parser.add_argument('--stride', type=int, default=64)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = VolumetricFFCModel()
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    print(f"Loaded checkpoint {args.checkpoint_path}")
    del checkpoint

    test_dataset = GridFullPatchedSubVolumeDataset(
        image_path=args.papyrus_path,
        patch_size=args.patch_size,
        z_start=args.z_start,
        z_size=args.z_size,
        load_labels=False,
        transform=None,
        stride=args.stride,
        train=False)

    print(f'Loaded dataset with {len(test_dataset)} samples.')

    result = detect_papyrus_ink(model=model,
                                dataset=test_dataset,
                                batch_size=args.batch_size,
                                threshold=args.threshold,
                                load_labels=False,
                                device=device,
                                epoch=0,
                                save_path=args.output_path)

    save_path = Path(args.output_path) / f'{args.checkpoint_uuid}'
    if not save_path.exists():
        save_path.mkdir(parents=True)
    save_image(result['output'], Path(save_path) / f'output_{args.checkpoint_uuid}_detection_prob.png')
    save_image(result['detection'], Path(save_path) / f'output_{args.checkpoint_uuid}_detection_map.png')

    sys.exit()


if __name__ == '__main__':
    main()
