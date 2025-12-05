"""Command-line helper to run the pose estimation stack on a single image."""

import argparse
from pathlib import Path

from pose_estimation import (
    load_image,
    preprocess_image,
    PoseExtractor,
    body25_to_humanoid_pose,
)


def run_pose_estimation(image_path: str, out_path: str | None, show: bool) -> None:
    image = load_image(image_path)
    preprocessed = preprocess_image(image)

    extractor = PoseExtractor()
    body25 = extractor.extract_keypoints(preprocessed)

    if len(body25) == 0:
        print("❌ No pose detected in the provided image.")
        return

    pose = body25_to_humanoid_pose(body25)
    print("✅ Pose detected! Humanoid joint vector (17 values):")
    print(pose)

    # Save and/or display skeleton overlay using the helper already in the package.
    if out_path:
        extractor.draw_skeleton(image, body25, out_path)
    if show:
        extractor.draw_skeleton(image, body25, None)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Mediapipe-based pose estimation pipeline on a single image."
    )
    parser.add_argument(
        "--image",
        "-i",
        required=True,
        help="Path to the image file (relative or absolute).",
    )
    parser.add_argument(
        "--out",
        "-o",
        default="out.jpg",
        help="Where to save the skeleton overlay image (empty string to skip saving).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the skeleton overlay in a window after processing.",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at '{image_path}'")

    out_path = args.out if args.out.strip() else None
    run_pose_estimation(str(image_path), out_path, args.show)


if __name__ == "__main__":
    main()
