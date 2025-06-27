import os
from sfacex_to_be_checked import LBPChiSquareAuthenticator
import json


def get_data_set():
    dirs = ["right", "left", "center", "down", "up"]
    folder_names = ["Look_Right", "Look_Left", "Look_Straight", "Look_Down", "Look_Up"]
    base = "./data/"
    paths = {}

    for folder, label in zip(folder_names, dirs):
        full_path = os.path.join(base, folder)
        if not os.path.exists(full_path):
            raise ValueError(
                f"Directory {full_path} does not exist. Please check the dataset paths."
            )

        imgs = os.listdir(full_path)
        filtered_imgs = [
            img for idx, img in enumerate(imgs) if idx % 2 == 0
        ]  # skip 1,3,5...
        paths[label] = [
            os.path.join(full_path, img)
            for img in filtered_imgs
            if img.endswith(".jpg") or img.endswith(".png")
        ]

    return paths


def print_dataset_info(paths):
    print("Dataset Information:")
    for label, images in paths.items():
        print(f"{label}: {len(images)} images")
    print("Total images:", sum(len(images) for images in paths.values()))


def enroll():
    paths = get_data_set()
    print_dataset_info(paths)
    positive_paths_person_A = []
    neagative_paths_person_A = []
    positive_path = "./positive/"
    negative_path = "./negative/"

    if os.path.exists(positive_path):
        positive_paths_person_A = [
            os.path.join(positive_path, img)
            for img in os.listdir(positive_path)
            if img.endswith(".jpg") or img.endswith(".png")
        ]

    if os.path.exists(negative_path):
        neagative_paths_person_A = [
            os.path.join(negative_path, img)
            for img in os.listdir(negative_path)
            if img.endswith(".jpg") or img.endswith(".png")
        ]

    print("--- Initializing Authenticator ---")

    authenticator = LBPChiSquareAuthenticator(
        target_size=(128, 128),
        grid_cells=(8, 8),
        # num_templates=5 # Specify the number of templates
    )

    print("\n--- Enrolling Person A ---")
    # Enroll using the first `num_templates` (i.e., 5) images from enroll_paths_person_A
    # Ensure enroll_paths_person_A has at least `authenticator.num_templates` valid paths.
    if len(positive_paths_person_A) < authenticator.initial_num_templates:
        print(
            f"Error: Not enough images for enrollment. Need {authenticator.initial_num_templates}, found {len(positive_paths_person_A)}"
        )
        enrollment_succeeded = False
    else:
        enrollment_succeeded = authenticator.enroll_person_with_poses(paths)

    # Check if enrollment was successful by looking at enrolled_templates_data
    if enrollment_succeeded and authenticator.enrolled_templates_data:
        print("\n--- Setting Decision Threshold via ROC ---")
        # Use all available positive samples for ROC, and all negative samples
        auth_positive_samples = positive_paths_person_A  # All 20 positive samples
        auth_negative_samples = neagative_paths_person_A  # All 25 negative samples

        # Ensure ROC sample paths exist
        if not all(os.path.exists(p) for p in auth_positive_samples) or not all(
            os.path.exists(p) for p in auth_negative_samples
        ):
            print(
                "Error: Missing positive or negative sample files for ROC analysis. Aborting threshold setting."
            )
        else:
            O_distance, _, a = authenticator.calculate_roc_and_find_threshold_template(
                auth_positive_samples, auth_negative_samples
            )

        if authenticator.threshold is not None:
            threshold_data = {"threshold": float(authenticator.threshold)}

            with open("data.json", "w") as f:
                json.dump(threshold_data, f, indent=4)

            print("Threshold value saved to 'data.json'")
            print(
                f"Enrollment complete. Optimal threshold: {authenticator.threshold:.4f}"
            )


# def recalculate_threshold():


def reset():
    positive_path = "./positive/"
    negative_path = "./negative/"

    paths = get_data_set()

    # if os.path.exists(positive_path):
    #     for img in os.listdir(positive_path):
    #         os.remove(os.path.join(positive_path, img))
    #     print(f"Removed all images from {positive_path}")

    # if os.path.exists(negative_path):
    #     for img in os.listdir(negative_path):
    #         os.remove(os.path.join(negative_path, img))
    #     print(f"Removed all images from {negative_path}")

    # Reset the dataset paths
    for label, images in paths.items():
        for img in images:
            if os.path.exists(img):
                os.remove(img)
                print(f"Removed image: {img}")

    print("Enrollment data reset completed.")


def create_dirs():
    positive_path = "./positive/"
    negative_path = "./negative/"
    base_path = "./data/"

    folder_names = ["Look_Right", "Look_Left", "Look_Straight", "Look_Down", "Look_Up"]

    if not os.path.exists(base_path):
        os.makedirs(base_path)
        print(f"Created base directory: {base_path}")

    if not os.path.exists(positive_path):
        os.makedirs(positive_path)
        print(f"Created directory: {positive_path}")

    if not os.path.exists(negative_path):
        os.makedirs(negative_path)
        print(f"Created directory: {negative_path}")

    for folder in folder_names:
        folder_path = os.path.join(base_path, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"Created directory: {folder_path}")
        else:
            print(f"Directory already exists: {folder_path}")
